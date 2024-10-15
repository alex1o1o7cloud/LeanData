import Mathlib

namespace NUMINAMATH_GPT_complex_fraction_simplification_l1720_172093

theorem complex_fraction_simplification (i : ℂ) (hi : i^2 = -1) : 
  ((2 - i) / (1 + 4 * i)) = (-2 / 17 - (9 / 17) * i) :=
  sorry

end NUMINAMATH_GPT_complex_fraction_simplification_l1720_172093


namespace NUMINAMATH_GPT_find_positive_real_solutions_l1720_172008

variable {x_1 x_2 x_3 x_4 x_5 : ℝ}

theorem find_positive_real_solutions
  (h1 : (x_1^2 - x_3 * x_5) * (x_2^2 - x_3 * x_5) ≤ 0)
  (h2 : (x_2^2 - x_4 * x_1) * (x_3^2 - x_4 * x_1) ≤ 0)
  (h3 : (x_3^2 - x_5 * x_2) * (x_4^2 - x_5 * x_2) ≤ 0)
  (h4 : (x_4^2 - x_1 * x_3) * (x_5^2 - x_1 * x_3) ≤ 0)
  (h5 : (x_5^2 - x_2 * x_4) * (x_1^2 - x_2 * x_4) ≤ 0)
  (hx1 : 0 < x_1)
  (hx2 : 0 < x_2)
  (hx3 : 0 < x_3)
  (hx4 : 0 < x_4)
  (hx5 : 0 < x_5) :
  x_1 = x_2 ∧ x_2 = x_3 ∧ x_3 = x_4 ∧ x_4 = x_5 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_real_solutions_l1720_172008


namespace NUMINAMATH_GPT_traveled_distance_is_9_l1720_172047

-- Let x be the usual speed in mph
variable (x : ℝ)
-- Let t be the usual time in hours
variable (t : ℝ)

-- Conditions
axiom condition1 : x * t = (x + 0.5) * (3 / 4 * t)
axiom condition2 : x * t = (x - 0.5) * (t + 3)

-- The journey distance d in miles
def distance_in_miles : ℝ := x * t

-- We can now state the theorem to prove that the distance he traveled is 9 miles
theorem traveled_distance_is_9 : distance_in_miles x t = 9 := by
  sorry

end NUMINAMATH_GPT_traveled_distance_is_9_l1720_172047


namespace NUMINAMATH_GPT_no_m_for_P_eq_S_m_le_3_for_P_implies_S_l1720_172089

namespace ProofProblem

def P (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def S (m x : ℝ) : Prop := |x - 1| ≤ m

theorem no_m_for_P_eq_S : ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S m x := sorry

theorem m_le_3_for_P_implies_S : ∀ (m : ℝ), (m ≤ 3) → (∀ x, S m x → P x) := sorry

end ProofProblem

end NUMINAMATH_GPT_no_m_for_P_eq_S_m_le_3_for_P_implies_S_l1720_172089


namespace NUMINAMATH_GPT_tom_bike_rental_hours_calculation_l1720_172034

variable (h : ℕ)
variable (base_cost : ℕ := 17)
variable (hourly_rate : ℕ := 7)
variable (total_paid : ℕ := 80)

theorem tom_bike_rental_hours_calculation (h : ℕ) 
  (base_cost : ℕ := 17) (hourly_rate : ℕ := 7) (total_paid : ℕ := 80) 
  (hours_eq : total_paid = base_cost + hourly_rate * h) : 
  h = 9 := 
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_tom_bike_rental_hours_calculation_l1720_172034


namespace NUMINAMATH_GPT_angle_equality_iff_l1720_172020

variables {A A' B B' C C' G : Point}

-- Define the angles as given in conditions
def angle_A'AC (A' A C : Point) : ℝ := sorry
def angle_ABB' (A B B' : Point) : ℝ := sorry
def angle_AC'C (A C C' : Point) : ℝ := sorry
def angle_AA'B (A A' B : Point) : ℝ := sorry

-- Main theorem statement
theorem angle_equality_iff :
  angle_A'AC A' A C = angle_ABB' A B B' ↔ angle_AC'C A C C' = angle_AA'B A A' B :=
sorry

end NUMINAMATH_GPT_angle_equality_iff_l1720_172020


namespace NUMINAMATH_GPT_length_of_square_side_l1720_172024

theorem length_of_square_side (length_of_string : ℝ) (num_sides : ℕ) (total_side_length : ℝ) 
  (h1 : length_of_string = 32) (h2 : num_sides = 4) (h3 : total_side_length = length_of_string) : 
  total_side_length / num_sides = 8 :=
by
  sorry

end NUMINAMATH_GPT_length_of_square_side_l1720_172024


namespace NUMINAMATH_GPT_hands_opposite_22_times_in_day_l1720_172039

def clock_hands_opposite_in_day : ℕ := 22

def minute_hand_speed := 12
def opposite_line_minutes := 30

theorem hands_opposite_22_times_in_day (minute_hand_speed: ℕ) (opposite_line_minutes : ℕ) : 
  minute_hand_speed = 12 →
  opposite_line_minutes = 30 →
  clock_hands_opposite_in_day = 22 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_hands_opposite_22_times_in_day_l1720_172039


namespace NUMINAMATH_GPT_integer_divisibility_l1720_172084

open Nat

theorem integer_divisibility {a b : ℕ} :
  (2 * b^2 + 1) ∣ (a^3 + 1) ↔ a = 2 * b^2 + 1 := sorry

end NUMINAMATH_GPT_integer_divisibility_l1720_172084


namespace NUMINAMATH_GPT_larger_number_is_1629_l1720_172048

theorem larger_number_is_1629 (x y : ℕ) (h1 : y - x = 1360) (h2 : y = 6 * x + 15) : y = 1629 := 
by 
  sorry

end NUMINAMATH_GPT_larger_number_is_1629_l1720_172048


namespace NUMINAMATH_GPT_evaluate_9_x_minus_1_l1720_172026

theorem evaluate_9_x_minus_1 (x : ℝ) (h : (3 : ℝ)^(2 * x) = 16) : (9 : ℝ)^(x - 1) = 16 / 9 := by
  sorry

end NUMINAMATH_GPT_evaluate_9_x_minus_1_l1720_172026


namespace NUMINAMATH_GPT_divisibility_of_product_l1720_172097

theorem divisibility_of_product (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a ∣ b^3) (h2 : b ∣ c^3) (h3 : c ∣ a^3) : abc ∣ (a + b + c) ^ 13 := by
  sorry

end NUMINAMATH_GPT_divisibility_of_product_l1720_172097


namespace NUMINAMATH_GPT_find_s_l1720_172056

theorem find_s (c d n r s : ℝ) 
(h1 : c * d = 3)
(h2 : ∃ p q : ℝ, (p + q = r) ∧ (p * q = s) ∧ (p = c + 1/d ∧ q = d + 1/c)) :
s = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_s_l1720_172056


namespace NUMINAMATH_GPT_jose_work_time_l1720_172074

-- Define the variables for days taken by Jose and Raju
variables (J R T : ℕ)

-- State the conditions:
-- 1. Raju completes work in 40 days
-- 2. Together, Jose and Raju complete work in 8 days
axiom ra_work : R = 40
axiom together_work : T = 8

-- State the theorem that needs to be proven:
theorem jose_work_time (J R T : ℕ) (h1 : R = 40) (h2 : T = 8) : J = 10 :=
sorry

end NUMINAMATH_GPT_jose_work_time_l1720_172074


namespace NUMINAMATH_GPT_original_numerator_l1720_172043

theorem original_numerator (n : ℕ) (hn : (n + 3) / (9 + 3) = 2 / 3) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_original_numerator_l1720_172043


namespace NUMINAMATH_GPT_kaleb_books_count_l1720_172030

/-- Kaleb's initial number of books. -/
def initial_books : ℕ := 34

/-- Number of books Kaleb sold. -/
def sold_books : ℕ := 17

/-- Number of new books Kaleb bought. -/
def new_books : ℕ := 7

/-- Prove the number of books Kaleb has now. -/
theorem kaleb_books_count : initial_books - sold_books + new_books = 24 := by
  sorry

end NUMINAMATH_GPT_kaleb_books_count_l1720_172030


namespace NUMINAMATH_GPT_harmonic_mean_of_x_and_y_l1720_172044

noncomputable def x : ℝ := 88 + (40 / 100) * 88
noncomputable def y : ℝ := x - (25 / 100) * x
noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 / ((1 / a) + (1 / b))

theorem harmonic_mean_of_x_and_y :
  harmonic_mean x y = 105.6 :=
by
  sorry

end NUMINAMATH_GPT_harmonic_mean_of_x_and_y_l1720_172044


namespace NUMINAMATH_GPT_prism_surface_area_equals_three_times_volume_l1720_172029

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

theorem prism_surface_area_equals_three_times_volume (x : ℝ) 
  (h : 2 * (log_base 5 x * log_base 6 x + log_base 5 x * log_base 10 x + log_base 6 x * log_base 10 x) 
        = 3 * (log_base 5 x * log_base 6 x * log_base 10 x)) :
  x = Real.exp ((2 / 3) * Real.log 300) :=
sorry

end NUMINAMATH_GPT_prism_surface_area_equals_three_times_volume_l1720_172029


namespace NUMINAMATH_GPT_find_f_of_f_neg2_l1720_172018

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f_of_f_neg2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_f_neg2_l1720_172018


namespace NUMINAMATH_GPT_coat_lifetime_15_l1720_172079

noncomputable def coat_lifetime : ℕ :=
  let cost_coat_expensive := 300
  let cost_coat_cheap := 120
  let years_cheap := 5
  let year_saving := 120
  let duration_comparison := 30
  let yearly_cost_cheaper := cost_coat_cheap / years_cheap
  let yearly_savings := year_saving / duration_comparison
  let cost_savings := yearly_cost_cheaper * duration_comparison - cost_coat_expensive * duration_comparison / (yearly_savings + (cost_coat_expensive / cost_coat_cheap))
  cost_savings

theorem coat_lifetime_15 : coat_lifetime = 15 := by
  sorry

end NUMINAMATH_GPT_coat_lifetime_15_l1720_172079


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l1720_172022

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A + B = 126) (h₂ : A = B + 20) (h₃ : A + B + C = 180) :
  max A (max B C) = 73 := sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l1720_172022


namespace NUMINAMATH_GPT_scarves_sold_at_new_price_l1720_172023

theorem scarves_sold_at_new_price :
  ∃ (p : ℕ), (∃ (c k : ℕ), (k = p * c) ∧ (p = 30) ∧ (c = 10)) ∧
  (∃ (new_c : ℕ), new_c = 165 / 10 ∧ k = new_p * new_c) ∧
  new_p = 18
:=
sorry

end NUMINAMATH_GPT_scarves_sold_at_new_price_l1720_172023


namespace NUMINAMATH_GPT_maximum_number_of_workers_l1720_172042

theorem maximum_number_of_workers :
  ∀ (n : ℕ), n ≤ 5 → 2 * n + 6 ≤ 16 :=
by
  intro n h
  have hn : n ≤ 5 := h
  linarith

end NUMINAMATH_GPT_maximum_number_of_workers_l1720_172042


namespace NUMINAMATH_GPT_inverse_proportion_function_increasing_l1720_172000

theorem inverse_proportion_function_increasing (m : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → (y = (m - 5) / x1) < (y = (m - 5) / x2)) ↔ m < 5 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_function_increasing_l1720_172000


namespace NUMINAMATH_GPT_log_expression_value_l1720_172036

theorem log_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) : 
  ((Real.log b / Real.log a) * (Real.log a / Real.log b))^2 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_log_expression_value_l1720_172036


namespace NUMINAMATH_GPT_terminal_side_third_quadrant_l1720_172006

theorem terminal_side_third_quadrant (α : ℝ) (k : ℤ) 
  (hα : (π / 2) + 2 * k * π < α ∧ α < π + 2 * k * π) : 
  ¬(π + 2 * k * π < α / 3 ∧ α / 3 < (3 / 2) * π + 2 * k * π) :=
by
  sorry

end NUMINAMATH_GPT_terminal_side_third_quadrant_l1720_172006


namespace NUMINAMATH_GPT_luncheon_cost_l1720_172014

variable (s c p : ℝ)
variable (eq1 : 5 * s + 9 * c + 2 * p = 6.50)
variable (eq2 : 7 * s + 14 * c + 3 * p = 9.45)
variable (eq3 : 4 * s + 8 * c + p = 5.20)

theorem luncheon_cost :
  s + c + p = 1.30 :=
by
  sorry

end NUMINAMATH_GPT_luncheon_cost_l1720_172014


namespace NUMINAMATH_GPT_value_of_x_l1720_172063

theorem value_of_x (x : ℤ) (h : 3 * x / 7 = 21) : x = 49 :=
sorry

end NUMINAMATH_GPT_value_of_x_l1720_172063


namespace NUMINAMATH_GPT_smallest_value_A_B_C_D_l1720_172012

theorem smallest_value_A_B_C_D :
  ∃ (A B C D : ℕ), 
  (A < B) ∧ (B < C) ∧ (C < D) ∧ -- A, B, C are in arithmetic sequence and B, C, D in geometric sequence
  (C = B + (B - A)) ∧  -- A, B, C form an arithmetic sequence with common difference d = B - A
  (C = (4 * B) / 3) ∧  -- Given condition
  (D = (4 * C) / 3) ∧ -- B, C, D form geometric sequence with common ratio 4/3
  ((∃ k, D = k * 9) ∧ -- D must be an integer, ensuring B must be divisible by 9
   A + B + C + D = 43) := 
sorry

end NUMINAMATH_GPT_smallest_value_A_B_C_D_l1720_172012


namespace NUMINAMATH_GPT_binom_six_two_l1720_172007

-- Define the binomial coefficient function
def binom (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- State the theorem
theorem binom_six_two : binom 6 2 = 15 := by
  sorry

end NUMINAMATH_GPT_binom_six_two_l1720_172007


namespace NUMINAMATH_GPT_largest_number_is_correct_l1720_172010

theorem largest_number_is_correct (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 3) : c = 33.25 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_is_correct_l1720_172010


namespace NUMINAMATH_GPT_path_count_correct_l1720_172081

-- Define the graph-like structure for the octagonal lattice with directional constraints
structure OctagonalLattice :=
  (vertices : Type)
  (edges : vertices → vertices → Prop) -- Directed edges

-- Define a path from A to B respecting the constraints
def path_num_lattice (L : OctagonalLattice) (A B : L.vertices) : ℕ :=
  sorry -- We assume a function counting valid paths exists here

-- Assert the specific conditions for the bug's movement
axiom LatticeStructure : OctagonalLattice
axiom vertex_A : LatticeStructure.vertices
axiom vertex_B : LatticeStructure.vertices

-- Example specific path counting for the problem's lattice
noncomputable def paths_from_A_to_B : ℕ :=
  path_num_lattice LatticeStructure vertex_A vertex_B

theorem path_count_correct : paths_from_A_to_B = 2618 :=
  sorry -- This is where the proof would go

end NUMINAMATH_GPT_path_count_correct_l1720_172081


namespace NUMINAMATH_GPT_find_xy_l1720_172082

theorem find_xy (x y : ℤ) 
  (h1 : (2 + 11 + 6 + x) / 4 = (14 + 9 + y) / 3) : 
  x = -35 ∧ y = -35 :=
by 
  sorry

end NUMINAMATH_GPT_find_xy_l1720_172082


namespace NUMINAMATH_GPT_probability_green_cube_l1720_172054

/-- A box contains 36 pink, 18 blue, 9 green, 6 red, and 3 purple cubes that are identical in size.
    Prove that the probability that a randomly selected cube is green is 1/8. -/
theorem probability_green_cube :
  let pink_cubes := 36
  let blue_cubes := 18
  let green_cubes := 9
  let red_cubes := 6
  let purple_cubes := 3
  let total_cubes := pink_cubes + blue_cubes + green_cubes + red_cubes + purple_cubes
  let probability := (green_cubes : ℚ) / total_cubes
  probability = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_probability_green_cube_l1720_172054


namespace NUMINAMATH_GPT_unique_solution_mnk_l1720_172080

theorem unique_solution_mnk :
  ∀ (m n k : ℕ), 3^n + 4^m = 5^k → (m, n, k) = (0, 1, 1) :=
by
  intros m n k h
  sorry

end NUMINAMATH_GPT_unique_solution_mnk_l1720_172080


namespace NUMINAMATH_GPT_mod_problem_l1720_172064

theorem mod_problem (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 21 [ZMOD 25]) : (x^2 ≡ 21 [ZMOD 25]) :=
sorry

end NUMINAMATH_GPT_mod_problem_l1720_172064


namespace NUMINAMATH_GPT_percent_of_x_eq_21_percent_l1720_172061

theorem percent_of_x_eq_21_percent (x : Real) : (0.21 * x = 0.30 * 0.70 * x) := by
  sorry

end NUMINAMATH_GPT_percent_of_x_eq_21_percent_l1720_172061


namespace NUMINAMATH_GPT_value_of_k_l1720_172070

open Nat

theorem value_of_k (k : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, 0 < n → S n = k * (n : ℝ) ^ 2 + (n : ℝ))
  (h_a : ∀ n : ℕ, 1 < n → a n = S n - S (n-1))
  (h_geom : ∀ m : ℕ, 0 < m → (a m) ≠ 0 → (a (2*m))^2 = a m * a (4*m)) :
  k = 0 ∨ k = 1 :=
sorry

end NUMINAMATH_GPT_value_of_k_l1720_172070


namespace NUMINAMATH_GPT_product_of_integers_l1720_172066

theorem product_of_integers (A B C D : ℕ) 
  (h1 : A + B + C + D = 100) 
  (h2 : 2^A = B - 6) 
  (h3 : C + 6 = D)
  (h4 : B + C = D + 10) : 
  A * B * C * D = 33280 := 
by
  sorry

end NUMINAMATH_GPT_product_of_integers_l1720_172066


namespace NUMINAMATH_GPT_charge_two_hours_l1720_172002

def charge_first_hour (F A : ℝ) : Prop := F = A + 25
def total_charge_five_hours (F A : ℝ) : Prop := F + 4 * A = 250
def total_charge_two_hours (F A : ℝ) : Prop := F + A = 115

theorem charge_two_hours (F A : ℝ) 
  (h1 : charge_first_hour F A)
  (h2 : total_charge_five_hours F A) : 
  total_charge_two_hours F A :=
by
  sorry

end NUMINAMATH_GPT_charge_two_hours_l1720_172002


namespace NUMINAMATH_GPT_lava_lamp_probability_l1720_172098

/-- Ryan has 4 red lava lamps and 2 blue lava lamps; 
    he arranges them in a row on a shelf randomly, and then randomly turns 3 of them on. 
    Prove that the probability that the leftmost lamp is blue and off, 
    and the rightmost lamp is red and on is 2/25. -/
theorem lava_lamp_probability : 
  let total_arrangements := (Nat.choose 6 2) 
  let total_on := (Nat.choose 6 3)
  let favorable_arrangements := (Nat.choose 4 1)
  let favorable_on := (Nat.choose 4 2)
  let favorable_outcomes := 4 * 6
  let probability := (favorable_outcomes : ℚ) / (total_arrangements * total_on : ℚ)
  probability = 2 / 25 := 
by
  sorry

end NUMINAMATH_GPT_lava_lamp_probability_l1720_172098


namespace NUMINAMATH_GPT_fractions_are_integers_l1720_172040

theorem fractions_are_integers (a b : ℕ) (h1 : 1 < a) (h2 : 1 < b) 
    (h3 : abs ((a : ℚ) / b - (a - 1) / (b - 1)) = 1) : 
    ∃ m n : ℤ, (a : ℚ) / b = m ∧ (a - 1) / (b - 1) = n := 
sorry

end NUMINAMATH_GPT_fractions_are_integers_l1720_172040


namespace NUMINAMATH_GPT_combined_weight_of_contents_l1720_172073

theorem combined_weight_of_contents
    (weight_pencil : ℝ := 28.3)
    (weight_eraser : ℝ := 15.7)
    (weight_paperclip : ℝ := 3.5)
    (weight_stapler : ℝ := 42.2)
    (num_pencils : ℕ := 5)
    (num_erasers : ℕ := 3)
    (num_paperclips : ℕ := 4)
    (num_staplers : ℕ := 2) :
    num_pencils * weight_pencil +
    num_erasers * weight_eraser +
    num_paperclips * weight_paperclip +
    num_staplers * weight_stapler = 287 := 
sorry

end NUMINAMATH_GPT_combined_weight_of_contents_l1720_172073


namespace NUMINAMATH_GPT_range_of_b_l1720_172031

-- Definitions
def polynomial_inequality (b : ℝ) (x : ℝ) : Prop := x^2 + b * x - b - 3/4 > 0

-- The main statement
theorem range_of_b (b : ℝ) : (∀ x : ℝ, polynomial_inequality b x) ↔ -3 < b ∧ b < -1 :=
by {
    sorry -- proof goes here
}

end NUMINAMATH_GPT_range_of_b_l1720_172031


namespace NUMINAMATH_GPT_fraction_length_EF_of_GH_l1720_172069

theorem fraction_length_EF_of_GH (GH GE EH GF FH EF : ℝ)
  (h1 : GE = 3 * EH)
  (h2 : GF = 4 * FH)
  (h3 : GE + EH = GH)
  (h4 : GF + FH = GH) :
  EF / GH = 1 / 20 := by 
  sorry

end NUMINAMATH_GPT_fraction_length_EF_of_GH_l1720_172069


namespace NUMINAMATH_GPT_rectangle_dimensions_l1720_172053

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : l = 2 * w)
  (h2 : 2 * l + 2 * w = 3 * (l * w)) : 
  w = 1 ∧ l = 2 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1720_172053


namespace NUMINAMATH_GPT_simplify_fraction_l1720_172068

variable {x y : ℝ}

theorem simplify_fraction (hx : x = 3) (hy : y = 4) : (12 * x * y^3) / (9 * x^3 * y^2) = 16 / 27 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1720_172068


namespace NUMINAMATH_GPT_last_digit_of_two_exp_sum_l1720_172041

theorem last_digit_of_two_exp_sum (m : ℕ) (h : 0 < m) : 
  ((2 ^ (m + 2007) + 2 ^ (m + 1)) % 10) = 0 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_last_digit_of_two_exp_sum_l1720_172041


namespace NUMINAMATH_GPT_linear_function_value_l1720_172072

theorem linear_function_value
  (a b c : ℝ)
  (h1 : 3 * a + b = 8)
  (h2 : -2 * a + b = 3)
  (h3 : -3 * a + b = c) :
  a^2 + b^2 + c^2 - a * b - b * c - a * c = 13 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_value_l1720_172072


namespace NUMINAMATH_GPT_shaded_rectangle_ratio_l1720_172065

/-- Define conditions involved in the problem -/
def side_length_large_square : ℕ := 50
def num_rows_cols_grid : ℕ := 5
def rows_spanned_rect : ℕ := 2
def cols_spanned_rect : ℕ := 3

/-- Calculate the side length of a small square in the grid -/
def side_length_small_square := side_length_large_square / num_rows_cols_grid

/-- Calculate the area of the large square -/
def area_large_square := side_length_large_square * side_length_large_square

/-- Calculate the area of the shaded rectangle -/
def area_shaded_rectangle :=
  (rows_spanned_rect * side_length_small_square) *
  (cols_spanned_rect * side_length_small_square)

/-- Prove the ratio of the shaded rectangle's area to the large square's area -/
theorem shaded_rectangle_ratio : 
  (area_shaded_rectangle : ℚ) / area_large_square = 6/25 := by
  sorry

end NUMINAMATH_GPT_shaded_rectangle_ratio_l1720_172065


namespace NUMINAMATH_GPT_water_bottle_size_l1720_172021

-- Define conditions
def glasses_per_day : ℕ := 4
def ounces_per_glass : ℕ := 5
def fills_per_week : ℕ := 4
def days_per_week : ℕ := 7

-- Theorem statement
theorem water_bottle_size :
  (glasses_per_day * ounces_per_glass * days_per_week) / fills_per_week = 35 :=
by
  sorry

end NUMINAMATH_GPT_water_bottle_size_l1720_172021


namespace NUMINAMATH_GPT_quadratic_translation_transformed_l1720_172025

-- The original function is defined as follows:
def original_func (x : ℝ) : ℝ := 2 * x^2

-- Translated function left by 3 units
def translate_left (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x + a)

-- Translated function down by 2 units
def translate_down (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f x - b

-- Combine both translations: left by 3 units and down by 2 units
def translated_func (x : ℝ) : ℝ := translate_down (translate_left original_func 3) 2 x

-- The theorem we want to prove
theorem quadratic_translation_transformed :
  translated_func x = 2 * (x + 3)^2 - 2 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_translation_transformed_l1720_172025


namespace NUMINAMATH_GPT_euler_totient_problem_l1720_172058

open Nat

def is_odd (n : ℕ) := n % 2 = 1

def is_power_of_2 (m : ℕ) := ∃ k : ℕ, m = 2^k

theorem euler_totient_problem (n : ℕ) (h1 : n > 0) (h2 : is_odd n) (h3 : is_power_of_2 (φ n)) (h4 : is_power_of_2 (φ (n + 1))) :
  is_power_of_2 (n + 1) ∨ n = 5 := 
sorry

end NUMINAMATH_GPT_euler_totient_problem_l1720_172058


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1720_172019

theorem quadratic_inequality_solution {a b : ℝ} 
  (h1 : (∀ x : ℝ, ax^2 - bx - 1 ≥ 0 ↔ (x = 1/3 ∨ x = 1/2))) : 
  ∃ a b : ℝ, (∀ x : ℝ, x^2 - b * x - a < 0 ↔ (-3 < x ∧ x < -2)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1720_172019


namespace NUMINAMATH_GPT_tiling_possible_l1720_172005

theorem tiling_possible (n x : ℕ) (hx : 7 * x = n^2) : ∃ k : ℕ, n = 7 * k :=
by sorry

end NUMINAMATH_GPT_tiling_possible_l1720_172005


namespace NUMINAMATH_GPT_average_score_of_class_l1720_172067

variable (students_total : ℕ) (group1_students : ℕ) (group2_students : ℕ)
variable (group1_avg : ℝ) (group2_avg : ℝ)

theorem average_score_of_class :
  students_total = 20 → 
  group1_students = 10 → 
  group2_students = 10 → 
  group1_avg = 80 → 
  group2_avg = 60 → 
  (group1_students * group1_avg + group2_students * group2_avg) / students_total = 70 := 
by
  intros students_total_eq group1_students_eq group2_students_eq group1_avg_eq group2_avg_eq
  rw [students_total_eq, group1_students_eq, group2_students_eq, group1_avg_eq, group2_avg_eq]
  simp
  sorry

end NUMINAMATH_GPT_average_score_of_class_l1720_172067


namespace NUMINAMATH_GPT_point_on_line_l1720_172055

theorem point_on_line (k : ℝ) (x y : ℝ) (h : x = -1/3 ∧ y = 4) (line_eq : 1 + 3 * k * x = -4 * y) : k = 17 :=
by
  rcases h with ⟨hx, hy⟩
  sorry

end NUMINAMATH_GPT_point_on_line_l1720_172055


namespace NUMINAMATH_GPT_smallest_integer_cube_ends_in_528_l1720_172075

theorem smallest_integer_cube_ends_in_528 :
  ∃ (n : ℕ), (n^3 % 1000 = 528 ∧ ∀ m : ℕ, (m^3 % 1000 = 528) → m ≥ n) ∧ n = 428 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_cube_ends_in_528_l1720_172075


namespace NUMINAMATH_GPT_probability_of_b_in_rabbit_l1720_172016

theorem probability_of_b_in_rabbit : 
  let word := "rabbit"
  let total_letters := 6
  let num_b_letters := 2
  (num_b_letters : ℚ) / total_letters = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_b_in_rabbit_l1720_172016


namespace NUMINAMATH_GPT_find_number_l1720_172038

-- Define the certain number x
variable (x : ℤ)

-- Define the conditions as given in part a)
def conditions : Prop :=
  x + 10 - 2 = 44

-- State the theorem that we need to prove
theorem find_number (h : conditions x) : x = 36 :=
by sorry

end NUMINAMATH_GPT_find_number_l1720_172038


namespace NUMINAMATH_GPT_percentage_of_additional_money_is_10_l1720_172083

-- Define the conditions
def months := 11
def payment_per_month := 15
def total_borrowed := 150

-- Define the function to calculate the total amount paid
def total_paid (months payment_per_month : ℕ) : ℕ :=
  months * payment_per_month

-- Define the function to calculate the additional amount paid
def additional_paid (total_paid total_borrowed : ℕ) : ℕ :=
  total_paid - total_borrowed

-- Define the function to calculate the percentage of the additional amount
def percentage_additional (additional_paid total_borrowed : ℕ) : ℕ :=
  (additional_paid * 100) / total_borrowed

-- State the theorem to prove the percentage of the additional money is 10%
theorem percentage_of_additional_money_is_10 :
  percentage_additional (additional_paid (total_paid months payment_per_month) total_borrowed) total_borrowed = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_additional_money_is_10_l1720_172083


namespace NUMINAMATH_GPT_selling_price_correct_l1720_172011

-- Define the conditions
def cost_per_cupcake : ℝ := 0.75
def total_cupcakes_burnt : ℕ := 24
def total_eaten_first : ℕ := 5
def total_eaten_later : ℕ := 4
def net_profit : ℝ := 24
def total_cupcakes_made : ℕ := 72
def total_cost : ℝ := total_cupcakes_made * cost_per_cupcake
def total_eaten : ℕ := total_eaten_first + total_eaten_later
def total_sold : ℕ := total_cupcakes_made - total_eaten
def revenue (P : ℝ) : ℝ := total_sold * P

-- Prove the correctness of the selling price P
theorem selling_price_correct : 
  ∃ P : ℝ, revenue P - total_cost = net_profit ∧ (P = 1.24) :=
by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l1720_172011


namespace NUMINAMATH_GPT_calculate_expression_l1720_172099

theorem calculate_expression : 
  |(-3)| - 2 * Real.tan (Real.pi / 4) + (-1:ℤ)^(2023) - (Real.sqrt 3 - Real.pi)^(0:ℤ) = -1 :=
  by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1720_172099


namespace NUMINAMATH_GPT_charity_race_finished_racers_l1720_172013

theorem charity_race_finished_racers :
  let initial_racers := 50
  let joined_after_20_minutes := 30
  let doubled_after_30_minutes := 2
  let dropped_racers := 30
  let total_racers_after_20_minutes := initial_racers + joined_after_20_minutes
  let total_racers_after_50_minutes := total_racers_after_20_minutes * doubled_after_30_minutes
  let finished_racers := total_racers_after_50_minutes - dropped_racers
  finished_racers = 130 := by
    sorry

end NUMINAMATH_GPT_charity_race_finished_racers_l1720_172013


namespace NUMINAMATH_GPT_actual_distance_traveled_l1720_172071

theorem actual_distance_traveled 
  (D : ℝ) (t : ℝ)
  (h1 : 8 * t = D)
  (h2 : 12 * t = D + 20) : 
  D = 40 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l1720_172071


namespace NUMINAMATH_GPT_find_a_l1720_172096

noncomputable def f (a x : ℝ) : ℝ := a^x - 4 * a + 3

theorem find_a (H : ∃ (a : ℝ), ∃ (x y : ℝ), f a x = y ∧ f y x = a ∧ x = 2 ∧ y = -1): ∃ a : ℝ, a = 2 :=
by
  obtain ⟨a, x, y, hx, hy, hx2, hy1⟩ := H
  --skipped proof
  sorry

end NUMINAMATH_GPT_find_a_l1720_172096


namespace NUMINAMATH_GPT_negation_of_universal_l1720_172046

theorem negation_of_universal : 
  (¬ (∀ x : ℝ, 2 * x^2 - x + 1 ≥ 0)) ↔ (∃ x : ℝ, 2 * x^2 - x + 1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_l1720_172046


namespace NUMINAMATH_GPT_isosceles_triangles_height_ratio_l1720_172049

theorem isosceles_triangles_height_ratio
  (b1 b2 h1 h2 : ℝ)
  (h1_ne_zero : h1 ≠ 0) 
  (h2_ne_zero : h2 ≠ 0)
  (equal_vertical_angles : ∀ (a1 a2 : ℝ), true) -- Placeholder for equal angles since it's not used directly
  (areas_ratio : (b1 * h1) / (b2 * h2) = 16 / 36)
  (similar_triangles : b1 / b2 = h1 / h2) :
  h1 / h2 = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangles_height_ratio_l1720_172049


namespace NUMINAMATH_GPT_john_total_amount_l1720_172051

-- Given conditions from a)
def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount

-- Problem statement
theorem john_total_amount : grandpa_amount + grandma_amount = 120 :=
by
  sorry

end NUMINAMATH_GPT_john_total_amount_l1720_172051


namespace NUMINAMATH_GPT_not_sum_of_squares_or_cubes_in_ap_l1720_172028

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a + b * b = n

def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a * a + b * b * b = n

def arithmetic_progression (a d k : ℕ) : ℕ :=
  a + d * k

theorem not_sum_of_squares_or_cubes_in_ap :
  ∀ k : ℕ, ¬ is_sum_of_two_squares (arithmetic_progression 31 36 k) ∧
           ¬ is_sum_of_two_cubes (arithmetic_progression 31 36 k) := by
  sorry

end NUMINAMATH_GPT_not_sum_of_squares_or_cubes_in_ap_l1720_172028


namespace NUMINAMATH_GPT_base_number_is_five_l1720_172077

variable (a x y : Real)

theorem base_number_is_five (h1 : xy = 1) (h2 : (a ^ (x + y) ^ 2) / (a ^ (x - y) ^ 2) = 625) : a = 5 := 
sorry

end NUMINAMATH_GPT_base_number_is_five_l1720_172077


namespace NUMINAMATH_GPT_no_integer_roots_if_coefficients_are_odd_l1720_172050

theorem no_integer_roots_if_coefficients_are_odd (a b c x : ℤ) 
  (h1 : Odd a) (h2 : Odd b) (h3 : Odd c) (h4 : a * x^2 + b * x + c = 0) : False := 
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_if_coefficients_are_odd_l1720_172050


namespace NUMINAMATH_GPT_solution_exists_l1720_172052

namespace EquationSystem
-- Given the conditions of the equation system:
def eq1 (a b c d : ℝ) := a * b + a * c = 3 * b + 3 * c
def eq2 (a b c d : ℝ) := b * c + b * d = 5 * c + 5 * d
def eq3 (a b c d : ℝ) := a * c + c * d = 7 * a + 7 * d
def eq4 (a b c d : ℝ) := a * d + b * d = 9 * a + 9 * b

-- We need to prove that the solutions are as described:
theorem solution_exists (a b c d : ℝ) :
  eq1 a b c d → eq2 a b c d → eq3 a b c d → eq4 a b c d →
  (a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) ∨ ∃ t : ℝ, a = t ∧ b = -t ∧ c = t ∧ d = -t :=
  by
    sorry
end EquationSystem

end NUMINAMATH_GPT_solution_exists_l1720_172052


namespace NUMINAMATH_GPT_sequence_sixth_term_l1720_172032

theorem sequence_sixth_term (a b c d : ℚ) : 
  (a = 1/4 * (5 + b)) →
  (b = 1/4 * (a + 45)) →
  (45 = 1/4 * (b + c)) →
  (c = 1/4 * (45 + d)) →
  d = 1877 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sixth_term_l1720_172032


namespace NUMINAMATH_GPT_watermelons_remaining_l1720_172092

theorem watermelons_remaining :
  let initial_watermelons := 10 * 12
  let yesterdays_sale := 0.40 * initial_watermelons
  let remaining_after_yesterday := initial_watermelons - yesterdays_sale
  let todays_sale := (1 / 4) * remaining_after_yesterday
  let remaining_after_today := remaining_after_yesterday - todays_sale
  let tomorrows_sales := 1.5 * todays_sale
  let remaining_after_tomorrow := remaining_after_today - tomorrows_sales
  remaining_after_tomorrow = 27 :=
by
  sorry

end NUMINAMATH_GPT_watermelons_remaining_l1720_172092


namespace NUMINAMATH_GPT_arrow_estimate_closest_to_9_l1720_172059

theorem arrow_estimate_closest_to_9 
  (a b : ℝ) (h₁ : a = 8.75) (h₂ : b = 9.0)
  (h : 8.75 < 9.0) :
  ∃ x ∈ Set.Icc a b, x = 9.0 :=
by
  sorry

end NUMINAMATH_GPT_arrow_estimate_closest_to_9_l1720_172059


namespace NUMINAMATH_GPT_sequence_eighth_term_is_sixteen_l1720_172037

-- Define the sequence based on given patterns
def oddPositionTerm (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

def evenPositionTerm (n : ℕ) : ℕ :=
  4 + 4 * (n - 1)

-- Formalize the proof problem
theorem sequence_eighth_term_is_sixteen : evenPositionTerm 4 = 16 :=
by 
  unfold evenPositionTerm
  sorry

end NUMINAMATH_GPT_sequence_eighth_term_is_sixteen_l1720_172037


namespace NUMINAMATH_GPT_contrapositive_negation_l1720_172004

-- Define the main condition of the problem
def statement_p (x y : ℝ) : Prop :=
  (x - 1) * (y + 2) = 0 → (x = 1 ∨ y = -2)

-- Prove the contrapositive of statement_p
theorem contrapositive (x y : ℝ) : 
  (x ≠ 1 ∧ y ≠ -2) → ¬ ((x - 1) * (y + 2) = 0) :=
by 
  sorry

-- Prove the negation of statement_p
theorem negation (x y : ℝ) : 
  ((x - 1) * (y + 2) = 0) → ¬ (x = 1 ∨ y = -2) :=
by 
  sorry

end NUMINAMATH_GPT_contrapositive_negation_l1720_172004


namespace NUMINAMATH_GPT_infinite_solutions_imply_values_l1720_172033

theorem infinite_solutions_imply_values (a b : ℝ) :
  (∀ x : ℝ, a * (2 * x + b) = 12 * x + 5) ↔ (a = 6 ∧ b = 5 / 6) :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_imply_values_l1720_172033


namespace NUMINAMATH_GPT_value_v3_at_1_horners_method_l1720_172015

def f (x : ℝ) : ℝ := 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem value_v3_at_1_horners_method :
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  v3 = 7.9 :=
by
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  exact sorry

end NUMINAMATH_GPT_value_v3_at_1_horners_method_l1720_172015


namespace NUMINAMATH_GPT_compute_fraction_l1720_172090

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x^2 - x + 1

theorem compute_fraction : (f (g (f 1))) / (g (f (g 1))) = 6801 / 281 := 
by 
  sorry

end NUMINAMATH_GPT_compute_fraction_l1720_172090


namespace NUMINAMATH_GPT_no_positive_integral_solutions_l1720_172035

theorem no_positive_integral_solutions (x y : ℕ) (h : x > 0) (k : y > 0) :
  x^4 * y^4 - 8 * x^2 * y^2 + 12 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integral_solutions_l1720_172035


namespace NUMINAMATH_GPT_ajay_total_gain_l1720_172088

noncomputable def ajay_gain : ℝ :=
  let cost1 := 15 * 14.50
  let cost2 := 10 * 13
  let total_cost := cost1 + cost2
  let total_weight := 15 + 10
  let selling_price := total_weight * 15
  selling_price - total_cost

theorem ajay_total_gain :
  ajay_gain = 27.50 := by
  sorry

end NUMINAMATH_GPT_ajay_total_gain_l1720_172088


namespace NUMINAMATH_GPT_velocity_at_3_seconds_l1720_172078

variable (t : ℝ)
variable (s : ℝ)

def motion_eq (t : ℝ) : ℝ := 1 + t + t^2

theorem velocity_at_3_seconds : 
  (deriv motion_eq 3) = 7 :=
by
  sorry

end NUMINAMATH_GPT_velocity_at_3_seconds_l1720_172078


namespace NUMINAMATH_GPT_probability_X_equals_3_l1720_172027

def total_score (a b : ℕ) : ℕ :=
  a + b

def prob_event_A_draws_yellow_B_draws_white : ℚ :=
  (2 / 5) * (3 / 4)

def prob_event_A_draws_white_B_draws_yellow : ℚ :=
  (3 / 5) * (2 / 4)

def prob_X_equals_3 : ℚ :=
  prob_event_A_draws_yellow_B_draws_white + prob_event_A_draws_white_B_draws_yellow

theorem probability_X_equals_3 :
  prob_X_equals_3 = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_X_equals_3_l1720_172027


namespace NUMINAMATH_GPT_measurable_masses_l1720_172086

theorem measurable_masses (k : ℤ) (h : -121 ≤ k ∧ k ≤ 121) : 
  ∃ (a b c d e : ℤ), k = a * 1 + b * 3 + c * 9 + d * 27 + e * 81 ∧ 
  (a = -1 ∨ a = 0 ∨ a = 1) ∧
  (b = -1 ∨ b = 0 ∨ b = 1) ∧
  (c = -1 ∨ c = 0 ∨ c = 1) ∧
  (d = -1 ∨ d = 0 ∨ d = 1) ∧
  (e = -1 ∨ e = 0 ∨ e = 1) :=
sorry

end NUMINAMATH_GPT_measurable_masses_l1720_172086


namespace NUMINAMATH_GPT_number_of_classes_min_wins_for_class2101_l1720_172060

-- Proof Problem for Q1
theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 := sorry

-- Proof Problem for Q2
theorem min_wins_for_class2101 (y : ℕ) (h : y + (9 - y) = 9 ∧ 2 * y + (9 - y) >= 14) : y >= 5 := sorry

end NUMINAMATH_GPT_number_of_classes_min_wins_for_class2101_l1720_172060


namespace NUMINAMATH_GPT_total_spent_on_video_games_l1720_172017

theorem total_spent_on_video_games (cost_basketball cost_racing : ℝ) (h_ball : cost_basketball = 5.20) (h_race : cost_racing = 4.23) : 
  cost_basketball + cost_racing = 9.43 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_on_video_games_l1720_172017


namespace NUMINAMATH_GPT_value_of_r_squared_plus_s_squared_l1720_172091

theorem value_of_r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 24) (h2 : r + s = 10) :
  r^2 + s^2 = 52 :=
sorry

end NUMINAMATH_GPT_value_of_r_squared_plus_s_squared_l1720_172091


namespace NUMINAMATH_GPT_min_expression_value_l1720_172076

theorem min_expression_value (x y z : ℝ) (xyz_eq : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n : ℝ, (∀ x y z : ℝ, x * y * z = 1 → 0 < x → 0 < y → 0 < z → 2 * x^2 + 8 * x * y + 32 * y^2 + 16 * y * z + 8 * z^2 ≥ n)
    ∧ n = 72 :=
sorry

end NUMINAMATH_GPT_min_expression_value_l1720_172076


namespace NUMINAMATH_GPT_right_triangle_leg_length_l1720_172057

theorem right_triangle_leg_length
  (a : ℕ) (c : ℕ) (h₁ : a = 8) (h₂ : c = 17) :
  ∃ b : ℕ, a^2 + b^2 = c^2 ∧ b = 15 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_leg_length_l1720_172057


namespace NUMINAMATH_GPT_M_inter_N_eq_M_l1720_172003

-- Definitions of the sets M and N
def M : Set ℝ := {x | abs (x - 1) < 1}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- The desired equality
theorem M_inter_N_eq_M : M ∩ N = M := 
by
  sorry

end NUMINAMATH_GPT_M_inter_N_eq_M_l1720_172003


namespace NUMINAMATH_GPT_blue_length_of_pencil_l1720_172045

theorem blue_length_of_pencil (total_length purple_length black_length blue_length : ℝ)
  (h1 : total_length = 6)
  (h2 : purple_length = 3)
  (h3 : black_length = 2)
  (h4 : total_length = purple_length + black_length + blue_length)
  : blue_length = 1 :=
by
  sorry

end NUMINAMATH_GPT_blue_length_of_pencil_l1720_172045


namespace NUMINAMATH_GPT_kho_kho_only_l1720_172087

theorem kho_kho_only (K H B total : ℕ) (h1 : K + B = 10) (h2 : B = 5) (h3 : K + H + B = 25) : H = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_kho_kho_only_l1720_172087


namespace NUMINAMATH_GPT_opposite_neg_three_over_two_l1720_172094

-- Define the concept of the opposite number
def opposite (x : ℚ) : ℚ := -x

-- State the problem: The opposite number of -3/2 is 3/2
theorem opposite_neg_three_over_two :
  opposite (- (3 / 2 : ℚ)) = (3 / 2 : ℚ) := 
  sorry

end NUMINAMATH_GPT_opposite_neg_three_over_two_l1720_172094


namespace NUMINAMATH_GPT_simplify_expression_l1720_172062

theorem simplify_expression : 
  18 * (8 / 15) * (3 / 4) = 12 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1720_172062


namespace NUMINAMATH_GPT_binomial_coefficient_divisible_by_p_l1720_172009

theorem binomial_coefficient_divisible_by_p (p k : ℕ) (hp : Nat.Prime p) (hk1 : 0 < k) (hk2 : k < p) :
  p ∣ (Nat.factorial p / (Nat.factorial k * Nat.factorial (p - k))) :=
by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_divisible_by_p_l1720_172009


namespace NUMINAMATH_GPT_artist_paints_37_sq_meters_l1720_172085

-- Define the structure of the sculpture
def top_layer : ℕ := 1
def middle_layer : ℕ := 5
def bottom_layer : ℕ := 11
def edge_length : ℕ := 1

-- Define the exposed surface areas
def exposed_surface_top_layer := 5 * top_layer
def exposed_surface_middle_layer := 1 * 5 + 4 * 4
def exposed_surface_bottom_layer := bottom_layer

-- Calculate the total exposed surface area
def total_exposed_surface_area := exposed_surface_top_layer + exposed_surface_middle_layer + exposed_surface_bottom_layer

-- The final theorem statement
theorem artist_paints_37_sq_meters (hyp1 : top_layer = 1)
  (hyp2 : middle_layer = 5)
  (hyp3 : bottom_layer = 11)
  (hyp4 : edge_length = 1)
  : total_exposed_surface_area = 37 := 
by
  sorry

end NUMINAMATH_GPT_artist_paints_37_sq_meters_l1720_172085


namespace NUMINAMATH_GPT_volume_of_prism_l1720_172095

noncomputable def volume_of_triangular_prism
  (area_lateral_face : ℝ)
  (distance_cc1_to_lateral_face : ℝ) : ℝ :=
  area_lateral_face * distance_cc1_to_lateral_face

theorem volume_of_prism (area_lateral_face : ℝ) 
    (distance_cc1_to_lateral_face : ℝ)
    (h_area : area_lateral_face = 4)
    (h_distance : distance_cc1_to_lateral_face = 2):
  volume_of_triangular_prism area_lateral_face distance_cc1_to_lateral_face = 4 := by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l1720_172095


namespace NUMINAMATH_GPT_complete_the_square_eqn_l1720_172001

theorem complete_the_square_eqn (x b c : ℤ) (h_eqn : x^2 - 10 * x + 15 = 0) (h_form : (x + b)^2 = c) : b + c = 5 := by
  sorry

end NUMINAMATH_GPT_complete_the_square_eqn_l1720_172001
