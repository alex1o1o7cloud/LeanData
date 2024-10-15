import Mathlib

namespace NUMINAMATH_GPT_remainder_when_divided_by_6_l750_75023

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : n % 12 = 8) : n % 6 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_6_l750_75023


namespace NUMINAMATH_GPT_necklace_cost_l750_75093

def bead_necklaces := 3
def gemstone_necklaces := 3
def total_necklaces := bead_necklaces + gemstone_necklaces
def total_earnings := 36

theorem necklace_cost :
  (total_earnings / total_necklaces) = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_necklace_cost_l750_75093


namespace NUMINAMATH_GPT_proof_problem_l750_75048

variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Define what it means for a function to be increasing on (-∞, 0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y < 0 → f x < f y

-- Define what it means for a function to be decreasing on (0, +∞)
def is_decreasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f y < f x

theorem proof_problem 
  (h_even : is_even_function f) 
  (h_inc_neg : is_increasing_on_neg f) : 
  (∀ x : ℝ, f (-x) - f x = 0) ∧ (is_decreasing_on_pos f) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l750_75048


namespace NUMINAMATH_GPT_percent_exceed_not_ticketed_l750_75035

-- Defining the given conditions
def total_motorists : ℕ := 100
def percent_exceed_limit : ℕ := 50
def percent_with_tickets : ℕ := 40

-- Calculate the number of motorists exceeding the limit and receiving tickets
def motorists_exceed_limit := total_motorists * percent_exceed_limit / 100
def motorists_with_tickets := total_motorists * percent_with_tickets / 100

-- Theorem: Percentage of motorists exceeding the limit but not receiving tickets
theorem percent_exceed_not_ticketed : 
  (motorists_exceed_limit - motorists_with_tickets) * 100 / motorists_exceed_limit = 20 := 
by
  sorry

end NUMINAMATH_GPT_percent_exceed_not_ticketed_l750_75035


namespace NUMINAMATH_GPT_identify_genuine_coins_l750_75070

section IdentifyGenuineCoins

variables (coins : Fin 25 → ℝ) 
          (is_genuine : Fin 25 → Prop) 
          (is_counterfeit : Fin 25 → Prop)

-- Conditions
axiom coin_total : ∀ i, is_genuine i ∨ is_counterfeit i
axiom genuine_count : ∃ s : Finset (Fin 25), s.card = 22 ∧ ∀ i ∈ s, is_genuine i
axiom counterfeit_count : ∃ t : Finset (Fin 25), t.card = 3 ∧ ∀ i ∈ t, is_counterfeit i
axiom genuine_weight : ∃ w : ℝ, ∀ i, is_genuine i → coins i = w
axiom counterfeit_weight : ∃ c : ℝ, ∀ i, is_counterfeit i → coins i = c
axiom counterfeit_lighter : ∀ (w c : ℝ), (∃ i, is_genuine i → coins i = w) ∧ (∃ j, is_counterfeit j → coins j = c) → c < w

-- Theorem: Identifying 6 genuine coins using two weighings
theorem identify_genuine_coins : ∃ s : Finset (Fin 25), s.card = 6 ∧ ∀ i ∈ s, is_genuine i :=
sorry

end IdentifyGenuineCoins

end NUMINAMATH_GPT_identify_genuine_coins_l750_75070


namespace NUMINAMATH_GPT_transformation_correct_l750_75012

noncomputable def original_function (x : ℝ) : ℝ := 2^x
noncomputable def transformed_function (x : ℝ) : ℝ := 2^x - 1
noncomputable def log_function (x : ℝ) : ℝ := Real.log x / Real.log 2 + 1

theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = log_function (original_function x) :=
by
  intros x
  rw [transformed_function, log_function, original_function]
  sorry

end NUMINAMATH_GPT_transformation_correct_l750_75012


namespace NUMINAMATH_GPT_sin_alpha_sub_beta_cos_beta_l750_75081

variables (α β : ℝ)
variables (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
variables (h1 : Real.sin α = 3 / 5)
variables (h2 : Real.tan (α - β) = -1 / 3)

theorem sin_alpha_sub_beta : Real.sin (α - β) = - Real.sqrt 10 / 10 :=
by
  sorry

theorem cos_beta : Real.cos β = 9 * Real.sqrt 10 / 50 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_sub_beta_cos_beta_l750_75081


namespace NUMINAMATH_GPT_painting_cost_3x_l750_75032

-- Define the dimensions of the original room and the painting cost
variables (L B H : ℝ)
def cost_of_painting (area : ℝ) : ℝ := 350

-- Create a definition for the calculation of area
def paint_area (L B H : ℝ) : ℝ := 2 * (L * H + B * H)

-- Define the new dimensions
def new_dimensions (L B H : ℝ) : ℝ × ℝ × ℝ := (3 * L, 3 * B, 3 * H)

-- Create a definition for the calculation of the new area
def new_paint_area (L B H : ℝ) : ℝ := 18 * (paint_area L B H)

-- Calculate the new cost
def new_cost (L B H : ℝ) : ℝ := 18 * cost_of_painting (paint_area L B H)

-- The theorem to be proved
theorem painting_cost_3x (L B H : ℝ) : new_cost L B H = 6300 :=
by 
  simp [new_cost, cost_of_painting, paint_area]
  sorry

end NUMINAMATH_GPT_painting_cost_3x_l750_75032


namespace NUMINAMATH_GPT_sum_of_coeffs_eq_59049_l750_75050

-- Definition of the polynomial
def poly (x y z : ℕ) : ℕ :=
  (2 * x - 3 * y + 4 * z) ^ 10

-- Conjecture: The sum of the numerical coefficients in poly when x, y, and z are set to 1 is 59049
theorem sum_of_coeffs_eq_59049 : poly 1 1 1 = 59049 := by
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_eq_59049_l750_75050


namespace NUMINAMATH_GPT_import_rate_for_rest_of_1997_l750_75080

theorem import_rate_for_rest_of_1997
    (import_1996: ℝ)
    (import_first_two_months_1997: ℝ)
    (excess_imports_1997: ℝ)
    (import_rate_first_two_months: ℝ)
    (expected_total_imports_1997: ℝ)
    (remaining_imports_1997: ℝ)
    (R: ℝ):
    excess_imports_1997 = 720e6 →
    expected_total_imports_1997 = import_1996 + excess_imports_1997 →
    remaining_imports_1997 = expected_total_imports_1997 - import_first_two_months_1997 →
    10 * R = remaining_imports_1997 →
    R = 180e6 :=
by
    intros h_import1996 h_import_first_two_months h_excess_imports h_import_rate_first_two_months 
           h_expected_total_imports h_remaining_imports h_equation
    sorry

end NUMINAMATH_GPT_import_rate_for_rest_of_1997_l750_75080


namespace NUMINAMATH_GPT_chocolate_oranges_initial_l750_75072

theorem chocolate_oranges_initial (p_c p_o G n_c x : ℕ) 
  (h_candy_bar_price : p_c = 5) 
  (h_orange_price : p_o = 10) 
  (h_goal : G = 1000) 
  (h_candy_bars_sold : n_c = 160) 
  (h_equation : G = p_o * x + p_c * n_c) : 
  x = 20 := 
by
  sorry

end NUMINAMATH_GPT_chocolate_oranges_initial_l750_75072


namespace NUMINAMATH_GPT_necessarily_negative_l750_75075

theorem necessarily_negative
  (a b c : ℝ)
  (ha : -2 < a ∧ a < -1)
  (hb : 0 < b ∧ b < 1)
  (hc : -1 < c ∧ c < 0) :
  b + c < 0 :=
sorry

end NUMINAMATH_GPT_necessarily_negative_l750_75075


namespace NUMINAMATH_GPT_school_competition_students_l750_75067

theorem school_competition_students (n : ℤ)
  (h1 : 100 < n) 
  (h2 : n < 200) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2) 
  (h5 : n % 6 = 2) :
  n = 122 ∨ n = 182 :=
sorry

end NUMINAMATH_GPT_school_competition_students_l750_75067


namespace NUMINAMATH_GPT_connie_earbuds_tickets_l750_75027

theorem connie_earbuds_tickets (total_tickets : ℕ) (koala_fraction : ℕ) (bracelet_tickets : ℕ) (earbud_tickets : ℕ) :
  total_tickets = 50 →
  koala_fraction = 2 →
  bracelet_tickets = 15 →
  (total_tickets / koala_fraction) + bracelet_tickets + earbud_tickets = total_tickets →
  earbud_tickets = 10 :=
by
  intros h_total h_koala h_bracelets h_sum
  sorry

end NUMINAMATH_GPT_connie_earbuds_tickets_l750_75027


namespace NUMINAMATH_GPT_max_red_tiles_l750_75058

theorem max_red_tiles (n : ℕ) (color : ℕ → ℕ → color) :
    (∀ i j, color i j ≠ color (i + 1) j ∧ color i j ≠ color i (j + 1) ∧ color i j ≠ color (i + 1) (j + 1) 
           ∧ color i j ≠ color (i - 1) j ∧ color i j ≠ color i (j - 1) ∧ color i j ≠ color (i - 1) (j - 1)) 
    → ∃ m ≤ 2500, ∀ i j, (color i j = red ↔ i * n + j < m) :=
sorry

end NUMINAMATH_GPT_max_red_tiles_l750_75058


namespace NUMINAMATH_GPT_no_square_with_odd_last_two_digits_l750_75055

def last_two_digits_odd (n : ℤ) : Prop :=
  (n % 10) % 2 = 1 ∧ ((n / 10) % 10) % 2 = 1

theorem no_square_with_odd_last_two_digits (n : ℤ) (k : ℤ) :
  (k^2 = n) → last_two_digits_odd n → False :=
by
  -- A placeholder for the proof
  sorry

end NUMINAMATH_GPT_no_square_with_odd_last_two_digits_l750_75055


namespace NUMINAMATH_GPT_find_b_from_ellipse_l750_75040

-- Definitions used in conditions
variables {F₁ F₂ : ℝ → ℝ} -- foci
variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Conditions
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse a b P.1 P.2
def perpendicular_vectors (P : ℝ × ℝ) : Prop := true -- Simplified, use correct condition in detailed proof
def area_of_triangle (P : ℝ × ℝ) (F₁ F₂ : ℝ → ℝ) : ℝ := 9

-- The target statement
theorem find_b_from_ellipse (P : ℝ × ℝ) (condition1 : point_on_ellipse a b P)
  (condition2 : perpendicular_vectors P) 
  (condition3 : area_of_triangle P F₁ F₂ = 9) : 
  b = 3 := 
sorry

end NUMINAMATH_GPT_find_b_from_ellipse_l750_75040


namespace NUMINAMATH_GPT_fraction_comparison_l750_75085

theorem fraction_comparison : 
  (1 / (Real.sqrt 2 - 1)) < (Real.sqrt 3 + 1) :=
sorry

end NUMINAMATH_GPT_fraction_comparison_l750_75085


namespace NUMINAMATH_GPT_factorial_equation_solution_l750_75066

theorem factorial_equation_solution (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → (a, b, c) = (3, 3, 4) :=
by
  sorry

end NUMINAMATH_GPT_factorial_equation_solution_l750_75066


namespace NUMINAMATH_GPT_problem_equivalent_proof_l750_75079

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem_equivalent_proof : ((sqrt 3 - 2) ^ 0 - Real.logb 2 (sqrt 2)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_proof_l750_75079


namespace NUMINAMATH_GPT_base_of_right_angled_triangle_l750_75029

theorem base_of_right_angled_triangle 
  (height : ℕ) (area : ℕ) (hypotenuse : ℕ) (b : ℕ) 
  (h_height : height = 8)
  (h_area : area = 24)
  (h_hypotenuse : hypotenuse = 10) 
  (h_area_eq : area = (1 / 2 : ℕ) * b * height)
  (h_pythagorean : hypotenuse^2 = height^2 + b^2) : 
  b = 6 := 
sorry

end NUMINAMATH_GPT_base_of_right_angled_triangle_l750_75029


namespace NUMINAMATH_GPT_total_shaded_area_l750_75099

theorem total_shaded_area 
  (side': ℝ) (d: ℝ) (s: ℝ)
  (h1: 12 / d = 4)
  (h2: d / s = 4) : 
  d = 3 →
  s = 3 / 4 →
  (π * (d / 2) ^ 2 + 8 * s ^ 2) = 9 * π / 4 + 9 / 2 :=
by
  intro h3 h4
  have h5 : d = 3 := h3
  have h6 : s = 3 / 4 := h4
  rw [h5, h6]
  sorry

end NUMINAMATH_GPT_total_shaded_area_l750_75099


namespace NUMINAMATH_GPT_james_prom_total_cost_l750_75083

-- Definitions and conditions
def ticket_cost : ℕ := 100
def num_tickets : ℕ := 2
def dinner_cost : ℕ := 120
def tip_rate : ℚ := 0.30
def limo_hourly_rate : ℕ := 80
def limo_hours : ℕ := 6

-- Calculation of each component
def total_ticket_cost : ℕ := ticket_cost * num_tickets
def total_tip : ℚ := tip_rate * dinner_cost
def total_dinner_cost : ℚ := dinner_cost + total_tip
def total_limo_cost : ℕ := limo_hourly_rate * limo_hours

-- Final total cost calculation
def total_cost : ℚ := total_ticket_cost + total_dinner_cost + total_limo_cost

-- Proving the final total cost
theorem james_prom_total_cost : total_cost = 836 := by sorry

end NUMINAMATH_GPT_james_prom_total_cost_l750_75083


namespace NUMINAMATH_GPT_trains_meet_at_noon_l750_75006

noncomputable def meeting_time_of_trains : Prop :=
  let distance_between_stations := 200
  let speed_of_train_A := 20
  let starting_time_A := 7
  let speed_of_train_B := 25
  let starting_time_B := 8
  let initial_distance_covered_by_A := speed_of_train_A * (starting_time_B - starting_time_A)
  let remaining_distance := distance_between_stations - initial_distance_covered_by_A
  let relative_speed := speed_of_train_A + speed_of_train_B
  let time_to_meet_after_B_starts := remaining_distance / relative_speed
  let meeting_time := starting_time_B + time_to_meet_after_B_starts
  meeting_time = 12

theorem trains_meet_at_noon : meeting_time_of_trains :=
by
  sorry

end NUMINAMATH_GPT_trains_meet_at_noon_l750_75006


namespace NUMINAMATH_GPT_coordinates_of_P_l750_75056

def point (x y : ℝ) := (x, y)

def A : (ℝ × ℝ) := point 1 1
def B : (ℝ × ℝ) := point 4 0

def vector_sub (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2)

def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

theorem coordinates_of_P
  (P : ℝ × ℝ)
  (hP : vector_sub P A = scalar_mult 3 (vector_sub B P)) :
  P = (11 / 2, -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l750_75056


namespace NUMINAMATH_GPT_luis_finish_fourth_task_l750_75018

-- Define the starting and finishing times
def start_time : ℕ := 540  -- 9:00 AM is 540 minutes from midnight
def finish_third_task : ℕ := 750  -- 12:30 PM is 750 minutes from midnight
def duration_one_task : ℕ := (750 - 540) / 3  -- Time for one task

-- Define the problem statement
theorem luis_finish_fourth_task :
  start_time = 540 →
  finish_third_task = 750 →
  3 * duration_one_task = finish_third_task - start_time →
  finish_third_task + duration_one_task = 820 :=
by
  -- You can place the proof for the theorem here
  sorry

end NUMINAMATH_GPT_luis_finish_fourth_task_l750_75018


namespace NUMINAMATH_GPT_xy_product_l750_75001

-- Define the proof problem with the conditions and required statement
theorem xy_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy_distinct : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 := 
  sorry

end NUMINAMATH_GPT_xy_product_l750_75001


namespace NUMINAMATH_GPT_tissue_magnification_l750_75007

theorem tissue_magnification
  (diameter_magnified : ℝ)
  (diameter_actual : ℝ)
  (h1 : diameter_magnified = 5)
  (h2 : diameter_actual = 0.005) :
  diameter_magnified / diameter_actual = 1000 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_tissue_magnification_l750_75007


namespace NUMINAMATH_GPT_sequence_and_sum_problems_l750_75089

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n-1) * d) / 2

def geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := b * r^(n-1)

noncomputable def sum_geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := 
(if r = 1 then b * n
 else b * (r^n - 1) / (r - 1))

theorem sequence_and_sum_problems :
  (∀ n : ℕ, arithmetic_sequence 19 (-2) n = 21 - 2 * n) ∧
  (∀ n : ℕ, sum_arithmetic_sequence 19 (-2) n = 20 * n - n^2) ∧
  (∀ n : ℕ, ∃ a_n : ℤ, (geometric_sequence 1 3 n + (a_n - geometric_sequence 1 3 n) = 21 - 2 * n + 3^(n-1)) ∧
    sum_geometric_sequence 1 3 n = (sum_arithmetic_sequence 19 (-2) n + (3^n - 1) / 2))
:= by
  sorry

end NUMINAMATH_GPT_sequence_and_sum_problems_l750_75089


namespace NUMINAMATH_GPT_number_of_students_l750_75051

/--
Statement: Several students are seated around a circular table. 
Each person takes one piece from a bag containing 120 pieces of candy 
before passing it to the next. Chris starts with the bag, takes one piece 
and also ends up with the last piece. Prove that the number of students
at the table could be 7 or 17.
-/
theorem number_of_students (n : Nat) (h : 120 > 0) :
  (∃ k, 119 = k * n ∧ n ≥ 1) → (n = 7 ∨ n = 17) :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l750_75051


namespace NUMINAMATH_GPT_symmetric_line_eq_l750_75090

theorem symmetric_line_eq (x y : ℝ) :  
  (x - 2 * y + 3 = 0) → (x + 2 * y + 3 = 0) :=
sorry

end NUMINAMATH_GPT_symmetric_line_eq_l750_75090


namespace NUMINAMATH_GPT_plane_stops_at_20_seconds_l750_75004

/-- The analytical expression of the function of the distance s the plane travels during taxiing 
after landing with respect to the time t is given by s = -1.5t^2 + 60t. 

Prove that the plane stops after taxiing for 20 seconds. -/

noncomputable def plane_distance (t : ℝ) : ℝ :=
  -1.5 * t^2 + 60 * t

theorem plane_stops_at_20_seconds :
  ∃ t : ℝ, t = 20 ∧ plane_distance t = plane_distance (20 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_plane_stops_at_20_seconds_l750_75004


namespace NUMINAMATH_GPT_intersection_eq_l750_75005

namespace Proof

universe u

-- Define the natural number set M
def M : Set ℕ := { x | x > 0 ∧ x < 6 }

-- Define the set N based on the condition |x-1| ≤ 2
def N : Set ℝ := { x | abs (x - 1) ≤ 2 }

-- Define the complement of N with respect to the real numbers
def ComplementN : Set ℝ := { x | x < -1 ∨ x > 3 }

-- Define the intersection of M and the complement of N
def IntersectMCompN : Set ℕ := { x | x ∈ M ∧ (x : ℝ) ∈ ComplementN }

-- Provide the theorem to be proved
theorem intersection_eq : IntersectMCompN = { 4, 5 } :=
by
  sorry

end Proof

end NUMINAMATH_GPT_intersection_eq_l750_75005


namespace NUMINAMATH_GPT_quadratic_function_points_relationship_l750_75041

theorem quadratic_function_points_relationship (c y1 y2 y3 : ℝ) 
  (h₁ : y1 = -((-1) ^ 2) + 2 * (-1) + c)
  (h₂ : y2 = -(2 ^ 2) + 2 * 2 + c)
  (h₃ : y3 = -(5 ^ 2) + 2 * 5 + c) :
  y2 > y1 ∧ y1 > y3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_points_relationship_l750_75041


namespace NUMINAMATH_GPT_no_solutions_abs_eq_quadratic_l750_75078

theorem no_solutions_abs_eq_quadratic (x : ℝ) : ¬ (|x - 3| = x^2 + 2 * x + 4) := 
by
  sorry

end NUMINAMATH_GPT_no_solutions_abs_eq_quadratic_l750_75078


namespace NUMINAMATH_GPT_range_of_m_l750_75071

open Set

-- Definitions and conditions
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0
def neg_p (x : ℝ) : Prop := ¬ p x
def neg_q (x m : ℝ) : Prop := ¬ q x m

-- Theorem statement
theorem range_of_m (x m : ℝ) (h₁ : ¬ p x → ¬ q x m) (h₂ : m > 0) : m ≥ 9 :=
  sorry

end NUMINAMATH_GPT_range_of_m_l750_75071


namespace NUMINAMATH_GPT_min_value_of_x_plus_y_l750_75037

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_value_of_x_plus_y_l750_75037


namespace NUMINAMATH_GPT_Juan_has_498_marbles_l750_75097

def ConnieMarbles : Nat := 323
def JuanMoreMarbles : Nat := 175
def JuanMarbles : Nat := ConnieMarbles + JuanMoreMarbles

theorem Juan_has_498_marbles : JuanMarbles = 498 := by
  sorry

end NUMINAMATH_GPT_Juan_has_498_marbles_l750_75097


namespace NUMINAMATH_GPT_total_subjects_l750_75096

theorem total_subjects (subjects_monica subjects_marius subjects_millie : ℕ)
  (h1 : subjects_monica = 10)
  (h2 : subjects_marius = subjects_monica + 4)
  (h3 : subjects_millie = subjects_marius + 3) :
  subjects_monica + subjects_marius + subjects_millie = 41 :=
by
  sorry

end NUMINAMATH_GPT_total_subjects_l750_75096


namespace NUMINAMATH_GPT_max_popsicles_with_10_dollars_l750_75074

def price (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 3 then 2
  else if n = 5 then 3
  else if n = 7 then 4
  else 0

theorem max_popsicles_with_10_dollars : ∀ (a b c d : ℕ),
  a * price 1 + b * price 3 + c * price 5 + d * price 7 = 10 →
  a + 3 * b + 5 * c + 7 * d ≤ 17 :=
sorry

end NUMINAMATH_GPT_max_popsicles_with_10_dollars_l750_75074


namespace NUMINAMATH_GPT_cookies_on_first_plate_l750_75042

theorem cookies_on_first_plate :
  ∃ a1 a2 a3 a4 a5 a6 : ℤ, 
  a2 = 7 ∧ 
  a3 = 10 ∧
  a4 = 14 ∧
  a5 = 19 ∧
  a6 = 25 ∧
  a2 = a1 + 2 ∧ 
  a3 = a2 + 3 ∧ 
  a4 = a3 + 4 ∧ 
  a5 = a4 + 5 ∧ 
  a6 = a5 + 6 ∧ 
  a1 = 5 :=
sorry

end NUMINAMATH_GPT_cookies_on_first_plate_l750_75042


namespace NUMINAMATH_GPT_multiply_63_57_l750_75061

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_GPT_multiply_63_57_l750_75061


namespace NUMINAMATH_GPT_total_pears_l750_75069

theorem total_pears (S P C : ℕ) (hS : S = 20) (hP : P = (S - S / 2)) (hC : C = (P + P / 5)) : S + P + C = 42 :=
by
  -- We state the theorem with the given conditions and the goal of proving S + P + C = 42.
  sorry

end NUMINAMATH_GPT_total_pears_l750_75069


namespace NUMINAMATH_GPT_friend_saves_per_week_l750_75052

theorem friend_saves_per_week
  (x : ℕ) 
  (you_have : ℕ := 160)
  (you_save_per_week : ℕ := 7)
  (friend_have : ℕ := 210)
  (weeks : ℕ := 25)
  (total_you_save : ℕ := you_have + you_save_per_week * weeks)
  (total_friend_save : ℕ := friend_have + x * weeks) 
  (h : total_you_save = total_friend_save) : x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_friend_saves_per_week_l750_75052


namespace NUMINAMATH_GPT_gcd_5670_9800_l750_75020

-- Define the two given numbers
def a := 5670
def b := 9800

-- State that the GCD of a and b is 70
theorem gcd_5670_9800 : Int.gcd a b = 70 := by
  sorry

end NUMINAMATH_GPT_gcd_5670_9800_l750_75020


namespace NUMINAMATH_GPT_equivalence_of_negation_l750_75015

-- Define the statement for the negation
def negation_stmt := ¬ ∃ x0 : ℝ, x0 ≤ 0 ∧ x0^2 ≥ 0

-- Define the equivalent statement after negation
def equivalent_stmt := ∀ x : ℝ, x ≤ 0 → x^2 < 0

-- The theorem stating that the negation_stmt is equivalent to equivalent_stmt
theorem equivalence_of_negation : negation_stmt ↔ equivalent_stmt := 
sorry

end NUMINAMATH_GPT_equivalence_of_negation_l750_75015


namespace NUMINAMATH_GPT_arithmetic_contains_geometric_l750_75064

theorem arithmetic_contains_geometric (a b : ℚ) (h : a^2 + b^2 ≠ 0) :
  ∃ (q : ℚ) (c : ℚ) (n₀ : ℕ) (n : ℕ → ℕ), (∀ k : ℕ, n (k+1) = n k + c * q^k) ∧
  ∀ k : ℕ, ∃ r : ℚ, a + b * n k = r * q^k :=
sorry

end NUMINAMATH_GPT_arithmetic_contains_geometric_l750_75064


namespace NUMINAMATH_GPT_solution_set_of_inequality_system_l750_75092

theorem solution_set_of_inequality_system (x : ℝ) : 
  (x + 5 < 4) ∧ (3 * x + 1 ≥ 2 * (2 * x - 1)) ↔ (x < -1) :=
  by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_system_l750_75092


namespace NUMINAMATH_GPT_sin_C_value_l750_75030

noncomputable def triangle_sine_proof (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : Real := by
  -- Utilizing the Law of Sines and given conditions to find sin C
  sorry

theorem sin_C_value (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : triangle_sine_proof A B C a b c hB hb = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_GPT_sin_C_value_l750_75030


namespace NUMINAMATH_GPT_sum_of_roots_l750_75082

theorem sum_of_roots (p q : ℝ) (h_eq : 2 * p + 3 * q = 6) (h_roots : ∀ x : ℝ, x ^ 2 - p * x + q = 0) : p = 2 := by
sorry

end NUMINAMATH_GPT_sum_of_roots_l750_75082


namespace NUMINAMATH_GPT_range_of_f_l750_75091

noncomputable def f (x : Real) : Real :=
  if x ≤ 1 then 2 * x + 1 else Real.log x + 1

theorem range_of_f (x : Real) : f x + f (x + 1) > 1 ↔ (x > -(3 / 4)) :=
  sorry

end NUMINAMATH_GPT_range_of_f_l750_75091


namespace NUMINAMATH_GPT_ms_cole_total_students_l750_75024

def number_of_students (S6 : Nat) (S4 : Nat) (S7 : Nat) : Nat :=
  S6 + S4 + S7

theorem ms_cole_total_students (S6 S4 S7 : Nat)
  (h1 : S6 = 40)
  (h2 : S4 = 4 * S6)
  (h3 : S7 = 2 * S4) :
  number_of_students S6 S4 S7 = 520 := by
  sorry

end NUMINAMATH_GPT_ms_cole_total_students_l750_75024


namespace NUMINAMATH_GPT_Sue_chewing_gums_count_l750_75008

theorem Sue_chewing_gums_count (S : ℕ) 
  (hMary : 5 = 5) 
  (hSam : 10 = 10) 
  (hTotal : 5 + 10 + S = 30) : S = 15 := 
by {
  sorry
}

end NUMINAMATH_GPT_Sue_chewing_gums_count_l750_75008


namespace NUMINAMATH_GPT_four_op_two_l750_75022

def op (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem four_op_two : op 4 2 = 18 := by
  sorry

end NUMINAMATH_GPT_four_op_two_l750_75022


namespace NUMINAMATH_GPT_roots_cubic_sum_cubes_l750_75063

theorem roots_cubic_sum_cubes (a b c : ℝ) 
    (h1 : 6 * a^3 - 803 * a + 1606 = 0)
    (h2 : 6 * b^3 - 803 * b + 1606 = 0)
    (h3 : 6 * c^3 - 803 * c + 1606 = 0) :
    (a + b)^3 + (b + c)^3 + (c + a)^3 = 803 := 
by
  sorry

end NUMINAMATH_GPT_roots_cubic_sum_cubes_l750_75063


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l750_75049

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n+1) = q * a n)
  (h_a1 : a 1 = 4)
  (h_a4 : a 4 = 1/2) :
  q = 1/2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l750_75049


namespace NUMINAMATH_GPT_fraction_equality_l750_75076

theorem fraction_equality (x : ℝ) :
  (4 + 2 * x) / (7 + 3 * x) = (2 + 3 * x) / (4 + 5 * x) ↔ x = -1 ∨ x = -2 := by
  sorry

end NUMINAMATH_GPT_fraction_equality_l750_75076


namespace NUMINAMATH_GPT_indeterminate_4wheelers_l750_75095

-- Define conditions and the main theorem to state that the number of 4-wheelers cannot be uniquely determined.
theorem indeterminate_4wheelers (x y : ℕ) (h : 2 * x + 4 * y = 58) : ∃ k : ℤ, y = ((29 : ℤ) - k - x) / 2 :=
by
  sorry

end NUMINAMATH_GPT_indeterminate_4wheelers_l750_75095


namespace NUMINAMATH_GPT_gcd_of_gx_and_x_l750_75084

theorem gcd_of_gx_and_x (x : ℤ) (hx : x % 11739 = 0) :
  Int.gcd ((3 * x + 4) * (5 * x + 3) * (11 * x + 5) * (x + 11)) x = 3 :=
sorry

end NUMINAMATH_GPT_gcd_of_gx_and_x_l750_75084


namespace NUMINAMATH_GPT_max_value_proof_l750_75077

noncomputable def maximum_value (x y z : ℝ) : ℝ := 
  (2/x) + (1/y) - (2/z) + 2

theorem max_value_proof {x y z : ℝ} 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0):
  maximum_value x y z ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_proof_l750_75077


namespace NUMINAMATH_GPT_x_squared_eq_r_floor_x_has_2_or_3_solutions_l750_75017

theorem x_squared_eq_r_floor_x_has_2_or_3_solutions (r : ℝ) (hr : r > 2) : 
  ∃! (s : Finset ℝ), s.card = 2 ∨ s.card = 3 ∧ ∀ x ∈ s, x^2 = r * (⌊x⌋) :=
by
  sorry

end NUMINAMATH_GPT_x_squared_eq_r_floor_x_has_2_or_3_solutions_l750_75017


namespace NUMINAMATH_GPT_consecutive_weights_sum_to_63_l750_75043

theorem consecutive_weights_sum_to_63 : ∃ n : ℕ, (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 63 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_weights_sum_to_63_l750_75043


namespace NUMINAMATH_GPT_initial_numbers_unique_l750_75054

theorem initial_numbers_unique 
  (A B C A' B' C' : ℕ) 
  (h1: 1 ≤ A ∧ A ≤ 50) 
  (h2: 1 ≤ B ∧ B ≤ 50) 
  (h3: 1 ≤ C ∧ C ≤ 50) 
  (final_ana : 104 = 2 * A + B + C)
  (final_beto : 123 = A + 2 * B + C)
  (final_caio : 137 = A + B + 2 * C) : 
  A = 13 ∧ B = 32 ∧ C = 46 :=
sorry

end NUMINAMATH_GPT_initial_numbers_unique_l750_75054


namespace NUMINAMATH_GPT_equivalent_single_discount_l750_75016

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.25
noncomputable def coupon_discount : ℝ := 0.10
noncomputable def final_price : ℝ := 33.75

theorem equivalent_single_discount :
  (1 - final_price / original_price) * 100 = 32.5 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_single_discount_l750_75016


namespace NUMINAMATH_GPT_table_tennis_possible_outcomes_l750_75094

-- Two people are playing a table tennis match. The first to win 3 games wins the match.
-- The match continues until a winner is determined.
-- Considering all possible outcomes (different numbers of wins and losses for each player are considered different outcomes),
-- prove that there are a total of 30 possible outcomes.

theorem table_tennis_possible_outcomes : 
  ∃ total_outcomes : ℕ, total_outcomes = 30 := 
by
  -- We need to prove that the total number of possible outcomes is 30
  sorry

end NUMINAMATH_GPT_table_tennis_possible_outcomes_l750_75094


namespace NUMINAMATH_GPT_cone_rolls_path_l750_75014

theorem cone_rolls_path (r h m n : ℝ) (rotations : ℕ) 
  (h_rotations : rotations = 20)
  (h_ratio : h / r = 3 * Real.sqrt 133)
  (h_m : m = 3)
  (h_n : n = 133) : 
  m + n = 136 := 
by sorry

end NUMINAMATH_GPT_cone_rolls_path_l750_75014


namespace NUMINAMATH_GPT_price_of_each_pizza_l750_75034

variable (P : ℝ)

theorem price_of_each_pizza (h1 : 4 * P + 5 = 45) : P = 10 := by
  sorry

end NUMINAMATH_GPT_price_of_each_pizza_l750_75034


namespace NUMINAMATH_GPT_least_number_of_pairs_l750_75025

theorem least_number_of_pairs :
  let students := 100
  let messages_per_student := 50
  ∃ (pairs_of_students : ℕ), pairs_of_students = 50 := sorry

end NUMINAMATH_GPT_least_number_of_pairs_l750_75025


namespace NUMINAMATH_GPT_intersection_equal_l750_75019

-- Define the sets M and N based on given conditions
def M : Set ℝ := {x : ℝ | x^2 - 3 * x - 28 ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

-- Define the intersection of M and N
def intersection : Set ℝ := {x : ℝ | (-4 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x ≤ 7)}

-- The statement to be proved
theorem intersection_equal : M ∩ N = intersection :=
by 
  sorry -- Skipping the proof

end NUMINAMATH_GPT_intersection_equal_l750_75019


namespace NUMINAMATH_GPT_length_of_AB_l750_75044

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

noncomputable def slope_of_line : ℝ := Real.tan (Real.pi / 6)

-- Equation of the line in point-slope form
noncomputable def line_eq (x : ℝ) : ℝ :=
  (slope_of_line * x) + 1

-- Intersection points of the line with the parabola y = (1/4)x^2
noncomputable def parabola_eq (x : ℝ) : ℝ :=
  (1/4) * x ^ 2

theorem length_of_AB :
  ∃ A B : ℝ × ℝ, 
    (A.2 = parabola_eq A.1) ∧
    (B.2 = parabola_eq B.1) ∧ 
    (A.2 = line_eq A.1) ∧
    (B.2 = line_eq B.1) ∧
    ((((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) ^ (1 / 2)) = 16 / 3) :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l750_75044


namespace NUMINAMATH_GPT_find_integer_l750_75031

theorem find_integer (n : ℤ) (h : 5 * (n - 2) = 85) : n = 19 :=
sorry

end NUMINAMATH_GPT_find_integer_l750_75031


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_property_l750_75057

theorem arithmetic_sequence_geometric_property (a : ℕ → ℤ) (d : ℤ) (h_d : d = 2)
  (h_a3 : a 3 = a 1 + 4) (h_a4 : a 4 = a 1 + 6)
  (geo_seq : (a 1 + 4) * (a 1 + 4) = a 1 * (a 1 + 6)) :
  a 2 = -6 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_property_l750_75057


namespace NUMINAMATH_GPT_mini_marshmallows_count_l750_75002

theorem mini_marshmallows_count (total_marshmallows large_marshmallows : ℕ) (h1 : total_marshmallows = 18) (h2 : large_marshmallows = 8) :
  total_marshmallows - large_marshmallows = 10 :=
by 
  sorry

end NUMINAMATH_GPT_mini_marshmallows_count_l750_75002


namespace NUMINAMATH_GPT_binary_remainder_div_8_l750_75053

theorem binary_remainder_div_8 (n : ℕ) (h : n = 0b101100110011) : n % 8 = 3 :=
by sorry

end NUMINAMATH_GPT_binary_remainder_div_8_l750_75053


namespace NUMINAMATH_GPT_compare_f_minus1_f_1_l750_75087

variable (f : ℝ → ℝ)

-- Given conditions
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x : ℝ, f x = x^2 + 2 * x * (f 2 - 2 * x))

-- Goal statement
theorem compare_f_minus1_f_1 : f (-1) > f 1 :=
by sorry

end NUMINAMATH_GPT_compare_f_minus1_f_1_l750_75087


namespace NUMINAMATH_GPT_find_triangle_sides_l750_75009

noncomputable def side_lengths (k c d : ℕ) : Prop :=
  let p1 := 26
  let p2 := 32
  let p3 := 30
  (2 * k = 6) ∧ (2 * k + 6 * c = p3) ∧ (2 * c + 2 * d = p1)

theorem find_triangle_sides (k c d : ℕ) (h1 : side_lengths k c d) : k = 3 ∧ c = 4 ∧ d = 5 := 
  sorry

end NUMINAMATH_GPT_find_triangle_sides_l750_75009


namespace NUMINAMATH_GPT_isosceles_right_triangle_sums_l750_75098

theorem isosceles_right_triangle_sums (m n : ℝ)
  (h1: (1 * 2 + m * m + 2 * n) = 0)
  (h2: (1 + m^2 + 4) = (4 + m^2 + n^2)) :
  m + n = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_isosceles_right_triangle_sums_l750_75098


namespace NUMINAMATH_GPT_normal_line_eq_l750_75038

variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem normal_line_eq (x_0 : ℝ) (h : x_0 = 1) :
  ∃ y_0 : ℝ, y_0 = f x_0 ∧ 
  ∀ x y : ℝ, y = -(x - 1) + y_0 ↔ f 1 = 0 ∧ y = -x + 1 :=
by
  sorry

end NUMINAMATH_GPT_normal_line_eq_l750_75038


namespace NUMINAMATH_GPT_perpendicular_line_equation_l750_75073

theorem perpendicular_line_equation (x y : ℝ) (h : 2 * x + y + 3 = 0) (hx : ∃ c : ℝ, x - 2 * y + c = 0) :
  (c = 7 ↔ ∀ p : ℝ × ℝ, p = (-1, 3) → (p.1 - 2 * p.2 + 7 = 0)) :=
sorry

end NUMINAMATH_GPT_perpendicular_line_equation_l750_75073


namespace NUMINAMATH_GPT_bobbit_worm_days_l750_75033

variable (initial_fish : ℕ)
variable (fish_added : ℕ)
variable (fish_eaten_per_day : ℕ)
variable (week_days : ℕ)
variable (final_fish : ℕ)
variable (d : ℕ)

theorem bobbit_worm_days (h1 : initial_fish = 60)
                         (h2 : fish_added = 8)
                         (h3 : fish_eaten_per_day = 2)
                         (h4 : week_days = 7)
                         (h5 : final_fish = 26) :
  60 - 2 * d + 8 - 2 * week_days = 26 → d = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_bobbit_worm_days_l750_75033


namespace NUMINAMATH_GPT_minimum_value_expression_l750_75047

theorem minimum_value_expression (x y z : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2) ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2)) :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l750_75047


namespace NUMINAMATH_GPT_correct_expression_l750_75000

theorem correct_expression (a b : ℝ) : (a^2 * b)^3 = (a^6 * b^3) := 
by
sorry

end NUMINAMATH_GPT_correct_expression_l750_75000


namespace NUMINAMATH_GPT_geometric_sequence_sum_l750_75028

theorem geometric_sequence_sum {a : ℕ → ℝ} (h : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) 
  (h_cond : (1 / (a 2 * a 4)) + (2 / (a 4 * a 4)) + (1 / (a 4 * a 6)) = 81) :
  (1 / a 3) + (1 / a 5) = 9 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l750_75028


namespace NUMINAMATH_GPT_min_x_plus_y_l750_75086

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_min_x_plus_y_l750_75086


namespace NUMINAMATH_GPT_cory_initial_money_l750_75013

variable (cost_per_pack : ℝ) (packs : ℕ) (additional_needed : ℝ) (total_cost : ℝ) (initial_money : ℝ)

-- Conditions
def cost_per_pack_def : Prop := cost_per_pack = 49
def packs_def : Prop := packs = 2
def additional_needed_def : Prop := additional_needed = 78
def total_cost_def : Prop := total_cost = packs * cost_per_pack
def initial_money_def : Prop := initial_money = total_cost - additional_needed

-- Theorem
theorem cory_initial_money : cost_per_pack = 49 ∧ packs = 2 ∧ additional_needed = 78 → initial_money = 20 := by
  intro h
  have h1 : cost_per_pack = 49 := h.1
  have h2 : packs = 2 := h.2.1
  have h3 : additional_needed = 78 := h.2.2
  -- sorry
  sorry

end NUMINAMATH_GPT_cory_initial_money_l750_75013


namespace NUMINAMATH_GPT_least_whole_number_for_ratio_l750_75011

theorem least_whole_number_for_ratio :
  ∃ x : ℕ, (6 - x) * 21 < (7 - x) * 16 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_least_whole_number_for_ratio_l750_75011


namespace NUMINAMATH_GPT_pufferfish_count_l750_75021

theorem pufferfish_count (s p : ℕ) (h1 : s = 5 * p) (h2 : s + p = 90) : p = 15 :=
by
  sorry

end NUMINAMATH_GPT_pufferfish_count_l750_75021


namespace NUMINAMATH_GPT_largest_positive_real_root_bound_l750_75039

theorem largest_positive_real_root_bound (b0 b1 b2 : ℝ)
  (h_b0 : abs b0 ≤ 1) (h_b1 : abs b1 ≤ 1) (h_b2 : abs b2 ≤ 1) :
  ∃ r : ℝ, r > 0 ∧ r^3 + b2 * r^2 + b1 * r + b0 = 0 ∧ 1.5 < r ∧ r < 2 := 
sorry

end NUMINAMATH_GPT_largest_positive_real_root_bound_l750_75039


namespace NUMINAMATH_GPT_leo_amount_after_settling_debts_l750_75059

theorem leo_amount_after_settling_debts (total_amount : ℝ) (ryan_share : ℝ) (ryan_owes_leo : ℝ) (leo_owes_ryan : ℝ) 
  (h1 : total_amount = 48) 
  (h2 : ryan_share = (2 / 3) * total_amount) 
  (h3 : ryan_owes_leo = 10) 
  (h4 : leo_owes_ryan = 7) : 
  (total_amount - ryan_share) + (ryan_owes_leo - leo_owes_ryan) = 19 :=
by
  sorry

end NUMINAMATH_GPT_leo_amount_after_settling_debts_l750_75059


namespace NUMINAMATH_GPT_calculation_is_correct_l750_75010

theorem calculation_is_correct :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := 
by
  sorry

end NUMINAMATH_GPT_calculation_is_correct_l750_75010


namespace NUMINAMATH_GPT_amount_paid_l750_75060

def original_price : ℕ := 15
def discount_percentage : ℕ := 40

theorem amount_paid (ticket_price : ℕ) (discount_pct : ℕ) (discount_amount : ℕ) (paid_amount : ℕ) 
  (h1 : ticket_price = original_price) 
  (h2 : discount_pct = discount_percentage) 
  (h3 : discount_amount = (discount_pct * ticket_price) / 100)
  (h4 : paid_amount = ticket_price - discount_amount) 
  : paid_amount = 9 := 
sorry

end NUMINAMATH_GPT_amount_paid_l750_75060


namespace NUMINAMATH_GPT_product_area_perimeter_square_EFGH_l750_75045

theorem product_area_perimeter_square_EFGH:
  let E := (5, 5)
  let F := (5, 1)
  let G := (1, 1)
  let H := (1, 5)
  let side_length := 4
  let area := side_length * side_length
  let perimeter := 4 * side_length
  area * perimeter = 256 :=
by
  sorry

end NUMINAMATH_GPT_product_area_perimeter_square_EFGH_l750_75045


namespace NUMINAMATH_GPT_inequality_true_l750_75026

variable (a b : ℝ)

theorem inequality_true (h : a > b ∧ b > 0) : (b^2 / a) < (a^2 / b) := by
  sorry

end NUMINAMATH_GPT_inequality_true_l750_75026


namespace NUMINAMATH_GPT_average_of_first_15_even_numbers_is_16_l750_75003

-- Define the sum of the first 15 even numbers
def sum_first_15_even_numbers : ℕ :=
  2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30

-- Define the average of the first 15 even numbers
def average_of_first_15_even_numbers : ℕ :=
  sum_first_15_even_numbers / 15

-- Prove that the average is equal to 16
theorem average_of_first_15_even_numbers_is_16 : average_of_first_15_even_numbers = 16 :=
by
  -- Sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_average_of_first_15_even_numbers_is_16_l750_75003


namespace NUMINAMATH_GPT_range_of_m_l750_75062

open Set

noncomputable def M (m : ℝ) : Set ℝ := {x | x ≤ m}
noncomputable def N : Set ℝ := {y | y ≥ 1}

theorem range_of_m (m : ℝ) : M m ∩ N = ∅ → m < 1 := by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_m_l750_75062


namespace NUMINAMATH_GPT_restaurant_customers_prediction_l750_75068

def total_customers_saturday (breakfast_customers_friday lunch_customers_friday dinner_customers_friday : ℝ) : ℝ :=
  let breakfast_customers_saturday := 2 * breakfast_customers_friday
  let lunch_customers_saturday := lunch_customers_friday + 0.25 * lunch_customers_friday
  let dinner_customers_saturday := dinner_customers_friday - 0.15 * dinner_customers_friday
  breakfast_customers_saturday + lunch_customers_saturday + dinner_customers_saturday

theorem restaurant_customers_prediction :
  let breakfast_customers_friday := 73
  let lunch_customers_friday := 127
  let dinner_customers_friday := 87
  total_customers_saturday breakfast_customers_friday lunch_customers_friday dinner_customers_friday = 379 := 
by
  sorry

end NUMINAMATH_GPT_restaurant_customers_prediction_l750_75068


namespace NUMINAMATH_GPT_solve_for_x_l750_75065

theorem solve_for_x (x : ℝ) (h : 1 / 3 + 1 / x = 2 / 3) : x = 3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l750_75065


namespace NUMINAMATH_GPT_correct_assignment_l750_75088

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end NUMINAMATH_GPT_correct_assignment_l750_75088


namespace NUMINAMATH_GPT_distinct_patterns_4x4_three_squares_l750_75046

noncomputable def count_distinct_patterns : ℕ :=
  sorry

theorem distinct_patterns_4x4_three_squares :
  count_distinct_patterns = 12 :=
by sorry

end NUMINAMATH_GPT_distinct_patterns_4x4_three_squares_l750_75046


namespace NUMINAMATH_GPT_parallelogram_angle_H_l750_75036

theorem parallelogram_angle_H (F H : ℝ) (h1 : F = 125) (h2 : F + H = 180) : H = 55 :=
by
  have h3 : H = 180 - F := by linarith
  rw [h1] at h3
  rw [h3]
  norm_num

end NUMINAMATH_GPT_parallelogram_angle_H_l750_75036
