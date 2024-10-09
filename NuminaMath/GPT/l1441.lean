import Mathlib

namespace planes_formed_through_three_lines_l1441_144147

theorem planes_formed_through_three_lines (L1 L2 L3 : ℝ × ℝ × ℝ → Prop) (P : ℝ × ℝ × ℝ) :
  (∀ (x : ℝ × ℝ × ℝ), L1 x → L2 x → L3 x → x = P) →
  (∃ n : ℕ, n = 1 ∨ n = 3) :=
sorry

end planes_formed_through_three_lines_l1441_144147


namespace simplify_expression_l1441_144110

variable (a b : ℝ)

theorem simplify_expression : (a + b) * (3 * a - b) - b * (a - b) = 3 * a ^ 2 + a * b :=
by
  sorry

end simplify_expression_l1441_144110


namespace division_subtraction_l1441_144107

theorem division_subtraction : 144 / (12 / 3) - 5 = 31 := by
  sorry

end division_subtraction_l1441_144107


namespace algebraic_expression_value_l1441_144128

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  (x - 1)^2 - x * (x - 3) + (x + 1) * (x - 1) = 3 :=
by
  sorry

end algebraic_expression_value_l1441_144128


namespace julian_comic_pages_l1441_144170

-- Definitions from conditions
def frames_per_page : ℝ := 143.0
def total_frames : ℝ := 1573.0

-- The theorem stating the proof problem
theorem julian_comic_pages : total_frames / frames_per_page = 11 :=
by
  sorry

end julian_comic_pages_l1441_144170


namespace min_value_of_function_l1441_144157

theorem min_value_of_function (x : ℝ) (h : x > -1) : 
  (∀ x₀ : ℝ, x₀ > -1 → (x₀ + 1 + 1 / (x₀ + 1) - 1) ≥ 1) ∧ (x = 0) :=
sorry

end min_value_of_function_l1441_144157


namespace find_function_ex_l1441_144135

theorem find_function_ex (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  (∀ x : ℝ, f x = x - a) :=
by
  intros h x
  sorry

end find_function_ex_l1441_144135


namespace amphibians_count_l1441_144108

-- Define the conditions
def frogs : Nat := 7
def salamanders : Nat := 4
def tadpoles : Nat := 30
def newt : Nat := 1

-- Define the total number of amphibians observed by Hunter
def total_amphibians : Nat := frogs + salamanders + tadpoles + newt

-- State the theorem
theorem amphibians_count : total_amphibians = 42 := 
by 
  -- proof goes here
  sorry

end amphibians_count_l1441_144108


namespace coordinates_of_P_l1441_144144

theorem coordinates_of_P (A B : ℝ × ℝ × ℝ) (m : ℝ) :
  A = (1, 0, 2) ∧ B = (1, -3, 1) ∧ (0, 0, m) = (0, 0, -3) :=
by 
  sorry

end coordinates_of_P_l1441_144144


namespace find_fourth_score_l1441_144171

theorem find_fourth_score
  (a b c : ℕ) (d : ℕ)
  (ha : a = 70) (hb : b = 80) (hc : c = 90)
  (average_eq : (a + b + c + d) / 4 = 70) :
  d = 40 := 
sorry

end find_fourth_score_l1441_144171


namespace height_percentage_difference_l1441_144112

theorem height_percentage_difference 
  (r1 h1 r2 h2 : ℝ) 
  (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2)
  (r2_eq_1_2_r1 : r2 = (6 / 5) * r1) :
  h1 = (36 / 25) * h2 :=
by
  sorry

end height_percentage_difference_l1441_144112


namespace tangent_line_l1441_144111

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line (x y : ℝ) (h : f 2 = 6) : 13 * x - y - 20 = 0 :=
by
  -- Insert proof here
  sorry

end tangent_line_l1441_144111


namespace profit_per_meter_is_25_l1441_144126

def sell_price : ℕ := 8925
def cost_price_per_meter : ℕ := 80
def meters_sold : ℕ := 85
def total_cost_price : ℕ := cost_price_per_meter * meters_sold
def total_profit : ℕ := sell_price - total_cost_price
def profit_per_meter : ℕ := total_profit / meters_sold

theorem profit_per_meter_is_25 : profit_per_meter = 25 := by
  sorry

end profit_per_meter_is_25_l1441_144126


namespace value_of_expression_l1441_144102

variable {a b m n x : ℝ}

def opposite (a b : ℝ) : Prop := a = -b
def reciprocal (m n : ℝ) : Prop := m * n = 1
def distance_to_2 (x : ℝ) : Prop := abs (x - 2) = 3

theorem value_of_expression (h1 : opposite a b) (h2 : reciprocal m n) (h3 : distance_to_2 x) :
  (a + b - m * n) * x + (a + b)^2022 + (- m * n)^2023 = 
  if x = 5 then -6 else if x = -1 then 0 else sorry :=
by
  sorry

end value_of_expression_l1441_144102


namespace linear_eq_k_l1441_144184

theorem linear_eq_k (k : ℝ) : (k - 3) * x ^ (|k| - 2) + 5 = k - 4 → |k| = 3 → k ≠ 3 → k = -3 :=
by
  intros h1 h2 h3
  sorry

end linear_eq_k_l1441_144184


namespace find_first_half_speed_l1441_144103

theorem find_first_half_speed (distance time total_time : ℝ) (v2 : ℝ)
    (h_distance : distance = 300) 
    (h_time : total_time = 11) 
    (h_v2 : v2 = 25) 
    (half_distance : distance / 2 = 150) :
    (150 / (total_time - (150 / v2)) = 30) :=
by
  sorry

end find_first_half_speed_l1441_144103


namespace samantha_sleep_hours_l1441_144125

def time_in_hours (hours minutes : ℕ) : ℕ :=
  hours + (minutes / 60)

def hours_slept (bed_time wake_up_time : ℕ) : ℕ :=
  if bed_time < wake_up_time then wake_up_time - bed_time + 12 else 24 - bed_time + wake_up_time

theorem samantha_sleep_hours : hours_slept 7 11 = 16 := by
  sorry

end samantha_sleep_hours_l1441_144125


namespace initial_gasohol_amount_l1441_144196

variable (x : ℝ)

def gasohol_ethanol_percentage (initial_gasohol : ℝ) := 0.05 * initial_gasohol
def mixture_ethanol_percentage (initial_gasohol : ℝ) := gasohol_ethanol_percentage initial_gasohol + 3

def optimal_mixture (total_volume : ℝ) := 0.10 * total_volume

theorem initial_gasohol_amount :
  ∀ (initial_gasohol : ℝ), 
  mixture_ethanol_percentage initial_gasohol = optimal_mixture (initial_gasohol + 3) →
  initial_gasohol = 54 :=
by
  intros
  sorry

end initial_gasohol_amount_l1441_144196


namespace sum_interior_numbers_eight_l1441_144173

noncomputable def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2 -- This is a general formula derived from the pattern

theorem sum_interior_numbers_eight :
  sum_interior_numbers 8 = 126 :=
by
  -- No proof required, so we use sorry.
  sorry

end sum_interior_numbers_eight_l1441_144173


namespace closed_chain_possible_l1441_144105

-- Define the angle constraint
def angle_constraint (θ : ℝ) : Prop :=
  θ ≥ 150

-- Define meshing condition between two gears
def meshed_gears (θ : ℝ) : Prop :=
  angle_constraint θ

-- Define the general condition for a closed chain of gears
def closed_chain (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → meshed_gears 150

theorem closed_chain_possible : closed_chain 61 :=
by sorry

end closed_chain_possible_l1441_144105


namespace simplify_expression_l1441_144129

theorem simplify_expression (z : ℝ) : (5 - 2*z^2) - (4*z^2 - 7) = 12 - 6*z^2 :=
by
  sorry

end simplify_expression_l1441_144129


namespace order_of_y1_y2_y3_l1441_144174

/-
Given three points A(-3, y1), B(3, y2), and C(4, y3) all lie on the parabola y = 2*(x - 2)^2 + 1,
prove that y2 < y3 < y1.
-/
theorem order_of_y1_y2_y3 :
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  sorry

end order_of_y1_y2_y3_l1441_144174


namespace sqrt_21_between_4_and_5_l1441_144133

theorem sqrt_21_between_4_and_5 : 4 < Real.sqrt 21 ∧ Real.sqrt 21 < 5 := 
by 
  sorry

end sqrt_21_between_4_and_5_l1441_144133


namespace max_individual_score_l1441_144176

open Nat

theorem max_individual_score (n : ℕ) (total_points : ℕ) (minimum_points : ℕ) (H1 : n = 12) (H2 : total_points = 100) (H3 : ∀ i : Fin n, 7 ≤ minimum_points) :
  ∃ max_points : ℕ, max_points = 23 :=
by 
  sorry

end max_individual_score_l1441_144176


namespace base_conversion_problem_l1441_144163

theorem base_conversion_problem :
  ∃ A B : ℕ, 0 ≤ A ∧ A < 8 ∧ 0 ≤ B ∧ B < 6 ∧
           8 * A + B = 6 * B + A ∧
           8 * A + B = 45 :=
by
  sorry

end base_conversion_problem_l1441_144163


namespace students_per_class_l1441_144154

variable (c : ℕ) (s : ℕ)

def books_per_month := 6
def months_per_year := 12
def books_per_year := books_per_month * months_per_year
def total_books_read := 72

theorem students_per_class : (s * c = 1 ∧ s * books_per_year = total_books_read) → s = 1 := by
  intros h
  have h1: books_per_year = total_books_read := by
    calc
      books_per_year = books_per_month * months_per_year := rfl
      _ = 6 * 12 := rfl
      _ = 72 := rfl
  sorry

end students_per_class_l1441_144154


namespace even_of_form_4a_plus_2_not_diff_of_squares_l1441_144191

theorem even_of_form_4a_plus_2_not_diff_of_squares (a x y : ℤ) : ¬ (4 * a + 2 = x^2 - y^2) :=
by sorry

end even_of_form_4a_plus_2_not_diff_of_squares_l1441_144191


namespace determine_m_l1441_144166

-- Define a complex number structure in Lean
structure ComplexNumber where
  re : ℝ  -- real part
  im : ℝ  -- imaginary part

-- Define the condition where the complex number is purely imaginary
def is_purely_imaginary (z : ComplexNumber) : Prop :=
  z.re = 0

-- State the Lean theorem
theorem determine_m (m : ℝ) (h : is_purely_imaginary (ComplexNumber.mk (m^2 - m) m)) : m = 1 :=
by
  sorry

end determine_m_l1441_144166


namespace route_down_distance_l1441_144186

noncomputable def rate_up : ℝ := 3
noncomputable def time_up : ℝ := 2
noncomputable def time_down : ℝ := 2
noncomputable def rate_down := 1.5 * rate_up

theorem route_down_distance : rate_down * time_down = 9 := by
  sorry

end route_down_distance_l1441_144186


namespace system_solution_l1441_144189

theorem system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : 0 < x₃) (h₄ : 0 < x₄) (h₅ : 0 < x₅)
  (h₆ : x₁ + x₂ = x₃^2) (h₇ : x₃ + x₄ = x₅^2) (h₈ : x₄ + x₅ = x₁^2) (h₉ : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
by 
  sorry

end system_solution_l1441_144189


namespace find_percentage_of_alcohol_l1441_144153

theorem find_percentage_of_alcohol 
  (Vx : ℝ) (Px : ℝ) (Vy : ℝ) (Py : ℝ) (Vp : ℝ) (Pp : ℝ)
  (hx : Px = 10) (hvx : Vx = 300) (hvy : Vy = 100) (hvxy : Vx + Vy = 400) (hpxy : Pp = 15) :
  (Vy * Py / 100) = 30 :=
by
  sorry

end find_percentage_of_alcohol_l1441_144153


namespace find_AX_bisect_ACB_l1441_144165

theorem find_AX_bisect_ACB (AC BX BC : ℝ) (h₁ : AC = 21) (h₂ : BX = 28) (h₃ : BC = 30) :
  ∃ (AX : ℝ), AX = 98 / 5 :=
by
  existsi 98 / 5
  sorry

end find_AX_bisect_ACB_l1441_144165


namespace quadratic_trinomials_unique_root_value_l1441_144131

theorem quadratic_trinomials_unique_root_value (p q : ℝ) :
  ∀ x, (x^2 + p * x + q) + (x^2 + q * x + p) = (2 * x^2 + (p + q) * x + (p + q)) →
  (((p + q = 0 ∨ p + q = 8) → (2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 8 ∨ 2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 32))) :=
by
  sorry

end quadratic_trinomials_unique_root_value_l1441_144131


namespace amount_after_a_year_l1441_144182

def initial_amount : ℝ := 90
def interest_rate : ℝ := 0.10

theorem amount_after_a_year : initial_amount * (1 + interest_rate) = 99 := 
by
  -- Here 'sorry' indicates that the proof is not provided.
  sorry

end amount_after_a_year_l1441_144182


namespace cups_remaining_l1441_144118

-- Each definition only directly appears in the conditions problem
def required_cups : ℕ := 7
def added_cups : ℕ := 3

-- The proof problem capturing Joan needs to add 4 more cups of flour.
theorem cups_remaining : required_cups - added_cups = 4 := 
by
  -- The proof is skipped using sorry.
  sorry

end cups_remaining_l1441_144118


namespace students_voted_both_l1441_144143

def total_students : Nat := 300
def students_voted_first : Nat := 230
def students_voted_second : Nat := 190
def students_voted_none : Nat := 40

theorem students_voted_both :
  students_voted_first + students_voted_second - (total_students - students_voted_none) = 160 :=
by
  sorry

end students_voted_both_l1441_144143


namespace minimum_effort_to_qualify_l1441_144180

def minimum_effort_to_qualify_for_mop (AMC_points_per_effort : ℕ := 6 * 1/3)
                                       (AIME_points_per_effort : ℕ := 10 * 1/7)
                                       (USAMO_points_per_effort : ℕ := 1 * 1/10)
                                       (required_amc_aime_points : ℕ := 200)
                                       (required_usamo_points : ℕ := 21) : ℕ :=
  let max_amc_points : ℕ := 150
  let effort_amc : ℕ := (max_amc_points / AMC_points_per_effort) * 3
  let remaining_aime_points : ℕ := 200 - max_amc_points
  let effort_aime : ℕ := (remaining_aime_points / AIME_points_per_effort) * 7
  let effort_usamo : ℕ := required_usamo_points * 10
  let total_effort : ℕ := effort_amc + effort_aime + effort_usamo
  total_effort

theorem minimum_effort_to_qualify : minimum_effort_to_qualify_for_mop 6 (10 * 1/7) (1 * 1/10) 200 21 = 320 := by
  sorry

end minimum_effort_to_qualify_l1441_144180


namespace total_cost_is_eight_times_l1441_144164

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l1441_144164


namespace parabola_intersections_l1441_144136

open Real

-- Definition of the two parabolas
def parabola1 (x : ℝ) : ℝ := 3*x^2 - 6*x + 2
def parabola2 (x : ℝ) : ℝ := 9*x^2 - 4*x - 5

-- Theorem stating the intersections are (-7/3, 9) and (0.5, -0.25)
theorem parabola_intersections : 
  {p : ℝ × ℝ | parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2} =
  {(-7/3, 9), (0.5, -0.25)} :=
by 
  sorry

end parabola_intersections_l1441_144136


namespace remainder_1125_1127_1129_div_12_l1441_144162

theorem remainder_1125_1127_1129_div_12 :
  (1125 * 1127 * 1129) % 12 = 3 :=
by
  -- Proof can be written here
  sorry

end remainder_1125_1127_1129_div_12_l1441_144162


namespace find_b_l1441_144117

-- Define the conditions as given in the problem
def poly1 (x : ℝ) : ℝ := x^2 - 2 * x - 1
def poly2 (x a b : ℝ) : ℝ := a * x^3 + b * x^2 + 1

-- Define the problem statement using these conditions
theorem find_b (a b : ℤ) (h : ∀ x, poly1 x = 0 → poly2 x a b = 0) : b = -3 :=
sorry

end find_b_l1441_144117


namespace gcd_45123_32768_l1441_144100

theorem gcd_45123_32768 : Nat.gcd 45123 32768 = 1 := by
  sorry

end gcd_45123_32768_l1441_144100


namespace radius_of_inscribed_circle_l1441_144177

noncomputable def inscribed_circle_radius (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_of_inscribed_circle :
  inscribed_circle_radius 6 8 10 = 2 :=
by
  sorry

end radius_of_inscribed_circle_l1441_144177


namespace find_pqr_eq_1680_l1441_144172

theorem find_pqr_eq_1680
  {p q r : ℤ} (hpqz : p ≠ 0) (hqqz : q ≠ 0) (hrqz : r ≠ 0)
  (h_sum : p + q + r = 30)
  (h_cond : (1:ℚ) / p + (1:ℚ) / q + (1:ℚ) / r + 390 / (p * q * r) = 1) :
  p * q * r = 1680 :=
sorry

end find_pqr_eq_1680_l1441_144172


namespace height_at_2_years_l1441_144146

variable (height : ℕ → ℕ) -- height function representing the height of the tree at the end of n years
variable (triples_height : ∀ n, height (n + 1) = 3 * height n) -- tree triples its height every year
variable (height_4 : height 4 = 81) -- height at the end of 4 years is 81 feet

-- We need the height at the end of 2 years
theorem height_at_2_years : height 2 = 9 :=
by {
  sorry
}

end height_at_2_years_l1441_144146


namespace students_more_than_guinea_pigs_l1441_144152

-- Definitions based on the problem's conditions
def students_per_classroom : Nat := 22
def guinea_pigs_per_classroom : Nat := 3
def classrooms : Nat := 5

-- The proof statement
theorem students_more_than_guinea_pigs :
  (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 95 :=
by
  sorry

end students_more_than_guinea_pigs_l1441_144152


namespace virginia_more_than_adrienne_l1441_144134

def teaching_years (V A D : ℕ) : Prop :=
  V + A + D = 102 ∧ D = 43 ∧ V = D - 9

theorem virginia_more_than_adrienne (V A : ℕ) (h : teaching_years V A 43) : V - A = 9 :=
by
  sorry

end virginia_more_than_adrienne_l1441_144134


namespace inequality_solution_l1441_144197

theorem inequality_solution (x : ℝ) : x ^ 2 < |x| + 2 ↔ -2 < x ∧ x < 2 :=
by
  sorry

end inequality_solution_l1441_144197


namespace first_part_results_count_l1441_144159

theorem first_part_results_count : 
    ∃ n, n * 10 + 90 + (25 - n) * 20 = 25 * 18 ∧ n = 14 :=
by
  sorry

end first_part_results_count_l1441_144159


namespace sum_of_reciprocals_of_squares_l1441_144179

theorem sum_of_reciprocals_of_squares (x y : ℕ) (hxy : x * y = 17) : 
  1 / (x:ℚ)^2 + 1 / (y:ℚ)^2 = 290 / 289 := 
by
  sorry

end sum_of_reciprocals_of_squares_l1441_144179


namespace fair_split_adjustment_l1441_144187

theorem fair_split_adjustment
    (A B : ℝ)
    (h : A < B)
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 120)
    (h2 : d2 = 150)
    (h3 : d3 = 180)
    (bernardo_pays_twice : ∀ D, (2 : ℝ) * D = d1 + d2 + d3) :
    (B - A) / 2 - 75 = ((d1 + d2 + d3) - 450) / 2 - (A - (d1 + d2 + d3) / 3) :=
by
  sorry

end fair_split_adjustment_l1441_144187


namespace angle_OA_plane_ABC_l1441_144130

noncomputable def sphere_radius (A B C : Type*) (O : Type*) : ℝ :=
  let surface_area : ℝ := 48 * Real.pi
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let radius := Real.sqrt (surface_area / (4 * Real.pi))
  radius

noncomputable def length_AC (A B C : Type*) : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let AC := Real.sqrt (AB ^ 2 + BC ^ 2 - 2 * AB * BC * Real.cos angle_ABC)
  AC

theorem angle_OA_plane_ABC 
(A B C O : Type*)
(radius : ℝ)
(AC : ℝ) :
radius = 2 * Real.sqrt 3 ∧
AC = 2 * Real.sqrt 3 ∧ 
(AB : ℝ) = 2 ∧ 
(BC : ℝ) = 4 ∧ 
(angle_ABC : ℝ) = Real.pi / 3
→ ∃ (angle_OA_plane_ABC : ℝ), angle_OA_plane_ABC = Real.arccos (Real.sqrt 3 / 3) :=
by
  intro h
  sorry

end angle_OA_plane_ABC_l1441_144130


namespace ineq_one_of_two_sqrt_amgm_l1441_144199

-- Lean 4 statement for Question 1
theorem ineq_one_of_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

-- Lean 4 statement for Question 2
theorem sqrt_amgm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 :=
sorry

end ineq_one_of_two_sqrt_amgm_l1441_144199


namespace greatest_possible_z_l1441_144116

theorem greatest_possible_z (x y z : ℕ) (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z)
  (hx_cond : 7 < x) (hy_cond : y < 15) (hx_lt_y : x < y) (hz_gt_zero : z > 0) 
  (hy_sub_x_div_z : (y - x) % z = 0) : z = 2 := 
sorry

end greatest_possible_z_l1441_144116


namespace chord_length_ne_l1441_144148

-- Define the ellipse
def ellipse (x y : ℝ) := (x^2 / 8) + (y^2 / 4) = 1

-- Define the first line
def line_l (k x : ℝ) := (k * x + 1)

-- Define the second line
def line_l_option_D (k x y : ℝ) := (k * x + y - 2)

-- Prove the chord length inequality for line_l_option_D
theorem chord_length_ne (k : ℝ) :
  ∀ x y : ℝ, ellipse x y →
  ∃ x1 x2 y1 y2 : ℝ, ellipse x1 y1 ∧ line_l k x1 = y1 ∧ ellipse x2 y2 ∧ line_l k x2 = y2 ∧
  ∀ x3 x4 y3 y4 : ℝ, ellipse x3 y3 ∧ line_l_option_D k x3 y3 = 0 ∧ ellipse x4 y4 ∧ line_l_option_D k x4 y4 = 0 →
  dist (x1, y1) (x2, y2) ≠ dist (x3, y3) (x4, y4) :=
sorry

end chord_length_ne_l1441_144148


namespace ken_climbing_pace_l1441_144155

noncomputable def sari_pace : ℝ := 350 -- Sari's pace in meters per hour, derived from 700 meters in 2 hours.

def ken_pace : ℝ := 500 -- We will need to prove this.

theorem ken_climbing_pace :
  let start_time_sari := 5
  let start_time_ken := 7
  let end_time_ken := 12
  let time_ken_climbs := end_time_ken - start_time_ken
  let sari_initial_headstart := 700 -- meters
  let sari_behind_ken := 50 -- meters
  let sari_total_climb := sari_pace * time_ken_climbs
  let total_distance_ken := sari_total_climb + sari_initial_headstart + sari_behind_ken
  ken_pace = total_distance_ken / time_ken_climbs :=
by
  sorry

end ken_climbing_pace_l1441_144155


namespace not_divisible_by_2006_l1441_144178

theorem not_divisible_by_2006 (k : ℤ) : ¬ ∃ m : ℤ, k^2 + k + 1 = 2006 * m :=
sorry

end not_divisible_by_2006_l1441_144178


namespace sphere_surface_area_l1441_144198

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : ∀ r, V = (4/3) * π * r^3)
  (h2 : V = 72 * π) : A = 36 * π * 2^(2/3) :=
by 
  sorry

end sphere_surface_area_l1441_144198


namespace find_third_side_l1441_144169

def vol_of_cube (side : ℝ) : ℝ := side ^ 3

def vol_of_box (length width height : ℝ) : ℝ := length * width * height

theorem find_third_side (n : ℝ) (vol_cube : ℝ) (num_cubes : ℝ) (l w : ℝ) (vol_box : ℝ) :
  num_cubes = 24 →
  vol_cube = 27 →
  l = 8 →
  w = 12 →
  vol_box = num_cubes * vol_cube →
  vol_box = vol_of_box l w n →
  n = 6.75 :=
by
  intros hcubes hc_vol hl hw hvbox1 hvbox2
  -- The proof goes here
  sorry

end find_third_side_l1441_144169


namespace original_amount_spent_l1441_144123

noncomputable def price_per_mango : ℝ := 383.33 / 115
noncomputable def new_price_per_mango : ℝ := 0.9 * price_per_mango

theorem original_amount_spent (N : ℝ) (H1 : (N + 12) * new_price_per_mango = N * price_per_mango) : 
  N * price_per_mango = 359.64 :=
by 
  sorry

end original_amount_spent_l1441_144123


namespace onions_total_l1441_144109

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ) 
  (h1: Sara_onions = 4) (h2: Sally_onions = 5) (h3: Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 :=
by
  sorry

end onions_total_l1441_144109


namespace shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l1441_144175

def false_weight_kgs (false_weight_g : ℕ) : ℚ := false_weight_g / 1000

def shopkeeper_gain_percentage (false_weight_g price_per_kg : ℕ) : ℚ :=
  let actual_price := false_weight_kgs false_weight_g * price_per_kg
  let gain := price_per_kg - actual_price
  (gain / actual_price) * 100

theorem shopkeeper_gain_first_pulse :
  shopkeeper_gain_percentage 950 10 = 5.26 := 
sorry

theorem shopkeeper_gain_second_pulse :
  shopkeeper_gain_percentage 960 15 = 4.17 := 
sorry

theorem shopkeeper_gain_third_pulse :
  shopkeeper_gain_percentage 970 20 = 3.09 := 
sorry

end shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l1441_144175


namespace highest_numbered_street_l1441_144138

theorem highest_numbered_street (L : ℕ) (d : ℕ) (H : L = 15000 ∧ d = 500) : 
    (L / d) - 2 = 28 :=
by
  sorry

end highest_numbered_street_l1441_144138


namespace exists_infinitely_many_n_l1441_144132

def sum_of_digits (m : ℕ) : ℕ := 
  m.digits 10 |>.sum

theorem exists_infinitely_many_n (S : ℕ → ℕ) (h_sum_of_digits : ∀ m, S m = sum_of_digits m) :
  ∀ N : ℕ, ∃ n ≥ N, S (3 ^ n) ≥ S (3 ^ (n + 1)) :=
by { sorry }

end exists_infinitely_many_n_l1441_144132


namespace tangents_collinear_F_minimum_area_triangle_l1441_144195

noncomputable def ellipse_condition : Prop :=
  ∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1

noncomputable def point_P_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4

noncomputable def tangent_condition (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop) : Prop :=
  -- Tangent lines meet the ellipse equation at points A and B
  ellipse A ∧ ellipse B

noncomputable def collinear (A F B : ℝ × ℝ) : Prop :=
  (A.2 - F.2) * (B.1 - F.1) = (B.2 - F.2) * (A.1 - F.1)

noncomputable def minimum_area (P A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((A.1 * B.2 + B.1 * P.2 + P.1 * A.2) - (A.2 * B.1 + B.2 * P.1 + P.2 * A.1))

theorem tangents_collinear_F (F : ℝ × ℝ) (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  collinear A F B :=
sorry

theorem minimum_area_triangle (F P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  minimum_area P A B = 9 / 2 :=
sorry

end tangents_collinear_F_minimum_area_triangle_l1441_144195


namespace enter_exit_ways_eq_sixteen_l1441_144122

theorem enter_exit_ways_eq_sixteen (n : ℕ) (h : n = 4) : n * n = 16 :=
by sorry

end enter_exit_ways_eq_sixteen_l1441_144122


namespace sum_divides_exp_sum_l1441_144181

theorem sum_divides_exp_sum (p a b c d : ℕ) [Fact (Nat.Prime p)] 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < p)
  (h6 : a^4 % p = b^4 % p) (h7 : b^4 % p = c^4 % p) (h8 : c^4 % p = d^4 % p) :
  (a + b + c + d) ∣ (a^2013 + b^2013 + c^2013 + d^2013) :=
sorry

end sum_divides_exp_sum_l1441_144181


namespace paint_price_and_max_boxes_l1441_144168

theorem paint_price_and_max_boxes (x y m : ℕ) 
  (h1 : x + 2 * y = 56) 
  (h2 : 2 * x + y = 64) 
  (h3 : 24 * m + 16 * (200 - m) ≤ 3920) : 
  x = 24 ∧ y = 16 ∧ m ≤ 90 := 
  by 
    sorry

end paint_price_and_max_boxes_l1441_144168


namespace tax_difference_is_250000_l1441_144141

noncomputable def old_tax_rate : ℝ := 0.20
noncomputable def new_tax_rate : ℝ := 0.30
noncomputable def old_income : ℝ := 1000000
noncomputable def new_income : ℝ := 1500000
noncomputable def old_taxes_paid := old_tax_rate * old_income
noncomputable def new_taxes_paid := new_tax_rate * new_income
noncomputable def tax_difference := new_taxes_paid - old_taxes_paid

theorem tax_difference_is_250000 : tax_difference = 250000 := by
  sorry

end tax_difference_is_250000_l1441_144141


namespace min_value_frac_sum_l1441_144193

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  ∃ (a b : ℝ), (a + 3 * b = 1) ∧ (a > 0) ∧ (b > 0) ∧ (∀ (a b : ℝ), (a + 3 * b = 1) → 0 < a → 0 < b → (1 / a + 3 / b) ≥ 16) :=
sorry

end min_value_frac_sum_l1441_144193


namespace smallest_possible_n_l1441_144190

theorem smallest_possible_n (n : ℕ) (h1 : n ≥ 100) (h2 : n < 1000)
  (h3 : n % 9 = 2) (h4 : n % 7 = 2) : n = 128 :=
by
  sorry

end smallest_possible_n_l1441_144190


namespace h_inverse_left_h_inverse_right_l1441_144145

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1
noncomputable def h (x : ℝ) : ℝ := f (g x)
noncomputable def h_inv (y : ℝ) : ℝ := 1 + (Real.sqrt (3 * y + 12)) / 4 -- Correct answer

-- Theorem statements to prove the inverse relationship
theorem h_inverse_left (x : ℝ) : h (h_inv x) = x :=
by
  sorry -- Proof of the left inverse

theorem h_inverse_right (y : ℝ) : h_inv (h y) = y :=
by
  sorry -- Proof of the right inverse

end h_inverse_left_h_inverse_right_l1441_144145


namespace solve_fraction_equation_l1441_144192

theorem solve_fraction_equation : ∀ (x : ℝ), (x + 2) / (2 * x - 1) = 1 → x = 3 :=
by
  intros x h
  sorry

end solve_fraction_equation_l1441_144192


namespace rhombus_perimeter_l1441_144151

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  ∃ p : ℝ, p = 16 * Real.sqrt 13 := by
sorry

end rhombus_perimeter_l1441_144151


namespace fifth_derivative_l1441_144160

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 - 7) * Real.log (x - 1)

theorem fifth_derivative :
  ∀ x, (deriv^[5] f) x = 8 * (x ^ 2 - 5 * x - 11) / ((x - 1) ^ 5) :=
by
  sorry

end fifth_derivative_l1441_144160


namespace find_a₁_l1441_144149

noncomputable def S_3 (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q^2

theorem find_a₁ (S₃_eq : S_3 a₁ q = a₁ + 3 * (a₁ * q)) (a₄_eq : a₁ * q^3 = 8) : a₁ = 1 :=
by
  -- proof skipped
  sorry

end find_a₁_l1441_144149


namespace geometric_sequence_8th_term_l1441_144185

theorem geometric_sequence_8th_term (a : ℚ) (r : ℚ) (n : ℕ) (h_a : a = 27) (h_r : r = 2/3) (h_n : n = 8) :
  a * r^(n-1) = 128 / 81 :=
by
  rw [h_a, h_r, h_n]
  sorry

end geometric_sequence_8th_term_l1441_144185


namespace square_integer_2209_implies_value_l1441_144113

theorem square_integer_2209_implies_value (x : ℤ) (h : x^2 = 2209) : (2*x + 1)*(2*x - 1) = 8835 :=
by sorry

end square_integer_2209_implies_value_l1441_144113


namespace binary_to_octal_of_101101110_l1441_144119

def binaryToDecimal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 2 + b) 0 (Nat.digits 2 n)

def decimalToOctal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 10 + b) 0 (Nat.digits 8 n)

theorem binary_to_octal_of_101101110 :
  decimalToOctal (binaryToDecimal 0b101101110) = 556 :=
by sorry

end binary_to_octal_of_101101110_l1441_144119


namespace ratio_of_ages_l1441_144115

-- Definitions of the conditions
def son_current_age : ℕ := 28
def man_current_age : ℕ := son_current_age + 30
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem
theorem ratio_of_ages : (man_age_in_two_years / son_age_in_two_years) = 2 :=
by
  -- Skipping the proof steps
  sorry

end ratio_of_ages_l1441_144115


namespace sin_tan_correct_value_l1441_144167

noncomputable def sin_tan_value (x y : ℝ) (h : x^2 + y^2 = 1) : ℝ :=
  let sin_alpha := y
  let tan_alpha := y / x
  sin_alpha * tan_alpha

theorem sin_tan_correct_value :
  sin_tan_value (3/5) (-4/5) (by norm_num) = 16/15 := 
by
  sorry

end sin_tan_correct_value_l1441_144167


namespace Chloe_pairs_shoes_l1441_144194

theorem Chloe_pairs_shoes (cost_per_shoe total_cost : ℤ) (h_cost: cost_per_shoe = 37) (h_total: total_cost = 1036) :
  (total_cost / cost_per_shoe) / 2 = 14 :=
by
  -- proof goes here
  sorry

end Chloe_pairs_shoes_l1441_144194


namespace find_sum_of_money_invested_l1441_144104

theorem find_sum_of_money_invested (P : ℝ) (h1 : SI_15 = P * (15 / 100) * 2)
                                    (h2 : SI_12 = P * (12 / 100) * 2)
                                    (h3 : SI_15 - SI_12 = 720) : 
                                    P = 12000 :=
by
  -- Skipping the proof
  sorry

end find_sum_of_money_invested_l1441_144104


namespace find_x_for_which_ffx_eq_fx_l1441_144114

def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem find_x_for_which_ffx_eq_fx :
  {x : ℝ | f (f x) = f x} = {0, 4, 5, -1} :=
by
  sorry

end find_x_for_which_ffx_eq_fx_l1441_144114


namespace contributions_before_john_l1441_144137

theorem contributions_before_john (n : ℕ) (A : ℚ) 
  (h1 : 1.5 * A = 75)
  (h2 : (n * A + 225) / (n + 1) = 75) : n = 6 :=
by {
  sorry
}

end contributions_before_john_l1441_144137


namespace largest_n_sum_pos_l1441_144156

section
variables {a : ℕ → ℤ}
variables {d : ℤ}
variables {n : ℕ}

axiom a_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom a1_pos : a 1 > 0
axiom a2013_2014_pos : a 2013 + a 2014 > 0
axiom a2013_2014_neg : a 2013 * a 2014 < 0

theorem largest_n_sum_pos :
  ∃ n : ℕ, (∀ k ≤ n, (k * (2 * a 1 + (k - 1) * d) / 2) > 0) → n = 4026 := sorry

end

end largest_n_sum_pos_l1441_144156


namespace solve_for_x_l1441_144161

theorem solve_for_x (x : ℤ) (h : 3 * x = 2 * x + 6) : x = 6 := by
  sorry

end solve_for_x_l1441_144161


namespace base_length_of_isosceles_triangle_l1441_144139

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end base_length_of_isosceles_triangle_l1441_144139


namespace Deepak_age_l1441_144120

theorem Deepak_age (A D : ℕ) (h1 : A / D = 4 / 3) (h2 : A + 6 = 26) : D = 15 :=
by
  sorry

end Deepak_age_l1441_144120


namespace cherries_initially_l1441_144127

theorem cherries_initially (x : ℕ) (h₁ : x - 6 = 10) : x = 16 :=
by
  sorry

end cherries_initially_l1441_144127


namespace hexagon_largest_angle_l1441_144101

theorem hexagon_largest_angle (x : ℝ) 
  (h_angles_sum : 80 + 100 + x + x + x + (2 * x + 20) = 720) : 
  (2 * x + 20) = 228 :=
by 
  sorry

end hexagon_largest_angle_l1441_144101


namespace cos_diff_l1441_144124

theorem cos_diff (x y : ℝ) (hx : x = Real.cos (20 * Real.pi / 180)) (hy : y = Real.cos (40 * Real.pi / 180)) :
  x - y = 1 / 2 :=
by sorry

end cos_diff_l1441_144124


namespace circle_line_intersection_l1441_144142

theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x - y + 1 = 0) ∧ ((x - a)^2 + y^2 = 2)) ↔ -3 ≤ a ∧ a ≤ 1 := 
by
  sorry

end circle_line_intersection_l1441_144142


namespace sqrt_product_simplification_l1441_144158

variable (q : ℝ)

theorem sqrt_product_simplification (hq : q ≥ 0) : 
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (12 * q^5) = 12 * q^4 * Real.sqrt (10 * q) :=
by
  sorry

end sqrt_product_simplification_l1441_144158


namespace optimal_purchasing_plan_l1441_144150

def price_carnation := 5
def price_lily := 10
def total_flowers := 300
def max_carnations (x : ℕ) : Prop := x ≤ 2 * (total_flowers - x)

theorem optimal_purchasing_plan :
  ∃ (x y : ℕ), (x + y = total_flowers) ∧ (x = 200) ∧ (y = 100) ∧ (max_carnations x) ∧ 
  ∀ (x' y' : ℕ), (x' + y' = total_flowers) → max_carnations x' →
    (price_carnation * x + price_lily * y ≤ price_carnation * x' + price_lily * y') :=
by
  sorry

end optimal_purchasing_plan_l1441_144150


namespace area_of_triangle_formed_by_tangent_line_l1441_144121

noncomputable def curve (x : ℝ) : ℝ := Real.log x - 2 * x

noncomputable def slope_of_tangent_at (x : ℝ) : ℝ := (1 / x) - 2

def point_of_tangency : ℝ × ℝ := (1, -2)

-- Define the tangent line equation at the point (1, -2)
noncomputable def tangent_line (x : ℝ) : ℝ := -x - 1

-- Define x and y intercepts of the tangent line
def x_intercept_of_tangent : ℝ := -1
def y_intercept_of_tangent : ℝ := -1

-- Define the area of the triangle formed by the tangent line and the coordinate axes
def triangle_area : ℝ := 0.5 * (-1) * (-1)

-- State the theorem to prove the area of the triangle
theorem area_of_triangle_formed_by_tangent_line : 
  triangle_area = 0.5 := by 
sorry

end area_of_triangle_formed_by_tangent_line_l1441_144121


namespace enthalpy_change_l1441_144188

def DeltaH_prods : Float := -286.0 - 297.0
def DeltaH_reacts : Float := -20.17
def HessLaw (DeltaH_prods DeltaH_reacts : Float) : Float := DeltaH_prods - DeltaH_reacts

theorem enthalpy_change : HessLaw DeltaH_prods DeltaH_reacts = -1125.66 := by
  -- Lean needs a proof, which is not needed per instructions
  sorry

end enthalpy_change_l1441_144188


namespace absolute_sum_value_l1441_144183

theorem absolute_sum_value (x1 x2 x3 x4 x5 : ℝ) 
(h : x1 + 1 = x2 + 2 ∧ x2 + 2 = x3 + 3 ∧ x3 + 3 = x4 + 4 ∧ x4 + 4 = x5 + 5 ∧ x5 + 5 = x1 + x2 + x3 + x4 + x5 + 6) :
  |(x1 + x2 + x3 + x4 + x5)| = 3.75 := 
by
  sorry

end absolute_sum_value_l1441_144183


namespace f_is_even_l1441_144106

noncomputable def f (x : ℝ) : ℝ := x ^ 2

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := 
by
  intros x
  sorry

end f_is_even_l1441_144106


namespace smallest_pos_multiple_6_15_is_30_l1441_144140

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l1441_144140
