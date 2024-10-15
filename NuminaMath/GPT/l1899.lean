import Mathlib

namespace NUMINAMATH_GPT_difference_of_numbers_l1899_189958

theorem difference_of_numbers (x y : ℝ) (h₁ : x + y = 25) (h₂ : x * y = 144) : |x - y| = 7 :=
sorry

end NUMINAMATH_GPT_difference_of_numbers_l1899_189958


namespace NUMINAMATH_GPT_fraction_addition_l1899_189967

theorem fraction_addition (x y : ℚ) (h : x / y = 2 / 3) : (x + y) / y = 5 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_addition_l1899_189967


namespace NUMINAMATH_GPT_max_values_of_x_max_area_abc_l1899_189991

noncomputable def m (x : ℝ) : ℝ × ℝ := ⟨2 * Real.sin x, Real.sin x - Real.cos x⟩
noncomputable def n (x : ℝ) : ℝ × ℝ := ⟨Real.sqrt 3 * Real.cos x, Real.sin x + Real.cos x⟩
noncomputable def f (x : ℝ) : ℝ := Prod.fst (m x) * Prod.fst (n x) + Prod.snd (m x) * Prod.snd (n x)

theorem max_values_of_x
  (k : ℤ) : ∃ x, x = k * Real.pi + Real.pi / 3 ∧ f x = 2 * Real.sin (2 * x - π / 6) :=
sorry

noncomputable def C : ℝ := Real.pi / 3
noncomputable def area_abc (a b c : ℝ) : ℝ := 1 / 2 * a * b * Real.sin C

theorem max_area_abc (a b : ℝ) (h₁ : c = Real.sqrt 3) (h₂ : f C = 2) :
  area_abc a b c ≤ 3 * Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_GPT_max_values_of_x_max_area_abc_l1899_189991


namespace NUMINAMATH_GPT_total_games_played_l1899_189956

def games_attended : ℕ := 14
def games_missed : ℕ := 25

theorem total_games_played : games_attended + games_missed = 39 :=
by
  sorry

end NUMINAMATH_GPT_total_games_played_l1899_189956


namespace NUMINAMATH_GPT_symmetric_point_correct_l1899_189952

-- Define the coordinates of point A
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D :=
  { x := -3
    y := -4
    z := 5 }

-- Define the symmetry function with respect to the plane xOz
def symmetric_xOz (p : Point3D) : Point3D :=
  { p with y := -p.y }

-- The expected coordinates of the point symmetric to A with respect to the plane xOz
def D_expected : Point3D :=
  { x := -3
    y := 4
    z := 5 }

-- Theorem stating that the symmetric point of A with respect to the plane xOz is D_expected
theorem symmetric_point_correct :
  symmetric_xOz A = D_expected := 
by 
  sorry

end NUMINAMATH_GPT_symmetric_point_correct_l1899_189952


namespace NUMINAMATH_GPT_egg_problem_l1899_189976

theorem egg_problem :
  ∃ (N F E : ℕ), N + F + E = 100 ∧ 5 * N + F + E / 2 = 100 ∧ (N = F ∨ N = E ∨ F = E) ∧ N = 10 ∧ F = 10 ∧ E = 80 :=
by
  sorry

end NUMINAMATH_GPT_egg_problem_l1899_189976


namespace NUMINAMATH_GPT_new_avg_weight_l1899_189937

-- Define the weights of individuals
variables (A B C D E : ℕ)
-- Conditions
axiom avg_ABC : (A + B + C) / 3 = 84
axiom avg_ABCD : (A + B + C + D) / 4 = 80
axiom E_def : E = D + 8
axiom A_80 : A = 80

theorem new_avg_weight (A B C D E : ℕ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 8) 
  (h4 : A = 80) 
  : (B + C + D + E) / 4 = 79 := 
by
  sorry

end NUMINAMATH_GPT_new_avg_weight_l1899_189937


namespace NUMINAMATH_GPT_largest_sum_l1899_189986

theorem largest_sum :
  max (max (max (max (1/4 + 1/9) (1/4 + 1/10)) (1/4 + 1/11)) (1/4 + 1/12)) (1/4 + 1/13) = 13/36 := 
sorry

end NUMINAMATH_GPT_largest_sum_l1899_189986


namespace NUMINAMATH_GPT_general_solution_of_differential_equation_l1899_189975

theorem general_solution_of_differential_equation (a₀ : ℝ) (x : ℝ) :
  ∃ y : ℝ → ℝ, (∀ x, deriv y x = (y x)^2) ∧ y x = a₀ / (1 - a₀ * x) :=
sorry

end NUMINAMATH_GPT_general_solution_of_differential_equation_l1899_189975


namespace NUMINAMATH_GPT_wheel_revolutions_l1899_189951

theorem wheel_revolutions (r_course r_wheel : ℝ) (laps : ℕ) (C_course C_wheel : ℝ) (d_total : ℝ) :
  r_course = 7 →
  r_wheel = 5 →
  laps = 15 →
  C_course = 2 * Real.pi * r_course →
  d_total = laps * C_course →
  C_wheel = 2 * Real.pi * r_wheel →
  ((d_total) / (C_wheel)) = 21 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_wheel_revolutions_l1899_189951


namespace NUMINAMATH_GPT_find_a_l1899_189936

noncomputable def slope1 (a : ℝ) : ℝ := -3 / (3^a - 3)
noncomputable def slope2 : ℝ := 2

theorem find_a (a : ℝ) (h : slope1 a * slope2 = -1) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1899_189936


namespace NUMINAMATH_GPT_jane_earnings_two_weeks_l1899_189954

def num_chickens : ℕ := 10
def num_eggs_per_chicken_per_week : ℕ := 6
def dollars_per_dozen : ℕ := 2
def dozens_in_12_eggs : ℕ := 12

theorem jane_earnings_two_weeks :
  (num_chickens * num_eggs_per_chicken_per_week * 2 / dozens_in_12_eggs * dollars_per_dozen) = 20 := by
  sorry

end NUMINAMATH_GPT_jane_earnings_two_weeks_l1899_189954


namespace NUMINAMATH_GPT_solve_for_x_l1899_189945

theorem solve_for_x (x : ℝ) (h : 2 * (1/x + 3/x / 6/x) - 1/x = 1.5) : x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1899_189945


namespace NUMINAMATH_GPT_particle_hits_origin_l1899_189939

def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x, 0 => 0
| 0, y => 0
| x+1, y+1 => 0.25 * P x (y+1) + 0.25 * P (x+1) y + 0.5 * P x y

theorem particle_hits_origin :
    ∃ m n : ℕ, m ≠ 0 ∧ m % 4 ≠ 0 ∧ P 5 5 = m / 4^n :=
sorry

end NUMINAMATH_GPT_particle_hits_origin_l1899_189939


namespace NUMINAMATH_GPT_alien_abduction_problem_l1899_189924

theorem alien_abduction_problem:
  ∀ (total_abducted people_taken_elsewhere people_taken_home people_returned: ℕ),
  total_abducted = 200 →
  people_taken_elsewhere = 10 →
  people_taken_home = 30 →
  people_returned = total_abducted - (people_taken_elsewhere + people_taken_home) →
  (people_returned : ℕ) / total_abducted * 100 = 80 := 
by
  intros total_abducted people_taken_elsewhere people_taken_home people_returned;
  intros h_total_abducted h_taken_elsewhere h_taken_home h_people_returned;
  sorry

end NUMINAMATH_GPT_alien_abduction_problem_l1899_189924


namespace NUMINAMATH_GPT_ratio_equiv_solve_x_l1899_189911

theorem ratio_equiv_solve_x (x : ℕ) (h : 3 / 12 = 3 / x) : x = 12 :=
sorry

end NUMINAMATH_GPT_ratio_equiv_solve_x_l1899_189911


namespace NUMINAMATH_GPT_monic_poly_7_r_8_l1899_189946

theorem monic_poly_7_r_8 :
  ∃ (r : ℕ → ℕ), (r 1 = 1) ∧ (r 2 = 2) ∧ (r 3 = 3) ∧ (r 4 = 4) ∧ (r 5 = 5) ∧ (r 6 = 6) ∧ (r 7 = 7) ∧ (∀ (n : ℕ), 8 < n → r n = n) ∧ r 8 = 5048 :=
sorry

end NUMINAMATH_GPT_monic_poly_7_r_8_l1899_189946


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1899_189930

theorem isosceles_triangle_perimeter
  (a b c : ℝ )
  (ha : a = 20)
  (hb : b = 20)
  (hc : c = (2/5) * 20)
  (triangle_ineq1 : a ≤ b + c)
  (triangle_ineq2 : b ≤ a + c)
  (triangle_ineq3 : c ≤ a + b) :
  a + b + c = 48 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1899_189930


namespace NUMINAMATH_GPT_rotation_90_deg_l1899_189993

theorem rotation_90_deg (z : ℂ) (r : ℂ → ℂ) (h : ∀ (x y : ℝ), r (x + y*I) = -y + x*I) :
  r (8 - 5*I) = 5 + 8*I :=
by sorry

end NUMINAMATH_GPT_rotation_90_deg_l1899_189993


namespace NUMINAMATH_GPT_trigonometric_identity_l1899_189970

open Real

theorem trigonometric_identity (α β : ℝ) :
  sin (2 * α) ^ 2 + sin β ^ 2 + cos (2 * α + β) * cos (2 * α - β) = 1 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1899_189970


namespace NUMINAMATH_GPT_unique_nat_number_sum_preceding_eq_self_l1899_189910

theorem unique_nat_number_sum_preceding_eq_self :
  ∃! (n : ℕ), (n * (n - 1)) / 2 = n :=
sorry

end NUMINAMATH_GPT_unique_nat_number_sum_preceding_eq_self_l1899_189910


namespace NUMINAMATH_GPT_complex_division_example_l1899_189974

theorem complex_division_example : (2 - (1 : ℂ) * Complex.I) / (1 - (1 : ℂ) * Complex.I) = (3 / 2) + (1 / 2) * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_complex_division_example_l1899_189974


namespace NUMINAMATH_GPT_roots_of_unity_real_root_l1899_189949

theorem roots_of_unity_real_root (n : ℕ) (h_even : n % 2 = 0) : ∃ z : ℝ, z ≠ 1 ∧ z^n = 1 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_unity_real_root_l1899_189949


namespace NUMINAMATH_GPT_total_hamburgers_sold_is_63_l1899_189920

-- Define the average number of hamburgers sold each day
def avg_hamburgers_per_day : ℕ := 9

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total hamburgers sold in a week
def total_hamburgers_sold : ℕ := avg_hamburgers_per_day * days_in_week

-- Prove that the total hamburgers sold in a week is 63
theorem total_hamburgers_sold_is_63 : total_hamburgers_sold = 63 :=
by
  sorry

end NUMINAMATH_GPT_total_hamburgers_sold_is_63_l1899_189920


namespace NUMINAMATH_GPT_units_digit_a_2017_l1899_189917

noncomputable def a_n (n : ℕ) : ℝ :=
  (Real.sqrt 2 + 1) ^ n - (Real.sqrt 2 - 1) ^ n

theorem units_digit_a_2017 : (Nat.floor (a_n 2017)) % 10 = 2 :=
  sorry

end NUMINAMATH_GPT_units_digit_a_2017_l1899_189917


namespace NUMINAMATH_GPT_sum_sin_double_angles_eq_l1899_189929

theorem sum_sin_double_angles_eq (
  α β γ : ℝ
) (h : α + β + γ = Real.pi) :
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 
  4 * Real.sin α * Real.sin β * Real.sin γ :=
sorry

end NUMINAMATH_GPT_sum_sin_double_angles_eq_l1899_189929


namespace NUMINAMATH_GPT_books_shelves_l1899_189992

def initial_books : ℝ := 40.0
def additional_books : ℝ := 20.0
def books_per_shelf : ℝ := 4.0

theorem books_shelves :
  (initial_books + additional_books) / books_per_shelf = 15 :=
by 
  sorry

end NUMINAMATH_GPT_books_shelves_l1899_189992


namespace NUMINAMATH_GPT_number_of_people_in_village_l1899_189979

variable (P : ℕ) -- Define the total number of people in the village

def people_not_working : ℕ := 50
def people_with_families : ℕ := 25
def people_singing_in_shower : ℕ := 75
def max_people_overlap : ℕ := 50

theorem number_of_people_in_village :
  P - people_not_working + P - people_with_families + P - people_singing_in_shower - max_people_overlap = P → 
  P = 100 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_in_village_l1899_189979


namespace NUMINAMATH_GPT_amy_soups_total_l1899_189907

def total_soups (chicken_soups tomato_soups : ℕ) : ℕ :=
  chicken_soups + tomato_soups

theorem amy_soups_total : total_soups 6 3 = 9 :=
by
  -- insert the proof here
  sorry

end NUMINAMATH_GPT_amy_soups_total_l1899_189907


namespace NUMINAMATH_GPT_no_correlation_pair_D_l1899_189941

-- Define the pairs of variables and their relationships
def pair_A : Prop := ∃ (fertilizer_applied grain_yield : ℝ), (fertilizer_applied ≠ 0 → grain_yield ≠ 0)
def pair_B : Prop := ∃ (review_time scores : ℝ), (review_time ≠ 0 → scores ≠ 0)
def pair_C : Prop := ∃ (advertising_expenses sales : ℝ), (advertising_expenses ≠ 0 → sales ≠ 0)
def pair_D : Prop := ∃ (books_sold revenue : ℕ), (revenue = books_sold * 5)

/-- Prove that pair D does not have a correlation in the context of the problem. --/
theorem no_correlation_pair_D : ¬pair_D :=
by
  sorry

end NUMINAMATH_GPT_no_correlation_pair_D_l1899_189941


namespace NUMINAMATH_GPT_voucher_placement_l1899_189909

/-- A company wants to popularize the sweets they market by hiding prize vouchers in some of the boxes.
The management believes the promotion is effective and the cost is bearable if a customer who buys 10 boxes has approximately a 50% chance of finding at least one voucher.
We aim to determine how often vouchers should be placed in the boxes to meet this requirement. -/
theorem voucher_placement (n : ℕ) (h_positive : n > 0) :
  (1 - (1 - 1/n)^10) ≥ 1/2 → n ≤ 15 :=
sorry

end NUMINAMATH_GPT_voucher_placement_l1899_189909


namespace NUMINAMATH_GPT_krystiana_monthly_earnings_l1899_189981

-- Definitions based on the conditions
def first_floor_cost : ℕ := 15
def second_floor_cost : ℕ := 20
def third_floor_cost : ℕ := 2 * first_floor_cost
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms_occupied : ℕ := 2

-- Statement to prove Krystiana's total monthly earnings are $165
theorem krystiana_monthly_earnings : 
  first_floor_cost * first_floor_rooms + 
  second_floor_cost * second_floor_rooms + 
  third_floor_cost * third_floor_rooms_occupied = 165 :=
by admit

end NUMINAMATH_GPT_krystiana_monthly_earnings_l1899_189981


namespace NUMINAMATH_GPT_estimate_blue_balls_l1899_189997

theorem estimate_blue_balls (total_balls : ℕ) (prob_yellow : ℚ)
  (h_total : total_balls = 80)
  (h_prob_yellow : prob_yellow = 0.25) :
  total_balls * (1 - prob_yellow) = 60 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_estimate_blue_balls_l1899_189997


namespace NUMINAMATH_GPT_parent_combinations_for_O_l1899_189948

-- Define the blood types
inductive BloodType
| A
| B
| O
| AB

open BloodType

-- Define the conditions given in the problem
def parent_not_AB (p : BloodType) : Prop :=
  p ≠ AB

def possible_parent_types : List BloodType :=
  [A, B, O]

-- The math proof problem
theorem parent_combinations_for_O :
  ∀ (mother father : BloodType),
    parent_not_AB mother →
    parent_not_AB father →
    mother ∈ possible_parent_types →
    father ∈ possible_parent_types →
    (possible_parent_types.length * possible_parent_types.length) = 9 := 
by
  intro mother father h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_parent_combinations_for_O_l1899_189948


namespace NUMINAMATH_GPT_find_length_DC_l1899_189916

noncomputable def length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) : ℕ :=
  let DC := 29
  DC

theorem find_length_DC (AB BC AD : ℕ) (BD : ℕ) (h1 : AB = 52) (h2 : BC = 21) (h3 : AD = 48) (h4 : AB^2 = AD^2 + BD^2) (h5 : BD^2 = 20^2) (h6 : 20^2 + BC^2 = DC^2) : length_DC AB BC AD BD h1 h2 h3 h4 h5 = 29 :=
  by
  sorry

end NUMINAMATH_GPT_find_length_DC_l1899_189916


namespace NUMINAMATH_GPT_average_pregnancies_per_kettle_l1899_189962

-- Define the given conditions
def num_kettles : ℕ := 6
def babies_per_pregnancy : ℕ := 4
def survival_rate : ℝ := 0.75
def total_expected_babies : ℕ := 270

-- Calculate surviving babies per pregnancy
def surviving_babies_per_pregnancy : ℝ := babies_per_pregnancy * survival_rate

-- Prove that the average number of pregnancies per kettle is 15
theorem average_pregnancies_per_kettle : ∃ P : ℝ, num_kettles * P * surviving_babies_per_pregnancy = total_expected_babies ∧ P = 15 :=
by
  sorry

end NUMINAMATH_GPT_average_pregnancies_per_kettle_l1899_189962


namespace NUMINAMATH_GPT_breadth_decrease_percentage_l1899_189918

theorem breadth_decrease_percentage
  (L B : ℝ)
  (hLpos : L > 0)
  (hBpos : B > 0)
  (harea_change : (1.15 * L) * (B - p/100 * B) = 1.035 * (L * B)) :
  p = 10 := 
sorry

end NUMINAMATH_GPT_breadth_decrease_percentage_l1899_189918


namespace NUMINAMATH_GPT_chip_sheets_per_pack_l1899_189927

noncomputable def sheets_per_pack (pages_per_day : ℕ) (days_per_week : ℕ) (classes : ℕ) 
                                  (weeks : ℕ) (packs : ℕ) : ℕ :=
(pages_per_day * days_per_week * classes * weeks) / packs

theorem chip_sheets_per_pack :
  sheets_per_pack 2 5 5 6 3 = 100 :=
sorry

end NUMINAMATH_GPT_chip_sheets_per_pack_l1899_189927


namespace NUMINAMATH_GPT_inequality_proof_l1899_189942

variable (x y z : ℝ)

theorem inequality_proof (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
  ≥ Real.sqrt (3 / 2 * (x + y + z)) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1899_189942


namespace NUMINAMATH_GPT_find_second_expression_l1899_189999

theorem find_second_expression (a : ℕ) (x : ℕ) 
  (h1 : (2 * a + 16 + x) / 2 = 74) (h2 : a = 28) : x = 76 := 
by
  sorry

end NUMINAMATH_GPT_find_second_expression_l1899_189999


namespace NUMINAMATH_GPT_line_intersects_circle_two_points_find_value_of_m_l1899_189955

open Real

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

theorem line_intersects_circle_two_points (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ),
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  x1 ≠ x2 ∨ y1 ≠ y2 := sorry

theorem find_value_of_m (m : ℝ) : ∀ (x1 y1 x2 y2 : ℝ), 
  line_l m x1 y1 → circle_C x1 y1 →
  line_l m x2 y2 → circle_C x2 y2 →
  dist (x1, y1) (x2, y2) = sqrt 17 → 
  m = sqrt 3 ∨ m = -sqrt 3 := sorry

end NUMINAMATH_GPT_line_intersects_circle_two_points_find_value_of_m_l1899_189955


namespace NUMINAMATH_GPT_triangle_side_length_l1899_189961

theorem triangle_side_length (a : ℝ) (B : ℝ) (C : ℝ) (c : ℝ) 
  (h₀ : a = 10) (h₁ : B = 60) (h₂ : C = 45) : 
  c = 10 * (Real.sqrt 3 - 1) :=
sorry

end NUMINAMATH_GPT_triangle_side_length_l1899_189961


namespace NUMINAMATH_GPT_remainder_t4_mod7_l1899_189983

def T : ℕ → ℕ
| 0 => 0 -- Not used
| 1 => 6
| n+1 => 6 ^ (T n)

theorem remainder_t4_mod7 : (T 4 % 7) = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_t4_mod7_l1899_189983


namespace NUMINAMATH_GPT_rect_RS_over_HJ_zero_l1899_189904

theorem rect_RS_over_HJ_zero :
  ∃ (A B C D H I J R S: ℝ × ℝ),
    (A = (0, 6)) ∧
    (B = (8, 6)) ∧
    (C = (8, 0)) ∧
    (D = (0, 0)) ∧
    (H = (5, 6)) ∧
    (I = (8, 4)) ∧
    (J = (3, 0)) ∧
    (R = (15 / 13, -12 / 13)) ∧
    (S = (15 / 13, -12 / 13)) ∧
    (RS = dist R S) ∧
    (HJ = dist H J) ∧
    (HJ ≠ 0) ∧
    (RS / HJ = 0) :=
sorry

end NUMINAMATH_GPT_rect_RS_over_HJ_zero_l1899_189904


namespace NUMINAMATH_GPT_max_underwear_pairs_l1899_189928

-- Define the weights of different clothing items
def weight_socks : ℕ := 2
def weight_underwear : ℕ := 4
def weight_shirt : ℕ := 5
def weight_shorts : ℕ := 8
def weight_pants : ℕ := 10

-- Define the washing machine limit
def max_weight : ℕ := 50

-- Define the current load of clothes Tony plans to wash
def current_load : ℕ :=
  1 * weight_pants +
  2 * weight_shirt +
  1 * weight_shorts +
  3 * weight_socks

-- State the theorem regarding the maximum number of additional pairs of underwear
theorem max_underwear_pairs : 
  current_load ≤ max_weight →
  (max_weight - current_load) / weight_underwear = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_underwear_pairs_l1899_189928


namespace NUMINAMATH_GPT_bread_left_in_pond_l1899_189982

theorem bread_left_in_pond (total_bread : ℕ) 
                           (half_bread_duck : ℕ)
                           (second_duck_bread : ℕ)
                           (third_duck_bread : ℕ)
                           (total_bread_thrown : total_bread = 100)
                           (half_duck_eats : half_bread_duck = total_bread / 2)
                           (second_duck_eats : second_duck_bread = 13)
                           (third_duck_eats : third_duck_bread = 7) :
                           total_bread - (half_bread_duck + second_duck_bread + third_duck_bread) = 30 :=
    by
    sorry

end NUMINAMATH_GPT_bread_left_in_pond_l1899_189982


namespace NUMINAMATH_GPT_curve_transformation_l1899_189933

variable (x y x0 y0 : ℝ)

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -2], ![0, 1]]

def C (x0 y0 : ℝ) : Prop := (x0 - y0)^2 + y0^2 = 1

def transform (x0 y0 : ℝ) : ℝ × ℝ :=
  let x := 2 * x0 - 2 * y0
  let y := y0
  (x, y)

def C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

theorem curve_transformation :
  ∀ x0 y0, C x0 y0 → C' (2 * x0 - 2 * y0) y0 := sorry

end NUMINAMATH_GPT_curve_transformation_l1899_189933


namespace NUMINAMATH_GPT_chocolates_problem_l1899_189914

-- Let denote the quantities as follows:
-- C: number of caramels
-- N: number of nougats
-- T: number of truffles
-- P: number of peanut clusters

def C_nougats_truffles_peanutclusters (C N T P : ℕ) :=
  N = 2 * C ∧
  T = C + 6 ∧
  C + N + T + P = 50 ∧
  P = 32

theorem chocolates_problem (C N T P : ℕ) :
  C_nougats_truffles_peanutclusters C N T P → C = 3 :=
by
  intros h
  have hN := h.1
  have hT := h.2.1
  have hSum := h.2.2.1
  have hP := h.2.2.2
  sorry

end NUMINAMATH_GPT_chocolates_problem_l1899_189914


namespace NUMINAMATH_GPT_cylindrical_to_cartesian_l1899_189987

theorem cylindrical_to_cartesian :
  ∀ (r θ z : ℝ), r = 2 → θ = π / 3 → z = 2 → 
  (r * Real.cos θ, r * Real.sin θ, z) = (1, Real.sqrt 3, 2) :=
by
  intros r θ z hr hθ hz
  sorry

end NUMINAMATH_GPT_cylindrical_to_cartesian_l1899_189987


namespace NUMINAMATH_GPT_find_percentage_l1899_189938

def problem_statement (n P : ℕ) := 
  n = (P / 100) * n + 84

theorem find_percentage : ∃ P, problem_statement 100 P ∧ (P = 16) :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l1899_189938


namespace NUMINAMATH_GPT_original_calculation_l1899_189988

theorem original_calculation
  (x : ℝ)
  (h : ((x * 3) + 14) * 2 = 946) :
  ((x / 3) + 14) * 2 = 130 :=
sorry

end NUMINAMATH_GPT_original_calculation_l1899_189988


namespace NUMINAMATH_GPT_ferry_speed_difference_l1899_189980

theorem ferry_speed_difference :
  let V_p := 6
  let Time_P := 3
  let Distance_P := V_p * Time_P
  let Distance_Q := 2 * Distance_P
  let Time_Q := Time_P + 1
  let V_q := Distance_Q / Time_Q
  V_q - V_p = 3 := by
  sorry

end NUMINAMATH_GPT_ferry_speed_difference_l1899_189980


namespace NUMINAMATH_GPT_simplify_expression_l1899_189935

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 5 + 2) + 2 / (Real.sqrt 7 - 2))) = 
  (6 * Real.sqrt 7 + 9 * Real.sqrt 5 + 6) / (118 + 12 * Real.sqrt 35) :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l1899_189935


namespace NUMINAMATH_GPT_brooke_added_balloons_l1899_189940

-- Definitions stemming from the conditions
def initial_balloons_brooke : Nat := 12
def added_balloons_brooke (x : Nat) : Nat := x
def initial_balloons_tracy : Nat := 6
def added_balloons_tracy : Nat := 24
def total_balloons_tracy : Nat := initial_balloons_tracy + added_balloons_tracy
def final_balloons_tracy : Nat := total_balloons_tracy / 2
def total_balloons (x : Nat) : Nat := initial_balloons_brooke + added_balloons_brooke x + final_balloons_tracy

-- Mathematical proof problem
theorem brooke_added_balloons (x : Nat) :
  total_balloons x = 35 → x = 8 := by
  sorry

end NUMINAMATH_GPT_brooke_added_balloons_l1899_189940


namespace NUMINAMATH_GPT_investment_ratio_l1899_189931

theorem investment_ratio (total_profit b_profit : ℝ) (a c b : ℝ) :
  total_profit = 150000 ∧ b_profit = 75000 ∧ a / c = 2 ∧ a + b + c = total_profit →
  a / b = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_investment_ratio_l1899_189931


namespace NUMINAMATH_GPT_annulus_area_l1899_189901

theorem annulus_area (r R x : ℝ) (hR_gt_r : R > r) (h_tangent : r^2 + x^2 = R^2) : 
  π * x^2 = π * (R^2 - r^2) :=
by
  sorry

end NUMINAMATH_GPT_annulus_area_l1899_189901


namespace NUMINAMATH_GPT_largest_of_seven_consecutive_numbers_l1899_189950

theorem largest_of_seven_consecutive_numbers (avg : ℕ) (h : avg = 20) :
  ∃ n : ℕ, n + 6 = 23 := 
by
  sorry

end NUMINAMATH_GPT_largest_of_seven_consecutive_numbers_l1899_189950


namespace NUMINAMATH_GPT_action_figures_per_shelf_l1899_189921

theorem action_figures_per_shelf (total_figures shelves : ℕ) (h1 : total_figures = 27) (h2 : shelves = 3) :
  (total_figures / shelves = 9) :=
by
  sorry

end NUMINAMATH_GPT_action_figures_per_shelf_l1899_189921


namespace NUMINAMATH_GPT_binomial_coefficient_12_10_l1899_189965

theorem binomial_coefficient_12_10 : Nat.choose 12 10 = 66 := by
  sorry  -- The actual proof is omitted as instructed.

end NUMINAMATH_GPT_binomial_coefficient_12_10_l1899_189965


namespace NUMINAMATH_GPT_time_to_fill_tank_l1899_189926

theorem time_to_fill_tank (T : ℝ) :
  (1 / 2 * T) + ((1 / 2 * T) / 4) = 10 → T = 16 :=
by { sorry }

end NUMINAMATH_GPT_time_to_fill_tank_l1899_189926


namespace NUMINAMATH_GPT_avg_speed_ratio_l1899_189915

theorem avg_speed_ratio 
  (dist_tractor : ℝ) (time_tractor : ℝ) 
  (dist_car : ℝ) (time_car : ℝ) 
  (speed_factor : ℝ) :
  dist_tractor = 575 -> 
  time_tractor = 23 ->
  dist_car = 450 ->
  time_car = 5 ->
  speed_factor = 2 ->

  (dist_car / time_car) / (speed_factor * (dist_tractor / time_tractor)) = 9/5 := 
by
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5]
  sorry

end NUMINAMATH_GPT_avg_speed_ratio_l1899_189915


namespace NUMINAMATH_GPT_smaller_interior_angle_of_parallelogram_l1899_189934

theorem smaller_interior_angle_of_parallelogram (x : ℝ) 
  (h1 : ∃ l, l = x + 90 ∧ x + l = 180) :
  x = 45 :=
by
  obtain ⟨l, hl1, hl2⟩ := h1
  simp only [hl1] at hl2
  linarith

end NUMINAMATH_GPT_smaller_interior_angle_of_parallelogram_l1899_189934


namespace NUMINAMATH_GPT_three_pizzas_needed_l1899_189932

noncomputable def masha_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "sausage" ∉ p

noncomputable def vanya_pizza (p : Set String) : Prop :=
  "mushrooms" ∈ p

noncomputable def dasha_pizza (p : Set String) : Prop :=
  "tomatoes" ∉ p

noncomputable def nikita_pizza (p : Set String) : Prop :=
  "tomatoes" ∈ p ∧ "mushrooms" ∉ p

noncomputable def igor_pizza (p : Set String) : Prop :=
  "mushrooms" ∉ p ∧ "sausage" ∈ p

theorem three_pizzas_needed (p1 p2 p3 : Set String) :
  (∃ p1, masha_pizza p1 ∧ vanya_pizza p1 ∧ dasha_pizza p1 ∧ nikita_pizza p1 ∧ igor_pizza p1) →
  (∃ p2, masha_pizza p2 ∧ vanya_pizza p2 ∧ dasha_pizza p2 ∧ nikita_pizza p2 ∧ igor_pizza p2) →
  (∃ p3, masha_pizza p3 ∧ vanya_pizza p3 ∧ dasha_pizza p3 ∧ nikita_pizza p3 ∧ igor_pizza p3) →
  ∀ p, ¬ ((masha_pizza p ∨ dasha_pizza p) ∧ vanya_pizza p ∧ (nikita_pizza p ∨ igor_pizza p)) :=
sorry

end NUMINAMATH_GPT_three_pizzas_needed_l1899_189932


namespace NUMINAMATH_GPT_coordinates_of_S_l1899_189944

variable (P Q R S : (ℝ × ℝ))
variable (hp : P = (3, -2))
variable (hq : Q = (3, 1))
variable (hr : R = (7, 1))
variable (h : Rectangle P Q R S)

def Rectangle (P Q R S : (ℝ × ℝ)) : Prop :=
  let (xP, yP) := P
  let (xQ, yQ) := Q
  let (xR, yR) := R
  let (xS, yS) := S
  (xP = xQ ∧ yR = yS) ∧ (xS = xR ∧ yP = yQ) 

theorem coordinates_of_S : S = (7, -2) := by
  sorry

end NUMINAMATH_GPT_coordinates_of_S_l1899_189944


namespace NUMINAMATH_GPT_factorize_expression_l1899_189973

theorem factorize_expression (x y : ℝ) : x^3 * y - 4 * x * y = x * y * (x - 2) * (x + 2) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l1899_189973


namespace NUMINAMATH_GPT_population_is_24000_l1899_189923

theorem population_is_24000 (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 := sorry

end NUMINAMATH_GPT_population_is_24000_l1899_189923


namespace NUMINAMATH_GPT_converse_negation_contrapositive_l1899_189989

variable {x : ℝ}

def P (x : ℝ) : Prop := x^2 - 3 * x + 2 ≠ 0
def Q (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ 2

theorem converse (h : Q x) : P x := by
  sorry

theorem negation (h : ¬ P x) : ¬ Q x := by
  sorry

theorem contrapositive (h : ¬ Q x) : ¬ P x := by
  sorry

end NUMINAMATH_GPT_converse_negation_contrapositive_l1899_189989


namespace NUMINAMATH_GPT_vector_BC_correct_l1899_189903

-- Define the conditions
def vector_AB : ℝ × ℝ := (-3, 2)
def vector_AC : ℝ × ℝ := (1, -2)

-- Define the problem to be proved
theorem vector_BC_correct :
  let vector_BC := (vector_AC.1 - vector_AB.1, vector_AC.2 - vector_AB.2)
  vector_BC = (4, -4) :=
by
  sorry -- The proof is not required, but the structure indicates where it would go

end NUMINAMATH_GPT_vector_BC_correct_l1899_189903


namespace NUMINAMATH_GPT_find_xyz_l1899_189995

theorem find_xyz (x y z : ℝ) 
  (h1: 3 * x - y + z = 8)
  (h2: x + 3 * y - z = 2) 
  (h3: x - y + 3 * z = 6) :
  x = 1 ∧ y = 3 ∧ z = 8 := by
  sorry

end NUMINAMATH_GPT_find_xyz_l1899_189995


namespace NUMINAMATH_GPT_Elise_savings_l1899_189908

theorem Elise_savings :
  let initial_dollars := 8
  let saved_euros := 11
  let euro_to_dollar := 1.18
  let comic_cost := 2
  let puzzle_pounds := 13
  let pound_to_dollar := 1.38
  let euros_to_dollars := saved_euros * euro_to_dollar
  let total_after_saving := initial_dollars + euros_to_dollars
  let after_comic := total_after_saving - comic_cost
  let pounds_to_dollars := puzzle_pounds * pound_to_dollar
  let final_amount := after_comic - pounds_to_dollars
  final_amount = 1.04 :=
by
  sorry

end NUMINAMATH_GPT_Elise_savings_l1899_189908


namespace NUMINAMATH_GPT_quadratic_no_real_roots_min_k_l1899_189902

theorem quadratic_no_real_roots_min_k :
  ∀ (k : ℤ), 
    (∀ x : ℝ, 3*x*(k*x-5) - 2*x^2 + 8 ≠ 0) ↔ 
    (k ≥ 3) := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_min_k_l1899_189902


namespace NUMINAMATH_GPT_goods_train_length_l1899_189978

theorem goods_train_length (speed_kmph : ℕ) (platform_length_m : ℕ) (time_s : ℕ) 
    (h_speed : speed_kmph = 72) (h_platform : platform_length_m = 250) (h_time : time_s = 24) : 
    ∃ train_length_m : ℕ, train_length_m = 230 := 
by 
  sorry

end NUMINAMATH_GPT_goods_train_length_l1899_189978


namespace NUMINAMATH_GPT_at_least_half_sectors_occupied_l1899_189972

theorem at_least_half_sectors_occupied (n : ℕ) (chips : Finset (Fin n.succ)) 
(h_chips_count: chips.card = n + 1) :
  ∃ (steps : ℕ), ∀ (t : ℕ), t ≥ steps → (∃ sector_occupied : Finset (Fin n), sector_occupied.card ≥ n / 2) :=
sorry

end NUMINAMATH_GPT_at_least_half_sectors_occupied_l1899_189972


namespace NUMINAMATH_GPT_construct_triangle_num_of_solutions_l1899_189969

theorem construct_triangle_num_of_solutions
  (r : ℝ) -- Circumradius
  (beta_gamma_diff : ℝ) -- Angle difference \beta - \gamma
  (KA1 : ℝ) -- Segment K A_1
  (KA1_lt_r : KA1 < r) -- Segment K A1 should be less than the circumradius
  (delta : ℝ := beta_gamma_diff) : 1 ≤ num_solutions ∧ num_solutions ≤ 2 :=
sorry

end NUMINAMATH_GPT_construct_triangle_num_of_solutions_l1899_189969


namespace NUMINAMATH_GPT_original_expenditure_mess_l1899_189966

theorem original_expenditure_mess : 
  ∀ (x : ℝ), 
  35 * x + 42 = 42 * (x - 1) + 35 * x → 
  35 * 12 = 420 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_original_expenditure_mess_l1899_189966


namespace NUMINAMATH_GPT_max_value_of_f_l1899_189996

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, ∃ k : ℤ, f x = 3 ∧ x = k * Real.pi :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1899_189996


namespace NUMINAMATH_GPT_probability_meeting_twin_probability_twin_in_family_expected_twin_pairs_l1899_189906

-- Problem 1
theorem probability_meeting_twin (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  (2 * p) / (p + 1) = (2 * p) / (p + 1) :=
by
  sorry

-- Problem 2
theorem probability_twin_in_family (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  (2 * p) / (2 * p + (1 - p) ^ 2) = (2 * p) / (2 * p + (1 - p) ^ 2) :=
by
  sorry

-- Problem 3
theorem expected_twin_pairs (N : ℕ) (p : ℝ) (h₀ : 0 < p) (h₁ : p < 1) :
  N * p / (p + 1) = N * p / (p + 1) :=
by
  sorry

end NUMINAMATH_GPT_probability_meeting_twin_probability_twin_in_family_expected_twin_pairs_l1899_189906


namespace NUMINAMATH_GPT_num_people_got_on_bus_l1899_189994

-- Definitions based on the conditions
def initialNum : ℕ := 4
def currentNum : ℕ := 17
def peopleGotOn (initial : ℕ) (current : ℕ) : ℕ := current - initial

-- Theorem statement
theorem num_people_got_on_bus : peopleGotOn initialNum currentNum = 13 := 
by {
  sorry -- Placeholder for the proof
}

end NUMINAMATH_GPT_num_people_got_on_bus_l1899_189994


namespace NUMINAMATH_GPT_initial_amount_solution_l1899_189943

noncomputable def initialAmount (P : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then P else (1 + 1/8) * initialAmount P (n - 1)

theorem initial_amount_solution (P : ℝ) (h₁ : initialAmount P 2 = 2025) : P = 1600 :=
  sorry

end NUMINAMATH_GPT_initial_amount_solution_l1899_189943


namespace NUMINAMATH_GPT_Cathy_total_money_l1899_189913

theorem Cathy_total_money 
  (Cathy_wallet : ℕ) 
  (dad_sends : ℕ) 
  (mom_sends : ℕ) 
  (h1 : Cathy_wallet = 12) 
  (h2 : dad_sends = 25) 
  (h3 : mom_sends = 2 * dad_sends) :
  (Cathy_wallet + dad_sends + mom_sends) = 87 :=
by
  sorry

end NUMINAMATH_GPT_Cathy_total_money_l1899_189913


namespace NUMINAMATH_GPT_acute_angle_implies_x_range_l1899_189925

open Real

def is_acute (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 > 0

theorem acute_angle_implies_x_range (x : ℝ) :
  is_acute (1, 2) (x, 4) → x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi (2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_implies_x_range_l1899_189925


namespace NUMINAMATH_GPT_speed_goods_train_l1899_189977

def length_train : ℝ := 50
def length_platform : ℝ := 250
def time_crossing : ℝ := 15

/-- The speed of the goods train in km/hr given the length of the train, the length of the platform, and the time to cross the platform. -/
theorem speed_goods_train :
  (length_train + length_platform) / time_crossing * 3.6 = 72 :=
by
  sorry

end NUMINAMATH_GPT_speed_goods_train_l1899_189977


namespace NUMINAMATH_GPT_num_integers_satisfying_abs_leq_bound_l1899_189953

theorem num_integers_satisfying_abs_leq_bound : ∃ n : ℕ, n = 19 ∧ ∀ x : ℤ, |x| ≤ 3 * Real.sqrt 10 → (x ≥ -9 ∧ x ≤ 9) := by
  sorry

end NUMINAMATH_GPT_num_integers_satisfying_abs_leq_bound_l1899_189953


namespace NUMINAMATH_GPT_value_of_a_minus_b_l1899_189947

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a - b = 5) (h2 : a - 2 * b = 4) : a - b = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l1899_189947


namespace NUMINAMATH_GPT_third_player_game_count_l1899_189998

theorem third_player_game_count (fp_games : ℕ) (sp_games : ℕ) (tp_games : ℕ) (total_games : ℕ) 
  (h1 : fp_games = 10) (h2 : sp_games = 21) (h3 : total_games = sp_games) 
  (h4 : total_games = fp_games + tp_games + 1): tp_games = 11 := 
  sorry

end NUMINAMATH_GPT_third_player_game_count_l1899_189998


namespace NUMINAMATH_GPT_percentage_reduction_of_faculty_l1899_189964

noncomputable def percentage_reduction (original reduced : ℝ) : ℝ :=
  ((original - reduced) / original) * 100

theorem percentage_reduction_of_faculty :
  percentage_reduction 226.74 195 = 13.99 :=
by sorry

end NUMINAMATH_GPT_percentage_reduction_of_faculty_l1899_189964


namespace NUMINAMATH_GPT_distinct_exponentiation_values_l1899_189984

theorem distinct_exponentiation_values : 
  ∃ (standard other1 other2 other3 : ℕ), 
    standard ≠ other1 ∧ 
    standard ≠ other2 ∧ 
    standard ≠ other3 ∧ 
    other1 ≠ other2 ∧ 
    other1 ≠ other3 ∧ 
    other2 ≠ other3 := 
sorry

end NUMINAMATH_GPT_distinct_exponentiation_values_l1899_189984


namespace NUMINAMATH_GPT_units_digit_7_pow_5_l1899_189968

theorem units_digit_7_pow_5 : (7 ^ 5) % 10 = 7 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_5_l1899_189968


namespace NUMINAMATH_GPT_Sara_lunch_bill_l1899_189919

theorem Sara_lunch_bill :
  let hotdog := 5.36
  let salad := 5.10
  let drink := 2.50
  let side_item := 3.75
  hotdog + salad + drink + side_item = 16.71 :=
by
  sorry

end NUMINAMATH_GPT_Sara_lunch_bill_l1899_189919


namespace NUMINAMATH_GPT_sum_of_three_consecutive_natural_numbers_not_prime_l1899_189959

theorem sum_of_three_consecutive_natural_numbers_not_prime (n : ℕ) : 
  ¬ Prime (n + (n+1) + (n+2)) := by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_natural_numbers_not_prime_l1899_189959


namespace NUMINAMATH_GPT_A_half_B_l1899_189905

-- Define the arithmetic series sum function
def series_sum (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define A and B according to the problem conditions
def A : ℕ := (Finset.range 2022).sum (λ m => series_sum (m + 1))

def B : ℕ := (Finset.range 2022).sum (λ m => (m + 1) * (m + 2))

-- The proof statement
theorem A_half_B : A = B / 2 :=
by
  sorry

end NUMINAMATH_GPT_A_half_B_l1899_189905


namespace NUMINAMATH_GPT_remaining_money_l1899_189971

-- Definitions
def cost_per_app : ℕ := 4
def num_apps : ℕ := 15
def total_money : ℕ := 66

-- Theorem
theorem remaining_money : total_money - (num_apps * cost_per_app) = 6 := by
  sorry

end NUMINAMATH_GPT_remaining_money_l1899_189971


namespace NUMINAMATH_GPT_point_not_in_third_quadrant_l1899_189960

theorem point_not_in_third_quadrant (A : ℝ × ℝ) (h : A.snd = -A.fst + 8) : ¬ (A.fst < 0 ∧ A.snd < 0) :=
sorry

end NUMINAMATH_GPT_point_not_in_third_quadrant_l1899_189960


namespace NUMINAMATH_GPT_polygon_sides_l1899_189957

theorem polygon_sides (R : ℝ) (n : ℕ) (h : R ≠ 0)
  (h_area : (1 / 2) * n * R^2 * Real.sin (360 / n * (Real.pi / 180)) = 4 * R^2) :
  n = 8 := 
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1899_189957


namespace NUMINAMATH_GPT_sam_more_than_sarah_l1899_189900

-- Defining the conditions
def street_width : ℤ := 25
def block_length : ℤ := 450
def block_width : ℤ := 350
def alleyway : ℤ := 25

-- Defining the distances run by Sarah and Sam
def sarah_long_side : ℤ := block_length + alleyway
def sarah_short_side : ℤ := block_width
def sam_long_side : ℤ := block_length + 2 * street_width
def sam_short_side : ℤ := block_width + 2 * street_width

-- Defining the total distance run by Sarah and Sam in one lap
def sarah_total_distance : ℤ := 2 * sarah_long_side + 2 * sarah_short_side
def sam_total_distance : ℤ := 2 * sam_long_side + 2 * sam_short_side

-- Proving the difference between Sam's and Sarah's running distances
theorem sam_more_than_sarah : sam_total_distance - sarah_total_distance = 150 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_sam_more_than_sarah_l1899_189900


namespace NUMINAMATH_GPT_train_speed_l1899_189922

-- Defining the lengths and time
def length_train : ℕ := 100
def length_bridge : ℕ := 300
def time_crossing : ℕ := 15

-- Defining the total distance
def total_distance : ℕ := length_train + length_bridge

-- Proving the speed of the train
theorem train_speed : (total_distance / time_crossing : ℚ) = 26.67 := by
  sorry

end NUMINAMATH_GPT_train_speed_l1899_189922


namespace NUMINAMATH_GPT_parallel_lines_slope_l1899_189963

theorem parallel_lines_slope (m : ℚ) (h : (x - y = 1) → (m + 3) * x + m * y - 8 = 0) :
  m = -3 / 2 :=
sorry

end NUMINAMATH_GPT_parallel_lines_slope_l1899_189963


namespace NUMINAMATH_GPT_blackjack_payment_l1899_189985

def casino_payout (b: ℤ) (r: ℤ): ℤ := b + r
def blackjack_payout (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ): ℤ :=
  (ratio_numerator * bet) / ratio_denominator

theorem blackjack_payment (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ) (payout: ℤ):
  ratio_numerator = 3 → 
  ratio_denominator = 2 → 
  bet = 40 →
  payout = blackjack_payout bet ratio_numerator ratio_denominator → 
  casino_payout bet payout = 100 :=
by
  sorry

end NUMINAMATH_GPT_blackjack_payment_l1899_189985


namespace NUMINAMATH_GPT_divisor_problem_l1899_189912

theorem divisor_problem :
  ∃ D : ℕ, 12401 = D * 76 + 13 ∧ D = 163 := 
by
  sorry

end NUMINAMATH_GPT_divisor_problem_l1899_189912


namespace NUMINAMATH_GPT_find_total_coins_l1899_189990

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end NUMINAMATH_GPT_find_total_coins_l1899_189990
