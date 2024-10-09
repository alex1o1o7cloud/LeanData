import Mathlib

namespace polynomial_remainder_l1273_127344

noncomputable def f (x : ℝ) : ℝ := x^4 + 2 * x^2 - 3
noncomputable def g (x : ℝ) : ℝ := x^2 + x - 2
noncomputable def r (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 3

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = g x * q x + r x :=
sorry

end polynomial_remainder_l1273_127344


namespace Cherie_boxes_l1273_127391

theorem Cherie_boxes (x : ℕ) :
  (2 * 8 + x * (8 + 9) = 33) → x = 1 :=
by
  intros h
  have h_eq : 16 + 17 * x = 33 := by simp [mul_add, mul_comm, h]
  linarith

end Cherie_boxes_l1273_127391


namespace parabola_directrix_eq_l1273_127376

noncomputable def equation_of_directrix (p : ℝ) : Prop :=
  (p > 0) ∧ (∀ (x y : ℝ), (x ≠ -5 / 4) → ¬ (y ^ 2 = 2 * p * x))

theorem parabola_directrix_eq (A_x A_y : ℝ) (hA : A_x = 2 ∧ A_y = 1)
  (h_perpendicular_bisector_fo : ∃ (f_x f_y : ℝ), f_x = 5 / 4 ∧ f_y = 0) :
  equation_of_directrix (5 / 2) :=
by {
  sorry
}

end parabola_directrix_eq_l1273_127376


namespace geometric_sequence_12th_term_l1273_127326

theorem geometric_sequence_12th_term 
  (a_4 a_8 : ℕ) (h4 : a_4 = 2) (h8 : a_8 = 162) :
  ∃ a_12 : ℕ, a_12 = 13122 :=
by
  sorry

end geometric_sequence_12th_term_l1273_127326


namespace smallest_number_l1273_127345

theorem smallest_number (x : ℕ) (h1 : (x + 7) % 8 = 0) (h2 : (x + 7) % 11 = 0) (h3 : (x + 7) % 24 = 0) : x = 257 :=
sorry

end smallest_number_l1273_127345


namespace relation_among_a_b_c_l1273_127337

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 3
noncomputable def c : ℝ := Real.sqrt 6 - Real.sqrt 2

theorem relation_among_a_b_c : a > c ∧ c > b :=
by {
  sorry
}

end relation_among_a_b_c_l1273_127337


namespace smallest_positive_period_minimum_value_of_f_l1273_127355

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x * sin (x + π / 3) - sqrt 3 * sin x ^ 2 + sin x * cos x

theorem smallest_positive_period :
  ∀ x, f (x + π) = f x :=
sorry

theorem minimum_value_of_f :
  ∀ k : ℤ, f (k * π - 5 * π / 12) = -2 :=
sorry

end smallest_positive_period_minimum_value_of_f_l1273_127355


namespace no_integers_satisfy_eq_l1273_127336

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^3 = 4 * n + 2) := 
  sorry

end no_integers_satisfy_eq_l1273_127336


namespace pizza_dough_milk_needed_l1273_127318

variable (milk_per_300 : ℕ) (flour_per_batch : ℕ) (total_flour : ℕ)

-- Definitions based on problem conditions
def milk_per_batch := milk_per_300
def batch_size := flour_per_batch
def used_flour := total_flour

-- The target proof statement
theorem pizza_dough_milk_needed (h1 : milk_per_batch = 60) (h2 : batch_size = 300) (h3 : used_flour = 1500) : 
  (used_flour / batch_size) * milk_per_batch = 300 :=
by
  rw [h1, h2, h3]
  sorry -- proof steps

end pizza_dough_milk_needed_l1273_127318


namespace S_nine_l1273_127386

noncomputable def S : ℕ → ℚ
| 3 => 8
| 6 => 10
| _ => 0  -- Placeholder for other values, as we're interested in these specific ones

theorem S_nine (S_3_eq : S 3 = 8) (S_6_eq : S 6 = 10) : S 9 = 21 / 2 :=
by
  -- Construct the proof here
  sorry

end S_nine_l1273_127386


namespace midpoint_of_segment_l1273_127312

def z1 : ℂ := 2 + 4 * Complex.I  -- Define the first endpoint
def z2 : ℂ := -6 + 10 * Complex.I  -- Define the second endpoint

theorem midpoint_of_segment :
  (z1 + z2) / 2 = -2 + 7 * Complex.I := by
  sorry

end midpoint_of_segment_l1273_127312


namespace number_line_y_l1273_127300

theorem number_line_y (step_length : ℕ) (steps_total : ℕ) (total_distance : ℕ) (y_step : ℕ) (y : ℕ) 
    (H1 : steps_total = 6) 
    (H2 : total_distance = 24) 
    (H3 : y_step = 4)
    (H4 : step_length = total_distance / steps_total) 
    (H5 : y = step_length * y_step) : 
  y = 16 := 
  by 
    sorry

end number_line_y_l1273_127300


namespace negation_of_proposition_l1273_127380

theorem negation_of_proposition : 
  (¬ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0)) ↔ (∃ x : ℝ, x^2 + 2*x + 5 = 0) :=
by sorry

end negation_of_proposition_l1273_127380


namespace min_value_of_expression_l1273_127362

theorem min_value_of_expression (x y : ℝ) (h : x^2 + x * y + y^2 = 3) : x^2 - x * y + y^2 ≥ 1 :=
by 
sorry

end min_value_of_expression_l1273_127362


namespace no_fixed_points_implies_no_double_fixed_points_l1273_127320

theorem no_fixed_points_implies_no_double_fixed_points (f : ℝ → ℝ) (hf : ∀ x, f x ≠ x) :
  ∀ x, f (f x) ≠ x :=
sorry

end no_fixed_points_implies_no_double_fixed_points_l1273_127320


namespace family_ages_l1273_127378

theorem family_ages:
  (∀ (Peter Harriet Jane Emily father: ℕ),
  ((Peter + 12 = 2 * (Harriet + 12)) ∧
   (Jane = Emily + 10) ∧
   (Peter = 60 / 3) ∧
   (Peter = Jane + 5) ∧
   (Aunt_Lucy = 52) ∧
   (Aunt_Lucy = 4 + Peter_Jane_mother) ∧
   (father - 20 = Aunt_Lucy)) →
  (Harriet = 4) ∧ (Peter = 20) ∧ (Jane = 15) ∧ (Emily = 5) ∧ (father = 72)) :=
sorry

end family_ages_l1273_127378


namespace angle_between_hands_at_3_15_l1273_127332

-- Definitions based on conditions
def minuteHandAngleAt_3_15 : ℝ := 90 -- The position of the minute hand at 3:15 is 90 degrees.

def hourHandSpeed : ℝ := 0.5 -- The hour hand moves at 0.5 degrees per minute.

def hourHandAngleAt_3_15 : ℝ := 3 * 30 + 15 * hourHandSpeed
-- The hour hand starts at 3 o'clock (90 degrees) and moves 0.5 degrees per minute.

-- Statement to prove
theorem angle_between_hands_at_3_15 : abs (minuteHandAngleAt_3_15 - hourHandAngleAt_3_15) = 82.5 :=
by
  sorry

end angle_between_hands_at_3_15_l1273_127332


namespace intersection_of_P_and_Q_l1273_127368

def P : Set ℝ := {x | Real.log x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}
def R : Set ℝ := {x | 1 < x ∧ x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = R := by
  sorry

end intersection_of_P_and_Q_l1273_127368


namespace distance_Bella_Galya_l1273_127396

theorem distance_Bella_Galya 
    (AB BV VG GD : ℝ)
    (DB : AB + 3 * BV + 2 * VG + GD = 700)
    (DV : AB + 2 * BV + 2 * VG + GD = 600)
    (DG : AB + 2 * BV + 3 * VG + GD = 650) :
    BV + VG = 150 := 
by
  -- Proof goes here
  sorry

end distance_Bella_Galya_l1273_127396


namespace annual_profits_l1273_127339

-- Define the profits of each quarter
def P1 : ℕ := 1500
def P2 : ℕ := 1500
def P3 : ℕ := 3000
def P4 : ℕ := 2000

-- State the annual profit theorem
theorem annual_profits : P1 + P2 + P3 + P4 = 8000 := by
  sorry

end annual_profits_l1273_127339


namespace line_through_intersection_points_of_circles_l1273_127393

theorem line_through_intersection_points_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (x^2 + y^2 + 2*x - 13 = 0) →
    (x - 2*y + 6 = 0) :=
by
  intro x y h
  -- Condition of circle 1
  have circle1 : x^2 + y^2 + 4*x - 4*y - 1 = 0 := h.left
  -- Condition of circle 2
  have circle2 : x^2 + y^2 + 2*x - 13 = 0 := h.right
  sorry

end line_through_intersection_points_of_circles_l1273_127393


namespace midpoint_product_l1273_127301

theorem midpoint_product (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 4) (hy1 : y1 = -3) (hx2 : x2 = -8) (hy2 : y2 = 7) :
  let midx := (x1 + x2) / 2
  let midy := (y1 + y2) / 2
  midx * midy = -4 :=
by
  sorry

end midpoint_product_l1273_127301


namespace triangle_side_lengths_l1273_127319

theorem triangle_side_lengths (x : ℤ) (h1 : x > 3) (h2 : x < 13) :
  (∃! (x : ℤ), (x > 3 ∧ x < 13) ∧ (4 ≤ x ∧ x ≤ 12)) :=
sorry

end triangle_side_lengths_l1273_127319


namespace printer_cost_l1273_127311

theorem printer_cost (num_keyboards : ℕ) (num_printers : ℕ) (total_cost : ℕ) (keyboard_cost : ℕ) (printer_cost : ℕ) :
  num_keyboards = 15 →
  num_printers = 25 →
  total_cost = 2050 →
  keyboard_cost = 20 →
  (total_cost - (num_keyboards * keyboard_cost)) / num_printers = printer_cost →
  printer_cost = 70 :=
by
  sorry

end printer_cost_l1273_127311


namespace matrix_sum_correct_l1273_127335

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![4, -3],
  ![2, 5]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-6, 8],
  ![-3, 7]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, 5],
  ![-1, 12]
]

theorem matrix_sum_correct : A + B = C := by
  sorry

end matrix_sum_correct_l1273_127335


namespace savanna_total_animals_l1273_127313

def num_lions_safari := 100
def num_snakes_safari := num_lions_safari / 2
def num_giraffes_safari := num_snakes_safari - 10
def num_elephants_safari := num_lions_safari / 4

def num_lions_savanna := num_lions_safari * 2
def num_snakes_savanna := num_snakes_safari * 3
def num_giraffes_savanna := num_giraffes_safari + 20
def num_elephants_savanna := num_elephants_safari * 5
def num_zebras_savanna := (num_lions_savanna + num_snakes_savanna) / 2

def total_animals_savanna := 
  num_lions_savanna 
  + num_snakes_savanna 
  + num_giraffes_savanna 
  + num_elephants_savanna 
  + num_zebras_savanna

open Nat
theorem savanna_total_animals : total_animals_savanna = 710 := by
  sorry

end savanna_total_animals_l1273_127313


namespace general_formula_for_sequence_a_l1273_127371

noncomputable def S (n : ℕ) : ℕ := 3^n + 1

def a (n : ℕ) : ℕ :=
if n = 1 then 4 else 2 * 3^(n-1)

theorem general_formula_for_sequence_a (n : ℕ) :
  a n = if n = 1 then 4 else 2 * 3^(n-1) :=
by {
  sorry
}

end general_formula_for_sequence_a_l1273_127371


namespace cubs_more_home_runs_l1273_127333

noncomputable def cubs_home_runs := 2 + 1 + 2
noncomputable def cardinals_home_runs := 1 + 1

theorem cubs_more_home_runs :
  cubs_home_runs - cardinals_home_runs = 3 :=
by
  -- Proof would go here, but we are using sorry to skip it
  sorry

end cubs_more_home_runs_l1273_127333


namespace shipping_cost_l1273_127395

def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def cost_per_crate : ℝ := 1.5

/-- Lizzy's total shipping cost for 540 pounds of fish packed in 30-pound crates at $1.5 per crate is $27. -/
theorem shipping_cost : (total_weight / weight_per_crate) * cost_per_crate = 27 := by
  sorry

end shipping_cost_l1273_127395


namespace min_value_of_expression_l1273_127374

theorem min_value_of_expression (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 5) : 
  (9 / x + 16 / y + 25 / z) ≥ 28.8 :=
by sorry

end min_value_of_expression_l1273_127374


namespace largest_T_l1273_127334

theorem largest_T (T : ℝ) (a b c d e : ℝ) 
  (h1: a ≥ 0) (h2: b ≥ 0) (h3: c ≥ 0) (h4: d ≥ 0) (h5: e ≥ 0)
  (h_sum : a + b = c + d + e)
  (h_T : T ≤ (Real.sqrt 30) / (30 + 12 * Real.sqrt 6)) : 
  Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2) ≥ T * (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d + Real.sqrt e)^2 :=
sorry

end largest_T_l1273_127334


namespace product_of_first_three_terms_of_arithmetic_sequence_l1273_127372

theorem product_of_first_three_terms_of_arithmetic_sequence {a d : ℕ} (ha : a + 6 * d = 20) (hd : d = 2) : a * (a + d) * (a + 2 * d) = 960 := by
  sorry

end product_of_first_three_terms_of_arithmetic_sequence_l1273_127372


namespace range_of_m_l1273_127341

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def B (m : ℝ) := { x : ℝ | x^2 - (2 * m + 1) * x + 2 * m < 0 }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → (-1 / 2 ≤ m ∧ m ≤ 1) :=
by
  sorry

end range_of_m_l1273_127341


namespace y_coordinate_sum_of_circle_on_y_axis_l1273_127346

-- Define the properties of the circle
def center := (-3, 1)
def radius := 8

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y - 1) ^ 2 = 64

-- Define the Lean theorem statement
theorem y_coordinate_sum_of_circle_on_y_axis 
  (h₁ : center = (-3, 1)) 
  (h₂ : radius = 8) 
  (h₃ : ∀ y : ℝ, circle_eq 0 y → (∃ y1 y2 : ℝ, y = y1 ∨ y = y2) ) : 
  ∃ y1 y2 : ℝ, (y1 + y2 = 2) ∧ (circle_eq 0 y1) ∧ (circle_eq 0 y2) := 
by 
  sorry

end y_coordinate_sum_of_circle_on_y_axis_l1273_127346


namespace no_negative_roots_l1273_127329

theorem no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 :=
by sorry

end no_negative_roots_l1273_127329


namespace check_not_coverable_boards_l1273_127317

def is_coverable_by_dominoes (m n : ℕ) : Prop :=
  (m * n) % 2 = 0

theorem check_not_coverable_boards:
  (¬is_coverable_by_dominoes 5 5) ∧ (¬is_coverable_by_dominoes 3 7) :=
by
  -- Proof steps are omitted.
  sorry

end check_not_coverable_boards_l1273_127317


namespace range_of_m_for_inversely_proportional_function_l1273_127363

theorem range_of_m_for_inversely_proportional_function 
  (m : ℝ)
  (h : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > x₁ → (m - 1) / x₂ < (m - 1) / x₁) : 
  m > 1 :=
sorry

end range_of_m_for_inversely_proportional_function_l1273_127363


namespace weight_lifting_ratio_l1273_127304

theorem weight_lifting_ratio :
  ∀ (F S : ℕ), F + S = 600 ∧ F = 300 ∧ 2 * F = S + 300 → F / S = 1 :=
by
  intro F S
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end weight_lifting_ratio_l1273_127304


namespace prove_f_of_pi_div_4_eq_0_l1273_127356

noncomputable
def tan_function (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x)

theorem prove_f_of_pi_div_4_eq_0 
  (ω : ℝ) (hω : ω > 0)
  (h_period : ∀ x : ℝ, tan_function ω (x + π / (4 * ω)) = tan_function ω x) :
  tan_function ω (π / 4) = 0 :=
by
  -- This is where the proof would go.
  sorry

end prove_f_of_pi_div_4_eq_0_l1273_127356


namespace area_of_triangle_l1273_127370

-- Define the vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- The goal is to prove the area of the triangle
theorem area_of_triangle (a b : ℝ × ℝ) : 
  a = (4, -1) → b = (-3, 3) → (|4 * 3 - (-1) * (-3)| / 2) = 9 / 2  :=
by
  intros
  sorry

end area_of_triangle_l1273_127370


namespace pipes_fill_cistern_time_l1273_127328

noncomputable def pipe_fill_time : ℝ :=
  let rateA := 1 / 80
  let rateC := 1 / 60
  let combined_rateAB := 1 / 20
  let rateB := combined_rateAB - rateA
  let combined_rateABC := rateA + rateB - rateC
  1 / combined_rateABC

theorem pipes_fill_cistern_time :
  pipe_fill_time = 30 := by
  sorry

end pipes_fill_cistern_time_l1273_127328


namespace max_a_squared_b_squared_c_squared_l1273_127375

theorem max_a_squared_b_squared_c_squared (a b c : ℤ)
  (h1 : a + b + c = 3)
  (h2 : a^3 + b^3 + c^3 = 3) :
  a^2 + b^2 + c^2 ≤ 57 :=
sorry

end max_a_squared_b_squared_c_squared_l1273_127375


namespace double_counted_page_number_l1273_127379

theorem double_counted_page_number (n x : ℕ) 
  (h1: 1 ≤ x ∧ x ≤ n)
  (h2: (n * (n + 1) / 2) + x = 1997) : 
  x = 44 := 
by
  sorry

end double_counted_page_number_l1273_127379


namespace elementary_schools_in_Lansing_l1273_127387

theorem elementary_schools_in_Lansing (total_students : ℕ) (students_per_school : ℕ) (h1 : total_students = 6175) (h2 : students_per_school = 247) : total_students / students_per_school = 25 := 
by sorry

end elementary_schools_in_Lansing_l1273_127387


namespace find_a9_l1273_127366

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {d a₁ : ℤ}

-- Conditions
def arithmetic_sequence := ∀ n : ℕ, a_n n = a₁ + n * d
def sum_first_n_terms := ∀ n : ℕ, S n = (n * (2 * a₁ + (n - 1) * d)) / 2

-- Specific Conditions for the problem
axiom condition1 : S 8 = 4 * a₁
axiom condition2 : a_n 6 = -2 -- Note that a_n is 0-indexed here.

theorem find_a9 : a_n 8 = 2 :=
by
  sorry

end find_a9_l1273_127366


namespace sum_of_three_numbers_l1273_127394

theorem sum_of_three_numbers
  (a b c : ℕ) (h_prime : Prime c)
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + a * c = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_l1273_127394


namespace total_cube_volume_l1273_127397

theorem total_cube_volume 
  (carl_cubes : ℕ)
  (carl_cube_side : ℕ)
  (kate_cubes : ℕ)
  (kate_cube_side : ℕ)
  (hcarl : carl_cubes = 4)
  (hcarl_side : carl_cube_side = 3)
  (hkate : kate_cubes = 6)
  (hkate_side : kate_cube_side = 4) :
  (carl_cubes * carl_cube_side ^ 3) + (kate_cubes * kate_cube_side ^ 3) = 492 :=
by
  sorry

end total_cube_volume_l1273_127397


namespace compound_interest_principal_amount_l1273_127373

theorem compound_interest_principal_amount :
  ∀ (r : ℝ) (n : ℕ) (t : ℕ) (CI : ℝ) (P : ℝ),
    r = 0.04 ∧ n = 1 ∧ t = 2 ∧ CI = 612 →
    (CI = P * (1 + r / n) ^ (n * t) - P) →
    P = 7500 :=
by
  intros r n t CI P h_conditions h_CI
  -- Proof not needed
  sorry

end compound_interest_principal_amount_l1273_127373


namespace gyeongyeon_total_path_l1273_127351

theorem gyeongyeon_total_path (D : ℝ) :
  (D / 4 + 250 = D / 2 - 300) -> D = 2200 :=
by
  intro h
  -- We would now proceed to show that D must equal 2200
  sorry

end gyeongyeon_total_path_l1273_127351


namespace relationship_x_x2_negx_l1273_127324

theorem relationship_x_x2_negx (x : ℝ) (h : x^2 + x < 0) : x < x^2 ∧ x^2 < -x :=
by
  sorry

end relationship_x_x2_negx_l1273_127324


namespace total_legs_in_farm_l1273_127321

theorem total_legs_in_farm (total_animals : ℕ) (total_cows : ℕ) (cow_legs : ℕ) (duck_legs : ℕ) 
  (h_total_animals : total_animals = 15) (h_total_cows : total_cows = 6) 
  (h_cow_legs : cow_legs = 4) (h_duck_legs : duck_legs = 2) :
  total_cows * cow_legs + (total_animals - total_cows) * duck_legs = 42 :=
by
  sorry

end total_legs_in_farm_l1273_127321


namespace min_value_proof_l1273_127348

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  2 / a + 2 / b + 2 / c

theorem min_value_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_abc : a + b + c = 9) : 
  minimum_value a b c ≥ 2 := 
by 
  sorry

end min_value_proof_l1273_127348


namespace find_abc_l1273_127315

-- Given conditions: a, b, c are positive real numbers and satisfy the given equations.
variables (a b c : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)
variable (h1 : a * (b + c) = 152)
variable (h2 : b * (c + a) = 162)
variable (h3 : c * (a + b) = 170)

theorem find_abc : a * b * c = 720 := 
  sorry

end find_abc_l1273_127315


namespace find_k_value_l1273_127349

theorem find_k_value
  (k : ℤ)
  (h : 3 * 2^2001 - 3 * 2^2000 - 2^1999 + 2^1998 = k * 2^1998) : k = 11 :=
by
  sorry

end find_k_value_l1273_127349


namespace handshake_max_participants_l1273_127377

theorem handshake_max_participants (N : ℕ) (hN : 5 < N) (hNotAllShaken: ∃ p1 p2 : ℕ, p1 ≠ p2 ∧ p1 < N ∧ p2 < N ∧ (∀ i : ℕ, i < N → i ≠ p1 → i ≠ p2 → ∃ j : ℕ, j < N ∧ j ≠ i ∧ j ≠ p1 ∧ j ≠ p2)) :
∃ k, k = N - 2 :=
by
  sorry

end handshake_max_participants_l1273_127377


namespace find_b_when_a_is_1600_l1273_127390

variable (a b : ℝ)

def inversely_vary (a b : ℝ) : Prop := a * b = 400

theorem find_b_when_a_is_1600 
  (h1 : inversely_vary 800 0.5)
  (h2 : inversely_vary a b)
  (h3 : a = 1600) :
  b = 0.25 := by
  sorry

end find_b_when_a_is_1600_l1273_127390


namespace midpoint_coordinates_l1273_127331

theorem midpoint_coordinates :
  let A := (7, 8)
  let B := (1, 2)
  let midpoint (p1 p2 : ℕ × ℕ) : ℕ × ℕ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint A B = (4, 5) :=
by
  sorry

end midpoint_coordinates_l1273_127331


namespace common_non_integer_root_eq_l1273_127381

theorem common_non_integer_root_eq (p1 p2 q1 q2 : ℤ) 
  (x : ℝ) (hx1 : x^2 + p1 * x + q1 = 0) (hx2 : x^2 + p2 * x + q2 = 0) 
  (hnint : ¬ ∃ (n : ℤ), x = n) : p1 = p2 ∧ q1 = q2 :=
sorry

end common_non_integer_root_eq_l1273_127381


namespace smallest_number_divisible_by_618_3648_60_inc_l1273_127357

theorem smallest_number_divisible_by_618_3648_60_inc :
  ∃ N : ℕ, (N + 1) % 618 = 0 ∧ (N + 1) % 3648 = 0 ∧ (N + 1) % 60 = 0 ∧ N = 1038239 :=
by
  sorry

end smallest_number_divisible_by_618_3648_60_inc_l1273_127357


namespace pow_product_l1273_127305

theorem pow_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := 
by {
  sorry
}

end pow_product_l1273_127305


namespace mean_of_two_numbers_l1273_127347

theorem mean_of_two_numbers (a b : ℝ) (mean_twelve : ℝ) (mean_fourteen : ℝ) 
  (h1 : mean_twelve = 60) 
  (h2 : mean_fourteen = 75) 
  (sum_twelve : 12 * mean_twelve = 720) 
  (sum_fourteen : 14 * mean_fourteen = 1050) 
  : (a + b) / 2 = 165 :=
by
  sorry

end mean_of_two_numbers_l1273_127347


namespace g_is_even_l1273_127358

noncomputable def g (x : ℝ) : ℝ := 5^(x^2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g x = g (-x) :=
by
  sorry

end g_is_even_l1273_127358


namespace two_planes_divide_at_most_4_parts_l1273_127323

-- Definitions related to the conditions
def Plane := ℝ × ℝ × ℝ → Prop -- Representing a plane in ℝ³ by an equation

-- Axiom: Two given planes
axiom plane1 : Plane
axiom plane2 : Plane

-- Conditions about their relationship
def are_parallel (p1 p2 : Plane) : Prop := 
  ∀ x y z, p1 (x, y, z) → p2 (x, y, z)

def intersect (p1 p2 : Plane) : Prop :=
  ∃ x y z, p1 (x, y, z) ∧ p2 (x, y, z)

-- Main theorem to state
theorem two_planes_divide_at_most_4_parts :
  (∃ p1 p2 : Plane, are_parallel p1 p2 ∨ intersect p1 p2) →
  (exists n : ℕ, n <= 4) :=
sorry

end two_planes_divide_at_most_4_parts_l1273_127323


namespace even_function_expression_l1273_127322

theorem even_function_expression (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = x * (2 * x - 1)) :
  ∀ x, x > 0 → f x = x * (2 * x + 1) :=
by 
  sorry

end even_function_expression_l1273_127322


namespace proof_equivalent_expression_l1273_127389

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2

theorem proof_equivalent_expression (x y : ℝ) :
  (dollar ((x + y) ^ 2) (dollar y x)) - (dollar (dollar x y) (dollar x y)) = 
  4 * (x + y) ^ 2 * ((x + y) ^ 2 - 1) :=
by
  sorry

end proof_equivalent_expression_l1273_127389


namespace vartan_spent_on_recreation_last_week_l1273_127308

variable (W P : ℝ)
variable (h1 : P = 0.20)
variable (h2 : W > 0)

theorem vartan_spent_on_recreation_last_week :
  (P * W) = 0.20 * W :=
by
  sorry

end vartan_spent_on_recreation_last_week_l1273_127308


namespace find_a_l1273_127309

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a {x0 a : ℝ} (h : f x0 a - g x0 a = 3) : a = -Real.log 2 - 1 :=
sorry

end find_a_l1273_127309


namespace toilet_paper_duration_l1273_127302

theorem toilet_paper_duration :
  let bill_weekday := 3 * 5
  let wife_weekday := 4 * 8
  let kid_weekday := 5 * 6
  let total_weekday := bill_weekday + wife_weekday + 2 * kid_weekday
  let bill_weekend := 4 * 6
  let wife_weekend := 5 * 10
  let kid_weekend := 6 * 5
  let total_weekend := bill_weekend + wife_weekend + 2 * kid_weekend
  let total_week := 5 * total_weekday + 2 * total_weekend
  let total_squares := 1000 * 300
  let weeks_last := total_squares / total_week
  let days_last := weeks_last * 7
  days_last = 2615 :=
sorry

end toilet_paper_duration_l1273_127302


namespace triangle_sides_inequality_l1273_127314

theorem triangle_sides_inequality
  {a b c : ℝ} (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0)
  (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  -- We would place the proof here if it were required
  sorry

end triangle_sides_inequality_l1273_127314


namespace expandProduct_l1273_127306

theorem expandProduct (x : ℝ) : 4 * (x - 5) * (x + 8) = 4 * x^2 + 12 * x - 160 := 
by 
  sorry

end expandProduct_l1273_127306


namespace cheese_left_after_10_customers_l1273_127353

theorem cheese_left_after_10_customers :
  ∀ (S : ℕ → ℚ), (∀ n, S n = (20 * n) / (n + 10)) →
  20 - S 10 = 10 := by
  sorry

end cheese_left_after_10_customers_l1273_127353


namespace arithmetic_seq_sum_l1273_127392

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 3 + a 4 + a 5 + a 6 + a 7 = 250) : a 2 + a 8 = 100 :=
sorry

end arithmetic_seq_sum_l1273_127392


namespace seating_arrangements_l1273_127367

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end seating_arrangements_l1273_127367


namespace eq_x_minus_y_l1273_127365

theorem eq_x_minus_y (x y : ℝ) : (x - y) * (x - y) = x^2 - 2 * x * y + y^2 :=
by
  sorry

end eq_x_minus_y_l1273_127365


namespace painted_cubes_count_l1273_127330

/-- A theorem to prove the number of painted small cubes in a larger cube. -/
theorem painted_cubes_count (total_cubes unpainted_cubes : ℕ) (a b : ℕ) :
  total_cubes = a * a * a →
  unpainted_cubes = (a - 2) * (a - 2) * (a - 2) →
  22 = unpainted_cubes →
  64 = total_cubes →
  ∃ m, m = total_cubes - unpainted_cubes ∧ m = 42 :=
by
  sorry

end painted_cubes_count_l1273_127330


namespace circle_tangent_line_k_range_l1273_127310

theorem circle_tangent_line_k_range
  (k : ℝ)
  (P Q : ℝ × ℝ)
  (c : ℝ × ℝ := (0, 1)) -- Circle center
  (r : ℝ := 1) -- Circle radius
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 2 * y = 0)
  (line_eq : ∀ (x y : ℝ), k * x + y + 3 = 0)
  (dist_pq : Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) = Real.sqrt 3) :
  k ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ici (Real.sqrt 3) :=
by
  sorry

end circle_tangent_line_k_range_l1273_127310


namespace ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l1273_127388

theorem ellipse_foci_on_x_axis_major_axis_twice_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m * y^2 = 1) → (∃ a b : ℝ, a = 1 ∧ b = Real.sqrt (1 / m) ∧ a = 2 * b) → m = 4 :=
by
  sorry

end ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l1273_127388


namespace total_pencils_sold_l1273_127316

theorem total_pencils_sold (price_reduced: Bool)
  (day1_students : ℕ) (first4_d1 : ℕ) (next3_d1 : ℕ) (last3_d1 : ℕ)
  (day2_students : ℕ) (first5_d2 : ℕ) (next6_d2 : ℕ) (last4_d2 : ℕ)
  (day3_students : ℕ) (first10_d3 : ℕ) (next10_d3 : ℕ) (last10_d3 : ℕ)
  (day1_total : day1_students = 10 ∧ first4_d1 = 4 ∧ next3_d1 = 3 ∧ last3_d1 = 3 ∧
    (first4_d1 * 5) + (next3_d1 * 7) + (last3_d1 * 3) = 50)
  (day2_total : day2_students = 15 ∧ first5_d2 = 5 ∧ next6_d2 = 6 ∧ last4_d2 = 4 ∧
    (first5_d2 * 4) + (next6_d2 * 9) + (last4_d2 * 6) = 98)
  (day3_total : day3_students = 2 * day2_students ∧ first10_d3 = 10 ∧ next10_d3 = 10 ∧ last10_d3 = 10 ∧
    (first10_d3 * 2) + (next10_d3 * 8) + (last10_d3 * 4) = 140) :
  (50 + 98 + 140 = 288) :=
sorry

end total_pencils_sold_l1273_127316


namespace fifth_term_arithmetic_sequence_l1273_127307

noncomputable def fifth_term (x y : ℚ) (a1 : ℚ := x + 2 * y) (a2 : ℚ := x - 2 * y) (a3 : ℚ := x + 2 * y^2) (a4 : ℚ := x / (2 * y)) (d : ℚ := -4 * y) : ℚ :=
    a4 + d

theorem fifth_term_arithmetic_sequence (x y : ℚ) (h1 : y ≠ 0) :
  (fifth_term x y - (-((x : ℚ) / 6) - 12)) = 0 :=
by
  sorry

end fifth_term_arithmetic_sequence_l1273_127307


namespace probability_of_selecting_3_co_captains_is_correct_l1273_127359

def teams : List ℕ := [4, 6, 7, 9]

def probability_of_selecting_3_co_captains (n : ℕ) : ℚ :=
  if n = 4 then 1/4
  else if n = 6 then 1/20
  else if n = 7 then 1/35
  else if n = 9 then 1/84
  else 0

def total_probability : ℚ :=
  (1/4) * (probability_of_selecting_3_co_captains 4 +
            probability_of_selecting_3_co_captains 6 +
            probability_of_selecting_3_co_captains 7 +
            probability_of_selecting_3_co_captains 9)

theorem probability_of_selecting_3_co_captains_is_correct :
  total_probability = 143 / 1680 :=
by
  -- The proof will be inserted here
  sorry

end probability_of_selecting_3_co_captains_is_correct_l1273_127359


namespace verify_base_case_l1273_127364

theorem verify_base_case : 1 + (1 / 2) + (1 / 3) < 2 :=
sorry

end verify_base_case_l1273_127364


namespace find_integer_sets_l1273_127382

noncomputable def satisfy_equation (A B C : ℤ) : Prop :=
  A ^ 2 - B ^ 2 - C ^ 2 = 1 ∧ B + C - A = 3

theorem find_integer_sets :
  { (A, B, C) : ℤ × ℤ × ℤ | satisfy_equation A B C } = {(9, 8, 4), (9, 4, 8), (-3, 2, -2), (-3, -2, 2)} :=
  sorry

end find_integer_sets_l1273_127382


namespace hardest_vs_least_worked_hours_difference_l1273_127398

-- Let x be the scaling factor for the ratio
-- The times worked are 2x, 3x, and 4x

def project_time_difference (x : ℕ) : Prop :=
  let time1 := 2 * x
  let time2 := 3 * x
  let time3 := 4 * x
  (time1 + time2 + time3 = 90) ∧ ((4 * x - 2 * x) = 20)

theorem hardest_vs_least_worked_hours_difference :
  ∃ x : ℕ, project_time_difference x :=
by
  sorry

end hardest_vs_least_worked_hours_difference_l1273_127398


namespace sample_second_grade_l1273_127343

theorem sample_second_grade (r1 r2 r3 sample_size : ℕ) (h1 : r1 = 3) (h2 : r2 = 3) (h3 : r3 = 4) (h_sample_size : sample_size = 50) : (r2 * sample_size) / (r1 + r2 + r3) = 15 := by
  sorry

end sample_second_grade_l1273_127343


namespace unit_circle_inequality_l1273_127352

theorem unit_circle_inequality 
  (a b c d : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (habcd : a * b + c * d = 1) 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) 
  (hx1 : x1^2 + y1^2 = 1)
  (hx2 : x2^2 + y2^2 = 1)
  (hx3 : x3^2 + y3^2 = 1)
  (hx4 : x4^2 + y4^2 = 1) :
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2 ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := 
sorry

end unit_circle_inequality_l1273_127352


namespace multiplication_24_12_l1273_127383

theorem multiplication_24_12 :
  let a := 24
  let b := 12
  let b1 := 10
  let b2 := 2
  let p1 := a * b2
  let p2 := a * b1
  let sum := p1 + p2
  b = b1 + b2 →
  p1 = a * b2 →
  p2 = a * b1 →
  sum = p1 + p2 →
  a * b = sum :=
by
  intros
  sorry

end multiplication_24_12_l1273_127383


namespace system_non_zero_solution_condition_l1273_127327

theorem system_non_zero_solution_condition (a b c : ℝ) :
  (∃ (x y z : ℝ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧
    x = b * y + c * z ∧
    y = c * z + a * x ∧
    z = a * x + b * y) ↔
  (2 * a * b * c + a * b + b * c + c * a - 1 = 0) :=
sorry

end system_non_zero_solution_condition_l1273_127327


namespace smallest_difference_l1273_127361

-- Definition for the given problem conditions.
def side_lengths (AB BC AC : ℕ) : Prop := 
  AB + BC + AC = 2023 ∧ AB < BC ∧ BC ≤ AC ∧ 
  AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB

theorem smallest_difference (AB BC AC : ℕ) 
  (h: side_lengths AB BC AC) : 
  ∃ (AB BC AC : ℕ), side_lengths AB BC AC ∧ (BC - AB = 1) :=
by
  sorry

end smallest_difference_l1273_127361


namespace simplify_and_evaluate_expression_l1273_127342

noncomputable def expression (a : ℝ) : ℝ :=
  ((a^2 - 1) / (a - 3) - a - 1) / ((a + 1) / (a^2 - 6 * a + 9))

theorem simplify_and_evaluate_expression : expression (3 - Real.sqrt 2) = -2 * Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_expression_l1273_127342


namespace inequality_solution_l1273_127340

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (2^x + 1)

lemma monotone_decreasing (a : ℝ) : ∀ x y : ℝ, x < y → f a y < f a x := 
sorry

lemma odd_function (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 0 := 
sorry

theorem inequality_solution (t : ℝ) (a : ℝ) (h_monotone : ∀ x y : ℝ, x < y → f a y < f a x)
    (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : t ≥ 4 / 3 ↔ f a (2 * t + 1) + f a (t - 5) ≤ 0 := 
sorry

end inequality_solution_l1273_127340


namespace fraction_meaningful_range_l1273_127303

theorem fraction_meaningful_range (x : ℝ) : 5 - x ≠ 0 ↔ x ≠ 5 :=
by sorry

end fraction_meaningful_range_l1273_127303


namespace total_people_l1273_127360

theorem total_people (M W C : ℕ) (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : C = 30) : M + W + C = 300 :=
by
  sorry

end total_people_l1273_127360


namespace min_length_QR_l1273_127350

theorem min_length_QR (PQ PR SR QS QR : ℕ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  QR > PR - PQ ∧ QR > QS - SR ↔ QR = 16 :=
by
  sorry

end min_length_QR_l1273_127350


namespace smallest_five_digit_multiple_of_18_correct_l1273_127369

def smallest_five_digit_multiple_of_18 : ℕ := 10008

theorem smallest_five_digit_multiple_of_18_correct :
  (smallest_five_digit_multiple_of_18 >= 10000) ∧ 
  (smallest_five_digit_multiple_of_18 < 100000) ∧ 
  (smallest_five_digit_multiple_of_18 % 18 = 0) :=
by
  sorry

end smallest_five_digit_multiple_of_18_correct_l1273_127369


namespace problem1_problem2_l1273_127325

-- Definitions for permutation and combination
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problems statements
theorem problem1 : 
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := by 
  sorry

theorem problem2 :
  C 200 198 + C 200 196 + 2 * C 200 197 = C 202 4 := by 
  sorry

end problem1_problem2_l1273_127325


namespace cost_of_goat_l1273_127338

theorem cost_of_goat (G : ℝ) (goat_count : ℕ) (llama_count : ℕ) (llama_multiplier : ℝ) (total_cost : ℝ) 
    (h1 : goat_count = 3)
    (h2 : llama_count = 2 * goat_count)
    (h3 : llama_multiplier = 1.5)
    (h4 : total_cost = 4800) : G = 400 :=
by
  sorry

end cost_of_goat_l1273_127338


namespace number_of_correct_conclusions_l1273_127354

-- Given conditions
variables {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : c > 3)
           (h₂ : a * 25 + b * 5 + c = 0)
           (h₃ : -b / (2 * a) = 2)
           (h₄ : a < 0)

-- Proof should show:
theorem number_of_correct_conclusions 
  (h₀ : a ≠ 0)
  (h₁ : c > 3)
  (h₂ : 25 * a + 5 * b + c = 0)
  (h₃ : - b / (2 * a) = 2)
  (h₄ : a < 0) :
  (a * b * c < 0) ∧ 
  (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂) ∧ (a * x₁^2 + b * x₁ + c = 2) ∧ (a * x₂^2 + b * x₂ + c = 2)) ∧ 
  (a < -3 / 5) := 
by
  sorry

end number_of_correct_conclusions_l1273_127354


namespace problem_proof_l1273_127385

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + (1 / Real.sqrt (2 - x))
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {y | y ≥ 1}
def CU_B : Set ℝ := {y | y < 1}
def U : Set ℝ := Set.univ

theorem problem_proof :
  (∀ x, x ∈ A ↔ -1 ≤ x ∧ x < 2) ∧
  (∀ y, y ∈ B ↔ y ≥ 1) ∧
  (A ∩ CU_B = {x | -1 ≤ x ∧ x < 1}) :=
by
  sorry

end problem_proof_l1273_127385


namespace initial_percentage_of_water_l1273_127399

variable (V : ℝ) (W : ℝ) (P : ℝ)

theorem initial_percentage_of_water 
  (h1 : V = 120) 
  (h2 : W = 8)
  (h3 : (V + W) * 0.25 = ((P / 100) * V) + W) : 
  P = 20 :=
by
  sorry

end initial_percentage_of_water_l1273_127399


namespace savings_relationship_l1273_127384

def combined_salary : ℝ := 3000
def salary_A : ℝ := 2250
def salary_B : ℝ := combined_salary - salary_A
def savings_A : ℝ := 0.05 * salary_A
def savings_B : ℝ := 0.15 * salary_B

theorem savings_relationship : savings_A = 112.5 ∧ savings_B = 112.5 := by
  have h1 : salary_B = 750 := by sorry
  have h2 : savings_A = 0.05 * 2250 := by sorry
  have h3 : savings_B = 0.15 * 750 := by sorry
  have h4 : savings_A = 112.5 := by sorry
  have h5 : savings_B = 112.5 := by sorry
  exact And.intro h4 h5

end savings_relationship_l1273_127384
