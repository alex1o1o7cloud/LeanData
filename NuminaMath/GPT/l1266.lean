import Mathlib

namespace inequality_interval_l1266_126646

def differentiable_on_R (f : ℝ → ℝ) : Prop := Differentiable ℝ f
def strictly_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y

theorem inequality_interval (f : ℝ → ℝ)
  (h_diff : differentiable_on_R f)
  (h_cond : ∀ x : ℝ, f x > deriv f x)
  (h_init : f 0 = 1) :
  ∀ x : ℝ, (x > 0) ↔ (f x / Real.exp x < 1) := 
by
  sorry

end inequality_interval_l1266_126646


namespace volume_after_increase_l1266_126610

variable (l w h : ℝ)

def original_volume (l w h : ℝ) : ℝ := l * w * h
def original_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def original_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)
def increased_volume (l w h : ℝ) : ℝ := (l + 2) * (w + 2) * (h + 2)

theorem volume_after_increase :
  original_volume l w h = 5000 →
  original_surface_area l w h = 1800 →
  original_edge_sum l w h = 240 →
  increased_volume l w h = 7048 := by
  sorry

end volume_after_increase_l1266_126610


namespace initial_percentage_of_milk_l1266_126681

theorem initial_percentage_of_milk (P : ℝ) :
  (P / 100) * 60 = (68 / 100) * 74.11764705882354 → P = 84 :=
by
  sorry

end initial_percentage_of_milk_l1266_126681


namespace triangle_side_lengths_consecutive_l1266_126685

theorem triangle_side_lengths_consecutive (n : ℕ) (a b c A : ℕ) 
  (h1 : a = n - 1) (h2 : b = n) (h3 : c = n + 1) (h4 : A = n + 2)
  (h5 : 2 * A * A = 3 * n^2 * (n^2 - 4)) :
  a = 3 ∧ b = 4 ∧ c = 5 :=
sorry

end triangle_side_lengths_consecutive_l1266_126685


namespace part_one_part_two_part_three_l1266_126624

def numberOfWaysToPlaceBallsInBoxes : ℕ :=
  4 ^ 4

def numberOfWaysOneBoxEmpty : ℕ :=
  Nat.choose 4 2 * (Nat.factorial 4 / Nat.factorial 1)

def numberOfWaysTwoBoxesEmpty : ℕ :=
  (Nat.choose 4 1 * (Nat.factorial 4 / Nat.factorial 2)) + (Nat.choose 4 2 * (Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)))

theorem part_one : numberOfWaysToPlaceBallsInBoxes = 256 := by
  sorry

theorem part_two : numberOfWaysOneBoxEmpty = 144 := by
  sorry

theorem part_three : numberOfWaysTwoBoxesEmpty = 120 := by
  sorry

end part_one_part_two_part_three_l1266_126624


namespace union_complement_l1266_126604

open Set Real

def P : Set ℝ := { x | x^2 - 4 * x + 3 ≤ 0 }
def Q : Set ℝ := { x | x^2 - 4 < 0 }

theorem union_complement :
  P ∪ (compl Q) = (Iic (-2)) ∪ Ici 1 :=
by
  sorry

end union_complement_l1266_126604


namespace inradius_circumradius_le_height_l1266_126623

theorem inradius_circumradius_le_height
    {α β γ : ℝ}
    (hα : 0 < α ∧ α ≤ 90)
    (hβ : 0 < β ∧ β ≤ 90)
    (hγ : 0 < γ ∧ γ ≤ 90)
    (α_ge_β : α ≥ β)
    (β_ge_γ : β ≥ γ)
    {r R h : ℝ} :
  r + R ≤ h := 
sorry

end inradius_circumradius_le_height_l1266_126623


namespace range_of_a_exists_distinct_x1_x2_eq_f_l1266_126632

noncomputable
def f (a x : ℝ) : ℝ :=
  if x < 1 then a * x + 1 - 4 * a else x^2 - 3 * a * x

theorem range_of_a_exists_distinct_x1_x2_eq_f :
  { a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2 } = 
  { a : ℝ | (a > (2 / 3)) ∨ (a ≤ 0) } :=
sorry

end range_of_a_exists_distinct_x1_x2_eq_f_l1266_126632


namespace unique_solution_iff_t_eq_quarter_l1266_126637

variable {x y t : ℝ}

theorem unique_solution_iff_t_eq_quarter : (∃! (x y : ℝ), (x ≥ y^2 + t * y ∧ y^2 + t * y ≥ x^2 + t)) ↔ t = 1 / 4 :=
by
  sorry

end unique_solution_iff_t_eq_quarter_l1266_126637


namespace cubic_identity_l1266_126640

theorem cubic_identity (x : ℝ) (hx : x + 1/x = -5) : x^3 + 1/x^3 = -110 := by
  sorry

end cubic_identity_l1266_126640


namespace find_purchase_price_l1266_126696

noncomputable def purchase_price (total_paid : ℝ) (interest_percent : ℝ) : ℝ :=
    total_paid / (1 + interest_percent)

theorem find_purchase_price :
  purchase_price 130 0.09090909090909092 = 119.09 :=
by
  -- Normally we would provide the full proof here, but it is omitted as per instructions
  sorry

end find_purchase_price_l1266_126696


namespace proposition_equivalence_l1266_126690

-- Definition of propositions p and q
variables (p q : Prop)

-- Statement of the problem in Lean 4
theorem proposition_equivalence :
  (p ∨ q) → ¬(p ∧ q) ↔ (¬((p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → (p ∨ q))) :=
sorry

end proposition_equivalence_l1266_126690


namespace log_inequality_l1266_126657

theorem log_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : 
  Real.log b / Real.log a + Real.log a / Real.log b ≤ -2 := sorry

end log_inequality_l1266_126657


namespace not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l1266_126629

theorem not_right_triangle_sqrt_3_sqrt_4_sqrt_5 :
  ¬ (Real.sqrt 3)^2 + (Real.sqrt 4)^2 = (Real.sqrt 5)^2 :=
by
  -- Start constructing the proof here
  sorry

end not_right_triangle_sqrt_3_sqrt_4_sqrt_5_l1266_126629


namespace length_of_brick_proof_l1266_126612

noncomputable def length_of_brick (courtyard_length courtyard_width : ℕ) (brick_width : ℕ) (total_bricks : ℕ) : ℕ :=
  let total_area_cm := courtyard_length * courtyard_width * 10000
  total_area_cm / (brick_width * total_bricks)

theorem length_of_brick_proof :
  length_of_brick 25 16 10 20000 = 20 :=
by
  unfold length_of_brick
  sorry

end length_of_brick_proof_l1266_126612


namespace calc_quotient_l1266_126680

theorem calc_quotient (a b : ℕ) (h1 : a - b = 177) (h2 : 14^2 = 196) : (a - b)^2 / 196 = 144 := 
by sorry

end calc_quotient_l1266_126680


namespace time_to_cross_pole_l1266_126651

-- Conditions
def train_speed_kmh : ℕ := 108
def train_length_m : ℕ := 210

-- Conversion functions
def km_per_hr_to_m_per_sec (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

-- Theorem to be proved
theorem time_to_cross_pole : (train_length_m : ℕ) / (km_per_hr_to_m_per_sec train_speed_kmh) = 7 := by
  -- we'll use sorry here to skip the actual proof steps.
  sorry

end time_to_cross_pole_l1266_126651


namespace transport_cost_in_euros_l1266_126672

def cost_per_kg : ℝ := 18000
def weight_g : ℝ := 300
def exchange_rate : ℝ := 0.95

theorem transport_cost_in_euros :
  (cost_per_kg * (weight_g / 1000) * exchange_rate) = 5130 :=
by sorry

end transport_cost_in_euros_l1266_126672


namespace Maria_high_school_students_l1266_126676

theorem Maria_high_school_students (M J : ℕ) (h1 : M = 4 * J) (h2 : M + J = 3600) : M = 2880 :=
sorry

end Maria_high_school_students_l1266_126676


namespace max_distance_l1266_126658

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := 
  (rho * Real.cos theta, rho * Real.sin theta)

noncomputable def curve_C (p : ℝ × ℝ) : Prop := 
  let x := p.1 
  let y := p.2 
  x^2 + y^2 - 2*y = 0

noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  (-3/5 * t + 2, 4/5 * t)

def x_axis_intersection (l : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := l 0 
  (x, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance {M : ℝ × ℝ} {N : ℝ × ℝ}
  (curve_c : (ℝ × ℝ) → Prop)
  (line_l : ℝ → ℝ × ℝ)
  (h1 : curve_c = curve_C)
  (h2 : line_l = line_l)
  (M_def : x_axis_intersection line_l = M)
  (hNP : curve_c N) :
  distance M N ≤ Real.sqrt 5 + 1 :=
sorry

end max_distance_l1266_126658


namespace positive_n_for_modulus_eq_l1266_126620

theorem positive_n_for_modulus_eq (n : ℕ) (h_pos : 0 < n) (h_eq : Complex.abs (5 + (n : ℂ) * Complex.I) = 5 * Real.sqrt 26) : n = 25 :=
by
  sorry

end positive_n_for_modulus_eq_l1266_126620


namespace remainder_of_sum_of_5_consecutive_numbers_mod_9_l1266_126615

theorem remainder_of_sum_of_5_consecutive_numbers_mod_9 :
  (9154 + 9155 + 9156 + 9157 + 9158) % 9 = 1 :=
by
  sorry

end remainder_of_sum_of_5_consecutive_numbers_mod_9_l1266_126615


namespace alley_width_l1266_126634

theorem alley_width (ℓ : ℝ) (m : ℝ) (n : ℝ): ℓ * (1 / 2 + Real.cos (70 * Real.pi / 180)) = ℓ * (Real.cos (60 * Real.pi / 180)) + ℓ * (Real.cos (70 * Real.pi / 180)) := by
  sorry

end alley_width_l1266_126634


namespace train_speed_correct_l1266_126667

-- Define the length of the train
def train_length : ℝ := 200

-- Define the time taken to cross the telegraph post
def cross_time : ℝ := 8

-- Define the expected speed of the train
def expected_speed : ℝ := 25

-- Prove that the speed of the train is as expected
theorem train_speed_correct (length time : ℝ) (h_length : length = train_length) (h_time : time = cross_time) : 
  (length / time = expected_speed) :=
by
  rw [h_length, h_time]
  sorry

end train_speed_correct_l1266_126667


namespace consecutive_numbers_product_l1266_126669

theorem consecutive_numbers_product (a b c d : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h4 : a + d = 109) :
  b * c = 2970 :=
by {
  -- Proof goes here
  sorry
}

end consecutive_numbers_product_l1266_126669


namespace number_of_participants_l1266_126649

-- Define the conditions and theorem
theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 231) : n = 22 :=
  sorry

end number_of_participants_l1266_126649


namespace orchids_to_roses_ratio_l1266_126601

noncomputable def total_centerpieces : ℕ := 6
noncomputable def roses_per_centerpiece : ℕ := 8
noncomputable def lilies_per_centerpiece : ℕ := 6
noncomputable def total_budget : ℕ := 2700
noncomputable def cost_per_flower : ℕ := 15
noncomputable def total_flowers : ℕ := total_budget / cost_per_flower

noncomputable def total_roses : ℕ := total_centerpieces * roses_per_centerpiece
noncomputable def total_lilies : ℕ := total_centerpieces * lilies_per_centerpiece
noncomputable def total_roses_and_lilies : ℕ := total_roses + total_lilies
noncomputable def total_orchids : ℕ := total_flowers - total_roses_and_lilies
noncomputable def orchids_per_centerpiece : ℕ := total_orchids / total_centerpieces

theorem orchids_to_roses_ratio : orchids_per_centerpiece / roses_per_centerpiece = 2 :=
by
  sorry

end orchids_to_roses_ratio_l1266_126601


namespace integer_pairs_m_n_l1266_126642

theorem integer_pairs_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (cond1 : ∃ k1 : ℕ, k1 * m = 3 * n ^ 2)
  (cond2 : ∃ k2 : ℕ, k2 ^ 2 = n ^ 2 + m) :
  ∃ a : ℕ, m = 3 * a ^ 2 ∧ n = a :=
by
  sorry

end integer_pairs_m_n_l1266_126642


namespace generatrix_length_l1266_126678

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end generatrix_length_l1266_126678


namespace gcd_n4_plus_16_n_plus_3_eq_1_l1266_126694

theorem gcd_n4_plus_16_n_plus_3_eq_1 (n : ℕ) (h : n > 16) : gcd (n^4 + 16) (n + 3) = 1 := 
sorry

end gcd_n4_plus_16_n_plus_3_eq_1_l1266_126694


namespace ratio_of_pentagon_side_to_rectangle_width_l1266_126687

-- Definitions based on the conditions
def pentagon_perimeter : ℝ := 60
def rectangle_perimeter : ℝ := 60
def rectangle_length (w : ℝ) : ℝ := 2 * w

-- The statement to be proven
theorem ratio_of_pentagon_side_to_rectangle_width :
  ∀ w : ℝ, 2 * (rectangle_length w + w) = rectangle_perimeter → (pentagon_perimeter / 5) / w = 6 / 5 :=
by
  sorry

end ratio_of_pentagon_side_to_rectangle_width_l1266_126687


namespace new_parabola_after_shift_l1266_126626

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the transformation functions for shifting the parabola
def shift_left (x : ℝ) (shift : ℝ) : ℝ := x + shift
def shift_down (y : ℝ) (shift : ℝ) : ℝ := y - shift

-- Prove the transformation yields the correct new parabola equation
theorem new_parabola_after_shift : 
  (∀ x : ℝ, (shift_down (original_parabola (shift_left x 2)) 3) = (x + 2)^2 - 2) :=
by
  sorry

end new_parabola_after_shift_l1266_126626


namespace range_of_a_l1266_126643

theorem range_of_a (x a : ℝ) (p : (x + 1)^2 > 4) (q : x > a) 
  (h : (¬((x + 1)^2 > 4)) → (¬(x > a)))
  (sufficient_but_not_necessary : (¬((x + 1)^2 > 4)) → (¬(x > a))) : a ≥ 1 :=
sorry

end range_of_a_l1266_126643


namespace total_amount_shared_l1266_126613

-- conditions as definitions
def Parker_share : ℕ := 50
def ratio_part_Parker : ℕ := 2
def ratio_total_parts : ℕ := 2 + 3 + 4
def value_of_one_part : ℕ := Parker_share / ratio_part_Parker

-- question translated to Lean statement with expected correct answer
theorem total_amount_shared : ratio_total_parts * value_of_one_part = 225 := by
  sorry

end total_amount_shared_l1266_126613


namespace total_employee_costs_in_February_l1266_126636

def weekly_earnings (hours_per_week : ℕ) (rate_per_hour : ℕ) : ℕ :=
  hours_per_week * rate_per_hour

def monthly_earnings 
  (hours_per_week : ℕ) 
  (rate_per_hour : ℕ) 
  (weeks_worked : ℕ) 
  (bonus_deduction : ℕ := 0) 
  : ℕ :=
  weeks_worked * weekly_earnings hours_per_week rate_per_hour + bonus_deduction

theorem total_employee_costs_in_February 
  (hours_Fiona : ℕ := 40) (rate_Fiona : ℕ := 20) (weeks_worked_Fiona : ℕ := 3)
  (hours_John : ℕ := 30) (rate_John : ℕ := 22) (overtime_hours_John : ℕ := 10)
  (hours_Jeremy : ℕ := 25) (rate_Jeremy : ℕ := 18) (bonus_Jeremy : ℕ := 200)
  (hours_Katie : ℕ := 35) (rate_Katie : ℕ := 21) (deduction_Katie : ℕ := 150)
  (hours_Matt : ℕ := 28) (rate_Matt : ℕ := 19) : 
  monthly_earnings hours_Fiona rate_Fiona weeks_worked_Fiona 
  + monthly_earnings hours_John rate_John 4 
    + overtime_hours_John * (rate_John * 3 / 2)
  + monthly_earnings hours_Jeremy rate_Jeremy 4 bonus_Jeremy
  + monthly_earnings hours_Katie rate_Katie 4 - deduction_Katie
  + monthly_earnings hours_Matt rate_Matt 4 = 13278 := 
by sorry

end total_employee_costs_in_February_l1266_126636


namespace greatest_x_l1266_126665

theorem greatest_x (x : ℕ) (h_pos : 0 < x) (h_ineq : (x^6) / (x^3) < 18) : x = 2 :=
by sorry

end greatest_x_l1266_126665


namespace inequality_condition_l1266_126653

variables {a b c : ℝ} {x : ℝ}

theorem inequality_condition (h : a * a + b * b < c * c) : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end inequality_condition_l1266_126653


namespace sufficient_condition_frac_ineq_inequality_transformation_problem_equivalence_l1266_126684

theorem sufficient_condition_frac_ineq (x : ℝ) : (1 < x ∧ x < 2) → ( (x + 1) / (x - 1) > 2) :=
by
  -- Given that 1 < x and x < 2, we need to show (x + 1) / (x - 1) > 2
  sorry

theorem inequality_transformation (x : ℝ) : ( (x + 1) / (x - 1) > 2) ↔ ( (x - 1) * (x - 3) < 0 ) :=
by
  -- Prove that (x + 1) / (x - 1) > 2 is equivalent to (x - 1)(x - 3) < 0
  sorry

theorem problem_equivalence (x : ℝ) : ( (x + 1) / (x - 1) > 2) → (1 < x ∧ x < 3) :=
by
  -- Prove that (x + 1) / (x - 1) > 2 implies 1 < x < 3
  sorry

end sufficient_condition_frac_ineq_inequality_transformation_problem_equivalence_l1266_126684


namespace length_of_AB_in_triangle_l1266_126693

open Real

theorem length_of_AB_in_triangle
  (AC BC : ℝ)
  (area : ℝ) :
  AC = 4 →
  BC = 3 →
  area = 3 * sqrt 3 →
  ∃ AB : ℝ, AB = sqrt 13 :=
by
  sorry

end length_of_AB_in_triangle_l1266_126693


namespace number_of_clown_mobiles_l1266_126622

def num_clown_mobiles (total_clowns clowns_per_mobile : ℕ) : ℕ :=
  total_clowns / clowns_per_mobile

theorem number_of_clown_mobiles :
  num_clown_mobiles 140 28 = 5 :=
by
  sorry

end number_of_clown_mobiles_l1266_126622


namespace geometric_mean_of_negatives_l1266_126668

theorem geometric_mean_of_negatives :
  ∃ x : ℝ, x^2 = (-2) * (-8) ∧ (x = 4 ∨ x = -4) := by
  sorry

end geometric_mean_of_negatives_l1266_126668


namespace fraction_of_seats_taken_l1266_126660

theorem fraction_of_seats_taken : 
  ∀ (total_seats broken_fraction available_seats : ℕ), 
    total_seats = 500 → 
    broken_fraction = 1 / 10 → 
    available_seats = 250 → 
    (total_seats - available_seats - total_seats * broken_fraction) / total_seats = 2 / 5 :=
by
  intro total_seats broken_fraction available_seats
  intro h1 h2 h3
  sorry

end fraction_of_seats_taken_l1266_126660


namespace total_Pokemon_cards_l1266_126630

def j : Nat := 6
def o : Nat := j + 2
def r : Nat := 3 * o
def t : Nat := j + o + r

theorem total_Pokemon_cards : t = 38 := by 
  sorry

end total_Pokemon_cards_l1266_126630


namespace intersection_complement_l1266_126603

open Set

/-- The universal set U as the set of all real numbers -/
def U : Set ℝ := @univ ℝ

/-- The set M -/
def M : Set ℝ := {-1, 0, 1}

/-- The set N defined by the equation x^2 + x = 0 -/
def N : Set ℝ := {x | x^2 + x = 0}

/-- The complement of set N in the universal set U -/
def C_U_N : Set ℝ := {x ∈ U | x ≠ -1 ∧ x ≠ 0}

theorem intersection_complement :
  M ∩ C_U_N = {1} :=
by
  sorry

end intersection_complement_l1266_126603


namespace smaller_angle_at_10_oclock_l1266_126671

def degreeMeasureSmallerAngleAt10 := 
  let totalDegrees := 360
  let numHours := 12
  let degreesPerHour := totalDegrees / numHours
  let hourHandPosition := 10
  let minuteHandPosition := 12
  let divisionsBetween := if hourHandPosition < minuteHandPosition then minuteHandPosition - hourHandPosition else hourHandPosition - minuteHandPosition
  degreesPerHour * divisionsBetween

theorem smaller_angle_at_10_oclock : degreeMeasureSmallerAngleAt10 = 60 :=
  by 
    let totalDegrees := 360
    let numHours := 12
    let degreesPerHour := totalDegrees / numHours
    have h1 : degreesPerHour = 30 := by norm_num
    let hourHandPosition := 10
    let minuteHandPosition := 12
    let divisionsBetween := minuteHandPosition - hourHandPosition
    have h2 : divisionsBetween = 2 := by norm_num
    show 30 * divisionsBetween = 60
    calc 
      30 * 2 = 60 := by norm_num

end smaller_angle_at_10_oclock_l1266_126671


namespace wheat_flour_one_third_l1266_126688

theorem wheat_flour_one_third (recipe_cups: ℚ) (third_recipe: ℚ) 
  (h1: recipe_cups = 5 + 2 / 3) (h2: third_recipe = recipe_cups / 3) :
  third_recipe = 1 + 8 / 9 :=
by
  sorry

end wheat_flour_one_third_l1266_126688


namespace mowing_difference_l1266_126692

-- Define the number of times mowed in spring and summer
def mowedSpring : ℕ := 8
def mowedSummer : ℕ := 5

-- Prove the difference between spring and summer mowing is 3
theorem mowing_difference : mowedSpring - mowedSummer = 3 := by
  sorry

end mowing_difference_l1266_126692


namespace convoy_length_after_checkpoint_l1266_126679

theorem convoy_length_after_checkpoint
  (L_initial : ℝ) (v_initial : ℝ) (v_final : ℝ) (t_fin : ℝ)
  (H_initial_len : L_initial = 300)
  (H_initial_speed : v_initial = 60)
  (H_final_speed : v_final = 40)
  (H_time_last_car : t_fin = (300 / 1000) / 60) :
  L_initial * v_final / v_initial - (v_final * ((300 / 1000) / 60)) = 200 :=
by
  sorry

end convoy_length_after_checkpoint_l1266_126679


namespace find_third_discount_percentage_l1266_126673

noncomputable def third_discount_percentage (x : ℝ) : Prop :=
  let item_price := 68
  let num_items := 3
  let first_discount := 0.15
  let second_discount := 0.10
  let total_initial_price := num_items * item_price
  let price_after_first_discount := total_initial_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount * (1 - x / 100) = 105.32

theorem find_third_discount_percentage : ∃ x : ℝ, third_discount_percentage x ∧ x = 32.5 :=
by
  sorry

end find_third_discount_percentage_l1266_126673


namespace total_cost_function_range_of_x_minimum_cost_when_x_is_2_l1266_126697

def transportation_cost (x : ℕ) : ℕ :=
  300 * x + 500 * (12 - x) + 400 * (10 - x) + 800 * (x - 2)

theorem total_cost_function (x : ℕ) : transportation_cost x = 200 * x + 8400 := by
  -- Simply restate the definition in the theorem form
  sorry

theorem range_of_x (x : ℕ) : 2 ≤ x ∧ x ≤ 10 := by
  -- Provide necessary constraints in theorem form
  sorry

theorem minimum_cost_when_x_is_2 : transportation_cost 2 = 8800 := by
  -- Final cost at minimum x
  sorry

end total_cost_function_range_of_x_minimum_cost_when_x_is_2_l1266_126697


namespace sum_of_inserted_numbers_l1266_126661

theorem sum_of_inserted_numbers (x y : ℝ) (h1 : x^2 = 2 * y) (h2 : 2 * y = x + 20) :
  x + y = 4 ∨ x + y = 17.5 :=
sorry

end sum_of_inserted_numbers_l1266_126661


namespace inscribed_circle_radius_l1266_126638

theorem inscribed_circle_radius (r : ℝ) (radius : ℝ) (angle_deg : ℝ): 
  radius = 6 ∧ angle_deg = 120 ∧ (∀ θ : ℝ, θ = 60) → r = 3 := 
by
  sorry

end inscribed_circle_radius_l1266_126638


namespace ellipse_focus_coordinates_l1266_126689

theorem ellipse_focus_coordinates (a b c : ℝ) (x1 y1 x2 y2 : ℝ) 
  (major_axis_length : 2 * a = 20) 
  (focal_relationship : c^2 = a^2 - b^2)
  (focus1_location : x1 = 3 ∧ y1 = 4) 
  (focus_c_calculation : c = Real.sqrt (x1^2 + y1^2)) :
  (x2 = -3 ∧ y2 = -4) := by
  sorry

end ellipse_focus_coordinates_l1266_126689


namespace new_avg_weight_l1266_126686

-- Definition of the conditions
def original_team_avg_weight : ℕ := 94
def original_team_size : ℕ := 7
def new_player_weight_1 : ℕ := 110
def new_player_weight_2 : ℕ := 60
def total_new_team_size : ℕ := original_team_size + 2

-- Computation of the total weight
def total_weight_original_team : ℕ := original_team_avg_weight * original_team_size
def total_weight_new_team : ℕ := total_weight_original_team + new_player_weight_1 + new_player_weight_2

-- Statement of the theorem
theorem new_avg_weight : total_weight_new_team / total_new_team_size = 92 := by
  -- Proof is omitted
  sorry

end new_avg_weight_l1266_126686


namespace total_snowfall_yardley_l1266_126621

theorem total_snowfall_yardley (a b c d : ℝ) (ha : a = 0.12) (hb : b = 0.24) (hc : c = 0.5) (hd : d = 0.36) :
  a + b + c + d = 1.22 :=
by
  sorry

end total_snowfall_yardley_l1266_126621


namespace max_n_value_l1266_126607

-- Define the arithmetic sequence
variable {a : ℕ → ℤ} (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)

-- Given conditions
variable (h1 : a 1 + a 3 + a 5 = 105)
variable (h2 : a 2 + a 4 + a 6 = 99)

-- Goal: Prove the maximum integer value of n is 10
theorem max_n_value (n : ℕ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 3 + a 5 = 105) (h2 : a 2 + a 4 + a 6 = 99) : n ≤ 10 → 
  (∀ m, (0 < m ∧ m ≤ n) → a (2 * m) ≥ 0) → n = 10 := 
sorry

end max_n_value_l1266_126607


namespace total_pencils_l1266_126699

def num_boxes : ℕ := 12
def pencils_per_box : ℕ := 17

theorem total_pencils : num_boxes * pencils_per_box = 204 := by
  sorry

end total_pencils_l1266_126699


namespace increasing_function_condition_l1266_126617

theorem increasing_function_condition (k : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k - 6) * x1 + (2 * k + 1) < (2 * k - 6) * x2 + (2 * k + 1)) ↔ (k > 3) :=
by
  -- To prove the statement, we would need to prove it in both directions.
  sorry

end increasing_function_condition_l1266_126617


namespace action_figure_price_l1266_126605

theorem action_figure_price (x : ℝ) (h1 : 2 + 4 * x = 30) : x = 7 :=
by
  -- The proof is provided here
  sorry

end action_figure_price_l1266_126605


namespace exposed_surface_area_equals_42_l1266_126644

-- Define the structure and exposed surface area calculations.
def surface_area_of_sculpture (layers : List Nat) : Nat :=
  (layers.headD 0 * 5) +  -- Top layer (5 faces exposed)
  (layers.getD 1 0 * 3 + layers.getD 1 0) +  -- Second layer
  (layers.getD 2 0 * 1 + layers.getD 2 0) +  -- Third layer
  (layers.getD 3 0 * 1) -- Bottom layer

-- Define the conditions
def number_of_layers : List Nat := [1, 4, 9, 6]

-- State the theorem
theorem exposed_surface_area_equals_42 :
  surface_area_of_sculpture number_of_layers = 42 :=
by
  sorry

end exposed_surface_area_equals_42_l1266_126644


namespace sampling_method_is_stratified_l1266_126631

-- Given conditions
def unit_population : ℕ := 500 + 1000 + 800
def elderly_ratio : ℕ := 5
def middle_aged_ratio : ℕ := 10
def young_ratio : ℕ := 8
def total_selected : ℕ := 230

-- Prove that the sampling method used is stratified sampling
theorem sampling_method_is_stratified :
  (500 + 1000 + 800 = unit_population) ∧
  (total_selected = 230) ∧
  (500 * 230 / unit_population = elderly_ratio) ∧
  (1000 * 230 / unit_population = middle_aged_ratio) ∧
  (800 * 230 / unit_population = young_ratio) →
  sampling_method = stratified_sampling :=
by
  sorry

end sampling_method_is_stratified_l1266_126631


namespace measure_of_angle_B_l1266_126654

theorem measure_of_angle_B (A B C a b c : ℝ) (h₁ : a = A.sin) (h₂ : b = B.sin) (h₃ : c = C.sin)
  (h₄ : (b - a) / (c + a) = c / (a + b)) :
  B = 2 * π / 3 :=
by
  sorry

end measure_of_angle_B_l1266_126654


namespace sun_volume_exceeds_moon_volume_by_387_cubed_l1266_126606

/-- Given Sun's distance to Earth is 387 times greater than Moon's distance to Earth. 
Given diameters:
- Sun's diameter: D_s
- Moon's diameter: D_m
Formula for volume of a sphere: V = (4/3) * pi * R^3
Derive that the Sun's volume exceeds the Moon's volume by 387^3 times. -/
theorem sun_volume_exceeds_moon_volume_by_387_cubed
  (D_s D_m : ℝ)
  (h : D_s = 387 * D_m) :
  (4/3) * Real.pi * (D_s / 2)^3 = 387^3 * (4/3) * Real.pi * (D_m / 2)^3 := by
  sorry

end sun_volume_exceeds_moon_volume_by_387_cubed_l1266_126606


namespace x_squared_plus_y_squared_l1266_126619

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^3 = 8) (h2 : x * y = 5) : 
  x^2 + y^2 = -6 := by
  sorry

end x_squared_plus_y_squared_l1266_126619


namespace cube_volume_of_surface_area_l1266_126655

theorem cube_volume_of_surface_area (S : ℝ) (V : ℝ) (a : ℝ) (h1 : S = 150) (h2 : S = 6 * a^2) (h3 : V = a^3) : V = 125 := by
  sorry

end cube_volume_of_surface_area_l1266_126655


namespace prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l1266_126647

theorem prop_P_subset_q_when_m_eq_1 :
  ∀ x : ℝ, ∀ m : ℝ, m = 1 → (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) ↔ (x ∈ {x | 0 ≤ x ∧ x ≤ 2}) := 
by sorry

theorem range_m_for_necessity_and_not_sufficiency :
  ∀ m : ℝ, (∀ x : ℝ, (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) → (x ∈ {x | 1 - m ≤ x ∧ x ≤ 1 + m})) ↔ (m ≥ 9) := 
by sorry

end prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l1266_126647


namespace percentage_not_drop_l1266_126627

def P_trip : ℝ := 0.40
def P_drop_given_trip : ℝ := 0.25
def P_not_drop : ℝ := 0.90

theorem percentage_not_drop :
  (1 - P_trip * P_drop_given_trip) = P_not_drop :=
sorry

end percentage_not_drop_l1266_126627


namespace min_value_of_sum_l1266_126670

theorem min_value_of_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2 * a + b) : a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end min_value_of_sum_l1266_126670


namespace equilibrium_proof_l1266_126662

noncomputable def equilibrium_constant (Γ_eq B_eq : ℝ) : ℝ :=
(Γ_eq ^ 3) / (B_eq ^ 3)

theorem equilibrium_proof (Γ_eq B_eq : ℝ) (K_c : ℝ) (B_initial : ℝ) (Γ_initial : ℝ)
  (hΓ : Γ_eq = 0.25) (hB : B_eq = 0.15) (hKc : K_c = 4.63) 
  (ratio : Γ_eq = B_eq + B_initial) (hΓ_initial : Γ_initial = 0) :
  equilibrium_constant Γ_eq B_eq = K_c ∧ 
  B_initial = 0.4 ∧ 
  Γ_initial = 0 := 
by
  sorry

end equilibrium_proof_l1266_126662


namespace sam_dimes_example_l1266_126656

theorem sam_dimes_example (x y : ℕ) (h₁ : x = 9) (h₂ : y = 7) : x + y = 16 :=
by 
  sorry

end sam_dimes_example_l1266_126656


namespace sqrt_three_cubes_l1266_126618

theorem sqrt_three_cubes : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := 
  sorry

end sqrt_three_cubes_l1266_126618


namespace rational_root_of_p_l1266_126698

noncomputable def p (n : ℕ) (x : ℚ) : ℚ :=
  x^n + (2 + x)^n + (2 - x)^n

theorem rational_root_of_p :
  ∀ n : ℕ, n > 0 → (∃ x : ℚ, p n x = 0) ↔ n = 1 := by
  sorry

end rational_root_of_p_l1266_126698


namespace polynomial_roots_and_coefficients_l1266_126663

theorem polynomial_roots_and_coefficients 
  (a b c d e : ℝ)
  (h1 : a = 2)
  (h2 : 256 * a + 64 * b + 16 * c + 4 * d + e = 0)
  (h3 : -81 * a + 27 * b - 9 * c + 3 * d + e = 0)
  (h4 : 625 * a + 125 * b + 25 * c + 5 * d + e = 0) :
  (b + c + d) / a = 151 := 
by
  sorry

end polynomial_roots_and_coefficients_l1266_126663


namespace no_infinite_non_constant_arithmetic_progression_with_powers_l1266_126628

theorem no_infinite_non_constant_arithmetic_progression_with_powers (a b : ℕ) (b_ge_2 : b ≥ 2) : 
  ¬ ∃ (f : ℕ → ℕ) (d : ℕ), (∀ n : ℕ, f n = (a^(b + n*d)) ∧ b ≥ 2) := sorry

end no_infinite_non_constant_arithmetic_progression_with_powers_l1266_126628


namespace initial_money_l1266_126691

def cost_of_game : Nat := 47
def cost_of_toy : Nat := 7
def number_of_toys : Nat := 3

theorem initial_money (initial_amount : Nat) (remaining_amount : Nat) :
  initial_amount = cost_of_game + remaining_amount →
  remaining_amount = number_of_toys * cost_of_toy →
  initial_amount = 68 := by
    sorry

end initial_money_l1266_126691


namespace circle_value_a_l1266_126641

noncomputable def represents_circle (a : ℝ) (x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0

theorem circle_value_a {a : ℝ} (h : ∀ x y : ℝ, represents_circle a x y) :
  a = -1 :=
by
  sorry

end circle_value_a_l1266_126641


namespace sine_shift_l1266_126650

variable (m : ℝ)

theorem sine_shift (h : Real.sin 5.1 = m) : Real.sin 365.1 = m :=
by
  sorry

end sine_shift_l1266_126650


namespace sum_constants_l1266_126614

def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x - 88

theorem sum_constants (a b c : ℝ) (h : ∀ x : ℝ, -4 * x^2 + 20 * x - 88 = a * (x + b)^2 + c) : 
  a + b + c = -70.5 :=
sorry

end sum_constants_l1266_126614


namespace june_1_friday_l1266_126652

open Nat

-- Define the days of the week as data type
inductive DayOfWeek : Type
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

open DayOfWeek

-- Define that June has 30 days
def june_days := 30

-- Hypotheses that June has exactly three Mondays and exactly three Thursdays
def three_mondays (d : DayOfWeek) : Prop := 
  ∃ days : Fin 30 → DayOfWeek, 
    (∀ n : Fin 30, days n = Monday → 3 ≤ n / 7) -- there are exactly three Mondays
  
def three_thursdays (d : DayOfWeek) : Prop := 
  ∃ days : Fin 30 → DayOfWeek, 
    (∀ n : Fin 30, days n = Thursday → 3 ≤ n / 7) -- there are exactly three Thursdays

-- Theorem to prove June 1 falls on a Friday given those conditions
theorem june_1_friday : ∀ (d : DayOfWeek), 
  three_mondays d → three_thursdays d → (d = Friday) :=
by
  sorry

end june_1_friday_l1266_126652


namespace mailing_ways_l1266_126611

-- Definitions based on the problem conditions
def countWays (letters mailboxes : ℕ) : ℕ := mailboxes^letters

-- The theorem to prove the mathematically equivalent proof problem
theorem mailing_ways (letters mailboxes : ℕ) (h_letters : letters = 3) (h_mailboxes : mailboxes = 4) : countWays letters mailboxes = 4^3 := 
by
  rw [h_letters, h_mailboxes]
  rfl

end mailing_ways_l1266_126611


namespace line_intersects_curve_l1266_126677

theorem line_intersects_curve (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ax₁ + 16 = x₁^3 ∧ ax₂ + 16 = x₂^3) →
  a = 12 :=
by
  sorry

end line_intersects_curve_l1266_126677


namespace lower_bound_fraction_sum_l1266_126659

open Real

theorem lower_bound_fraction_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  (1 / (3 * a) + 3 / b) ≥ 8 / 3 :=
by 
  sorry

end lower_bound_fraction_sum_l1266_126659


namespace percentage_increase_l1266_126683

theorem percentage_increase (A B : ℝ) (y : ℝ) (h : A > B) (h1 : B > 0) (h2 : C = A + B) (h3 : C = (1 + y / 100) * B) : y = 100 * (A / B) := 
sorry

end percentage_increase_l1266_126683


namespace directrix_of_parabola_l1266_126600

theorem directrix_of_parabola (y x : ℝ) (h : y = 4 * x^2) : y = - (1 / 16) :=
sorry

end directrix_of_parabola_l1266_126600


namespace numerator_multiple_of_prime_l1266_126616

theorem numerator_multiple_of_prime (n : ℕ) (hp : Nat.Prime (3 * n + 1)) :
  (2 * n - 1) % (3 * n + 1) = 0 :=
sorry

end numerator_multiple_of_prime_l1266_126616


namespace hunting_dog_catches_fox_l1266_126682

theorem hunting_dog_catches_fox :
  ∀ (V_1 V_2 : ℝ) (t : ℝ),
  V_1 / V_2 = 10 ∧ 
  t * V_2 = (10 / (V_2) + t) →
  (V_1 * t) = 100 / 9 :=
by
  intros V_1 V_2 t h
  sorry

end hunting_dog_catches_fox_l1266_126682


namespace four_consecutive_integers_divisible_by_24_l1266_126635

noncomputable def product_of_consecutive_integers (n : ℤ) : ℤ :=
  n * (n + 1) * (n + 2) * (n + 3)

theorem four_consecutive_integers_divisible_by_24 (n : ℤ) :
  24 ∣ product_of_consecutive_integers n :=
by
  sorry

end four_consecutive_integers_divisible_by_24_l1266_126635


namespace loss_per_metre_l1266_126666

-- Definitions for given conditions
def TSP : ℕ := 15000           -- Total Selling Price
def CPM : ℕ := 40              -- Cost Price per Metre
def TMS : ℕ := 500             -- Total Metres Sold

-- Definition for the expected Loss Per Metre
def LPM : ℕ := 10

-- Statement to prove that the loss per metre is 10
theorem loss_per_metre :
  (CPM * TMS - TSP) / TMS = LPM :=
by
sorry

end loss_per_metre_l1266_126666


namespace smallest_four_digit_divisible_by_25_l1266_126674

theorem smallest_four_digit_divisible_by_25 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 25 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 25 = 0 → n ≤ m := by
  -- Prove that the smallest four-digit number divisible by 25 is 1000
  sorry

end smallest_four_digit_divisible_by_25_l1266_126674


namespace min_value_x_plus_one_over_x_plus_two_l1266_126602

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1 / (x + 2) ∧ y ≥ 0 := 
sorry

end min_value_x_plus_one_over_x_plus_two_l1266_126602


namespace percentage_increase_l1266_126633

def x (y: ℝ) : ℝ := 1.25 * y
def z : ℝ := 250
def total_amount (x y z : ℝ) : ℝ := x + y + z

theorem percentage_increase (y: ℝ) : (total_amount (x y) y z = 925) → ((y - z) / z) * 100 = 20 := by
  sorry

end percentage_increase_l1266_126633


namespace eq_satisfied_for_all_y_l1266_126609

theorem eq_satisfied_for_all_y (x : ℝ) : 
  (∀ y: ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by
  sorry

end eq_satisfied_for_all_y_l1266_126609


namespace find_m_and_other_root_l1266_126645

theorem find_m_and_other_root (m x_2 : ℝ) :
  (∃ (x_1 : ℝ), x_1 = -1 ∧ x_1^2 + m * x_1 - 5 = 0) →
  m = -4 ∧ ∃ (x_2 : ℝ), x_2 = 5 ∧ x_2^2 + m * x_2 - 5 = 0 :=
by
  sorry

end find_m_and_other_root_l1266_126645


namespace mark_weekly_leftover_l1266_126608

def initial_hourly_wage := 40
def raise_percentage := 5 / 100
def daily_hours := 8
def weekly_days := 5
def old_weekly_bills := 600
def personal_trainer_cost := 100

def new_hourly_wage := initial_hourly_wage * (1 + raise_percentage)
def weekly_hours := daily_hours * weekly_days
def weekly_earnings := new_hourly_wage * weekly_hours
def new_weekly_expenses := old_weekly_bills + personal_trainer_cost
def leftover_per_week := weekly_earnings - new_weekly_expenses

theorem mark_weekly_leftover : leftover_per_week = 980 := by
  sorry

end mark_weekly_leftover_l1266_126608


namespace find_g_neg_6_l1266_126664

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end find_g_neg_6_l1266_126664


namespace jason_current_money_l1266_126625

/-- Definition of initial amounts and earnings -/
def fred_initial : ℕ := 49
def jason_initial : ℕ := 3
def fred_current : ℕ := 112
def jason_earned : ℕ := 60

/-- The main theorem -/
theorem jason_current_money : jason_initial + jason_earned = 63 := 
by
  -- proof omitted for this example
  sorry

end jason_current_money_l1266_126625


namespace find_p_q_l1266_126639

theorem find_p_q (p q : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 + p * x + q)
  (h_min : ∀ x, x = q → f x = (p + q)^2) : 
  (p = 0 ∧ q = 0) ∨ (p = -1 ∧ q = 1 / 2) :=
by
  sorry

end find_p_q_l1266_126639


namespace planar_molecules_l1266_126648

structure Molecule :=
  (name : String)
  (formula : String)
  (is_planar : Bool)

def propylene : Molecule := 
  { name := "Propylene", formula := "C3H6", is_planar := False }

def vinyl_chloride : Molecule := 
  { name := "Vinyl Chloride", formula := "C2H3Cl", is_planar := True }

def benzene : Molecule := 
  { name := "Benzene", formula := "C6H6", is_planar := True }

def toluene : Molecule := 
  { name := "Toluene", formula := "C7H8", is_planar := False }

theorem planar_molecules : 
  (vinyl_chloride.is_planar = True) ∧ (benzene.is_planar = True) := 
by
  sorry

end planar_molecules_l1266_126648


namespace number_of_ways_to_take_pieces_l1266_126675

theorem number_of_ways_to_take_pieces : 
  (Nat.choose 6 4) = 15 := 
by
  sorry

end number_of_ways_to_take_pieces_l1266_126675


namespace graph_passes_through_point_l1266_126695

theorem graph_passes_through_point (a : ℝ) (x y : ℝ) (h : a < 0) : (1 - a)^0 - 1 = -1 :=
by
  sorry

end graph_passes_through_point_l1266_126695
