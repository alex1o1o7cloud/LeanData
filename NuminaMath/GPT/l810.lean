import Mathlib

namespace shaded_region_perimeter_l810_810665

-- Define the radius and central angle based on given conditions
def radius : ℝ := 7
def central_angle : ℝ := 135 * (Real.pi / 180) -- converting degrees to radians

-- Define the circumference of the full circle
def circumference : ℝ := 2 * Real.pi * radius

-- Calculate the arc length for 5/8 of the circumference
def arc_length : ℝ := (5 / 8) * circumference

-- Calculate the perimeter of the shaded region
def shaded_perimeter : ℝ := 2 * radius + arc_length

-- Our main theorem stating the perimeter of the shaded region
theorem shaded_region_perimeter : shaded_perimeter = 14 + (35 / 4) * Real.pi := by
  sorry

end shaded_region_perimeter_l810_810665


namespace paolo_sevilla_birthday_l810_810343

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l810_810343


namespace friends_attended_birthday_l810_810355

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l810_810355


namespace line_is_x_axis_l810_810638

theorem line_is_x_axis (A B C : ℝ) (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : A = 0 ∧ B ≠ 0 ∧ C = 0 :=
by sorry

end line_is_x_axis_l810_810638


namespace barbell_cost_l810_810286

theorem barbell_cost (num_barbells : ℤ) (total_money_given : ℤ) 
  (change_received : ℤ) (total_cost : ℤ) (each_barbell_cost : ℤ) 
  (h1 : num_barbells = 3) (h2 : total_money_given = 850) 
  (h3 : change_received = 40) (h4 : total_cost = total_money_given - change_received)
  (h5 : each_barbell_cost = total_cost / num_barbells) 
  : each_barbell_cost = 270 :=
by
  rw [h2, h3] at h4
  rw [← h4, h1] at h5
  exact calc 
    each_barbell_cost = (total_money_given - change_received) / num_barbells : h5
                    ... = (850 - 40) / 3 : by rw [h2, h3, h1]
                    ... = 810 / 3 : rfl
                    ... = 270 : rfl

end barbell_cost_l810_810286


namespace greatest_x_lcm_l810_810814

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l810_810814


namespace log_diff_l810_810472

theorem log_diff : (Real.log (12:ℝ) / Real.log (2:ℝ)) - (Real.log (3:ℝ) / Real.log (2:ℝ)) = 2 := 
by
  sorry

end log_diff_l810_810472


namespace incorrect_inequality_l810_810215

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬ (-2 * a > -2 * b) :=
by {
  have h_neg_mul : -2 * a < -2 * b,
  linarith,
  exact h_neg_mul,
  sorry
}

end incorrect_inequality_l810_810215


namespace Time_on_Bus_l810_810461

def XiaoMingTravel : Type := unit

constants 
  (T_subway : ℕ)
  (T_bus_alone : ℕ)
  (T_total : ℕ)
  (T_transfer : ℕ)

axiom T_subway_def : T_subway = 30
axiom T_bus_alone_def : T_bus_alone = 50
axiom T_total_def : T_total = 40
axiom T_transfer_def : T_transfer = 6

theorem Time_on_Bus (T_bus : ℕ) : T_bus = 10 :=
by
  -- Using the given conditions and information in a proof
  sorry

end Time_on_Bus_l810_810461


namespace greatest_x_lcm_105_l810_810794

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l810_810794


namespace cost_of_each_barbell_l810_810282

theorem cost_of_each_barbell (total_given change_received total_barbells : ℕ)
  (h1 : total_given = 850)
  (h2 : change_received = 40)
  (h3 : total_barbells = 3) :
  (total_given - change_received) / total_barbells = 270 :=
by
  sorry

end cost_of_each_barbell_l810_810282


namespace final_result_l810_810008

-- Given conditions: x ≠ 0 and n sequences
variables {x : ℝ} (n : ℕ)
hypothesis h : x ≠ 0

-- The final result after n sequences
theorem final_result : (∃ y : ℝ, y = x^( (-4)^n )) :=
sorry

end final_result_l810_810008


namespace estimation_problems_l810_810939

noncomputable def average_root_cross_sectional_area (x : list ℝ) : ℝ :=
  (list.sum x) / (list.length x)

noncomputable def average_volume (y : list ℝ) : ℝ :=
  (list.sum y) / (list.length y)

noncomputable def sample_correlation_coefficient (x y : list ℝ) : ℝ :=
  let n := list.length x
      avg_x := average_root_cross_sectional_area x
      avg_y := average_volume y
      sum_xy := (list.zip_with (*) x y).sum
      sum_x2 := (x.map (λ xi, xi * xi)).sum
      sum_y2 := (y.map (λ yi, yi * yi)).sum
  in (sum_xy - n * avg_x * avg_y) / (real.sqrt ((sum_x2 - n * avg_x^2) * (sum_y2 - n * avg_y^2)))

theorem estimation_problems :
  let x := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
      y := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
      X := 186
  in
    average_root_cross_sectional_area x = 0.06 ∧
    average_volume y = 0.39 ∧
    abs (sample_correlation_coefficient x y - 0.97) < 0.01 ∧
    (average_volume y / average_root_cross_sectional_area x) * X = 1209 :=
by
  sorry

end estimation_problems_l810_810939


namespace sin_solution_set_l810_810423

open Real

theorem sin_solution_set (x : ℝ) : 
  (3 * sin x = 1 + cos (2 * x)) ↔ ∃ k : ℤ, x = k * π + (-1) ^ k * (π / 6) :=
by
  sorry

end sin_solution_set_l810_810423


namespace petyas_claim_is_false_l810_810376

theorem petyas_claim_is_false (S T : ℝ) 
  (h1 : (S / 8) ≤ (T / 2)) 
  (h2 : (5 * T / 2) ≤ (S / 2)) : 
  false :=
by 
  have h3 : S ≤ 4 * T,
  { linarith, },
  have h4 : 5 * T ≤ S,
  { linarith, },
  have h5 : 5 * T ≤ 4 * T,
  { linarith, },
  have h6 : ¬ (5 * T ≤ 4 * T),
  { linarith, },
  contradiction

end petyas_claim_is_false_l810_810376


namespace num_of_valid_three_digit_numbers_l810_810632

def valid_three_digit_numbers : ℕ :=
  let valid_numbers : List (ℕ × ℕ × ℕ) :=
    [(2, 3, 4), (4, 6, 8)]
  valid_numbers.length

theorem num_of_valid_three_digit_numbers :
  valid_three_digit_numbers = 2 :=
by
  sorry

end num_of_valid_three_digit_numbers_l810_810632


namespace friends_attended_birthday_l810_810360

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l810_810360


namespace expand_product_l810_810560

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := 
by 
  sorry

end expand_product_l810_810560


namespace g_zero_eq_zero_l810_810750

noncomputable def g : ℝ → ℝ :=
  sorry

axiom functional_equation (a b : ℝ) :
  g (3 * a + 2 * b) + g (3 * a - 2 * b) = 2 * g (3 * a) + 2 * g (2 * b)

theorem g_zero_eq_zero : g 0 = 0 :=
by
  let a := 0
  let b := 0
  have eqn := functional_equation a b
  sorry

end g_zero_eq_zero_l810_810750


namespace five_student_committees_from_eight_l810_810127

theorem five_student_committees_from_eight : nat.choose 8 5 = 56 := by
  sorry

end five_student_committees_from_eight_l810_810127


namespace f_2010_l810_810154

noncomputable def f (a b α β x : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β)

theorem f_2010 (a b α β : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : α ≠ 0) (h₄ : β ≠ 0)
  (h₅ : f a b α β 2009 = -1) : f a b α β 2010 = 1 :=
by
  sorry

end f_2010_l810_810154


namespace quadratic_solution_unique_l810_810392

theorem quadratic_solution_unique (b : ℝ) (hb : b ≠ 0) (hdisc : 30 * 30 - 4 * b * 10 = 0) :
  ∃ x : ℝ, bx ^ 2 + 30 * x + 10 = 0 ∧ x = -2 / 3 :=
by
  sorry

end quadratic_solution_unique_l810_810392


namespace decimal_expansion_of_13_over_625_l810_810114

theorem decimal_expansion_of_13_over_625 : (13 : ℚ) / 625 = 0.0208 :=
by sorry

end decimal_expansion_of_13_over_625_l810_810114


namespace cost_of_each_barbell_l810_810283

theorem cost_of_each_barbell (total_given change_received total_barbells : ℕ)
  (h1 : total_given = 850)
  (h2 : change_received = 40)
  (h3 : total_barbells = 3) :
  (total_given - change_received) / total_barbells = 270 :=
by
  sorry

end cost_of_each_barbell_l810_810283


namespace ratio_A_B_l810_810842

variable (A B C : ℕ)

theorem ratio_A_B 
  (h1: A + B + C = 98) 
  (h2: B = 30) 
  (h3: (B : ℚ) / C = 5 / 8) 
  : (A : ℚ) / B = 2 / 3 :=
sorry

end ratio_A_B_l810_810842


namespace bricks_needed_for_wall_l810_810879

noncomputable def number_of_bricks_needed
    (brick_length : ℕ)
    (brick_width : ℕ)
    (brick_height : ℕ)
    (wall_length_m : ℕ)
    (wall_height_m : ℕ)
    (wall_thickness_cm : ℕ) : ℕ :=
  let wall_length_cm := wall_length_m * 100
  let wall_height_cm := wall_height_m * 100
  let wall_volume := wall_length_cm * wall_height_cm * wall_thickness_cm
  let brick_volume := brick_length * brick_width * brick_height
  (wall_volume + brick_volume - 1) / brick_volume -- This rounds up to the nearest whole number.

theorem bricks_needed_for_wall : number_of_bricks_needed 5 11 6 8 6 2 = 2910 :=
sorry

end bricks_needed_for_wall_l810_810879


namespace greatest_x_lcm_l810_810817

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l810_810817


namespace T_10_mod_5_l810_810573

def T (n : ℕ) : ℕ :=
  -- Placeholder definition for T, the real definition would be the recursive generation
  sorry

-- Main theorem statement
theorem T_10_mod_5 : T 10 % 5 = 3 :=
by
  -- Define the sequences and constraints, real constants would follow the recursive rules outlined above
  let c : ℕ → ℕ := sorry
  let d : ℕ → ℕ := sorry
  let c1 : ℕ → ℕ := λ n, if n = 1 then 1 else sorry
  let c2 : ℕ → ℕ := λ n, if n = 1 then 0 else sorry
  let d1 : ℕ → ℕ := λ n, if n = 1 then 1 else sorry
  let d2 : ℕ → ℕ := λ n, if n = 1 then 0 else sorry

  -- Recursive relations
  have h1 : ∀ n, c1 (n+1) = d1 n + d2 n := sorry
  have h2 : ∀ n, c2 (n+1) = c1 n := sorry
  have h3 : ∀ n, d1 (n+1) = c1 n + c2 n := sorry
  have h4 : ∀ n, d2 (n+1) = d1 n := sorry

  -- Total sequences calculation
  let T := (λ n, c1 n + c2 n + d1 n + d2 n)
  
  -- Base case
  have T1 : T 1 = 2 := by simp only [c1, c2, d1, d2]; exact rfl

  -- Final proof, skipping intermediate computations
  -- T(10) = 58 and 58 % 5 = 3
  have T10 : T 10 = 58 := sorry
  show T 10 % 5 = 3, from by rw [T10]; norm_num

end T_10_mod_5_l810_810573


namespace inverse_functions_symmetry_l810_810635

theorem inverse_functions_symmetry {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1) :
  ∀ x : ℝ, a ^ x = exp (log a * x) ∧ (∀ y : ℝ, y > 0 → log a (a ^ y) = y) →
    ∀ x : ℝ, a ^ x = exp (log a * x) ∧ log a y = x ↔ log a y = x ∧ y = a ^ x :=
begin
  sorry
end

end inverse_functions_symmetry_l810_810635


namespace base_p_prime_values_zero_l810_810949

theorem base_p_prime_values_zero :
  (∀ p : ℕ, p.Prime → 2008 * p^3 + 407 * p^2 + 214 * p + 226 = 243 * p^2 + 382 * p + 471 → False) :=
by
  sorry

end base_p_prime_values_zero_l810_810949


namespace line_intersects_midpoint_l810_810763

theorem line_intersects_midpoint (c : ℤ) : 
  (∃x y : ℤ, 2 * x - y = c ∧ x = (1 + 5) / 2 ∧ y = (3 + 11) / 2) → c = -1 := by
  sorry

end line_intersects_midpoint_l810_810763


namespace prove_payment_prove_cost_effectiveness_more_effective_plan_l810_810381

-- Define the cost of soccer ball and jump rope
def soccer_ball_price := 100
def jump_rope_price := 20

-- Define the payment functions for stores A and B
def payment_store_A (x : ℕ) : ℕ := 20 * x + 2400
def payment_store_B (x : ℕ) : ℕ := 18 * x + 2700

-- Define the comparison function for cost-effectiveness for x=100
def cost_effective_store (x : ℕ) : string :=
  if payment_store_A x < payment_store_B x then
    "store A"
  else
    "store B"

-- Define the more cost-effective purchasing plan and calculate the cost for x=100
noncomputable def more_cost_effective_plan (x : ℕ) : ℕ :=
  let cost_A := 3000 -- Cost for 30 soccer balls and 30 free jump ropes from store A
  let cost_B := 18 * (x - 30) -- Cost for remaining jump ropes from store B at 90% discount
  cost_A + cost_B

-- The statement for the proof problems without providing the proof itself
theorem prove_payment (x : ℕ) (h : x > 30) :
  payment_store_A x = 20 * x + 2400 ∧ payment_store_B x = 18 * x + 2700 := by sorry

theorem prove_cost_effectiveness (h : 100 > 30) :
  (cost_effective_store 100 = "store A") ∧
  (payment_store_A 100 = 4400) ∧
  (payment_store_B 100 = 4500) := by sorry

theorem more_effective_plan (h : 100 > 30) :
  more_cost_effective_plan 100 = 4260 := by sorry

end prove_payment_prove_cost_effectiveness_more_effective_plan_l810_810381


namespace car_gas_cost_l810_810957

def car_mpg_city : ℝ := 30
def car_mpg_highway : ℝ := 40
def city_distance_one_way : ℝ := 60
def highway_distance_one_way : ℝ := 200
def gas_cost_per_gallon : ℝ := 3
def total_gas_cost : ℝ := 42

theorem car_gas_cost :
  (city_distance_one_way / car_mpg_city * 2 + highway_distance_one_way / car_mpg_highway * 2) * gas_cost_per_gallon = total_gas_cost := 
  sorry

end car_gas_cost_l810_810957


namespace mod_inv_sum_l810_810861

-- Define modular inverse function
def mod_inv (a n : ℕ) : ℕ := let (g, x, y) := xgcd a n in ((x % n + n) % n)

-- Definitions for conditions
def five_inv_mod_31 : ℕ := mod_inv 5 31
def twenty_five_inv_mod_31 : ℕ := mod_inv (5^2) 31

-- Main statement to prove
theorem mod_inv_sum : (five_inv_mod_31 + twenty_five_inv_mod_31) % 31 = 26 := 
  by sorry

end mod_inv_sum_l810_810861


namespace table_tennis_arrangements_l810_810047

-- Defining the conditions
def total_players : ℕ := 5
def experienced_players : ℕ := 2
def new_players : ℕ := 3
def positions : ℕ := 3

-- Main theorem
theorem table_tennis_arrangements :
  (∃ (arrangements : ℕ), arrangements = 57) :=
begin
  -- proof steps skipped
  sorry
end

end table_tennis_arrangements_l810_810047


namespace no_cell_with_sum_2018_l810_810891

theorem no_cell_with_sum_2018 : ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 4900 → (5 * x = 2018 → false) := 
by
  intros x hx
  have h_bound : 1 ≤ x ∧ x ≤ 4900 := hx
  sorry

end no_cell_with_sum_2018_l810_810891


namespace probability_and_expected_value_of_area_ratios_l810_810732

-- Triangle definitions and points
variables {A B C P Q M : Type}
variables (AP PB AQ QC PM AM PQM PQ PQM ABC : ℝ)
axioms
  (hAP_PB : AP = PB * (1 / 3))
  (hAQ_QC : AQ = QC / 2)
  (hABC : ABC > 0)  -- Area of triangle ABC is positive
  (hRandomM : ∃ M:BC, true) -- M is chosen randomly on the side BC

-- Non-computable definitions required
noncomputable def area_triangle_ABC : ℝ := ABC
noncomputable def area_triangle_PQM : ℝ := PQM

noncomputable def probability_area_Ratio : ℝ := 5 / 6
noncomputable def expected_value_ratio : ℝ := 1 / 4

-- Proof statement
theorem probability_and_expected_value_of_area_ratios :
  (∃ P Q M:Type, hAP_PB ∧ hAQ_QC ∧ PQM ≤ ABC / 6 → probability_area_Ratio = 5 / 6) ∧
  (∃ P Q M:Type, hAP_PB ∧ hAQ_QC ∧ expected_value_ratio = 1 / 4) :=
begin
  sorry  -- Proof
end

end probability_and_expected_value_of_area_ratios_l810_810732


namespace f_prime_at_1_l810_810216

def f (x : ℝ) (fp1 : ℝ) : ℝ := (1/3) * x^3 - fp1 * x^2 + x + 5

theorem f_prime_at_1 {fp1 : ℝ} : 
  (∀ x : ℝ, derivative (λ y, f y fp1) x = x^2 - 2*fp1*x + 1) → 
  fp1 = 2/3 := 
by
  intro h 
  have key := h 1
  rw [key, derivative] at key
  sorry

end f_prime_at_1_l810_810216


namespace line_properties_l810_810912

structure Point3D := (x y z : ℝ)

def direction_vector (p1 p2 : Point3D) : Point3D :=
{ x := p2.x - p1.x,
  y := p2.y - p1.y,
  z := p2.z - p1.z }

def parametric_line (p : Point3D) (d : Point3D) (t : ℝ) : Point3D :=
{ x := p.x + t * d.x,
  y := p.y + t * d.y,
  z := p.z + t * d.z }

noncomputable def find_y_coordinate (x_val : ℝ) (p : Point3D) (d : Point3D) : ℝ :=
let t := (x_val - p.x) / d.x in
(p.y + t * d.y)

noncomputable def intersection_with_z_zero_plane (p : Point3D) (d : Point3D) : Point3D :=
let t := - p.z / d.z in
parametric_line p d t

theorem line_properties (p1 p2 : Point3D) (h : p1 = ⟨1, 3, 2⟩ ∧ p2 = ⟨4, 3, -1⟩) :
  (find_y_coordinate 7 p1 (direction_vector p1 p2) = 3) ∧
  (intersection_with_z_zero_plane p1 (direction_vector p1 p2) = ⟨3, 3, 0⟩) :=
begin
  sorry
end

end line_properties_l810_810912


namespace diamond_value_l810_810173

def diamond (a b : ℝ) : ℝ := sorry

axiom diamond_property1 (a b : ℝ) : 
  (a * b) ∈ ℝ ∧ (a(b * b))

axiom diamond_comm (a : ℝ) : 
  (a ∈ ℝ )
  
axiom diamond_id (a : ℝ) : 
  1 ∈ ℝ ∧ 1

theorem diamond_value : 
  diamond (1/2) (4/3) = 2/3 := 
  by
  sorry

end diamond_value_l810_810173


namespace sum_of_permutations_of_3_digits_l810_810882

theorem sum_of_permutations_of_3_digits :
  let digits := [2, 4, 5] in
  let permutations := digits.permutations.map (λ l, l.foldl (λ n d, 10 * n + d) 0) in
  permutations.sum = 2442 :=
by
  sorry

end sum_of_permutations_of_3_digits_l810_810882


namespace apple_cost_correct_l810_810436

noncomputable def apple_cost := 0.80

theorem apple_cost_correct (total_apples total_oranges : ℕ) (cost_orange total_earnings : ℝ) (remaining_apples remaining_oranges : ℕ)
    (total_apples = 50) (total_oranges = 40) (cost_orange = 0.50) (total_earnings = 49) (remaining_apples = 10) (remaining_oranges = 6) :
    let sold_apples := total_apples - remaining_apples
    let sold_oranges := total_oranges - remaining_oranges
    let earnings_oranges := sold_oranges * cost_orange
    let earnings_apples := total_earnings - earnings_oranges
    let cost_apple := earnings_apples / sold_apples
in cost_apple = apple_cost :=
by {
    sorry,
}

end apple_cost_correct_l810_810436


namespace cyclic_quadrilateral_l810_810853

open Classical

variables {α : Type*} [MetricSpace α]

structure Circle (α : Type*) [MetricSpace α] :=
(center : α)
(radius : ℝ)

noncomputable def isTangential (P : α) (c₁ c₂ : Circle α) : Prop := 
  dist c₁.center P = c₁.radius ∧ dist c₂.center P = c₂.radius

noncomputable def isChord (P₁ P₂ : α) (c : Circle α) : Prop :=
  dist c.center P₁ ≤ c.radius ∧ dist c.center P₂ ≤ c.radius

noncomputable def isCyclicQuad (A B C D : α) : Prop :=
  ∃ (O : α) (r : ℝ), dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

variables {α}

theorem cyclic_quadrilateral
  (M P A B H : α) (ω₁ ω₂ : Circle α)
  (h_int : dist M ω₁.center ≤ ω₁.radius ∧ dist M ω₂.center ≤ ω₂.radius)
  (h_P : dist M P = dist P H)
  (h_tan_1 : isTangential M ω₁ ω₂)
  (h_chord_1 : isChord M A ω₁)
  (h_tan_2 : isTangential M ω₂ ω₁)
  (h_chord_2 : isChord M B ω₂) :
  isCyclicQuad M A H B :=
sorry

end cyclic_quadrilateral_l810_810853


namespace three_distinguishable_coins_outcomes_l810_810643

theorem three_distinguishable_coins_outcomes : 
  let coin_1_outcomes := 2 
  let coin_2_outcomes := 2 
  let coin_3_outcomes := 2 
  in coin_1_outcomes * coin_2_outcomes * coin_3_outcomes = 8 := 
by 
  let coin_1_outcomes := 2 
  let coin_2_outcomes := 2 
  let coin_3_outcomes := 2 
  calc
    2 * 2 * 2 = 4 * 2 : by sorry
         ... = 8 : by sorry

end three_distinguishable_coins_outcomes_l810_810643


namespace explicit_formula_and_tangent_line_l810_810713

noncomputable def f (x a : ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

theorem explicit_formula_and_tangent_line :
  (∃ a : ℝ, (∀ x, deriv (f x a) x = 0 → x = 3) 
              ∧ f 1 3 = 16 
              ∧ (∀ x, deriv (f x 3) x = 6 * x^2 - 24 * x + 18) 
              ∧ (f 1 3 = 16) 
              ∧ (deriv (f x 3) 1 = 0) ) →
  (∃ (fx : ℝ → ℝ) (y : ℝ),
    (fx = (λ x, 2 * x^3 - 12 * x^2 + 18 * x + 8) 
    ∧ y = 16) ) :=
begin
  sorry
end

end explicit_formula_and_tangent_line_l810_810713


namespace square_area_from_diagonal_l810_810502

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  (let s := d / Real.sqrt 2 in s * s) = 144 := by
sorry

end square_area_from_diagonal_l810_810502


namespace num_five_student_committees_l810_810124

theorem num_five_student_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 5) : choose n k = 56 :=
by
  rw [h_n, h_k]
  -- rest of the proof would go here
  sorry

end num_five_student_committees_l810_810124


namespace valid_y_values_l810_810520

def valid_triangle_sides (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_valid_y (y : ℕ) : Prop :=
  valid_triangle_sides 8 11 (y^2 - 1)

theorem valid_y_values : {y : ℕ // is_valid_y y} = {{y : ℕ // y = 3} ∪ {y : ℕ // y = 4}} :=
by
  sorry

end valid_y_values_l810_810520


namespace cyclic_quadrilateral_solutions_l810_810549

theorem cyclic_quadrilateral_solutions (a b c d : ℤ) :
  let s := (a + b + c + d) / 2 in
  (∃ (s : ℤ), s = (a + b + c + d) / 2 ∧
  ∃ (T : ℤ), T = (s - a) * (s - b) * (s - c) * (s - d) ∧
  √(T) = a + b + c + d) →
  List.mem (a, b, c, d) [(4, 4, 4, 4), (8, 5, 5, 2), (14, 6, 5, 5), (6, 6, 3, 3)] :=
begin
  intro h,
  cases h with s hs,
  sorry
end

end cyclic_quadrilateral_solutions_l810_810549


namespace distinct_collections_engineering_l810_810324

theorem distinct_collections_engineering : 
  let letters := "ENGINEERING".toList in
  let vowels := ['E', 'E', 'E', 'E', 'E'] in
  let consonants := ['N', 'N', 'G', 'G', 'G', 'R', 'I'] in
  (choose vowels 3) * (choose consonants 3) = 13 :=
by
  admit

end distinct_collections_engineering_l810_810324


namespace smallest_positive_integer_divisible_by_10_11_12_13_l810_810568

theorem smallest_positive_integer_divisible_by_10_11_12_13 : 
  ∃ n : ℕ, n > 0 ∧ (∀ d ∈ {10, 11, 12, 13}, d ∣ n) ∧ n = 8580 :=
by
  use 8580
  split
  · -- proof that 8580 is positive
    exact Nat.zero_lt_succ 8579
  split
  · -- proof that 10, 11, 12, and 13 divide 8580
    intros d hd
    fin_cases hd
    · exact dvd.intro 858 "8580 = 858 * 10"
    · exact dvd.intro 780 "8580 = 780 * 11"
    · exact dvd.intro 715 "8580 = 715 * 12"
    · exact dvd.intro 660 "8580 = 660 * 13"
  · -- proof that n = 8580
    refl

end smallest_positive_integer_divisible_by_10_11_12_13_l810_810568


namespace dog_catches_rabbit_in_4_minutes_l810_810090

def dog_speed_mph : ℝ := 24
def rabbit_speed_mph : ℝ := 15
def rabbit_head_start : ℝ := 0.6

theorem dog_catches_rabbit_in_4_minutes : 
  (∃ t : ℝ, t > 0 ∧ 0.4 * t = 0.25 * t + 0.6) → ∃ t : ℝ, t = 4 :=
sorry

end dog_catches_rabbit_in_4_minutes_l810_810090


namespace tens_digit_of_sum_l810_810872

-- Define periodic pattern for the last two digits of powers of 2
def periodic_tens_digit_of_2 (n : ℕ) : ℕ :=
  let cycle := [76, 52, 04, 08, 16, 32, 64, 28, 56, 12, 24, 48, 96, 92, 84, 68, 36, 72, 44, 88] in
  (cycle[(n % 20)] / 10) % 10

-- Define tens digit for powers of 5
def tens_digit_of_power_of_5 (n : ℕ) : ℕ :=
  if n = 0 then 1 else 5

theorem tens_digit_of_sum (n m : ℕ) : (n % 20) = 0 → (m ≥ 2) → ((periodic_tens_digit_of_2 n + tens_digit_of_power_of_5 m) % 10 = 9) :=
by
  intros h₁ h₂
  have hd1 : periodic_tens_digit_of_2 n = 7, from sorry
  have hd2 : tens_digit_of_power_of_5 m = 2, from sorry
  have hsum := hd1 + hd2
  have h10 : (hsum % 10) = 9, from sorry
  exact h10

end tens_digit_of_sum_l810_810872


namespace tree_volume_estimation_proof_l810_810934

noncomputable def average_root_cross_sectional_area (x_i : list ℝ) := (x_i.sum) / (x_i.length)
noncomputable def average_volume (y_i : list ℝ) := (y_i.sum) / (y_i.length)
noncomputable def correlation_coefficient (x_i y_i : list ℝ) : ℝ :=
  let n := x_i.length in
  let x_bar := average_root_cross_sectional_area x_i in
  let y_bar := average_volume y_i in
  let numerator := (list.zip x_i y_i).sum (λ ⟨x, y⟩, (x - x_bar) * (y - y_bar)) in
  let denominator_x := (x_i.sum (λ x, (x - x_bar)^2)) in
  let denominator_y := (y_i.sum (λ y, (y - y_bar)^2)) in
  numerator / ((denominator_x * denominator_y).sqrt)

noncomputable def total_volume_estimate (total_area avg_y avg_x : ℝ) := (avg_y / avg_x) * total_area

theorem tree_volume_estimation_proof :
  let x_i := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06] in
  let y_i := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40] in
  let total_area := 186 in
  average_root_cross_sectional_area x_i = 0.06 ∧
  average_volume y_i = 0.39 ∧
  correlation_coefficient x_i y_i ≈ 0.97 ∧
  total_volume_estimate total_area 0.39 0.06 = 1209 :=
by
  sorry

end tree_volume_estimation_proof_l810_810934


namespace clothing_profit_l810_810917

theorem clothing_profit 
  (purchase_cost : ℕ) (selling_price : ℕ) 
  (daily_sales : ℕ) (increase_in_sales_per_yuan : ℕ) 
  (target_profit : ℕ) (price_reduction : ℕ) :
  purchase_cost = 80 →
  selling_price = 120 →
  daily_sales = 30 →
  increase_in_sales_per_yuan = 3 →
  target_profit = 1800 →
  
  let profit_per_piece (x : ℕ) := selling_price - purchase_cost - x in
  let daily_increase_in_sales (x : ℕ) := increase_in_sales_per_yuan * x in
  let total_sales (x : ℕ) := daily_sales + daily_increase_in_sales x in
  let daily_profit (x : ℕ) := profit_per_piece x * total_sales x in
  
  daily_profit price_reduction = target_profit → 
  price_reduction = 20 :=
begin
  intros,
  sorry
end

end clothing_profit_l810_810917


namespace band_row_lengths_l810_810006

theorem band_row_lengths : 
  let divisors := { n : ℕ | n ∣ 90 ∧ 6 ≤ n ∧ n ≤ 25 } in 
  (finset.card (finset.filter (λ n, n ∈ divisors) (finset.range (25 + 1)))) = 5 :=
by
  have divisors : finset ℕ := (finset.filter (λ n, n ∈ { n | n ∣ 90 ∧ 6 ≤ n ∧ n ≤ 25 }) (finset.range (25 + 1)))
  have card_divisors : finset.card divisors = 5
  sorry

end band_row_lengths_l810_810006


namespace total_buttons_l810_810271

-- Define the conditions
def shirts_per_kid : Nat := 3
def number_of_kids : Nat := 3
def buttons_per_shirt : Nat := 7

-- Define the statement to prove
theorem total_buttons : shirts_per_kid * number_of_kids * buttons_per_shirt = 63 := by
  sorry

end total_buttons_l810_810271


namespace vehicle_value_this_year_l810_810429

variable (V_last_year : ℝ) (V_this_year : ℝ)

-- Conditions
def last_year_value : ℝ := 20000
def this_year_value : ℝ := 0.8 * last_year_value

theorem vehicle_value_this_year :
  V_last_year = last_year_value →
  V_this_year = this_year_value →
  V_this_year = 16000 := sorry

end vehicle_value_this_year_l810_810429


namespace line_segment_represents_feet_l810_810497

-- Define the conditions
def scale_factor := 500 -- One inch represents 500 feet
def line_segment_length := 6.5 -- Line segment in the drawing is 6.5 inches

-- Define the main theorem to prove the question equals the answer given the conditions.
theorem line_segment_represents_feet 
  (scale_factor_eq : scale_factor = 500)
  (line_segment_length_eq : line_segment_length = 6.5) : 
  line_segment_length * scale_factor = 3250 := 
by
  sorry

end line_segment_represents_feet_l810_810497


namespace fort_blocks_count_l810_810672

/-- A fort with dimensions 12 feet long, 10 feet wide, and 5 feet high is built using one-foot cubical blocks,
with a floor and four walls that are each one foot thick. Prove that the number of one-foot blocks used 
to construct the fort is 280. -/
theorem fort_blocks_count : 
  ∀ (length width height : ℕ) (thickness : ℕ),
  length = 12 →
  width = 10 →
  height = 5 →
  thickness = 1 →
  let volume_original := length * width * height,
      volume_interior  := (length - 2 * thickness) * (width - 2 * thickness) * (height - thickness),
      volume_blocks := volume_original - volume_interior 
  in volume_blocks = 280 :=
begin
  intros,
  rw [←volume_original, ←volume_interior, ←volume_blocks],
  sorry
end

end fort_blocks_count_l810_810672


namespace arithmetic_sequence_a5_l810_810662

variable (a : ℕ → ℝ)

theorem arithmetic_sequence_a5 (h : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
by
  sorry

end arithmetic_sequence_a5_l810_810662


namespace three_connected_l810_810222

variable {G : Type} [Graph G]

def edge_maximal (G : Graph) : Prop :=
  ∀ (G' : Graph), (G ⊆ G') → (TK₅ ⊆ G' ∨ TK_{3,3} ⊆ G') → (G = G')

theorem three_connected
  (h_size : |G| ≥ 4)
  (h_edge_max : edge_maximal G)
  (H1 : ∀ (G' : Graph), (G' ⊆ G) → ¬(TK₅ ⊆ G') ∧ ¬(TK_{3,3} ⊆ G')) :
  3_connected G :=
by sorry

end three_connected_l810_810222


namespace determine_dresses_l810_810579

-- Definitions
def girl := {name: String, dress: String}
def Anya   : girl := {name := "Anya", dress := ""}
def Valya  : girl := {name := "Valya", dress := ""}
def Galya  : girl := {name := "Galya", dress := ""}
def Nina   : girl := {name := "Nina", dress := ""}

-- Dresses
def Green := "green"
def Blue := "blue"
def White := "white"
def Pink := "pink"

-- Conditions
def girl_in_green := Galya
def girl_in_blue  := Valya
def girl_in_white := Anya
def girl_in_pink  := Nina

axiom between {A B C : girl} : girl → girl → girl → Prop
axiom condition1 : between Valya girl_in_green Nina
axiom condition2 : between girl_in_pink Anya Valya

-- Theorem Statement
theorem determine_dresses :
  girl_in_white.dress = White ∧
  girl_in_blue.dress = Blue ∧
  girl_in_green.dress = Green ∧
  girl_in_pink.dress = Pink := by
  sorry

end determine_dresses_l810_810579


namespace linear_correlation_is_very_strong_regression_line_residual_first_sample_effect_of_exclusion_l810_810031

noncomputable def resident_income : List ℝ := [32.2, 31.1, 32.9,  35.7,  37.1,  38.0,  39.0,  43.0,  44.6,  46.0]

noncomputable def goods_sales : List ℝ := [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0]

noncomputable def sum_x : ℝ := 379.6
noncomputable def sum_y : ℝ := 391
noncomputable def sum_x_squared_dev : ℝ := 246.904
noncomputable def sum_y_squared_dev : ℝ := 568.9
noncomputable def sum_xy_dev : ℝ := r * math.sqrt(sum_y_squared_dev * sum_x_squared_dev) where r := 0.95

-- Prove the linear correlation
theorem linear_correlation_is_very_strong : sum_xy_dev / (math.sqrt(sum_x_squared_dev) * math.sqrt(sum_y_squared_dev)) ≈ 0.95 → 
  (0.95 ≈ 1 ∨ 0.95 ≈ -1) :=
sorry

-- Calculate the regression line
theorem regression_line : sum_xy_dev / sum_x_squared_dev ≈ 1.44 ∧ 39.1 - 1.44 * 37.96 ≈ -15.56 → 
  (∀ x, y = 1.44 * x - 15.56) :=
sorry

-- Calculating the residual
theorem residual_first_sample : y - (1.44 * 32.2 - 15.56) ≈ -5.81 :=
sorry

-- Effect on regression line when excluding the sample point
theorem effect_of_exclusion (residual_lt_regression : y < 1.44 * x - 15.56):
  sum_xy_dev * math.sqrt(sum_x_squared_dev) * math.sqrt(sum_y_squared_dev) < sum_xy_dev * math.sqrt(sum_x_squared_dev) :=
sorry

end linear_correlation_is_very_strong_regression_line_residual_first_sample_effect_of_exclusion_l810_810031


namespace decoded_word_is_correct_l810_810850

-- Assume that we have a way to represent figures and encoded words
structure Figure1
structure Figure2

-- Assume the existence of a key that maps arrow patterns to letters
def decode (f1 : Figure1) (f2 : Figure2) : String := sorry

theorem decoded_word_is_correct (f1 : Figure1) (f2 : Figure2) :
  decode f1 f2 = "КОМПЬЮТЕР" :=
by
  sorry

end decoded_word_is_correct_l810_810850


namespace find_lambda_l810_810300

variable {a : ℕ → ℕ} 

def Sn (n : ℕ) : ℕ := n * a 1 + (n * (n - 1)) / 2 * (a 2 - a 1)

def Tn (n : ℕ) : ℚ := (Finset.range n).sum (λ k, 1 / (Sn (k+1) : ℚ))

theorem find_lambda (h1 : Sn 4 = 2 / 3 * Sn 5) (h2 : Sn 7 = 28) :
  ∃ λ, (λ = 2) ∧ ∀ n, Tn n < λ := 
begin
  sorry,
end

end find_lambda_l810_810300


namespace sum_of_possible_N_equals_zero_l810_810018

theorem sum_of_possible_N_equals_zero (S : ℝ) : 
  ∑ N in {N : ℝ | N ≠ 0 ∧ N - 4 / N + N^3 = S}, N = 0 :=
sorry

end sum_of_possible_N_equals_zero_l810_810018


namespace number_of_functions_is_1600_l810_810686

-- Define the set B
def B := Finset.range 8

-- Define the function type
def func := (B → B)

-- Define the condition g(g(g(x))) is constant
def is_constant (g : func) : Prop :=
  ∃ d : B, ∀ x : B, g (g (g x)) = d

-- Calculate the number of such functions and find the remainder when divided by 2000
theorem number_of_functions_is_1600 :
  let M := (8 * ∑ j in Finset.range 7, Nat.choose 7 j * j^(8 - j)) in
  M % 2000 = 1600 :=
by
  sorry

end number_of_functions_is_1600_l810_810686


namespace quadrilateral_angles_l810_810655

theorem quadrilateral_angles 
  (A B C D : Type) 
  (a d b c : Float)
  (hAD : a = d ∧ d = c) 
  (hBDC_twice_BDA : ∃ x : Float, b = 2 * x) 
  (hBDA_CAD_ratio : ∃ x : Float, d = 2/3 * x) :
  (∃ α β γ δ : Float, 
    α = 75 ∧ 
    β = 135 ∧ 
    γ = 60 ∧ 
    δ = 90) := 
sorry

end quadrilateral_angles_l810_810655


namespace monotonically_decreasing_intervals_l810_810104

noncomputable theory

open Real

def is_monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x₁ x₂⦄, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ > f x₂

def k_intervals (k : ℤ) : set ℝ := {x | k * π - π / 3 < x ∧ x < k * π + π / 6}

def y (x : ℝ) : ℝ := 3 - 2 * cos(2 * x - π / 3)

theorem monotonically_decreasing_intervals :
  ∀ k : ℤ, is_monotonically_decreasing y (k_intervals k) :=
sorry

end monotonically_decreasing_intervals_l810_810104


namespace b4_lt_b7_l810_810046

noncomputable def b : ℕ → ℚ
| 1       := 1 + 1 / (1 : ℚ)
| (n + 1) := (1 + 1 / ((b n) + 1 / (1 : ℚ)))

theorem b4_lt_b7 (α : ℕ → ℕ) (hα : ∀ n, α n > 0) : b 4 < b 7 :=
by
  sorry

end b4_lt_b7_l810_810046


namespace probability_x_gt_5y_l810_810729

-- Define the rectangular region as described
structure Point where
  x : ℝ
  y : ℝ

-- Define the rectangular region boundaries
def inRectangle (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 1000 ∧ 0 ≤ p.y ∧ p.y ≤ 1005

-- Define the inequality condition
def condition (p : Point) : Prop :=
  p.x > 5 * p.y

-- Define the problem of finding the probability
theorem probability_x_gt_5y :
  (∫ (p : Point) in {p | inRectangle p ∧ condition p}.toMeasure, 1) /
  (∫ (p : Point) in {p | inRectangle p}.toMeasure, 1) = 20 / 201 :=
sorry

end probability_x_gt_5y_l810_810729


namespace sum_last_three_coefficients_l810_810455

theorem sum_last_three_coefficients :
  let expr := (λ (a : ℚ), (1 - 1 / a)^8)
  let coefficients := [1, -8, 28] in
  coefficients.sum = 21 :=
by
  sorry

end sum_last_three_coefficients_l810_810455


namespace correct_inequalities_l810_810546

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem correct_inequalities (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) : 
  f(x₁) * f(-x₁) ≤ 0 ∧ f(x₁) + f(x₂) ≥ f(-x₁) + f(-x₂) :=
by
  sorry

end correct_inequalities_l810_810546


namespace remainder_of_1394_div_40_l810_810988

theorem remainder_of_1394_div_40 :
  let n := 1394 in
  n % 20 = 14 →
  n % 2535 = 1929 →
  n % 40 = 34 :=
by
  intros n h₁ h₂
  sorry

end remainder_of_1394_div_40_l810_810988


namespace pencils_profit_l810_810020

/-- Prove that selling 1500 pencils yields a profit of exactly $150 given the conditions
    that the shop owner purchases 2000 pencils at $0.15 each and sells each pencil for $0.30. -/
theorem pencils_profit :
  let pencils_purchased := 2000
  let purchase_price_per_pencil := 0.15
  let total_cost := pencils_purchased * purchase_price_per_pencil
  let desired_profit := 150
  let selling_price_per_pencil := 0.30
  let total_revenue_needed := total_cost + desired_profit
  let pencils_needed_to_sell := total_revenue_needed / selling_price_per_pencil
  in pencils_needed_to_sell = 1500 :=
by
  sorry

end pencils_profit_l810_810020


namespace measure_angle_B_l810_810715

theorem measure_angle_B (l k m : Line) (P Q : Point) 
  (h1 : parallel l k) 
  (h2 : intersects m l P)
  (h3 : intersects m k Q)
  (h4 : angle_at l m P = 110)
  (h5 : angle_at k m Q = 70) :
  angle_supplement m k Q = 110 := 
sorry

end measure_angle_B_l810_810715


namespace simplify_fraction_product_l810_810384

theorem simplify_fraction_product :
  4 * (18 / 5) * (35 / -63) * (8 / 14) = - (32 / 7) :=
by sorry

end simplify_fraction_product_l810_810384


namespace greatest_possible_x_max_possible_x_l810_810775

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l810_810775


namespace simplify_and_evaluate_l810_810744

def expr (a b : ℝ) : ℝ := 3 * a + 2 * (a - 1/2 * b^2) - (a - 2 * b^2)

theorem simplify_and_evaluate : expr (-2) 1 = -7 :=
by
  -- skip the detailed proof steps
  sorry

end simplify_and_evaluate_l810_810744


namespace five_student_committees_from_eight_l810_810131

theorem five_student_committees_from_eight : nat.choose 8 5 = 56 := by
  sorry

end five_student_committees_from_eight_l810_810131


namespace coordinates_of_P_correct_l810_810295

def point (ℝ : Type*) := ℝ × ℝ × ℝ

def A : point ℝ := (1, 2, 3)
def B : point ℝ := (4, 5, 6)
def ratio_m : ℝ := 3
def ratio_n : ℝ := 5
def t : ℝ := 5 / 8
def u : ℝ := 3 / 8

def P := (t * A.1 + u * B.1, t * A.2 + u * B.2, t * A.3 + u * B.3)

theorem coordinates_of_P_correct :
    P = ((ratio_n * B.1 + ratio_m * A.1) / (ratio_m + ratio_n), 
         (ratio_n * B.2 + ratio_m * A.2) / (ratio_m + ratio_n),
         (ratio_n * B.3 + ratio_m * A.3) / (ratio_m + ratio_n)) := by
  sorry

end coordinates_of_P_correct_l810_810295


namespace B_days_proof_l810_810009

-- Definitions of given conditions
def A_days := 30
def A_work_per_day := 1 / (A_days: ℝ)
def B_days (x: ℝ) := x
def B_work_per_day (x: ℝ) := 1 / (B_days x)
def total_days := 10
def fraction_completed := 0.5833333333333334
def fraction_left := 0.41666666666666663

-- To be proved
theorem B_days_proof : ∃ (x : ℝ), B_days x = 40 ∧ total_days * (A_work_per_day + B_work_per_day x) = fraction_completed :=
sorry

end B_days_proof_l810_810009


namespace sum_of_arithmetic_sequence_l810_810170

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ):
  (S 4 = S 8 - S 4) →
  (S 4 = S 12 - S 8) →
  (S 4 = S 16 - S 12) →
  S 16 / S 4 = 10 :=
by
  intros h1 h2 h3
  sorry

end sum_of_arithmetic_sequence_l810_810170


namespace greatest_x_lcm_105_l810_810793

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l810_810793


namespace angle_bisectors_perpendicular_and_intersect_on_AD_l810_810654

variable (A B C D : Type) [affine_space A B]
variables (AB BC CD AD : line A) 
variables (P Q : point A)

-- Assumption that AB || DC and BC = AB + CD
axiom parallel_AB_DC : parallel AB DC
axiom length_BC_eq_AB_plus_CD : length BC = length AB + length CD

-- Define the angle bisectors
def angle_bisector_ABC (A B C : point A) : line A := sorry
def angle_bisector_BCD (B C D : point A) : line A := sorry

-- Intersection point F on AD
axiom intersection_F_on_AD : ∃ (F : point A), F ∈ AD ∧ F = line_intersection (angle_bisector_ABC A B C) (angle_bisector_BCD B C D)

-- The main proof statement
theorem angle_bisectors_perpendicular_and_intersect_on_AD 
  (parallel_AB_DC : parallel AB DC) 
  (length_BC_eq_AB_plus_CD : length BC = length AB + length CD) :
  let angle_bisector_ABC := angle_bisector_ABC A B C in
  let angle_bisector_BCD := angle_bisector_BCD B C D in
  perpendicular angle_bisector_ABC angle_bisector_BCD ∧ 
  ∃ F ∈ AD, F = line_intersection angle_bisector_ABC angle_bisector_BCD :=
sorry

end angle_bisectors_perpendicular_and_intersect_on_AD_l810_810654


namespace triangle_area_enclosed_by_three_lines_l810_810857

theorem triangle_area_enclosed_by_three_lines :
  let line1 := λ x, (1 / 3) * x + 2
  let line2 := λ x, 3 * x - 6
  let point_A := (3, 3)
  let point_B := (4.5, 7.5)
  let point_C := (7.5, 4.5)
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ), 
    point_A = (x1, y1) ∧ point_B = (x2, y2) ∧ point_C = (x3, y3) ∧
    (y1 = line1 x1 ∧ y2 = line2 x2 ∧ x3 + y3 = 12) ∧
    let area := (1 / 2) * | x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) |
    area = 6 :=
begin
  sorry
end

end triangle_area_enclosed_by_three_lines_l810_810857


namespace first_part_lent_years_l810_810516

-- Conditions
def total_sum : ℝ := 2665
def second_part : ℝ := 1332.5
def interest_rate_first : ℝ := 0.03
def interest_rate_second : ℝ := 0.05
def time_second : ℝ := 3
def interest_second : ℝ := second_part * interest_rate_second * time_second := by
  norm_num [second_part, interest_rate_second, time_second]

-- Define the first part
def first_part : ℝ := total_sum - second_part

-- Define interest on the first part for n years
def interest_first (n : ℝ) : ℝ := first_part * interest_rate_first * n

-- Prove the number of years the first part is lent is equal to 5
theorem first_part_lent_years : ∃ n : ℝ, interest_first n = interest_second ∧ n = 5 := by
  use 5
  field_simp [first_part, total_sum, second_part, interest_rate_first, interest_rate_second, time_second]
  norm_num
  ring_nf
  sorry

end first_part_lent_years_l810_810516


namespace Ceva_theorem_l810_810733

variables {A B C K L M P : Point}
variables {BK KC CL LA AM MB : ℝ}

-- Assume P is inside the triangle ABC and KP, LP, and MP intersect BC, CA, and AB at points K, L, and M respectively
-- We need to prove the ratio product property according to Ceva's theorem
theorem Ceva_theorem 
  (h1: BK / KC = b)
  (h2: CL / LA = c)
  (h3: AM / MB = a)
  (h4: (b * c * a = 1)): 
  (BK / KC) * (CL / LA) * (AM / MB) = 1 :=
sorry

end Ceva_theorem_l810_810733


namespace circle_Q_eq_no_suitable_a_l810_810597

open Real

noncomputable def circle := {x | x.1^2 + x.2^2 - 6*x.1 + 4*x.2 + 4 = 0}
def point_P := (2, 0)

-- Question 1: Prove that the equation of circle Q with MN as its diameter is (x-2)^2 + y^2 = 4
theorem circle_Q_eq : 
  ∀ M N : ℝ × ℝ, (M ∈ circle) → (N ∈ circle) → 
    dist M N = 4 → 
    ∃ Q : ℝ × ℝ → Prop, 
    (∀ p : ℝ × ℝ, Q p ↔ (p.1 - 2)^2 + p.2^2 = 4) := 
by
  sorry

-- Question 2: Prove that there is no real number a such that the line passing through point P perpendicularly bisects chord AB
theorem no_suitable_a : 
  ¬ ∃ a : ℝ, 
    (∀ A B : ℝ × ℝ, (A ∈ circle) → (B ∈ circle) → 
    A ≠ B → (a * A.1 - A.2 + 1 = 0) → (a * B.1 - B.2 + 1 = 0) → 
    ∃ l : ℝ × ℝ → Prop, 
    (∀ p : ℝ × ℝ, l p ↔ dist p point_P = dist p (3, -2))) := 
by
  sorry

end circle_Q_eq_no_suitable_a_l810_810597


namespace probability_red_or_white_l810_810465

-- Define the total number of marbles and the counts of blue and red marbles.
def total_marbles : Nat := 60
def blue_marbles : Nat := 5
def red_marbles : Nat := 9

-- Define the remainder to calculate white marbles.
def white_marbles : Nat := total_marbles - (blue_marbles + red_marbles)

-- Lean proof statement to show the probability of selecting a red or white marble.
theorem probability_red_or_white :
  (red_marbles + white_marbles) / total_marbles = 11 / 12 :=
by
  sorry

end probability_red_or_white_l810_810465


namespace regular_polyhedra_similarity_l810_810736

theorem regular_polyhedra_similarity
  (P1 P2 : RegularPolyhedron)
  (h_kind : P1.kind = P2.kind) :
  Similar P1 P2 :=
sorry

end regular_polyhedra_similarity_l810_810736


namespace greatest_x_lcm_105_l810_810787

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l810_810787


namespace tom_walks_distance_l810_810674

theorem tom_walks_distance (t : ℝ) (d : ℝ) :
  t = 15 ∧ d = (1 / 18) * t → d ≈ 0.8 :=
by
  sorry

end tom_walks_distance_l810_810674


namespace sum_of_five_terms_sequence_l810_810261

def sequence (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n a_n, 2 * a_n)

theorem sum_of_five_terms_sequence : 
  sequence 1 + sequence 2 + sequence 3 + sequence 4 + sequence 5 = 31 :=
by
  sorry

end sum_of_five_terms_sequence_l810_810261


namespace percentage_of_men_speaking_french_l810_810477

theorem percentage_of_men_speaking_french {total_employees men women french_speaking_employees french_speaking_women french_speaking_men : ℕ}
    (h1 : total_employees = 100)
    (h2 : men = 60)
    (h3 : women = 40)
    (h4 : french_speaking_employees = 50)
    (h5 : french_speaking_women = 14)
    (h6 : french_speaking_men = french_speaking_employees - french_speaking_women)
    (h7 : french_speaking_men * 100 / men = 60) : true :=
by
  sorry

end percentage_of_men_speaking_french_l810_810477


namespace paolo_sevilla_birthday_l810_810342

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l810_810342


namespace right_angled_triangle_l810_810528

theorem right_angled_triangle {a b c : ℝ} (h : a = 1 ∧ b = real.sqrt 2 ∧ c = real.sqrt 3) :
  (a^2 + b^2 = c^2) ∧ (¬ (real.sqrt 2)^2 + (real.sqrt 3)^2 = 2^2) ∧ 
  (¬ 5^2 + 4^2 = 6^2) ∧ (¬ 9^2 + 16^2 = 25^2) :=
by
  have h₁ : 1^2 + (real.sqrt 2)^2 = (real.sqrt 3)^2 := by sorry
  have h₂ : (real.sqrt 2)^2 + (real.sqrt 3)^2 ≠ 2^2 := by sorry
  have h₃ : 5^2 + 4^2 ≠ 6^2 := by sorry
  have h₄ : 9^2 + 16^2 ≠ 25^2 := by sorry
  exact ⟨h₁, h₂, h₃, h₄⟩

end right_angled_triangle_l810_810528


namespace b4_lt_b7_l810_810039

noncomputable theory
open_locale big_operators

def α (k: ℕ) := k + 1 -- This makes α_{k+1} in ℕ^*
def b : ℕ → ℚ
| 0 := 1 + 1 / α 0
| (n + 1) := 1 + 1 / (α 0 + ∑ i in finset.range (n + 1), 1 / b i)

theorem b4_lt_b7: b 3 < b 6 := 
sorry

end b4_lt_b7_l810_810039


namespace probability_vertical_side_l810_810014

/- Definition of the problem -/

def boundary_condition (x y : Nat) : Prop :=
  (x = 0 ∧ 0 ≤ y ∧ y ≤ 6) ∨ (x = 6 ∧ 0 ≤ y ∧ y ≤ 6) ∨ 
  (y = 0 ∧ 0 ≤ x ∧ x ≤ 6) ∨ (y = 6 ∧ 0 ≤ x ∧ x ≤ 6)

noncomputable def Q : ℕ × ℕ → ℚ
| (0, y) => 1
| (6, y) => 1
| (x, 0) => 0
| (x, 6) => 0
| (2, 3) => 1/2 * Q (3, 3) + 1/4 * Q (2, 2) + 1/4 * Q (2, 4)
| (3, 3) => 1/4 * Q (2, 3) + 1/4 * Q (4, 3) + 1/4 * Q (3, 2) + 1/4 * Q (3, 4)
| _ => 0

theorem probability_vertical_side : Q (2, 3) = (5 : ℚ) / 8 :=
by sorry

end probability_vertical_side_l810_810014


namespace cylinder_cone_volume_ratio_l810_810485

theorem cylinder_cone_volume_ratio (r h : ℝ) :
  let V_cylinder := π * r^2 * h,
      V_cone := (1 / 3) * π * r^2 * h
  in V_cylinder / V_cone = 3 :=
by
  sorry

end cylinder_cone_volume_ratio_l810_810485


namespace size_of_former_apartment_l810_810678

open Nat

theorem size_of_former_apartment
  (former_rent_rate : ℕ)
  (new_apartment_cost : ℕ)
  (savings_per_year : ℕ)
  (split_factor : ℕ)
  (savings_per_month : ℕ)
  (share_new_rent : ℕ)
  (former_rent : ℕ)
  (apartment_size : ℕ)
  (h1 : former_rent_rate = 2)
  (h2 : new_apartment_cost = 2800)
  (h3 : savings_per_year = 1200)
  (h4 : split_factor = 2)
  (h5 : savings_per_month = savings_per_year / 12)
  (h6 : share_new_rent = new_apartment_cost / split_factor)
  (h7 : former_rent = share_new_rent + savings_per_month)
  (h8 : apartment_size = former_rent / former_rent_rate) :
  apartment_size = 750 :=
by
  sorry

end size_of_former_apartment_l810_810678


namespace volume_of_tetrahedron_EFGH_l810_810658

theorem volume_of_tetrahedron_EFGH
  (EF : ℝ) (area_EFG : ℝ) (area_EFH : ℝ) (angle_EFG_EFH : ℝ)
  (hEF : EF = 4)
  (hEFG : area_EFG = 18)
  (hEFH : area_EFH = 16)
  (hAngle : angle_EFG_EFH = 45) :
  ∃ (V : ℝ), V = 68 * real.sqrt 2 / 3 :=
by
  sorry

end volume_of_tetrahedron_EFGH_l810_810658


namespace logarithm_expression_l810_810115

theorem logarithm_expression (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : 
    (log 2 (a^2) + log 4 (1 / a^2)) * (log a 2 + log (a^2) (1 / 2)) = 1 / 2 := 
by
  sorry

end logarithm_expression_l810_810115


namespace MrKishore_petrol_expense_l810_810940

theorem MrKishore_petrol_expense 
  (rent milk groceries education misc savings salary expenses petrol : ℝ)
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_education : education = 2500)
  (h_misc : misc = 700)
  (h_savings : savings = 1800)
  (h_salary : salary = 18000)
  (h_expenses_equation : expenses = rent + milk + groceries + education + petrol + misc)
  (h_savings_equation : savings = salary * 0.10)
  (h_total_equation : salary = expenses + savings) :
  petrol = 2000 :=
by
  sorry

end MrKishore_petrol_expense_l810_810940


namespace find_number_l810_810444

theorem find_number (n : ℕ) (a : ℕ) (d : ℕ) (h_d : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (h_eq : n = 10 * a + d) (h_ineq : n > 5 * d) : n = 25 :=
sorry

end find_number_l810_810444


namespace sum_of_digits_l810_810521

theorem sum_of_digits (N : ℕ) (h : N * (N + 1) / 2 = 3003) : (7 + 7) = 14 := by
  sorry

end sum_of_digits_l810_810521


namespace area_of_S_l810_810391

-- Define the rectangle and its properties
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)

-- Define the specific rectangle as given in the problem
def ABCD : Rectangle := 
  { length := 18, width := 12 }

-- Define the region S as described in the problem
def isClosestToCenter (rect : Rectangle) (p : ℝ × ℝ) : Prop :=
  let center := (rect.length / 2, rect.width / 2)
  let distances := 
    [ (0, 0), (rect.length, 0), (0, rect.width), (rect.length, rect.width) ].map 
      (λ vertex => (vertex.1 - p.1) ^ 2 + (vertex.2 - p.2) ^ 2)
  let min_distance := (center.1 - p.1) ^ 2 + (center.2 - p.2) ^ 2
  min_distance < distances.minD

def S (rect : Rectangle) : set (ℝ × ℝ) := 
  { p | isClosestToCenter rect p }

-- The theorem to prove
theorem area_of_S (rect : Rectangle) (h1 : rect.length = 18) (h2 : rect.width = 12) :
  let side_length := rect.length / 2
  let side_width := rect.width / 2
  side_length * side_width = 54 := by
  sorry

end area_of_S_l810_810391


namespace line_intersection_y_axis_l810_810951

theorem line_intersection_y_axis :
  let p1 := (2, 8)
      p2 := (4, 14)
      m := (p2.2 - p1.2) / (p2.1 - p1.1)  -- slope calculation
      b := p1.2 - m * p1.1  -- y-intercept calculation
  in (b = 2) → (m = 3) → (p1 ≠ p2) → 
    (0, b) = @ y-intercept of the line passing through p1 and p2 :=
by
  intros p1 p2 m b h1 h2 h3
  -- placeholder for actual proof
  sorry

end line_intersection_y_axis_l810_810951


namespace boy_two_girls_work_completion_days_l810_810027

-- Work rates definitions
def man_work_rate := 1 / 6
def woman_work_rate := 1 / 18
def girl_work_rate := 1 / 12
def team_work_rate := 1 / 3

-- Boy's work rate
def boy_work_rate := 1 / 36

-- Combined work rate of boy and two girls
def boy_two_girls_work_rate := boy_work_rate + 2 * girl_work_rate

-- Prove that the number of days it will take for a boy and two girls to complete the work is 36 / 7
theorem boy_two_girls_work_completion_days : (1 / boy_two_girls_work_rate) = 36 / 7 :=
by
  sorry

end boy_two_girls_work_completion_days_l810_810027


namespace evaluate_expression_l810_810603

section

variables {a b c : ℝ}

def condition := (a / (45 - a) + b / (85 - b) + c / (75 - c) = 9)

theorem evaluate_expression (h : condition) : 
  (9 / (45 - a) + 17 / (85 - b) + 15 / (75 - c) = 2.4) :=
sorry

end

end evaluate_expression_l810_810603


namespace problem_statement_l810_810034

open Real

noncomputable def correlation_coefficient (x y : List ℝ) : ℝ :=
  let n := x.length
  let x̄ := List.sum x / n.toReal
  let ȳ := List.sum y / n.toReal
  let numerator := List.sum (List.zipWith (λ xi yi, (xi - x̄) * (yi - ȳ)) x y)
  let denominator := sqrt (List.sum (List.map (λ xi, (xi - x̄)^2) x) * List.sum (List.map (λ yi, (yi - ȳ)^2) y))
  numerator / denominator

def residual_correct (x y : List ℝ) (predicted_y : ℝ) (actual_y : ℝ) : ℝ :=
  actual_y - predicted_y

theorem problem_statement :
  -- Given conditions
  let x := [32.2, 31.1, 32.9, 35.7, 37.1, 38.0, 39.0, 43.0, 44.6, 46.0]
  let y := [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0]
  let n := 10
  let sum_x := 379.6
  let sum_y := 391
  let sum_x_squared := 246.904
  let sum_y_squared := 568.9
  let covariance_xy := m
  let mean_x := 37.96
  let mean_y := 39.1
  let r := correlation_coefficient x y
  -- Proving the statements
  r ≈ 0.95 →
  ∃ a b : ℝ, 
    b = 1.44 ∧
    a = -15.56 ∧
    ∀ xi yi, 
    (xi, yi) ∈ List.zip x y →
    residual_correct xi yi (b * xi + a) ≈ -5.81 :=
by sorry

end problem_statement_l810_810034


namespace area_of_extended_quadrilateral_l810_810383

theorem area_of_extended_quadrilateral
  (EFGH : convex_quadrilateral)
  (a b c d : point)
  (EF FE' FG GG' GH HH' HE EE' : ℝ)
  (area_EFGH : ℝ)
  (h1 : EFGH.EF = 7)
  (h2 : EFGH.FE' = 7)
  (h3 : EFGH.FG = 7)
  (h4 : EFGH.GG' = 7)
  (h5 : EFGH.GH = 7)
  (h6 : EFGH.HH' = 7)
  (h7 : EFGH.HE = 5)
  (h8 : EFGH.EE' = 5)
  (h9 : EFGH.area = 18) :
  ∃ extension_area : ℝ, extension_area = 54 := 
sorry

end area_of_extended_quadrilateral_l810_810383


namespace bridge_length_is_correct_l810_810518

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_cross : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600 in
  let total_distance := train_speed_mps * time_to_cross in
  total_distance - train_length

theorem bridge_length_is_correct :
  length_of_bridge 110 60 20.99832013438925 = 240 :=
by
  sorry

end bridge_length_is_correct_l810_810518


namespace max_x_y2_z3_l810_810703

theorem max_x_y2_z3 (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) : 
    x + y^2 + z^3 ≤ 1 :=
begin
  -- skipping proof
  sorry
end

example : ∃ x y z : ℝ, x + y + z = 1 ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y^2 + z^3 = 1 :=
begin
  use [1, 0, 0],
  split,
  { exact rfl },
  split,
  { linarith },
  split,
  { linarith },
  split,
  { linarith },
  dsimp,
  ring,
end

end max_x_y2_z3_l810_810703


namespace intersection_A_complement_B_l810_810623

open Set

noncomputable def I := {x : ℝ | True}
noncomputable def A := {x : ℝ | (x + 1) / (x + 3) ≤ 1 / 2}
noncomputable def B := {x : ℝ | 2^(|x + 1|) < 4}

noncomputable def C_I_B := {x : ℝ | x ≤ -3 ∨ x ≥ 1}

theorem intersection_A_complement_B : A ∩ C_I_B = {1} :=
by
  sorry

end intersection_A_complement_B_l810_810623


namespace greatest_x_lcm_105_l810_810796

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l810_810796


namespace area_of_square_l810_810511

-- Define the diagonal length condition.
def diagonal_length : ℝ := 12 * real.sqrt 2

-- Define the side length of the square computed from the diagonal using the 45-45-90 triangle property.
def side_length : ℝ := diagonal_length / real.sqrt 2

-- Define the area of the square in terms of its side length.
def square_area : ℝ := side_length * side_length

-- Prove that the area is indeed 144 square centimeters.
theorem area_of_square (d : ℝ) (h : d = 12 * real.sqrt 2) : (d / real.sqrt 2) * (d / real.sqrt 2) = 144 :=
by
  rw [h, ←real.mul_div_cancel (12 * real.sqrt 2) (real.sqrt 2)],
  { norm_num },
  { exact real.sqrt_ne_zero'.2 (by norm_num) }

end area_of_square_l810_810511


namespace cost_of_each_barbell_l810_810289

variables (barbells : ℕ) (money_given money_change : ℝ) (total_cost_per_barbell : ℝ)

-- Given conditions
def conditions := barbells = 3 ∧ money_given = 850 ∧ money_change = 40

-- Theorem statement: Proving the cost of each barbell is $270
theorem cost_of_each_barbell (h : conditions) : total_cost_per_barbell = 270 :=
by
  -- We are using sorry to indicate we are skipping the proof
  sorry

#eval sorry -- Placeholder to verify if the code is syntactically correct

end cost_of_each_barbell_l810_810289


namespace five_student_committees_from_eight_l810_810133

theorem five_student_committees_from_eight : nat.choose 8 5 = 56 := by
  sorry

end five_student_committees_from_eight_l810_810133


namespace compute_expression_l810_810304

noncomputable def complexify (θ : ℝ) : ℂ :=
  complex.exp (θ * complex.I)

theorem compute_expression :
  let x := complex.exp ((2 * real.pi) / 9  * complex.I),
  (2 * x + x^2) * (2 * x^2 + x^4) * (2 * x^3 + x^6) * (2 * x^4 + x^8) = 33 :=
by sorry

end compute_expression_l810_810304


namespace possible_50_turtles_members_l810_810667

-- Define students and their friendship
constant students : Fin 100 → Type
def friends (a b : students) : Prop := sorry

-- Define the initial conditions
constant students_count : Nat := 100
constant cheetahs_founded_date : Nat
constant february_19th : Nat
def cheetahs_initial_members : Set (students) := sorry
def cheetahs_full_members (date : Nat) : Set (students) :=
  if date = february_19th then sorry else sorry

-- Define the spread condition
def joins_club (s : students) (c : Set students) : Prop :=
  sorry -- condition to check if a student joins a club

-- Prove that the conditions allow 50 students in the Turtles club on February 19th
theorem possible_50_turtles_members :
  ∃ turtles : Set students, turtles.count = 50 ∧
    (∀ s, (s ∈ cheetahs_full_members february_19th) → (P → s ∈ turtles)) where P := sorry := sorry

end possible_50_turtles_members_l810_810667


namespace max_dim_t_normal_space_l810_810948

def t_normal (A : Matrix (Fin n) (Fin n) ℂ) : Prop := A ⬝ A.transpose = A.transpose ⬝ A

theorem max_dim_t_normal_space (n : ℕ) : 
  ∃ (S : Submodule ℂ (Matrix (Fin n) (Fin n) ℂ)), 
    (∀ A ∈ S, t_normal A) ∧ 
    FiniteDimensional.finrank ℂ S = n * (n + 1) / 2 := 
sorry

end max_dim_t_normal_space_l810_810948


namespace find_f_half_l810_810585

variable {α : Type} [DivisionRing α]

theorem find_f_half {f : α → α} (h : ∀ x, f (1 - 2 * x) = 1 / (x^2)) : f (1 / 2) = 16 :=
by
  sorry

end find_f_half_l810_810585


namespace find_value_am2_bm_minus_7_l810_810217

variable {a b m : ℝ}

theorem find_value_am2_bm_minus_7
  (h : a * m^2 + b * m + 5 = 0) : a * m^2 + b * m - 7 = -12 :=
by
  sorry

end find_value_am2_bm_minus_7_l810_810217


namespace combination_eight_choose_five_l810_810137

theorem combination_eight_choose_five : 
  ∀ (n k : ℕ), n = 8 ∧ k = 5 → Nat.choose n k = 56 :=
by 
  intros n k h
  obtain ⟨hn, hk⟩ := h
  rw [hn, hk]
  exact Nat.choose_eq 8 5
  sorry  -- This signifies that the proof needs to be filled in, but we'll skip it as per instructions.

end combination_eight_choose_five_l810_810137


namespace greatest_x_lcm_l810_810801

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l810_810801


namespace height_of_fifth_tree_l810_810426

theorem height_of_fifth_tree 
  (h₁ : tallest_tree = 108) 
  (h₂ : second_tallest_tree = 54 - 6) 
  (h₃ : third_tallest_tree = second_tallest_tree / 4) 
  (h₄ : fourth_shortest_tree = (second_tallest_tree + third_tallest_tree) - 2) 
  (h₅ : fifth_tree = 0.75 * (tallest_tree + second_tallest_tree + third_tallest_tree + fourth_shortest_tree)) : 
  fifth_tree = 169.5 :=
by
  sorry

end height_of_fifth_tree_l810_810426


namespace find_a_l810_810609

theorem find_a : 
  ∃ a : ℝ, 
  (∀ x y : ℝ, (a * x + y - 2) = 0 → (x-1)^2 + (y-a)^2 = 4) ∧
  (∀ A B C : ℝ×ℝ, ((a * (fst A) + (snd A) - 2) = 0 → 
                  (a * (fst B) + (snd B) - 2) = 0 → 
                  (fst C = 1 ∧ snd C = a ∧ (fst C - fst A)^2 + (snd C - snd A)^2 = 4) →
                  (fst C - fst B)^2 + (snd C - snd B)^2 = 4) →
  (√(2^2-1) = √3)) → 
  a = 4 + sqrt 15 ∨ a = 4 - sqrt 15 :=
begin
  sorry
end

end find_a_l810_810609


namespace b4_lt_b7_l810_810038

noncomputable theory
open_locale big_operators

def α (k: ℕ) := k + 1 -- This makes α_{k+1} in ℕ^*
def b : ℕ → ℚ
| 0 := 1 + 1 / α 0
| (n + 1) := 1 + 1 / (α 0 + ∑ i in finset.range (n + 1), 1 / b i)

theorem b4_lt_b7: b 3 < b 6 := 
sorry

end b4_lt_b7_l810_810038


namespace number_of_x_values_l810_810207

theorem number_of_x_values : 
  (∃ x_values : Finset ℕ, (∀ x ∈ x_values, 10 ≤ x ∧ x < 25) ∧ x_values.card = 15) :=
by
  sorry

end number_of_x_values_l810_810207


namespace louis_age_currently_31_l810_810231

-- Definitions
variable (C L : ℕ)
variable (h1 : C + 6 = 30)
variable (h2 : C + L = 55)

-- Theorem statement
theorem louis_age_currently_31 : L = 31 :=
by
  sorry

end louis_age_currently_31_l810_810231


namespace required_line_equation_l810_810987

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Line structure with general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- A point P on a line
def on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Perpendicular condition between two lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- The known line
def known_line : Line := {a := 1, b := -2, c := 3}

-- The given point
def P : Point := {x := -1, y := 3}

noncomputable def required_line : Line := {a := 2, b := 1, c := -1}

-- The theorem to be proved
theorem required_line_equation (l : Line) (P : Point) :
  (on_line P l) ∧ (perpendicular l known_line) ↔ l = required_line :=
  by
    sorry

end required_line_equation_l810_810987


namespace number_of_friends_l810_810333

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l810_810333


namespace sum_of_solutions_l810_810112

theorem sum_of_solutions (y : ℝ) (h : y + 16 / y = 12) : y = 4 ∨ y = 8 → 4 + 8 = 12 :=
by sorry

end sum_of_solutions_l810_810112


namespace chess_tournament_matches_l810_810648

theorem chess_tournament_matches (n : ℕ) (h : n = 32) :
  ∃ m : ℕ, m = 31 :=
by
  have m := n - 1
  use m
  rw [h]
  simp
  sorry

end chess_tournament_matches_l810_810648


namespace avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l810_810931

open Real
open List

-- Conditions
def x_vals : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y_vals : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
def sum_x : ℝ := 0.6
def sum_y : ℝ := 3.9
def sum_x_squared : ℝ := 0.038
def sum_y_squared : ℝ := 1.6158
def sum_xy : ℝ := 0.2474
def total_root_area : ℝ := 186

-- Proof problems
theorem avg_root_area : (List.sum x_vals / 10) = 0.06 := by
  sorry

theorem avg_volume : (List.sum y_vals / 10) = 0.39 := by
  sorry

theorem correlation_coefficient : 
  let mean_x := List.sum x_vals / 10;
  let mean_y := List.sum y_vals / 10;
  let numerator := List.sum (List.zipWith (λ x y => (x - mean_x) * (y - mean_y)) x_vals y_vals);
  let denominator := sqrt ((List.sum (List.map (λ x => (x - mean_x) ^ 2) x_vals)) * (List.sum (List.map (λ y => (y - mean_y) ^ 2) y_vals)));
  (numerator / denominator) = 0.97 := by 
  sorry

theorem total_volume_estimate : 
  let avg_x := sum_x / 10;
  let avg_y := sum_y / 10;
  (avg_y / avg_x) * total_root_area = 1209 := by
  sorry

end avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l810_810931


namespace B_completes_in_40_days_l810_810896

noncomputable def BCompletesWorkInDays (x : ℝ) : ℝ :=
  let A_rate := 1 / 45
  let B_rate := 1 / x
  let work_done_together := 9 * (A_rate + B_rate)
  let work_done_B_alone := 23 * B_rate
  let total_work := 1
  work_done_together + work_done_B_alone

theorem B_completes_in_40_days :
  BCompletesWorkInDays 40 = 1 :=
by
  sorry

end B_completes_in_40_days_l810_810896


namespace sin_cos_sum_l810_810751

theorem sin_cos_sum (θ a : ℝ) (h1 : 0 ≤ θ ∧ θ ≤ π / 2) (h2 : sin (2 * θ) = a) : 
  sin θ + cos θ = sqrt (1 + a) :=
sorry

end sin_cos_sum_l810_810751


namespace greatest_x_lcm_105_l810_810791

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l810_810791


namespace sin_cos_identity_proof_l810_810606

noncomputable def identity_to_prove : ℝ :=
  let x := Real.pi / 12
  in Real.sin x ^ 4 - Real.cos x ^ 4

theorem sin_cos_identity_proof : 
  let x := Real.pi / 12 in
  identity_to_prove = - (Real.sqrt 3) / 2 :=
by
  -- Proof is omitted, just the statement is needed
  sorry

end sin_cos_identity_proof_l810_810606


namespace cyclic_shift_permutation_sum_condition_l810_810576

theorem cyclic_shift_permutation_sum_condition (n : ℕ) :
  (∃ (a b : ℕ → ℕ), (cyclic_shift a ∧ permutation b ∧ ∀ i, 1 + a i + b i = (n + 3) / 2)) ↔ (n % 2 = 1) :=
sorry

def cyclic_shift (a : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ), ∀ i, a i = (i + k) % n + 1

def permutation (b : ℕ → ℕ) : Prop :=
  ∃ (l : list ℕ), list.perm l (list.range n) ∧ ∀ i, b i = l.nth i 

end cyclic_shift_permutation_sum_condition_l810_810576


namespace max_handshakes_l810_810476

-- Definitions based on the given conditions
def num_people := 30
def handshake_formula (n : ℕ) := n * (n - 1) / 2

-- Formal statement of the problem
theorem max_handshakes : handshake_formula num_people = 435 :=
by
  -- Calculation here would be carried out in the proof, but not included in the statement itself.
  sorry

end max_handshakes_l810_810476


namespace arrangement_non_adjacent_l810_810007

theorem arrangement_non_adjacent :
  let total_arrangements := Nat.factorial 30
  let adjacent_arrangements := 2 * Nat.factorial 29
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements = 28 * Nat.factorial 29 :=
by
  sorry

end arrangement_non_adjacent_l810_810007


namespace smallest_n_l810_810705

theorem smallest_n (n : ℕ) (x : Fin n → ℝ)
  (h1 : ∀ i, |x i| < 1)
  (h2 : (Finset.univ.sum (λ i, |x i|)) = 19 + |Finset.univ.sum (λ i, x i)|) :
  n = 20 :=
by sorry

end smallest_n_l810_810705


namespace barbell_cost_l810_810284

theorem barbell_cost (num_barbells : ℤ) (total_money_given : ℤ) 
  (change_received : ℤ) (total_cost : ℤ) (each_barbell_cost : ℤ) 
  (h1 : num_barbells = 3) (h2 : total_money_given = 850) 
  (h3 : change_received = 40) (h4 : total_cost = total_money_given - change_received)
  (h5 : each_barbell_cost = total_cost / num_barbells) 
  : each_barbell_cost = 270 :=
by
  rw [h2, h3] at h4
  rw [← h4, h1] at h5
  exact calc 
    each_barbell_cost = (total_money_given - change_received) / num_barbells : h5
                    ... = (850 - 40) / 3 : by rw [h2, h3, h1]
                    ... = 810 / 3 : rfl
                    ... = 270 : rfl

end barbell_cost_l810_810284


namespace hot_dogs_per_pack_l810_810875

-- Define the givens / conditions
def total_hot_dogs : ℕ := 36
def buns_pack_size : ℕ := 9
def same_quantity (h : ℕ) (b : ℕ) := h = b

-- State the theorem to be proven
theorem hot_dogs_per_pack : ∃ h : ℕ, (total_hot_dogs / h = buns_pack_size) ∧ same_quantity (total_hot_dogs / h) (total_hot_dogs / buns_pack_size) := 
sorry

end hot_dogs_per_pack_l810_810875


namespace total_distance_traveled_l810_810022

def distance_from_earth_to_planet_x : ℝ := 0.5
def distance_from_planet_x_to_planet_y : ℝ := 0.1
def distance_from_planet_y_to_earth : ℝ := 0.1

theorem total_distance_traveled : 
  distance_from_earth_to_planet_x + distance_from_planet_x_to_planet_y + distance_from_planet_y_to_earth = 0.7 :=
by
  sorry

end total_distance_traveled_l810_810022


namespace rectangle_area_with_circles_l810_810966

theorem rectangle_area_with_circles :
  ∀ (r : ℝ), r = 3 → (∀ (length width : ℝ), length = 2 * r + 6 → width = 2 * r + 6 → (length * width) = 144) :=
begin
  intros r hr length width hlength hwidth,
  rw hr,
  simp [hlength, hwidth],
  norm_num,
end

end rectangle_area_with_circles_l810_810966


namespace magnitude_z_is_sqrt2_l810_810157

noncomputable def magnitude_z_eq_sqrt2 : Prop :=
  let z := (2 * complex.i) / (1 + complex.i) in
  complex.abs z = real.sqrt 2

theorem magnitude_z_is_sqrt2 : magnitude_z_eq_sqrt2 :=
  sorry

end magnitude_z_is_sqrt2_l810_810157


namespace birthday_friends_count_l810_810353

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l810_810353


namespace non_zero_digits_l810_810213

theorem non_zero_digits (n : ℕ) (h : n = 2) :
  let x := 80 / (2^4 * 5^6)
  in count_non_zero_digits_right_of_decimal x = n := 
by
  let x := 80 / (2^4 * 5^6)
  have : x = 1 / (5^5) := by
    calc
      80 / (2^4 * 5^6)
          = (2^4 * 5) / (2^4 * 5^6) : by sorry
      ... = 1 / (5^5) : by sorry
  show count_non_zero_digits_right_of_decimal x = n 
  from sorry

end non_zero_digits_l810_810213


namespace four_neg_x_equals_eight_l810_810223

-- Definitions based on the problem conditions
def four_pow_two_x_eq_sixty_four (x : ℝ) : Prop := 4^(2*x) = 64

-- The statement we want to prove
theorem four_neg_x_equals_eight (x : ℝ) (h : four_pow_two_x_eq_sixty_four x) : 4^(-x) = 1/8 :=
by
  -- Proof would go here
  sorry

end four_neg_x_equals_eight_l810_810223


namespace primes_p_q_divisibility_l810_810636

theorem primes_p_q_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hq_eq : q = p + 2) :
  (p + q) ∣ (p ^ q + q ^ p) := 
sorry

end primes_p_q_divisibility_l810_810636


namespace find_a9_l810_810652

variable (a : ℕ → ℝ)  -- Define a sequence a_n.

-- Define the conditions for the arithmetic sequence.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

variables (h_arith_seq : is_arithmetic_sequence a)
          (h_a3 : a 3 = 8)   -- Condition a_3 = 8
          (h_a6 : a 6 = 5)   -- Condition a_6 = 5 

-- State the theorem.
theorem find_a9 : a 9 = 2 := by
  sorry

end find_a9_l810_810652


namespace area_of_square_l810_810514

-- Define the diagonal length condition.
def diagonal_length : ℝ := 12 * real.sqrt 2

-- Define the side length of the square computed from the diagonal using the 45-45-90 triangle property.
def side_length : ℝ := diagonal_length / real.sqrt 2

-- Define the area of the square in terms of its side length.
def square_area : ℝ := side_length * side_length

-- Prove that the area is indeed 144 square centimeters.
theorem area_of_square (d : ℝ) (h : d = 12 * real.sqrt 2) : (d / real.sqrt 2) * (d / real.sqrt 2) = 144 :=
by
  rw [h, ←real.mul_div_cancel (12 * real.sqrt 2) (real.sqrt 2)],
  { norm_num },
  { exact real.sqrt_ne_zero'.2 (by norm_num) }

end area_of_square_l810_810514


namespace calories_350_grams_mint_lemonade_l810_810716

-- Definitions for the weights of ingredients in grams
def lemon_juice_weight := 150
def sugar_weight := 200
def water_weight := 300
def mint_weight := 50
def total_weight := lemon_juice_weight + sugar_weight + water_weight + mint_weight

-- Definitions for the caloric content per specified weight
def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def mint_calories_per_10g := 7
def water_calories := 0

-- Calculate total calories from each ingredient
def lemon_juice_calories := (lemon_juice_calories_per_100g * lemon_juice_weight) / 100
def sugar_calories := (sugar_calories_per_100g * sugar_weight) / 100
def mint_calories := (mint_calories_per_10g * mint_weight) / 10

-- Calculate total calories in the lemonade
def total_calories := lemon_juice_calories + sugar_calories + mint_calories + water_calories

noncomputable def calories_in_350_grams : ℕ := (total_calories * 350) / total_weight

-- Theorem stating the number of calories in 350 grams of Marco’s lemonade
theorem calories_350_grams_mint_lemonade : calories_in_350_grams = 440 := 
by
  sorry

end calories_350_grams_mint_lemonade_l810_810716


namespace problem_solution_l810_810302

noncomputable def f : ℝ → ℝ := sorry

theorem problem_solution : (∃ f : ℝ → ℝ, (∀ x y : ℝ, f(x) * f(y) - f(x + y) = x - y) ∧
                              (let values := (set_of (λ v, v = f(3))) in
                               ∃ n s, (n = values.card) ∧ (s = values.sum) ∧ (n * s = -3))) :=
{
  sorry
}

end problem_solution_l810_810302


namespace smallest_among_list_l810_810053

theorem smallest_among_list : ∀ (l : List ℚ), l = [0, -2/3, 1, -3] → l.min = -3 :=
by
  intros l hl
  rw [hl]
  exact sorry

end smallest_among_list_l810_810053


namespace train_cross_platform_time_l810_810479

theorem train_cross_platform_time
  (train_length : ℕ) (time_to_cross_pole : ℕ) (platform_length : ℕ)
  (h_train_length : train_length = 300)
  (h_time_to_cross_pole : time_to_cross_pole = 18)
  (h_platform_length : platform_length = 350) :
  let speed := train_length / time_to_cross_pole,
      total_distance := train_length + platform_length,
      time_to_cross_platform := total_distance / speed
  in  time_to_cross_platform = 39 :=
by
  sorry

end train_cross_platform_time_l810_810479


namespace meetings_before_first_lap_l810_810851

/-- 
Given two boys starting from the same point on a circular track with a circumference of 190 feet, 
running in opposite directions with speeds of 7 ft/s and 12 ft/s respectively,
prove that the number of meetings (excluding start and finish) before the faster boy completes his first lap is 1.
-/
theorem meetings_before_first_lap (circumference : ℕ) (speed1 speed2 : ℕ) (h_circ : circumference = 190) (h_speed1 : speed1 = 7) (h_speed2 : speed2 = 12) :
  let relative_speed := speed1 + speed2 in
  let time_to_complete_lap := circumference / speed2 in
  (relative_speed * time_to_complete_lap) / circumference = 1 := by
  sorry

end meetings_before_first_lap_l810_810851


namespace number_of_friends_l810_810338

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l810_810338


namespace sum_f_eq_45_2_l810_810572

def f (n : ℕ) : ℝ := if (real.log n / real.log 4).is_rational then real.log n / real.log 4 else 0

theorem sum_f_eq_45_2 : ∑ n in finset.range (1023 + 1), f n = 45 / 2 :=
by
  sorry

end sum_f_eq_45_2_l810_810572


namespace birthday_friends_count_l810_810368

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l810_810368


namespace number_of_real_solutions_l810_810690

noncomputable def system_of_equations_solutions_count (x : ℝ) : Prop :=
  3 * x^2 - 45 * (⌊x⌋:ℝ) + 60 = 0 ∧ 2 * x - 3 * (⌊x⌋:ℝ) + 1 = 0

theorem number_of_real_solutions : ∃ (x₁ x₂ x₃ : ℝ), system_of_equations_solutions_count x₁ ∧ system_of_equations_solutions_count x₂ ∧ system_of_equations_solutions_count x₃ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ :=
sorry

end number_of_real_solutions_l810_810690


namespace number_of_valid_paths_l810_810063

structure Grid := 
  (rows : ℕ)
  (cols : ℕ)
  (black_dots : ℕ)

def initial_point := (0, 0) -- point A at the top-left corner assuming (0,0) coordinates.

def is_valid_path (path : list (ℕ × ℕ)) (grid : Grid) : Prop :=
  (path.head = initial_point) ∧ 
  (path.last = initial_point) ∧
  (∀ p ∈ path.tail.drop_last, p ≠ initial_point) ∧
  (list.nodup path) ∧ 
  (path.length = grid.black_dots)

def valid_paths (grid : Grid) : finset (list (ℕ × ℕ)) :=
  {path | is_valid_path path grid}.to_finset

theorem number_of_valid_paths : 
  valid_paths {rows := 3, cols := 3, black_dots := 16} = 12 :=
by sorry

end number_of_valid_paths_l810_810063


namespace tree_volume_estimation_proof_l810_810935

noncomputable def average_root_cross_sectional_area (x_i : list ℝ) := (x_i.sum) / (x_i.length)
noncomputable def average_volume (y_i : list ℝ) := (y_i.sum) / (y_i.length)
noncomputable def correlation_coefficient (x_i y_i : list ℝ) : ℝ :=
  let n := x_i.length in
  let x_bar := average_root_cross_sectional_area x_i in
  let y_bar := average_volume y_i in
  let numerator := (list.zip x_i y_i).sum (λ ⟨x, y⟩, (x - x_bar) * (y - y_bar)) in
  let denominator_x := (x_i.sum (λ x, (x - x_bar)^2)) in
  let denominator_y := (y_i.sum (λ y, (y - y_bar)^2)) in
  numerator / ((denominator_x * denominator_y).sqrt)

noncomputable def total_volume_estimate (total_area avg_y avg_x : ℝ) := (avg_y / avg_x) * total_area

theorem tree_volume_estimation_proof :
  let x_i := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06] in
  let y_i := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40] in
  let total_area := 186 in
  average_root_cross_sectional_area x_i = 0.06 ∧
  average_volume y_i = 0.39 ∧
  correlation_coefficient x_i y_i ≈ 0.97 ∧
  total_volume_estimate total_area 0.39 0.06 = 1209 :=
by
  sorry

end tree_volume_estimation_proof_l810_810935


namespace total_matches_l810_810647

theorem total_matches (n : ℕ) (h : n = 14) : 
  let matches := n * (n - 1) 
  in matches = 182 :=
by
  let n := 14
  show 14 * (14 - 1) = 182 from
    sorry

end total_matches_l810_810647


namespace lattice_labeling_l810_810088
-- Import the required libraries

-- Define the main problem conditions and theorem
theorem lattice_labeling (ℓ : ℕ × ℕ → ℕ)
  (h0 : ℓ (0,0) = 0)
  (h1 : ∀ x y, {ℓ (x, y), ℓ (x, y+1), ℓ (x+1, y)} = {(n : ℕ), n, n+1, n+2 -> n})
  : ∃ k : ℕ, ℓ (2000, 2024) = k ∧ k ≤ 6048 ∧ k % 3 = 0 := sorry

end lattice_labeling_l810_810088


namespace compute_expression_l810_810965

theorem compute_expression :
  ((-1)^2 + |(-2)| + (-3)^0 - (1 / 5)^(-1)) = -2 := by
  sorry

end compute_expression_l810_810965


namespace parabola_equation_l810_810565

theorem parabola_equation :
  ∃ (a b c : ℝ), (∀ (x : ℝ), y = 3 * x^2 + (-18) * x + 25) ∧ 
                 ∀ (x : ℝ), (y = a * (x-3)^2 - 2) &&
                 a = 3 &&
                 (3 * (4-3)^2 - 2 = 1) &&
                 (3x^2 - 18x + 25 = y) :=
by sorry

end parabola_equation_l810_810565


namespace greatest_x_lcm_l810_810799

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l810_810799


namespace existence_of_close_points_l810_810727

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the set S as an infinite set of points
noncomputable def S : set Point := sorry

-- Define the conditions
axiom finite_points_in_unit_square :
  ∀ (top_left: Point), { p : Point | p ∈ S ∧ top_left.x ≤ p.x ∧ p.x < top_left.x + 1 ∧ top_left.y ≤ p.y ∧ p.y < top_left.y + 1 }.finite

-- Define distance function
def dist (p1 p2 : Point) : ℝ :=
  real.sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the proof statement
theorem existence_of_close_points :
  ∃ (A B : Point) (A_ne_B : A ≠ B), (A ∈ S ∧ B ∈ S) ∧
  ∀ X ∈ S, dist X A ≥ 0.999 * dist A B ∧ dist X B ≥ 0.999 * dist A B :=
sorry

end existence_of_close_points_l810_810727


namespace angle_BOC_eq_angle_AOD_l810_810162

-- Define points in the quadrilateral and intersections as given
variables {A B C D E F P O : Type} 
variables [ConvexQuadrilateral A B C D E F P O]

-- State the conditions
variables (intersect_AB_CD : line A B ∩ line C D = E)
variables (intersect_AD_BC : line A D ∩ line B C = F)
variables (intersect_AC_BD : line A C ∩ line B D = P)
variables (perp_from_P_to_EF : ∃ (O : Type), perpendicular_from P (line E F) = O ∧ ∃ l : line, on_line O l ∧ l = line E F)

-- Prove the desired angle equality
theorem angle_BOC_eq_angle_AOD :
  ∠ B O C = ∠ A O D :=
begin
  sorry
end

end angle_BOC_eq_angle_AOD_l810_810162


namespace num_five_student_committees_l810_810121

theorem num_five_student_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 5) : choose n k = 56 :=
by
  rw [h_n, h_k]
  -- rest of the proof would go here
  sorry

end num_five_student_committees_l810_810121


namespace alice_paper_cranes_l810_810941

theorem alice_paper_cranes (T : ℕ)
  (h1 : T / 2 - T / 10 = 400) : T = 1000 :=
sorry

end alice_paper_cranes_l810_810941


namespace crayons_left_l810_810835

-- Define the initial number of crayons and the number taken
def initial_crayons : ℕ := 7
def crayons_taken : ℕ := 3

-- Prove the number of crayons left in the drawer
theorem crayons_left : initial_crayons - crayons_taken = 4 :=
by
  sorry

end crayons_left_l810_810835


namespace triangle_area_sqrt2_l810_810644

noncomputable def area_triangle (a b C : ℝ) : ℝ := 
  (1 / 2) * a * b * Real.sin C

theorem triangle_area_sqrt2 (a b : ℝ) (h_geom_seq : a * b = 4) : 
  area_triangle a b (Real.pi / 4) = Real.sqrt 2 :=
by
  have hC : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := Real.sin_pi_div_four
  rw [area_triangle, h_geom_seq, hC]
  simp
  sorry

end triangle_area_sqrt2_l810_810644


namespace expand_product_l810_810561

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 := 
by 
  sorry

end expand_product_l810_810561


namespace jim_traveled_distance_l810_810280

noncomputable section

-- Define the variables and constants
def skateboard_speed := 15  -- in km/h
def walk_speed := 4        -- in km/h
def total_time := 56 / 60  -- total time in hours

-- Define the function to calculate total distance
def total_distance (d: ℝ) : ℝ :=
  ( (2 / 3) * d / skateboard_speed ) + ( (1 / 3) * d / walk_speed )

-- The theorem we want to prove
theorem jim_traveled_distance :
  ∃ d: ℝ, total_distance d = total_time ∧ (Real.floor ((d + 0.25) * 2) / 2 = 7.5) :=
sorry

end jim_traveled_distance_l810_810280


namespace part_I_part_II_part_III_l810_810316

noncomputable def f (x a b : ℝ) := (x^3) / 3 - (a + 1) * x^2 + 4 * a * x + b

theorem part_I (a b : ℝ) (h₁ : ∃ x, x = 3 ∧ f x a b = 1 / 2 ∧ ∂ f x a b / ∂ x = 0) :
  a = 3 / 2 ∧ b = -4 :=
sorry

theorem part_II (a : ℝ) :
  if a > 1 then (intervals_of_monotonic_increase a = (-∞, 2) ∪ (2 * a, +∞)) ∧
  if a = 1 then (intervals_of_monotonic_increase a = -∞ ∪ ∞) ∧
  if a < 1 then (intervals_of_monotonic_increase a = (-∞, 2 * a) ∪ (2, +∞)) :=
sorry

theorem part_III (a : ℝ) (h₂ : ∃ x, x ∈ (-1, 1) ∧ ∂ f x a b / ∂ x = 0) :
  - 1 / 2 < a < 1 / 2 :=
sorry

end part_I_part_II_part_III_l810_810316


namespace train_speed_kmph_l810_810492

theorem train_speed_kmph (length time : ℝ) (h_length : length = 90) (h_time : time = 8.999280057595392) :
  (length / time) * 3.6 = 36.003 :=
by
  rw [h_length, h_time]
  norm_num
  sorry -- the norm_num tactic might simplify this enough, otherwise further steps would be added here.

end train_speed_kmph_l810_810492


namespace quadrilateral_pyramid_exists_l810_810975

open EuclideanGeometry

def point := ℝ × ℝ × ℝ

variables (A B C D M N : point)

-- Conditions
axiom AD_perpendicular_ABC 
  (A B C D : point) : ∃ (normal_vector : point), 
    let x_axis : point := (1, 0, 0),
        y_axis : point := (0, 1, 0),
        z_axis : point := (0, 0, 1) in
    D - A = z_axis ∧ B - A = x_axis ∧ C - A = y_axis

axiom M_on_AB 
  (A B M : point) : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ M = (1 - t) • A + t • B

axiom N_on_AC 
  (A C N : point) : ∃ (u : ℝ), 0 < u ∧ u < 1 ∧ N = (1 - u) • A + u • C

-- Proof Problem Statement
theorem quadrilateral_pyramid_exists 
  (A B C D M N : point)
  (h1 : AD_perpendicular_ABC A B C D)
  (h2 : M_on_AB A B M) 
  (h3 : N_on_AC A C N) : 
  ∃ (quad_pyramid : Prop), 
    quad_pyramid ∧ 
    (let BMN := {B, M, N, C};
         planes BMD CND : set (set point) := 
      λ V, ∃ (uv : ℝ),
          V = uv • B + (1 - uv) • M) in 

    ∃ (h : plane BMD ⊥ BMN ∧ plane CND ⊥ BMN), 
    true :=
sorry

end quadrilateral_pyramid_exists_l810_810975


namespace chords_return_to_start_l810_810749

theorem chords_return_to_start (m : ℕ) (h : ∀ (AB BC : ℝ), AB = BC → m * 60 = 360 → m = 3 ) :
  ∃ (n : ℕ), n = 3 :=
by
  use 3
  apply h
  { sorry },
  { sorry }

end chords_return_to_start_l810_810749


namespace neither_prime_nor_composite_probability_l810_810641

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

def is_neither_prime_nor_composite (n : ℕ) : Prop :=
  ¬is_prime n ∧ ¬is_composite n

theorem neither_prime_nor_composite_probability : 
  let draw_one : ℕ := 1 in
  let total_pieces : ℕ := 100 in
  let count_neither_prime_nor_composite := (finset.range 101).filter is_neither_prime_nor_composite in
  (count_neither_prime_nor_composite.card = 1 ∧ 
  (1 : ℝ) / total_pieces = 1 / 100) := by
  sorry

end neither_prime_nor_composite_probability_l810_810641


namespace arrows_to_travel_l810_810156

def directed_graph (V : Type) := 
  ∀ (a b : V), (a ≠ b) → (∃ (p : V → V → Prop), 
  (p a b ∨ (∃ c, p a c ∧ p c b)) ∧ ¬(p a b ∧ p b a))

theorem arrows_to_travel (n : ℕ) (hn : n > 4) :
  ∃(V : Type), ∃(G : directed_graph V), fintype.card V = n :=
by 
  sorry

end arrows_to_travel_l810_810156


namespace find_c_value_l810_810704

theorem find_c_value
    (a b c : ℝ)
    (h1 : (27 * a) - (6 * b) - 3 = 0)
    (h2 : (3 * a) + (2 * b) - 3 = 8)
    (h3 : ∀ x, (f x : ℝ) = a * x^3 + b * x^2 - 3 * x + c)
    (h4 : ∀ x, (f_derivative x : ℝ) = 3 * a * x^2 + 2 * b * x - 3)
    (h5 : ∀ x : ℝ, f_derivative x = 0 ↔ x = -3 ∨ x = 1 / 3)
    (h6 : f (-1) = 10)
    : c = 4 := sorry

# where f and f_derivative match the definitions
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^3 + b * x^2 - 3 * x + c
def f_derivative (x : ℝ) (a b : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x - 3

end find_c_value_l810_810704


namespace parallelogram_trapezoid_or_parallelogram_l810_810387

-- Definition of a parallelogram
structure Parallelogram (ℝ : Type*) [AddCommGroup ℝ] [Module ℝ ℝ] :=
(A B C D : ℝ)
(parallel_AB_CD : A - B = D - C)
(parallel_AD_BC : A - D = B - C)

-- Points M, N, K, L as intersection points
structure IntersectionPoints (ℝ : Type*) [LinearOrderedField ℝ] :=
(M N K L : ℝ)

-- Problem statement in Lean 4
theorem parallelogram_trapezoid_or_parallelogram
  (ℝ : Type*) [LinearOrderedField ℝ]
  (P : Parallelogram ℝ)
  (M N K L : IntersectionPoints ℝ) :
  -- The points M, N, K, L are vertices of either
  -- a trapezoid or a parallelogram
  sorry

end parallelogram_trapezoid_or_parallelogram_l810_810387


namespace greatest_value_of_x_l810_810811

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l810_810811


namespace friends_at_birthday_l810_810361

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l810_810361


namespace birthday_friends_count_l810_810370

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l810_810370


namespace friends_at_birthday_l810_810364

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l810_810364


namespace vehicle_value_this_year_l810_810430

variable (V_last_year : ℝ) (V_this_year : ℝ)

-- Conditions
def last_year_value : ℝ := 20000
def this_year_value : ℝ := 0.8 * last_year_value

theorem vehicle_value_this_year :
  V_last_year = last_year_value →
  V_this_year = this_year_value →
  V_this_year = 16000 := sorry

end vehicle_value_this_year_l810_810430


namespace intersection_sum_is_six_l810_810240

-- Define the coordinates of points A, B, and C
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Midpoints D and E
def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Equations of lines AE and CD
def lineAE (x : ℝ) : ℝ := -8 / 5 * x + 8
def lineCD (x : ℝ) : ℝ := -2 / 5 * x + 4

-- Prove that the sum of the coordinates of F is 6
theorem intersection_sum_is_six :
  let F : ℝ × ℝ := (
    (lineAE (10 / 3) + lineCD (10 / 3)) / 2,
    (lineAE (8 / 3) + lineCD (8 / 3)) / 2
  ) in
  F.1 + F.2 = 6 := by
  -- Perform the proof here
  sorry

end intersection_sum_is_six_l810_810240


namespace triangle_carpet_area_l810_810752

theorem triangle_carpet_area (base_feet height_feet feet_per_yard : ℝ)
  (h_base : base_feet = 15)
  (h_height : height_feet = 12)
  (h_feet_per_yard : feet_per_yard = 3) :
  let base_yards := base_feet / feet_per_yard,
      height_yards := height_feet / feet_per_yard,
      area_yards := (1 / 2) * base_yards * height_yards in
  area_yards = 10 :=
by
  -- We are skipping the proof here.
  sorry

end triangle_carpet_area_l810_810752


namespace friends_at_birthday_l810_810366

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l810_810366


namespace arithmetic_sequence_a9_l810_810256

variable {a : ℕ → ℤ} {d : ℤ}

-- Condition 1: Definition of arithmetic sequence term difference
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

-- Condition 2: Specific values in the sequence
axiom a3_2 : a 3 - a 2 = -2
axiom a7_eq : a 7 = -2

-- Theorem: To prove a9 equals -6
theorem arithmetic_sequence_a9 : ∃ d, arithmetic_sequence a d ∧ d = -2 ∧ a 9 = -6 :=
  by
  -- Unwrap the conditions and definitions
  use -2
  split
  · sorry  -- (Prove that the sequence has a common difference of -2)
  · split
    · exact a3_2
    · exact sorry  -- (Use the sequence property to prove a 9 = -6)

end arithmetic_sequence_a9_l810_810256


namespace domain_of_f_l810_810757

def f (x : ℝ) : ℝ := sqrt (4 * x + 1)

theorem domain_of_f (x : ℝ) : (4 * x + 1 ≥ 0) ↔ (x ∈ Set.Ici (-1 / 4)) := by
  sorry

end domain_of_f_l810_810757


namespace balls_in_boxes_l810_810653

theorem balls_in_boxes:
  ∃ (x y z : ℕ), 
  x + y + z = 320 ∧ 
  6 * x + 11 * y + 15 * z = 1001 ∧
  x > 0 ∧ y > 0 ∧ z > 0 :=
by
  sorry

end balls_in_boxes_l810_810653


namespace part1_part2_l810_810400

def f (x : ℝ) : ℝ := 3 / (9 ^ x + 3)

theorem part1 (x : ℝ) : f(x) + f(1 - x) = 1 := by
  sorry

theorem part2 : 
  let S := ∑ n in finset.range 2016, f ((n + 1) / 2017) in 
  S = 1008 := by
  sorry

end part1_part2_l810_810400


namespace band_gigs_count_l810_810005

-- Definitions of earnings per role and total earnings
def leadSingerEarnings := 30
def guitaristEarnings := 25
def bassistEarnings := 20
def drummerEarnings := 25
def keyboardistEarnings := 20
def backupSingerEarnings := 15
def totalEarnings := 2055

-- Calculate total per gig earnings
def totalPerGigEarnings :=
  leadSingerEarnings + guitaristEarnings + bassistEarnings + drummerEarnings + keyboardistEarnings + backupSingerEarnings

-- Statement to prove the number of gigs played is 15
theorem band_gigs_count :
  totalEarnings / totalPerGigEarnings = 15 := 
by { sorry }

end band_gigs_count_l810_810005


namespace mean_of_arithmetic_sequence_l810_810537

theorem mean_of_arithmetic_sequence : 
  let a_n := λ n => 7 + (n - 1) in
  let first_term := 7 in
  let last_term := 7 + (35 - 1) in
  let sum_of_terms := (35 / 2) * (first_term + last_term) in
  let mean := sum_of_terms / 35 in
  mean = 24 := by
  let a_n := λ n => 7 + (n - 1)
  let first_term := 7
  let last_term := 7 + (35 - 1)
  let sum_of_terms := (35 / 2) * (first_term + last_term)
  let mean := sum_of_terms / 35
  sorry

end mean_of_arithmetic_sequence_l810_810537


namespace median_of_list_l810_810450

theorem median_of_list : 
  let list := (List.range (3031)).map (fun x => x) ++ (List.range (3031)).map (fun x => x*x)
  list = list.sort <|
  let median := if list.length % 2 = 0 then (list[list.length / 2 - 1] + list[list.length / 2]) / 2
                else list[list.length / 2]
  median = 2975.5 := sorry

end median_of_list_l810_810450


namespace common_chord_length_of_two_circles_l810_810596

-- Define the equations of the circles C1 and C2
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 2 * y - 4 = 0
def circle2 (x y : ℝ) : Prop := (x + 3 / 2)^2 + (y - 3 / 2)^2 = 11 / 2

-- The theorem stating the length of the common chord
theorem common_chord_length_of_two_circles :
  ∃ l : ℝ, (∀ (x y : ℝ), circle1 x y ↔ circle2 x y) → l = 2 :=
by simp [circle1, circle2]; sorry

end common_chord_length_of_two_circles_l810_810596


namespace ticket_cost_is_25_l810_810498

-- Define the given conditions
def num_tickets_first_show : ℕ := 200
def num_tickets_second_show : ℕ := 3 * num_tickets_first_show
def total_tickets : ℕ := num_tickets_first_show + num_tickets_second_show
def total_revenue_in_dollars : ℕ := 20000

-- Claim to prove
theorem ticket_cost_is_25 : ∃ x : ℕ, total_tickets * x = total_revenue_in_dollars ∧ x = 25 :=
by
  -- sorry is used here to skip the proof
  sorry

end ticket_cost_is_25_l810_810498


namespace billing_function_low_billing_function_high_calculate_160_kwh_calculate_consumption_l810_810899

-- Define the conditions for the billing rates
def billing_rate_low (x : ℕ) (h : 0 ≤ x ∧ x ≤ 200) : ℝ := 0.6 * x
def billing_rate_high (x : ℕ) (h : 200 < x) : ℝ := 0.65 * x - 10

-- The function relationships
theorem billing_function_low (x : ℕ) (h : 0 ≤ x ∧ x ≤ 200) : billing_rate_low x h = 0.6 * x :=
by
  sorry

theorem billing_function_high (x : ℕ) (h : 200 < x) : billing_rate_high x h = 0.65 * x - 10 :=
by
  sorry

-- Money to pay for 160 kilowatt-hours
theorem calculate_160_kwh : billing_rate_low 160 (by decide) = 96 :=
by
  sorry

-- Calculation for 146 yuan payment
theorem calculate_consumption (y : ℝ) (h : y = 146) : ∃ x, billing_rate_high x (by decide) = y :=
by
  use 240
  sorry

end billing_function_low_billing_function_high_calculate_160_kwh_calculate_consumption_l810_810899


namespace problem1_problem2_l810_810198

noncomputable def f (x : ℝ) (a : ℝ) := (1 / 2) * x^2 + 2 * a * x
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) := 3 * a^2 * Real.log x + b

theorem problem1 (a b x₀ : ℝ) (h : x₀ = a):
  a > 0 →
  (1 / 2) * x₀^2 + 2 * a * x₀ = 3 * a^2 * Real.log x₀ + b →
  x₀ + 2 * a = 3 * a^2 / x₀ →
  b = (5 * a^2 / 2) - 3 * a^2 * Real.log a := sorry

theorem problem2 (a b : ℝ):
  -2 ≤ b ∧ b ≤ 2 →
  ∀ x > 0, x < 4 →
  ∀ x, x - b + 3 * a^2 / x ≥ 0 →
  a ≥ Real.sqrt 3 / 3 ∨ a ≤ -Real.sqrt 3 / 3 := sorry

end problem1_problem2_l810_810198


namespace number_of_friends_l810_810326

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l810_810326


namespace fixed_point_trajectory_l810_810163

theorem fixed_point_trajectory 
  (F : ℝ × ℝ) (l : ℝ → Prop) (P : ℝ × ℝ)
  (hF : F = (1, 0)) (hl : ∀ x, l x ↔ x = 4) 
  (hP : ∀ x y, P = (x, y) → (real.sqrt ((x-1)^2 + y^2)) / |x-4| = 1/2) : 
  (∀ x y, P = (x, y) → x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ Q : ℝ × ℝ, (Q = (1, 0) ∨ Q = (7, 0)) ∧ (∀ M N : ℝ × ℝ, (M = (4, 6*y1 / (m*y1+3)) ∧ N = (4, 6*y2 / (m*y2+3))) →
  (Q.1 - 4)^2 + 36*y1*y2 / (m^2*y1*y2 + 3*m*(y1+y2) + 9) = 0)) :=
begin
  sorry
end

end fixed_point_trajectory_l810_810163


namespace greatest_possible_x_max_possible_x_l810_810776

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l810_810776


namespace smallest_n_for_multiple_of_9_l810_810390

theorem smallest_n_for_multiple_of_9 (x y : ℤ) (hx : x ≡ 4 [ZMOD 9]) (hy : y ≡ -4 [ZMOD 9]) : ∃ (n : ℕ), n > 0 ∧ x^2 + x * y + y^2 + n ≡ 0 [ZMOD 9] :=
begin
  use 2,
  sorry
end

end smallest_n_for_multiple_of_9_l810_810390


namespace orthocenter_locus_l810_810062

noncomputable def A (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) : ℝ × ℝ :=
  (x, real.sqrt (1 - x^2))

def B : ℝ × ℝ := (-3, -1)
def C : ℝ × ℝ := (2, -1)

theorem orthocenter_locus :
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 →
    ∃ y : ℝ, y = (6 - x - x^2) / (1 + real.sqrt (1 - x^2)) - 1 :=
by
  intro x hx
  use (6 - x - x^2) / (1 + real.sqrt (1 - x^2)) - 1
  sorry

end orthocenter_locus_l810_810062


namespace meg_final_result_l810_810719

theorem meg_final_result :
  let start := 100
  let first_increase := (20 / 100) * start
  let after_first_increase := start + first_increase
  let second_increase := (50 / 100) * after_first_increase
  let final_result := after_first_increase + second_increase
  final_result = 180 :=
by
  let start := 100
  let first_increase := (20 / 100) * start
  let after_first_increase := start + first_increase
  let second_increase := (50 / 100) * after_first_increase
  let final_result := after_first_increase + second_increase
  show final_result = 180 from
  sorry

end meg_final_result_l810_810719


namespace linear_correlation_is_very_strong_regression_line_residual_first_sample_effect_of_exclusion_l810_810029

noncomputable def resident_income : List ℝ := [32.2, 31.1, 32.9,  35.7,  37.1,  38.0,  39.0,  43.0,  44.6,  46.0]

noncomputable def goods_sales : List ℝ := [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0]

noncomputable def sum_x : ℝ := 379.6
noncomputable def sum_y : ℝ := 391
noncomputable def sum_x_squared_dev : ℝ := 246.904
noncomputable def sum_y_squared_dev : ℝ := 568.9
noncomputable def sum_xy_dev : ℝ := r * math.sqrt(sum_y_squared_dev * sum_x_squared_dev) where r := 0.95

-- Prove the linear correlation
theorem linear_correlation_is_very_strong : sum_xy_dev / (math.sqrt(sum_x_squared_dev) * math.sqrt(sum_y_squared_dev)) ≈ 0.95 → 
  (0.95 ≈ 1 ∨ 0.95 ≈ -1) :=
sorry

-- Calculate the regression line
theorem regression_line : sum_xy_dev / sum_x_squared_dev ≈ 1.44 ∧ 39.1 - 1.44 * 37.96 ≈ -15.56 → 
  (∀ x, y = 1.44 * x - 15.56) :=
sorry

-- Calculating the residual
theorem residual_first_sample : y - (1.44 * 32.2 - 15.56) ≈ -5.81 :=
sorry

-- Effect on regression line when excluding the sample point
theorem effect_of_exclusion (residual_lt_regression : y < 1.44 * x - 15.56):
  sum_xy_dev * math.sqrt(sum_x_squared_dev) * math.sqrt(sum_y_squared_dev) < sum_xy_dev * math.sqrt(sum_x_squared_dev) :=
sorry

end linear_correlation_is_very_strong_regression_line_residual_first_sample_effect_of_exclusion_l810_810029


namespace greatest_x_lcm_105_l810_810782

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l810_810782


namespace louis_age_currently_31_l810_810230

-- Definitions
variable (C L : ℕ)
variable (h1 : C + 6 = 30)
variable (h2 : C + L = 55)

-- Theorem statement
theorem louis_age_currently_31 : L = 31 :=
by
  sorry

end louis_age_currently_31_l810_810230


namespace probability_two_cards_sum_eleven_l810_810852

theorem probability_two_cards_sum_eleven
  (S : Finset ℕ)
  (hS : S = {2, 3, 4, 5, 6, 7, 8, 9, 10})
  (hDeck : ∀ x ∈ S, x ∈ range 2 11)
  (hStd : card S = 36)
  (hTotal : 52) :
  ∃ p : ℚ, p = 44 / 221 :=
by
  sorry

end probability_two_cards_sum_eleven_l810_810852


namespace payment_to_C_correct_l810_810466

variable (A_work_days : ℕ) (B_work_days : ℕ) (total_payment : ℝ) (total_days : ℕ) (C_payment : ℝ)

def payment_to_C (A_work_days B_work_days total_payment total_days : ℕ) : ℝ :=
  (total_payment : ℝ) * (1 - (3 / A_work_days + 3 / B_work_days)) 

theorem payment_to_C_correct :
  A_work_days = 6 →
  B_work_days = 8 →
  total_payment = 3200 →
  total_days = 3 →
  payment_to_C A_work_days B_work_days total_payment total_days = 400 := 
by
  intros hA hB hP hT
  rw [hA, hB, hP, hT]
  unfold payment_to_C
  norm_num
  sorry

end payment_to_C_correct_l810_810466


namespace find_hypotenuse_l810_810245

-- Let a, b be the legs of the right triangle, c be the hypotenuse.
-- Let h be the altitude to the hypotenuse and r be the radius of the inscribed circle.
variable (a b c h r : ℝ)

-- Assume conditions of a right-angled triangle
def right_angled (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Given the altitude to the hypotenuse
def altitude (h c : ℝ) : Prop :=
  ∃ a b : ℝ, right_angled a b c ∧ h = a * b / c

-- Given the radius of the inscribed circle
def inscribed_radius (r a b c : ℝ) : Prop :=
  r = (a + b - c) / 2

-- The proof problem statement
theorem find_hypotenuse (a b c h r : ℝ) 
  (h_right_angled : right_angled a b c)
  (h_altitude : altitude h c)
  (h_inscribed_radius : inscribed_radius r a b c) : 
  c = 2 * r^2 / (h - 2 * r) :=
  sorry

end find_hypotenuse_l810_810245


namespace fourth_tree_more_than_third_l810_810438

-- Definitions of the conditions
def first_tree_growth := 1
def second_tree_growth := first_tree_growth * 2
def third_tree_growth := 2

def total_growth_with_fourth (fourth_tree_growth : ℕ) : ℕ :=
  4 * first_tree_growth + 4 * second_tree_growth + 4 * third_tree_growth + 4 * fourth_tree_growth

-- Main statement to be proved
theorem fourth_tree_more_than_third :
  ∃ (fourth_tree_growth : ℕ), total_growth_with_fourth fourth_tree_growth = 32 ∧ fourth_tree_growth - third_tree_growth = 1 :=
begin
  sorry,
end

end fourth_tree_more_than_third_l810_810438


namespace measure_of_one_interior_angle_of_regular_nonagon_is_140_l810_810449

-- Define the number of sides for a nonagon
def number_of_sides_nonagon : ℕ := 9

-- Define the formula for the sum of the interior angles of a regular n-gon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- The sum of the interior angles of a nonagon
def sum_of_interior_angles_nonagon : ℕ := sum_of_interior_angles number_of_sides_nonagon

-- The measure of one interior angle of a regular n-gon
def measure_of_one_interior_angle (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- The measure of one interior angle of a regular nonagon
def measure_of_one_interior_angle_nonagon : ℕ := measure_of_one_interior_angle number_of_sides_nonagon

-- The final theorem statement
theorem measure_of_one_interior_angle_of_regular_nonagon_is_140 : 
  measure_of_one_interior_angle_nonagon = 140 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_nonagon_is_140_l810_810449


namespace measure_of_angle_y_is_40_l810_810257

-- Define the problem setup and conditions
variables (p q : Line) -- lines p and q
def lines_parallel : Prop := parallel p q
def angle_on_q : ℝ := 40

noncomputable def measure_of_angle_y : ℝ := 40

-- Formal statement of the problem
theorem measure_of_angle_y_is_40 (hpq : lines_parallel p q) (haq : angle_on_q = 40) :
  measure_of_angle_y = 40 :=
sorry

end measure_of_angle_y_is_40_l810_810257


namespace incorrect_C_l810_810712

variable (D : ℝ → ℝ)

-- Definitions to encapsulate conditions
def range_D : Set ℝ := {0, 1}
def is_even := ∀ x, D x = D (-x)
def is_periodic := ∀ T > 0, ∃ p, ∀ x, D (x + p) = D x
def is_monotonic := ∀ x y, x < y → D x ≤ D y

-- The proof statement
theorem incorrect_C : ¬ is_periodic D :=
sorry

end incorrect_C_l810_810712


namespace min_k_condition_l810_810238

open List

/-- Define the reordering of columns in Table 1 to Table 2. -/
def reorder_columns (table1 : list (list ℝ)) : list (list ℝ) :=
  conjugateTranspose (map sort (conjugateTranspose table1))

/-- The smallest positive integer k such that if ∑_{j=1}^{25} x_{ij} ≤ 1 for all
    1 ≤ i ≤ 100, then ∑_{j=1}^{25} x'_{ij} ≤ 1 for all i ≥ k, where x'_{ij} is the element 
    of ith row jth column in the ordered table. -/
theorem min_k_condition {table1 : list (list ℝ)}
  (h_table1_len : table1.length = 100)
  (h_col_len : ∀ row ∈ table1, row.length = 25)
  (h_nonneg : ∀ row ∈ table1, ∀ x ∈ row, 0 ≤ x)
  (h_row_sum_le1 : ∀ row ∈ table1, ∑ x in row, x ≤ 1) :
  ∃ k : ℕ, k = 97 ∧ ∀ table2 = reorder_columns table1,
    ∀ i ≥ k, ∑ x in table2.nth_le i sorry ≤ 1 :=
sorry

end min_k_condition_l810_810238


namespace problem1_problem2_l810_810196

def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

theorem problem1 (a : ℝ) (h_pos : 0 < a) : (∃ x ≥ 0, deriv (f a) x = 0) → a = 1 := 
sorry

theorem problem2 (a : ℝ) (h_pos : 0 < a) : (∀ x ≥ 0, f a x ≥ Real.log 2) -> 1 ≤ a := 
sorry

end problem1_problem2_l810_810196


namespace square_area_from_diagonal_l810_810499

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  (let s := d / Real.sqrt 2 in s * s) = 144 := by
sorry

end square_area_from_diagonal_l810_810499


namespace tonya_large_lemonade_sales_l810_810847

theorem tonya_large_lemonade_sales 
  (price_small : ℝ)
  (price_medium : ℝ)
  (price_large : ℝ)
  (total_revenue : ℝ)
  (revenue_small : ℝ)
  (revenue_medium : ℝ)
  (n : ℝ)
  (h_price_small : price_small = 1)
  (h_price_medium : price_medium = 2)
  (h_price_large : price_large = 3)
  (h_total_revenue : total_revenue = 50)
  (h_revenue_small : revenue_small = 11)
  (h_revenue_medium : revenue_medium = 24)
  (h_revenue_large : n = (total_revenue - revenue_small - revenue_medium) / price_large) :
  n = 5 :=
sorry

end tonya_large_lemonade_sales_l810_810847


namespace range_of_k_l810_810612

noncomputable def f : ℝ → ℝ
| x := if x ≤ 1 then Real.exp x else f (x - 1)

def g (k x : ℝ) : ℝ := k * x + 1

theorem range_of_k (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g k x1 ∧ f x2 = g k x2 ↔ ∃ k_range : Set ℝ, k ∈ k_range :=
sorry

end range_of_k_l810_810612


namespace sum_of_solutions_l810_810995

theorem sum_of_solutions (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → (1 / Real.sin x + 1 / Real.cos x = 4)) : 
  ∑ x in {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 1 / Real.sin x + 1 / Real.cos x = 4}, x = Real.pi :=
by
  sorry

end sum_of_solutions_l810_810995


namespace watch_current_price_l810_810021

-- Definitions based on conditions
def original_price : ℝ := 15
def first_reduction_rate : ℝ := 0.25
def second_reduction_rate : ℝ := 0.40

-- The price after the first reduction
def first_reduced_price : ℝ := original_price * (1 - first_reduction_rate)

-- The price after the second reduction
def final_price : ℝ := first_reduced_price * (1 - second_reduction_rate)

-- The theorem that needs to be proved
theorem watch_current_price : final_price = 6.75 :=
by
  -- Proof goes here
  sorry

end watch_current_price_l810_810021


namespace smallest_n_l810_810699

noncomputable def f (x : ℝ) : ℝ := abs (3 * (x - x.floor) - 1.5)

def fractional_part (x : ℝ) : ℝ := x - x.floor

def condition (n : ℕ) : Prop :=
  ∃ (x : ℝ) [3000 solutions to n * f (x * f x) = x]

theorem smallest_n : ∃ n, condition n ∧ ∀ m, condition m → m ≥ n := 
begin
  sorry
end

end smallest_n_l810_810699


namespace contradiction_method_example_l810_810874

theorem contradiction_method_example (p q : ℝ) (h : p^3 + q^3 = 2) : p + q ≤ 2 :=
by
  -- We will assume the opposite to derive a contradiction
  have contra : (p + q > 2) → False := sorry
  apply classical.by_contradiction
  exact contra

end contradiction_method_example_l810_810874


namespace find_simple_interest_principal_l810_810422

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ n

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t / 100

theorem find_simple_interest_principal : 
  (simple_interest P 8 3 = 1 / 2 * compound_interest 4000 10 2) → 
  P = 1750 := 
by
  sorry

end find_simple_interest_principal_l810_810422


namespace sin_neg_angle_cos_neg_angle_trig_compute_l810_810961

theorem sin_neg_angle (θ : ℝ) : sin (-θ) = -sin θ := sorry

theorem cos_neg_angle (θ : ℝ) : cos (-θ) = cos θ := sorry

lemma sin_30 : sin (real.pi / 6) = 1 / 2 := sorry

lemma cos_30 : cos (real.pi / 6) = sqrt 3 / 2 := sorry

theorem trig_compute : 
  sin (-real.pi / 6) = -1 / 2 ∧ cos (-real.pi / 6) = sqrt 3 / 2 :=
by 
  have h1 : sin (-real.pi / 6) = -sin (real.pi / 6) := sin_neg_angle (real.pi / 6)
  have h2 : sin (real.pi / 6) = 1 / 2 := sin_30
  have h3 : cos (-real.pi / 6) = cos (real.pi / 6) := cos_neg_angle (real.pi / 6)
  have h4 : cos (real.pi / 6) = sqrt 3 / 2 := cos_30
  exact ⟨by rw [h1, h2], by rw [h3, h4]⟩

end sin_neg_angle_cos_neg_angle_trig_compute_l810_810961


namespace wendy_pictures_l810_810447

theorem wendy_pictures (album1_pics rest_albums albums each_album_pics : ℕ)
    (h1 : album1_pics = 44)
    (h2 : rest_albums = 5)
    (h3 : each_album_pics = 7)
    (h4 : albums = rest_albums * each_album_pics)
    (h5 : albums = 5 * 7):
  album1_pics + albums = 79 :=
by
  -- We leave the proof as an exercise
  sorry

end wendy_pictures_l810_810447


namespace integer_sequences_l810_810564

theorem integer_sequences (n : ℕ) (a : Fin n → ℕ) :
  (∀ i : Fin n, 1 ≤ a i ∧ a i ≤ n) ∧
  (∀ i j : Fin n, |a i - a j| = |i.val - j.val|) →
  (∀ i : Fin n, a i = i.val + 1) ∨ (∀ i : Fin n, a i = n - i.val) :=
by
  sorry

end integer_sequences_l810_810564


namespace square_area_from_diagonal_l810_810509

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) : ∃ A : ℝ, A = 144 :=
by
  let s := d / Real.sqrt 2
  have s_eq : s = 12 := by
    rw [h]
    field_simp
    norm_num
  use s * s
  rw [s_eq]
  norm_num
  sorry

end square_area_from_diagonal_l810_810509


namespace barbell_cost_l810_810285

theorem barbell_cost (num_barbells : ℤ) (total_money_given : ℤ) 
  (change_received : ℤ) (total_cost : ℤ) (each_barbell_cost : ℤ) 
  (h1 : num_barbells = 3) (h2 : total_money_given = 850) 
  (h3 : change_received = 40) (h4 : total_cost = total_money_given - change_received)
  (h5 : each_barbell_cost = total_cost / num_barbells) 
  : each_barbell_cost = 270 :=
by
  rw [h2, h3] at h4
  rw [← h4, h1] at h5
  exact calc 
    each_barbell_cost = (total_money_given - change_received) / num_barbells : h5
                    ... = (850 - 40) / 3 : by rw [h2, h3, h1]
                    ... = 810 / 3 : rfl
                    ... = 270 : rfl

end barbell_cost_l810_810285


namespace exactly_one_correct_l810_810944

def P1 : Prop := ∀ (r : ℚ) (x : ℝ), irrational x → irrational (r + x)
def P2 : Prop := ∀ (r : ℚ) (x : ℝ), irrational x → irrational (r * x)
def P3 : Prop := ∀ (x y : ℝ), irrational x → irrational y → irrational (x + y)
def P4 : Prop := ∀ (x y : ℝ), irrational x → irrational y → irrational (x * y)

theorem exactly_one_correct : (∃! i, i ∈ [1, 2, 3, 4] ∧ (if i = 1 then P1 else if i = 2 then P2 else if i = 3 then P3 else P4)) := by
  sorry

end exactly_one_correct_l810_810944


namespace number_of_friends_l810_810330

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l810_810330


namespace number_of_sets_l810_810409

theorem number_of_sets (A : Set ℕ) : ∃ s : Finset (Set ℕ), 
  (∀ x ∈ s, ({1} ⊂ x ∧ x ⊆ {1, 2, 3, 4})) ∧ s.card = 7 :=
sorry

end number_of_sets_l810_810409


namespace max_leap_years_l810_810543

theorem max_leap_years (years_per_period : ℕ) (total_years : ℕ) (leap_year_freq : ℕ) (h1 : total_years = 200) (h2 : leap_year_freq = 5) :
  (total_years / leap_year_freq) = 40 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul (by norm_num)

-- sorry

end max_leap_years_l810_810543


namespace find_f_pi_over_12_l810_810614

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.tan (3 * x + φ)

theorem find_f_pi_over_12 :
  ∃ φ : ℝ, |φ| ≤ Real.pi / 4 ∧
    (∀ x : ℝ, f x φ = f (- x - Real.pi / 9) φ) ∧ 
    f (Real.pi / 12) φ = 2 - Real.sqrt 3 :=
begin
  sorry
end

end find_f_pi_over_12_l810_810614


namespace line_intersects_midpoint_l810_810764

theorem line_intersects_midpoint (c : ℤ) : 
  (∃x y : ℤ, 2 * x - y = c ∧ x = (1 + 5) / 2 ∧ y = (3 + 11) / 2) → c = -1 := by
  sorry

end line_intersects_midpoint_l810_810764


namespace median_of_sequence_l810_810453

theorem median_of_sequence : 
  let sequence := (List.range 3030).map (λ n, n + 1) ++ (List.range 3030).map (λ n, (n + 1) ^ 2)
  let sorted_sequence := sequence.qsort (· ≤ ·)
  (sorted_sequence[3030-1] + sorted_sequence[3030]) / 2 = 2975.5 :=
by
  have h_seq_len : sequence.length = 6060 :=
    by sorry
  have h_sorted_seq : List.sorted (· ≤ ·) sorted_sequence :=
    by sorry
  have h_median_pos : sequence.length / 2 = 3030 :=
    by sorry
  have h_median_values : sorted_sequence[3030-1] = 2975 ∧ sorted_sequence[3030] = 2976 :=
    by sorry
  calc
    (sorted_sequence[3030-1] + sorted_sequence[3030]) / 2 = (2975 + 2976) / 2 := by rw [h_median_values]
    ... = 2975.5 := by norm_num

end median_of_sequence_l810_810453


namespace total_pages_l810_810025

theorem total_pages (total1 total2 : ℕ) (h1 : total1 = 385) (cond1 : ∀ n : ℕ, total1 = (∑ k in range n, (35 + 5 * k)) + 35) (cond2 : ∀ n : ℕ, total2 = (∑ k in range n, (45 + 5 * k)) + 40) : total1 = total2 :=
by sorry

end total_pages_l810_810025


namespace complex_quadrant_lemma_cos_neg4_sin_neg4_e_neg4i_in_second_quadrant_l810_810683

theorem complex_quadrant_lemma (x : ℝ) : e ^ (-4 * complex.I) = complex.of_real (cos (-4)) + complex.I * (sin (-4)) :=
by sorry

theorem cos_neg4 : cos (-4) < 0 :=
by sorry

theorem sin_neg4 : sin (-4) > 0 :=
by sorry

theorem e_neg4i_in_second_quadrant : ∃ (c : ℂ), c = e^(-4*complex.I) ∧ c.re < 0 ∧ c.im > 0 :=
by sorry

end complex_quadrant_lemma_cos_neg4_sin_neg4_e_neg4i_in_second_quadrant_l810_810683


namespace max_x_lcm_15_21_105_l810_810769

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l810_810769


namespace find_C_find_area_l810_810153

-- Define the triangle and conditions
variables {A B C a b c : ℝ}
axiom triangle_ABC : a^2 - ab - 2b^2 = 0

-- (1) Prove that C = π/3 when B = π/6
theorem find_C (hB : B = π / 6) (h : a^2 - ab - 2b^2 = 0) : C = π / 3 := 
  sorry

-- (2) Prove that the area of triangle ABC is 14√3 when C = 2π/3 and c = 14
theorem find_area (hC : C = 2 * π / 3) (hc : c = 14) (h : a^2 - ab - 2b^2 = 0) : 
  let area := (1 / 2) * a * b * Real.sin C in area = 14 * Real.sqrt 3 := 
  sorry

end find_C_find_area_l810_810153


namespace vector_magnitude_problem_l810_810204

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

def a : ℝ × ℝ := (-1, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_magnitude_problem :
  magnitude (vector_add a (scalar_mul 2 b)) = Real.sqrt 2 := by
  sorry

end vector_magnitude_problem_l810_810204


namespace circle_trajectory_l810_810902

theorem circle_trajectory (a b : ℝ) :
  ∃ x y : ℝ, (b - 3)^2 + a^2 = (b + 3)^2 → x^2 = 12 * y := 
sorry

end circle_trajectory_l810_810902


namespace rectangle_is_axisymmetric_and_centrally_symmetric_l810_810051

def Shape : Type
| EquilateralTriangle
| Rectangle
| Parallelogram
| RegularPentagon

def isAxisymmetric : Shape → Prop
| Shape.EquilateralTriangle := true
| Shape.Rectangle := true
| Shape.Parallelogram := false
| Shape.RegularPentagon := true

def isCentrallySymmetric : Shape → Prop
| Shape.EquilateralTriangle := false
| Shape.Rectangle := true
| Shape.Parallelogram := true
| Shape.RegularPentagon := false

theorem rectangle_is_axisymmetric_and_centrally_symmetric :
  ∃ shape, isAxisymmetric shape ∧ isCentrallySymmetric shape ∧ shape = Shape.Rectangle :=
by
  existsi Shape.Rectangle
  split
  · exact true.intro
  · split
    · exact true.intro
    · rfl

end rectangle_is_axisymmetric_and_centrally_symmetric_l810_810051


namespace greatest_x_lcm_105_l810_810781

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l810_810781


namespace alex_buys_17p3_pounds_of_corn_l810_810080

noncomputable def pounds_of_corn (c b : ℝ) : Prop :=
    c + b = 30 ∧ 1.05 * c + 0.39 * b = 23.10

theorem alex_buys_17p3_pounds_of_corn :
    ∃ c b, pounds_of_corn c b ∧ c = 17.3 :=
by
    sorry

end alex_buys_17p3_pounds_of_corn_l810_810080


namespace friends_at_birthday_l810_810362

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l810_810362


namespace height_relationship_l810_810858

noncomputable def h_ratio (r1 h1 r2 h2 : ℝ) : ℝ := (π * r1^2 * h1) / (π * r2 ^ 2 * h2)

theorem height_relationship (r1 h1 r2 h2 : ℝ) 
  (V_same : π * r1 ^ 2 * h1 = π * r2 ^ 2 * h2)
  (r2_rel : r2 = (6/5) * r1) : 
  h1 = 1.44 * h2 := by 
  -- Proof goes here
  sorry

end height_relationship_l810_810858


namespace tan_c_div_tan_a_plus_tan_c_div_tan_b_l810_810651

-- Define the setup of an acute triangle ABC with sides a, b, and c
variables {A B C : ℝ} [is_acute A] [is_acute B] [is_acute C]
variables (a b c : ℝ)

-- Define the condition of the problem
axiom acute_triangle_condition (h : (b / a) + (a / b) = 6 * real.cos C)

-- State the theorem we need to prove
theorem tan_c_div_tan_a_plus_tan_c_div_tan_b (h : (b / a) + (a / b) = 6 * real.cos C) : 
  (real.tan C / real.tan A) + (real.tan C / real.tan B) = 4 :=
sorry

end tan_c_div_tan_a_plus_tan_c_div_tan_b_l810_810651


namespace main_theorem_l810_810689

noncomputable def a (n : ℕ) : ℝ := sorry -- Definition to be drawn from the original condition
noncomputable def b (n : ℕ) : ℝ := sorry -- Definition to be drawn from the original condition

axiom conditions (n : ℕ) (h : n ≥ 101) :
  a n = real.sqrt ((1 / 100) * (∑ j in finset.range 100, b (n - j))) ∧
  b n = real.sqrt ((1 / 100) * (∑ j in finset.range 100, a (n - j)))

theorem main_theorem :
  ∃ n ≥ 101, |a n - b n| < 0.001 := by
  sorry

end main_theorem_l810_810689


namespace sum_of_a1_l810_810591

noncomputable def a : ℕ → ℝ := sorry -- sequence definition, should meet a_{n+1} = k a_n + 2k - 2
def k : ℝ := sorry -- constant k such that k ≠ 0 and k ≠ 1
axiom h_recursion : ∀ n : ℕ, a (n + 1) = k * a n + 2 * k - 2
axiom h_conditions : ∀ i : ℕ, 2 ≤ i ∧ i ≤ 5 → a i ∈ {-272, -32, -2, 8, 88, 888}

theorem sum_of_a1 : 
  let possible_a1 := {a_1 : ℝ | ∀ n : ℕ, a (n + 1) = k * a n + 2 * k - 2 ∧ (∃ i : ℕ, 2 ≤ i ∧ i ≤ 5 ∧ a i ∈ {-272, -32, -2, 8, 88, 888})} in
  ∑ a1 in possible_a1, a1 = 2402 / 3 := sorry

end sum_of_a1_l810_810591


namespace unit_vector_in_direction_l810_810152

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem unit_vector_in_direction {a b : ℝ × ℝ} (h_a : a = (5, 4)) (h_b : b = (3, 2)) :
  ∃ c : ℝ × ℝ, 
    (c = ((1 / magnitude (2 • a - 3 • b)) • (2 • a - 3 • b)) ∨ c = -((1 / magnitude (2 • a - 3 • b)) • (2 • a - 3 • b))) ∧ 
    c = (⟨(real.sqrt 5) / 5, 2 * (real.sqrt 5) / 5⟩) ∨ 
    c = (⟨-(real.sqrt 5) / 5, -2 * (real.sqrt 5) / 5⟩) :=
sorry

end unit_vector_in_direction_l810_810152


namespace remainder_when_a_squared_times_b_divided_by_n_l810_810303

theorem remainder_when_a_squared_times_b_divided_by_n (n : ℕ) (a : ℤ) (h1 : a * 3 ≡ 1 [ZMOD n]) : 
  (a^2 * 3) % n = a % n := 
by
  sorry

end remainder_when_a_squared_times_b_divided_by_n_l810_810303


namespace parity_E_2023_2024_2025_l810_810970

-- Definitions of the sequence and conditions
def E : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 0
| (n + 3) := E (n + 2) + E (n + 1)

-- Boolean function to check the parity (Even/Odd)
def is_even (n : ℕ) : Bool := n % 2 == 0

def is_odd (n : ℕ) : Bool := ¬is_even n

-- The theorem statement with the given conditions and expected outputs
theorem parity_E_2023_2024_2025 :
  is_even (E 2023) ∧ is_odd (E 2024) ∧ is_odd (E 2025) := 
by
  -- The proof is omitted by using sorry.
  sorry

end parity_E_2023_2024_2025_l810_810970


namespace max_min_y_l810_810990

noncomputable def y (x : ℝ) : ℝ := (Real.sin x)^(2:ℝ) + 2 * (Real.sin x) * (Real.cos x) + 3 * (Real.cos x)^(2:ℝ)

theorem max_min_y : 
  ∀ x : ℝ, 
  2 - Real.sqrt 2 ≤ y x ∧ y x ≤ 2 + Real.sqrt 2 :=
by sorry

end max_min_y_l810_810990


namespace cos_arcsin_of_3_5_tan_arcsin_of_3_5_l810_810072

noncomputable def cos_arcsin_result (x : ℝ) (h : 0 < x ∧ x < 1) : ℝ :=
  let θ := real.arcsin x in
  real.cos θ

noncomputable def tan_arcsin_result (x : ℝ) (h : 0 < x ∧ x < 1) : ℝ :=
  let θ := real.arcsin x in
  real.sin θ / real.cos θ

theorem cos_arcsin_of_3_5 : cos_arcsin_result (3 / 5) ⟨by norm_num, by norm_num⟩ = 4 / 5 :=
sorry

theorem tan_arcsin_of_3_5 : tan_arcsin_result (3 / 5) ⟨by norm_num, by norm_num⟩ = 3 / 4 :=
sorry

end cos_arcsin_of_3_5_tan_arcsin_of_3_5_l810_810072


namespace abc_sum_l810_810268

-- Definitions for the conditions
variables {a b c x y : ℝ}
def cond1 : 1 / (a + x) = 6
def cond2 : 1 / (b + y) = 3
def cond3 : 1 / (c + x + y) = 2

-- The theorem to be proved
theorem abc_sum : cond1 ∧ cond2 ∧ cond3 → c = a + b := sorry

end abc_sum_l810_810268


namespace total_flowers_l810_810834

def number_of_pots : ℕ := 141
def flowers_per_pot : ℕ := 71

theorem total_flowers : number_of_pots * flowers_per_pot = 10011 :=
by
  -- formal proof goes here
  sorry

end total_flowers_l810_810834


namespace sum_of_solutions_l810_810110

theorem sum_of_solutions (S : Set ℝ) (h : ∀ y ∈ S, y + 16 / y = 12) :
  ∃ t : ℝ, (∀ y ∈ S, y = 8 ∨ y = 4) ∧ t = 12 := by
  sorry

end sum_of_solutions_l810_810110


namespace sum_one_over_one_plus_nth_roots_of_unity_l810_810595

theorem sum_one_over_one_plus_nth_roots_of_unity (n : ℕ) (hn1 : n > 1) (hn2 : n % 2 = 1) :
  ∃ (x : Fin n → Complex), (∀ i : Fin n, x i = Complex.exp (2 * Real.pi * Complex.I * i / n)) ∧ 
  (∑ i in Finset.range (n-1), 1 / (1 + x ⟨i, Nat.lt_of_lt_pred hn1⟩)) = (n - 1) / 2 := sorry

end sum_one_over_one_plus_nth_roots_of_unity_l810_810595


namespace log_sum_geom_seq_l810_810185

noncomputable def geomSeq (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem log_sum_geom_seq (a r : ℝ) (h_pos: ∀ n, 0 < geomSeq a r n)
  (h_eq: geomSeq a r 1 * geomSeq a r 100 + geomSeq a r 3 * geomSeq a r 98 = 8) :
  (List.sum (List.map (fun n => Real.log2 (geomSeq a r n)) (List.range 100).map (· + 1))) = 100 := sorry

end log_sum_geom_seq_l810_810185


namespace gasoline_tank_capacity_l810_810911

theorem gasoline_tank_capacity
  (y : ℝ)
  (h_initial: y * (5 / 6) - y * (1 / 3) = 20) :
  y = 40 :=
sorry

end gasoline_tank_capacity_l810_810911


namespace least_number_to_subtract_l810_810105

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (r : ℕ) (h : n = 3674958423) (div : k = 47) (rem : r = 30) :
  (n % k = r) → 3674958423 % 47 = 30 :=
by
  sorry

end least_number_to_subtract_l810_810105


namespace square_area_l810_810504

theorem square_area {d : ℝ} (h : d = 12 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 144 ∧ ( ∃ s : ℝ, s = d / Real.sqrt 2 ∧ A = s^2 ) :=
by
  sorry

end square_area_l810_810504


namespace find_y_l810_810664

-- Definitions for the conditions in the problem
def straight_line (a b c : Point) (line : Line) : Prop :=
  line.contains a ∧ line.contains b ∧ line.contains c

def exterior_angle_theorem (triangle : Triangle) (exterior_angle non_adj_angle1 non_adj_angle2 : Angle) : Prop :=
  exterior_angle = non_adj_angle1 + non_adj_angle2

-- Points and angles given in the problem
variables (A B C D : Point)
variables (line_ABC : Line)
variables (angle_ABD angle_BCD angle_y : Angle)
variables (triangle_BCD : Triangle)

-- The conditions given in the problem
axiom line_ABC_is_straight : straight_line A B C line_ABC
axiom angle_ABD_is_148 : angle_ABD = 148
axiom angle_BCD_is_58 : angle_BCD = 58
axiom D_below_line : D.is_below line_ABC -- Assume a predicate that describes the position of D relative to the line.

-- The theorem to be proven
theorem find_y : 
  exterior_angle_theorem triangle_BCD angle_ABD angle_BCD angle_y →
  angle_y = 90 :=
by
  -- Initial setup using the conditions
  intros h1
  sorry

end find_y_l810_810664


namespace paolo_sevilla_birthday_l810_810344

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l810_810344


namespace existence_of_planes_l810_810605

-- Definitions for skew lines, parallel and perpendicular planes are assumed to be from geometry module or defined appropriately in Mathlib.
-- Here we just state the existence statements provided in the problem.

variable {Point Line Plane : Type}
variable [Geometry Point Line Plane]

def skew (m n : Line) : Prop := 
¬(∃ p : Point, p ∈ m ∧ p ∈ n) ∧ ¬(∃ (α : Plane), m ⊂ α ∧ n ⊂ α)

theorem existence_of_planes (m n : Line) (h_skew : skew m n) :
    (∃ (α : Plane), m ⊂ α ∧ n ∥ α) ∧
    (∃ (γ : Plane), (∀ p ∈ m, dist p γ = dist p γ)) ∧
    (∃ (α β : Plane), m ⊂ α ∧ n ⊂ β ∧ α ⊥ β) := sorry

end existence_of_planes_l810_810605


namespace area_of_ABC_l810_810265

noncomputable def TriangleABC :=
  let S_BQC := 1
  let AK_BK_ratio := (1 : ℕ, 2 : ℕ)
  let CL_LB_ratio := (2 : ℕ, 1 : ℕ)
  let S_ABC := (7 / 4 : ℚ)
  S_ABC

theorem area_of_ABC (h1 : AK_BK_ratio = (1, 2))
                    (h2 : CL_LB_ratio = (2, 1))
                    (h3 : S_BQC = 1) : 
        TriangleABC = (7 / 4 : ℚ) :=
by sorry

end area_of_ABC_l810_810265


namespace calculate_final_amount_l810_810955

def calculate_percentage (percentage : ℝ) (amount : ℝ) : ℝ :=
  percentage * amount

theorem calculate_final_amount :
  let A := 3000
  let B := 0.20
  let C := 0.35
  let D := 0.05
  D * (C * (B * A)) = 10.50 := by
    sorry

end calculate_final_amount_l810_810955


namespace platform_length_l810_810478

theorem platform_length 
  (train_length : ℕ)
  (time_cross_tree : ℕ)
  (time_cross_platform : ℕ)
  (speed : ℕ)
  : train_length = 1200 → 
    time_cross_tree = 120 → 
    time_cross_platform = 210 → 
    speed = train_length / time_cross_tree → 
    ∃ (platform_length : ℕ), platform_length = 900 :=
by
  intros
  have h1 : speed = 10 := by sorry -- Calculation of the speed
  have h2 : 210 * 10 = train_length + platform_length := by sorry -- Equation setup
  use 900
  sorry -- Final steps to establish platform_length = 900

end platform_length_l810_810478


namespace four_digit_numbers_ending_in_45_divisible_by_5_l810_810208

theorem four_digit_numbers_ending_in_45_divisible_by_5 :
  ∃ (a b c : ℕ), a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (∃ n, n = 10000 * a + 1000 * b + 100 * c + 45) ∧ ∃ (k : ℕ), 10000 * a + 1000 * b + 100 * c + 45 = 5 * k :=
  sorry

end four_digit_numbers_ending_in_45_divisible_by_5_l810_810208


namespace perpendicular_line_plane_l810_810179

variables {m : ℝ}

theorem perpendicular_line_plane (h : (4 / 2) = (2 / 1) ∧ (2 / 1) = (m / -1)) : m = -2 :=
by
  sorry

end perpendicular_line_plane_l810_810179


namespace greatest_x_lcm_l810_810813

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l810_810813


namespace largest_integer_n_apples_l810_810026

theorem largest_integer_n_apples (t : ℕ) (a : ℕ → ℕ) (h1 : t = 150) 
    (h2 : ∀ i : ℕ, 100 ≤ a i ∧ a i ≤ 130) :
  ∃ n : ℕ, n = 5 ∧ (∀ i j : ℕ, a i = a j → i = j → 5 ≤ i ∧ 5 ≤ j) :=
by
  sorry

end largest_integer_n_apples_l810_810026


namespace proof_problem_l810_810035

noncomputable def problem_data :=
  { n : ℕ := 10,
    x : Fin 10 → ℝ := λ i, [32.2, 31.1, 32.9, 35.7, 37.1, 38.0, 39.0, 43.0, 44.6, 46.0][i],
    y : Fin 10 → ℝ := λ i, [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0][i],
    sum_xi : ℝ := 379.6,
    sum_yi : ℝ := 391,
    sum_squares_x : ℝ := 246.904,
    sum_squares_y : ℝ := 568.9,
    sum_products : ℝ := sorry, -- m should be determined
    r : ℝ := 0.95 }

def correlation_is_strong (d : problem_data) : Prop :=
  d.r ≈ 0.95 → (d.r > 0.5) -- assuming a linear correlation is considered strong if r > 0.5

def empirical_regression_equation (d : problem_data) : Prop :=
  ∃ a b, b = 1.44 ∧ a = -15.56 ∧ ∀ x, d.y x = b * d.x x + a

def residual_and_slope_judgement (d : problem_data) : Prop :=
  let residual := d.y 0 - (1.44 * d.x 0 - 15.56) in
  residual = -5.81

theorem proof_problem (d : problem_data) :
  correlation_is_strong d ∧ empirical_regression_equation d ∧ residual_and_slope_judgement d :=
sorry

end proof_problem_l810_810035


namespace problem_l810_810697

noncomputable def b : ℕ → ℝ
| 0       := 7 / 17
| (n + 1) := 2 * (b n)^2 - 1

def seq_condition (n : ℕ) : Prop :=
∀ n > 0, |(List.range n).map b ∏| ≤ 1 / (2^n) * (17 / 4 * Real.sqrt 15)

theorem problem (d := (17 / (4 * Real.sqrt 15))) :
  100 * d ≈ 110 :=
by
  sorry

end problem_l810_810697


namespace probability_non_intersecting_rods_l810_810107

-- Define the given conditions
noncomputable def α := by sorry
noncomputable def β := by sorry

-- Proving the required probability
theorem probability_non_intersecting_rods :
  (∃ α β, 0 ≤ α ∧ α ≤ π ∧ 0 ≤ β ∧ β < 2 * π ∧
  (β < (π / 2 - α / 2)) ∧ (α < (π / 2 - β / 2))) →
  (1 - 1 / 12 = 11 / 12) :=
by sorry

end probability_non_intersecting_rods_l810_810107


namespace find_angle_BMC_l810_810395

noncomputable def triangle_angle (A B C : Point) : ℝ := sorry

noncomputable def circle_through_points (A B : Point) (C : Point) : Circle := sorry

noncomputable def point_on_median (A B C : Point) : Point := sorry

noncomputable def angle_BMC (A B C M : Point) : ℝ := sorry

-- Given conditions
variables (A B C M : Point) (α : ℝ)
-- Conditions
hypothesis (h1 : triangle_angle A B C = α) -- ∠A = α
hypothesis (h2 : circle_through_points A B C) -- Circle through A, B is tangent to BC at C
hypothesis (h3 : point_on_median A B C = M) -- Circle intersects median AD at M

-- Proof statement
theorem find_angle_BMC :
  angle_BMC A B C M = 180 - α :=
sorry

end find_angle_BMC_l810_810395


namespace cards_in_D_l810_810837

-- Define the problem in Lean
variable (A B C D : ℕ)

-- Assume the conditions given in the problem
axiom hA1 : A = C + 16 -- A's statement
axiom hB1 : D = C + 6  -- B's statement
axiom hC1 : A = D + 9  -- C's statement
axiom hD1 : D + 2 = 3 * C -- D's statement
axiom hL : min A (min B (min C D)) = (if hA1 then D else if hB1 then C else if hC1 then B else A)

theorem cards_in_D (hA1 : A = C + 16) (hB1 : D = C + 6) (hC1 : A = D + 9) (hD1 : D + 2 = 3 * C) (hL : min A (min B (min C D)) = (if hA1 then D else if hB1 then C else if hC1 then B else A)) : D = 10 :=
sorry

end cards_in_D_l810_810837


namespace dog_catches_rabbit_in_4_minutes_l810_810092

noncomputable def time_to_catch_up (dog_speed rabbit_speed head_start : ℝ) : ℝ :=
  head_start / (dog_speed - rabbit_speed) * 60

theorem dog_catches_rabbit_in_4_minutes :
  time_to_catch_up 24 15 0.6 = 4 :=
by
  unfold time_to_catch_up
  norm_num
  rfl

end dog_catches_rabbit_in_4_minutes_l810_810092


namespace monthly_expenses_last_month_was_2888_l810_810742

def basic_salary : ℕ := 1250
def commission_rate : ℚ := 0.10
def total_sales : ℕ := 23600
def savings_rate : ℚ := 0.20

theorem monthly_expenses_last_month_was_2888 :
  let commission := commission_rate * total_sales
  let total_earnings := basic_salary + commission
  let savings := savings_rate * total_earnings
  let monthly_expenses := total_earnings - savings
  monthly_expenses = 2888 := by
  sorry

end monthly_expenses_last_month_was_2888_l810_810742


namespace proof_problem_l810_810708

noncomputable def f (x : ℝ) : ℝ := x^2 + (-4) * x + (7 / 2)

theorem proof_problem :
  ∀ x ∈ Icc (1 : ℝ) (3 : ℝ), |f x| ≤ 1 / 2 →
  underbrace f (f ( \ldots f_{2017} (3 + √7) / 2) \ldots) = (3 - √7) / 2 :=
begin
  assume h : ∀ x ∈ Icc (1 : ℝ) (3 : ℝ), |f x| ≤ 1 / 2,
  sorry
end

end proof_problem_l810_810708


namespace expand_expression_l810_810981

theorem expand_expression : ∀ (x : ℝ), 2 * (x + 3) * (x^2 - 2*x + 7) = 2*x^3 + 2*x^2 + 2*x + 42 := 
by
  intro x
  sorry

end expand_expression_l810_810981


namespace inclination_angle_l810_810405

theorem inclination_angle (x y : ℝ) : 
  let line_eq := x * real.cos (140 * real.pi / 180) + y * real.sin (40 * real.pi / 180) + 1 = 0 in
  ∃ θ : ℝ, θ = 50 ∧ (∀ (x y : ℝ), (x * real.cos (140 * real.pi / 180) + y * real.sin (40 * real.pi / 180) + 1 = 0) → θ = (θ * (real.pi / 180)))
  :=
begin
  sorry
end

end inclination_angle_l810_810405


namespace fleas_difference_l810_810908

-- Define the initial number of fleas and subsequent fleas after each treatment.
def initial_fleas (F : ℝ) := F
def after_first_treatment (F : ℝ) := F * 0.40
def after_second_treatment (F : ℝ) := (after_first_treatment F) * 0.55
def after_third_treatment (F : ℝ) := (after_second_treatment F) * 0.70
def after_fourth_treatment (F : ℝ) := (after_third_treatment F) * 0.80

-- Given condition
axiom final_fleas : initial_fleas 20 = after_fourth_treatment 20

-- Prove the number of fleas before treatment minus the number after treatment is 142
theorem fleas_difference (F : ℝ) (h : initial_fleas F = after_fourth_treatment 20) : 
  F - 20 = 142 :=
by {
  sorry
}

end fleas_difference_l810_810908


namespace distinct_sums_at_least_half_n_nplus1_l810_810158

theorem distinct_sums_at_least_half_n_nplus1 (n : ℕ) 
  (h_n_pos : 0 < n) (a : Finₓ n → ℕ) (h_distinct : Function.Injective a) : 
  ∃ s : Finset ℕ, s.card ≥ n*(n+1)/2 ∧ 
    (∀ t : Finset (Finₓ n), t ≠ ∅ → s = s ∪ (∑ i in t, a i)) :=
sorry

end distinct_sums_at_least_half_n_nplus1_l810_810158


namespace cost_of_each_barbell_l810_810281

theorem cost_of_each_barbell (total_given change_received total_barbells : ℕ)
  (h1 : total_given = 850)
  (h2 : change_received = 40)
  (h3 : total_barbells = 3) :
  (total_given - change_received) / total_barbells = 270 :=
by
  sorry

end cost_of_each_barbell_l810_810281


namespace max_x_lcm_15_21_105_l810_810768

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l810_810768


namespace area_of_trapezoid_DBCE_l810_810525

-- Define the conditions used in the problem
variables {A B C D E : Type} -- Points in the plane
variables {area : Type} [HasZero area] -- Area type with zero element
variables {area_triangle : A → B → C → area}
variables {similar : A → B → C → A → D → E → Prop}

-- Conditions
variables {triangle_ABC_isosceles : AB = AC}
variables {smallest_triangle_area : Π (P Q R : A), similar P Q R A B C → area_triangle P Q R = 1}
variables {triangle_ABC_area : area_triangle A B C = 40}

-- The proof problem
theorem area_of_trapezoid_DBCE :
  area_of_trapezoid D B C E = 20 :=
begin
  sorry
end

end area_of_trapezoid_DBCE_l810_810525


namespace roots_of_polynomial_l810_810108

noncomputable def polynomial := λ x : ℂ, 5 * x ^ 5 + 18 * x ^ 3 - 45 * x ^ 2 + 30 * x

theorem roots_of_polynomial : 
  ∀ x : ℂ, polynomial x = 0 ↔ x = 0 ∨ x = 1 / 5 ∨ x = complex.I * real.sqrt 3 ∨ x = - complex.I * real.sqrt 3 :=
sorry

end roots_of_polynomial_l810_810108


namespace sum_of_squares_inequality_l810_810685

theorem sum_of_squares_inequality (n : ℕ) (x : Finₓ n → ℕ) (h : ∀ i j : Finₓ n, i ≠ j → x i ≠ x j) :
  (∑ i : Finₓ n, (x i)^2) ≥ ((2 * n + 1) / 3) * ∑ i : Finₓ n, x i := by
  sorry

end sum_of_squares_inequality_l810_810685


namespace range_H_l810_810871

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem range_H : set.range H = set.Icc 3 5 := 
sorry

end range_H_l810_810871


namespace A_can_give_B_head_start_l810_810010

theorem A_can_give_B_head_start :
  let vA := 1000 / 4.5 in
  let vB := 1000 / 5 in
  vA - vB = 22.22 ∧ 5 * (vA - vB) = 111 :=
begin
  intros,
  sorry
end

end A_can_give_B_head_start_l810_810010


namespace problem_statement_l810_810317

theorem problem_statement (x y z : ℕ) (h₁ : x = 40) (h₂ : y = 20) (h₃ : z = 5) :
  (0.8 * (3 * (2 * x))^2 - 0.8 * real.sqrt ((y / 4)^3 * z^3)) = 45980 :=
by
  sorry

end problem_statement_l810_810317


namespace range_of_a_l810_810186

def isTangentLine (a t : ℝ) : Prop := 
  let f := λ x, -x^3 + (3 * a / 2) * x^2 - 6 * x
  let f' := λ x, -3 * x^2 + 3 * a * x - 6
  let tangentLine := (f t) + f' t * (0 - t)
  tangentLine = -1

def curveHasTwoTangents (a : ℝ) : Prop := 
  ∃ t₁ t₂ : ℝ, t₁ > 0 ∧ t₂ > 0 ∧ t₁ ≠ t₂ ∧ isTangentLine a t₁ ∧ isTangentLine a t₂

theorem range_of_a : { a : ℝ | curveHasTwoTangents a } = {a : ℝ | a > 2} :=
sorry

end range_of_a_l810_810186


namespace jack_buttons_total_l810_810272

theorem jack_buttons_total :
  (3 * 3) * 7 = 63 :=
by
  sorry

end jack_buttons_total_l810_810272


namespace problem1_problem2_l810_810195

def f (x : ℝ) : ℝ := |x - 3|

theorem problem1 :
  {x : ℝ | f x < 2 + |x + 1|} = {x : ℝ | 0 < x} := sorry

theorem problem2 (m n : ℝ) (h_mn : m > 0) (h_nn : n > 0) (h : (1 / m) + (1 / n) = 2 * m * n) :
  m * f n + n * f (-m) ≥ 6 := sorry

end problem1_problem2_l810_810195


namespace equation_of_line_l_l810_810103

theorem equation_of_line_l (
  passes_through_intersection : ∃ P : ℝ × ℝ, 
    (P.1 * 2 - P.2 * 3 - 3 = 0) ∧ 
    (P.1 + P.2 + 2 = 0) ∧ 
  parallel_to_line : ∀ Q : ℝ × ℝ, 
    (Q.2 = -3 * Q.1) → 
    ∃ R : ℝ × ℝ, Q = R) :
  ∃ a b c : ℝ, a * 15 + b * 5 + c * 16 = 0 :=
by
  sorry

end equation_of_line_l_l810_810103


namespace sqrt_pattern_l810_810322

theorem sqrt_pattern (n : ℕ) (h : n ≥ 2) : 
    sqrt (n - n / (n^2 + 1)) = n * sqrt (n / (n^2 + 1)) :=
by sorry

end sqrt_pattern_l810_810322


namespace part1_part2_l810_810617

noncomputable def f : ℝ → ℝ := sorry

def x : ℕ → ℝ 
| 0 := 1/2
| (n + 1) := 2 * x n / (1 + (x n)^2)

axiom f_prop : ∀ x y : ℝ, f x + f y = f ((x + y) / (1 + x * y))
axiom f_half : f (1/2) = -1
axiom domain : ∀ n, x n ∈ set.Ioo (-1) 1

theorem part1 : ∀ n, f (x n) = -2 ^ (n - 1) := sorry

theorem part2 : 
  1 + (finset.range n).sum (λ k, f (1 / (k^2 + 3*k + 1))) + f (1 / (n + 2)) = f 0 := 
sorry

end part1_part2_l810_810617


namespace probability_diana_beats_apollo_l810_810085

/-- Define the event that one die roll is greater than another -/
def diana_wins (diana apollo : ℕ) : Prop := diana > apollo

/-- Define the probability calculation -/
def probability (success total : ℕ) : ℚ := success / total

/-- Our statement: the probability that Diana's number is larger than Apollo's number is 7/16 -/
theorem probability_diana_beats_apollo : 
  probability 
    (Finset.card 
      (Finset.filter (λ (p : ℕ × ℕ), diana_wins p.1 p.2) 
        (Finset.product (Finset.range 8) (Finset.range 8))))
    (Finset.card (Finset.product (Finset.range 8) (Finset.range 8))) 
  = 7 / 16 :=
sorry

end probability_diana_beats_apollo_l810_810085


namespace number_of_friends_l810_810331

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l810_810331


namespace problem_statement_l810_810824

theorem problem_statement (x : ℝ) (h : x + x⁻¹ = 3) : x^7 - 6 * x^5 + 5 * x^3 - x = 0 :=
sorry

end problem_statement_l810_810824


namespace students_taking_art_l810_810900

def total_students := 500
def students_taking_music := 40
def students_taking_both := 10
def students_taking_neither := 450

theorem students_taking_art : ∃ A, total_students = students_taking_music - students_taking_both + (A - students_taking_both) + students_taking_both + students_taking_neither ∧ A = 20 :=
by
  sorry

end students_taking_art_l810_810900


namespace regression_analysis_correct_statements_l810_810253

theorem regression_analysis_correct_statements :
  (let statement1 := ∀ residual_plot, evenly_scattered_horizontal_band(residual_plot) -> appropriate_model(residual_plot),
       statement2 := ∀ model, larger_R2_better_fit(model),
       statement3 := ∀ model1 model2, smaller_sum_of_squares_residuals_better_fit(model1, model2)
   in (statement1 ∧ statement2 ∧ statement3) = 3) :=
by
  sorry

end regression_analysis_correct_statements_l810_810253


namespace no_solution_for_inequalities_l810_810578

theorem no_solution_for_inequalities (x : ℝ) :
  ¬(5 * x^2 - 7 * x + 1 < 0 ∧ x^2 - 9 * x + 30 < 0) :=
sorry

end no_solution_for_inequalities_l810_810578


namespace range_of_t_l810_810194

noncomputable def f (a x : ℝ) : ℝ := x * abs (x - a) + 2 * x

theorem range_of_t (t : ℝ) :
  (∃ a ∈ set.Icc (-3 : ℝ) 3, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    f a x1 = t * f a a ∧ f a x2 = t * f a a ∧ f a x3 = t * f a a) ↔
  t ∈ set.Ioo 1 (25/24) :=
by sorry

end range_of_t_l810_810194


namespace line_equation_slope_intercept_l810_810489

theorem line_equation_slope_intercept (x y : ℝ) :
  (∃ x y : ℝ, (2, -1) • (x - 4, y + 3) = 0) → ∃ m b : ℝ, (m, b) = (2, -11) := 
by
  sorry

end line_equation_slope_intercept_l810_810489


namespace friends_attended_birthday_l810_810359

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l810_810359


namespace shift_graph_l810_810604

def f (x : ℝ) : ℝ := Real.sin x
def f'' (x : ℝ) : ℝ := (Real.sin'' x)

theorem shift_graph :
  ∀ x : ℝ, f'' (2 * x + π / 3) = Real.sin (2 * (x + 5 * π / 12)) :=
by
  sorry

end shift_graph_l810_810604


namespace find_n_plus_t_l810_810058

theorem find_n_plus_t (n t : ℕ) (h1 : n > 0) (h2 : t > 0)
    (h3 : (∑ i in Finset.range (k+1), n + i) = 374)
    (h4 : (∑ i in Finset.range (k+1), t + i) = 319) :
    n + t = 53 :=
by 
  sorry

end find_n_plus_t_l810_810058


namespace cost_of_house_l810_810278

noncomputable def cost_of_trailer : ℤ := 120000
noncomputable def loan_period_years : ℕ := 20
noncomputable def loan_period_months : ℕ := loan_period_years * 12
noncomputable def additional_payment_house : ℤ := 1500

theorem cost_of_house :
  let T := cost_of_trailer / loan_period_months in
  let monthly_payment_house := T + additional_payment_house in
  let total_cost_house := monthly_payment_house * loan_period_months in
  total_cost_house = 480000 :=
by
  sorry

end cost_of_house_l810_810278


namespace average_salary_excluding_manager_l810_810396

theorem average_salary_excluding_manager
    (A : ℝ)
    (manager_salary : ℝ)
    (total_employees : ℕ)
    (salary_increase : ℝ)
    (h1 : total_employees = 24)
    (h2 : manager_salary = 4900)
    (h3 : salary_increase = 100)
    (h4 : 24 * A + manager_salary = 25 * (A + salary_increase)) :
    A = 2400 := by
  sorry

end average_salary_excluding_manager_l810_810396


namespace no_solution_mod_8_no_solution_mod_9_solution_mod_11_no_solution_mod_121_l810_810577

theorem no_solution_mod_8 (n : ℤ) : ¬ (n^2 - 6 * n - 2 = 0 [MOD 8]) :=
sorry

theorem no_solution_mod_9 (n : ℤ) : ¬ (n^2 - 6 * n - 2 = 0 [MOD 9]) :=
sorry

theorem solution_mod_11 (n k : ℤ) : (n = 3 + 11 * k) ↔ (n^2 - 6 * n - 2 = 0 [MOD 11]) :=
sorry

theorem no_solution_mod_121 (n : ℤ) : ¬ (n^2 - 6 * n - 2 = 0 [MOD 121]) :=
sorry

end no_solution_mod_8_no_solution_mod_9_solution_mod_11_no_solution_mod_121_l810_810577


namespace distance_between_parallel_lines_l810_810756

theorem distance_between_parallel_lines (A B c1 c2 : Real) (hA : A = 2) (hB : B = 3) 
(hc1 : c1 = -3) (hc2 : c2 = 2) : 
    (abs (c1 - c2) / Real.sqrt (A^2 + B^2)) = (5 * Real.sqrt 13 / 13) := by
  sorry

end distance_between_parallel_lines_l810_810756


namespace irrational_number_among_candidates_l810_810947

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

theorem irrational_number_among_candidates :
  let a := (1 : ℝ) / (7 : ℝ)
  let b := Real.sqrt 2
  let c := Real.cbrt 8
  let d := (1010010001 : ℝ) / (1000000000 : ℝ)
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
  b ≠ c ∧ b ≠ d ∧
  c ≠ d ∧
  is_irrational b :=
by {
  sorry,
}

end irrational_number_among_candidates_l810_810947


namespace number_of_houses_around_square_l810_810717

namespace HouseCounting

-- Definitions for the conditions
def M (k : ℕ) : ℕ := k
def J (k : ℕ) : ℕ := k

-- The main theorem stating the solution
theorem number_of_houses_around_square (n : ℕ)
  (h1 : M 5 % n = J 12 % n)
  (h2 : J 5 % n = M 30 % n) : n = 32 :=
sorry

end HouseCounting

end number_of_houses_around_square_l810_810717


namespace louis_current_age_l810_810234

/-- 
  In 6 years, Carla will be 30 years old. 
  The sum of the current ages of Carla and Louis is 55. 
  Prove that Louis is currently 31 years old.
--/
theorem louis_current_age (C L : ℕ) 
  (h1 : C + 6 = 30) 
  (h2 : C + L = 55) 
  : L = 31 := 
sorry

end louis_current_age_l810_810234


namespace num_five_student_committees_l810_810126

theorem num_five_student_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 5) : choose n k = 56 :=
by
  rw [h_n, h_k]
  -- rest of the proof would go here
  sorry

end num_five_student_committees_l810_810126


namespace general_term_correct_max_n_correct_l810_810182

-- Definitions and conditions
noncomputable def geometric_seq (a₁ : ℝ) (q : ℝ) : ℕ → ℝ
| 0       := a₁
| (n + 1) := geometric_seq q a₁ n * q

noncomputable def S₄ (a₁ : ℝ) (q : ℝ) : ℝ :=
(geometric_seq a₁ q 0) + (geometric_seq a₁ q 1) + (geometric_seq a₁ q 2) + (geometric_seq a₁ q 3)

def condition1 (a₁ : ℝ) (q : ℝ) : Prop :=
S₄ a₁ q = 5

def condition2 (a₁ : ℝ) (q : ℝ) : Prop :=
4 * a₁ + geometric_seq a₁ q 1 = 3 * geometric_seq a₁ q 1

-- Proof goals
noncomputable def general_term : ℕ → ℝ := λ n, (1 / 3) * 2^(n - 1)

theorem general_term_correct (a₁ : ℝ) (q : ℝ) (h1 : condition1 a₁ q) (h2 : condition2 a₁ q) :
∀ n, geometric_seq a₁ q n = general_term n := sorry

-- Definitions and computations for the arithmetic sequence
noncomputable def arith_seq (a₁ : ℝ) : ℕ → ℝ
| 0       := 2
| (n + 1) := 2 + (n + 1) * (-a₁)

noncomputable def T_n (a₁ : ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := T_n n + arith_seq a₁ n

def condition3 (a₁ : ℝ) (n : ℕ) : Prop :=
T_n a₁ (n - 1) > 0

theorem max_n_correct (a₁ : ℝ) (h3 : a₁ = 1 / 3) : ∃ n, n = 13 ∧ condition3 a₁ n := sorry

end general_term_correct_max_n_correct_l810_810182


namespace domain_transformation_l810_810180

variable {α : Type*}
variable {f : α → α}
variable {x y : α}
variable (h₁ : ∀ x, -1 < x ∧ x < 1)

theorem domain_transformation (h₁ : ∀ x, -1 < x ∧ x < 1) : ∀ x, 0 < x ∧ x < 1 →
  ((-1 < (2 * x - 1) ∧ (2 * x - 1) < 1)) :=
by
  intro x
  intro h
  have h₂ : -1 < 2 * x - 1 := sorry
  have h₃ : 2 * x - 1 < 1 := sorry
  exact ⟨h₂, h₃⟩

end domain_transformation_l810_810180


namespace greatest_x_lcm_105_l810_810784

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l810_810784


namespace no_consecutive_sum_eq_30_l810_810210

open Nat

theorem no_consecutive_sum_eq_30 :
  ¬ ∃ (a n : ℕ), n ≥ 2 ∧ (∑ i in range (a + n) \ set.range a, i) = 30 := by
  sorry

end no_consecutive_sum_eq_30_l810_810210


namespace intersection_M_N_l810_810202

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end intersection_M_N_l810_810202


namespace polynomial_root_coefficient_B_l810_810569

noncomputable def thirdSymmetricSum (s : Finset ℕ) (k : ℕ) : ℕ :=
  Finset.sum (s.powerset.filter (λ t, t.card = k)) (λ t, t.prod id)

theorem polynomial_root_coefficient_B :
  ∃ (E F G : ℤ) (roots : Finset ℕ), 
  (∀ r ∈ roots, r > 0) ∧
  roots.sum id = 15 ∧
  roots.prod id = 64 ∧
  roots.card = 6 ∧
  (-1 : ℤ)^3 * thirdSymmetricSum roots 3 = -287 :=
by
  sorry

end polynomial_root_coefficient_B_l810_810569


namespace volume_maximization_perpendicular_l810_810734

noncomputable def tetrahedron_volume_maximized (u v w t m : ℝ) :=
  (u > 0) → (v > 0) → (w > 0) →
  (∀ t, t = 1 / 2 * u * (if u ⟂ v then v else sorry)) →
  (∀ m, m = (if w ⟂ (u * v) then w else sorry)) →
  (∀ K, K = 1 / 3 * t * m) →
  (t = 1 / 2 * u * v) → (m = w) → K 

theorem volume_maximization_perpendicular (u v w : ℝ) :
  tetrahedron_volume_maximized u v w (1/2 * u * v) w :=
sorry

end volume_maximization_perpendicular_l810_810734


namespace mary_probability_l810_810718

open BigOperators

noncomputable def probability_at_least_half_correct : ℚ :=
  ∑ k in finset.range 17 \ finset.range 8, (nat.choose 16 k) * (1 / 4)^k * (3 / 4)^(16 - k)

theorem mary_probability :
  probability_at_least_half_correct = 
    ∑ k in finset.range 17 \ finset.range 8, (nat.choose 16 k) * (1 / 4)^k * (3 / 4)^(16 - k) :=
begin
  sorry
end

end mary_probability_l810_810718


namespace necessary_and_sufficient_condition_for_geometric_sequence_l810_810425

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {c : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n+1) = r * a_n n

theorem necessary_and_sufficient_condition_for_geometric_sequence :
  (∀ n : ℕ, S_n n = 2^n + c) →
  (∀ n : ℕ, a_n n = S_n n - S_n (n-1)) →
  is_geometric_sequence a_n ↔ c = -1 :=
by
  sorry

end necessary_and_sufficient_condition_for_geometric_sequence_l810_810425


namespace monthly_expenses_last_month_l810_810741

def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.10
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.20

def commission := total_sales * commission_rate
def total_earnings := basic_salary + commission
def savings := total_earnings * savings_rate
def monthly_expenses := total_earnings - savings

theorem monthly_expenses_last_month :
  monthly_expenses = 2888 := 
by sorry

end monthly_expenses_last_month_l810_810741


namespace expected_winnings_l810_810487

-- Define the probability of rolling any given number on a fair 6-sided die
def prob (n : ℕ) : ℚ :=
  if 1 ≤ n ∧ n ≤ 6 then 1 / 6 else 0

-- Define the winnings for a given roll n
def winnings (n : ℕ) : ℕ := (6 - n) ^ 2

-- Calculate the expected value of winnings
def expected_value : ℚ :=
  ∑ n in Finset.range 6, prob (n + 1) * (winnings (n + 1) : ℚ)

-- The expected value should be equal to 55/6
theorem expected_winnings : expected_value = 55 / 6 := by
  sorry

end expected_winnings_l810_810487


namespace one_white_one_red_probability_l810_810894

theorem one_white_one_red_probability :
  let total_ways := Nat.choose 16 2
  let ways_to_choose_white := Nat.choose 7 1
  let ways_to_choose_red := Nat.choose 1 1
  let successful_outcomes := ways_to_choose_white * ways_to_choose_red
  total_ways > 0 -> (successful_outcomes / total_ways.toRat) = (7 / 120 : ℚ):=
by
  intro total_ways_pos
  have total_ways_val : total_ways = 120 := by
    rw [Nat.choose]
    simp
  have successful_outcomes_val : successful_outcomes = 7 := by
    rw [Nat.choose, Nat.choose]
    simp
  rw [total_ways_val, successful_outcomes_val]
  norm_num
  sorry

end one_white_one_red_probability_l810_810894


namespace friends_attended_birthday_l810_810354

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l810_810354


namespace ratio_FC_AC_l810_810244

open Triangle

-- Defining the basic geometric properties in the context of right triangle with altitude, median, and bisector
variable (A B C D E F : Point)
variable {ABC : Triangle}
variable [RightTriangle ABC B]
variable [AngleA : Angle A B C = 30°]
variable [HeightBD : HeightFrom B D]
variable [MedianDE : MedianFrom D E]
variable [BisectorEF : AngleBisectorFrom E F]

-- Statement of the proof
theorem ratio_FC_AC (h : RightTriangle ABC B ∧ HeightFrom B D ∧ MedianFrom D E ∧ AngleBisectorFrom E F)
  : ratio F C A C = 1 / 8 := 
  sorry

end ratio_FC_AC_l810_810244


namespace distance_walked_l810_810017

noncomputable def david_location : ℤ × ℤ := (2, -25)
noncomputable def emma_location : ℤ × ℤ := (-3, 19)
noncomputable def felix_location : ℚ × ℚ := (-1/2, -6)

def midpoint (p1 p2 : ℚ × ℚ) : ℚ × ℚ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def vertical_distance (y1 y2 : ℚ) : ℚ :=
  y2 - y1

theorem distance_walked : 
  vertical_distance 
    (midpoint (david_location, emma_location)).2 
    (felix_location).2 = 3 :=
by
  sorry

end distance_walked_l810_810017


namespace friends_attended_birthday_l810_810358

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l810_810358


namespace linda_original_savings_l810_810714

theorem linda_original_savings (S : ℝ) (f : ℝ) (a : ℝ) (t : ℝ) 
  (h1 : f = 7 / 13 * S) (h2 : a = 3 / 13 * S) 
  (h3 : t = S - f - a) (h4 : t = 180) (h5 : a = 360) : 
  S = 1560 :=
by 
  sorry

end linda_original_savings_l810_810714


namespace intersection_of_A_and_B_l810_810581

def setA : Set ℝ := {y | ∃ x : ℝ, y = 2 * x}
def setB : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem intersection_of_A_and_B : setA ∩ setB = {y | y ≥ 0} :=
by
  sorry

end intersection_of_A_and_B_l810_810581


namespace chord_length_l810_810620

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 12 * x - 5 * y = 3

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 8 * y + 16 = 0

-- Defining the theorem to calculate length of chord AB
theorem chord_length :
  ∀ (A B : ℝ × ℝ),
    line_eq A.1 A.2 →
    circle_eq A.1 A.2 →
    line_eq B.1 B.2 →
    circle_eq B.1 B.2 →
    dist A B = 4 * real.sqrt 2 :=
by sorry

end chord_length_l810_810620


namespace value_this_year_l810_810428

def last_year_value : ℝ := 20000
def depreciation_factor : ℝ := 0.8

theorem value_this_year :
  last_year_value * depreciation_factor = 16000 :=
by
  sorry

end value_this_year_l810_810428


namespace square_area_l810_810473

theorem square_area (A B C D : Point) (P Q R S : Point)
  (H1 : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)
  (H2 : is_square A B C D)
  (HP : on_side P A B)
  (HQ : on_side Q B C)
  (HR : on_side R C D)
  (HS : on_side S D A)
  (HP_coords : P = (31, 27))
  (HQ_coords : Q = (42, 43))
  (HR_coords : R = (60, 27))
  (HS_coords : S = (46, 16)) :
  area A B C D = 672.8 :=
sorry

end square_area_l810_810473


namespace hiring_probabilities_l810_810906

-- Definitions based on the conditions
def n := 10
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)
def arrangements := factorial n
def A (k : Nat) : Nat := sorry -- Number of different application orders in which the kth most capable person can be hired

-- Lean 4 statement for the proof
theorem hiring_probabilities (A : Nat → Nat) 
  (perm : Fin n! → Vector (Fin n) n)  -- permutations of the n applicants
  (first_3_not_hired_condition : ∀ (p : Fin (3 * 3!)), ¬(hired_at_step p 1 ∨ hired_at_step p 2 ∨ hired_at_step p 3)) -- no first 3 hired
  (best_among_first_three_check : ∀ i, 0 < i ∧ i < 4 → p (i - 1) < p i) -- assuming ranks of first 3
  (hired_if_best : ∀ i, 4 ≤ i → (∀ j, j < i → p j < p i) → hired_at_step p i) -- hired if ability surpasses all previous interviews
  (last_if_none : ∀ p, (∀ i < 9, ¬hired_at_step p i) → hired_at_step p 10) -- if none hired before last, hire last
  : 
  (A 1 > A 2 ∧ A 2 > A 3 ∧ A 3 > A 4 ∧ A 4 > A 5 ∧ A 5 > A 6 ∧ A 6 > A 7 ∧ A 7 > A 8 ∧ A 8 = A 9 = A 10) ∧
  ((A 1 + A 2 + A 3 > 0.7 * arrangements) ∧
   (A 8 + A 9 + A 10 ≤ 0.1 * arrangements)) := sorry

-- Definitions to be used within the theorem
def hired_at_step (p : Fin n! → Vector (Fin n) n) (i : Nat) : Bool := sorry

end hiring_probabilities_l810_810906


namespace tribe_leadership_choices_l810_810056

open Nat

theorem tribe_leadership_choices (n m k l : ℕ) (h : n = 15) : 
  (choose 14 2 * choose 12 3 * choose 9 3 * 15 = 27392400) := 
  by sorry

end tribe_leadership_choices_l810_810056


namespace no_three_digit_numbers_divisible_by_30_l810_810630

def digits_greater_than_6 (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d > 6

theorem no_three_digit_numbers_divisible_by_30 :
  ∀ n, (100 ≤ n ∧ n < 1000 ∧ digits_greater_than_6 n ∧ n % 30 = 0) → false :=
by
  sorry

end no_three_digit_numbers_divisible_by_30_l810_810630


namespace five_student_committees_from_eight_l810_810134

theorem five_student_committees_from_eight : nat.choose 8 5 = 56 := by
  sorry

end five_student_committees_from_eight_l810_810134


namespace lemonade_sales_l810_810849

theorem lemonade_sales (total_amount small_amount medium_amount large_price sales_price_small sales_price_medium earnings_small earnings_medium : ℕ) (h1 : total_amount = 50) (h2 : sales_price_small = 1) (h3 : sales_price_medium = 2) (h4 : large_price = 3) (h5 : earnings_small = 11) (h6 : earnings_medium = 24) : large_amount = 5 :=
by
  sorry

end lemonade_sales_l810_810849


namespace lemango_eating_mangos_l810_810682

theorem lemango_eating_mangos :
  ∃ (mangos_eaten : ℕ → ℕ), 
    (mangos_eaten 1 * (2^6 - 1) = 364 * (2 - 1)) ∧
    (mangos_eaten 6 = 128) :=
by
  sorry

end lemango_eating_mangos_l810_810682


namespace combined_shoe_size_l810_810279

-- Definitions based on conditions
def Jasmine_size : ℕ := 7
def Alexa_size : ℕ := 2 * Jasmine_size
def Clara_size : ℕ := 3 * Jasmine_size

-- Statement to prove
theorem combined_shoe_size : Jasmine_size + Alexa_size + Clara_size = 42 :=
by
  sorry

end combined_shoe_size_l810_810279


namespace sequence_general_term_l810_810262

-- Define the sequence {a_n} with the given initial term and recursive formula
def sequence (n : ℕ) : ℚ
| 0       := 1  -- Note: using 0 to represent a1 for simplicity
| (n + 1) := 3 * sequence n / (3 + sequence n)

-- Conjectured general term formula for the sequence: a_n = 3 / (n + 2)
def conjectured_formula (n : ℕ) : ℚ := 3 / (n + 2)

-- The problem statement to prove in Lean
theorem sequence_general_term (n : ℕ) : sequence n = conjectured_formula n := by
  sorry

end sequence_general_term_l810_810262


namespace dog_catches_rabbit_in_4_minutes_l810_810089

def dog_speed_mph : ℝ := 24
def rabbit_speed_mph : ℝ := 15
def rabbit_head_start : ℝ := 0.6

theorem dog_catches_rabbit_in_4_minutes : 
  (∃ t : ℝ, t > 0 ∧ 0.4 * t = 0.25 * t + 0.6) → ∃ t : ℝ, t = 4 :=
sorry

end dog_catches_rabbit_in_4_minutes_l810_810089


namespace smallest_value_l810_810555

theorem smallest_value : 54 * Real.sqrt 3 < 144 ∧ 54 * Real.sqrt 3 < 108 * Real.sqrt 6 - 108 * Real.sqrt 2 := by
  sorry

end smallest_value_l810_810555


namespace regular_tetrahedron_division_infinite_planes_l810_810968

theorem regular_tetrahedron_division_infinite_planes :
  ∀ T : geometric_shape, is_regular_tetrahedron T → 
  (∃ P : set (plane T), ∀ p ∈ P, passes_through_center p T) ∧ infinite P := 
sorry

end regular_tetrahedron_division_infinite_planes_l810_810968


namespace concentric_circles_ratio_l810_810417

theorem concentric_circles_ratio (r R : ℝ) (h₁ : (π * R^2) / (π * r^2) = 4) : R - r = r :=
by
  -- First, we find R in terms of r using the given ratio.
  have h2 : (R^2) / (r^2) = 4,
  {
    rw [←div_div, div_self, one_mul] at h₁,
    exact h₁,
    exact pi_ne_zero,
  },

  -- Take square root of both sides.
  have h3 : (R / r)^2 = 4,
  {
    rw [div_sq, h2],
  },

  -- Simplifying sqrt gives R / r = 2.
  have h4 : R / r = 2,
  {
    exact eq_of_sq_eq_sq (or.inl zero_lt_two),
  },

  -- Therefore, R = 2r.
  have h5 : R = 2 * r,
  {
    rw [eq_div_iff_mul_eq],
    exact h4,
    exact ne_of_gt zero_lt_two,
  },

  -- Subtract r from 2r to get the difference.
  have h6 : R - r = 2 * r - r,
  {
    exact eq.symm (congr_arg (has_sub.sub (R - r)) h5),
  },

  -- Simplify the right-hand side.
  rw [two_mul, sub_add_cancel] at h6,
  exact eq.symm h6,
sorry

end concentric_circles_ratio_l810_810417


namespace discount_difference_l810_810531

noncomputable def single_discount (amount : ℝ) (rate : ℝ) : ℝ :=
  amount * (1 - rate)

noncomputable def successive_discounts (amount : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (λ acc rate => acc * (1 - rate)) amount

theorem discount_difference:
  let amount := 12000
  let single_rate := 0.35
  let successive_rates := [0.25, 0.08, 0.02]
  single_discount amount single_rate - successive_discounts amount successive_rates = 314.4 := 
  sorry

end discount_difference_l810_810531


namespace alice_burger_spending_l810_810087

theorem alice_burger_spending :
  let daily_burgers := 4
  let burger_cost := 13
  let days_in_june := 30
  let mondays_wednesdays := 8
  let fridays := 4
  let fifth_purchase_coupons := 6
  let discount_10_percent := 0.9
  let discount_50_percent := 0.5
  let full_price := days_in_june * daily_burgers * burger_cost
  let discount_10 := mondays_wednesdays * daily_burgers * burger_cost * discount_10_percent
  let fridays_cost := (daily_burgers - 1) * fridays * burger_cost
  let discount_50 := fifth_purchase_coupons * burger_cost * discount_50_percent
  full_price - discount_10 - fridays_cost - discount_50 + fridays_cost = 1146.6 := by sorry

end alice_burger_spending_l810_810087


namespace greatest_x_lcm_l810_810804

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l810_810804


namespace affine_transform_as_composition_of_stretches_l810_810735

variables {α : Type*} [affine_space α]

-- Define an affine transformation in an affine space
def affine_transformation (T : α → α) : Prop :=
∀ (P Q R : α), collinear P Q R → collinear (T P) (T Q) (T R) 

-- Define stretch transformation
def stretch_transformation (S: α → α) (k: ℝ) : Prop :=
∀ (P Q : α), distance_PQ (S P) (S Q) = k * distance_PQ P Q

-- Prove the main theorem
theorem affine_transform_as_composition_of_stretches 
    (T : α → α) 
    (h_affine : affine_transformation T)
    (A B C : α):
    ∃ (S₁ S₂: α → α) 
    (k₁ k₂: ℝ), stretch_transformation S₁ k₁ ∧ stretch_transformation S₂ k₂ ∧ 
    (∃ A' B' C' : α, similar (A':: B':: C':: []) (T A:: T B:: T C:: []) ) ∧ 
    (T (triangle A B C)) = S₂ (S₁ (triangle A B C)) :=
sorry

end affine_transform_as_composition_of_stretches_l810_810735


namespace combination_eight_choose_five_l810_810135

theorem combination_eight_choose_five : 
  ∀ (n k : ℕ), n = 8 ∧ k = 5 → Nat.choose n k = 56 :=
by 
  intros n k h
  obtain ⟨hn, hk⟩ := h
  rw [hn, hk]
  exact Nat.choose_eq 8 5
  sorry  -- This signifies that the proof needs to be filled in, but we'll skip it as per instructions.

end combination_eight_choose_five_l810_810135


namespace mutually_exclusive_not_complementary_pairs_l810_810924

-- Conditions
def E1 : Prop := "miss the target"
def E2 : Prop := "hit the target"
def E3 : Prop := "the number of rings hit on the target is greater than 4"
def E4 : Prop := "the number of rings hit on the target is not less than 5"

-- The statement of the proof problem
theorem mutually_exclusive_not_complementary_pairs :
  (num_pairs (λ {E1, E3}, mutually_exclusive_not_complementary) + 
   num_pairs (λ {E1, E4}, mutually_exclusive_not_complementary)) = 2 :=
sorry

end mutually_exclusive_not_complementary_pairs_l810_810924


namespace no_distinct_integer_poly_cycle_l810_810737

noncomputable def distinct_integer_poly_cycle := 
  ∀ (p : ℤ[X]) (n : ℕ), n ≥ 3 → 
    ∀ (x : Fin n → ℤ), Function.Injective x → 
      ¬ (p.eval (x 0) = x 1 ∧ p.eval (x 1) = x 2 ∧ p.eval (x 2) = x 3 ∧ ... ∧ p.eval (x (n-1)) = x 0)

theorem no_distinct_integer_poly_cycle (p : ℤ[X]) (n : ℕ) (h_n : n ≥ 3)
  (x : Fin n → ℤ) (h_inj : Function.Injective x) :
  ¬ (p.eval (x 0) = x 1 ∧ p.eval (x 1) = x 2 ∧ ∀ i < n - 1, p.eval (x i) = x (i + 1) ∧ p.eval (x (n - 1)) = x 0) := 
sorry

end no_distinct_integer_poly_cycle_l810_810737


namespace find_GH_l810_810657

/-- In right triangle GHI, we have ∠G = 40°, ∠H = 90°, and HI = 12.
    The length of GH is approximately 14.3 to the nearest tenth. -/
theorem find_GH
  (G H I : Type)
  [triangle : RightTriangle G H I]
  (angle_G : ∠G = 40)
  (angle_H : ∠H = 90)
  (side_HI : distance H I = 12) :
  distance G H ≈ 14.3 :=
sorry

end find_GH_l810_810657


namespace range_H_l810_810870

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem range_H : set.range H = set.Icc 3 5 := 
sorry

end range_H_l810_810870


namespace stratified_sampling_first_level_l810_810923

-- Definitions from the conditions
def num_senior_teachers : ℕ := 90
def num_first_level_teachers : ℕ := 120
def num_second_level_teachers : ℕ := 170
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_second_level_teachers
def sample_size : ℕ := 38

-- Definition of the stratified sampling result
def num_first_level_selected : ℕ := (num_first_level_teachers * sample_size) / total_teachers

-- The statement to be proven
theorem stratified_sampling_first_level : num_first_level_selected = 12 :=
by
  sorry

end stratified_sampling_first_level_l810_810923


namespace greatest_x_lcm_l810_810800

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l810_810800


namespace prob_240_yuan_refund_l810_810976


def spinner_probability (n : ℕ) (p : ℚ) : ℚ := (Nat.choose 3 n) * (p^n) * ((1 - p)^(3-n))

def refund_probability : ℚ :=
1 - (spinner_probability 0 (1/6)) - (spinner_probability 3 (1/6))

theorem prob_240_yuan_refund : refund_probability = 5/12 := by
  sorry

end prob_240_yuan_refund_l810_810976


namespace center_is_8_in_3x3_array_l810_810943

theorem center_is_8_in_3x3_array :
  ∃ (arr : array (Fin 3) (array (Fin 3) ℕ)),
  (∀ i j, arr i j ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (∀ x y ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, ((|x - y| = 1) → (∃ i j k l, ((arr i j = x) ∧ (arr k l = y) ∧ ((|i - k|, |j - l|) = (1, 0) ∨ (|i - k|, |j - l|) = (0, 1))))) ∧
  (arr 0 0 + arr 0 2 + arr 2 0 + arr 2 2 = 20) ∧
  (arr 1 1 = 8) := sorry

end center_is_8_in_3x3_array_l810_810943


namespace minimum_value_m_l810_810402

noncomputable def f (x : ℝ) (phi : ℝ) : ℝ :=
  Real.sin (2 * x + phi)

theorem minimum_value_m (phi : ℝ) (m : ℝ) (h1 : |phi| < Real.pi / 2)
  (h2 : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x (Real.pi / 6) ≤ m) :
  m = -1 / 2 :=
by
  sorry

end minimum_value_m_l810_810402


namespace proof_problem_l810_810037

noncomputable def problem_data :=
  { n : ℕ := 10,
    x : Fin 10 → ℝ := λ i, [32.2, 31.1, 32.9, 35.7, 37.1, 38.0, 39.0, 43.0, 44.6, 46.0][i],
    y : Fin 10 → ℝ := λ i, [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0][i],
    sum_xi : ℝ := 379.6,
    sum_yi : ℝ := 391,
    sum_squares_x : ℝ := 246.904,
    sum_squares_y : ℝ := 568.9,
    sum_products : ℝ := sorry, -- m should be determined
    r : ℝ := 0.95 }

def correlation_is_strong (d : problem_data) : Prop :=
  d.r ≈ 0.95 → (d.r > 0.5) -- assuming a linear correlation is considered strong if r > 0.5

def empirical_regression_equation (d : problem_data) : Prop :=
  ∃ a b, b = 1.44 ∧ a = -15.56 ∧ ∀ x, d.y x = b * d.x x + a

def residual_and_slope_judgement (d : problem_data) : Prop :=
  let residual := d.y 0 - (1.44 * d.x 0 - 15.56) in
  residual = -5.81

theorem proof_problem (d : problem_data) :
  correlation_is_strong d ∧ empirical_regression_equation d ∧ residual_and_slope_judgement d :=
sorry

end proof_problem_l810_810037


namespace mod_inv_sum_l810_810862

-- Define modular inverse function
def mod_inv (a n : ℕ) : ℕ := let (g, x, y) := xgcd a n in ((x % n + n) % n)

-- Definitions for conditions
def five_inv_mod_31 : ℕ := mod_inv 5 31
def twenty_five_inv_mod_31 : ℕ := mod_inv (5^2) 31

-- Main statement to prove
theorem mod_inv_sum : (five_inv_mod_31 + twenty_five_inv_mod_31) % 31 = 26 := 
  by sorry

end mod_inv_sum_l810_810862


namespace compute_expression_l810_810075

noncomputable def expr1 := (16 / 81 : ℚ) ^ (-3 / 4 : ℚ)
noncomputable def expr2 := real.log (3 / 7) + real.log 70
noncomputable def expr3 := real.sqrt ((real.log 3) ^ 2 - real.log 9 + 1)

theorem compute_expression :
  expr1 + expr2 + expr3 = 43 / 8 :=
sorry

end compute_expression_l810_810075


namespace modular_inverse_sum_correct_l810_810098

theorem modular_inverse_sum_correct :
  (3 * 8 + 9 * 13) % 56 = 29 :=
by
  sorry

end modular_inverse_sum_correct_l810_810098


namespace star_chain_result_l810_810969

def star (x y : ℝ) : ℝ :=
  (Real.sqrt (x^2 + 3 * x * y + y^2 - 2 * x - 2 * y + 4)) / (x * y + 4)

theorem star_chain_result :
  ((List.foldl star) 1 [2007, 2006, 2005, ...(List.range 2004).reverse]) = Real.sqrt 15 / 9 :=
by
  sorry

end star_chain_result_l810_810969


namespace lateral_surface_area_frustum_example_l810_810910

noncomputable def lateral_surface_area_frustum (R r h : ℝ) : ℝ :=
  π * (R + r) * sqrt ((R - r) ^ 2 + h ^ 2)

theorem lateral_surface_area_frustum_example :
  lateral_surface_area_frustum 10 4 9 = 14 * π * sqrt 117 :=
by
  sorry

end lateral_surface_area_frustum_example_l810_810910


namespace sum_of_bottom_three_circles_l810_810410

open BigOperators

def circles := {1, 2, 3, 4, 5, 6, 7, 8}

variables (p q r s t u v w : ℕ)

noncomputable def conditions :=
  p * q * r = 30 ∧
  u * v * w = 20 ∧
  v * t * s = 28 ∧
  {p, q, r, s, t, u, v, w} = circles

theorem sum_of_bottom_three_circles :
  conditions p q r s t u v w → u + v + w = 17 :=
begin
  sorry
end

end sum_of_bottom_three_circles_l810_810410


namespace greatest_possible_x_max_possible_x_l810_810778

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l810_810778


namespace decrease_neg_of_odd_and_decrease_nonneg_l810_810172

-- Define the properties of the function f
variable (f : ℝ → ℝ)

-- f is odd
def odd_function : Prop := ∀ x : ℝ, f (-x) = - f x

-- f is decreasing on [0, +∞)
def decreasing_on_nonneg : Prop := ∀ x1 x2 : ℝ, (0 ≤ x1) → (0 ≤ x2) → (x1 < x2 → f x1 > f x2)

-- Goal: f is decreasing on (-∞, 0)
def decreasing_on_neg : Prop := ∀ x1 x2 : ℝ, (x1 < 0) → (x2 < 0) → (x1 < x2) → f x1 > f x2

-- The theorem to be proved
theorem decrease_neg_of_odd_and_decrease_nonneg 
  (h_odd : odd_function f) (h_decreasing_nonneg : decreasing_on_nonneg f) :
  decreasing_on_neg f :=
sorry

end decrease_neg_of_odd_and_decrease_nonneg_l810_810172


namespace plane_speed_due_west_l810_810491

theorem plane_speed_due_west (v : ℝ) : 
  (∀ t : ℝ, t = 2100 / (325 + v) → t = 3.5) → v = 275 :=
by 
  intros hT.
  apply_fun (λ x, x * (325 + v)) at hT.
  simp at hT.
  exact hT
  sorry

end plane_speed_due_west_l810_810491


namespace cost_of_each_barbell_l810_810287

variables (barbells : ℕ) (money_given money_change : ℝ) (total_cost_per_barbell : ℝ)

-- Given conditions
def conditions := barbells = 3 ∧ money_given = 850 ∧ money_change = 40

-- Theorem statement: Proving the cost of each barbell is $270
theorem cost_of_each_barbell (h : conditions) : total_cost_per_barbell = 270 :=
by
  -- We are using sorry to indicate we are skipping the proof
  sorry

#eval sorry -- Placeholder to verify if the code is syntactically correct

end cost_of_each_barbell_l810_810287


namespace determinant_of_matrix_l810_810963

theorem determinant_of_matrix (x y : ℝ) :
  det ![![x, 2], ![3, y]] = x * y - 6 :=
 sorry

end determinant_of_matrix_l810_810963


namespace brocard_angle_cot_eq_2_l810_810840

theorem brocard_angle_cot_eq_2 
  (O1 O2 O3 : Type) [metric_space O1] [metric_space O2] [metric_space O3]
  (line : set (affine_plane ℝ)) 
  (mutual_points : set ℝ) 
  (A B C: ℝ)
  (hTangent1: O1 ∈ mutual_points)
  (hTangent2: O2 ∈ mutual_points)
  (hTangent3: O3 ∈ mutual_points)
  (hTouchLine1: ∃ P ∈ line, ∃ r₁ > 0, ∀ x ∈ O1, ∥P - x∥ = r₁)
  (hTouchLine2: ∃ Q ∈ line, ∃ r₂ > 0, ∀ x ∈ O2, ∥Q - x∥ = r₂)
  (hTouchLine3: ∃ R ∈ line, ∃ r₃ > 0, ∀ x ∈ O3, ∥R - x∥ = r₃)
  (hMutual: A ∈ mutual_points ∧ B ∈ mutual_points ∧ C ∈ mutual_points):
  cot (brocard_angle A B C) = 2 := sorry

end brocard_angle_cot_eq_2_l810_810840


namespace Evan_dog_weight_l810_810556

-- Define the weights of the dogs as variables
variables (E I : ℕ)

-- Conditions given in the problem
def Evan_dog_weight_wrt_Ivan (I : ℕ) : ℕ := 7 * I
def dogs_total_weight (E I : ℕ) : Prop := E + I = 72

-- Correct answer we need to prove
theorem Evan_dog_weight (h1 : Evan_dog_weight_wrt_Ivan I = E)
                          (h2 : dogs_total_weight E I)
                          (h3 : I = 9) : E = 63 :=
by
  sorry

end Evan_dog_weight_l810_810556


namespace area_bounded_region_l810_810401

theorem area_bounded_region :
  (∃ (x y : ℝ), y^2 + 2 * x * y + 50 * |x| = 500) →
  ∃ (area : ℝ), area = 1250 :=
by
  sorry

end area_bounded_region_l810_810401


namespace median_of_sequence_l810_810452

theorem median_of_sequence : 
  let sequence := (List.range 3030).map (λ n, n + 1) ++ (List.range 3030).map (λ n, (n + 1) ^ 2)
  let sorted_sequence := sequence.qsort (· ≤ ·)
  (sorted_sequence[3030-1] + sorted_sequence[3030]) / 2 = 2975.5 :=
by
  have h_seq_len : sequence.length = 6060 :=
    by sorry
  have h_sorted_seq : List.sorted (· ≤ ·) sorted_sequence :=
    by sorry
  have h_median_pos : sequence.length / 2 = 3030 :=
    by sorry
  have h_median_values : sorted_sequence[3030-1] = 2975 ∧ sorted_sequence[3030] = 2976 :=
    by sorry
  calc
    (sorted_sequence[3030-1] + sorted_sequence[3030]) / 2 = (2975 + 2976) / 2 := by rw [h_median_values]
    ... = 2975.5 := by norm_num

end median_of_sequence_l810_810452


namespace jill_total_time_l810_810677

def distance_up : ℝ := 900
def speed_up : ℝ := 9
def distance_down : ℝ := 900
def speed_down : ℝ := 12

theorem jill_total_time :
  let time_up := distance_up / speed_up in
  let time_down := distance_down / speed_down in
  let total_time := time_up + time_down in
  total_time = 175 :=
by
  sorry

end jill_total_time_l810_810677


namespace max_rootless_quads_l810_810592

theorem max_rootless_quads (n : ℕ) :
  let rootless (a b c : ℝ) := b^2 - 4 * a * c < 0 in
  (∀ moves, first_player_maximizes_rootless n moves) → (∃ (k : ℕ), k ≥ (n + 1) / 2) :=
begin
  sorry
end

end max_rootless_quads_l810_810592


namespace tennis_tournament_matches_l810_810832

theorem tennis_tournament_matches (n : ℕ) (h₁ : n = 128) (h₂ : ∃ m : ℕ, m = 32) (h₃ : ∃ k : ℕ, k = 96) (h₄ : ∀ i : ℕ, i > 1 → i ≤ n → ∃ j : ℕ, j = 1 + (i - 1)) :
  ∃ total_matches : ℕ, total_matches = 127 := 
by 
  sorry

end tennis_tournament_matches_l810_810832


namespace total_buttons_l810_810269

-- Define the conditions
def shirts_per_kid : Nat := 3
def number_of_kids : Nat := 3
def buttons_per_shirt : Nat := 7

-- Define the statement to prove
theorem total_buttons : shirts_per_kid * number_of_kids * buttons_per_shirt = 63 := by
  sorry

end total_buttons_l810_810269


namespace part1_unique_zero_part2_inequality_l810_810190

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x + 1 / x

theorem part1_unique_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

theorem part2_inequality (n : ℕ) (h : n > 0) : 
  Real.log ((n + 1) / n) < 1 / Real.sqrt (n^2 + n) := by
  sorry

end part1_unique_zero_part2_inequality_l810_810190


namespace distance_AD_35_l810_810730

-- Definitions based on conditions
variables (A B C D : Point)
variable (distance : Point → Point → ℝ)
variable (angle : Point → Point → Point → ℝ)
variable (dueEast : Point → Point → Prop)
variable (northOf : Point → Point → Prop)

-- Conditions
def conditions : Prop :=
  dueEast A B ∧
  angle A B C = 90 ∧
  distance A C = 15 * Real.sqrt 3 ∧
  angle B A C = 30 ∧
  northOf D C ∧
  distance C D = 10

-- The question: Proving the distance between points A and D
theorem distance_AD_35 (h : conditions A B C D distance angle dueEast northOf) :
  distance A D = 35 :=
sorry

end distance_AD_35_l810_810730


namespace correct_sampling_probability_correct_l810_810442

structure City :=
  (factories_A factories_B factories_C : ℕ)
  (total_factories : ℕ)
  (sampling_ratio : ℚ)

def myCity : City :=
  { factories_A := 9,
    factories_B := 18,
    factories_C := 18,
    total_factories := 45,
    sampling_ratio := 1 / 9 }

def sampled_factories (c : City) : ℕ × ℕ × ℕ :=
  (c.factories_A * c.sampling_ratio.num / c.sampling_ratio.denom,
   c.factories_B * c.sampling_ratio.num / c.sampling_ratio.denom,
   c.factories_C * c.sampling_ratio.num / c.sampling_ratio.denom)

theorem correct_sampling (c : City) :
  sampled_factories c = (1, 2, 2) :=
by
  cases c
  simp [sampled_factories, myCity, sampling_ratio]
  sorry

def pairs_and_probability : ℚ :=
  let pairs := [("A", "B1"), ("A", "B2"), ("A", "C1"), ("A", "C2"), ("B1", "B2"), ("B1", "C1"),
                ("B1", "C2"), ("B2", "C1"), ("B2", "C2"), ("C1", "C2")]
  let pairs_with_C := [("A", "C1"), ("A", "C2"), ("B1", "C1"), ("B1", "C2"), ("B2", "C1"), ("B2", "C2"),
                       ("C1", "C2")]
  pairs_with_C.length / pairs.length

theorem probability_correct :
  pairs_and_probability = 7 / 10 :=
by
  simp [pairs_and_probability]
  sorry

end correct_sampling_probability_correct_l810_810442


namespace negate_exponential_inequality_l810_810822

theorem negate_exponential_inequality :
  ¬ (∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x :=
by
  sorry

end negate_exponential_inequality_l810_810822


namespace math_problem_l810_810611

theorem math_problem
  (f : ℝ → ℝ) (Hf : ∀ x, f x = 3 * x / (2 * x + 3))
  (a : ℕ → ℝ) (Ha1 : a 1 = 1)
  (Ha_rec : ∀ n, a (n + 1) = f (a n))
  (b : ℕ → ℝ) (Hb1 : b 1 = 3)
  (Hb_rec : ∀ n, n ≥ 2 → b n = (a (n - 1)) * (a n))
  (S : ℕ → ℝ) (HS : ∀ n, S n = (finset.range (n + 1)).sum (λ k, b k))
  (m : ℕ) (HS_bound : ∀ n, S n < (m - 2014) / 2) :
  (a 2 = 3 / 5) ∧ (a 3 = 3 / 7) ∧ (a 4 = 1 / 3) ∧ 
  (∀ n, 1 / a (n + 1) = 1 / a n + 2 / 3) ∧ 
  m ≥ 2023 :=
by sorry

end math_problem_l810_810611


namespace sequence_sum_problem_l810_810183

theorem sequence_sum_problem 
  (a : ℕ → ℕ)
  (b : ℕ → ℕ)
  (h1 : ∀ n, S_n = 2 * a n - 1) 
  (h2 : b 1 = a 1) 
  (h3 : b 2 - b 1 = a 2 + 1)
  (c : ℕ → ℚ := λ n, b n / a n) 
  (T : ℕ → ℚ := λ n, ∑ i in finset.range n, c (i + 1)) :
  (∀ n, a n = 2 ^ (n - 1)) → 
  (∀ n, b n = 3 * n - 2) →
  (∀ n, T n = 8 - (3 * n + 4) / (2 ^ (n - 1))) :=
by
  sorry

end sequence_sum_problem_l810_810183


namespace square_area_from_diagonal_l810_810508

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) : ∃ A : ℝ, A = 144 :=
by
  let s := d / Real.sqrt 2
  have s_eq : s = 12 := by
    rw [h]
    field_simp
    norm_num
  use s * s
  rw [s_eq]
  norm_num
  sorry

end square_area_from_diagonal_l810_810508


namespace pyramid_new_volume_l810_810918

theorem pyramid_new_volume
  (V_initial : ℝ) (s h : ℝ) (V_initial_eq : V_initial = (1 / 3) * s^2 * h)
  (new_s : ℝ) (new_h : ℝ)
  (triple_s : new_s = 3 * s)
  (double_h : new_h = 2 * h) :
  let V_new := (1 / 3) * new_s^2 * new_h
  in V_new = 432 :=
by
  -- Initial volume condition
  have V_initial_eq_24 : (1 / 3) * s^2 * h = 24 := by
    rw V_initial_eq; exact rfl
  -- Expression for s^2 h
  have sh_72 : s^2 * h = 72 := by
    rw ← V_initial_eq_24; ring
  -- Calculate new volume
  let V_new := (1 / 3) * (3 * s)^2 * (2 * h)
  have V_new_eq : V_new = 6 * s^2 * h := by
    simp [triple_s, double_h]; ring
  -- Substitute s^2 h = 72
  rw [V_new_eq, sh_72]; simp; done
  sorry

end pyramid_new_volume_l810_810918


namespace find_a_range_l810_810586

theorem find_a_range (a : ℝ) :
  (∀ (p q : ℝ), 0 < p ∧ p < q ∧ q < 1 → (a * log p - 2 * p^2 - (a * log q - 2 * q^2)) / (p - q) > 1) ↔ a ∈ Set.Ici 5 := 
sorry

end find_a_range_l810_810586


namespace extreme_value_condition_l810_810408

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 3 + x + 1

theorem extreme_value_condition (a : ℝ) : 
  (∃ x : ℝ, derivative (f a) x = 0) ↔ a < 0 :=
sorry

end extreme_value_condition_l810_810408


namespace yuna_grandfather_age_l810_810463

def age_yuna : ℕ := 8
def age_father : ℕ := age_yuna + 20
def age_grandfather : ℕ := age_father + 25

theorem yuna_grandfather_age : age_grandfather = 53 := by
  sorry

end yuna_grandfather_age_l810_810463


namespace b_4_lt_b_7_l810_810042

-- Define the sequence
def b (α : ℕ → ℕ) (n : ℕ) : ℚ :=
  let rec aux : ℕ → ℚ
    | 0 => 0
    | k+1 => (aux k) + (1 / (α k + 1)) 
  in 1 + 1 / aux n

-- Define the conditions
variable (α : ℕ → ℕ)
variable hα : ∀ k, α k ∈ Nat.succ <$> Finset.range 1

-- The proof problem
theorem b_4_lt_b_7 (α : ℕ → ℕ) (hα : ∀ k, α k ∈ Nat.succ <$> Finset.range 1) : 
  b α 4 < b α 7 := by
  sorry

end b_4_lt_b_7_l810_810042


namespace range_of_a_l810_810639

open Real

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 * exp x1 - a = 0) ∧ (x2 * exp x2 - a = 0)) ↔ -1 / exp 1 < a ∧ a < 0 :=
sorry

end range_of_a_l810_810639


namespace circle_area_208pi_l810_810550

theorem circle_area_208pi :
  let P := (5 : ℝ, -2 : ℝ)
  let Q := (-7 : ℝ, 6 : ℝ)
  let distance := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let r := distance
  let area := Real.pi * r^2
  area = 208 * Real.pi :=
by
  let P := (5 : ℝ, -2 : ℝ)
  let Q := (-7 : ℝ, 6 : ℝ)
  let distance := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let r := distance
  let area := Real.pi * r^2
  have : area = 208 * Real.pi := sorry
  exact this

end circle_area_208pi_l810_810550


namespace total_trucks_l810_810321

-- Define the number of trucks Namjoon has
def trucks_namjoon : ℕ := 3

-- Define the number of trucks Taehyung has
def trucks_taehyung : ℕ := 2

-- Prove that together, Namjoon and Taehyung have 5 trucks
theorem total_trucks : trucks_namjoon + trucks_taehyung = 5 := by 
  sorry

end total_trucks_l810_810321


namespace mean_sharpening_instances_l810_810960

def pencil_sharpening_instances : List ℕ :=
  [13, 8, 13, 21, 7, 23, 15, 19, 12, 9, 28, 6, 17, 29, 31, 10, 4, 20, 16, 12, 2, 18, 27, 22, 5, 14, 31, 29, 8, 25]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem mean_sharpening_instances :
  mean pencil_sharpening_instances = 18.1 := by
  sorry

end mean_sharpening_instances_l810_810960


namespace triangle_inequality_l810_810224

variables {a b c : ℝ}

def sides_of_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (h : sides_of_triangle a b c) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_l810_810224


namespace area_of_triangle_OAB_is_5_l810_810692

-- Define the parameters and assumptions
def OA : ℝ × ℝ := (-2, 1)
def OB : ℝ × ℝ := (4, 3)

noncomputable def area_triangle_OAB (OA OB : ℝ × ℝ) : ℝ :=
  1 / 2 * (OA.1 * OB.2 - OA.2 * OB.1)

-- The theorem we want to prove:
theorem area_of_triangle_OAB_is_5 : area_triangle_OAB OA OB = 5 := by
  sorry

end area_of_triangle_OAB_is_5_l810_810692


namespace total_cost_of_crayons_l810_810721

-- Definition of the initial conditions
def usual_price : ℝ := 2.5
def discount_rate : ℝ := 0.15
def packs_initial : ℕ := 4
def packs_to_buy : ℕ := 2

-- Calculate the discounted price for one pack
noncomputable def discounted_price : ℝ :=
  usual_price - (usual_price * discount_rate)

-- Calculate the total cost of packs after purchase and validate it
theorem total_cost_of_crayons :
  (packs_initial * usual_price) + (packs_to_buy * discounted_price) = 14.25 :=
by
  sorry

end total_cost_of_crayons_l810_810721


namespace time_per_window_is_correct_l810_810914

-- Definition of the problem's conditions
def total_windows := 14
def installed_windows := 5
def remaining_windows := total_windows - installed_windows
def total_hours := 36

-- Calculate time per window and prove that it is equal to 4 hours
theorem time_per_window_is_correct : total_hours / remaining_windows = 4 := by
  have h1 : remaining_windows = total_windows - installed_windows := rfl
  have h2 : total_windows - installed_windows = 9 := by norm_num
  have h3 : remaining_windows = 9 := h1 ▸ h2
  have h4 : total_hours / remaining_windows = total_hours / 9 := by rw h3
  have h5 : total_hours = 36 := rfl
  have h6 : 36 / 9 = 4 := by norm_num
  exact eq.trans h4 h6

end time_per_window_is_correct_l810_810914


namespace findCompoundInterestRate_l810_810826

def simpleInterest (P R T : ℕ) := (P * R * T) / 100

def compoundInterest (P r T : ℚ) := P * ((1 + r / 100) ^ T - 1)

def givenValues : Prop :=
  let principalSI := 3225
  let rateSI := 8
  let timeSI := 5
  let si := simpleInterest principalSI rateSI timeSI

  let principalCI := 8000
  let timeCI := 2

  si = 258 ∧ 
  ∃ (r : ℚ), si = 1 / 2 * compoundInterest principalCI r timeCI

theorem findCompoundInterestRate (r : ℚ) (h : givenValues) : r ≈ 3.17 := 
sorry

end findCompoundInterestRate_l810_810826


namespace greatest_possible_x_max_possible_x_l810_810777

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l810_810777


namespace ellipse_solutions_l810_810594

def ellipse_eqn (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = 1)
  (arith_seq : 2 * b^2 = a^2 + c^2) : Prop :=
  (a^2 = 3) ∧ (b^2 = 2) ∧ (c^2 = 1)

def chord_length (a b : ℝ) (intersect_pts : ∀ x y : ℝ, x + y = 1 → (x^2 / a^2 + y^2 / b^2 = 1)) : Prop :=
  (sqrt(1+1) * abs((3 + 2*sqrt 6) / 5 - (3 - 2*sqrt 6) / 5) = 8 * sqrt 3 / 5)

def major_axis_range (a : ℝ) (eccentricity : ℝ → Prop) : Prop :=
  (sqrt 5 ≤ 2 * a ∧ 2 * a ≤ sqrt 6)

theorem ellipse_solutions (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = 1)
  (arith_seq : 2 * b^2 = a^2 + c^2)
  (intersect_pts : ∀ x y : ℝ, x + y = 1 → (x^2 / a^2 + y^2 / b^2 = 1))
  (eccentricity_cond : ∀ e : ℝ, sqrt 3 / 3 ≤ e ∧  e ≤ sqrt 2 / 2 → (e = sqrt ((a^2 - b^2) / a^2))) :
Proof :=
begin
  let ellipse_proof := ellipse_eqn a b c h1 h2 h3 arith_seq,
  -- Proving the ellipse equation is equivalent to the given conditions
  have h4 : a^2 = 3, from ellipse_proof.1,
  have h5 : b^2 = 2, from ellipse_proof.2,

  -- Proving the length of the chord
  let chord_proof := chord_length 3 2 intersect_pts,
  -- Prove length of AB

  -- Proving the range of the length of the major axis
  let major_axis_proof := major_axis_range a (λ e, sqrt 3 / 3 ≤ e ∧  e ≤ sqrt 2 / 2 → (e = sqrt ((a^2 - b^2) / a^2))),
  -- Prove range of major axis

  sorry
end

end ellipse_solutions_l810_810594


namespace derivative_of_y_l810_810102

noncomputable def y (x : ℝ) : ℝ := 
  Real.cos (Real.log 13) - (1 / 44) * (Real.cos (22 * x))^2 / Real.sin (44 * x)

theorem derivative_of_y (x : ℝ) : 
  deriv (y x) = 1 / (4 * (Real.sin (22 * x))^2) := by
  sorry

end derivative_of_y_l810_810102


namespace probability_win_all_games_l810_810394

variable (p : ℚ) (n : ℕ)

-- Define the conditions
def probability_of_winning := p = 2 / 3
def number_of_games := n = 6
def independent_games := true

-- The theorem we want to prove
theorem probability_win_all_games (h₁ : probability_of_winning p)
                                   (h₂ : number_of_games n)
                                   (h₃ : independent_games) :
  p^n = 64 / 729 :=
sorry

end probability_win_all_games_l810_810394


namespace scientific_notation_of_solubility_product_l810_810746

theorem scientific_notation_of_solubility_product :
  (∃ (K_sp : ℝ), K_sp = 0.0000000028) → 0.0000000028 = 2.8 * 10^(-9) :=
begin
  intro exists_K_sp,
  cases exists_K_sp with K_sp K_sp_eq,
  rw K_sp_eq,
  sorry
end

end scientific_notation_of_solubility_product_l810_810746


namespace h_j_h_of_3_l810_810700

def h (x : ℤ) : ℤ := 5 * x + 2
def j (x : ℤ) : ℤ := 3 * x + 4

theorem h_j_h_of_3 : h (j (h 3)) = 277 := by
  sorry

end h_j_h_of_3_l810_810700


namespace collinear_points_l810_810707

noncomputable section

variables {A B C D P Q S T : Type*}
variables [field A] [field B] [field C] [field D]
variables [add_comm_monoid P] [add_comm_monoid Q] [add_comm_monoid S] [add_comm_monoid T]

-- Define points A, B, C, D on a conic E
variables (E : Type*) [has_mem A E] [has_mem B E] [has_mem C E] [has_mem D E]

-- Intersection of tangents to E at A, B defines point P
axiom tangent_intersection_AB (E : Type*) [has_mem A E] [has_mem B E] : P

-- Intersection of tangents to E at C, D defines point Q
axiom tangent_intersection_CD (E : Type*) [has_mem C E] [has_mem D E] : Q

-- Intersection of line (AC) and (BD) defines point T
axiom line_intersection_AC_BD (E : Type*) [has_mem A E] [has_mem C E] [has_mem B E] [has_mem D E] : T

-- Intersection of line (AD) and (BC) defines point S
axiom line_intersection_AD_BC (E : Type*) [has_mem A E] [has_mem D E] [has_mem B E] [has_mem C E] : S

-- Prove that points P, Q, S, and T are collinear
theorem collinear_points (E : Type*) [has_mem A E] [has_mem B E] [has_mem C E] [has_mem D E] [add_comm_monoid P] [add_comm_monoid Q] [add_comm_monoid S] [add_comm_monoid T]
  (tangent_intersection_AB E) (tangent_intersection_CD E) (line_intersection_AC_BD E) (line_intersection_AD_BC E):
  collinear P Q S T :=
begin
  sorry
end

end collinear_points_l810_810707


namespace parabola_zero_sum_l810_810403

theorem parabola_zero_sum (x a b : ℝ)
  (h1 : ∀ x, (x-2)^2 + 3)  -- Original equation
  (h2 : ∀ x, -((x-1)-2)^2 + 3)  -- After rotation
  (h3 : ∀ x, -((x+1)-2)^2 + 3)  -- After shift left
  (h4 : ∀ x, -(x+1)^2 + 1)  -- After shift down
  (zero_eq_x : h4 x = 0)
  (zero_eq_a_b : ∀ a b, h4 a = 0 ∧ h4 b = 0)
: a + b = -2 :=
sorry

end parabola_zero_sum_l810_810403


namespace expand_product_l810_810559

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 :=
by
  sorry

end expand_product_l810_810559


namespace geometry_proof_problem_l810_810885

noncomputable def P : Point := sorry
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry

def tangent_at_P (τ: Circle) := sorry
def intersect_at (τ₁ τ₂: Circle) (pt: Point) := sorry
def distinct_points (pts : List Point) := sorry

axiom four_distinct_circles : ∃ (τ₁ τ₂ τ₃ τ₄ : Circle), 
  (tangent_at_P τ₁) ∧ (tangent_at_P τ₃) ∧ (tangent_at_P τ₂) ∧ (tangent_at_P τ₄) ∧
  (intersect_at τ₁ τ₂ A) ∧ (intersect_at τ₂ τ₃ B) ∧ (intersect_at τ₃ τ₄ C) ∧ (intersect_at τ₄ τ₁ D) ∧
  (distinct_points [A, B, C, D, P])

axiom tangent_circles_condition : ∀ (τ₁ τ₂ τ₃ τ₄ : Circle), 
  tangent_at_P τ₁ ∧ tangent_at_P τ₃ ∧ tangent_at_P τ₂ ∧ tangent_at_P τ₄ ∧
  intersect_at τ₁ τ₂ A ∧ intersect_at τ₂ τ₃ B ∧ intersect_at τ₃ τ₄ C ∧ intersect_at τ₄ τ₁ D → 
  distinct_points [A, B, C, D, P]

theorem geometry_proof_problem : 
  ∀ (τ₁ τ₂ τ₃ τ₄ : Circle), 
  (tangent_at_P τ₁) ∧ (tangent_at_P τ₃) ∧ (tangent_at_P τ₂) ∧ (tangent_at_P τ₄) ∧
  (intersect_at τ₁ τ₂ A) ∧ (intersect_at τ₂ τ₃ B) ∧ (intersect_at τ₃ τ₄ C) ∧ (intersect_at τ₄ τ₁ D) ∧
  (distinct_points [A, B, C, D, P]) →
  (AB * BC) / (AD * DC) = (PB^2) / (PD^2) :=
begin
  sorry
end

end geometry_proof_problem_l810_810885


namespace range_H_l810_810869

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem range_H : set.range H = {5} :=
sorry

end range_H_l810_810869


namespace prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l810_810580

noncomputable def total_outcomes := 24
noncomputable def outcomes_two_correct := 6
noncomputable def outcomes_at_least_two_correct := 7
noncomputable def outcomes_all_incorrect := 9

theorem prob_two_correct : (outcomes_two_correct : ℚ) / total_outcomes = 1 / 4 := by
  sorry

theorem prob_at_least_two_correct : (outcomes_at_least_two_correct : ℚ) / total_outcomes = 7 / 24 := by
  sorry

theorem prob_all_incorrect : (outcomes_all_incorrect : ℚ) / total_outcomes = 3 / 8 := by
  sorry

end prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l810_810580


namespace roots_equation_1352_l810_810698

theorem roots_equation_1352 {c d : ℝ} (hc : c^2 - 6 * c + 8 = 0) (hd : d^2 - 6 * d + 8 = 0) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 1352 :=
by
  sorry

end roots_equation_1352_l810_810698


namespace max_f_eq_max_g_exp_sub_log_gt_2x_two_pow_pi_le_pi_sq_min_mn_l810_810825

-- Definition of the functions f and g
def f (x : ℝ) := (Real.log x) / x
def g (x : ℝ) := x / Real.exp x

-- Option A: Prove the maximum value of f(x) equals the maximum value of g(x)
theorem max_f_eq_max_g : (∀ x > 0, f(x) ≤ 1/e) ∧ (∃ x > 0, f(x) = 1/e) ∧ (∀ x, g(x) ≤ 1/e) ∧ (∃ x, g(x) = 1/e) := sorry

-- Option B: Prove e^x - ln x > 2x for x > 0
theorem exp_sub_log_gt_2x (x : ℝ) (h : x > 0) : Real.exp x - Real.log x > 2 * x := sorry

-- Option C: Prove 2^π ≤ π²
theorem two_pow_pi_le_pi_sq : 2^Real.pi ≤ Real.pi^2 := sorry

-- Option D: If ln(m)/m = n/e^n < 0, prove mn has a minimum value of -1/e
theorem min_mn (m n : ℝ) (h1 : m > 0) (h2 : (Real.log m) / m = n / Real.exp n) (h3 : (Real.log m) / m < 0) : m * n ≥ -1/Real.exp 1 := sorry

end max_f_eq_max_g_exp_sub_log_gt_2x_two_pow_pi_le_pi_sq_min_mn_l810_810825


namespace identify_roles_l810_810927

-- Define the types for the personas
inductive Persona : Type
| A
| B
| C

open Persona

-- Define the roles
inductive Role : Type
| Knight  -- always tells truth
| Liar    -- always lies
| Spy     -- can lie or tell truth

open Role

-- Define predicates for each statement
def statement_A : Persona → Prop
| C => false  -- A said "C is a liar"

def statement_C : Persona → Prop
| A => true   -- C said "A is a knight"

def C_claim_identity : Role → Prop
| Spy => true  -- C said "I am the spy"
| _  => false

-- Define the proof problem
theorem identify_roles (rA rB rC : Role) (h_A : statement_A C) (h_C : statement_C A) (h_C_identity : C_claim_identity Spy) :
  rA = Knight ∧ rB = Spy ∧ rC = Liar :=
by
  -- Add your proof here
  sorry

end identify_roles_l810_810927


namespace reflection_over_vector_l810_810994

open Real EuclideanSpace

theorem reflection_over_vector : 
  let v : ℝ × ℝ := (2, 6)
  let u : ℝ × ℝ := (2, 1)
  let proj_v_u : ℝ × ℝ := ((v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)) * u
  let r : ℝ × ℝ := 2 * proj_v_u - v
  r = (6, -2) 
:= by
  let v : ℝ × ℝ := (2, 6)
  let u : ℝ × ℝ := (2, 1)
  let proj_v_u : ℝ × ℝ := ((v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)) * u
  let r : ℝ × ℝ := 2 * proj_v_u - v
  sorry

end reflection_over_vector_l810_810994


namespace repeating_decimal_sum_l810_810418

theorem repeating_decimal_sum (a b c d : ℕ) (h1 : (5 : ℚ) / 13 = 0.abcdabcdabcd) :
  a + b + c + d = 20 :=
sorry

end repeating_decimal_sum_l810_810418


namespace minimum_distance_to_line_l810_810621

noncomputable def curve_min_distance : ℝ :=
  let ℂ := {p : ℝ × ℝ | ∃ α : ℝ, 0 ≤ α ∧ α < 2 * π ∧ p.1 = 2 * Real.cos α ∧ p.2 = Real.sin α} in 
  let line := {p : ℝ × ℝ | p.1 - 2 * p.2 - 4 * Real.sqrt 2 = 0} in
  let dist (p : ℝ × ℝ) := abs (p.1 - 2 * p.2 - 4 * Real.sqrt 2) / Real.sqrt 5 in
  Real.Inf (dist '' ℂ)

theorem minimum_distance_to_line : curve_min_distance = 2 * Real.sqrt 10 / 5 :=
begin
  -- Proof omitted
  sorry
end

end minimum_distance_to_line_l810_810621


namespace five_student_committees_l810_810148

theorem five_student_committees (n k : ℕ) (hn : n = 8) (hk : k = 5) : 
  nat.choose n k = 56 := by
  rw [hn, hk]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end five_student_committees_l810_810148


namespace geometric_sequence_sum_l810_810829

theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℚ) (a : ℚ) :
  (∀ n, S n = (1 / 2) * 3^(n + 1) - a) →
  S 1 - (S 2 - S 1)^2 = (S 2 - S 1) * (S 3 - S 2) →
  a = 3 / 2 :=
by
  intros hSn hgeo
  sorry

end geometric_sequence_sum_l810_810829


namespace hired_is_c6_l810_810905

variable (Candidate : Type) [Fintype Candidate] [DecidableEq Candidate]

-- Define candidates
variable (c1 c2 c3 c4 c5 c6 : Candidate)

-- Define the predictions
def A_pred : Candidate → Prop := fun c => c ≠ c6
def B_pred : Candidate → Prop := fun c => c = c4 ∨ c = c5
def C_pred : Candidate → Prop := fun c => c = c1 ∨ c = c2 ∨ c = c3
def D_pred : Candidate → Prop := fun c => c ≠ c1 ∧ c ≠ c2 ∧ c ≠ c3

-- Define the main hypothesis
axiom main_hypothesis : ∀ c, (A_pred c → B_pred c → false) ∨ (A_pred c → C_pred c → false) ∨ 
                                    (A_pred c → D_pred c → false) ∨ 
                                    (B_pred c → C_pred c → false) ∨ (B_pred c → D_pred c → false) ∨ 
                                    (C_pred c → D_pred c → false)

-- Define the hired candidate
def Hired : Candidate :=
    if A_pred c4 ∧ B_pred c4 ∧ D_pred c4 then c6 else sorry -- Corresponding logic to be inserted
    
theorem hired_is_c6 : Hired = c6 :=
by
  sorry

end hired_is_c6_l810_810905


namespace slope_of_intersections_l810_810574

-- Define the two lines' equations in terms of real numbers x, y, s
def line1 (s x y : ℝ) : Prop := 2 * x + 3 * y = 8 * s + 4
def line2 (s x y : ℝ) : Prop := x + 2 * y = 3 * s - 1

-- The intersection point must satisfy both line equations
def intersection_point (s x y : ℝ) : Prop := line1 s x y ∧ line2 s x y

-- We are to prove the slope of the line on which all intersection points (x, y) lie
theorem slope_of_intersections : ∀ s : ℝ, ∃ m : ℝ, ∀ x y : ℝ, intersection_point s x y → (y = m * x + _ ) :=
sorry

end slope_of_intersections_l810_810574


namespace greatest_value_of_x_l810_810808

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l810_810808


namespace sin_product_difference_proof_l810_810554

noncomputable def sin_product_difference : Prop :=
  sin 70 * sin 65 - sin 20 * sin 25 = sqrt 2 / 2

theorem sin_product_difference_proof : sin_product_difference :=
  by sorry

end sin_product_difference_proof_l810_810554


namespace point_of_tangency_is_correct_l810_810993

theorem point_of_tangency_is_correct : 
  (∃ (x y : ℝ), y = x^2 + 20 * x + 63 ∧ x = y^2 + 56 * y + 875 ∧ x = -19 / 2 ∧ y = -55 / 2) :=
by
  sorry

end point_of_tangency_is_correct_l810_810993


namespace ellipse_area_l810_810101

theorem ellipse_area :
  ∃ a b : ℝ, 
    (∀ x y : ℝ, (x^2 - 2 * x + 9 * y^2 + 18 * y + 16 = 0) → 
    (a = 2 ∧ b = (2 / 3) ∧ (π * a * b = 4 * π / 3))) :=
sorry

end ellipse_area_l810_810101


namespace tree_volume_estimation_proof_l810_810936

noncomputable def average_root_cross_sectional_area (x_i : list ℝ) := (x_i.sum) / (x_i.length)
noncomputable def average_volume (y_i : list ℝ) := (y_i.sum) / (y_i.length)
noncomputable def correlation_coefficient (x_i y_i : list ℝ) : ℝ :=
  let n := x_i.length in
  let x_bar := average_root_cross_sectional_area x_i in
  let y_bar := average_volume y_i in
  let numerator := (list.zip x_i y_i).sum (λ ⟨x, y⟩, (x - x_bar) * (y - y_bar)) in
  let denominator_x := (x_i.sum (λ x, (x - x_bar)^2)) in
  let denominator_y := (y_i.sum (λ y, (y - y_bar)^2)) in
  numerator / ((denominator_x * denominator_y).sqrt)

noncomputable def total_volume_estimate (total_area avg_y avg_x : ℝ) := (avg_y / avg_x) * total_area

theorem tree_volume_estimation_proof :
  let x_i := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06] in
  let y_i := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40] in
  let total_area := 186 in
  average_root_cross_sectional_area x_i = 0.06 ∧
  average_volume y_i = 0.39 ∧
  correlation_coefficient x_i y_i ≈ 0.97 ∧
  total_volume_estimate total_area 0.39 0.06 = 1209 :=
by
  sorry

end tree_volume_estimation_proof_l810_810936


namespace find_a_and_b_l810_810301

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_and_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 40) : a + b = 7 :=
by
  sorry

end find_a_and_b_l810_810301


namespace union_of_sets_l810_810001

noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 4, 6}

theorem union_of_sets : A ∪ B = {1, 2, 4, 6} := 
by 
sorry

end union_of_sets_l810_810001


namespace no_solution_integral_pairs_l810_810972

theorem no_solution_integral_pairs (a b : ℤ) : (1 / (a : ℚ) + 1 / (b : ℚ) = -1 / (a + b : ℚ)) → false :=
by
  sorry

end no_solution_integral_pairs_l810_810972


namespace friends_meeting_both_movie_and_games_l810_810890

theorem friends_meeting_both_movie_and_games 
  (T M P G M_and_P P_and_G M_and_P_and_G : ℕ) 
  (hT : T = 31) 
  (hM : M = 10) 
  (hP : P = 20) 
  (hG : G = 5) 
  (hM_and_P : M_and_P = 4) 
  (hP_and_G : P_and_G = 0) 
  (hM_and_P_and_G : M_and_P_and_G = 2) : (M + P + G - M_and_P - T + M_and_P_and_G - 2) = 2 := 
by 
  sorry

end friends_meeting_both_movie_and_games_l810_810890


namespace greatest_possible_x_max_possible_x_l810_810780

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l810_810780


namespace number_of_ordered_triples_l810_810297

def S : Finset ℕ := Finset.range 21 \ 0  -- Set S = {1,2,...,20}

def succ (a b : ℕ) : Prop := (0 < a - b ∧ a - b ≤ 10) ∨ (b - a > 10)

theorem number_of_ordered_triples : 
  (∑ x in S, ∑ y in S, ∑ z in S, if (succ x y ∧ succ y z ∧ succ z x) then 1 else 0) = 720 := 
sorry

end number_of_ordered_triples_l810_810297


namespace total_buttons_l810_810270

-- Define the conditions
def shirts_per_kid : Nat := 3
def number_of_kids : Nat := 3
def buttons_per_shirt : Nat := 7

-- Define the statement to prove
theorem total_buttons : shirts_per_kid * number_of_kids * buttons_per_shirt = 63 := by
  sorry

end total_buttons_l810_810270


namespace problem_solution_l810_810660

-- Define the conditions
def C1_parametric (a b : ℝ) (ϕ : ℝ) : ℝ × ℝ := (a * Real.cos ϕ, b * Real.sin ϕ)

noncomputable def C1_eq (x y a b : ℝ) : Prop :=
  x^2/16 + y^2/4 = 1

noncomputable def C2_eq (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ

theorem problem_solution
  (a b x y ρ1 ρ2 θ : ℝ)
  (h₀ : a = 4 ∧ b = 2)
  (hC1 : C1_eq x y a b)
  (hC2 : C2_eq ρ1 θ)
  (hA : C1_eq (ρ1 * Real.cos θ) (ρ1 * Real.sin θ) a b)
  (hB : C1_eq (ρ2 * Real.cos (θ + Real.pi / 2)) (ρ2 * Real.sin (θ + Real.pi / 2)) a b) :
  1 / (ρ1^2 + ρ2^2) = 5 / 16 :=
sorry

end problem_solution_l810_810660


namespace relationship_between_n_and_m_l810_810314

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

noncomputable def geometric_sequence (a q : ℝ) (m : ℕ) : ℝ :=
  a * q ^ (m - 1)

theorem relationship_between_n_and_m
  (a d q : ℝ) (n m : ℕ)
  (h_d_ne_zero : d ≠ 0)
  (h1 : arithmetic_sequence a d 1 = geometric_sequence a q 1)
  (h2 : arithmetic_sequence a d 3 = geometric_sequence a q 3)
  (h3 : arithmetic_sequence a d 7 = geometric_sequence a q 5)
  (q_pos : 0 < q) (q_sqrt2 : q^2 = 2)
  :
  n = 2 ^ ((m + 1) / 2) - 1 := sorry

end relationship_between_n_and_m_l810_810314


namespace problem_M_l810_810633

theorem problem_M (M : ℤ) (h : 1989 + 1991 + 1993 + 1995 + 1997 + 1999 + 2001 = 14000 - M) : M = 35 :=
by
  sorry

end problem_M_l810_810633


namespace evaluate_expression_quartic_l810_810571

theorem evaluate_expression_quartic (n : ℤ) (h : n ≥ 13) : 
  ∃ p : polynomial ℤ, ∀ (x : ℤ), p = polynomial.X^4 + 5 * polynomial.X^3 + 4 * polynomial.X^2 + polynomial.X ∧ 
  (∃ c : ℤ, p.eval x = c * x * (x + 1) ∧ 
  ((n+1) * n * (n^2 + 5 * n + 4) = c * p.eval n)) :=
by {
  sorry
}

end evaluate_expression_quartic_l810_810571


namespace line_through_A_is_correct_l810_810640

theorem line_through_A_is_correct (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1)
  (A : Point) (hA : A = (2, 1)) :
  (∀ x y, y = log a (x - 1) + 1 → A = (2, 1)) → 
  (line_through A (2, 1) 2 = ({p : Point | x = 2}) ∨
    line_through A (2, 1) 2 = ({p : Point | 3 * p.1 + 4 * p.2 = 10})) :=
by
  sorry

end line_through_A_is_correct_l810_810640


namespace volume_hex_prism_l810_810928

-- Define the conditions and variables as per the problem
def side_length (a : ℝ) : Prop := a = 5
def height (m : ℝ) : Prop := m = √69
def base_area (A_t : ℝ) : Prop := A_t = 37.5 * √3
def volume (V : ℝ) : Prop := V = 37.5 * √207

theorem volume_hex_prism (a m A_t V: ℝ) (h1: side_length a) (h2: height m) (h3: base_area A_t) :
  volume V :=
by
  -- The proof steps are omitted as per instruction, hence using sorry
  sorry

end volume_hex_prism_l810_810928


namespace birthday_friends_count_l810_810349

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l810_810349


namespace find_magnitude_l810_810177

open Real

variables (a b : ℝ^3)
variable (θ : ℝ)

def norm (v : ℝ^3) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

axiom norm_a : norm a = 1
axiom norm_b : norm b = 1
axiom angle_ab : θ = pi / 3
axiom dot_product : a.dot b = 1 * 1 * cos θ

theorem find_magnitude : 
  norm (2 • a - b) = sqrt 3 := 
sorry

end find_magnitude_l810_810177


namespace find_b_l810_810997

noncomputable def polynomial_has_four_non_real_roots 
  (a b c d : ℝ) (z w : ℂ) : Prop :=
  z ≠ conj z ∧ w ≠ conj w ∧
  (z * w = -7 + 4i) ∧
  (conj z + conj w = -2 - 6i)

theorem find_b 
  (a b c d : ℝ) (z w : ℂ) 
  (h : polynomial_has_four_non_real_roots a b c d z w) : 
  b = 26 :=
sorry

end find_b_l810_810997


namespace value_this_year_l810_810427

def last_year_value : ℝ := 20000
def depreciation_factor : ℝ := 0.8

theorem value_this_year :
  last_year_value * depreciation_factor = 16000 :=
by
  sorry

end value_this_year_l810_810427


namespace quadratic_identity_1_quadratic_identity_2_l810_810553

noncomputable def f (x a b : ℝ) := x^2 + a * x + b
noncomputable def f' (x a : ℝ) := 2 * x + a

theorem quadratic_identity_1 :
  ∀ x : ℝ, f (x + 1) (-5/3) (2/3) =
    (-1/2) * ( ∫ t in 0..1, (3 * x^2 + 4 * x * t) * f' t (-5/3) ) := 
by sorry

theorem quadratic_identity_2 :
  ∀ x : ℝ, f (x + 1) (-2/3) (-1/3) =
    1 * ( ∫ t in 0..1, (3 * x^2 + 4 * x * t) * f' t (-2/3) ) := 
by sorry

end quadratic_identity_1_quadratic_identity_2_l810_810553


namespace dog_catches_rabbit_in_4_minutes_l810_810091

noncomputable def time_to_catch_up (dog_speed rabbit_speed head_start : ℝ) : ℝ :=
  head_start / (dog_speed - rabbit_speed) * 60

theorem dog_catches_rabbit_in_4_minutes :
  time_to_catch_up 24 15 0.6 = 4 :=
by
  unfold time_to_catch_up
  norm_num
  rfl

end dog_catches_rabbit_in_4_minutes_l810_810091


namespace smallest_population_multiple_of_3_l810_810414

theorem smallest_population_multiple_of_3 : 
  ∃ (a : ℕ), ∃ (b c : ℕ), 
  a^2 + 50 = b^2 + 1 ∧ b^2 + 51 = c^2 ∧ 
  (∃ m : ℕ, a * a = 576 ∧ 576 = 3 * m) :=
by
  sorry

end smallest_population_multiple_of_3_l810_810414


namespace floor_ceil_eq_l810_810175

theorem floor_ceil_eq (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌊x⌋ - x = 0 :=
by
  sorry

end floor_ceil_eq_l810_810175


namespace part1_part2_l810_810624

open Set

variable {R : Type} [OrderedRing R]

def U : Set R := univ
def A : Set R := {x | x^2 - 2*x - 3 > 0}
def B : Set R := {x | 4 - x^2 <= 0}

theorem part1 : A ∩ B = {x | -2 ≤ x ∧ x < -1} :=
sorry

theorem part2 : (U \ A) ∪ (U \ B) = {x | x < -2 ∨ x > -1} :=
sorry

end part1_part2_l810_810624


namespace triangle_inscribed_circle_radius_not_max_tetrahedron_inscribed_sphere_radius_not_max_l810_810168

-- Part 1: Triangle with Sides of Unit Length
theorem triangle_inscribed_circle_radius_not_max (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) :
    let ρ := x * sqrt (1 - x^2) / (1 + x),
        ρ_base1 := (sqrt 3) / 6
    in ρ ≠ ρ_base1 := 
sorry

-- Part 2: Regular Tetrahedron with Edge Length of Unit Length
theorem tetrahedron_inscribed_sphere_radius_not_max (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1 / sqrt 2) :
    let ρ := x * sqrt (1 - 2 * x^2) / (sqrt (1 - x^2) + x),
        ρ_base1 := (x : ℝ) -> if x = 0.5 then 0.2588 else 0.2597
    in ρ ≠ ρ_base1 0.5 :=
sorry

end triangle_inscribed_circle_radius_not_max_tetrahedron_inscribed_sphere_radius_not_max_l810_810168


namespace raise_3000_yuan_probability_l810_810845

def prob_correct_1 : ℝ := 0.9
def prob_correct_2 : ℝ := 0.5
def prob_correct_3 : ℝ := 0.4
def prob_incorrect_3 : ℝ := 1 - prob_correct_3

def fund_first : ℝ := 1000
def fund_second : ℝ := 2000
def fund_third : ℝ := 3000

def prob_raise_3000_yuan : ℝ := prob_correct_1 * prob_correct_2 * prob_incorrect_3

theorem raise_3000_yuan_probability :
  prob_raise_3000_yuan = 0.27 :=
by
  sorry

end raise_3000_yuan_probability_l810_810845


namespace proof_area_of_intersection_is_zero_l810_810441

-- Define the vertices of the rectangle
def vertex_1 : ℝ × ℝ := (5, 11)
def vertex_2 : ℝ × ℝ := (16, 11)
def vertex_3 : ℝ × ℝ := (16, -8)
def vertex_4 : ℝ × ℝ := (5, -8)

-- Define the center and radius of the circle
def circle_center : ℝ × ℝ := (5, 1)
def radius : ℝ := 5

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := (x - circle_center.1)^2 + (y - circle_center.2)^2 = radius^2

-- Define a function to check if a point is inside the circle
def inside_circle (p : ℝ × ℝ) : Prop := circle_eq p.1 p.2

-- Define the vertices of the rectangle
def rectangle_vertices : list (ℝ × ℝ) := [vertex_1, vertex_2, vertex_3, vertex_4]

-- Define a dummy function to represent the area of intersection is 0
-- Note that this is a placeholder for the actual computation.
noncomputable def area_of_intersection : ℝ := 0

-- Prove that area of intersection is 0
theorem proof_area_of_intersection_is_zero :
  area_of_intersection = 0 := 
by sorry

end proof_area_of_intersection_is_zero_l810_810441


namespace louis_current_age_l810_810233

/-- 
  In 6 years, Carla will be 30 years old. 
  The sum of the current ages of Carla and Louis is 55. 
  Prove that Louis is currently 31 years old.
--/
theorem louis_current_age (C L : ℕ) 
  (h1 : C + 6 = 30) 
  (h2 : C + L = 55) 
  : L = 31 := 
sorry

end louis_current_age_l810_810233


namespace parabola_x_intercepts_count_l810_810971

theorem parabola_x_intercepts_count : 
  ∀ (a b c : ℝ), 
  a = 3 → b = 5 → c = -8 → 
  (∃ x : ℝ, y = 3 * x^2 + 5 * x - 8 = 0) → 
  (b^2 - 4 * a * c > 0) → 
  ∃ x1 x2 : ℝ, y = 3 * x1^2 + 5 * x1 - 8 = 0 ∧ y = 3 * x2^2 + 5 * x2 - 8 = 0 ∧ x1 ≠ x2 := 
by 
   sorry

end parabola_x_intercepts_count_l810_810971


namespace lemonade_sales_l810_810848

theorem lemonade_sales (total_amount small_amount medium_amount large_price sales_price_small sales_price_medium earnings_small earnings_medium : ℕ) (h1 : total_amount = 50) (h2 : sales_price_small = 1) (h3 : sales_price_medium = 2) (h4 : large_price = 3) (h5 : earnings_small = 11) (h6 : earnings_medium = 24) : large_amount = 5 :=
by
  sorry

end lemonade_sales_l810_810848


namespace louis_current_age_l810_810235

-- Define the constants for years to future and future age of Carla
def years_to_future : ℕ := 6
def carla_future_age : ℕ := 30

-- Define the sum of current ages
def sum_current_ages : ℕ := 55

-- State the theorem
theorem louis_current_age :
  ∃ (c l : ℕ), (c + years_to_future = carla_future_age) ∧ (c + l = sum_current_ages) ∧ (l = 31) :=
sorry

end louis_current_age_l810_810235


namespace greatest_x_lcm_105_l810_810790

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l810_810790


namespace tom_sent_roses_for_7_days_l810_810443

theorem tom_sent_roses_for_7_days (dozen: ℕ) (roses_per_day: ℕ) (total_roses: ℕ) :
    dozen = 12 → roses_per_day = 2 * dozen → total_roses = 168 → total_roses / roses_per_day = 7 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  exact Nat.div_eq (168 / 24) h3
  sorry

end tom_sent_roses_for_7_days_l810_810443


namespace dice_probability_l810_810841

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_square (n : ℕ) : Prop := n = 1 ∨ n = 4
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5

theorem dice_probability :
  let outcomes := 6 * 6 * 6 in
  let red_success := 3 in
  let green_success := 2 in
  let blue_success := 3 in
  let successful_combinations := red_success * green_success * blue_success in
  let probability := successful_combinations.to_rat / outcomes.to_rat in
  probability = (1 / 12 : ℚ) :=
by
  sorry

end dice_probability_l810_810841


namespace suitable_survey_is_D_l810_810459

-- Define the surveys
def survey_A := "Survey on the viewing of the movie 'The Long Way Home' by middle school students in our city"
def survey_B := "Survey on the germination rate of a batch of rose seeds"
def survey_C := "Survey on the water quality of the Jialing River"
def survey_D := "Survey on the health codes of students during the epidemic"

-- Define what it means for a survey to be suitable for a comprehensive census
def suitable_for_census (survey : String) : Prop :=
  survey = survey_D

-- Define the main theorem statement
theorem suitable_survey_is_D : suitable_for_census survey_D :=
by
  -- We assume sorry here to skip the proof
  sorry

end suitable_survey_is_D_l810_810459


namespace friends_at_birthday_l810_810367

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l810_810367


namespace num_unique_four_digit_numbers_l810_810382

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}
def evens : Finset ℕ := digits.filter is_even
def odds : Finset ℕ := digits.filter is_odd

theorem num_unique_four_digit_numbers : 
  ∃ (c : ℕ), 
  c = Nat.choose evens.card 2 * Nat.choose odds.card 2 * Nat.factorial 4 ∧
  c = 378 := 
sorry

end num_unique_four_digit_numbers_l810_810382


namespace no_prime_5n3_l810_810220

theorem no_prime_5n3 (n : ℕ) (h1 : ∃ x : ℕ, x^2 = 2 * n + 1)
                            (h2 : ∃ y : ℕ, y^2 = 3 * n + 1) : ¬ prime (5 * n + 3) := by
    sorry

end no_prime_5n3_l810_810220


namespace product_vs_average_l810_810538

theorem product_vs_average :
  let diff1 := 0.65 * 1500 - 0.55 * 1800,
      diff2 := 0.45 * 2300 - 0.30 * 2100,
      diff3 := 0.75 * 3200 - 0.40 * 2800,
      sum_diffs := diff1 + diff2 + diff3,
      product_diffs := diff1 * diff2 * diff3,
      avg_squares := (diff1^2 + diff2^2 + diff3^2) / 3 in
  product_diffs - avg_squares = -8328883.333 :=
by {
  let diff1 := 0.65 * 1500 - 0.55 * 1800,
  let diff2 := 0.45 * 2300 - 0.30 * 2100,
  let diff3 := 0.75 * 3200 - 0.40 * 2800,
  let product_diffs := diff1 * diff2 * diff3,
  let avg_squares := (diff1^2 + diff2^2 + diff3^2) / 3,
  have h_diff1 : diff1 = -15 := by norm_num,
  have h_diff2 : diff2 = 405 := by norm_num,
  have h_diff3 : diff3 = 1280 := by norm_num,
  have h_product_diffs : product_diffs = -7728000 := by norm_num,
  have h_avg_squares : avg_squares = 600883.33 := by norm_num,
  calc
    product_diffs - avg_squares = -7728000 - 600883.333 : by rw [h_product_diffs, h_avg_squares]
                          ... = -8328883.333 : by norm_num
}

end product_vs_average_l810_810538


namespace complete_square_to_d_l810_810093

-- Conditions given in the problem
def quadratic_eq (x : ℝ) : Prop := x^2 + 10 * x + 7 = 0

-- Equivalent Lean 4 statement of the problem
theorem complete_square_to_d (x : ℝ) (c d : ℝ) (h : quadratic_eq x) (hc : c = 5) : (x + c)^2 = d → d = 18 :=
by sorry

end complete_square_to_d_l810_810093


namespace solve_quadratic_eq_solve_linear_system_l810_810888

theorem solve_quadratic_eq (x : ℚ) : 4 * (x - 1) ^ 2 - 25 = 0 ↔ x = 7 / 2 ∨ x = -3 / 2 := 
by sorry

theorem solve_linear_system (x y : ℚ) : (2 * x - y = 4) ∧ (3 * x + 2 * y = 1) ↔ (x = 9 / 7 ∧ y = -10 / 7) :=
by sorry

end solve_quadratic_eq_solve_linear_system_l810_810888


namespace circle_tangent_to_line_at_parabola_focus_l810_810399

noncomputable def parabola_focus : (ℝ × ℝ) := (2, 0)

def line_eq (p : ℝ × ℝ) : Prop := p.2 = p.1

def circle_eq (center radius : ℝ) (p : ℝ × ℝ) : Prop := 
  (p.1 - center)^2 + p.2^2 = radius

theorem circle_tangent_to_line_at_parabola_focus : 
  ∀ p : ℝ × ℝ, (circle_eq 2 2 p ↔ (line_eq p ∧ p = parabola_focus)) := by
  sorry

end circle_tangent_to_line_at_parabola_focus_l810_810399


namespace num_five_student_committees_l810_810125

theorem num_five_student_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 5) : choose n k = 56 :=
by
  rw [h_n, h_k]
  -- rest of the proof would go here
  sorry

end num_five_student_committees_l810_810125


namespace line_intersects_y_axis_at_0_2_l810_810952

theorem line_intersects_y_axis_at_0_2 :
  ∃ y : ℝ, (2, 8) ≠ (4, 14) ∧ ∀ x: ℝ, (3 * x + y = 2) ∧ x = 0 → y = 2 :=
by
  sorry

end line_intersects_y_axis_at_0_2_l810_810952


namespace find_lambda_l810_810626

open Real

variable (AB AC : ℝ^3)
variable (λ : ℝ)
variable (P : ℝ^3)

noncomputable def AP := λ • AB + AC
noncomputable def BC := AC - AB

axiom magnitude_AB : ∥AB∥ = 3
axiom magnitude_AC : ∥AC∥ = 2
axiom angle_ABC : Real.Angle (AB) (AC) = π * (2/3)
axiom perpendicular_condition : (AP λ AC AB) ∙ (BC AC AB) = 0

theorem find_lambda (h1 : ∥AB∥ = 3)
                    (h2 : ∥AC∥ = 2)
                    (h3 : ∀ AB AC, Real.Angle AB AC = π * (2 / 3))
                    (h4 : (AP λ AC AB) ∙ (BC AC AB) = 0) :
  λ = 7 / 3 := sorry

end find_lambda_l810_810626


namespace parallelogram_altitude_length_l810_810252

-- Conditions
variables {A B C D E F : Point}
variable [parallelogram ABCD]
variable (DE : Line) -- altitude to base AB
variable (DF : Line) -- altitude to base AD
variable (DC : ℝ) (EB : ℝ) (DE_len : ℝ)
variable [DC_eq : DC = 15] [EB_eq : EB = 3] [DE_eq : DE_len = 5]

-- Theorem statement
theorem parallelogram_altitude_length (DF_len : ℝ) 
    (AB_eq_DC : AB = 15)
    (area_eq : AB * DE_len = 75)
    (AD_eq : AD = 15) :
    DF_len = 5 :=
  sorry

end parallelogram_altitude_length_l810_810252


namespace sin_cos_power_identity_l810_810583

variables {x : ℝ} {n : ℕ}
hypothesis h1 : sin x + cos x = -1

theorem sin_cos_power_identity (h1 : sin x + cos x = -1) : sin x ^ n + cos x ^ n = (-1:ℝ) ^ n :=
sorry

end sin_cos_power_identity_l810_810583


namespace surface_area_circumscribed_sphere_l810_810178

theorem surface_area_circumscribed_sphere (a h : ℝ) (ha : a = 2) (hh : h = 3) : 
  let R := (1 / 2) * Real.sqrt (h^2 + (a * Real.sqrt 2)^2)
  in 4 * Real.pi * R^2 = 17 * Real.pi := 
by
  -- Insert conditions
  rw [ha, hh]
  -- Expand and simplify as per the provided solution steps
  have d := a * Real.sqrt 2,
  rw ha at d,
  rw [← Real.sqrt_mul (2:ℝ) _] at d,
  let R := (1 / 2) * Real.sqrt (h^2 + d^2),
  rw hh at R,
  rw [← Real.sqrt_add_pow_mul_rat] at R,
  sorry -- Proof steps skipped

end surface_area_circumscribed_sphere_l810_810178


namespace AngleB_Conditions_l810_810294

-- Definitions
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (R : ℝ) -- Circumradius
variables (b : ℝ) -- Side length opposite angle B
variables (BO BH : ℝ) -- Distances

-- Conditions
def Circumcenter (O : A) : Prop := ∃ (A B C : A), ∀ (P : A), dist P O = R
def Orthocenter (H : A) : Prop := ∃ (A B C : A), ∀ (P : A), dist (P, H) = R

-- Question
def AngleB (angles : Set ℝ) : Prop :=
  ∃ (B : ℝ), (B ∈ angles)

-- Lean 4 statement to prove
theorem AngleB_Conditions (O H : A) (angle_B : ℝ) (angles : Set ℝ) :
  Circumcenter O → Orthocenter H → BO = BH → angles = {60, 120} → AngleB angles :=
by
  sorry

end AngleB_Conditions_l810_810294


namespace estimation_problems_l810_810938

noncomputable def average_root_cross_sectional_area (x : list ℝ) : ℝ :=
  (list.sum x) / (list.length x)

noncomputable def average_volume (y : list ℝ) : ℝ :=
  (list.sum y) / (list.length y)

noncomputable def sample_correlation_coefficient (x y : list ℝ) : ℝ :=
  let n := list.length x
      avg_x := average_root_cross_sectional_area x
      avg_y := average_volume y
      sum_xy := (list.zip_with (*) x y).sum
      sum_x2 := (x.map (λ xi, xi * xi)).sum
      sum_y2 := (y.map (λ yi, yi * yi)).sum
  in (sum_xy - n * avg_x * avg_y) / (real.sqrt ((sum_x2 - n * avg_x^2) * (sum_y2 - n * avg_y^2)))

theorem estimation_problems :
  let x := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
      y := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
      X := 186
  in
    average_root_cross_sectional_area x = 0.06 ∧
    average_volume y = 0.39 ∧
    abs (sample_correlation_coefficient x y - 0.97) < 0.01 ∧
    (average_volume y / average_root_cross_sectional_area x) * X = 1209 :=
by
  sorry

end estimation_problems_l810_810938


namespace combination_eight_choose_five_l810_810142

theorem combination_eight_choose_five : 
  ∀ (n k : ℕ), n = 8 ∧ k = 5 → Nat.choose n k = 56 :=
by 
  intros n k h
  obtain ⟨hn, hk⟩ := h
  rw [hn, hk]
  exact Nat.choose_eq 8 5
  sorry  -- This signifies that the proof needs to be filled in, but we'll skip it as per instructions.

end combination_eight_choose_five_l810_810142


namespace five_student_committees_from_eight_l810_810128

theorem five_student_committees_from_eight : nat.choose 8 5 = 56 := by
  sorry

end five_student_committees_from_eight_l810_810128


namespace investment_of_c_l810_810467

variable (P_a P_b P_c C_a C_b C_c : ℝ)

theorem investment_of_c (h1 : P_b = 3500) 
                        (h2 : P_a - P_c = 1399.9999999999998) 
                        (h3 : C_a = 8000) 
                        (h4 : C_b = 10000) 
                        (h5 : P_a / C_a = P_b / C_b) 
                        (h6 : P_c / C_c = P_b / C_b) : 
                        C_c = 40000 := 
by 
  sorry

end investment_of_c_l810_810467


namespace polygons_homothetic_l810_810306

variables {n : ℕ} (A : ℕ → (ℝ × ℝ)) -- This represents the n-gon vertices A_1, A_2, ..., A_n.

def centroid (A : ℕ → (ℝ × ℝ)) (N : ℕ) : (ℝ × ℝ) :=
    let vs := finset.univ.sum (λ i : fin N, A i.1) in
    ((vs.1 : ℝ) / N, (vs.2 : ℝ) / N)

noncomputable def centroid_n_gon : (ℝ × ℝ) := centroid A n

noncomputable def centroid_n_minus_1_gon (i : fin n) : (ℝ × ℝ) :=
    centroid (λ j, if j < i.val then A j else A (j+1)) (n-1)

theorem polygons_homothetic :
    ∃ (M : (ℝ × ℝ)) (r : ℝ), r = -1 / (n-1) ∧
    centroid_n_gon A = M ∧
    ∀ i, centroid_n_minus_1_gon A ⟨i, sorry⟩ = (r • (A i - M) + M) :=
sorry

end polygons_homothetic_l810_810306


namespace paolo_sevilla_birthday_l810_810345

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l810_810345


namespace problem_statement_l810_810693

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vec_dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem problem_statement : vec_dot (vec_add a (vec_scalar_mul 2 b)) c = -3 := 
by
  sorry

end problem_statement_l810_810693


namespace arithmetic_sequence_formula_minimum_lambda_l810_810169

noncomputable def a_n (n : ℕ) : ℕ := n + 1
def S_4 := 14
def is_geometric (a1 a3 a7 : ℕ) := (a3 * a3 = a1 * a7)

theorem arithmetic_sequence_formula 
  (h1 : S_4 = (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4))
  (h2 : is_geometric (a_n 1) (a_n 3) (a_n 7)) :
  ∀ n : ℕ, a_n n = n + 1 :=
sorry

def T_n (n : ℕ) : ℝ := (n : ℝ) / (2 * (n + 2))
def a_n1 (n : ℕ) : ℝ := (n : ℝ) + 2

theorem minimum_lambda :
  ∃ λ : ℝ, (∀ n : ℕ, T_n n ≤ λ * a_n1 n) ∧ λ = 1/16 :=
sorry

end arithmetic_sequence_formula_minimum_lambda_l810_810169


namespace calculate_x_n_minus_inverse_x_n_l810_810219

theorem calculate_x_n_minus_inverse_x_n
  (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π) (x : ℝ) (h : x - 1/x = 2 * Real.sin θ) (n : ℕ) (hn : 0 < n) :
  x^n - 1/x^n = 2 * Real.sinh (n * θ) :=
by sorry

end calculate_x_n_minus_inverse_x_n_l810_810219


namespace june_1_friday_l810_810646

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

end june_1_friday_l810_810646


namespace price_of_each_pair_of_cleats_l810_810627

theorem price_of_each_pair_of_cleats 
  (cards_price bat_price glove_discount glove_original_price total_earnings : ℕ)
  (cleats_total cleats_count : ℕ)
  (h1 : cards_price = 25)
  (h2 : bat_price = 10)
  (h3 : glove_discount = 20)
  (h4 : glove_original_price = 30)
  (h5 : total_earnings = 79)
  (h6 : cleats_count = 2) :
  cleats_total / cleats_count = 10 := 
by
  let glove_sale_price := glove_original_price - (glove_original_price * glove_discount / 100)
  let total_without_cleats := cards_price + bat_price + glove_sale_price
  let cleats_total := total_earnings - total_without_cleats
  have h7 : glove_sale_price = 24 := by sorry -- proof omitted
  have h8 : total_without_cleats = 59 := by sorry -- proof omitted
  have h9 : cleats_total = 20 := by sorry -- proof omitted
  show cleats_total / cleats_count = 10 from by sorry -- proof provided

end price_of_each_pair_of_cleats_l810_810627


namespace max_chord_length_ellipse_l810_810821

theorem max_chord_length_ellipse : 
  ∀ (t : ℝ), let line := λ (x : ℝ), x + t in
  (let chord_length := 
    if (3:ℝ)*x^2 + (4:ℝ)*t*x + (2:ℝ)*t^2 - 2 = 0 
    then real.sqrt(2) * real.sqrt(((-4 * t) / 3)^2 - 4 * ((2 * t^2 - 2) / 3)) / 3
    else 0 in
  chord_length ≤ (4 * real.sqrt(3)) / 3) :=
sorry

end max_chord_length_ellipse_l810_810821


namespace magnitude_ge_one_l810_810161

theorem magnitude_ge_one (a : ℝ) : 
  let z := a + (1 : ℂ) * complex.I in 
  complex.abs z ≥ 1 :=
sorry

end magnitude_ge_one_l810_810161


namespace pascal_triangle_odd_rows_count_l810_810076

/-- 
  Identifies the number of rows that consist entirely of odd numbers 
  (excluding the 1 at each end) among the first 20 rows of Pascal's triangle.
-/
theorem pascal_triangle_odd_rows_count :
  let S := {n | n ∈ finset.range 21 ∧ ∀ k ∈ finset.range (n + 1), (nat.choose n k) % 2 = 1}
  in S.card = 3 := 
by {
  -- S defined as the set of row indices where all elements in the row (except the ends) are odd
  let S := {n | n ∈ finset.range 21 ∧ ∀ k ∈ finset.range (n + 1), (nat.choose n k) % 2 = 1},
  -- Cardinality of S is 3
  exact S.card = 3,
  sorry
}

end pascal_triangle_odd_rows_count_l810_810076


namespace friends_attended_birthday_l810_810357

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l810_810357


namespace number_of_sets_A_l810_810552

theorem number_of_sets_A : 
  (∃ (s : set (set ℕ)), (∀ A ∈ s, {2} ⊆ A ∧ A ⊆ {1, 2, 3}) ∧ s.card = 3) :=
sorry

end number_of_sets_A_l810_810552


namespace proof_main_l810_810255

noncomputable def ellipseC_eq : Prop :=
  ∀ (P : ℝ × ℝ), 
    let F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
    let F2 : ℝ × ℝ := (Real.sqrt 2, 0)
    dist P F1 + dist P F2 = 4 ↔
    ∃ x y : ℝ, P = (x, y) ∧ (x^2 / 4 + y^2 / 2 = 1)

noncomputable def line_AB_tangent_to_circle : Prop :=
  ∀ (A B : ℝ × ℝ),
    let O : ℝ × ℝ := (0, 0)
    let circle : ℝ × ℝ → Prop := λ P, P.1 ^ 2 + P.2 ^ 2 = 2
    let y_eq_2 : ℝ × ℝ → Prop := λ P, P.2 = 2
    (A ∈ ellipseC_eq) →
    (y_eq_2 B) →
    (O.1 * B.1 + O.2 * B.2 = 0) →
    ∃ line : ℝ → ℝ,
      (∀ t : ℝ, B.1 = A.1 + t ∧ B.2 = A.2 + t / A.1) ∧
      tangent line circle

theorem proof_main : ellipseC_eq ∧ line_AB_tangent_to_circle := sorry

end proof_main_l810_810255


namespace equal_distances_l810_810291

noncomputable def is_square (A B C D : ℝ × ℝ) :=
  (A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2) ∧
  (B.1 - A.1 = C.1 - B.1 ∧ C.2 - B.2 = D.2 - C.2) ∧
  (A.1 - D.1 = B.1 - A.1)

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

theorem equal_distances (
  A B C D M N : ℝ × ℝ) 
  (h1 : is_square A B C D)
  (h2 : M.1 ∈ Set.Icc 0 (B.1 - A.1) ∧ M.2 = A.2)
  (h3 : N.2 ∈ Set.Icc (B.2 - B.1) C.2 ∧ N.1 = B.1)
  (h4 : ∠(M, D, N) = 45)
  (R := midpoint M N) : 
  let P : ℝ × ℝ := (some_hull_definition) -- Placeholder
  let Q : ℝ × ℝ := (another_hull_definition) -- Placeholder
  (P = Q) → (dist R P = dist R Q) := 
by sorry

end equal_distances_l810_810291


namespace max_x_lcm_15_21_105_l810_810772

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l810_810772


namespace min_terms_to_represent_in_zero_country_l810_810243

/-- In "-0 Country", numbers are represented by sums of terms with digits 1 and 0.
Prove that the minimum number of terms needed to represent 20120204 is 4. -/
theorem min_terms_to_represent_in_zero_country : 
  ∀ S : list ℕ, (∀ x ∈ S, (∀ d ∈ x.digits 10, d = 0 ∨ d = 1)) ∧ S.sum = 20120204 → S.length ≥ 4 :=
by
  -- To be proved
  sorry

end min_terms_to_represent_in_zero_country_l810_810243


namespace B_completes_in_40_days_l810_810895

noncomputable def BCompletesWorkInDays (x : ℝ) : ℝ :=
  let A_rate := 1 / 45
  let B_rate := 1 / x
  let work_done_together := 9 * (A_rate + B_rate)
  let work_done_B_alone := 23 * B_rate
  let total_work := 1
  work_done_together + work_done_B_alone

theorem B_completes_in_40_days :
  BCompletesWorkInDays 40 = 1 :=
by
  sorry

end B_completes_in_40_days_l810_810895


namespace sequence_behavior_l810_810305

noncomputable def arithmetic_mean (x y : ℝ) := (x + y) / 2
noncomputable def geometric_mean (x y : ℝ) := real.sqrt (x * y)
noncomputable def harmonic_mean (x y : ℝ) := 2 / (1 / x + 1 / y)
noncomputable def quadratic_mean (x y : ℝ) := real.sqrt ((x^2 + y^2) / 2)

theorem sequence_behavior (x y : ℝ) (h₀ : 1 < x) (h₁ : 1 < y) (hₓ : x ≠ y) :
  let A := λ n : ℕ, nat.rec (arithmetic_mean x y) (λ _ A_n, (A_n + nat.rec (harmonic_mean x y) (λ _ H_n, H_n / (1 + ((H_n) ⁻¹)) sorry) sorry) / 2) n,
      G := λ n : ℕ, nat.rec (geometric_mean x y) (λ _ _, (A (n - 1) * A (n - 1))⁻¹⁻) n,
      H := λ n : ℕ, nat.rec (harmonic_mean x y) (λ _ H_n, 2 / (1 / (A (n - 1)) + (1 / H_n))) n,
      Q := λ n : ℕ, nat.rec (quadratic_mean x y) (λ _ Q_n, real.sqrt ((Q_n^2 + H (n - 1)^2) / 2)) n in
  (∀ n, A n ≥ A (n + 1)) ∧ (∀ n, G n = G 0) ∧ (∀ n, H n ≤ H (n + 1)) ∧ (∀ n, Q n ≤ Q (n + 1)) := sorry

end sequence_behavior_l810_810305


namespace area_of_triangle_is_8sqrt2_l810_810618

-- Define the hyperbola with eccentricity condition
def hyperbola_asymptote_area (e m : ℝ) (hyp : e = real.sqrt 3) : ℝ :=
  if h : m = 2 then
    let x := 4
    let y := 2 * real.sqrt 2
    -- Area of the triangle is
    1 / 2 * x * y
  else 0

-- Since m should satisfy the eccentricity condition
def m := 2

theorem area_of_triangle_is_8sqrt2 : hyperbola_asymptote_area (real.sqrt 3) m (by norm_num) = 8 * real.sqrt 2 :=
  sorry

end area_of_triangle_is_8sqrt2_l810_810618


namespace max_value_of_b_l810_810991

theorem max_value_of_b (b x : ℝ) :
  b * real.sqrt b * (x^2 - 10 * x + 25) + real.sqrt b / (x^2 - 10 * x + 25) <= 
  (1/5) * real.root b 4 ^ 3 * abs (real.sin (real.pi * x / 10)) ↔ 
  b <= (1/10000) :=
sorry

end max_value_of_b_l810_810991


namespace q0_r0_eq_three_l810_810701

variable (p q r s : Polynomial ℝ)
variable (hp_const : p.coeff 0 = 2)
variable (hs_eq : s = p * q * r)
variable (hs_const : s.coeff 0 = 6)

theorem q0_r0_eq_three : (q.coeff 0) * (r.coeff 0) = 3 := by
  sorry

end q0_r0_eq_three_l810_810701


namespace problem_statement_l810_810032

open Real

noncomputable def correlation_coefficient (x y : List ℝ) : ℝ :=
  let n := x.length
  let x̄ := List.sum x / n.toReal
  let ȳ := List.sum y / n.toReal
  let numerator := List.sum (List.zipWith (λ xi yi, (xi - x̄) * (yi - ȳ)) x y)
  let denominator := sqrt (List.sum (List.map (λ xi, (xi - x̄)^2) x) * List.sum (List.map (λ yi, (yi - ȳ)^2) y))
  numerator / denominator

def residual_correct (x y : List ℝ) (predicted_y : ℝ) (actual_y : ℝ) : ℝ :=
  actual_y - predicted_y

theorem problem_statement :
  -- Given conditions
  let x := [32.2, 31.1, 32.9, 35.7, 37.1, 38.0, 39.0, 43.0, 44.6, 46.0]
  let y := [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0]
  let n := 10
  let sum_x := 379.6
  let sum_y := 391
  let sum_x_squared := 246.904
  let sum_y_squared := 568.9
  let covariance_xy := m
  let mean_x := 37.96
  let mean_y := 39.1
  let r := correlation_coefficient x y
  -- Proving the statements
  r ≈ 0.95 →
  ∃ a b : ℝ, 
    b = 1.44 ∧
    a = -15.56 ∧
    ∀ xi yi, 
    (xi, yi) ∈ List.zip x y →
    residual_correct xi yi (b * xi + a) ≈ -5.81 :=
by sorry

end problem_statement_l810_810032


namespace range_of_inequality_l810_810171

noncomputable def even_function_monotonically_increasing_on_nonneg (f : ℝ → ℝ) :=
  (∀ x, f x = f (-x)) ∧ (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)

theorem range_of_inequality (f : ℝ → ℝ)
  (hf : even_function_monotonically_increasing_on_nonneg f) :
  {x : ℝ | 0 ≤ x ∧ f x > f (2 * x - 1)} = set.Ioo (1 / 3) 1 :=
by
  sorry

end range_of_inequality_l810_810171


namespace unique_real_root_interval_l810_810181

theorem unique_real_root_interval {a : ℝ} : 
  (∃! x : ℝ, x ∈ set.Ioi 1 ∧ x * real.log x + (3 - a) * x + a = 0) → (5 < a ∧ a < 6) :=
by sorry

end unique_real_root_interval_l810_810181


namespace solve_quadratic_l810_810587

theorem solve_quadratic (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
sorry

end solve_quadratic_l810_810587


namespace circle_numbers_exist_l810_810061

theorem circle_numbers_exist :
  ∃ (a b c d e f : ℚ),
    a = 2 ∧
    b = 3 ∧
    c = 3 / 2 ∧
    d = 1 / 2 ∧
    e = 1 / 3 ∧
    f = 2 / 3 ∧
    a = b * f ∧
    b = a * c ∧
    c = b * d ∧
    d = c * e ∧
    e = d * f ∧
    f = e * a := by
  sorry

end circle_numbers_exist_l810_810061


namespace product_of_possible_values_l810_810258

theorem product_of_possible_values : 
  ∀ x ∈ ({ x : ℝ | |x - 5| - 6 = 1 }), x = 12 ∨ x = -2 → 
  (∀ a b, (a, b) ∈ ({(12, -2), (-2, 12)} : set (ℝ × ℝ)) → a * b = -24) :=
by
  sorry

end product_of_possible_values_l810_810258


namespace birthday_friends_count_l810_810350

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l810_810350


namespace max_x_lcm_15_21_105_l810_810767

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l810_810767


namespace inequality_always_true_l810_810529

theorem inequality_always_true {x : ℝ} (hx : 0 < x) : e^x >= e :=
by
  sorry

end inequality_always_true_l810_810529


namespace total_games_season_is_636_l810_810023

-- Define the number of teams and their divisions
def teams_per_division : ℕ := 8
def total_divisions : ℕ := 3
def total_teams : ℕ := total_divisions * teams_per_division

-- Define the number of games played within and between divisions
def intra_division_games (teams_per_div : ℕ) : ℕ := (teams_per_div * (teams_per_div - 1) / 2) * 3
def inter_division_games (teams_per_div : ℕ) (total_divs : ℕ) : ℕ := (teams_per_div * (total_divs - 1) * teams_per_div) * 2

-- Define the total games considering each game is counted for both teams
def total_games_in_season (teams_per_div : ℕ) (total_divs : ℕ) : ℕ :=
  let teams_count := teams_per_div * total_divs in
  ((intra_division_games teams_per_div + inter_division_games teams_per_div total_divs) * teams_count) / 2

-- Prove the total number of games is as expected
theorem total_games_season_is_636 : total_games_in_season teams_per_division total_divisions = 636 := by
  sorry

end total_games_season_is_636_l810_810023


namespace Mikail_money_left_after_purchase_l810_810320

def Mikail_age_tomorrow : ℕ := 9  -- Defining Mikail's age tomorrow as 9.

def gift_per_year : ℕ := 5  -- Defining the gift amount per year of age as $5.

def video_game_cost : ℕ := 80  -- Defining the cost of the video game as $80.

def calculate_gift (age : ℕ) : ℕ := age * gift_per_year  -- Function to calculate the gift money he receives based on his age.

-- The statement we need to prove:
theorem Mikail_money_left_after_purchase : 
    calculate_gift Mikail_age_tomorrow < video_game_cost → calculate_gift Mikail_age_tomorrow - video_game_cost = 0 :=
by
  sorry

end Mikail_money_left_after_purchase_l810_810320


namespace min_value_dot_product_ab_l810_810625

-- Define the vectors a and b
def vec_a (x : ℝ) := (1, x)
def vec_b (x : ℝ) := (x, x + 1)

-- Define the dot product function for 2D vectors
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

-- Define the function to find the value of the dot product with vectors a and b
def dot_product_ab (x : ℝ) : ℝ :=
  dot_product (vec_a x) (vec_b x)

-- The theorem to prove that the minimum value of the dot product of vectors a and b is -1
theorem min_value_dot_product_ab : 
  ∃ x : ℝ, dot_product_ab x = -1 :=
by
  sorry

end min_value_dot_product_ab_l810_810625


namespace tomatoes_initially_l810_810909

-- Conditions
def tomatoes_picked_yesterday : ℕ := 56
def tomatoes_picked_today : ℕ := 41
def tomatoes_left_after_yesterday : ℕ := 104

-- The statement to prove
theorem tomatoes_initially : tomatoes_left_after_yesterday + tomatoes_picked_yesterday + tomatoes_picked_today = 201 :=
  by
  -- Proof steps would go here
  sorry

end tomatoes_initially_l810_810909


namespace best_model_is_model_4_l810_810260

noncomputable def Model1_R2 : ℝ := 0.03
noncomputable def Model2_R2 : ℝ := 0.6
noncomputable def Model3_R2 : ℝ := 0.4
noncomputable def Model4_R2 : ℝ := 0.97

def best_fitting_model (model_R2: List ℝ): ℝ :=
  model_R2.foldl max 0

theorem best_model_is_model_4 :
  best_fitting_model [Model1_R2, Model2_R2, Model3_R2, Model4_R2] = Model4_R2 :=
by
  -- Proof goes here
  sorry

end best_model_is_model_4_l810_810260


namespace trig_identity_l810_810071

noncomputable def sin_deg (x : ℝ) := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) := Real.cos (x * Real.pi / 180)
noncomputable def tan_deg (x : ℝ) := Real.tan (x * Real.pi / 180)

theorem trig_identity :
  (2 * sin_deg 50 + sin_deg 10 * (1 + Real.sqrt 3 * tan_deg 10) * Real.sqrt 2 * (sin_deg 80)^2) = Real.sqrt 6 :=
by
  sorry

end trig_identity_l810_810071


namespace minimize_I_l810_810588

def H (p q : ℝ) := -3 * p * q + 2 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

def I (p : ℝ) : ℝ := Real.sup (Set.image (H p) (Set.Icc 0 1))

theorem minimize_I : ∃ p : ℝ, (0 ≤ p ∧ p ≤ 1) ∧ I p = I (9/14) ∧ ∀ q ∈ Set.Icc (0 : ℝ) 1, I p ≤ I q :=
by
  sorry

end minimize_I_l810_810588


namespace pq_greater_than_one_l810_810760

/-- Given the function y = -x^2 + px + q which intersects the x-axis at points (a, 0) and (b, 0) 
with b < 1 < a, we want to prove that p + q > 1. -/
theorem pq_greater_than_one (a b p q : ℝ) 
  (h1 : b < 1) 
  (h2 : 1 < a) 
  (h3 : y_eq : ∀ x, -x^2 + p * x + q = 0 ↔ (x = a) ∨ (x = b)) :
  p + q > 1 := 
by 
  sorry

end pq_greater_than_one_l810_810760


namespace probability_specific_female_selected_l810_810919

open Function

section

def total_students : ℕ := 50
def male_students : ℕ := 30
def female_students_not_specific : ℕ := 19
def female_students_specific : ℕ := 1
def total_students_with_conditions : ℕ := male_students + female_students_not_specific + female_students_specific -- that is 30 + 19 + 1

def choose (n k : ℕ) :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem probability_specific_female_selected :
  choose total_students 5 > 0 → 
  total_students = 50 → 
  male_students + female_students_not_specific + female_students_specific = 50 →
  ∃ probability : ℚ, probability = (choose (total_students - 1) 4 : ℚ) / (choose total_students 5 : ℚ) ∧
  probability = 1 / 10 :=
by
  intros h_positive h_total h_group
  use (choose (total_students - 1) 4 : ℚ) / (choose total_students 5 : ℚ)
  split
  { sorry }
  { sorry }

end

end probability_specific_female_selected_l810_810919


namespace b4_lt_b7_l810_810044

noncomputable def b : ℕ → ℚ
| 1       := 1 + 1 / (1 : ℚ)
| (n + 1) := (1 + 1 / ((b n) + 1 / (1 : ℚ)))

theorem b4_lt_b7 (α : ℕ → ℕ) (hα : ∀ n, α n > 0) : b 4 < b 7 :=
by
  sorry

end b4_lt_b7_l810_810044


namespace b_4_lt_b_7_l810_810041

-- Define the sequence
def b (α : ℕ → ℕ) (n : ℕ) : ℚ :=
  let rec aux : ℕ → ℚ
    | 0 => 0
    | k+1 => (aux k) + (1 / (α k + 1)) 
  in 1 + 1 / aux n

-- Define the conditions
variable (α : ℕ → ℕ)
variable hα : ∀ k, α k ∈ Nat.succ <$> Finset.range 1

-- The proof problem
theorem b_4_lt_b_7 (α : ℕ → ℕ) (hα : ∀ k, α k ∈ Nat.succ <$> Finset.range 1) : 
  b α 4 < b α 7 := by
  sorry

end b_4_lt_b_7_l810_810041


namespace rotate_W_results_in_M_l810_810831

-- Define the axisymmetric property of the letter W
def is_axisymmetric (letter : Char) : Prop :=
  letter = 'W'

-- Define the transformation of rotating a letter by 180 degrees around a point O on its axis of symmetry
def rotate_180 (letter : Char) (point : Point) : Char :=
  if is_axisymmetric letter then 'M' else letter

-- Define a point as needed, since 'Point' is not used further and only represents 'any point O'
structure Point where
  x : Float
  y : Float

-- Proposition stating that rotating letter 'W' by 180 degrees around any point on its axis of symmetry results in 'M'
theorem rotate_W_results_in_M (O : Point) : rotate_180 'W' O = 'M' :=
  by
  sorry

end rotate_W_results_in_M_l810_810831


namespace total_pushups_l810_810464

def Zachary_pushups : ℕ := 44
def David_pushups : ℕ := Zachary_pushups + 58

theorem total_pushups : Zachary_pushups + David_pushups = 146 := by
  sorry

end total_pushups_l810_810464


namespace equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l810_810983
open BigOperators

-- First, we define the three equations and their constraints
def equation1_solution (k : ℤ) : ℤ × ℤ := (2 - 5 * k, -1 + 3 * k)
def equation2_solution (k : ℤ) : ℤ × ℤ := (8 - 5 * k, -4 + 3 * k)
def equation3_solution (k : ℤ) : ℤ × ℤ := (16 - 39 * k, -25 + 61 * k)

-- Define the proof that the supposed solutions hold for each equation
theorem equation1_solution_valid (k : ℤ) : 3 * (equation1_solution k).1 + 5 * (equation1_solution k).2 = 1 :=
by
  -- Proof steps would go here
  sorry

theorem equation2_solution_valid (k : ℤ) : 3 * (equation2_solution k).1 + 5 * (equation2_solution k).2 = 4 :=
by
  -- Proof steps would go here
  sorry

theorem equation3_solution_valid (k : ℤ) : 183 * (equation3_solution k).1 + 117 * (equation3_solution k).2 = 3 :=
by
  -- Proof steps would go here
  sorry

end equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l810_810983


namespace find_scalars_l810_810299

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 7], ![-3, -1]]
def M_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![-17, 7], ![-3, -20]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem find_scalars :
  ∃ p q : ℤ, M_squared = p • M + q • I ∧ (p, q) = (1, -19) := sorry

end find_scalars_l810_810299


namespace number_is_10_l810_810483

theorem number_is_10 (x : ℕ) (h : x * 15 = 150) : x = 10 :=
sorry

end number_is_10_l810_810483


namespace louis_current_age_l810_810237

-- Define the constants for years to future and future age of Carla
def years_to_future : ℕ := 6
def carla_future_age : ℕ := 30

-- Define the sum of current ages
def sum_current_ages : ℕ := 55

-- State the theorem
theorem louis_current_age :
  ∃ (c l : ℕ), (c + years_to_future = carla_future_age) ∧ (c + l = sum_current_ages) ∧ (l = 31) :=
sorry

end louis_current_age_l810_810237


namespace greatest_possible_x_max_possible_x_l810_810773

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l810_810773


namespace garden_columns_l810_810242

theorem garden_columns (rows : ℕ) (tree_distance : ℕ) (boundary_distance : ℕ) (garden_length : ℕ) 
  (h_rows : rows = 10)
  (h_tree_distance : tree_distance = 2)
  (h_boundary_distance : boundary_distance = 5)
  (h_garden_length : garden_length = 32) :
  let available_length := garden_length - 2 * boundary_distance in
  let spaces_between_trees := available_length / tree_distance in
  let columns := spaces_between_trees + 1 in
  columns = 12 :=
by
  sorry

end garden_columns_l810_810242


namespace eval_expression_l810_810000

open Real

theorem eval_expression :
  (0.8^5 - (0.5^6 / 0.8^4) + 0.40 + 0.5^3 - log 0.3 + sin (π / 6)) = 2.51853302734375 :=
  sorry

end eval_expression_l810_810000


namespace height_of_pile_correct_l810_810856

-- Definitions based on the conditions
def diameter_pipe : ℝ := 10
def radius_pipe : ℝ := diameter_pipe / 2

-- The height of a stack of 3 pipes in crate B
def height_pile_of_pipes : ℝ := 
  let height_triangle := (radius_pipe * Math.sqrt 3) in
  radius_pipe + height_triangle + radius_pipe

-- Goal: Prove the height of the pile is 10 + 5√3 cm
theorem height_of_pile_correct : height_pile_of_pipes = 10 + 5 * Real.sqrt 3 := 
  sorry

end height_of_pile_correct_l810_810856


namespace rubbish_equal_N2_rubbish_equal_N10_l810_810432

-- Define the conditions and questions for N = 2
theorem rubbish_equal_N2 (a1 a2 : ℕ) : (2 * a1 = a2) ↔ 
    (a1' = (a1 + (a2 / 2)) / 2 ∧
     a2' = ((a2 / 2) + (a1 + (a2 / 2)) / 2) / 2 →
    (a1' = a1) ∧ (a2' = a2)) :=
sorry

-- Define the conditions and questions for N = 10
theorem rubbish_equal_N10 (a : Fin 10 → ℕ) : 
    (a 0 = 1 ∧ a 1 = 2 ∧ a 2 = 4 ∧ a 3 = 8 ∧ a 4 = 16 ∧ a 5 = 32 ∧ a 6 = 64 ∧ a 7 = 128 ∧ a 8 = 256 ∧ a 9 = 512) ↔ 
    ∀ k, k < 10 →
    (let a' (i : Fin 10) := if i = 0 then
      1 + ∑ j in finset.range 9, (a (Fin.ofNat j) / 2)
    else
      a (Fin.mk d.succ.val (Nat.succ_lt_succ_iff.2 (Fin.is_lt _))) / 2
 in (∀ i, a' i = a i)) :=
sorry

end rubbish_equal_N2_rubbish_equal_N10_l810_810432


namespace supplementary_angle_difference_l810_810407

theorem supplementary_angle_difference (a b : ℝ) (h1 : a + b = 180) (h2 : 5 * b = 3 * a) : abs (a - b) = 45 :=
  sorry

end supplementary_angle_difference_l810_810407


namespace area_of_parallelogram_eq_area_of_rectangle_l810_810920

variables (base height area_parallelogram : ℕ)

theorem area_of_parallelogram_eq_area_of_rectangle
  (h_base : base = 6)
  (h_area_rectangle : base * height = 24)
  (h_parallelogram_same_dimensions : base * height = area_parallelogram) :
  area_parallelogram = 24 :=
by
  rcases h_base with rfl
  rcases h_area_rectangle with rfl
  rcases h_parallelogram_same_dimensions with rfl
  sorry

end area_of_parallelogram_eq_area_of_rectangle_l810_810920


namespace set_D_is_empty_l810_810049

theorem set_D_is_empty :
  {x : ℝ | x^2 + 2 = 0} = ∅ :=
by {
  sorry
}

end set_D_is_empty_l810_810049


namespace num_five_student_committees_l810_810122

theorem num_five_student_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 5) : choose n k = 56 :=
by
  rw [h_n, h_k]
  -- rest of the proof would go here
  sorry

end num_five_student_committees_l810_810122


namespace inequality_solution_sum_of_squares_geq_sum_of_products_l810_810004

-- Problem 1
theorem inequality_solution (x : ℝ) : (0 < x ∧ x < 2/3) ↔ (x + 2) / (2 - 3 * x) > 1 :=
by
  sorry

-- Problem 2
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end inequality_solution_sum_of_squares_geq_sum_of_products_l810_810004


namespace louis_age_currently_31_l810_810229

-- Definitions
variable (C L : ℕ)
variable (h1 : C + 6 = 30)
variable (h2 : C + L = 55)

-- Theorem statement
theorem louis_age_currently_31 : L = 31 :=
by
  sorry

end louis_age_currently_31_l810_810229


namespace sum_F_eq_8204_l810_810308

noncomputable def F (m : ℕ) := int.floor (Real.log2 m.toReal)

theorem sum_F_eq_8204 : (∑ m in Finset.range 1024, F (m + 1)) = 8204 := sorry

end sum_F_eq_8204_l810_810308


namespace average_percentage_reduction_l810_810011

theorem average_percentage_reduction 
  (initial_price final_price : ℝ) 
  (reductions : ℕ) 
  (h_initial : initial_price = 25) 
  (h_final : final_price = 16) 
  (h_reductions : reductions = 2) :
  ∃ x : ℝ, (initial_price * (1 - x) ^ reductions = final_price) ∧ x = 0.2 :=
by {
  existsi 0.2,
  split,
  {
    rw [h_initial, h_final, h_reductions],
    norm_num,
    ring,
  },
  {
    refl,
  },
}

end average_percentage_reduction_l810_810011


namespace five_student_committees_from_eight_l810_810129

theorem five_student_committees_from_eight : nat.choose 8 5 = 56 := by
  sorry

end five_student_committees_from_eight_l810_810129


namespace distinct_numbers_exist_l810_810099

theorem distinct_numbers_exist (a b c d : ℕ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : a ≠ d) (h₄ : b ≠ c) (h₅ : b ≠ d) (h₆ : c ≠ d) :
  (∃ n₁ n₂ : ℕ, n₁ * n₁ = a^2 + 2 * c * d + b^2 ∧ n₂ * n₂ = c^2 + 2 * a * b + d^2) :=
by
  use 1
  use 6
  use 2
  use 3
  sorry

end distinct_numbers_exist_l810_810099


namespace prime_factor_diff_l810_810608

noncomputable def gcd (a b : ℕ) : ℕ := sorry
noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem prime_factor_diff (k a : ℕ) (p : ℕ → ℕ)
    (hk : k ≥ 2)
    (hp_odd : ∀ i, 1 ≤ i ∧ i ≤ k → (i = 2 * n + 1 ∧ is_prime (p i) ∧ ∃ n, n ∈ ℕ))
    (hgcd1 : gcd a (p 1) = 1)
    (hgcd_rest : ∀ i, 1 ≤ i ∧ i ≤ k → gcd a (p i) = 1) :
    ∃ q, is_prime q ∧ q ∣ (a ^ (p 1 - 1) * (∏ i in finset.range k, (p i - 1)) - 1) ∧ ∀ i, 1 ≤ i ∧ i ≤ k → q ≠ (p i) := sorry

end prime_factor_diff_l810_810608


namespace find_m_of_complement_l810_810298

open Set

variable {U : Set ℕ} {m : ℝ}

theorem find_m_of_complement :
  U = {0, 1, 2, 3} →
  (A = {x ∈ U | x^2 + m * x = 0}) →
  (compl U A = {1, 2}) →
  m = -3 := by
  sorry

end find_m_of_complement_l810_810298


namespace find_selling_price_l810_810416

-- Define the cost price of the article
def cost_price : ℝ := 47

-- Define the profit when the selling price is Rs. 54
def profit : ℝ := 54 - cost_price

-- Assume that the profit is the same as the loss
axiom profit_equals_loss : profit = 7

-- Define the selling price that yields the same loss as the profit
def selling_price_loss : ℝ := cost_price - profit

-- Now state the theorem to prove that the selling price for loss is Rs. 40
theorem find_selling_price : selling_price_loss = 40 :=
sorry

end find_selling_price_l810_810416


namespace problem_statement_l810_810601

noncomputable def a : ℝ := log 5 / log 10  -- a = log_{10} 5
noncomputable def b : ℝ := log 2 / log 10  -- b = log_{10} 2

theorem problem_statement : a + b = 1 := by
  sorry

end problem_statement_l810_810601


namespace sqrt_x_div_sqrt_y_as_fraction_l810_810188

theorem sqrt_x_div_sqrt_y_as_fraction (x y : ℝ) 
  (h : ( (2 / 5)^2 + (1 / 7)^2 ) / ( (1 / 3)^2 + (1 / 8)^2 ) = 25 * x / (73 * y)) : 
  real.sqrt x / real.sqrt y = 356 / 175 :=
by sorry

end sqrt_x_div_sqrt_y_as_fraction_l810_810188


namespace greatest_value_of_x_l810_810807

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l810_810807


namespace scientific_notation_of_0_0000000028_l810_810747

theorem scientific_notation_of_0_0000000028 : (0.0000000028 : ℝ) = 2.8 * 10^(-9) :=
sorry

end scientific_notation_of_0_0000000028_l810_810747


namespace subsequence_ordered_l810_810823

theorem subsequence_ordered {a : Fin 101 → Nat} (hperm : ∀ i, a i ∈ Finset.range 102) :
  ∃ S : Finset (Fin 101), S.card = 11 ∧ (∀ i j ∈ S, i < j → a i ≤ a j) ∨ (∀ i j ∈ S, i < j → a i ≥ a j) := 
begin
  sorry
end

end subsequence_ordered_l810_810823


namespace sequence_decreasing_l810_810307

open BigOperators

noncomputable def sequence (n : ℕ) : ℚ :=
∑ k in Finset.range n, 1 / (k + 1 : ℚ) * (n - k : ℚ)

theorem sequence_decreasing {n : ℕ} (h : n ≥ 2) :
  sequence (n + 1) < sequence n :=
sorry

end sequence_decreasing_l810_810307


namespace bruce_mango_purchase_l810_810535

theorem bruce_mango_purchase (m : ℕ) 
  (cost_grapes : 8 * 70 = 560)
  (cost_total : 560 + 55 * m = 1110) : 
  m = 10 :=
by
  sorry

end bruce_mango_purchase_l810_810535


namespace jenna_hike_duration_l810_810534

-- Definitions from conditions
def initial_speed : ℝ := 25
def exhausted_speed : ℝ := 10
def total_distance : ℝ := 140
def total_time : ℝ := 8

-- The statement to prove:
theorem jenna_hike_duration : ∃ x : ℝ, 25 * x + 10 * (8 - x) = 140 ∧ x = 4 := by
  sorry

end jenna_hike_duration_l810_810534


namespace greatest_possible_x_max_possible_x_l810_810774

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l810_810774


namespace bisect_exterior_angle_l810_810259

variable {A B C D : Type}
variables [inst1: LinearOrder A] [inst2: LinearOrder B] [inst3: LinearOrder C] 
variables (AD BD CD AB AC : ℝ)

theorem bisect_exterior_angle (h: D ∈ (BC) (A, B, C) : Prop) :
  AD^2 = BD * CD - AB * AC ↔ bisects_exterior_angle_class AD A :=
sorry

end bisect_exterior_angle_l810_810259


namespace average_monthly_balance_l810_810545

theorem average_monthly_balance 
  (initial_balance : ℕ)
  (changes : List Int)
  (final_balance : ℕ) 
  (average_balance : ℕ) : 
  initial_balance = 120 →
  changes = [80, -50, 70, 0, 100] →
  final_balance = initial_balance + List.sum changes →
  average_balance = (initial_balance + 200 + 150 + 220 + 220 + 320) / 6 :=
begin
  intros,
  sorry
end

end average_monthly_balance_l810_810545


namespace no_such_matrix_exists_l810_810989

theorem no_such_matrix_exists :
  ∀ (N : Matrix (Fin 2) (Fin 2) ℝ), 
    (∀ (x y z w : ℝ), 
      N.mul (Matrix.of ![![x, y], ![z, w]]) = Matrix.of ![![2 * x, 3 * y], ![4 * z, 5 * w]]) → 
      N = 0 :=
by
  sorry

end no_such_matrix_exists_l810_810989


namespace chess_team_boys_l810_810012

-- Definitions based on the conditions
def members : ℕ := 30
def attendees : ℕ := 20

-- Variables representing boys (B) and girls (G)
variables (B G : ℕ)

-- Defining the conditions
def condition1 : Prop := B + G = members
def condition2 : Prop := (2 * G) / 3 + B = attendees

-- The problem statement: proving that B = 0
theorem chess_team_boys (h1 : condition1 B G) (h2 : condition2 B G) : B = 0 :=
  sorry

end chess_team_boys_l810_810012


namespace chef_earns_less_than_manager_by_3_dollars_l810_810064

noncomputable def wage_of_manager : ℝ := 7.50
def wage_of_dishwasher (M : ℝ) : ℝ := M / 2
def wage_of_chef (D : ℝ) : ℝ := D * 1.2

theorem chef_earns_less_than_manager_by_3_dollars :
  let M := wage_of_manager
  let D := wage_of_dishwasher M
  let C := wage_of_chef D
  M - C = 3.00 :=
by
  let M := wage_of_manager
  let D := wage_of_dishwasher M
  let C := wage_of_chef D
  sorry

end chef_earns_less_than_manager_by_3_dollars_l810_810064


namespace B_contribution_l810_810877

-- Define the given values as conditions
def A_investment: Int := 3500
def B_ratio: Int := 8
def A_profit: Int := 2
def B_profit: Int := 3
def total_months: Int := 12
def profit_ratio: Ratio Int := ⟨A_profit, B_profit⟩ 
def time_b_investment: Int := total_months - B_ratio

-- Define the proof statement
theorem B_contribution : 
  ∃ (x: Int), (A_investment * total_months) / (x * time_b_investment) = A_profit / B_profit ∧ x = 1575 :=
by
  sorry

end B_contribution_l810_810877


namespace combination_eight_choose_five_l810_810139

theorem combination_eight_choose_five : 
  ∀ (n k : ℕ), n = 8 ∧ k = 5 → Nat.choose n k = 56 :=
by 
  intros n k h
  obtain ⟨hn, hk⟩ := h
  rw [hn, hk]
  exact Nat.choose_eq 8 5
  sorry  -- This signifies that the proof needs to be filled in, but we'll skip it as per instructions.

end combination_eight_choose_five_l810_810139


namespace points_interior_of_triangles_l810_810166

theorem points_interior_of_triangles (A : Finset (ℝ × ℝ)) (n : ℕ) (h₁ : A.card = n) (h₂ : ∀ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ∈ A → p₂ ∈ A → p₃ ∈ A → p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ → ¬ collinear ℝ {p₁, p₂, p₃}) :
  ∃ (B : Finset (ℝ × ℝ)), B.card = 2 * n - 5 ∧ (∀ (p₁ p₂ p₃ : ℝ × ℝ), p₁ ∈ A → p₂ ∈ A → p₃ ∈ A → p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ → ∃ b ∈ B, b ∈ interior (convex_hull ℝ ({p₁, p₂, p₃} : Set (ℝ × ℝ)))) :=
by
  sorry

end points_interior_of_triangles_l810_810166


namespace largest_value_among_options_l810_810709

noncomputable def x : ℝ := 10 ^ -2024

theorem largest_value_among_options : (5 / x) > (3 + x) ∧ (5 / x) > (3 - x) ∧ (5 / x) > (2 * x) ∧ (5 / x) > (x / 5) :=
by
  sorry

end largest_value_among_options_l810_810709


namespace alice_has_ball_after_two_turns_l810_810524

noncomputable def probability_alice_has_ball_twice_turns : ℚ :=
  let P_AB_A : ℚ := 1/2 * 1/3
  let P_ABC_A : ℚ := 1/2 * 1/3 * 1/2
  let P_AA : ℚ := 1/2 * 1/2
  P_AB_A + P_ABC_A + P_AA

theorem alice_has_ball_after_two_turns :
  probability_alice_has_ball_twice_turns = 1/2 := 
by
  sorry

end alice_has_ball_after_two_turns_l810_810524


namespace work_completion_problem_l810_810897

theorem work_completion_problem :
  (∃ x : ℕ, 9 * (1 / 45 + 1 / x) + 23 * (1 / x) = 1) → x = 40 :=
sorry

end work_completion_problem_l810_810897


namespace find_sum_of_integers_l810_810116

theorem find_sum_of_integers (x y : ℕ) (h_diff : x - y = 8) (h_prod : x * y = 180) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : x + y = 28 :=
by
  sorry

end find_sum_of_integers_l810_810116


namespace number_of_friends_l810_810339

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l810_810339


namespace num_days_hired_l810_810843

-- Step c: Definitions from the conditions
def cost_per_hour_per_guard : ℕ := 20
def num_guards : ℕ := 2
def hours_per_day : ℕ := 8
def total_weekly_cost : ℕ := 2240
def daily_cost := cost_per_hour_per_guard * num_guards * hours_per_day

-- The theorem statement to be proved.
theorem num_days_hired (cost_per_hour_per_guard num_guards hours_per_day total_weekly_cost : ℕ) 
    (h_cost_per_guard : cost_per_hour_per_guard = 20)
    (h_num_guards : num_guards = 2)
    (h_hours_per_day : hours_per_day = 8)
    (h_total_weekly_cost : total_weekly_cost = 2240) :
    total_weekly_cost / (cost_per_hour_per_guard * num_guards * hours_per_day) = 7 :=
by
  rw [h_cost_per_guard, h_num_guards, h_hours_per_day, h_total_weekly_cost]
  sorry

end num_days_hired_l810_810843


namespace overall_ratio_men_women_l810_810434

variables (m_w_diff players_total beginners_m beginners_w intermediate_m intermediate_w advanced_m advanced_w : ℕ)

def total_men : ℕ := beginners_m + intermediate_m + advanced_m
def total_women : ℕ := beginners_w + intermediate_w + advanced_w

theorem overall_ratio_men_women 
  (h1 : beginners_m = 2) 
  (h2 : beginners_w = 4)
  (h3 : intermediate_m = 3) 
  (h4 : intermediate_w = 5) 
  (h5 : advanced_m = 1) 
  (h6 : advanced_w = 3) 
  (h7 : m_w_diff = 4)
  (h8 : total_men = 6)
  (h9 : total_women = 12)
  (h10 : players_total = 18) :
  total_men / total_women = 1 / 2 :=
by {
  sorry
}

end overall_ratio_men_women_l810_810434


namespace proof_problem_l810_810036

noncomputable def problem_data :=
  { n : ℕ := 10,
    x : Fin 10 → ℝ := λ i, [32.2, 31.1, 32.9, 35.7, 37.1, 38.0, 39.0, 43.0, 44.6, 46.0][i],
    y : Fin 10 → ℝ := λ i, [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0][i],
    sum_xi : ℝ := 379.6,
    sum_yi : ℝ := 391,
    sum_squares_x : ℝ := 246.904,
    sum_squares_y : ℝ := 568.9,
    sum_products : ℝ := sorry, -- m should be determined
    r : ℝ := 0.95 }

def correlation_is_strong (d : problem_data) : Prop :=
  d.r ≈ 0.95 → (d.r > 0.5) -- assuming a linear correlation is considered strong if r > 0.5

def empirical_regression_equation (d : problem_data) : Prop :=
  ∃ a b, b = 1.44 ∧ a = -15.56 ∧ ∀ x, d.y x = b * d.x x + a

def residual_and_slope_judgement (d : problem_data) : Prop :=
  let residual := d.y 0 - (1.44 * d.x 0 - 15.56) in
  residual = -5.81

theorem proof_problem (d : problem_data) :
  correlation_is_strong d ∧ empirical_regression_equation d ∧ residual_and_slope_judgement d :=
sorry

end proof_problem_l810_810036


namespace minimum_value_of_x_plus_2y_l810_810599

open Real

theorem minimum_value_of_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 18 := by
  sorry

end minimum_value_of_x_plus_2y_l810_810599


namespace five_student_committees_l810_810149

theorem five_student_committees (n k : ℕ) (hn : n = 8) (hk : k = 5) : 
  nat.choose n k = 56 := by
  rw [hn, hk]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end five_student_committees_l810_810149


namespace max_points_on_D_5cm_from_Q_l810_810378

-- Definitions of the given conditions
def circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  { p | dist p center = radius }

def point := ℝ × ℝ

variable (Q : point)
variable (D_center : point)
variable (D_radius : ℝ := 4)
variable (Q_outside_D : dist Q D_center > D_radius)

-- Definition of the circles
def circle_D := circle D_center D_radius
def circle_Q := circle Q 5

-- Hypothesis stating Q is outside the circle D
axiom Q_outside_circle_D : dist Q D_center > D_radius

-- The theorem statement to prove the maximum number of intersection points is 2
theorem max_points_on_D_5cm_from_Q : 
  ∃ (A B : point), A ∈ circle_Q ∧ A ∈ circle_D ∧ B ∈ circle_Q ∧ B ∈ circle_D ∧ A ≠ B :=
sorry

end max_points_on_D_5cm_from_Q_l810_810378


namespace sum_of_first_50_terms_l810_810696

noncomputable def a (n : ℕ) (d_a : ℝ) := 30 + (n - 1) * d_a
noncomputable def b (n : ℕ) (d_b : ℝ) := 60 + (n - 1) * d_b
noncomputable def sum_seq (n : ℕ) := (∑ i in finset.range n, a i (d_a := (60 / 49)) + b i (d_b := ((60 / 49))))

theorem sum_of_first_50_terms :
  a 1 + b 1 = 90 ∧
  a 50 + b 50 = 150 → 
  sum_seq 50 = 41250 :=
begin
  intro h,
  sorry -- Proof is skipped here
end

end sum_of_first_50_terms_l810_810696


namespace problem_statement_l810_810475

def a : ℝ := 60.5
def b : ℝ := 0.56
def c : ℝ := Real.log 0.56 / Real.log 10

theorem problem_statement : c < b ∧ b < a := 
by
  have h₁ : a = 60.5 := by rfl
  have h₂ : 0 < b := by norm_num
  have h₃ : b < 1 := by norm_num
  have h₄ : c < 0 := by 
    have : log 0.56 / log 10 = logb 10 0.56 := by simp [Real.logb_div_log]
    rw this
    exact logb_nonpos 0.56 one_le_log
  exact ⟨h₄, h₃⟩

end problem_statement_l810_810475


namespace problem_xy_squared_and_product_l810_810221

theorem problem_xy_squared_and_product (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 ∧ x * y = 96 :=
by
  sorry

end problem_xy_squared_and_product_l810_810221


namespace polynomial_coefficients_divisible_by_prime_l810_810310

theorem polynomial_coefficients_divisible_by_prime
  (p : ℕ) (hp : Nat.Prime p)
  (n : ℕ) 
  (f : Polynomial (Fin n → ℤ))
  (hdeg : ∀ i, (Polynomial.degree (Polynomial.c (f i))) < p)
  (hdiv : ∀ (x : Fin n → ℤ), f.eval x % p = 0) :
  ∀ i, (f.coeff i) % p = 0 := 
  sorry

end polynomial_coefficients_divisible_by_prime_l810_810310


namespace hyperbola_eccentricity_is_two_l810_810404

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : ℝ :=
  let c := real.sqrt (a ^ 2 + b ^ 2) in
  c / a

theorem hyperbola_eccentricity_is_two (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : hyperbola_eccentricity a b h_a h_b = 2 :=
  sorry

end hyperbola_eccentricity_is_two_l810_810404


namespace AB_equals_DE_l810_810290

-- Define points and conditions
variables (A B C D E : Point)
variables [collinear : Collinear A B C]
variables (Gamma1 Gamma2 Gamma3 Gamma : Circle)
variables [tangent_Gamma1 : Tangent Gamma1 Gamma]
variables [tangent_Gamma3 : Tangent Gamma3 Gamma]
variables (l : Line)
variables [perpendicular_l_AC : Perpendicular l (Line_through A C)]
variables (tangent_l_Gamma : TangentLine l Gamma)
variables (D_tangent_Gamma3 : Point_of_Tangency D Gamma Gamma3)
variables (E_on_line : OnLine E l)
variables [diameter_through_D : DiameterThrough D E Gamma]

-- Prove statement AB = DE
theorem AB_equals_DE : Distance A B = Distance D E := sorry

end AB_equals_DE_l810_810290


namespace survey_problem_l810_810904

-- Definitions
def P_B (X : ℝ) : ℝ := X - 20
def P_A_and_B : ℝ := 23
def P_neither : ℝ := 23
def P_A_or_B : ℝ := 100 - P_neither

-- Theorem statement
theorem survey_problem (X : ℝ) : P_A_or_B = 77 → P_B X = X - 20 → P_A_and_B = 23 → P_neither = 23 → P_A X = 120 - X :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end survey_problem_l810_810904


namespace remaining_days_to_complete_work_l810_810681

noncomputable def kiran_time_for_total_work : ℕ := 18 -- Kiran's total time to complete the work alone (days)
noncomputable def kiran_work_rate : ℝ := 1 / 18 -- Kiran's work rate (work per day)
noncomputable def rahul_work_rate : ℝ := 2 * kiran_work_rate -- Rahul's work rate (work per day)
noncomputable def combined_work_rate : ℝ := kiran_work_rate + rahul_work_rate -- Combined work rate
noncomputable def remaining_work : ℝ := 2 / 3 -- Remaining work fraction

theorem remaining_days_to_complete_work :
  let remaining_days := remaining_work / combined_work_rate in
  remaining_days = 4 :=
by
  sorry

end remaining_days_to_complete_work_l810_810681


namespace part1_part2_l810_810628

variable (x y z : ℕ)

theorem part1 (h1 : 3 * x + 5 * y = 98) (h2 : 8 * x + 3 * y = 158) : x = 16 ∧ y = 10 :=
sorry

theorem part2 (hx : x = 16) (hy : y = 10) (hz : 16 * z + 10 * (40 - z) ≤ 550) : z ≤ 25 :=
sorry

end part1_part2_l810_810628


namespace number_of_friends_l810_810332

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l810_810332


namespace area_of_square_l810_810512

-- Define the diagonal length condition.
def diagonal_length : ℝ := 12 * real.sqrt 2

-- Define the side length of the square computed from the diagonal using the 45-45-90 triangle property.
def side_length : ℝ := diagonal_length / real.sqrt 2

-- Define the area of the square in terms of its side length.
def square_area : ℝ := side_length * side_length

-- Prove that the area is indeed 144 square centimeters.
theorem area_of_square (d : ℝ) (h : d = 12 * real.sqrt 2) : (d / real.sqrt 2) * (d / real.sqrt 2) = 144 :=
by
  rw [h, ←real.mul_div_cancel (12 * real.sqrt 2) (real.sqrt 2)],
  { norm_num },
  { exact real.sqrt_ne_zero'.2 (by norm_num) }

end area_of_square_l810_810512


namespace louis_current_age_l810_810232

/-- 
  In 6 years, Carla will be 30 years old. 
  The sum of the current ages of Carla and Louis is 55. 
  Prove that Louis is currently 31 years old.
--/
theorem louis_current_age (C L : ℕ) 
  (h1 : C + 6 = 30) 
  (h2 : C + L = 55) 
  : L = 31 := 
sorry

end louis_current_age_l810_810232


namespace find_circle_equation_l810_810598

noncomputable def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

theorem find_circle_equation (A B : ℝ × ℝ) (f : ℝ × ℝ → Prop)
  (hA : A = (1, 1))
  (hB : B = (2, 0))
  (h_intercept : ∀ P Q : ℝ × ℝ, P ∈ C → Q ∈ C → x - y - 2 = 0 → (dist P Q = sqrt 2)) :
  ∃ x y : ℝ, circle_equation x y :=
sorry

end find_circle_equation_l810_810598


namespace estimation_problems_l810_810937

noncomputable def average_root_cross_sectional_area (x : list ℝ) : ℝ :=
  (list.sum x) / (list.length x)

noncomputable def average_volume (y : list ℝ) : ℝ :=
  (list.sum y) / (list.length y)

noncomputable def sample_correlation_coefficient (x y : list ℝ) : ℝ :=
  let n := list.length x
      avg_x := average_root_cross_sectional_area x
      avg_y := average_volume y
      sum_xy := (list.zip_with (*) x y).sum
      sum_x2 := (x.map (λ xi, xi * xi)).sum
      sum_y2 := (y.map (λ yi, yi * yi)).sum
  in (sum_xy - n * avg_x * avg_y) / (real.sqrt ((sum_x2 - n * avg_x^2) * (sum_y2 - n * avg_y^2)))

theorem estimation_problems :
  let x := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
      y := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
      X := 186
  in
    average_root_cross_sectional_area x = 0.06 ∧
    average_volume y = 0.39 ∧
    abs (sample_correlation_coefficient x y - 0.97) < 0.01 ∧
    (average_volume y / average_root_cross_sectional_area x) * X = 1209 :=
by
  sorry

end estimation_problems_l810_810937


namespace number_of_friends_l810_810329

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l810_810329


namespace f_range_f_at_B_l810_810206

-- Definitions of vectors and the function f(x)
def m (x : ℝ) : ℝ × ℝ := (sqrt 3 * real.sin x, 1 - sqrt 2 * real.sin x)
def n (x : ℝ) : ℝ × ℝ := (2 * real.cos x, 1 + sqrt 2 * real.sin x)
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Problem 1: Range of f(x) for x ∈ [0, π/2]
theorem f_range : ∀ x ∈ set.Icc (0 : ℝ) (real.pi / 2), f x ∈ set.Icc (-1) 2 := by
  sorry

-- Triangle setup for Problem 2
variables {a b c : ℝ} {A B C : ℝ}

-- Given conditions for Problem 2
axiom ratio_ba : b / a = sqrt 3
axiom sinB_cosA : real.sin B * real.cos A / real.sin A = 2 - real.cos B

-- Problem 2: Evaluate f at B
theorem f_at_B : f B = 1 := by
  sorry

end f_range_f_at_B_l810_810206


namespace math_problem_l810_810695

variables {a b : ℝ}
hypothesis h1 : a ≠ b
hypothesis h2 : a > 0
hypothesis h3 : b > 0
hypothesis h4 : b * (log a) - a * (log b) = a - b

-- Prove the following propositions
theorem math_problem :
  (a + b - a * b > 1) ∧ 
  (a + b > 2) ∧ 
  (1 / a + 1 / b > 2) :=
sorry

end math_problem_l810_810695


namespace x_is_integer_if_conditions_hold_l810_810212

theorem x_is_integer_if_conditions_hold (x : ℝ)
  (h1 : ∃ (k : ℤ), x^2 - x = k)
  (h2 : ∃ (n : ℕ), n ≥ 3 ∧ ∃ (m : ℤ), x^n - x = m) :
  ∃ (z : ℤ), x = z :=
sorry

end x_is_integer_if_conditions_hold_l810_810212


namespace tan_α_eq_neg_4_over_3_l810_810151

-- Definitions based on the conditions from part (a)
def α : ℝ := sorry  -- Angle α in radians
axiom cos_α_eq_neg_3_over_5 : cos α = - (3 / 5)
axiom α_in_interval : 0 < α ∧ α < π

-- The target statement to prove
theorem tan_α_eq_neg_4_over_3 (h1 : cos α = - (3 / 5)) (h2 : 0 < α ∧ α < π) : tan α = - (4 / 3) :=
sorry

end tan_α_eq_neg_4_over_3_l810_810151


namespace b4_lt_b7_l810_810040

noncomputable theory
open_locale big_operators

def α (k: ℕ) := k + 1 -- This makes α_{k+1} in ℕ^*
def b : ℕ → ℚ
| 0 := 1 + 1 / α 0
| (n + 1) := 1 + 1 / (α 0 + ∑ i in finset.range (n + 1), 1 / b i)

theorem b4_lt_b7: b 3 < b 6 := 
sorry

end b4_lt_b7_l810_810040


namespace jack_buttons_total_l810_810274

theorem jack_buttons_total :
  (3 * 3) * 7 = 63 :=
by
  sorry

end jack_buttons_total_l810_810274


namespace find_angle_l810_810100

theorem find_angle (x : ℝ) (h : sin (4 * x) * sin (6 * x) = cos (4 * x) * cos (6 * x)) : x = real.pi / 10 := sorry

end find_angle_l810_810100


namespace expand_product_l810_810558

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 :=
by
  sorry

end expand_product_l810_810558


namespace PA2_PB2_PC2_minus_PG2_constant_l810_810293

variables {A B C P : EuclideanGeometry.Point}
variables [circumcircle_ABC : EuclideanGeometry.on_circumcircle P A B C]
variables {G : EuclideanGeometry.Point} [centroid_G : EuclideanGeometry.is_centroid G A B C]
variables {a b c R : ℝ}  -- side lengths and circumradius

theorem PA2_PB2_PC2_minus_PG2_constant
  (hA : EuclideanGeometry.distance P A = R)
  (hB : EuclideanGeometry.distance P B = R)
  (hC : EuclideanGeometry.distance P C = R)
  : EuclideanGeometry.distance P A ^ 2 + EuclideanGeometry.distance P B ^ 2 + EuclideanGeometry.distance P C ^ 2 
    - EuclideanGeometry.distance P G ^ 2 = (1 / 2) * (a^2 + b^2 + c^2) + 3 * R^2 :=
sorry

end PA2_PB2_PC2_minus_PG2_constant_l810_810293


namespace probability_one_miss_in_three_attempts_l810_810462

theorem probability_one_miss_in_three_attempts (success_rate : ℝ) (h : success_rate = 0.9) :
  let miss_rate := 1 - success_rate in
  let scenario_1 := miss_rate * success_rate * success_rate in
  let scenario_2 := success_rate * miss_rate * success_rate in
  let scenario_3 := success_rate * success_rate * miss_rate in
  (scenario_1 + scenario_2 + scenario_3) = 0.243 :=
by
  sorry

end probability_one_miss_in_three_attempts_l810_810462


namespace problem_statement_l810_810033

open Real

noncomputable def correlation_coefficient (x y : List ℝ) : ℝ :=
  let n := x.length
  let x̄ := List.sum x / n.toReal
  let ȳ := List.sum y / n.toReal
  let numerator := List.sum (List.zipWith (λ xi yi, (xi - x̄) * (yi - ȳ)) x y)
  let denominator := sqrt (List.sum (List.map (λ xi, (xi - x̄)^2) x) * List.sum (List.map (λ yi, (yi - ȳ)^2) y))
  numerator / denominator

def residual_correct (x y : List ℝ) (predicted_y : ℝ) (actual_y : ℝ) : ℝ :=
  actual_y - predicted_y

theorem problem_statement :
  -- Given conditions
  let x := [32.2, 31.1, 32.9, 35.7, 37.1, 38.0, 39.0, 43.0, 44.6, 46.0]
  let y := [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0]
  let n := 10
  let sum_x := 379.6
  let sum_y := 391
  let sum_x_squared := 246.904
  let sum_y_squared := 568.9
  let covariance_xy := m
  let mean_x := 37.96
  let mean_y := 39.1
  let r := correlation_coefficient x y
  -- Proving the statements
  r ≈ 0.95 →
  ∃ a b : ℝ, 
    b = 1.44 ∧
    a = -15.56 ∧
    ∀ xi yi, 
    (xi, yi) ∈ List.zip x y →
    residual_correct xi yi (b * xi + a) ≈ -5.81 :=
by sorry

end problem_statement_l810_810033


namespace min_cost_max_profit_l810_810486

-- Definitions of the given functions and parameters
def P (x : ℝ) : ℝ := 12500 / x + 40 + 0.05 * x
def Q (x : ℝ) : ℝ := 170 - 0.05 * x
def profit (x : ℝ) : ℝ := x * Q(x) - x * P(x)

-- Stating the first proof problem
theorem min_cost : P 500 = 90 :=
by
  sorry

-- Stating the second proof problem
theorem max_profit : profit 650 = 29750 :=
by
  have h1 : profit (x : ℝ) = -0.1 * x^2 + 130 * x - 12500 := by
    sorry
  sorry

end min_cost_max_profit_l810_810486


namespace possible_theta_l810_810613

-- Define the function f(x)
def f (x θ : ℝ) : ℝ := sin (x + θ) + cos (x + θ)

-- Define the condition that f(x) = f(-x) for all x in ℝ
def f_is_even (θ : ℝ) : Prop :=
  ∀ x : ℝ, f x θ = f (-x) θ

-- State the main theorem in Lean 4
theorem possible_theta (θ : ℝ) : f_is_even θ → θ = π / 4 := by
  sorry

end possible_theta_l810_810613


namespace birthday_friends_count_l810_810351

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l810_810351


namespace hancho_milk_l810_810323

def initial_milk : ℝ := 1
def ye_seul_milk : ℝ := 0.1
def ga_young_milk : ℝ := ye_seul_milk + 0.2
def remaining_milk : ℝ := 0.3

theorem hancho_milk : (initial_milk - (ye_seul_milk + ga_young_milk + remaining_milk)) = 0.3 :=
by
  sorry

end hancho_milk_l810_810323


namespace gumball_problem_l810_810629

theorem gumball_problem:
  ∀ (total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs: ℕ),
    total_gumballs = 45 →
    given_to_Todd = 4 →
    given_to_Alisha = 2 * given_to_Todd →
    remaining_gumballs = 6 →
    given_to_Todd + given_to_Alisha + given_to_Bobby + remaining_gumballs = total_gumballs →
    given_to_Bobby = 45 - 18 →
    4 * given_to_Alisha - given_to_Bobby = 5 :=
by
  intros total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs ht hTodd hAlisha hRemaining hSum hBobby
  rw [ht, hTodd] at *
  rw [hAlisha, hRemaining] at *
  sorry

end gumball_problem_l810_810629


namespace tan_zero_l810_810962

theorem tan_zero : Real.tan 0 = 0 := 
by
  sorry

end tan_zero_l810_810962


namespace H_intersects_plane_at_most_five_points_l810_810548

/-
Problem Statement:
Define a set of points in space that has at least one but no more than five points in any plane.
-/
noncomputable def H_points (x : ℝ) : ℝ × ℝ := (x^3, x^5)

theorem H_intersects_plane_at_most_five_points (A B C D : ℝ) (h : ¬(A = 0 ∧ B = 0 ∧ C = 0)) :
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
  (A * x₁ + B * (x₁ ^ 3) + C * (x₁ ^ 5) + D = 0) ∧
  (A * x₂ + B * (x₂ ^ 3) + C * (x₂ ^ 5) + D = 0) ∧
  (A * x₃ + B * (x₃ ^ 3) + C * (x₃ ^ 5) + D = 0) ∧
  (A * x₄ + B * (x₄ ^ 3) + C * (x₄ ^ 5) + D = 0) ∧
  (A * x₅ + B * (x₅ ^ 3) + C * (x₅ ^ 5) + D = 0) ∧
  ∀ x, (A * x + B * (x ^ 3) + C * (x ^ 5) + D = 0) → (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ ∨ x = x₅).
Proof := sorry

end H_intersects_plane_at_most_five_points_l810_810548


namespace monthly_expenses_last_month_was_2888_l810_810743

def basic_salary : ℕ := 1250
def commission_rate : ℚ := 0.10
def total_sales : ℕ := 23600
def savings_rate : ℚ := 0.20

theorem monthly_expenses_last_month_was_2888 :
  let commission := commission_rate * total_sales
  let total_earnings := basic_salary + commission
  let savings := savings_rate * total_earnings
  let monthly_expenses := total_earnings - savings
  monthly_expenses = 2888 := by
  sorry

end monthly_expenses_last_month_was_2888_l810_810743


namespace five_student_committees_l810_810145

theorem five_student_committees (n k : ℕ) (hn : n = 8) (hk : k = 5) : 
  nat.choose n k = 56 := by
  rw [hn, hk]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end five_student_committees_l810_810145


namespace birthday_friends_count_l810_810371

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l810_810371


namespace S4_equals_30_l810_810584

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ n : ℕ, a (n+1) = a n * q

def sum_geometric_sequence (a : ℕ → ℝ) (n : ℕ) :=
a 1 * (1 - q^(n + 1)) / (1 - q)

theorem S4_equals_30
  (h1 : is_geometric_sequence a q)
  (h2 : a 2 * a 3 = 2 * a 1)
  (h3 : (a 4 + 2 * a 7) / 2 = 5 / 4)
  : sum_geometric_sequence a 4 = 30 :=
sorry

end S4_equals_30_l810_810584


namespace domino_covering_l810_810249

theorem domino_covering (m n : ℕ) (h_m : 2 ≤ m) (h_n : 2 ≤ n)
  (dominoes : set (set (ℕ × ℕ))) (no_overlap : ∀ d1 d2 ∈ dominoes, d1 ∩ d2 = ∅)
  (covers_two : ∀ d ∈ dominoes, ∃ (x1 y1 x2 y2 : ℕ), d = {(x1, y1), (x2, y2)} ∧ (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 + 1 = y2) ∨ y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 + 1 = x2)))
  (max_coverage : ∀ (extra_domino : set (ℕ × ℕ)), ¬(extra_domino ∩ doms = ∅ ∧ ∃ x y x' y', extra_domino = {(x, y), (x', y')}). :=
  sorry): (Σ^2 / (m * n)) := sorry

end domino_covering_l810_810249


namespace scalene_triangles_count_l810_810992

theorem scalene_triangles_count :
  ∃ n : ℕ, n = 23 ∧ ∀ (a b c : ℕ), a < b → b < c → a + b + c < 20 → a + b > c → a + c > b → b + c > a → sorry

end scalene_triangles_count_l810_810992


namespace range_of_f_l810_810203

open Real

theorem range_of_f {x : ℝ} :
  let m := (sin x, cos x),
      n := (sqrt 3, -1)
  in (m.1 * n.1 + m.2 * n.2 = 1) →
     let f := cos (2 * x) + 4 * cos (sin x)
     -- Question: proving the range
     in -3 ≤ f ∧ f ≤ 3 / 2 :=
by
  intros m n h f
  sorry

end range_of_f_l810_810203


namespace B_work_days_proof_l810_810482

-- Define the main variables
variables (W : ℝ) (x : ℝ) (daysA : ℝ) (daysBworked : ℝ) (daysAremaining : ℝ)

-- Given conditions from the problem
def A_work_days : ℝ := 6
def B_work_days : ℝ := x
def B_worked_days : ℝ := 10
def A_remaining_days : ℝ := 2

-- We are asked to prove this statement
theorem B_work_days_proof (h1 : daysA = A_work_days)
                           (h2 : daysBworked = B_worked_days)
                           (h3 : daysAremaining = A_remaining_days) 
                           (hx : (W/6 = (W - 10*W/x) / 2)) : x = 15 :=
by 
  -- Proof omitted
  sorry 

end B_work_days_proof_l810_810482


namespace intersection_complement_example_l810_810313

open Set

theorem intersection_complement_example :
  let P := {1, 2, 3, 4, 5}
  let Q := {4, 5, 6}
  let U := univ : Set ℝ
  P ∩ (U \ Q) = {1, 2, 3} :=
by
  let P := {1, 2, 3, 4, 5}
  let Q := {4, 5, 6}
  let U := univ : Set ℝ
  show P ∩ (U \ Q) = {1, 2, 3}
  sorry

end intersection_complement_example_l810_810313


namespace num_permutations_DOG_l810_810209

theorem num_permutations_DOG : fintype.card (perm (fin 3)) = 6 :=
by
  -- We acknowledge that fin 3 corresponds to a three-element set,
  -- and perm (fin 3) signifies permuting those three elements.
  sorry

end num_permutations_DOG_l810_810209


namespace number_of_birdhouses_l810_810070

-- Definitions for the conditions
def cost_per_nail : ℝ := 0.05
def cost_per_plank : ℝ := 3.0
def planks_per_birdhouse : ℕ := 7
def nails_per_birdhouse : ℕ := 20
def total_cost : ℝ := 88.0

-- Total cost calculation per birdhouse
def cost_per_birdhouse := planks_per_birdhouse * cost_per_plank + nails_per_birdhouse * cost_per_nail

-- Proving that the number of birdhouses is 4
theorem number_of_birdhouses : total_cost / cost_per_birdhouse = 4 := by
  sorry

end number_of_birdhouses_l810_810070


namespace john_has_22_dimes_l810_810830

theorem john_has_22_dimes (d q : ℕ) (h1 : d = q + 4) (h2 : 10 * d + 25 * q = 680) : d = 22 :=
by
sorry

end john_has_22_dimes_l810_810830


namespace stats_not_increase_l810_810068

def game_scores_before : List ℝ := [48, 55, 55, 60, 62]
def game_score_6th : ℝ := 53

def median (l : List ℝ) : ℝ :=
  (l.sorted.nth (l.length / 2) + l.sorted.nth ((l.length - 1) / 2)) / 2.0

def mode (l : List ℝ) : ℝ :=
  l.mode

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem stats_not_increase :
  median (game_scores_before ++ [game_score_6th]) ≤ median game_scores_before ∧
  mode (game_scores_before ++ [game_score_6th]) ≤ mode game_scores_before ∧
  mean (game_scores_before ++ [game_score_6th]) ≤ mean game_scores_before :=
by
  sorry

end stats_not_increase_l810_810068


namespace square_area_from_diagonal_l810_810507

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) : ∃ A : ℝ, A = 144 :=
by
  let s := d / Real.sqrt 2
  have s_eq : s = 12 := by
    rw [h]
    field_simp
    norm_num
  use s * s
  rw [s_eq]
  norm_num
  sorry

end square_area_from_diagonal_l810_810507


namespace linear_correlation_is_very_strong_regression_line_residual_first_sample_effect_of_exclusion_l810_810030

noncomputable def resident_income : List ℝ := [32.2, 31.1, 32.9,  35.7,  37.1,  38.0,  39.0,  43.0,  44.6,  46.0]

noncomputable def goods_sales : List ℝ := [25.0, 30.0, 34.0, 37.0, 39.0, 41.0, 42.0, 44.0, 48.0, 51.0]

noncomputable def sum_x : ℝ := 379.6
noncomputable def sum_y : ℝ := 391
noncomputable def sum_x_squared_dev : ℝ := 246.904
noncomputable def sum_y_squared_dev : ℝ := 568.9
noncomputable def sum_xy_dev : ℝ := r * math.sqrt(sum_y_squared_dev * sum_x_squared_dev) where r := 0.95

-- Prove the linear correlation
theorem linear_correlation_is_very_strong : sum_xy_dev / (math.sqrt(sum_x_squared_dev) * math.sqrt(sum_y_squared_dev)) ≈ 0.95 → 
  (0.95 ≈ 1 ∨ 0.95 ≈ -1) :=
sorry

-- Calculate the regression line
theorem regression_line : sum_xy_dev / sum_x_squared_dev ≈ 1.44 ∧ 39.1 - 1.44 * 37.96 ≈ -15.56 → 
  (∀ x, y = 1.44 * x - 15.56) :=
sorry

-- Calculating the residual
theorem residual_first_sample : y - (1.44 * 32.2 - 15.56) ≈ -5.81 :=
sorry

-- Effect on regression line when excluding the sample point
theorem effect_of_exclusion (residual_lt_regression : y < 1.44 * x - 15.56):
  sum_xy_dev * math.sqrt(sum_x_squared_dev) * math.sqrt(sum_y_squared_dev) < sum_xy_dev * math.sqrt(sum_x_squared_dev) :=
sorry

end linear_correlation_is_very_strong_regression_line_residual_first_sample_effect_of_exclusion_l810_810030


namespace find_c_l810_810761

-- Define points and the line equation.
def point_A := (1, 3)
def point_B := (5, 11)
def midpoint (A B : ℚ × ℚ) : ℚ × ℚ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- The line equation 2x - y = c
def line_eq (x y c : ℚ) : Prop :=
  2 * x - y = c

-- Define the proof problem
theorem find_c : 
  let M := midpoint point_A point_B in
  line_eq M.1 M.2 (-1) :=
by
  sorry

end find_c_l810_810761


namespace green_balls_in_bag_l810_810481

theorem green_balls_in_bag (b : ℕ) (P_blue : ℚ) (g : ℕ) (h1 : b = 8) (h2 : P_blue = 1 / 3) (h3 : P_blue = (b : ℚ) / (b + g)) :
  g = 16 :=
by
  sorry

end green_balls_in_bag_l810_810481


namespace correct_N_value_l810_810471

theorem correct_N_value :
  ∀ (N : ℕ), N > 1 →
  (∀ d : list ℕ, (∀ i, i < d.length → d[i + 1] = N / d[i]) →
                 d.head = 1 ∧ d.last = N ∧ (∀ i < d.length - 1, (d[i], d[i+1]) ∈ d) →
                 list.sum (list.map (λ i, Nat.gcd i (i+1)) (list.finRange (d.length - 1))) = N - 2) →
  N = 3 :=
by
  intros N hN d hDivisors h1 hN hPairs hSum
  sorry

end correct_N_value_l810_810471


namespace average_score_of_class_l810_810527

theorem average_score_of_class (n : ℕ) (k : ℕ) (jimin_score : ℕ) (jungkook_score : ℕ) (avg_others : ℕ) 
  (total_students : n = 40) (excluding_students : k = 38) 
  (avg_excluding_others : avg_others = 79) 
  (jimin : jimin_score = 98) 
  (jungkook : jungkook_score = 100) : 
  (98 + 100 + (38 * 79)) / 40 = 80 :=
sorry

end average_score_of_class_l810_810527


namespace interval_of_decrease_log_a_x_square_minus_2x_minus_3_l810_810189

theorem interval_of_decrease_log_a_x_square_minus_2x_minus_3 
  (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) 
  (h_monotonic: ∀ x, (0 < x) ∧ (x < 1) → log a (2^x - 1) < 0) :
  (∃ I : set ℝ, I = set.Iio (-1) ∧ ∀ x ∈ I, 
   ∀ y ∈ I, x ≤ y → (log a (x^2 - 2*x - 3) ≥ log a (y^2 - 2*y - 3))) :=
by
  sorry

end interval_of_decrease_log_a_x_square_minus_2x_minus_3_l810_810189


namespace sum_of_solutions_l810_810109

theorem sum_of_solutions (S : Set ℝ) (h : ∀ y ∈ S, y + 16 / y = 12) :
  ∃ t : ℝ, (∀ y ∈ S, y = 8 ∨ y = 4) ∧ t = 12 := by
  sorry

end sum_of_solutions_l810_810109


namespace shaded_figure_area_l810_810985

noncomputable def area_of_shaded_figure (R : ℝ) (α : ℝ) (angle : α = 30 * π / 180) : ℝ :=
  (π * R^2) / 3

theorem shaded_figure_area (R : ℝ) (hα : α = 30 * π / 180) :
  area_of_shaded_figure R 30 hα = (π * R^2) / 3 :=
by
  sorry

end shaded_figure_area_l810_810985


namespace angle_EMD_eq_DMF_l810_810246

def triangle := Type

structure Triangle (α : Type) :=
(A B C : α)

variables {α : Type} [EuclideanGeometry α]

def is_angle_bisector (A B C D : α) : Prop :=
  sorry -- definition of angle bisector

def perpendicular_foot (P Q R : α) : α := sorry -- definition of the foot of perpendicular

def orthogonal (P Q R : α) : Prop :=
  sorry -- definition of orthogonal

noncomputable def D_point (A C : α) : α := sorry -- definition of point D on AC

noncomputable def M_point (B C : α) : α := sorry -- definition of point M on BC

theorem angle_EMD_eq_DMF
  (t : Triangle α)
  (h_diff_lengths : t.A ≠ t.B ∧ t.B ≠ t.C ∧ t.A ≠ t.C)
  (D : α) (hD_on_AC : D = D_point t.A t.C)
  (hD_bisector : is_angle_bisector t.B t.A t.C D)
  (E F : α)
  (hE_perp : E = perpendicular_foot t.A D)
  (hF_perp : F = perpendicular_foot t.C D)
  (M : α) 
  (hM_on_BC : M = M_point t.B t.C)
  (hM_perp : orthogonal D M t.B):
  ∠EMD = ∠DMF :=
sorry

end angle_EMD_eq_DMF_l810_810246


namespace find_abs_z3_l810_810663

variables (z1 z2 z3 : ℂ)
variable (λ : ℂ)
variable (h1 : λ.re < 0)
variable (h2 : z1 = λ * z2)
variable (h3 : abs (z1 - z2) = 13)
variable (h4 : abs (z1)^2 + abs (z3)^2 + abs (z1 * z3) = 144)
variable (h5 : abs (z2)^2 + abs (z3)^2 - abs (z2 * z3) = 25)

theorem find_abs_z3 :
  abs z3 = 40 * √3 / 13 :=
sorry

end find_abs_z3_l810_810663


namespace complex_expression_proof_l810_810311

open Complex

theorem complex_expression_proof {x y z : ℂ}
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 15)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 18 :=
by
  sorry

end complex_expression_proof_l810_810311


namespace greatest_x_lcm_l810_810819

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l810_810819


namespace cot_B_minus_cot_C_is_2_l810_810670

variables {A B C D : Type}
variables [Triangle A B C]

def median_AD_is_30_degrees (A B C : Point) (D : Point) : Prop :=
  ∃ (θ : Real), θ = 30 ∧ median AD = θ

def length_BC_is_a (B C : Point) (a : Real) : Prop :=
  distance B C = a

theorem cot_B_minus_cot_C_is_2 (A B C : Point) (a : Real) (D : Point) :
  median_AD_is_30_degrees A B C D → length_BC_is_a B C a → |cot B - cot C| = 2 :=
by
  intros h1 h2
  sorry

end cot_B_minus_cot_C_is_2_l810_810670


namespace jack_buttons_l810_810275

theorem jack_buttons :
  ∀ (shirts_per_kid kids buttons_per_shirt : ℕ),
  shirts_per_kid = 3 →
  kids = 3 →
  buttons_per_shirt = 7 →
  (shirts_per_kid * kids * buttons_per_shirt) = 63 :=
by
  intros shirts_per_kid kids buttons_per_shirt h1 h2 h3
  rw [h1, h2, h3]
  calc
    3 * 3 * 7 = 9 * 7 : by rw mul_assoc
            ... = 63   : by norm_num

end jack_buttons_l810_810275


namespace polynomial_expansion_l810_810557

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 + 4 * x - 7) * (2 * x^4 - 3 * x^2 + 5) =
  6 * x^7 + 12 * x^5 - 9 * x^4 - 21 * x^3 - 11 * x + 35 :=
by
  sorry

end polynomial_expansion_l810_810557


namespace find_dividend_l810_810986

theorem find_dividend (divisor : ℕ) (partial_quotient : ℕ) (dividend : ℕ) 
                       (h_divisor : divisor = 12)
                       (h_partial_quotient : partial_quotient = 909809) 
                       (h_calculation : dividend = divisor * partial_quotient) : 
                       dividend = 10917708 :=
by
  rw [h_divisor, h_partial_quotient] at h_calculation
  exact h_calculation


end find_dividend_l810_810986


namespace area_of_square_l810_810513

-- Define the diagonal length condition.
def diagonal_length : ℝ := 12 * real.sqrt 2

-- Define the side length of the square computed from the diagonal using the 45-45-90 triangle property.
def side_length : ℝ := diagonal_length / real.sqrt 2

-- Define the area of the square in terms of its side length.
def square_area : ℝ := side_length * side_length

-- Prove that the area is indeed 144 square centimeters.
theorem area_of_square (d : ℝ) (h : d = 12 * real.sqrt 2) : (d / real.sqrt 2) * (d / real.sqrt 2) = 144 :=
by
  rw [h, ←real.mul_div_cancel (12 * real.sqrt 2) (real.sqrt 2)],
  { norm_num },
  { exact real.sqrt_ne_zero'.2 (by norm_num) }

end area_of_square_l810_810513


namespace m_gt_n_l810_810226

variable (m n : ℝ)

-- Definition of points A and B lying on the line y = -2x + 1
def point_A_on_line : Prop := m = -2 * (-1) + 1
def point_B_on_line : Prop := n = -2 * 3 + 1

-- Theorem stating that m > n given the conditions
theorem m_gt_n (hA : point_A_on_line m) (hB : point_B_on_line n) : m > n :=
by
  -- To avoid the proof part, which we skip as per instructions
  sorry

end m_gt_n_l810_810226


namespace least_value_expression_l810_810867

theorem least_value_expression (x y : ℝ) : ∃ (x y : ℝ), (xy + 1)^2 + (x + y)^2 = 1 ∧ ∀ (x y : ℝ), (xy + 1)^2 + (x + y)^2 ≥ 1 := 
begin
  sorry
end

end least_value_expression_l810_810867


namespace scientific_notation_of_solubility_product_l810_810745

theorem scientific_notation_of_solubility_product :
  (∃ (K_sp : ℝ), K_sp = 0.0000000028) → 0.0000000028 = 2.8 * 10^(-9) :=
begin
  intro exists_K_sp,
  cases exists_K_sp with K_sp K_sp_eq,
  rw K_sp_eq,
  sorry
end

end scientific_notation_of_solubility_product_l810_810745


namespace find_m_l810_810706

noncomputable def z1 (m : ℝ) : ℂ := m^2 - 3*m + complex.I * m^2
noncomputable def z2 (m : ℝ) : ℂ := 4 + complex.I * (5*m + 6)

theorem find_m (m : ℝ) (h : z1 m - z2 m = 0) : m = -1 :=
sorry

end find_m_l810_810706


namespace sum_ka_k_le_one_l810_810445

theorem sum_ka_k_le_one (n : ℕ) (a : ℕ → ℝ) (h : ∀ x : ℝ, |∑ k in Finset.range (n + 1), a k * Real.sin (k * x)| ≤ |Real.sin x|):
  |∑ k in Finset.range (n + 1), k * a k| ≤ 1 := 
sorry

end sum_ka_k_le_one_l810_810445


namespace problem_number_selection_probability_l810_810411

theorem problem_number_selection_probability :
  let p := (10 * ((5^4) - (4^4))) / (10^4) in
  10000 * p = 3690 :=
by
  sorry

end problem_number_selection_probability_l810_810411


namespace showUpPeopleFirstDay_l810_810958

def cansFood := 2000
def people1stDay (cansTaken_1stDay : ℕ) := cansFood - 1500 = cansTaken_1stDay
def peopleSnapped_1stDay := 500

theorem showUpPeopleFirstDay :
  (people1stDay peopleSnapped_1stDay) → (peopleSnapped_1stDay / 1) = 500 := 
by 
  sorry

end showUpPeopleFirstDay_l810_810958


namespace birthday_friends_count_l810_810348

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l810_810348


namespace composite_shape_to_square_l810_810081

-- Define the initial composite geometric shape
def initial_figure : GeometricShape := composite_shape

-- Define the concept that the composite shape can be divided into parts
def can_be_divided (shape: GeometricShape) (parts: ℕ) : Prop :=
  -- Here we say shape can be divided into "parts" number of parts
  ∃ (ps: list GeometricShape), ps.length = parts ∧ combined_area ps = area shape

-- Define the concept of fit together to form a square
def form_square (pieces: list GeometricShape) : Prop :=
  -- Pieces can be rearranged to form a square
  ∃ (square: GeometricShape), is_square square ∧ combined_area pieces = area square

-- Noncomputable due to geometric nature (exact cuts and rearrangements might not be constructive)
noncomputable def can_form_square_from_initial (parts_count: ℕ) : Prop :=
  parts_count = 4 ∧ can_be_divided initial_figure parts_count ∧ form_square (divide initial_figure parts_count)

-- The theorem stating the main problem
theorem composite_shape_to_square: can_form_square_from_initial 4 :=
  sorry

end composite_shape_to_square_l810_810081


namespace sum_integer_solutions_l810_810084

theorem sum_integer_solutions (n : ℤ) (h1 : |n^2| < |n - 5|^2) (h2 : |n - 5|^2 < 16) : n = 2 := 
sorry

end sum_integer_solutions_l810_810084


namespace math_problem_proof_l810_810956

theorem math_problem_proof : 
  ((9 - 8 + 7) ^ 2 * 6 + 5 - 4 ^ 2 * 3 + 2 ^ 3 - 1) = 347 := 
by sorry

end math_problem_proof_l810_810956


namespace number_of_friends_l810_810336

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l810_810336


namespace part1_part2_l810_810671

-- Part 1: Prove measure of angle A given the condition 2a*sin B = sqrt(3)b
theorem part1 (a b c A B C : ℝ) (h1 : 2 * a * sin B = sqrt 3 * b) : A = π / 3 ∨ A = 2 * π / 3 :=
sorry

-- Part 2: Prove the maximum value of c/b + b/c given the altitude condition on side BC
theorem part2 (a b c A B C : ℝ) (h2 : sin A = sqrt 3 / 2) (h3 : (a^2) = 2 * b * c * sin A) : 
  2 * sqrt 2 = max (c / b + b / c) (b / c + c / b) :=
sorry

end part1_part2_l810_810671


namespace gcd_problem_l810_810176

theorem gcd_problem (x : ℤ) (h : ∃ k, x = 2 * 2027 * k) :
  Int.gcd (3 * x ^ 2 + 47 * x + 101) (x + 23) = 1 :=
sorry

end gcd_problem_l810_810176


namespace range_of_a_if_odd_symmetric_points_l810_810118

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a

theorem range_of_a_if_odd_symmetric_points (a : ℝ): 
  (∃ x₀ : ℝ, x₀ ≠ 0 ∧ f x₀ a = -f (-x₀) a) → (1 < a) :=
by 
  sorry

end range_of_a_if_odd_symmetric_points_l810_810118


namespace shortest_distance_from_MA_to_BC_l810_810711

noncomputable def cube_distance (a : ℝ) : ℝ :=
  let distance := (2 * real.sqrt 5) / 5 * a
  distance

theorem shortest_distance_from_MA_to_BC (a : ℝ) :
  let cube := cube_distance a in
  cube = (2 * real.sqrt 5 / 5) * a :=
by
  sorry

end shortest_distance_from_MA_to_BC_l810_810711


namespace right_pyramid_volume_l810_810019

noncomputable def volume_of_right_pyramid (base_area lateral_face_area total_surface_area : ℝ) : ℝ := 
  let height := (10 : ℝ) / 3
  (1 / 3) * base_area * height

theorem right_pyramid_volume (total_surface_area base_area lateral_face_area : ℝ)
  (h0 : total_surface_area = 300)
  (h1 : base_area + 3 * lateral_face_area = total_surface_area)
  (h2 : lateral_face_area = base_area / 3) 
  : volume_of_right_pyramid base_area lateral_face_area total_surface_area = 500 / 3 := 
by
  sorry

end right_pyramid_volume_l810_810019


namespace birch_trees_probability_l810_810015

/--
A gardener plants four pine trees, five oak trees, and six birch trees in a row. He plants them in random order, each arrangement being equally likely.
Prove that no two birch trees are next to one another is \(\frac{2}{45}\).
--/
theorem birch_trees_probability: (∃ (m n : ℕ), (m = 2) ∧ (n = 45) ∧ (no_two_birch_trees_adjacent_probability = m / n)) := 
sorry

end birch_trees_probability_l810_810015


namespace right_triangle_cos_square_sum_obtuse_triangle_cos_square_sum_acute_triangle_cos_square_sum_l810_810106

variable (A B C : ℝ)

-- Conditions for right triangle
def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ (A = π / 2 ∨ B = π / 2 ∨ C = π / 2)

-- Conditions for obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2)

-- Conditions for acute triangle
def is_acute_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ A < π / 2 ∧ B < π / 2 ∧ C < π / 2

-- Proof statement for right triangle
theorem right_triangle_cos_square_sum (h : is_right_triangle A B C) : 
  cos A ^ 2 + cos B ^ 2 + cos C ^ 2 = 1 := 
sorry

-- Proof statement for obtuse triangle
theorem obtuse_triangle_cos_square_sum (h : is_obtuse_triangle A B C) : 
  1 < cos A ^ 2 + cos B ^ 2 + cos C ^ 2 ∧ cos A ^ 2 + cos B ^ 2 + cos C ^ 2 < 3 := 
sorry

-- Proof statement for acute triangle
theorem acute_triangle_cos_square_sum (h : is_acute_triangle A B C) :
  3 / 4 ≤ cos A ^ 2 + cos B ^ 2 + cos C ^ 2 ∧ cos A ^ 2 + cos B ^ 2 + cos C ^ 2 < 1 := 
sorry

end right_triangle_cos_square_sum_obtuse_triangle_cos_square_sum_acute_triangle_cos_square_sum_l810_810106


namespace max_value_of_xy_plus_yz_l810_810600

theorem max_value_of_xy_plus_yz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : xy + yz ≤ sqrt(2) / 2 :=
sorry

end max_value_of_xy_plus_yz_l810_810600


namespace compute_expression_l810_810542

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 :=
by sorry

end compute_expression_l810_810542


namespace count_integer_points_on_line_between_C_and_D_l810_810490

def C := (2 : ℤ, 3 : ℤ)
def D := (101 : ℤ, 503 : ℤ)

def line_through_points (p1 p2 : ℤ × ℤ) (x y : ℤ) : Prop :=
  (y - p1.2) * (p2.1 - p1.1) = (p2.2 - p1.2) * (x - p1.1)

def strictly_between (x a b : ℤ) : Prop :=
  a < x ∧ x < b

theorem count_integer_points_on_line_between_C_and_D :
  let points_on_line := { p : ℤ × ℤ | line_through_points C D p.1 p.2 }
  let valid_points := { p : ℤ × ℤ | p ∈ points_on_line ∧ strictly_between p.1 2 101 }
  ∃ n, n = 17 ∧ finite valid_points ∧ (valid_points.card = n) :=
sorry

end count_integer_points_on_line_between_C_and_D_l810_810490


namespace cos_of_right_triangle_l810_810254

   theorem cos_of_right_triangle {
     a b c : ℝ
   } (h : a = 4)
     (h' : b = 3) 
     (h'' : c = Real.sqrt (a^2 + b^2)) 
     (right_angle : angle_ABC = 90°) : 
     Real.cos (angle_A) = 3 / 5 := 
   by
     sorry
   
end cos_of_right_triangle_l810_810254


namespace complex_number_satisfies_eq_l810_810155

theorem complex_number_satisfies_eq (z : ℂ) (h : 1 + z * complex.I = z + complex.I) : z = -1 :=
begin
  sorry,
end

end complex_number_satisfies_eq_l810_810155


namespace max_x_lcm_15_21_105_l810_810765

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l810_810765


namespace total_pieces_on_chessboard_l810_810060

-- Given conditions about initial chess pieces and lost pieces.
def initial_pieces_each : Nat := 16
def pieces_lost_arianna : Nat := 3
def pieces_lost_samantha : Nat := 9

-- The remaining pieces for each player.
def remaining_pieces_arianna : Nat := initial_pieces_each - pieces_lost_arianna
def remaining_pieces_samantha : Nat := initial_pieces_each - pieces_lost_samantha

-- The total remaining pieces on the chessboard.
def total_remaining_pieces : Nat := remaining_pieces_arianna + remaining_pieces_samantha

-- The theorem to prove
theorem total_pieces_on_chessboard : total_remaining_pieces = 20 :=
by
  sorry

end total_pieces_on_chessboard_l810_810060


namespace simple_interest_principal_l810_810419

theorem simple_interest_principal
  (P_CI : ℝ)
  (r_CI t_CI : ℝ)
  (CI : ℝ)
  (P_SI : ℝ)
  (r_SI t_SI SI : ℝ)
  (h_compound_interest : (CI = P_CI * (1 + r_CI / 100)^t_CI - P_CI))
  (h_simple_interest : SI = (1 / 2) * CI)
  (h_SI_formula : SI = P_SI * r_SI * t_SI / 100) :
  P_SI = 1750 :=
by
  have P_CI := 4000
  have r_CI := 10
  have t_CI := 2
  have r_SI := 8
  have t_SI := 3
  have CI := 840
  have SI := 420
  sorry

end simple_interest_principal_l810_810419


namespace intersection_l810_810201

noncomputable def M : Set ℝ := { x : ℝ | Real.sqrt (x + 1) ≥ 0 }
noncomputable def N : Set ℝ := { x : ℝ | x^2 + x - 2 < 0 }

theorem intersection (x : ℝ) : x ∈ (M ∩ N) ↔ -1 ≤ x ∧ x < 1 := by
  sorry

end intersection_l810_810201


namespace sum_of_distances_l810_810691

noncomputable def parabola : ℝ → ℝ := λ x, 2 * x^2

def circle_intersects (P : (ℝ × ℝ) → Prop) : Prop := 
  ∃ k h r : ℝ, (∀ x y : ℝ, P (x, y) ↔ (x - k)^2 + (y - h)^2 = r^2)

def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def focus : ℝ × ℝ := (0, 1 / 8)

theorem sum_of_distances : 
  parabola (-1) = 2 ∧ 
  parabola 5 = 50 ∧ 
  parabola 0 = 0 ∧ 
  circle_intersects (λ p, p.2 = parabola p.1) →
  (distance focus (-1, 2) + 
  distance focus (5, 50) + 
  distance focus (0, 0) + 
  distance focus (-4, 32)) = 169 :=
by
  intros h1 h2 h3 h4,
  sorry

end sum_of_distances_l810_810691


namespace num_five_student_committees_l810_810119

theorem num_five_student_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 5) : choose n k = 56 :=
by
  rw [h_n, h_k]
  -- rest of the proof would go here
  sorry

end num_five_student_committees_l810_810119


namespace limit_fraction_l810_810539

theorem limit_fraction :
  (Real.tendsto (fun n : ℕ => (2 * n : ℝ) / (4 * n + 1)) atTop (𝓝 (1 / 2))) :=
by
  sorry

end limit_fraction_l810_810539


namespace average_salary_rest_workers_l810_810650

def average_salary_all_workers : ℝ := 850
def number_of_technicians : ℝ := 7
def average_salary_technicians : ℝ := 1000
def total_number_of_workers : ℝ := 22

theorem average_salary_rest_workers :
  let total_salary_all_workers := average_salary_all_workers * total_number_of_workers,
      total_salary_technicians := average_salary_technicians * number_of_technicians,
      total_salary_rest_workers := total_salary_all_workers - total_salary_technicians,
      number_of_rest_workers := total_number_of_workers - number_of_technicians,
      average_salary_rest_workers := total_salary_rest_workers / number_of_rest_workers
  in average_salary_rest_workers = 780 :=
begin
  sorry
end

end average_salary_rest_workers_l810_810650


namespace incorrect_fraction_addition_l810_810929

theorem incorrect_fraction_addition (a b x y : ℤ) (h1 : 0 < b) (h2 : 0 < y) (h3 : (a + x) * (b * y) = (a * y + b * x) * (b + y)) :
  ∃ k : ℤ, x = -a * k^2 ∧ y = b * k :=
by
  sorry

end incorrect_fraction_addition_l810_810929


namespace find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l810_810164

def f : ℝ → ℝ :=
  sorry

noncomputable def f_properties : Prop :=
  (∀ x y : ℝ, x < 0 → f x < 0 → f x + f y = f (x * y) / f (x + y)) ∧ f 1 = 1

theorem find_f2_f_neg1 :
  f_properties →
  f 2 = 1 / 2 ∧ f (-1) = -1 :=
sorry

theorem f_is_odd :
  f_properties →
  ∀ x : ℝ, f x = -f (-x) :=
sorry

theorem f_monotonic_on_negatives :
  f_properties →
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2 :=
sorry

end find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l810_810164


namespace find_angle_C_find_area_l810_810645

-- Problem conditions as definitions
variables {A B C : ℝ}
variables {a b c : ℝ}
variable h1 : a * cos B + b * cos A = 2 * c * cos C
variable h2 : a + b = 4
variable h3 : c = 2

-- First proposition: finding the angle C
theorem find_angle_C (h1 : a * cos B + b * cos A = 2 * c * cos C) : C = π / 3 :=
sorry

-- Second proposition: finding the area given a + b = 4 and c = 2
theorem find_area (h1 : a * cos B + b * cos A = 2 * c * cos C) (h2 : a + b = 4) (h3 : c = 2) :
  (1 / 2) * a * b * sin C = sqrt 3 :=
sorry

end find_angle_C_find_area_l810_810645


namespace dihedral_angle_ge_l810_810495

-- Define the problem conditions and goal in Lean
theorem dihedral_angle_ge (n : ℕ) (h : 3 ≤ n) (ϕ : ℝ) :
  ϕ ≥ π * (1 - 2 / n) := 
sorry

end dihedral_angle_ge_l810_810495


namespace find_GH_l810_810656

/-- In right triangle GHI, we have ∠G = 40°, ∠H = 90°, and HI = 12.
    The length of GH is approximately 14.3 to the nearest tenth. -/
theorem find_GH
  (G H I : Type)
  [triangle : RightTriangle G H I]
  (angle_G : ∠G = 40)
  (angle_H : ∠H = 90)
  (side_HI : distance H I = 12) :
  distance G H ≈ 14.3 :=
sorry

end find_GH_l810_810656


namespace children_had_to_sit_again_l810_810839

theorem children_had_to_sit_again (total children_passed : ℝ) (h1 : total = 698.0) (h2 : children_passed = 105.0) : total - children_passed = 593.0 :=
by
  rw [h1, h2]
  norm_num
  sorry -- This will skip the proof steps for now.

end children_had_to_sit_again_l810_810839


namespace percentage_of_80_subtracted_from_12_percent_of_160_gives_11_point_2_l810_810454

theorem percentage_of_80_subtracted_from_12_percent_of_160_gives_11_point_2 :
  ∃ (x : ℤ), (0.12 * 160) - ((x / 100) * 80) = 11.2 ∧ x = 10 :=
by
  use 10
  simp
  norm_num
  sorry

end percentage_of_80_subtracted_from_12_percent_of_160_gives_11_point_2_l810_810454


namespace smallest_abundant_number_not_multiple_of_10_l810_810055

-- Definition of proper divisors of a number n
def properDivisors (n : ℕ) : List ℕ := 
  (List.range n).filter (λ d => d > 0 ∧ n % d = 0)

-- Definition of an abundant number
def isAbundant (n : ℕ) : Prop := 
  (properDivisors n).sum > n

-- Definition of not being a multiple of 10
def notMultipleOf10 (n : ℕ) : Prop := 
  n % 10 ≠ 0

-- Statement to prove
theorem smallest_abundant_number_not_multiple_of_10 :
  ∃ n, isAbundant n ∧ notMultipleOf10 n ∧ ∀ m, (isAbundant m ∧ notMultipleOf10 m) → n ≤ m :=
by
  sorry

end smallest_abundant_number_not_multiple_of_10_l810_810055


namespace marked_points_on_same_arc_of_120_degrees_l810_810474

-- defining what it means for points to be on a circle
variables {Point : Type*} (circle : set Point) (is_on_circle : Point → Prop)

-- n points marked on the circle
variables (n : ℕ) (marked_points : fin n → Point)

-- conditions given in the problem
def arc_less_than_120_degrees (A B : Point) (hA : is_on_circle A) (hB : is_on_circle B) : Prop :=
  -- assuming a function measure_arc that measures the arc between two points on the circle
  -- this should be defined appropriately in the context of the problem
  measure_arc A B < 120

-- problem to be proved in Lean
theorem marked_points_on_same_arc_of_120_degrees
  (h_circle : ∀ (i : fin n), is_on_circle (marked_points i))
  (h_arc : ∀ (i j : fin n) (h_i : is_on_circle (marked_points i)) (h_j : is_on_circle (marked_points j)),
    arc_less_than_120_degrees (marked_points i) (marked_points j) h_i h_j) :
  ∃ (A B : Point) (hA : is_on_circle A) (hB : is_on_circle B),
    (∀ (i : fin n), measure_arc A (marked_points i) < 120 ∧ measure_arc (marked_points i) B < 120) :=
by { sorry }

end marked_points_on_same_arc_of_120_degrees_l810_810474


namespace tan_product_cos_conditions_l810_810582

variable {α β : ℝ}

theorem tan_product_cos_conditions
  (h1 : Real.cos (α + β) = 2 / 3)
  (h2 : Real.cos (α - β) = 1 / 3) :
  Real.tan α * Real.tan β = -1 / 3 :=
sorry

end tan_product_cos_conditions_l810_810582


namespace student_correct_answers_l810_810468

-- Defining the conditions as variables and equations
def correct_answers (c w : ℕ) : Prop :=
  c + w = 60 ∧ 4 * c - w = 160

-- Stating the problem: proving the number of correct answers is 44
theorem student_correct_answers (c w : ℕ) (h : correct_answers c w) : c = 44 :=
by 
  sorry

end student_correct_answers_l810_810468


namespace b_4_lt_b_7_l810_810043

-- Define the sequence
def b (α : ℕ → ℕ) (n : ℕ) : ℚ :=
  let rec aux : ℕ → ℚ
    | 0 => 0
    | k+1 => (aux k) + (1 / (α k + 1)) 
  in 1 + 1 / aux n

-- Define the conditions
variable (α : ℕ → ℕ)
variable hα : ∀ k, α k ∈ Nat.succ <$> Finset.range 1

-- The proof problem
theorem b_4_lt_b_7 (α : ℕ → ℕ) (hα : ∀ k, α k ∈ Nat.succ <$> Finset.range 1) : 
  b α 4 < b α 7 := by
  sorry

end b_4_lt_b_7_l810_810043


namespace exponent_division_simplification_l810_810536

theorem exponent_division_simplification :
  ((18^18 / 18^17)^2 * 9^2) / 3^4 = 324 :=
by
  sorry

end exponent_division_simplification_l810_810536


namespace angle_AOD_l810_810241

theorem angle_AOD (x : ℝ) 
  (h1 : x = 36)
  (h2 : ∠AOD = 4 * ∠BOC)
  (h3 : ∠BOC = x)
  : ∠AOD = 144 :=
by
  sorry

end angle_AOD_l810_810241


namespace cars_meet_after_40_minutes_l810_810731

noncomputable def time_to_meet 
  (BC CD : ℝ) (speed : ℝ) 
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) : ℝ :=
  (BC + CD) / speed * 40 / 60

-- Define the condition that must hold: cars meet at 40 minutes
theorem cars_meet_after_40_minutes
  (BC CD : ℝ) (speed : ℝ)
  (constant_speed : ∀ t, t > 0 → speed = (BC + CD) / t) :
  time_to_meet BC CD speed constant_speed = 40 := sorry

end cars_meet_after_40_minutes_l810_810731


namespace find_monotonically_increasing_and_odd_l810_810946

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x < f y

noncomputable def functions : list (ℝ → ℝ) := [λ x, Real.tan x, λ x, -1/x, λ x, x - Real.cos x, λ x, Real.exp x - Real.exp (-x)]

theorem find_monotonically_increasing_and_odd :
  (∃ f ∈ functions, is_odd_function f ∧ is_monotonically_increasing f) ∧
  (∀ g ∈ functions, is_odd_function g ∧ is_monotonically_increasing g → g = (λ x, Real.exp x - Real.exp (-x))) :=
by
  sorry

end find_monotonically_increasing_and_odd_l810_810946


namespace scientific_notation_of_0_0000000028_l810_810748

theorem scientific_notation_of_0_0000000028 : (0.0000000028 : ℝ) = 2.8 * 10^(-9) :=
sorry

end scientific_notation_of_0_0000000028_l810_810748


namespace polynomial_sum_of_squares_l810_810889

theorem polynomial_sum_of_squares
  (p : Polynomial ℝ)
  (p_0 : ℝ)
  (h_pos : 0 < p.leading_coeff)
  (h_no_real_roots : ∀ x : ℝ, ¬ (Polynomial.eval x p) = 0) :
  ∃ f g : Polynomial ℝ, p = f^2 + g^2 := 
sorry

end polynomial_sum_of_squares_l810_810889


namespace greatest_x_lcm_l810_810816

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l810_810816


namespace closed_chain_possible_l810_810884

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

end closed_chain_possible_l810_810884


namespace jack_buttons_total_l810_810273

theorem jack_buttons_total :
  (3 * 3) * 7 = 63 :=
by
  sorry

end jack_buttons_total_l810_810273


namespace vector_calculation_l810_810159

def vector_a : ℝ × ℝ := (1, -1)
def vector_b : ℝ × ℝ := (-1, 2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

theorem vector_calculation :
  (dot_product (vector_add (scalar_mult 2 vector_a) vector_b) vector_a) = 1 :=
by
  sorry

end vector_calculation_l810_810159


namespace parabola_focus_l810_810424
noncomputable def parabola_standard_equation (x y : ℝ) : Prop :=
  (∃ p : ℝ, p = 8 ∧ y^2 = 2 * p * x)

theorem parabola_focus (x y : ℝ) :
  (∃ h : ℝ, h = 4 ∧ ∀ (x y : ℝ), ((x^2 / 16) - (y^2 / 9) = 1) → ((x, y) = (h, 0))) →
  parabola_standard_equation x y :=
by 
  intros hExists
  rcases hExists with ⟨h, h_value, focus_condition⟩
  have hx4 : h = 4 := h_value
  have foc : ∀ (x y : ℝ), ((x^2 / 16) - (y^2 / 9) = 1) → ((x, y) = (4, 0)) := focus_condition
  sorry

end parabola_focus_l810_810424


namespace total_green_ducks_percentage_l810_810239

def ducks_in_park_A : ℕ := 200
def green_percentage_A : ℕ := 25

def ducks_in_park_B : ℕ := 350
def green_percentage_B : ℕ := 20

def ducks_in_park_C : ℕ := 120
def green_percentage_C : ℕ := 50

def ducks_in_park_D : ℕ := 60
def green_percentage_D : ℕ := 25

def ducks_in_park_E : ℕ := 500
def green_percentage_E : ℕ := 30

theorem total_green_ducks_percentage (green_ducks_A green_ducks_B green_ducks_C green_ducks_D green_ducks_E total_ducks : ℕ)
  (h_A : green_ducks_A = ducks_in_park_A * green_percentage_A / 100)
  (h_B : green_ducks_B = ducks_in_park_B * green_percentage_B / 100)
  (h_C : green_ducks_C = ducks_in_park_C * green_percentage_C / 100)
  (h_D : green_ducks_D = ducks_in_park_D * green_percentage_D / 100)
  (h_E : green_ducks_E = ducks_in_park_E * green_percentage_E / 100)
  (h_total_ducks : total_ducks = ducks_in_park_A + ducks_in_park_B + ducks_in_park_C + ducks_in_park_D + ducks_in_park_E) :
  (green_ducks_A + green_ducks_B + green_ducks_C + green_ducks_D + green_ducks_E) * 100 / total_ducks = 2805 / 100 :=
by sorry

end total_green_ducks_percentage_l810_810239


namespace garden_area_difference_l810_810494

noncomputable def garden_difference : ℝ := 
  let length := 60
  let width := 20
  let area_rect := length * width
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  let area_circle := Real.pi * radius^2
  area_circle - area_rect

theorem garden_area_difference : garden_difference ≈ 837.62 := by
  sorry

end garden_area_difference_l810_810494


namespace square_area_l810_810505

theorem square_area {d : ℝ} (h : d = 12 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 144 ∧ ( ∃ s : ℝ, s = d / Real.sqrt 2 ∧ A = s^2 ) :=
by
  sorry

end square_area_l810_810505


namespace die_probabilities_l810_810024

-- Definitions related to the problem
def is_prime (n : ℕ) : Prop := 
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def eight_sided_die_faces := {n : ℕ | n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8}

def roll_die_five_times := (list ℕ)

-- Statements about the problem
theorem die_probabilities (m n : ℕ) (coprime : nat.gcd m n = 1) : 
(eight_sided_die_faces.prod | ⟨l, hl⟩ ← roll_die_five_times, is_perfect_square l.prod) = m / n → m + n = 468 :=
begin
  sorry
end

end die_probabilities_l810_810024


namespace five_student_committees_l810_810147

theorem five_student_committees (n k : ℕ) (hn : n = 8) (hk : k = 5) : 
  nat.choose n k = 56 := by
  rw [hn, hk]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end five_student_committees_l810_810147


namespace find_simple_interest_principal_l810_810421

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ n

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t / 100

theorem find_simple_interest_principal : 
  (simple_interest P 8 3 = 1 / 2 * compound_interest 4000 10 2) → 
  P = 1750 := 
by
  sorry

end find_simple_interest_principal_l810_810421


namespace sqrt_sub_pow_add_sin_l810_810540

theorem sqrt_sub_pow_add_sin :
  sqrt 9 - (1 / 2)^0 + 2 * real.sin (real.pi / 6) = 3 :=
by sorry

end sqrt_sub_pow_add_sin_l810_810540


namespace five_student_committees_from_eight_l810_810132

theorem five_student_committees_from_eight : nat.choose 8 5 = 56 := by
  sorry

end five_student_committees_from_eight_l810_810132


namespace linear_function_properties_l810_810200

theorem linear_function_properties (x : ℝ) :
  (∀ x, -3 * x + 2 ≠ -y ∧ 2 > 0 ∧ -3 < 0 →  (¬∃ x, y=-3 * x + 2 ∧ x<0 ∧ -3<0<y)) ∧
  (∀ x, ∀ y, y = 2 * x - 4 → y + 5 = 2 * x + 1) := by
  sorry

end linear_function_properties_l810_810200


namespace number_of_friends_l810_810328

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l810_810328


namespace tom_walking_distance_l810_810675

noncomputable def walking_rate_miles_per_minute : ℝ := 1 / 18
def walking_time_minutes : ℝ := 15
def expected_distance_miles : ℝ := 0.8

theorem tom_walking_distance :
  walking_rate_miles_per_minute * walking_time_minutes = expected_distance_miles :=
by
  -- Calculation steps and conversion to decimal are skipped
  sorry

end tom_walking_distance_l810_810675


namespace distinct_values_of_expression_is_3_l810_810073

def evaluateExpression (e : (ℕ → ℕ → ℕ)) (x : ℕ) : ℕ :=
  match x with
  -- Define the base cases and recursive computation for expressions
  | 0 => 1
  | (n + 1) => e 3 (evaluateExpression e n)

-- Expressions with different valid placements of parentheses for 3^{3^{3^{3}}}
def expr1 := 3 ^ (3 ^ (3 ^ 3))
def expr2 := 3 ^ ((3 ^ 3) ^ 3)
def expr3 := ((3 ^ 3) ^ 3) ^ 3
def expr4 := 3 ^ (3 ^ (3 ^ 3))
def expr5 := (3 ^ (3 ^ 3)) ^ 3

-- Proof statement
def numberOfDistinctValues : Prop := 
  {expr1, expr2, expr3, expr4, expr5}.card = 3

-- Theorem to state the problem
theorem distinct_values_of_expression_is_3 : numberOfDistinctValues :=
by
  sorry

end distinct_values_of_expression_is_3_l810_810073


namespace angle_ratio_l810_810250

-- Conditions
variables {O : Type} [circle O] (D E F G : O)
  (hDE : arc_length D E = 160) (hEF : arc_length E F = 56)
  (hG_on_arc_DF : G ∈ minor_arc D F) (hOG_perp_DF : perpendicular (line_segment O G) (line_segment D F))

-- Statement
theorem angle_ratio (h_acute_triangle : acute_triangle D E F) :
  let ω₁ := (angle O G F) in
  let ω₂ := (angle D E F) in
  ratio ω₁ ω₂ = 9/10 :=
sorry

end angle_ratio_l810_810250


namespace charlie_has_largest_final_answer_l810_810942

theorem charlie_has_largest_final_answer :
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  charlie > alice ∧ charlie > bob :=
by
  -- Definitions of intermediate variables
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  -- Comparison assertions
  sorry

end charlie_has_largest_final_answer_l810_810942


namespace birthday_friends_count_l810_810372

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l810_810372


namespace smallest_4_digit_multiple_of_112_l810_810878

theorem smallest_4_digit_multiple_of_112 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 112 ∣ n ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ 112 ∣ m → n ≤ m :=
begin
  use 1008,
  repeat { sorry },
end

end smallest_4_digit_multiple_of_112_l810_810878


namespace birthday_friends_count_l810_810352

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l810_810352


namespace num_five_student_committees_l810_810123

theorem num_five_student_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 5) : choose n k = 56 :=
by
  rw [h_n, h_k]
  -- rest of the proof would go here
  sorry

end num_five_student_committees_l810_810123


namespace man_speed_42_minutes_7_km_l810_810913

theorem man_speed_42_minutes_7_km 
  (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ)
  (h1 : distance = 7) 
  (h2 : time_minutes = 42) 
  (h3 : time_hours = time_minutes / 60) :
  distance / time_hours = 10 := by
  sorry

end man_speed_42_minutes_7_km_l810_810913


namespace convert_polar_to_rectangular_l810_810079

theorem convert_polar_to_rectangular :
  ∀ r θ : ℝ, r = 4 ∧ θ = (3 * Real.pi / 4) →
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  x = -2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 :=
by
  intros r θ h
  cases h
  have h1 : Real.cos (3 * Real.pi / 4) = -1 / Real.sqrt 2 := sorry
  have h2 : Real.sin (3 * Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  simp [h1, h2]
  split
  case x =>
    sorry
  case y =>
    sorry

end convert_polar_to_rectangular_l810_810079


namespace anton_happy_ways_l810_810883

-- Define the components required for the problem: Bracelets, Stones, Clasps
inductive Bracelet
| silver | gold | steel

inductive Stone
| cubic_zirconia | emerald | quartz | diamond | agate

inductive Clasp
| classic | butterfly | buckle

-- Define a structure for a Watch
structure Watch :=
(bracelet : Bracelet)
(stone : Stone)
(clasp : Clasp)

-- Define a condition to check if three watches make Anton happy
def watches_make_anton_happy (w1 w2 w3 : Watch) : Prop :=
  (w1.bracelet ≠ w2.bracelet ∧ w1.bracelet ≠ w3.bracelet ∧ w2.bracelet ≠ w3.bracelet) ∧
  (w1.stone ≠ w2.stone ∧ w1.stone ≠ w3.stone ∧ w2.stone ≠ w3.stone) ∧
  (w1.clasp ≠ w2.clasp ∧ w1.clasp ≠ w3.clasp ∧ w2.clasp ≠ w3.clasp) ∧
  (w1.bracelet = Bracelet.steel ∧ w1.clasp = Clasp.classic ∧ w1.stone = Stone.cubic_zirconia) ∧
  ((w2.bracelet = Bracelet.gold ∧ w3.bracelet = Bracelet.silver) ∨ 
   (w2.bracelet = Bracelet.silver ∧ w3.bracelet = Bracelet.gold))

-- The theorem to prove that the number of ways to make Anton happy is 72
theorem anton_happy_ways : ∃! n : ℕ, n = 72 ∧
  ∃ w1 w2 w3 : Watch, watches_make_anton_happy w1 w2 w3 :=
begin
  sorry
end

end anton_happy_ways_l810_810883


namespace find_a_eq_half_l810_810616

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * Real.log x
noncomputable def g (x a : ℝ) : ℝ := f x a - 2 * a * x

theorem find_a_eq_half (a : ℝ) (h₀ : a > 0) (h₁ : ∃ x, g x a = 0) :
  a = 1 / 2 :=
begin
  sorry
end

end find_a_eq_half_l810_810616


namespace friends_at_birthday_l810_810363

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l810_810363


namespace unique_critical_point_extremum_number_of_zeros_l810_810589

noncomputable def f (a x : ℝ) : ℝ := x + a / x - (a - 1) * Real.log x - 2

theorem unique_critical_point_extremum (a : ℝ) (ha : f a a = 0) (h_unique : ∀ x ≠ a, f' a x ≠ 0) :
  a = 1 ∨ a = Real.exp 1 := sorry

theorem number_of_zeros (a : ℝ) :
  (1 < a ∧ a < Real.exp 1 → ∀ x ∈ set.Icc 1 (Real.exp 1), f a x ≠ 0) ∧
  ((a ≤ 1 ∨ Real.exp 1 ≤ a) → ∃! x ∈ set.Icc 1 (Real.exp 1), f a x = 0) := sorry

end unique_critical_point_extremum_number_of_zeros_l810_810589


namespace necessary_and_sufficient_condition_l810_810192

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a / x

theorem necessary_and_sufficient_condition :
  ∀ (a : ℝ), (1 < a ∧ a < 16) ↔ 
    (∀ (x : ℝ), x > 0 → (f' x a < 0 ↔ x < real.sqrt a) ∧ (f' x a > 0 ↔ x > real.sqrt a)) 
    ∧ (∃ (x : ℝ), x = real.sqrt a ∧ (∀ (y : ℝ), y > 0 → f y a ≥ f x a)) :=
sorry

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 1 - a / x^2

end necessary_and_sufficient_condition_l810_810192


namespace volume_after_2_hours_l810_810926

-- Define initial volume and increase rate
def initial_volume : ℝ := 500  -- in cm³
def increase_rate : ℝ := 2 / 5

-- Define function for volume after n hours
def volume_after_hours (hours : ℕ) : ℝ :=
  let rec calc_volume (hours : ℕ) (current_volume : ℝ) : ℝ :=
    match hours with
    | 0 => current_volume
    | hours + 1 => calc_volume hours (current_volume + increase_rate * current_volume)
  calc_volume hours initial_volume

-- Prove the volume after 2 hours is 980 cm³
theorem volume_after_2_hours : volume_after_hours 2 = 980 := by
  sorry

end volume_after_2_hours_l810_810926


namespace pyramid_cosine_zero_l810_810167

noncomputable def pyramidAngleCosine (A B C D P E : ℝ → ℝ × ℝ × ℝ) : ℝ :=
  let base_side := 2
  let slant_edge := 4
  let BD := Math.sqrt ((base_side ^ 2) + (base_side ^ 2))
  let PE := slant_edge / 2
  let BE := BD / 2
  let CE := Math.sqrt ((BE ^ 2) + (base_side ^ 2))
  let cos_angle_BCE := (base_side ^ 2 + BE ^ 2 - CE ^ 2) / (2 * base_side * BE)
  0

theorem pyramid_cosine_zero (A B C D P E : ℝ → ℝ × ℝ × ℝ) :
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  sqrt((2 : ℝ) ^ 2 + (2 : ℝ) ^ 2) = 2 * sqrt(2) →
  sqrt((sqrt(2) ^ 2) + (2 : ℝ) ^ 2) = sqrt(6) →
  let base_side := 2
  let BE := sqrt(2)
  let CE := sqrt(6)
  let cos_angle_BCE := (base_side ^ 2 + BE ^ 2 - CE ^ 2) / (2 * base_side * BE)
  cos_angle_BCE = 0 :=
sorry

end pyramid_cosine_zero_l810_810167


namespace greatest_x_lcm_l810_810803

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l810_810803


namespace mode_and_median_equal_l810_810844

theorem mode_and_median_equal 
  (sizes : List ℚ)
  (quantities : List ℕ)
  (h_sizes : sizes = [25, 25.5, 26, 26.5, 27])
  (h_quantities : quantities = [2, 4, 2, 1, 1]) :
  (mode sizes quantities = 25.5) ∧ (median sizes quantities = 25.5) :=
by
  -- insert proof here
  sorry

end mode_and_median_equal_l810_810844


namespace combination_eight_choose_five_l810_810136

theorem combination_eight_choose_five : 
  ∀ (n k : ℕ), n = 8 ∧ k = 5 → Nat.choose n k = 56 :=
by 
  intros n k h
  obtain ⟨hn, hk⟩ := h
  rw [hn, hk]
  exact Nat.choose_eq 8 5
  sorry  -- This signifies that the proof needs to be filled in, but we'll skip it as per instructions.

end combination_eight_choose_five_l810_810136


namespace balazs_missed_number_l810_810065

theorem balazs_missed_number (n k : ℕ) 
  (h1 : n * (n + 1) / 2 = 3000 + k)
  (h2 : 1 ≤ k)
  (h3 : k < n) : k = 3 := by
  sorry

end balazs_missed_number_l810_810065


namespace zongzi_cost_price_after_festival_maximize_zongzi_profit_l810_810753

-- Defining the given conditions and proving the questions

theorem zongzi_cost_price_after_festival :
  ∃ x : ℝ, (240 / x - 4 = 240 / (x + 2)) ∧ x = 10 :=
by
  -- Proof sketch: Solve the system of equations provided in the problem 
  -- and show that the correct x is 10.
  simp
  existsi (10 : ℝ)
  sorry

theorem maximize_zongzi_profit :
  ∃ m w : ℝ, 
    (12 * m + 10 * (400 - m) ≤ 4600) ∧ 
    (w = (20 - 12) * m + (16 - 10) * (400 - m)) ∧ 
    m = 300 ∧ w = 3000 :=
by
  -- Proof sketch: Manipulate provided inequality and profit function, 
  -- and show the maximum profit w is 3000 when m is 300.
  simp
  existsi (300 : ℝ, 3000 : ℝ)
  sorry

end zongzi_cost_price_after_festival_maximize_zongzi_profit_l810_810753


namespace parallel_lines_equivalence_l810_810590

def lines_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  ∀ {x y : ℝ}, (a1 * x + b1 * y + c1 = 0) ∧ (a2 * x + b2 * y + c2 = 0) → 
                (a1 * b2 - a2 * b1 = 0)

theorem parallel_lines_equivalence (a1 b1 c1 a2 b2 c2 : ℝ) (p : Prop) :
  (p = lines_parallel a1 b1 c1 a2 b2 c2) →
  -- f(p) denotes the number of true statements among the original proposition,
  -- its converse, negation, and contrapositive.
  f(p) = 2 :=
by
  sorry

end parallel_lines_equivalence_l810_810590


namespace student_marks_l810_810515

variable (max_marks : ℕ) (passing_percentage : ℕ) (marks_failed_by : ℕ)

theorem student_marks :
  max_marks = 400 → passing_percentage = 33 → marks_failed_by = 40 →
  let passing_marks := (passing_percentage * max_marks) / 100 in
  let obtained_marks := passing_marks - marks_failed_by in
  obtained_marks = 92 :=
by
  intros h1 h2 h3 
  let passing_marks := (passing_percentage * max_marks) / 100
  let obtained_marks := passing_marks - marks_failed_by
  rw [h1, h2, h3]
  have : passing_marks = 132 := by norm_num
  rw this
  norm_num
  sorry

end student_marks_l810_810515


namespace total_rainfall_in_january_l810_810977

theorem total_rainfall_in_january 
  (r1 r2 : ℝ)
  (h1 : r2 = 1.5 * r1)
  (h2 : r2 = 18) : 
  r1 + r2 = 30 := by
  sorry

end total_rainfall_in_january_l810_810977


namespace jack_buttons_l810_810276

theorem jack_buttons :
  ∀ (shirts_per_kid kids buttons_per_shirt : ℕ),
  shirts_per_kid = 3 →
  kids = 3 →
  buttons_per_shirt = 7 →
  (shirts_per_kid * kids * buttons_per_shirt) = 63 :=
by
  intros shirts_per_kid kids buttons_per_shirt h1 h2 h3
  rw [h1, h2, h3]
  calc
    3 * 3 * 7 = 9 * 7 : by rw mul_assoc
            ... = 63   : by norm_num

end jack_buttons_l810_810276


namespace circumcircles_equal_circumcircle_abc_l810_810266

variable {A B C D F G : Type}
variable [triangle : Triangle A B C]
variable [point_on_ab : D ∈ segment A B]
variable [parallel_dg_bc : Line.parallel (Line.mk D G) (Line.mk B C)]
variable [parallel_df_ac : Line.parallel (Line.mk D F) (Line.mk A C)]
variable [sim_adg_abc : Similar (Triangle.mk A D G) (Triangle.mk A B C)]
variable [sim_dbf_abc : Similar (Triangle.mk D B F) (Triangle.mk A B C)]
variable {R R₁ R₂ : ℝ}
variable [circumradius_abc : Circumradius (Triangle.mk A B C) R]
variable [circumradius_adg : Circumradius (Triangle.mk A D G) R₁]
variable [circumradius_dbf : Circumradius (Triangle.mk D B F) R₂]
variable [ratio_circumradius : (R₁ / Line.length A D) = (R₂ / Line.length D B) ∧ (R / Line.length A B)]

theorem circumcircles_equal_circumcircle_abc : 
  2 * Real.pi * R₁ + 2 * Real.pi * R₂ = 2 * Real.pi * R := 
by
  sorry

end circumcircles_equal_circumcircle_abc_l810_810266


namespace polygon_diagonals_with_restricted_vertices_l810_810484

theorem polygon_diagonals_with_restricted_vertices
  (vertices : ℕ) (non_contributing_vertices : ℕ)
  (h_vertices : vertices = 35)
  (h_non_contributing_vertices : non_contributing_vertices = 5) :
  (vertices - non_contributing_vertices) * (vertices - non_contributing_vertices - 3) / 2 = 405 :=
by {
  sorry
}

end polygon_diagonals_with_restricted_vertices_l810_810484


namespace x_is_perfect_square_l810_810739

theorem x_is_perfect_square {x y : ℕ} (hx : x > 0) (hy : y > 0) (h : (x^2 + y^2 - x) % (2 * x * y) = 0) : ∃ z : ℕ, x = z^2 :=
by
  -- The proof will proceed here
  sorry

end x_is_perfect_square_l810_810739


namespace range_of_a_l810_810193

def f (x : ℝ) : ℝ := x^3 + x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f (x^2 + a) + f (a * x) > 2) → 0 < a ∧ a < 4 := 
by 
  sorry

end range_of_a_l810_810193


namespace square_area_l810_810503

theorem square_area {d : ℝ} (h : d = 12 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 144 ∧ ( ∃ s : ℝ, s = d / Real.sqrt 2 ∧ A = s^2 ) :=
by
  sorry

end square_area_l810_810503


namespace max_x_lcm_15_21_105_l810_810770

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l810_810770


namespace minimal_black_points_l810_810325

open Nat

theorem minimal_black_points (n : ℕ) (hn : n ≥ 3) :
  let N := if (n+1) % 3 = 0 then n-1 else n in
  ∀ (S : Finset (Fin (2 * n - 1))),
  S.card = N →
  ∃ (a b : Fin (2 * n - 1)),
  a ∈ S ∧ b ∈ S ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧
   (b + k) % (2 * n - 1) == a % (2 * n - 1) ∨ 
   (a + k) % (2 * n - 1) == b % (2 * n - 1)) :=
by
  sorry

end minimal_black_points_l810_810325


namespace dice_sum_even_given_odd_product_l810_810570

theorem dice_sum_even_given_odd_product :
  ∀ (die1 die2 die3 die4 die5 : ℕ) (h1 : die1 ∈ {1, 3, 5}) (h2 : die2 ∈ {1, 3, 5}) (h3 : die3 ∈ {1, 3, 5}) (h4 : die4 ∈ {1, 3, 5}) (h5 : die5 ∈ {1, 3, 5}),
    ((die1 * die2 * die3 * die4 * die5) % 2 = 1) →
    ((die1 + die2 + die3 + die4 + die5) % 2 = 0) →
    false :=
begin
  sorry -- Proof needs to be provided here, based on the solution steps.
end

end dice_sum_even_given_odd_product_l810_810570


namespace greatest_value_of_x_l810_810810

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l810_810810


namespace five_student_committees_l810_810143

theorem five_student_committees (n k : ℕ) (hn : n = 8) (hk : k = 5) : 
  nat.choose n k = 56 := by
  rw [hn, hk]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end five_student_committees_l810_810143


namespace greatest_value_of_x_l810_810812

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l810_810812


namespace speed_of_man_in_still_water_l810_810016

variables (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 5 = 36 ∧ (v_m - v_s) * 7 = 22 → v_m = 5.17 :=
by 
  sorry

end speed_of_man_in_still_water_l810_810016


namespace train_distance_train_horsepower_l810_810916

noncomputable theory

-- Definitions based on conditions in the problem.
def train_weight : ℝ := 18 * 10^4 -- in kg
def friction_coefficient : ℝ := 0.005
def max_speed : ℝ := 12 -- in m/s
def time_to_max_speed : ℝ := 60 -- in seconds

-- Proof problems based on questions and correct answers.
theorem train_distance (P : ℝ) (v : ℝ) (t : ℝ) (h : v / t = 0.2) : 
  (1 / 2) * (v / t * t^2) = 360 :=
by sorry

theorem train_horsepower (P : ℝ) (v : ℝ) (t : ℝ) : 
  ((1 / 2) * P * v^2) / (t * 746) = 290 :=
by sorry

-- Applying conditions to the theorems.
example : train_distance train_weight max_speed time_to_max_speed (by norm_num1) := sorry
example : train_horsepower train_weight max_speed time_to_max_speed := sorry

end train_distance_train_horsepower_l810_810916


namespace largest_root_of_g_l810_810566

noncomputable def g (x : ℝ) : ℝ := 12 * x^4 - 17 * x^2 + 5

theorem largest_root_of_g :
  ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → y ≤ x ∧ x = (sqrt 5) / 2 := 
sorry

end largest_root_of_g_l810_810566


namespace C_increases_with_n_l810_810078

variables (n e R r : ℝ)
variables (h_pos_e : e > 0) (h_pos_R : R > 0)
variables (h_pos_r : r > 0) (h_R_nr : R > n * r)
noncomputable def C : ℝ := (e * n) / (R - n * r)

theorem C_increases_with_n (h_pos_e : e > 0) (h_pos_R : R > 0)
(h_pos_r : r > 0) (h_R_nr : R > n * r) (hn1 hn2 : ℝ)
(h_inequality : hn1 < hn2) : 
((e*hn1) / (R - hn1*r)) < ((e*hn2) / (R - hn2*r)) :=
by sorry

end C_increases_with_n_l810_810078


namespace find_second_angle_l810_810758

noncomputable def angle_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem find_second_angle
  (A B C : ℝ)
  (hA : A = 32)
  (hC : C = 2 * A - 12)
  (hB : B = 3 * A)
  (h_sum : angle_in_triangle A B C) :
  B = 96 :=
by sorry

end find_second_angle_l810_810758


namespace minimum_groups_required_l810_810086

noncomputable def group_even :: ℕ → list (list ℕ) -- (This represents the even number groups as discussed)
| 0         := [[]]
| (n + 1)   := ...

noncomputable def group_odd :: ℕ → list (list ℕ) -- (This represents the odd number groups as discussed)
| 0         := [[]]
| (n + 1)   := ...

noncomputable def group_splitting (n : ℕ) (k_even : ℕ) (k_odd: ℕ) : list (list ℕ) :=
  let even_groups := group_even k_even,
      odd_groups := group_odd k_odd in
  even_groups ++ odd_groups

theorem minimum_groups_required : 
  ∀ n, n = 2010 → ∃ k, k = 503 ∧ 
  ∀ (groups : list (list ℕ)), groups = group_splitting n 1005 1005 →
  ∀ g ∈ groups, (∀ x y z ∈ g, gcd x (gcd y z) = 1) := sorry

end minimum_groups_required_l810_810086


namespace volume_of_cube_l810_810680

-- Definition of the problem conditions
variable (a : ℝ) (V : ℝ) -- a: side length, V: volume of the cube
axiom surface_area_150 : 6 * a^2 = 150

-- Statement to prove
theorem volume_of_cube : V = a^3 → V = 125 := by
  assume h : V = a^3
  have ha : a^2 = 150 / 6 := by
    linarith [surface_area_150]
  have a_eq_5 : a = real.sqrt 25 := by
    rw ha
    norm_num
  have a_val : a = 5 := by
    rw real.sqrt_eq_iff_sq_eq
    { linarith }
    { linarith }
  rw a_val at h
  norm_num at h

end volume_of_cube_l810_810680


namespace factorize_expr_l810_810097

theorem factorize_expr (y : ℝ) : 3 * y ^ 2 - 6 * y + 3 = 3 * (y - 1) ^ 2 :=
by
  sorry

end factorize_expr_l810_810097


namespace integer_value_of_a_l810_810887

theorem integer_value_of_a (a x y z k : ℤ) :
  (x = k) ∧ (y = 4 * k) ∧ (z = 5 * k) ∧ (y = 9 * a^2 - 2 * a - 8) ∧ (z = 10 * a + 2) → a = 5 :=
by 
  sorry

end integer_value_of_a_l810_810887


namespace greatest_x_lcm_105_l810_810783

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l810_810783


namespace greatest_x_lcm_105_l810_810786

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l810_810786


namespace paolo_sevilla_birthday_l810_810346

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l810_810346


namespace greatest_x_lcm_l810_810820

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l810_810820


namespace problem_ABC_sum_l810_810984

-- Let A, B, and C be positive integers such that A and C, B and C, and A and B
-- have no common factor greater than 1.
-- If they satisfy the equation A * log_100 5 + B * log_100 4 = C,
-- then we need to prove that A + B + C = 4.

theorem problem_ABC_sum (A B C : ℕ) (h1 : 1 < A ∧ 1 < B ∧ 1 < C)
    (h2 : A.gcd B = 1 ∧ B.gcd C = 1 ∧ A.gcd C = 1)
    (h3 : A * Real.log 5 / Real.log 100 + B * Real.log 4 / Real.log 100 = C) :
    A + B + C = 4 :=
sorry

end problem_ABC_sum_l810_810984


namespace min_abs_val_of_36_power_minus_5_power_l810_810526

theorem min_abs_val_of_36_power_minus_5_power :
  ∃ (m n : ℕ), |(36^m : ℤ) - (5^n : ℤ)| = 11 := sorry

end min_abs_val_of_36_power_minus_5_power_l810_810526


namespace length_DE_eq_15_l810_810263

noncomputable theory

variables {A B C D E : Type} [inner_product_space ℝ E]

-- Define the geometric setup
axiom BC_length : dist B C = 30 * real.sqrt 2
axiom angle_C : angle B C A = real.pi / 4
axiom perpendicular_bisector_D : midpoint B C = D
axiom perpendicular_bisector_E : (orthogonal_projection (affine_span ℝ (set.insert C (set.of_point_coordinates [D]))) A) = E

theorem length_DE_eq_15 : dist D E = 15 :=
by sorry

end length_DE_eq_15_l810_810263


namespace polygon_no_necessary_90_degree_angle_l810_810059

open EuclideanGeometry

noncomputable def concave_polygon_has_no_necessary_90_degree_angle (P : Polygon ℝ) : Prop :=
  (∀ (i j k : ℕ),
    i ≠ j ∧ j ≠ k ∧ k ≠ i →
    are_consecutive (vertices P i) (vertices P j) (vertices P k) →
    is_right_triangle (vertices P i) (vertices P j) (vertices P k)) →
  ¬ (∀ i, exists j, internal_angle (vertices P i) = 90 ∨ internal_angle (vertices P i) = 270)

theorem polygon_no_necessary_90_degree_angle (P : Polygon ℝ) :
  concave_polygon_has_no_necessary_90_degree_angle P :=
by
  sorry

end polygon_no_necessary_90_degree_angle_l810_810059


namespace greatest_x_lcm_l810_810797

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l810_810797


namespace volume_second_cube_l810_810431

open Real

-- Define the ratio of the edges of the cubes
def edge_ratio (a b : ℝ) := a / b = 3 / 1

-- Define the volume of the first cube
def volume_first_cube (a : ℝ) := a^3 = 27

-- Define the edge of the second cube based on the edge of the first cube
def edge_second_cube (a b : ℝ) := a / 3 = b

-- Statement of the problem in Lean 4
theorem volume_second_cube 
  (a b : ℝ) 
  (h_edge_ratio : edge_ratio a b) 
  (h_volume_first : volume_first_cube a) 
  (h_edge_second : edge_second_cube a b) : 
  b^3 = 1 := 
sorry

end volume_second_cube_l810_810431


namespace general_term_formula_find_n_value_l810_810593

noncomputable def sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) : Prop :=
(∀ n, a n = (1/2) * (q^(n-1))) ∧ q < 1 ∧ a 1 = 1/2 ∧ 7 * a 2 = 2 * S 3

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  sequence_general_term a S q → (∀ n, a n = (1/2)^n) :=
by sorry

noncomputable def find_n (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
(∀ n, b n = Real.log 2 (1 - S (n + 1))) ∧ 
(∀ n, (∑ i in Finset.range n, 1 / (b (2 * i + 1) * b (2 * i + 3))) = 5 / 21)

theorem find_n_value (n : ℕ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  find_n b S → n = 20 :=
by sorry

end general_term_formula_find_n_value_l810_810593


namespace maximal_votes_l810_810684

theorem maximal_votes (k : ℕ) (n : ℕ) (citizens : Finset (ℝ × ℝ))
  (h_distinct : ∀ (x y : (ℝ × ℝ)), x ∈ citizens → y ∈ citizens → x ≠ y → dist x y ≠ dist x y)
  (h_size : citizens.card ≥ (k + 1))
  (votes : (ℝ × ℝ) → Finset (ℝ × ℝ) → ℝ × ℝ) :
  ∃ c ∈ citizens, Finset.filter (λ x, votes x citizens = c) citizens = 5 * k := 
sorry

end maximal_votes_l810_810684


namespace maple_leaf_recipients_l810_810724

theorem maple_leaf_recipients :
  let students := 15 in
  let meetings := 5 in
  (students ^ meetings) = 759375 :=
by
  sorry

end maple_leaf_recipients_l810_810724


namespace prize_distribution_correct_l810_810377

-- Definition of the problem conditions
def equally_skilled (A B : Type) : Prop := sorry
def best_of_seven (A B : Type) : Prop := sorry
def win_count_A (games : ℕ) (wins : ℕ) : Prop := sorry
def win_count_B (games : ℕ) (wins : ℕ) : Prop := sorry

-- Tuple representing the probability distribution
structure prize_distribution :=
(A_share : ℕ)
(B_share : ℕ)

-- The main proof problem statement
theorem prize_distribution_correct (A B : Type) (games_won_A : ℕ) (games_won_B : ℕ) (total_prize : ℕ) 
  (h1 : equally_skilled A B) 
  (h2 : best_of_seven A B) 
  (h3 : games_won_A = 3) 
  (h4 : games_won_B = 2) 
  (h5 : total_prize = 10000) : 
  prize_distribution :=
{ A_share := 7500, 
  B_share := 2500 }

-- Proof placeholder
begin
  sorry
end

end prize_distribution_correct_l810_810377


namespace factorization_l810_810563

theorem factorization (y : ℝ) : (16 * y^2 - 48 * y + 36) = (4 * y - 6)^2 :=
by
  sorry

end factorization_l810_810563


namespace fish_caught_300_l810_810759

def fish_caught_at_dawn (F : ℕ) : Prop :=
  (3 * F / 5) = 180

theorem fish_caught_300 : ∃ F, fish_caught_at_dawn F ∧ F = 300 := 
by 
  use 300 
  have h1 : 3 * 300 / 5 = 180 := by norm_num 
  exact ⟨h1, rfl⟩

end fish_caught_300_l810_810759


namespace mass_percentage_of_Cl_in_bleach_l810_810069

-- Definitions based on conditions
def Na_molar_mass : Float := 22.99
def Cl_molar_mass : Float := 35.45
def O_molar_mass : Float := 16.00

def NaClO_molar_mass : Float := Na_molar_mass + Cl_molar_mass + O_molar_mass

def mass_NaClO (mass_na: Float) (mass_cl: Float) (mass_o: Float) : Float :=
  mass_na + mass_cl + mass_o

def mass_of_NaClO : Float := 100.0

def mass_of_Cl_in_NaClO (mass_of_NaClO: Float) : Float :=
  (Cl_molar_mass / NaClO_molar_mass) * mass_of_NaClO

-- Statement to prove
theorem mass_percentage_of_Cl_in_bleach :
  let mass_Cl := mass_of_Cl_in_NaClO mass_of_NaClO
  (mass_Cl / mass_of_NaClO) * 100 = 47.61 :=
by 
  -- Skip the proof
  sorry

end mass_percentage_of_Cl_in_bleach_l810_810069


namespace find_1005th_term_l810_810998

-- Definitions
def is_mean (a : ℕ → ℕ) := ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, a (i+1)) / n = n + 1

-- Theorem statement
theorem find_1005th_term (a : ℕ → ℕ) (h : is_mean a) : a 1005 = 2010 := by
  sorry

end find_1005th_term_l810_810998


namespace harmonic_not_integer_l810_810318

theorem harmonic_not_integer (n : ℕ) (h : n > 1) : 
  ¬ (∃ k : ℤ, (1 + ∑ i in Finset.range n, 1/(i+1)) = k) :=
sorry

end harmonic_not_integer_l810_810318


namespace greatest_x_lcm_105_l810_810795

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l810_810795


namespace exists_trailing_zeros_1971_not_exists_trailing_zeros_1972_l810_810267

def count_trailing_zeros (n : ℕ) : ℕ :=
  (List.iota 1 (Nat.log n 5 + 1)).sum (λ k, n / 5^k)

theorem exists_trailing_zeros_1971
    : ∃ n, count_trailing_zeros n = 1971 :=
sorry

theorem not_exists_trailing_zeros_1972
    : ¬ ∃ n, count_trailing_zeros n = 1972 :=
sorry

end exists_trailing_zeros_1971_not_exists_trailing_zeros_1972_l810_810267


namespace simplify_expression_l810_810863

theorem simplify_expression (x : ℝ) : 
  3 - 5*x - 6*x^2 + 9 + 11*x - 12*x^2 - 15 + 17*x + 18*x^2 - 2*x^3 = -2*x^3 + 23*x - 3 :=
by
  sorry

end simplify_expression_l810_810863


namespace smallest_positive_period_interval_of_monotonic_increase_number_of_zeros_l810_810205
noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x / 2 + Real.pi / 6) + 1 / 2

noncomputable def g (x : ℝ) : ℝ :=
  Real.sin (x / 2 - Real.pi / 6) + 1 / 2

theorem smallest_positive_period :
  ∀ ⦃T⦄, (∀ x, f (x + T) = f x) ↔ T = 4 * Real.pi := sorry

theorem interval_of_monotonic_increase (k : ℤ) :
  ∀ x, x ∈ Set.Icc (4 * k * Real.pi - 4 * Real.pi / 3) (4 * k * Real.pi + 2 * Real.pi / 3) 
  → (Real.deriv f x) > 0 := sorry

theorem number_of_zeros (k : ℝ) :
  ∀ x, x ∈ Set.Icc 0 (7 * Real.pi / 3) →
  if k ∉ Set.Icc 0 (3/2 : ℝ) then g(x) - k = 0 ↔ 0 
  else if 0 ≤ k ∧ k < 1 then g(x) - k = 0 ↔ 2 
  else g(x) - k = 0 ↔ 1 := sorry

end smallest_positive_period_interval_of_monotonic_increase_number_of_zeros_l810_810205


namespace smallest_y_l810_810642

def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  -- Assume this function returns the prime factorization of n in the form [(p1, e1), (p2, e2), ...]
  sorry

def is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n

theorem smallest_y (y : ℕ) :
  (∃ k : ℕ, (2^1 * 3^2 * 5^1 * 7^2 * y = k * k)) ∧ (∀ z : ℕ, (z > 0) → (∃ k : ℕ, (2^1 * 3^2 * 5^1 * 7^2 * z = k * k)) -> y ≤ z) ↔ y = 10 :=
begin
  sorry
end

end smallest_y_l810_810642


namespace darts_partition_count_l810_810066

theorem darts_partition_count : 
  (card {l : List ℕ | l.sum = 5 ∧ l.length = 4 ∧ l.sorted (≥)}) = 6 :=
by
  sorry

end darts_partition_count_l810_810066


namespace angle_bisector_divides_DE_ratio_l810_810728

variable (A B C D E O : Type)
variables [RightTriangle A B C] [SquareOnHypotenuseOutward A B D E]
variables (AC BC : ℝ)
variables (AC_1 : AC = 1) (BC_3 : BC = 3)
variable (isHypotenuseConsistent : hypotenuse A B C = sqrt 10)
variable (DE : segment D E)

theorem angle_bisector_divides_DE_ratio:
  divides_in_ratio (angle_bisector A C B) DE (1 : ℝ) (3 : ℝ) :=
sorry

end angle_bisector_divides_DE_ratio_l810_810728


namespace sales_discount_l810_810907

theorem sales_discount
  (P N : ℝ)  -- original price and number of items sold
  (H1 : (1 - D / 100) * 1.3 = 1.17) -- condition when discount D is applied
  (D : ℝ)  -- sales discount percentage
  : D = 10 := by
  sorry

end sales_discount_l810_810907


namespace divisible_by_101_l810_810460

theorem divisible_by_101 (k : ℕ) (hk : k > 0) : 
  let n := k * 101 - 1 
  in 101 ∣ n^3 + 1 ∧ 101 ∣ n^2 - 1 :=
by
  let n := k * 101 - 1
  have h₁ : 101 ∣ n^3 + 1 := sorry
  have h₂ : 101 ∣ n^2 - 1 := sorry
  exact ⟨h₁, h₂⟩

end divisible_by_101_l810_810460


namespace burmese_pythons_required_l810_810225

theorem burmese_pythons_required (single_python_rate : ℕ) (total_alligators : ℕ) (total_weeks : ℕ) (required_pythons : ℕ) :
  single_python_rate = 1 →
  total_alligators = 15 →
  total_weeks = 3 →
  required_pythons = total_alligators / total_weeks →
  required_pythons = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at *
  simp at h4
  sorry

end burmese_pythons_required_l810_810225


namespace find_1005th_term_l810_810999

-- Definitions
def is_mean (a : ℕ → ℕ) := ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, a (i+1)) / n = n + 1

-- Theorem statement
theorem find_1005th_term (a : ℕ → ℕ) (h : is_mean a) : a 1005 = 2010 := by
  sorry

end find_1005th_term_l810_810999


namespace J_3_15_10_eq_68_over_15_l810_810575

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_15_10_eq_68_over_15 : J 3 15 10 = 68 / 15 := by
  sorry

end J_3_15_10_eq_68_over_15_l810_810575


namespace ways_to_arrange_6_people_l810_810439

-- Definitions based on the problem conditions
def person_A_can_be_head_or_tail (row : List ℕ) (A : ℕ) : Prop :=
  row.head? = some A ∨ row.last? = some A

def persons_B_and_C_adjacent (row : List ℕ) (B C : ℕ) : Prop :=
  let indices := List.indexes row
  let b_index := List.indexOf B row
  let c_index := List.indexOf C row
  (b_index, c_index) ∈ [(c_index - 1, c_index), (c_index + 1, c_index)]

-- Main theorem statement
theorem ways_to_arrange_6_people (row : List ℕ) (A B C D E F : ℕ) :
  person_A_can_be_head_or_tail row A ∧ persons_B_and_C_adjacent row B C →
  List.permutations [A, B, C, D, E, F].length = 96 :=
  sorry

end ways_to_arrange_6_people_l810_810439


namespace fraction_identity_l810_810214

theorem fraction_identity
  (m : ℝ)
  (h : (m - 1) / m = 3) : (m^2 + 1) / m^2 = 5 :=
by
  sorry

end fraction_identity_l810_810214


namespace max_plus_min_l810_810184

noncomputable def f (x : ℝ) : ℝ := (3 * Real.exp (Real.abs x) - x * Real.cos x) / Real.exp (Real.abs x)

theorem max_plus_min (p q : ℝ) (h₁ : ∃ x ∈ Set.Icc (-(Real.pi / 2)) (Real.pi / 2), f x = p)
  (h₂ : ∃ y ∈ Set.Icc (-(Real.pi / 2)) (Real.pi / 2), f y = q) :
  p + q = 6 :=
sorry

end max_plus_min_l810_810184


namespace matrix_product_example_l810_810074

def matrix_mult {α : Type*} [semiring α] {m n p : Type*}
  [fintype m] [fintype n] [fintype p] (A : matrix m n α) (B : matrix n p α) : matrix m p α :=
λ i j, ∑ k, A i k * B k j

theorem matrix_product_example :
  matrix_mult
    (λ i j, if (i, j) = (0, 0) then 3 else if (i, j) = (0, 1) then -2
            else if (i, j) = (1, 0) then -1 else if (i, j) = (1, 1) then 5 else 0)
    (λ i j, if (i, j) = (0, 0) then 4 else if (i, j) = (1, 0) then -3 else 0) =
    (λ i j, if (i, j) = (0, 0) then 18 else if (i, j) = (1, 0) then -19 else 0) :=
by
  sorry

end matrix_product_example_l810_810074


namespace eval_fraction_of_factorials_l810_810094

theorem eval_fraction_of_factorials : ((factorial (factorial 4)) / (factorial 4)) = (factorial 23) := by
  sorry

end eval_fraction_of_factorials_l810_810094


namespace sum_remainders_mod_13_l810_810117

theorem sum_remainders_mod_13 :
  ∀ (a b c d e : ℕ),
  a % 13 = 3 →
  b % 13 = 5 →
  c % 13 = 7 →
  d % 13 = 9 →
  e % 13 = 11 →
  (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end sum_remainders_mod_13_l810_810117


namespace tom_walks_distance_l810_810673

theorem tom_walks_distance (t : ℝ) (d : ℝ) :
  t = 15 ∧ d = (1 / 18) * t → d ≈ 0.8 :=
by
  sorry

end tom_walks_distance_l810_810673


namespace difference_of_consecutive_9_digit_palindromes_l810_810292

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

def is_9_digit_palindrome (n : ℕ) : Prop :=
  is_palindrome n ∧ 100000000 ≤ n ∧ n < 1000000000

theorem difference_of_consecutive_9_digit_palindromes (M N : ℕ)
  (hM : is_9_digit_palindrome M)
  (hN : is_9_digit_palindrome N)
  (hNM : N > M)
  (h_consec : ∀ k, M < k ∧ k < N → ¬ is_9_digit_palindrome k) :
  N - M = 100000011 := by
  sorry

end difference_of_consecutive_9_digit_palindromes_l810_810292


namespace select_interpreters_l810_810649

structure InterpreterGroup (α : Type) :=
  (speaksJapanese : α → Prop)
  (speaksMalay : α → Prop)
  (speaksFarsi : α → Prop)
  (japanese_count : ∀ s : set α, speaksJapanese s → s.card = 24)
  (malay_count : ∀ s : set α, speaksMalay s → s.card = 24)
  (farsi_count : ∀ s : set α, speaksFarsi s → s.card = 24)

def exists_desired_subgroup (α : Type) [Fintype α] 
  (group : InterpreterGroup α) : Prop :=
  ∃ (s_j s_m s_f : set α),
    (∀ a ∈ s_j, group.speaksJapanese a) ∧ s_j.card = 12 ∧
    (∀ a ∈ s_m, group.speaksMalay a) ∧ s_m.card = 12 ∧
    (∀ a ∈ s_f, group.speaksFarsi a) ∧ s_f.card = 12

theorem select_interpreters (α : Type) [Fintype α] 
  (group : InterpreterGroup α) : exists_desired_subgroup α group :=
sorry

end select_interpreters_l810_810649


namespace identity_sum_form_l810_810738

theorem identity_sum_form :
  ∑ k in Finset.range 10, (2 * (k + 1)) / ((x : ℝ)^2 - ((k + 1)^2)) = 
    11 * ∑ k in Finset.range 10, 1 / ((x - (k + 1)) * (x + (11 - (k + 1)))) :=
  sorry

end identity_sum_form_l810_810738


namespace distinct_floor_values_l810_810836

theorem distinct_floor_values
  (n : ℕ) (h₀ : n > 3) 
  (a : Fin n → ℕ) 
  (h₁ : ∀ i, a i < (n - 1)!)
  (h₂ : Function.Injective a) :
  ∃ (i j k l : Fin n), i ≠ j ∧ k ≠ l ∧
  a i > a j ∧ a k > a l ∧ 
  (⌊(a i : ℝ) / a j⌋ : ℤ) = ⌊(a k : ℝ) / a l⌋ := 
sorry

end distinct_floor_values_l810_810836


namespace sufficient_but_not_necessary_condition_l810_810827

theorem sufficient_but_not_necessary_condition :
  ∀ x : ℝ, (x = 2 → (x - 2) * (x + 5) = 0) ∧ ((x - 2) * (x + 5) = 0 → x = 2 → False) :=
by
  intros x,
  split,
  {
    intro h,
    rw h,
    ring,
  },
  {
    intros h1 h2,
    simp at h1,
    cases h1 with h3 h4,
    { contradiction },
    { norm_num at h4 },
  }

end sufficient_but_not_necessary_condition_l810_810827


namespace cost_of_each_sale_puppy_l810_810532

-- Conditions
def total_cost (total: ℚ) : Prop := total = 800
def non_sale_puppy_cost (cost: ℚ) : Prop := cost = 175
def num_puppies (num: ℕ) : Prop := num = 5

-- Question to Prove
theorem cost_of_each_sale_puppy (total cost : ℚ) (num: ℕ):
  total_cost total →
  non_sale_puppy_cost cost →
  num_puppies num →
  (total - 2 * cost) / (num - 2) = 150 := 
sorry

end cost_of_each_sale_puppy_l810_810532


namespace bouquet_cost_l810_810533

theorem bouquet_cost (cost_18_roses : ℝ) (num_roses_18 : ℕ) (num_roses_36 : ℕ) 
(discount : ℝ) (price_directly_proportional : Prop)
(h1 : cost_18_roses = 30)
(h2 : num_roses_18 = 18)
(h3 : num_roses_36 = 36)
(h4 : discount = 0.10)
(h5 : price_directly_proportional) :
  let proportionality_factor := cost_18_roses / num_roses_18,
      initial_cost_36_roses := num_roses_36 * proportionality_factor,
      discounted_cost := initial_cost_36_roses * (1 - discount)
  in discounted_cost = 54 :=
by
  sorry

end bouquet_cost_l810_810533


namespace friends_attended_birthday_l810_810356

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l810_810356


namespace decreasing_pairs_count_l810_810978

variable {a : Fin 2008 → ℕ}
variable (increas_pairs : Fin 2007 → Prop)
variable (decreas_pairs : Fin 2007 → Prop)

def valid_increasing_pairs : Prop :=
  ∃ f : Fin 2007 → Prop, (∀ i, f i ↔ a i < a (i + 1)) ∧ (Fintype.card {i | f i} = 103)

def valid_decreasing_pairs : Prop :=
  ∃ g : Fin 2007 → Prop, (∀ i, g i ↔ a i > a (i + 1)) 

theorem decreasing_pairs_count :
  ∀ (a : Fin 2008 → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ 5) →
  valid_increasing_pairs a increas_pairs →
  ∃ (g : Fin 2007 → Prop), valid_decreasing_pairs a g ∧ (Fintype.card {i | g i} ≥ 25) := sorry

end decreasing_pairs_count_l810_810978


namespace avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l810_810932

open Real
open List

-- Conditions
def x_vals : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y_vals : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
def sum_x : ℝ := 0.6
def sum_y : ℝ := 3.9
def sum_x_squared : ℝ := 0.038
def sum_y_squared : ℝ := 1.6158
def sum_xy : ℝ := 0.2474
def total_root_area : ℝ := 186

-- Proof problems
theorem avg_root_area : (List.sum x_vals / 10) = 0.06 := by
  sorry

theorem avg_volume : (List.sum y_vals / 10) = 0.39 := by
  sorry

theorem correlation_coefficient : 
  let mean_x := List.sum x_vals / 10;
  let mean_y := List.sum y_vals / 10;
  let numerator := List.sum (List.zipWith (λ x y => (x - mean_x) * (y - mean_y)) x_vals y_vals);
  let denominator := sqrt ((List.sum (List.map (λ x => (x - mean_x) ^ 2) x_vals)) * (List.sum (List.map (λ y => (y - mean_y) ^ 2) y_vals)));
  (numerator / denominator) = 0.97 := by 
  sorry

theorem total_volume_estimate : 
  let avg_x := sum_x / 10;
  let avg_y := sum_y / 10;
  (avg_y / avg_x) * total_root_area = 1209 := by
  sorry

end avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l810_810932


namespace sushil_marks_ratio_l810_810393

theorem sushil_marks_ratio
  (E M Science : ℕ)
  (h1 : E + M + Science = 170)
  (h2 : E = M / 4)
  (h3 : Science = 17) :
  E = 31 :=
by
  sorry

end sushil_marks_ratio_l810_810393


namespace smallest_boxes_l810_810880

-- Definitions based on the conditions:
def divisible_by (n d : Nat) : Prop := ∃ k, n = d * k

-- The statement to be proved:
theorem smallest_boxes (n : Nat) : 
  divisible_by n 5 ∧ divisible_by n 24 -> n = 120 :=
by sorry

end smallest_boxes_l810_810880


namespace square_area_l810_810506

theorem square_area {d : ℝ} (h : d = 12 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 144 ∧ ( ∃ s : ℝ, s = d / Real.sqrt 2 ∧ A = s^2 ) :=
by
  sorry

end square_area_l810_810506


namespace movies_left_to_watch_l810_810433

theorem movies_left_to_watch (total_movies : ℕ) (movies_watched : ℕ) : total_movies = 17 ∧ movies_watched = 7 → (total_movies - movies_watched) = 10 :=
by
  sorry

end movies_left_to_watch_l810_810433


namespace greatest_value_of_x_l810_810806

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l810_810806


namespace avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l810_810933

open Real
open List

-- Conditions
def x_vals : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y_vals : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
def sum_x : ℝ := 0.6
def sum_y : ℝ := 3.9
def sum_x_squared : ℝ := 0.038
def sum_y_squared : ℝ := 1.6158
def sum_xy : ℝ := 0.2474
def total_root_area : ℝ := 186

-- Proof problems
theorem avg_root_area : (List.sum x_vals / 10) = 0.06 := by
  sorry

theorem avg_volume : (List.sum y_vals / 10) = 0.39 := by
  sorry

theorem correlation_coefficient : 
  let mean_x := List.sum x_vals / 10;
  let mean_y := List.sum y_vals / 10;
  let numerator := List.sum (List.zipWith (λ x y => (x - mean_x) * (y - mean_y)) x_vals y_vals);
  let denominator := sqrt ((List.sum (List.map (λ x => (x - mean_x) ^ 2) x_vals)) * (List.sum (List.map (λ y => (y - mean_y) ^ 2) y_vals)));
  (numerator / denominator) = 0.97 := by 
  sorry

theorem total_volume_estimate : 
  let avg_x := sum_x / 10;
  let avg_y := sum_y / 10;
  (avg_y / avg_x) * total_root_area = 1209 := by
  sorry

end avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l810_810933


namespace birthday_friends_count_l810_810373

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l810_810373


namespace paidAmount_Y_l810_810854

theorem paidAmount_Y (X Y : ℝ) (h1 : X + Y = 638) (h2 : X = 1.2 * Y) : Y = 290 :=
by
  sorry

end paidAmount_Y_l810_810854


namespace jack_buttons_l810_810277

theorem jack_buttons :
  ∀ (shirts_per_kid kids buttons_per_shirt : ℕ),
  shirts_per_kid = 3 →
  kids = 3 →
  buttons_per_shirt = 7 →
  (shirts_per_kid * kids * buttons_per_shirt) = 63 :=
by
  intros shirts_per_kid kids buttons_per_shirt h1 h2 h3
  rw [h1, h2, h3]
  calc
    3 * 3 * 7 = 9 * 7 : by rw mul_assoc
            ... = 63   : by norm_num

end jack_buttons_l810_810277


namespace find_ab_l810_810315

noncomputable def f (x : ℝ) : ℝ := |Real.log2 x|

theorem find_ab (a b : ℝ) (h1 : f (a + 1) = f (b + 2)) (h2 : f (10 * a + 6 * b + 22) = 4) (h3 : a < b) : 
  a * b = 2 / 15 := 
sorry

end find_ab_l810_810315


namespace sum_squares_of_roots_l810_810541

theorem sum_squares_of_roots :
  (∀ y : ℝ, y ≥ 0 → y^3 - 8*y^2 + 9*y - 2 = 0) → (∑ y in {r, s, t}, r^2) = 46 :=
by
  -- Define the polynomial and conditions
  let f (y : ℝ) := y^3 - 8*y^2 + 9*y - 2
  assume h : ∀ y : ℝ, y ≥ 0 → f y = 0
  sorry

end sum_squares_of_roots_l810_810541


namespace polynomial_int_values_l810_810456

variable {R : Type*} [CommRing R] [IsDomain R]

theorem polynomial_int_values (f : Polynomial R) (n : ℕ)
  (h : ∀ k : ℕ, k < n + 1 → (f.eval k).isInt) :
  ∀ x : ℤ, (f.eval x).isInt :=
sorry

end polynomial_int_values_l810_810456


namespace monthly_expenses_last_month_l810_810740

def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.10
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.20

def commission := total_sales * commission_rate
def total_earnings := basic_salary + commission
def savings := total_earnings * savings_rate
def monthly_expenses := total_earnings - savings

theorem monthly_expenses_last_month :
  monthly_expenses = 2888 := 
by sorry

end monthly_expenses_last_month_l810_810740


namespace resting_time_is_thirty_l810_810723

-- Defining the conditions as Lean 4 definitions
def speed := 10 -- miles per hour
def time_first_part := 30 -- minutes
def distance_second_part := 15 -- miles
def distance_third_part := 20 -- miles
def total_time := 270 -- minutes

-- Function to convert hours to minutes
def hours_to_minutes (h : ℕ) : ℕ := h * 60

-- Problem statement in Lean 4: Proving the resting time is 30 minutes
theorem resting_time_is_thirty :
  let distance_first := speed * (time_first_part / 60)
  let time_second_part := (distance_second_part / speed) * 60
  let time_third_part := (distance_third_part / speed) * 60
  let times_sum := time_first_part + time_second_part + time_third_part
  total_time = times_sum + 30 := 
  sorry

end resting_time_is_thirty_l810_810723


namespace line_intersects_y_axis_at_0_2_l810_810953

theorem line_intersects_y_axis_at_0_2 :
  ∃ y : ℝ, (2, 8) ≠ (4, 14) ∧ ∀ x: ℝ, (3 * x + y = 2) ∧ x = 0 → y = 2 :=
by
  sorry

end line_intersects_y_axis_at_0_2_l810_810953


namespace friends_at_birthday_l810_810365

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l810_810365


namespace greatest_x_lcm_105_l810_810789

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l810_810789


namespace necessary_but_not_sufficient_for_ln_l810_810886

theorem necessary_but_not_sufficient_for_ln (a : ℝ) : (a < 1 → ln a < 0) ∧ (ln a < 0 → 0 < a) ∧ ¬ (a < 1 → ln a < 0) :=
by
  sorry

end necessary_but_not_sufficient_for_ln_l810_810886


namespace problem_l810_810187

noncomputable def ellipse_foci : Set ℝ → Set ℝ → (Set ℝ × Set ℝ) := sorry
noncomputable def intersection (line ellipse : ℝ → ℝ) : Set (ℝ × ℝ) := sorry
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem problem (line : ℝ → ℝ) (ellipse : ℝ → ℝ) (F : ℝ × ℝ) (A B : ℝ × ℝ) :
  (ellipse = (λ x y, (x^2 / 4) + (y^2 / 3) - 1))
  ∧ (F = (-1, 0))
  ∧ (line = (λ x, sqrt(3) * (x + 1)))
  ∧ (A ∈ intersection line ellipse)
  ∧ (B ∈ intersection line ellipse)
  ∧ (A ≠ B)
  → (1 / (distance A F)) + (1 / (distance B F)) = 4 / 3 :=
by sorry

end problem_l810_810187


namespace cost_per_person_l810_810679

theorem cost_per_person (total_cost : ℕ) (num_people : ℕ) (expected_cost_per_person : ℕ) : 
  total_cost = 12100 → num_people = 11 → expected_cost_per_person = 1100 → 
  total_cost / num_people = expected_cost_per_person :=
by
  -- Start by assuming the conditions
  intros h_total_cost h_num_people h_expected_cost_per_person
  -- Simplify the division based on these conditions
  rw [h_total_cost, h_num_people, h_expected_cost_per_person]
  -- Check division with natural numbers
  exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num) sorry

end cost_per_person_l810_810679


namespace distance_to_airport_l810_810523

-- Definitions based on the conditions
def initial_speed := 40 -- in miles per hour
def increased_speed := 60 -- in miles per hour
def initial_distance := 40 -- miles, since he initially covers 40 miles
def stop_time := 0.25 -- hours, since he stops for 15 minutes
def early_arrival := 0.25 -- hours, since he arrives 15 minutes early

-- Derived conditions to prove the total distance
theorem distance_to_airport (t : ℝ) (d : ℝ) 
  (h1 : d = initial_speed * (t + 1 + stop_time))
  (h2 : d - initial_distance = increased_speed * (t - early_arrival)) : d = 190 := 
by 
  sorry

end distance_to_airport_l810_810523


namespace range_of_S_exists_l810_810607

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x ^ 2 / 2 + y ^ 2 = 1

noncomputable def is_symmetric_about_y (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = Q.2

noncomputable def lhs_focus : ℝ × ℝ := (-1, 0)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  ( (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 ) ^ (1 / 2)

noncomputable def pf1_qf1 (P Q: ℝ × ℝ) (lhs_focus: ℝ × ℝ) :=
  distance P lhs_focus + distance Q lhs_focus = 2 * real.sqrt 2

noncomputable def line_through_focus (k : ℝ) (lhs_focus : ℝ × ℝ) (A B : ℝ × ℝ) :=
  A.2 = k * (A.1 + 1) ∧ B.2 = k * (B.1 + 1) -- line equation y = kx + k

noncomputable def AF1_eq_lambda_FB1 (A B lhs_focus : ℝ × ℝ) (λ : ℝ) :=
  (lhs_focus.1 - A.1, lhs_focus.2 - A.2) = (λ * (B.1 + lhs_focus.1), λ * B.2)

noncomputable def point_D : ℝ × ℝ := (-1, -1)

noncomputable def area_triangle (A B D : ℝ × ℝ) : ℝ :=
  abs (A.1 * (B.2 - D.2) + B.1 * (D.2 - A.2) + D.1 * (A.2 - B.2)) / 2

noncomputable def range_S (A B D lhs_focus: ℝ × ℝ) (S : ℝ) (k λ : ℝ) : Prop :=
  3 ≤ λ ∧ λ ≤ 2 + real.sqrt 3 ∧
  λ ≥ 0 ∧ A.2 = -λ * B.2 ∧
  S = area_triangle A B point_D ∧
  distance lhs_focus (0,0) = 1 ∧
  line_through_focus k lhs_focus A B ∧
  (real.sqrt 2 / 2) ≤ k ∧ k ≤ 1 ∧
  (2/3) ≤ S ∧ S ≤ (real.sqrt 3)/2

theorem range_of_S_exists : 
  ∃ (S : ℝ), 
    ∀ (P Q A B lhs_focus : ℝ × ℝ) (k λ : ℝ), 
      ellipse_equation P.1 P.2 ∧
      ellipse_equation Q.1 Q.2 ∧
      is_symmetric_about_y P Q ∧ 
      pf1_qf1 P Q lhs_focus ∧ 
      line_through_focus k lhs_focus A B ∧ 
      AF1_eq_lambda_FB1 A B lhs_focus λ ∧ 
      range_S A B point_D lhs_focus S k λ :=
begin
  sorry
end

end range_of_S_exists_l810_810607


namespace julia_fascinating_last_digits_l810_810725

theorem julia_fascinating_last_digits : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, (∃ y : ℕ, x = 10 * y) → x % 10 < 10) :=
by
  sorry

end julia_fascinating_last_digits_l810_810725


namespace sqrt_3_between_inequalities_l810_810228

theorem sqrt_3_between_inequalities (n : ℕ) (h1 : 1 + (3 : ℝ) / (n + 1) < Real.sqrt 3) (h2 : Real.sqrt 3 < 1 + (3 : ℝ) / n) : n = 4 := 
sorry

end sqrt_3_between_inequalities_l810_810228


namespace work_completion_problem_l810_810898

theorem work_completion_problem :
  (∃ x : ℕ, 9 * (1 / 45 + 1 / x) + 23 * (1 / x) = 1) → x = 40 :=
sorry

end work_completion_problem_l810_810898


namespace prove_visual_transformation_l810_810457

noncomputable def spinning_coin_transformation (coin : Type) (tabletop : Type) :=
  (stationary_coin_2D : coin → Prop) →
  (rapid_spinning_blurs_edges : (coin → coin) → Prop) →
  (transformation_to_3D_perception_eq_sphere : (coin → tabletop) → Prop)

axiom stationary_coin_2D (c : coin) : c -- stationary coin is a 2D object

axiom rapid_spinning_blurs_edges (spin : coin → coin) : 
  -- rapid spinning motion blurs the edges of the coin

axiom visual_transformation (c : coin) (t : tabletop) :
  rapid_spinning_blurs_edges spin → (transformation_to_3D_perception_eq_sphere t)

theorem prove_visual_transformation (coin : Type) (tabletop : Type) :
  (stationary_coin_2D : coin → Prop) →
  (rapid_spinning_blurs_edges : (coin → coin) → Prop) →
  (transformation_to_3D_perception_eq_sphere : (coin → tabletop) → Prop) :=
begin
  sorry
end

end prove_visual_transformation_l810_810457


namespace sqrt_div_equality_l810_810562

noncomputable def sqrt_div (x y : ℝ) : ℝ := Real.sqrt x / Real.sqrt y

theorem sqrt_div_equality (x y : ℝ)
  (h : ( ( (1/3 : ℝ) ^ 2 + (1/4 : ℝ) ^ 2 ) / ( (1/5 : ℝ) ^ 2 + (1/6 : ℝ) ^ 2 ) = 25 * x / (73 * y) )) :
  sqrt_div x y = 5 / 2 :=
sorry

end sqrt_div_equality_l810_810562


namespace female_athletes_drawn_is_7_l810_810517

-- Given conditions as definitions
def male_athletes := 64
def female_athletes := 56
def drawn_male_athletes := 8

-- The function that represents the equation in stratified sampling
def stratified_sampling_eq (x : Nat) : Prop :=
  (drawn_male_athletes : ℚ) / (male_athletes) = (x : ℚ) / (female_athletes)

-- The theorem which states that the solution to the problem is x = 7
theorem female_athletes_drawn_is_7 : ∃ x : Nat, stratified_sampling_eq x ∧ x = 7 :=
by
  sorry

end female_athletes_drawn_is_7_l810_810517


namespace circular_garden_radius_l810_810876

theorem circular_garden_radius (r : ℝ) (h : 2 * Real.pi * r = (1 / 8) * Real.pi * r^2) : r = 16 :=
sorry

end circular_garden_radius_l810_810876


namespace total_money_together_is_l810_810067

def Sam_has : ℚ := 750.50
def Billy_has (S : ℚ) : ℚ := 4.5 * S - 345.25
def Lila_has (B S : ℚ) : ℚ := 2.25 * (B - S)
def Total_money (S B L : ℚ) : ℚ := S + B + L

theorem total_money_together_is :
  Total_money Sam_has (Billy_has Sam_has) (Lila_has (Billy_has Sam_has) Sam_has) = 8915.88 :=
by sorry

end total_money_together_is_l810_810067


namespace square_area_from_diagonal_l810_810500

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  (let s := d / Real.sqrt 2 in s * s) = 144 := by
sorry

end square_area_from_diagonal_l810_810500


namespace crease_length_l810_810915

/-- 
  Given a right triangle with side lengths 5, 12, and 13 inches.
  When the triangle is folded such that point A falls on point C,
  the length of the crease is 5 inches.
-/

theorem crease_length {a b c : ℕ}
  (h_triangle : a = 5 ∧ b = 12 ∧ c = 13)
  (h_pythagorean : a^2 + b^2 = c^2) :
  let crease := 5 in
  crease = 5 :=
by 
{ sorry }

end crease_length_l810_810915


namespace square_area_from_diagonal_l810_810510

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) : ∃ A : ℝ, A = 144 :=
by
  let s := d / Real.sqrt 2
  have s_eq : s = 12 := by
    rw [h]
    field_simp
    norm_num
  use s * s
  rw [s_eq]
  norm_num
  sorry

end square_area_from_diagonal_l810_810510


namespace series_sum_correct_l810_810003

-- Define the general term in the series
def term (n : ℕ) : ℚ := n / (2 ^ n)

-- Define the finite sum of terms in the series
noncomputable def series_sum : ℚ := (∑ n in Finset.range 10, term (n + 1))

-- Problem statement in Lean 4
theorem series_sum_correct : series_sum = 509 / 256 :=
by
  sorry

end series_sum_correct_l810_810003


namespace roof_difference_l810_810881

theorem roof_difference (W L : ℝ) (h1 : L = 5 * W) (h2 : L * W = 1024) : |L - W| ≈ 57.24 := by
  sorry

end roof_difference_l810_810881


namespace prime_count_between_10_and_30_l810_810631

-- Define the range of interest
def in_range (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 30

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime numbers in the range
def primes_in_range : List ℕ := (List.filter (λ n => is_prime n) (List.range' 11 19))

-- Statement to be proven
theorem prime_count_between_10_and_30 : primes_in_range.length = 6 := by
  sorry

end prime_count_between_10_and_30_l810_810631


namespace count_real_rooted_quadratics_l810_810544

noncomputable def count_valid_equations : ℕ := 
  let S := {1, 2, 3, 4, 5, 7}
  let valid (b c : ℤ) : Prop := b ∈ S ∧ c ∈ S ∧ b^2 - 4 * c ≥ 0
  let valid_pairs := { (b, c) | b ∈ S ∧ c ∈ S ∧ b^2 - 4 * c ≥ 0 }
  valid_pairs.toFinset.card
  
theorem count_real_rooted_quadratics : count_valid_equations = 18 := 
  by sorry

end count_real_rooted_quadratics_l810_810544


namespace length_of_longer_side_l810_810903

noncomputable def radius : ℝ := 6
noncomputable def area_circle : ℝ := π * radius ^ 2
noncomputable def area_rectangle : ℝ := 3 * area_circle
noncomputable def shorter_side : ℝ := 2 * radius

theorem length_of_longer_side :
  (∃ L : ℝ, shorter_side * L = area_rectangle ∧ L = 9 * π) :=
begin
  use (9 * π),
  split,
  {
    show shorter_side * (9 * π) = area_rectangle,
    calc
      shorter_side * (9 * π) = (2 * radius) * (9 * π)  : by refl
      ...                     = 2 * 6 * 9 * π          : by {unfold radius, linarith}
      ...                     = 108 * π                : by linarith
      ...                     = area_rectangle         : by {unfold area_rectangle, unfold area_circle, linarith},
  },
  {
    refl,
  }
end

end length_of_longer_side_l810_810903


namespace n_squared_divisible_by_12_l810_810211

theorem n_squared_divisible_by_12 (n : ℕ) : 12 ∣ n^2 * (n^2 - 1) :=
  sorry

end n_squared_divisible_by_12_l810_810211


namespace intersection_points_l810_810389

theorem intersection_points (g : ℝ → ℝ) (hg_inv : Function.Injective g) : 
  ∃ n, n = 3 ∧ ∀ x, g (x^3) = g (x^5) ↔ x = 0 ∨ x = 1 ∨ x = -1 :=
by {
  sorry
}

end intersection_points_l810_810389


namespace triangle_area_l810_810864

theorem triangle_area (A B C : Real × Real)
  (hA : A = (0, 0))
  (hB : B = (4, 0))
  (hC : C = (4, 10)) :
  let base := (B.1 - A.1).abs
  let height := (C.2 - B.2).abs
  (1 / 2) * base * height = 20 :=
by
  intros
  rw [hA, hB, hC]
  simp  -- Simplifies to base = 4 and height = 10
  sorry  -- The proof is omitted as requested

end triangle_area_l810_810864


namespace S2_lt_S1_and_S1_lt_S3_l810_810160

noncomputable def S1 := ∫ x in (1:ℝ)..2, x^2
noncomputable def S2 := ∫ x in (1:ℝ)..2, 1/x
noncomputable def S3 := ∫ x in (1:ℝ)..2, Real.exp x

theorem S2_lt_S1_and_S1_lt_S3 :
  S2 < S1 ∧ S1 < S3 :=
by
  have S1_val : S1 = 7 / 3 := by
    simp [S1]
    norm_num
  have S2_val : S2 = Real.log 2 := by
    simp [S2]
    norm_num
  have S3_val : S3 = Real.exp 2 - Real.exp 1 := by
    simp [S3]
    norm_num
  rw [S1_val, S2_val, S3_val]
  split
  norm_num
  refine Real.log_pos _ (by norm_num)
  norm_num
  refine Real.exp_pos _
  norm_num
sorry

end S2_lt_S1_and_S1_lt_S3_l810_810160


namespace altitude_product_difference_eq_zero_l810_810264

variables (A B C P Q H : Type*) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited H]
variable {HP HQ BP PC AQ QC AH BH : ℝ}

-- Given conditions
axiom altitude_intersects_at_H : true
axiom HP_val : HP = 3
axiom HQ_val : HQ = 7

-- Statement to prove
theorem altitude_product_difference_eq_zero (h_BP_PC : BP * PC = 3 / (AH + 3))
                                           (h_AQ_QC : AQ * QC = 7 / (BH + 7))
                                           (h_AH_BQ_ratio : AH / BH = 3 / 7) :
  (BP * PC) - (AQ * QC) = 0 :=
by sorry

end altitude_product_difference_eq_zero_l810_810264


namespace ball_hits_ground_time_l810_810398

noncomputable def height (t : ℝ) : ℝ := -16 * t^2 - 20 * t + 70

theorem ball_hits_ground_time :
  ∃ (t : ℝ), t ≥ 0 ∧ abs (t - 1.56) < 0.01 ∧ height t = 0 :=
begin
  sorry
end

end ball_hits_ground_time_l810_810398


namespace liars_are_C_and_D_l810_810057
open Classical 

-- We define inhabitants and their statements
inductive Inhabitant
| A | B | C | D

open Inhabitant

axiom is_liar : Inhabitant → Prop

-- Statements by the inhabitants:
-- A: "At least one of us is a liar."
-- B: "At least two of us are liars."
-- C: "At least three of us are liars."
-- D: "None of us are liars."

def statement_A : Prop := is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D
def statement_B : Prop := (is_liar A ∧ is_liar B) ∨ (is_liar A ∧ is_liar C) ∨ (is_liar A ∧ is_liar D) ∨
                          (is_liar B ∧ is_liar C) ∨ (is_liar B ∧ is_liar D) ∨ (is_liar C ∧ is_liar D)
def statement_C : Prop := (is_liar A ∧ is_liar B ∧ is_liar C) ∨ (is_liar A ∧ is_liar B ∧ is_liar D) ∨
                          (is_liar A ∧ is_liar C ∧ is_liar D) ∨ (is_liar B ∧ is_liar C ∧ is_liar D)
def statement_D : Prop := ¬(is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D)

-- Given that there are some liars
axiom some_liars_exist : ∃ x, is_liar x

-- Lean proof statement
theorem liars_are_C_and_D : is_liar C ∧ is_liar D ∧ ¬(is_liar A) ∧ ¬(is_liar B) :=
by
  sorry

end liars_are_C_and_D_l810_810057


namespace hyperbola_eccentricity_correct_l810_810602

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : a > 0) (k : b > 0)
  (h_eq : a + Real.sqrt (a^2 + b^2) = Real.sqrt 3 * b) : ℝ :=
let c := Real.sqrt (a^2 + b^2) in
let e := c / a in
2

theorem hyperbola_eccentricity_correct (a b : ℝ) (h : a > 0) (k : b > 0)
  (h_eq : a + Real.sqrt (a^2 + b^2) = Real.sqrt 3 * b) :
  hyperbola_eccentricity a b h k h_eq = 2 :=
by
  sorry

end hyperbola_eccentricity_correct_l810_810602


namespace length_of_one_of_these_halves_l810_810496

noncomputable def length_of_half (L : ℝ) (n : ℕ) : ℝ :=
  (L / n) / 2

theorem length_of_one_of_these_halves :
  length_of_half 153 7 ≈ 10.93 :=
by
  sorry

end length_of_one_of_these_halves_l810_810496


namespace line_intersection_y_axis_l810_810950

theorem line_intersection_y_axis :
  let p1 := (2, 8)
      p2 := (4, 14)
      m := (p2.2 - p1.2) / (p2.1 - p1.1)  -- slope calculation
      b := p1.2 - m * p1.1  -- y-intercept calculation
  in (b = 2) → (m = 3) → (p1 ≠ p2) → 
    (0, b) = @ y-intercept of the line passing through p1 and p2 :=
by
  intros p1 p2 m b h1 h2 h3
  -- placeholder for actual proof
  sorry

end line_intersection_y_axis_l810_810950


namespace only_set_B_is_right_angle_triangle_l810_810458

def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem only_set_B_is_right_angle_triangle :
  is_right_angle_triangle 3 4 5 ∧ ¬is_right_angle_triangle 1 2 2 ∧ ¬is_right_angle_triangle 3 4 9 ∧ ¬is_right_angle_triangle 4 5 7 :=
by
  -- proof steps omitted
  sorry

end only_set_B_is_right_angle_triangle_l810_810458


namespace incorrect_option_D_l810_810901

-- Define the data set
def temperatures : List ℤ := [-7, -4, -2, 1, -2, 2]

-- Define function to calculate mean
def mean (lst : List ℤ) : ℤ :=
  (lst.sum / lst.length.toInt)

-- Define function to calculate median
def median (lst : List ℤ) : ℤ :=
  let sorted := lst.qsort (· < ·)
  if sorted.length % 2 == 0 then
    (sorted.nth! (sorted.length / 2 - 1) + sorted.nth! (sorted.length / 2)) / 2
  else
    sorted.nth! (sorted.length / 2)

-- Define function to calculate mode
def mode (lst : List ℤ) : Option ℤ :=
  lst.foldl (λ (acc : Option (ℤ × Nat)) x, match acc with
    | none => some (x, 1)
    | some (y, n) => if x == y then some (y, n + 1) else if n == 1 then some (x, 1) else acc
  ) none |>.map (λ p => p.1)

-- Define function to calculate variance
def variance (lst : List ℤ) (mean_val : ℤ) : ℤ :=
  (lst.map (λ x => (x - mean_val) * (x - mean_val))).sum / lst.length.toInt

theorem incorrect_option_D : 
  mean temperatures = -2 ∧
  median temperatures = -2 ∧
  mode temperatures = some -2 ∧
  variance temperatures (mean temperatures) = 9 →
  OptionDIncorrect :=
by
  sorry -- Proof not required

noncomputable def OptionDIncorrect := variance temperatures (mean temperatures) ≠ 7

end incorrect_option_D_l810_810901


namespace sum_intervals_eq_log2013_2015_l810_810077

-- Define the function g(x) as described in the conditions
def g (x : ℝ) : ℝ := floor x * (2013^(x - floor x) - 1)

-- The main theorem
theorem sum_intervals_eq_log2013_2015 :
  (∑ k in finset.range 2014, real.log 2013 (↑(k + 1) / ↑k)) = real.log 2013 2015 := sorry

end sum_intervals_eq_log2013_2015_l810_810077


namespace louis_current_age_l810_810236

-- Define the constants for years to future and future age of Carla
def years_to_future : ℕ := 6
def carla_future_age : ℕ := 30

-- Define the sum of current ages
def sum_current_ages : ℕ := 55

-- State the theorem
theorem louis_current_age :
  ∃ (c l : ℕ), (c + years_to_future = carla_future_age) ∧ (c + l = sum_current_ages) ∧ (l = 31) :=
sorry

end louis_current_age_l810_810236


namespace smallest_possible_X_l810_810688

theorem smallest_possible_X (T : ℕ) (h1 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) (h2 : T % 24 = 0) :
  ∃ (X : ℕ), X = T / 24 ∧ X = 4625 :=
  sorry

end smallest_possible_X_l810_810688


namespace greatest_x_lcm_l810_810802

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l810_810802


namespace geom_seq_common_ratio_arith_seq_ratio_l810_810312

section GeometricSequence

variables (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)

-- Geometric sequence conditions
hypothesis geomSum : (∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) -- Sum formula for geometric sequence
hypothesis frac : S 5 / S 10 = 1 / 3

-- Proof problem statement for geometric sequence
theorem geom_seq_common_ratio : q = (2:ℝ)^(1/5) := by sorry

end GeometricSequence

section ArithmeticSequence

variables (S : ℕ → ℝ)

-- Arithmetic sequence conditions
hypotheses
  (arithSum5 : S 5 = m)
  (arithSum10 : S 10 = 3 * m)
  (arithFrac : S 5 / S 10 = 1 / 3)

-- Derived condition
noncomputable def derivedSum20 := 10 * m

-- Proof problem statement for arithmetic sequence
theorem arith_seq_ratio : S 10 / derivedSum20 = 3 / 10 := by sorry

end ArithmeticSequence

end geom_seq_common_ratio_arith_seq_ratio_l810_810312


namespace irrational_number_l810_810530

theorem irrational_number : 
  ∃ x ∈ ({Real.pi, Real.sqrt 4, 22 / 7, Real.sqrt 3 8} : set ℝ), irrational x ∧ ∀ y ∈ ({Real.pi, Real.sqrt 4, 22 / 7, Real.sqrt 3 8} : set ℝ), irrational y → y = x :=
by
  sorry

end irrational_number_l810_810530


namespace five_student_committees_l810_810146

theorem five_student_committees (n k : ℕ) (hn : n = 8) (hk : k = 5) : 
  nat.choose n k = 56 := by
  rw [hn, hk]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end five_student_committees_l810_810146


namespace range_of_a_l810_810083

def S : Set ℝ := {x | (x - 2) ^ 2 > 9 }
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8 }

theorem range_of_a (a : ℝ) : (S ∪ T a) = Set.univ ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end range_of_a_l810_810083


namespace hyperbola_equation_center_origin_asymptote_l810_810165

theorem hyperbola_equation_center_origin_asymptote
  (center_origin : ∀ x y : ℝ, x = 0 ∧ y = 0)
  (focus_parabola : ∃ x : ℝ, 4 * x^2 = 8 * x)
  (asymptote : ∀ x y : ℝ, x + y = 0):
  ∃ a b : ℝ, a^2 = 2 ∧ b^2 = 2 ∧ (x^2 / 2) - (y^2 / 2) = 1 := 
sorry

end hyperbola_equation_center_origin_asymptote_l810_810165


namespace digit_at_position_2020_l810_810702

def sequence_digit (n : Nat) : Nat :=
  -- Function to return the nth digit of the sequence formed by concatenating the integers from 1 to 1000
  sorry

theorem digit_at_position_2020 : sequence_digit 2020 = 7 :=
  sorry

end digit_at_position_2020_l810_810702


namespace greatest_x_lcm_l810_810798

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end greatest_x_lcm_l810_810798


namespace impossible_to_be_ahead_l810_810218

def Teena_speed := 85  -- in mph
def Yoe_speed := 60    -- in mph
def Lona_speed := 70   -- in mph
def Teena_behind_Yoe := 17.5  -- in miles
def Lona_behind_Teena := 20   -- in miles

theorem impossible_to_be_ahead :
  ¬ (∃ t : ℝ, t >= 0 ∧
       (Teena_speed * t - (Yoe_speed * t + Teena_behind_Yoe)) = 25 ∧
       (Teena_speed * t - (Lona_speed * t - Lona_behind_Teena)) = 10) :=
begin
  sorry
end

end impossible_to_be_ahead_l810_810218


namespace hours_learning_english_each_day_l810_810096

theorem hours_learning_english_each_day (total_hours : ℕ) (days : ℕ) (learning_hours_per_day : ℕ) 
  (h1 : total_hours = 12) 
  (h2 : days = 2) 
  (h3 : total_hours = learning_hours_per_day * days) : 
  learning_hours_per_day = 6 := 
by
  sorry

end hours_learning_english_each_day_l810_810096


namespace initial_cell_count_l810_810860

-- Defining the constants and parameters given in the problem
def doubling_time : ℕ := 20 -- minutes
def culture_time : ℕ := 240 -- minutes (4 hours converted to minutes)
def final_bacterial_cells : ℕ := 4096

-- Definition to find the number of doublings
def num_doublings (culture_time doubling_time : ℕ) : ℕ :=
  culture_time / doubling_time

-- Definition for exponential growth formula
def exponential_growth (initial_cells : ℕ) (doublings : ℕ) : ℕ :=
  initial_cells * (2 ^ doublings)

-- The main theorem to be proven
theorem initial_cell_count :
  exponential_growth 1 (num_doublings culture_time doubling_time) = final_bacterial_cells :=
  sorry

end initial_cell_count_l810_810860


namespace part1_part2_l810_810296

-- Definitions based on conditions
def P (i : ℕ) : Set (Set ℕ) :=
  {S | ∃ n, S = {m | 1 ≤ m ∧ m ≤ i}}

def a₁₂ = 9
def a (i j : ℕ) : ℕ :=
  (2^j - 1)^i

-- Lean statements
theorem part1 (P₁ P₂ : Set ℕ): 
  a₁₂ = 9 := 
  by
  sorry

theorem part2 (i j : ℕ) (P₁ P₂ : Set ℕ) (hij : P₁ ∩ P₂ = ∅):
  a i j = (2^j - 1)^i := 
  by
  sorry

end part1_part2_l810_810296


namespace probability_john_reads_on_both_days_l810_810415

-- Defining the basic events
noncomputable def reads_book_on_monday := 0.8
noncomputable def plays_soccer_on_tuesday := 0.5

-- Defining conditional event probabilities
noncomputable def reads_book_on_tuesday_given_conditions := reads_book_on_monday * plays_soccer_on_tuesday

-- The main theorem to be proved.
theorem probability_john_reads_on_both_days :
  reads_book_on_tuesday_given_conditions = 0.32 := by
  sorry

end probability_john_reads_on_both_days_l810_810415


namespace tom_walking_distance_l810_810676

noncomputable def walking_rate_miles_per_minute : ℝ := 1 / 18
def walking_time_minutes : ℝ := 15
def expected_distance_miles : ℝ := 0.8

theorem tom_walking_distance :
  walking_rate_miles_per_minute * walking_time_minutes = expected_distance_miles :=
by
  -- Calculation steps and conversion to decimal are skipped
  sorry

end tom_walking_distance_l810_810676


namespace weight_of_one_apple_l810_810480

-- Conditions
def total_weight_of_bag_with_apples : ℝ := 1.82
def weight_of_empty_bag : ℝ := 0.5
def number_of_apples : ℕ := 6

-- The proposition to prove: the weight of one apple
theorem weight_of_one_apple : (total_weight_of_bag_with_apples - weight_of_empty_bag) / number_of_apples = 0.22 := 
by
  sorry

end weight_of_one_apple_l810_810480


namespace degree_sum_interior_angles_of_star_l810_810054

-- Definitions based on conditions provided.
def extended_polygon_star (n : Nat) (h : n ≥ 6) : Nat := 
  180 * (n - 2)

-- Theorem to prove the degree-sum of the interior angles.
theorem degree_sum_interior_angles_of_star (n : Nat) (h : n ≥ 6) : 
  extended_polygon_star n h = 180 * (n - 2) :=
by
  sorry

end degree_sum_interior_angles_of_star_l810_810054


namespace first_player_winning_strategy_l810_810446

-- Define the problem in Lean 4
theorem first_player_winning_strategy (n : ℕ) (h : n ≥ 4) : 
  (∃ i, i ∈ [0, 1, 2] ∧ (n % 4 = i)) ↔ 
  (∃ strategy : nat → nat × nat, 
    (∀ x, strategy x ≠ (x, 0) ∧ strategy x ≠ (0, x)) ∧ 
    (∀ x, x < n → (strategy x).1 + (strategy x).2 = x)) :=
sorry

end first_player_winning_strategy_l810_810446


namespace five_student_committees_l810_810144

theorem five_student_committees (n k : ℕ) (hn : n = 8) (hk : k = 5) : 
  nat.choose n k = 56 := by
  rw [hn, hk]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end five_student_committees_l810_810144


namespace pairwise_products_sum_leq_one_fifth_l810_810828

theorem pairwise_products_sum_leq_one_fifth 
  (a b c d e : ℝ) 
  (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) 
  (h2 : a + b + c + d + e = 1) : 
  ∃ p q r s t : ℝ, 
    {p, q, r, s, t} = {a, b, c, d, e} ∧ 
    p * q + q * r + r * s + s * t + t * p ≤ 1/5 := 
sorry

end pairwise_products_sum_leq_one_fifth_l810_810828


namespace decimal_expansion_of_13_over_625_l810_810113

theorem decimal_expansion_of_13_over_625 : (13 : ℚ) / 625 = 0.0208 :=
by sorry

end decimal_expansion_of_13_over_625_l810_810113


namespace conjugate_in_fourth_quadrant_l810_810710

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

theorem conjugate_in_fourth_quadrant (z : ℂ) :
  (2 - complex.i) * z = 1 + complex.i →
  ∃ (q : ℕ), q = 4 ∧ 
  let z_conj := complex_conjugate z
  in z_conj.re > 0 ∧ z_conj.im < 0 :=
by
  sorry

end conjugate_in_fourth_quadrant_l810_810710


namespace correct_function_b_l810_810945

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f x < f y

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f x > f y

def function_b (x : ℝ) : ℝ :=
  -x^2 + abs(x)

theorem correct_function_b :
  is_even function_b ∧ is_monotonically_decreasing function_b 1 (real.to_nnreal ℝ + ∞) :=
sorry

end correct_function_b_l810_810945


namespace common_ratio_l810_810661

variable (a_1 d : ℝ)
def a_n (n : ℕ) : ℝ := a_1 + d * n
def is_geometric_sequence (b c e : ℝ) : Prop := c * c = b * e

theorem common_ratio (h: is_geometric_sequence a_1 (a_n a_1 d 2) (a_n a_1 d 3)) :
  (a_n a_1 d 2 = 0 ∨ a_1 = -4 * d) → 
  let q := if a_n a_1 d 2 = 0 then 1 else (a_n a_1 d 2 / a_1) in
  q = 1 ∨ q = 1/2 :=
sorry

end common_ratio_l810_810661


namespace polar_to_cartesian_hyperbola_l810_810755

-- Define polar coordinate relationship
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define given condition in polar coordinates
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos (2 * θ) = 1

-- Define the Cartesian condition which we need to prove
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 - y^2 = 1

-- Theorem stating the equivalence
theorem polar_to_cartesian_hyperbola (ρ θ : ℝ) (h : polar_equation ρ θ) :
  ∃ x y : ℝ, polar_to_cartesian ρ θ = (x, y) ∧ cartesian_equation x y :=
by
  sorry

end polar_to_cartesian_hyperbola_l810_810755


namespace c_investment_l810_810522

theorem c_investment (a_investment b_investment a_profit total_profit : ℝ) (x : ℝ) (h : a_investment / (a_investment + b_investment + x) = a_profit / total_profit) :
  x = 10500 :=
by
  have h1 : a_investment = 6300 := rfl
  have h2 : b_investment = 4200 := rfl
  have h3 : total_profit = 12300 := rfl
  have h4 : a_profit = 3690 := rfl
  sorry

end c_investment_l810_810522


namespace smallest_possible_X_l810_810687

theorem smallest_possible_X (T : ℕ) (h1 : ∀ d ∈ T.digits 10, d = 0 ∨ d = 1) (h2 : T % 24 = 0) :
  ∃ (X : ℕ), X = T / 24 ∧ X = 4625 :=
  sorry

end smallest_possible_X_l810_810687


namespace min_value_expression_l810_810973

open Real

/-- The minimum value of (14 - x) * (8 - x) * (14 + x) * (8 + x) is -4356. -/
theorem min_value_expression (x : ℝ) : ∃ (a : ℝ), a = (14 - x) * (8 - x) * (14 + x) * (8 + x) ∧ a ≥ -4356 :=
by
  use -4356
  sorry

end min_value_expression_l810_810973


namespace t_mobile_first_two_lines_cost_l810_810722

theorem t_mobile_first_two_lines_cost :
  ∃ T : ℝ,
  (T + 16 * 3) = (45 + 14 * 3 + 11) → T = 50 :=
by
  sorry

end t_mobile_first_two_lines_cost_l810_810722


namespace max_x_lcm_15_21_105_l810_810771

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l810_810771


namespace five_student_committees_from_eight_l810_810130

theorem five_student_committees_from_eight : nat.choose 8 5 = 56 := by
  sorry

end five_student_committees_from_eight_l810_810130


namespace correct_statements_l810_810052

noncomputable def statement_1 (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x ∈ set.Icc (2 * a - 1) (a + 4), f x = a * x^2 + (2 * a + b) * x + 2 ∧ (f (-x) = f x)

noncomputable def statement_2 (f : ℝ → ℝ) : Prop :=
∀ x, f x = min (-2 * x + 2) (-2 * x^2 + 4 * x + 2) ∧ f x ≤ 1

noncomputable def statement_3 (f : ℝ → ℝ) (a : ℝ) : Prop :=
∀ x, f x = abs (2 * x + a) ∧ ∀ y, y ≥ 3 → 2 * y + a ≥ 0

noncomputable def statement_4 (f : ℝ → ℝ) : Prop :=
∀ x y, f(x * y) = x * f(y) + y * f(x) ∧ (∀ z, f(-z) = -f(z))

theorem correct_statements :
  (∀ f a b, statement_1 f a b → b = 2) ∧
  (∀ f, statement_2 f → False) ∧
  (∀ f a, statement_3 f a → a = -6) ∧
  (∀ f, statement_4 f → ∀ z : ℝ, f(-z) = -f(z)) :=
by {
  intros,
  sorry
}

end correct_statements_l810_810052


namespace circumscribed_sphere_radii_equal_circumscribed_circle_radii_equal_l810_810380

-- Definitions to be used from conditions in a)
variables {P : Type} [regular_polyhedron P]
variables {P_dual : Type} [dual_regular_polyhedron P_dual]

-- Given conditions
variables (r_1 r_2 R_1 R_2 : ℝ)

-- Assert the conditions of inscribed sphere radii being equal
def radii_inscribed_equal (r_1 r_2 : ℝ) := r_1 = r_2

-- Part (a): proving the equality of circumscribed sphere radii
theorem circumscribed_sphere_radii_equal
  (h₁ : radii_inscribed_equal r_1 r_2)
  (H₁ : P)
  (H₂ : P_dual)
  (R_1 R_2 : ℝ) :
  (R₁ = R₂) :=
sorry

-- Defining the relation between radius of circumscribed circle
def radius_circumscribed_circle (R r : ℝ) : ℝ := sqrt (R^2 - r^2)

-- Part (b): proving the equality of circumscribed circles of faces radii
theorem circumscribed_circle_radii_equal
  (h₁ : radii_inscribed_equal r_1 r_2)
  (H₁ : P)
  (H₂ : P_dual)
  (R₁ R₂ : ℝ)
  (h_a : circumscribed_sphere_radii_equal r_1 r_2 H₁ H₂ R₁ R₂) :
  (radius_circumscribed_circle R₁ r₁ = radius_circumscribed_circle R₂ r₂) :=
sorry

end circumscribed_sphere_radii_equal_circumscribed_circle_radii_equal_l810_810380


namespace calculate_allocations_l810_810319

variable (new_revenue : ℝ)
variable (ratio_employee_salaries ratio_stock_purchases ratio_rent ratio_marketing_costs : ℕ)

theorem calculate_allocations :
  let total_ratio := ratio_employee_salaries + ratio_stock_purchases + ratio_rent + ratio_marketing_costs
  let part_value := new_revenue / total_ratio
  let employee_salary_alloc := ratio_employee_salaries * part_value
  let rent_alloc := ratio_rent * part_value
  let marketing_costs_alloc := ratio_marketing_costs * part_value
  employee_salary_alloc + rent_alloc + marketing_costs_alloc = 7800 :=
by
  sorry

end calculate_allocations_l810_810319


namespace greatest_x_lcm_105_l810_810788

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l810_810788


namespace mr_william_farm_tax_l810_810982

noncomputable def total_tax_collected : ℝ := 3840
noncomputable def mr_william_percentage : ℝ := 16.666666666666668 / 100  -- Convert percentage to decimal

theorem mr_william_farm_tax : (total_tax_collected * mr_william_percentage) = 640 := by
  sorry

end mr_william_farm_tax_l810_810982


namespace domain_of_f_squared_l810_810615

theorem domain_of_f_squared (f : ℝ → ℝ) :
  (∀ x, 1 ≤ log 2 (x + 1) ∧ log 2 (x + 1) ≤ 15 → f (log 2 (x + 1)) ∈ set.Icc 1 15) →
  (∀ y, (1 ≤ y ∧ y ≤ 2) ∨ (-2 ≤ y ∧ y ≤ -1) → f (y^2) ≠ 0) :=
by
  sorry

end domain_of_f_squared_l810_810615


namespace equidistant_points_count_l810_810551

theorem equidistant_points_count {O : Point} {r : ℝ} (circle : Circle O r) (tangent1 tangent2 : Line)
  (h1 : is_tangent circle tangent1 ∧ dist O tangent1 = r) 
  (h2 : is_tangent circle tangent2 ∧ dist O tangent2 = 2 * r)
  (parallel : parallel tangent1 tangent2) :
  ∃! P : Point, eq_dist_from_circle_and_tangents P circle tangent1 tangent2 := 
sorry

end equidistant_points_count_l810_810551


namespace combination_eight_choose_five_l810_810141

theorem combination_eight_choose_five : 
  ∀ (n k : ℕ), n = 8 ∧ k = 5 → Nat.choose n k = 56 :=
by 
  intros n k h
  obtain ⟨hn, hk⟩ := h
  rw [hn, hk]
  exact Nat.choose_eq 8 5
  sorry  -- This signifies that the proof needs to be filled in, but we'll skip it as per instructions.

end combination_eight_choose_five_l810_810141


namespace angle_SRQ_proof_l810_810666

theorem angle_SRQ_proof (l k R S Q : Type) [inhabited l] [inhabited k] [inhabited R] [inhabited S] [inhabited Q] 
  (parallel_lk : ∀ (x : l) (y : k), x ∥ y)
  (perpendicular_RQ_lk : ∀ (r : R) (q : Q), ∃ (n m : ℝ), r = ⟨n, m⟩ ∧ q = ⟨n, m⟩ ∧ l r ⟂ k q)
  (angle_RSQ : ℝ) (h2 : angle_RSQ = 120) :
  ∃ (angle_SRQ : ℝ), angle_SRQ = 30 :=
by
  sorry

end angle_SRQ_proof_l810_810666


namespace weight_ratio_l810_810637

variable (J : ℕ) (T : ℕ) (L : ℕ) (S : ℕ)

theorem weight_ratio (h_jake_weight : J = 152) (h_total_weight : J + S = 212) (h_weight_loss : L = 32) :
    (J - L) / (T - J) = 2 :=
by
  sorry

end weight_ratio_l810_810637


namespace hex_B1F4_to_decimal_l810_810967

def hex_to_decimal (B1F4 : string) : ℕ :=
  let B := 11
  let one := 1
  let F := 15
  let four := 4
  B * 16^3 + one * 16^2 + F * 16^1 + four

theorem hex_B1F4_to_decimal : hex_to_decimal "B1F4" = 45556 :=
by
  -- Compute each part separately to ensure clarity
  let B := 11
  let one := 1
  let F := 15
  let four := 4
  have B_term : B * 16^3 = 45056 := by sorry
  have one_term : one * 16^2 = 256 := by sorry
  have F_term : F * 16^1 = 240 := by sorry
  have four_term : four * 16^0 = 4 := by sorry
  calc
    hex_to_decimal "B1F4" = 45056 + 256 + 240 + 4 := by sorry
    ... = 45556 := by sorry

end hex_B1F4_to_decimal_l810_810967


namespace part1_min_tan_A_part2_find_c_l810_810247

variables (A B C a b c : ℝ)
variables (ABC_acute: 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variables (sides_opposite: a = A ∧ b = B ∧ c = C)
variables (trig_identity: cos C * sin (A - B) = cos B * sin (C - A))

noncomputable def min_tan_A : ℝ := sqrt 3

theorem part1_min_tan_A (h : ABC_acute ∧ sides_opposite ∧ trig_identity) :
  (∀ tA, tan A = tA → tA ≥ min_tan_A) :=
sorry

variables (tan_A_eq_2 : tan A = 2)
variables (a_eq_4sqrt5 : a = 4 * sqrt 5)

noncomputable def possible_c1 : ℝ := 5 * sqrt 2
noncomputable def possible_c2 : ℝ := 3 * sqrt 10

theorem part2_find_c (h : ABC_acute ∧ sides_opposite ∧ trig_identity ∧ tan_A_eq_2 ∧ a_eq_4sqrt5) : (c = possible_c1 ∨ c = possible_c2) :=
sorry

end part1_min_tan_A_part2_find_c_l810_810247


namespace find_sum_of_coefficients_l810_810174

theorem find_sum_of_coefficients (O A B C D : Point)
  (h_non_collinear: ¬ collinear {A, B, C})
  (h_coplanar: coplanar {A, B, C, D})
  (x y z : ℝ)
  (h_vector_equation: vector_equals (OA O A) (2 * x * (OB O B) + 3 * y * (OC O C) + 4 * z * (OD O D)))
  : 2 * x + 3 * y + 4 * z = -1 :=
sorry

end find_sum_of_coefficients_l810_810174


namespace range_of_a_l810_810619

def valid_real_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a

theorem range_of_a :
  (∀ a : ℝ, (¬ valid_real_a a)) ↔ (a < 1 ∨ a > 3) :=
sorry

end range_of_a_l810_810619


namespace group_students_l810_810437

-- Define the conditions and the final theorem statement
theorem group_students :
  (∃ (n_a n_b n_c n_ab n_bc n_ca n_abc : ℕ),
    n_a + n_ab + n_ca + n_abc = 50 ∧
    n_b + n_ab + n_bc + n_abc = 50 ∧
    n_c + n_bc + n_ca + n_abc = 50) →
  (∃ (groups : list (ℕ → bool)),
    groups.length = 5 ∧
    (∀ group ∈ groups, 
      (count (λ n, group n && speaks_english n) = 10 ∧
       count (λ n, group n && speaks_french n) = 10 ∧
       count (λ n, group n && speaks_spanish n) = 10)) 
    ∧ (all_students_in_groups groups) ∧ (no_duplicate_students groups)) :=
sorry -- Proof goes here

end group_students_l810_810437


namespace baxter_earnings_l810_810388

theorem baxter_earnings (a_days : ℕ) (a_students : ℕ) 
                        (b_days : ℕ) (b_students : ℕ) (b_bonus : ℕ) 
                        (c_days : ℕ) (c_students : ℕ) 
                        (total_payment : ℝ) 
                        (daily_wage : ℝ)
                        (total_days: ℝ) 
                        (adjusted_payment: ℝ)
                        (baxter_days : ℝ) 
                        (earnings : ℝ) : 
  a_students * a_days = 20 ∧
  b_students * b_days = 18 ∧
  c_students * c_days = 48 ∧
  total_payment = 920 ∧
  total_days = (a_students * a_days + b_students * b_days + c_students * c_days) ∧
  daily_wage = adjusted_payment / total_days ∧
  adjusted_payment = total_payment - (b_students * b_bonus) ∧
  earnings = (daily_wage * baxter_days) + (b_students * b_bonus) ∧
  round (earnings * 100) / 100 = 204.42 :=
by
  sorry

end baxter_earnings_l810_810388


namespace football_starting_lineup_count_l810_810488

variable (n_team_members n_offensive_linemen : ℕ)
variable (H_team_members : 12 = n_team_members)
variable (H_offensive_linemen : 5 = n_offensive_linemen)

theorem football_starting_lineup_count :
  n_team_members = 12 → n_offensive_linemen = 5 →
  (n_offensive_linemen * (n_team_members - 1) * (n_team_members - 2) * ((n_team_members - 3) * (n_team_members - 4) / 2)) = 19800 := 
by
  intros
  sorry

end football_starting_lineup_count_l810_810488


namespace sum_of_positive_ks_l810_810974

theorem sum_of_positive_ks : 
  (∑ k in {k : ℤ | ∃ α β : ℤ, α*β = -18 ∧ α+β = k ∧ k > 0}.to_finset, k) = 27 :=
by
  sorry

end sum_of_positive_ks_l810_810974


namespace orthogonal_projection_circumcenter_l810_810413

theorem orthogonal_projection_circumcenter (A B C M : ℝ³) (hMA : dist M A = dist M B) (hMB : dist M B = dist M C) :
  let M₁ := orthogonal_projection (affine_span ℝ {A, B, C}) M in
  dist M₁ A = dist M₁ B ∧ dist M₁ B = dist M₁ C := 
by 
  sorry

end orthogonal_projection_circumcenter_l810_810413


namespace square_area_from_diagonal_l810_810501

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  (let s := d / Real.sqrt 2 in s * s) = 144 := by
sorry

end square_area_from_diagonal_l810_810501


namespace greatest_x_lcm_105_l810_810785

theorem greatest_x_lcm_105 (x: ℕ): (Nat.lcm x 15 = Nat.lcm 21 105) → (x ≤ 105 ∧ Nat.dvd 105 x) → x = 105 :=
by
  sorry

end greatest_x_lcm_105_l810_810785


namespace simplified_expression_eq_sqrt2_l810_810385

noncomputable def simplify_expression (y : ℝ) : ℝ :=
  sqrt (y + 2 + 3 * sqrt (2 * y - 5)) - sqrt (y - 2 + sqrt (2 * y - 5))

theorem simplified_expression_eq_sqrt2 (y : ℝ) (x : ℝ)
  (h1 : sqrt (2 * y - 5) = x)
  (h2 : y ≥ 5 / 2) :
  simplify_expression y = sqrt 2 := by
  sorry

end simplified_expression_eq_sqrt2_l810_810385


namespace num_five_student_committees_l810_810120

theorem num_five_student_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 5) : choose n k = 56 :=
by
  rw [h_n, h_k]
  -- rest of the proof would go here
  sorry

end num_five_student_committees_l810_810120


namespace number_of_friends_l810_810334

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l810_810334


namespace number_of_friends_l810_810335

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l810_810335


namespace number_of_fractions_satisfying_conditions_l810_810412

theorem number_of_fractions_satisfying_conditions : 
  {a : ℕ // 1 < a ∧ a < 7}.card = 5 :=
by
  sorry

end number_of_fractions_satisfying_conditions_l810_810412


namespace folded_triangle_AC1_C1B_correct_l810_810519

theorem folded_triangle_AC1_C1B_correct 
  (AB BC CA : ℝ)
  (h_AB : AB = 6)
  (h_BC : BC = 5)
  (h_CA : CA = 4)
  (h_fold : ∃ (C_1 : ℝ) (K M : ℝ),
    C_1 ∈ set.Icc 0 AB ∧ 
    (∠(A, C, K) = ∠(C_1, B, M)) ∧ 
    (∠(C, K, M) = ∠(C_1, M, K))
  ) :
    ∃ (AC1 C1B : ℝ),
      AC1 = 10 / 3 ∧ C1B = 8 / 3 := 
begin
  sorry
end

end folded_triangle_AC1_C1B_correct_l810_810519


namespace problem1_problem2_l810_810197

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + a / (x + 1) + a / 2

theorem problem1 (a : ℝ) (h : a = 9 / 2) :
  (∀ x, 0 < x < 1 / 2 → (f' a x > 0)) ∧ 
  (∀ x, 2 < x → (f' a x > 0)) ∧ 
  (∀ x, 1 / 2 < x < 2 → (f' a x < 0)) :=
sorry

theorem problem2 (a : ℝ) (h : ∀ x, 0 < x → f a x ≤ (a / 2) * (x + 1)) : 
  a = 4 / 3 :=
sorry

-- Helper definition of the derivative f'
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ :=
  (2 * x - 1) * (x - 2) / (2 * x * (x + 1) ^ 2)

end problem1_problem2_l810_810197


namespace general_formula_for_bn_sum_T1_to_T2022_l810_810610

-- Definitions of sequences and given conditions
variable {a_n S_n b_n c_n T_n : ℕ → ℝ}
variable {n : ℕ}

-- Given Conditions
axiom b1_eq_a1 : b_n 1 = a_n 1
axiom two_an_plus_sn_eq_n : ∀ n, 2 * a_n n + S_n n = n
axiom bn_next_eq_an_next_minus_an : ∀ n, b_n (n + 1) = a_n (n + 1) - a_n n

-- Goals
theorem general_formula_for_bn :
  ∀ n, b_n n = (1 / 2) * (2 / 3) ^ n := by
  sorry

theorem sum_T1_to_T2022 :
  (∑ k in Finset.range 2022, T_n (k + 1)) =
  - (1 / 3) * (1 - (1 / 2) ^ 2022) := by
  sorry

end general_formula_for_bn_sum_T1_to_T2022_l810_810610


namespace greatest_possible_x_max_possible_x_l810_810779

theorem greatest_possible_x (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : x ≤ 105 :=
by
  -- Proof goes here
  sorry

-- As a corollary, we can state the maximum value of x
theorem max_possible_x : 105 ≤ 105 :=
by
  -- Proof goes here
  exact le_refl 105

end greatest_possible_x_max_possible_x_l810_810779


namespace find_a_solution_set_a_negative_l810_810199

-- Definitions
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (a - 1) * x - 1 ≥ 0

-- Problem 1: Prove the value of 'a'
theorem find_a (h : ∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ -1/2)) :
  a = -2 :=
sorry

-- Problem 2: Prove the solution sets when a < 0
theorem solution_set_a_negative (h : a < 0) :
  (a = -1 → (∀ x : ℝ, quadratic_inequality a x ↔ x = -1)) ∧
  (a < -1 → (∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ 1/a))) ∧
  (-1 < a ∧ a < 0 → (∀ x : ℝ, quadratic_inequality a x ↔ (1/a ≤ x ∧ x ≤ -1))) :=
sorry

end find_a_solution_set_a_negative_l810_810199


namespace cost_difference_of_buses_l810_810251

-- Definitions from the conditions
def bus_cost_equations (x y : ℝ) :=
  (x + 2 * y = 260) ∧ (2 * x + y = 280)

-- The statement to prove
theorem cost_difference_of_buses (x y : ℝ) (h : bus_cost_equations x y) :
  x - y = 20 :=
sorry

end cost_difference_of_buses_l810_810251


namespace paolo_sevilla_birthday_l810_810340

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l810_810340


namespace greatest_x_lcm_l810_810818

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l810_810818


namespace tangent_circles_BC_length_l810_810855

theorem tangent_circles_BC_length
  (rA rB : ℝ) (A B C : ℝ × ℝ) (distAB distAC : ℝ) 
  (hAB : rA + rB = distAB)
  (hAC : distAB + 2 = distAC) 
  (h_sim : ∀ AD BE BC AC : ℝ, AD / BE = rA / rB → BC / AC = rB / rA) :
  BC = 52 / 7 := sorry

end tangent_circles_BC_length_l810_810855


namespace count_distinct_integer_values_of_sum_l810_810979

theorem count_distinct_integer_values_of_sum :
  let x : ℕ → ℝ := λ i, if i % 2 = 0 then (if i / 2 % 2 = 0 then sqrt 2 - 1 else sqrt 2 + 1) else (if (i - 1) / 2 % 2 = 0 then sqrt 2 - 1 else sqrt 2 + 1)
  in (∑ k in Finset.range 1002, x (2*k) * x (2*k + 1)) = 2005 :=
by
  sorry

end count_distinct_integer_values_of_sum_l810_810979


namespace paolo_sevilla_birthday_l810_810341

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l810_810341


namespace Jean_speed_proof_l810_810959

noncomputable def Chantal_flat_distance := 8
noncomputable def Chantal_uphill_distance := 4
noncomputable def total_distance := 12
noncomputable def Chantal_flat_speed := 3
noncomputable def Chantal_uphill_speed := 1.5
noncomputable def Chantal_downhill_speed := 2.25
noncomputable def Jean_delay := 2
noncomputable def meeting_time := (Chantal_flat_distance / Chantal_flat_speed) 
                                  + (Chantal_uphill_distance / Chantal_uphill_speed) 
                                  + (Chantal_uphill_distance / Chantal_downhill_speed)

theorem Jean_speed_proof : 
  let Jean_time := (meeting_time - Jean_delay) in
  let Jean_distance := Chantal_uphill_distance in
  (Jean_distance / Jean_time) = 18 / 11 :=
by
  sorry

end Jean_speed_proof_l810_810959


namespace solve_for_x_l810_810386

theorem solve_for_x : ∃ x : ℝ, (x + 36) / 3 = (7 - 2 * x) / 6 ∧ x = -65 / 4 := by
  sorry

end solve_for_x_l810_810386


namespace alice_speed_proof_l810_810028

-- Problem definitions
def distance : ℕ := 1000
def abel_speed : ℕ := 50
def abel_arrival_time := distance / abel_speed
def alice_delay : ℕ := 1  -- Alice starts 1 hour later
def earlier_arrival_abel : ℕ := 6  -- Abel arrives 6 hours earlier than Alice

noncomputable def alice_speed : ℕ := (distance / (abel_arrival_time + earlier_arrival_abel))

theorem alice_speed_proof : alice_speed = 200 / 3 := by
  sorry -- proof not required as per instructions

end alice_speed_proof_l810_810028


namespace ada_originally_in_seat2_l810_810996

inductive Seat
| S1 | S2 | S3 | S4 | S5 deriving Inhabited, DecidableEq

def moveRight : Seat → Option Seat
| Seat.S1 => some Seat.S2
| Seat.S2 => some Seat.S3
| Seat.S3 => some Seat.S4
| Seat.S4 => some Seat.S5
| Seat.S5 => none

def moveLeft : Seat → Option Seat
| Seat.S1 => none
| Seat.S2 => some Seat.S1
| Seat.S3 => some Seat.S2
| Seat.S4 => some Seat.S3
| Seat.S5 => some Seat.S4

structure FriendState :=
  (bea ceci dee edie : Seat)
  (ada_left : Bool) -- Ada is away for snacks, identified by her not being in the seat row.

def initial_seating := FriendState.mk Seat.S2 Seat.S3 Seat.S4 Seat.S5 true

def final_seating (init : FriendState) : FriendState :=
  let bea' := match moveRight init.bea with
              | some pos => pos
              | none => init.bea
  let ceci' := init.ceci -- Ceci moves left then back, net zero movement
  let (dee', edie') := match moveRight init.dee, init.dee with
                      | some new_ee, ed => (new_ee, ed) -- Dee and Edie switch and Edie moves right
                      | _, _ => (init.dee, init.edie) -- If moves are invalid
  FriendState.mk bea' ceci' dee' edie' init.ada_left

theorem ada_originally_in_seat2 (init : FriendState) : init = initial_seating → final_seating init ≠ initial_seating → init.bea = Seat.S2 :=
by
  intro h_init h_finalne
  sorry -- Proof steps go here

end ada_originally_in_seat2_l810_810996


namespace median_of_list_l810_810451

theorem median_of_list : 
  let list := (List.range (3031)).map (fun x => x) ++ (List.range (3031)).map (fun x => x*x)
  list = list.sort <|
  let median := if list.length % 2 = 0 then (list[list.length / 2 - 1] + list[list.length / 2]) / 2
                else list[list.length / 2]
  median = 2975.5 := sorry

end median_of_list_l810_810451


namespace greatest_value_of_x_l810_810805

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l810_810805


namespace coefficient_x2_of_product_l810_810865

noncomputable def poly1 : ℝ[X] := 3 * X^4 - 2 * X^3 + 4 * X^2 - 5 * X + 2
noncomputable def poly2 : ℝ[X] := X^3 - 4 * X^2 + 3 * X + 6

theorem coefficient_x2_of_product : (poly1 * poly2).coeff 2 = 1 := by
  sorry

end coefficient_x2_of_product_l810_810865


namespace find_c_l810_810762

-- Define points and the line equation.
def point_A := (1, 3)
def point_B := (5, 11)
def midpoint (A B : ℚ × ℚ) : ℚ × ℚ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- The line equation 2x - y = c
def line_eq (x y c : ℚ) : Prop :=
  2 * x - y = c

-- Define the proof problem
theorem find_c : 
  let M := midpoint point_A point_B in
  line_eq M.1 M.2 (-1) :=
by
  sorry

end find_c_l810_810762


namespace arrangement_count_l810_810440

theorem arrangement_count (teachers students : Finset ℕ) (h_tea: teachers.card = 3) (h_stu: students.card = 3) :
  ∃ n, (∀ (arr: List ℕ), no_adjacent students arr ∧ arrangement arr teachers students) ∧ n = 144 :=
by
  sorry

-- Helper definitions
def no_adjacent (students : Finset ℕ) (arr : List ℕ) : Prop := sorry

def arrangement (arr : List ℕ) (teachers students : Finset ℕ) : Prop := sorry

end arrangement_count_l810_810440


namespace birthday_friends_count_l810_810374

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l810_810374


namespace area_of_triangle_OAB_slope_of_line_OM_l810_810668

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | let ⟨x, y⟩ := p in (x^2 / 4) + (y^2) = 1 }

noncomputable def line (t : ℝ) : Set (ℝ × ℝ) :=
  { p | let ⟨x, y⟩ := p in y = x + t }

def is_right_focus (p : ℝ × ℝ) : Prop :=
  p = (real.sqrt 3, 0)

def triangle_area (O A B : ℝ × ℝ) : ℝ :=
  let ⟨ox, oy⟩ := O, ⟨ax, ay⟩ := A, ⟨bx, by⟩ := B in
  abs ((ax - ox) * (by - oy) - (bx - ox) * (ay - oy)) / 2

def slope_of_OM (O M : ℝ × ℝ) : ℝ :=
  let ⟨ox, oy⟩ := O, ⟨mx, my⟩ := M in
  (my - oy) / (mx - ox)

theorem area_of_triangle_OAB (t : ℝ) (ht : t = - real.sqrt 3) (A B : ℝ × ℝ)
  (hA : A ∈ ellipse) (hB : B ∈ ellipse) (S : Set (ℝ × ℝ)) (hS : S = line t)
  (FOCUS : is_right_focus (real.sqrt 3, 0))
  (IntersectionA : A ∈ S) (IntersectionB : B ∈ S) :
  triangle_area (0, 0) A B = 2 * real.sqrt 3 / 5 :=
sorry

theorem slope_of_line_OM (t : ℝ) (ht1 : -real.sqrt 5 < t) (ht2 : t < real.sqrt 5) (ht0 : t ≠ 0) (A B : ℝ × ℝ)
  (hA : A ∈ ellipse) (hB : B ∈ ellipse) (S : Set (ℝ × ℝ)) (hS : S = line t)
  (M : ℝ × ℝ) (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) (IntersectionA : A ∈ S) (IntersectionB : B ∈ S) :
  slope_of_OM (0, 0) M = -1 / 4 :=
sorry

end area_of_triangle_OAB_slope_of_line_OM_l810_810668


namespace part1_min_tan_A_part2_find_c_l810_810248

variables (A B C a b c : ℝ)
variables (ABC_acute: 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variables (sides_opposite: a = A ∧ b = B ∧ c = C)
variables (trig_identity: cos C * sin (A - B) = cos B * sin (C - A))

noncomputable def min_tan_A : ℝ := sqrt 3

theorem part1_min_tan_A (h : ABC_acute ∧ sides_opposite ∧ trig_identity) :
  (∀ tA, tan A = tA → tA ≥ min_tan_A) :=
sorry

variables (tan_A_eq_2 : tan A = 2)
variables (a_eq_4sqrt5 : a = 4 * sqrt 5)

noncomputable def possible_c1 : ℝ := 5 * sqrt 2
noncomputable def possible_c2 : ℝ := 3 * sqrt 10

theorem part2_find_c (h : ABC_acute ∧ sides_opposite ∧ trig_identity ∧ tan_A_eq_2 ∧ a_eq_4sqrt5) : (c = possible_c1 ∨ c = possible_c2) :=
sorry

end part1_min_tan_A_part2_find_c_l810_810248


namespace total_cost_of_tennis_balls_l810_810720

theorem total_cost_of_tennis_balls
  (packs : ℕ) (balls_per_pack : ℕ) (cost_per_ball : ℕ)
  (h1 : packs = 4) (h2 : balls_per_pack = 3) (h3 : cost_per_ball = 2) : 
  packs * balls_per_pack * cost_per_ball = 24 := by
  sorry

end total_cost_of_tennis_balls_l810_810720


namespace length_segment_OS_l810_810013

theorem length_segment_OS {
  (O P T S Q : Type)
  [core_metric_space O] [core_metric_space P] [core_metric_space T] [core_metric_space S] [core_metric_space Q]
  (radiusO radiusP : ℝ)
  (h1 : radiusO = 10)
  (h2 : radiusP = 3)
  (h3 : dist O P = radiusO + radiusP)
  (h4 : dist T S = 2 * real.sqrt 30)
  (h5 : dist O T = radiusO)
  (h6 : dist P S = radiusP) :
  dist O S = 2 * real.sqrt 55 :=
by
  sorry

end length_segment_OS_l810_810013


namespace initial_average_mark_l810_810754

theorem initial_average_mark (A : ℕ) (A_excluded : ℕ := 20) (A_remaining : ℕ := 90) (n_total : ℕ := 14) (n_excluded : ℕ := 5) :
    (n_total * A = n_excluded * A_excluded + (n_total - n_excluded) * A_remaining) → A = 65 :=
by 
  intros h
  sorry

end initial_average_mark_l810_810754


namespace height_of_middle_brother_l810_810838

theorem height_of_middle_brother (h₁ h₂ h₃ : ℝ) (h₁_le_h₂ : h₁ ≤ h₂) (h₂_le_h₃ : h₂ ≤ h₃)
  (avg_height : (h₁ + h₂ + h₃) / 3 = 1.74) (avg_height_tallest_shortest : (h₁ + h₃) / 2 = 1.75) :
  h₂ = 1.72 :=
by
  -- Proof to be filled here
  sorry

end height_of_middle_brother_l810_810838


namespace geom_seq_increasing_sufficient_necessary_l810_810694

theorem geom_seq_increasing_sufficient_necessary (a : ℕ → ℝ) (r : ℝ) (h_geo : ∀ n : ℕ, a n = a 0 * r ^ n) 
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) : 
  (a 0 < a 1 ∧ a 1 < a 2) ↔ (∀ n : ℕ, a n < a (n + 1)) :=
sorry

end geom_seq_increasing_sufficient_necessary_l810_810694


namespace NaClO_concentration_decreases_over_time_l810_810930

theorem NaClO_concentration_decreases_over_time
  (V : ℝ) (C : ℝ) (D : ℝ) (t : ℝ)
  (H_v : V = 480) (H_c : C = 0.25) 
  (H_d : D = 1.19) (H_exposure : t > 0) :
  ∀ t' > t, concentration_of_NaClO (V - loss_due_to_exposure t' t) < concentration_of_NaClO V :=
sorry

end NaClO_concentration_decreases_over_time_l810_810930


namespace volume_of_inscribed_cubes_l810_810921

noncomputable def tetrahedron_cube_volume (a m : ℝ) : ℝ × ℝ :=
  let V1 := (a * m / (a + m))^3
  let V2 := (a * m / (a + (Real.sqrt 2) * m))^3
  (V1, V2)

theorem volume_of_inscribed_cubes (a m : ℝ) (ha : 0 < a) (hm : 0 < m) :
  tetrahedron_cube_volume a m = 
  ( (a * m / (a + m))^3, 
    (a * m / (a + (Real.sqrt 2) * m))^3 ) :=
  by
    sorry

end volume_of_inscribed_cubes_l810_810921


namespace total_charge_for_trip_l810_810469

def initial_fee : ℝ := 2.25
def charge_per_increment : ℝ := 0.4
def trip_distance : ℝ := 3.6
def increment_distance : ℝ := 2 / 5

def total_charge (distance : ℝ) : ℝ :=
  initial_fee + (distance / increment_distance) * charge_per_increment

theorem total_charge_for_trip :
  total_charge trip_distance = 5.85 :=
by
  sorry

end total_charge_for_trip_l810_810469


namespace center_of_4x4_matrix_l810_810726

theorem center_of_4x4_matrix :
  ∃ (M : Matrix (Fin 4) (Fin 4) ℕ), 
    (∀ i j : Fin 4, 1 ≤ M i j ∧ M i j ≤ 16) ∧
    (∀ i j i' j' : Fin 4, M i' j' = M i j + 1 → (i' = i ∧ (j' = j + 1 ∨ j' = j - 1) ∨ j' = j ∧ (i' = i + 1 ∨ i' = i - 1))) ∧
    (M 0 0 + M 0 3 + M 3 0 + M 3 3 = 34) ∧
    M 1 1 = 10 :=
begin
  sorry
end

end center_of_4x4_matrix_l810_810726


namespace clay_weight_in_second_box_l810_810893

/-- Define the properties of the first and second boxes -/
structure Box where
  height : ℕ
  width : ℕ
  length : ℕ
  weight : ℕ

noncomputable def box1 : Box :=
  { height := 2, width := 3, length := 5, weight := 40 }

noncomputable def box2 : Box :=
  { height := 2 * 2, width := 3 * 3, length := 5, weight := 240 }

theorem clay_weight_in_second_box : 
  box2.weight = (box2.height * box2.width * box2.length) / 
                (box1.height * box1.width * box1.length) * box1.weight :=
by
  sorry

end clay_weight_in_second_box_l810_810893


namespace five_student_committees_l810_810150

theorem five_student_committees (n k : ℕ) (hn : n = 8) (hk : k = 5) : 
  nat.choose n k = 56 := by
  rw [hn, hk]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end five_student_committees_l810_810150


namespace greatest_value_of_x_l810_810809

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end greatest_value_of_x_l810_810809


namespace range_H_l810_810868

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem range_H : set.range H = {5} :=
sorry

end range_H_l810_810868


namespace at_least_three_correct_guesses_l810_810397

-- Define the colors of the rainbow as an enumeration
inductive RainbowColor
| Red
| Orange
| Yellow
| Green
| Blue
| Indigo
| Violet

open RainbowColor

/-- There are seven different colors -/
def colors : List RainbowColor := [Red, Orange, Yellow, Green, Blue, Indigo, Violet]

/-- A gnome can see the colors of the hats on the other five gnomes -/
structure Gnome (hats : List RainbowColor) :=
(visible_hats : List RainbowColor) -- The five visible hats

/-- Determine the hidden hat color based on the cyclic pattern and visible hats -/
def hidden_hat_guess (g : Gnome) : RainbowColor :=
  -- Assume a function that correctly processes the strategy to guess nearest hat
  sorry -- Implementation detail omitted

/-- Ensure that at least three gnomes guess correctly given the strategy -/
theorem at_least_three_correct_guesses (hats : List RainbowColor)
  ( hidden : RainbowColor ) (gnomes : List (Gnome hats) ) :
  (∃ correct_guesses : List RainbowColor, correct_guesses.length ≥ 3 ∧ 
    ∀ g : Gnome hats, hidden_hat_guess g ∈ correct_guesses) :=
sorry

end at_least_three_correct_guesses_l810_810397


namespace union_A_B_m_eq_3_range_of_m_l810_810622

def A (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def B (x m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem union_A_B_m_eq_3 :
  A x ∨ B x 3 ↔ (-3 : ℝ) ≤ x ∧ x ≤ 5 := sorry

theorem range_of_m (h : ∀ x, A x ∨ B x m ↔ A x) : m ≤ (5 / 2) := sorry

end union_A_B_m_eq_3_range_of_m_l810_810622


namespace cost_of_fencing_per_meter_l810_810406

theorem cost_of_fencing_per_meter
  (breadth : ℝ)
  (length : ℝ)
  (cost : ℝ)
  (length_eq : length = breadth + 40)
  (total_cost : cost = 5300)
  (length_given : length = 70) :
  cost / (2 * length + 2 * breadth) = 26.5 :=
by
  sorry

end cost_of_fencing_per_meter_l810_810406


namespace number_of_friends_l810_810337

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l810_810337


namespace general_eq_line_rectangular_eq_curve_max_area_triangle_l810_810669

def parametric_line (t : ℝ) : ℝ × ℝ := (6 - (sqrt 3) / 2 * t, 1 / 2 * t)

def polar_curve (θ : ℝ) : ℝ := 6 * cos θ

theorem general_eq_line :
  ∀ t : ℝ, ∃ x y : ℝ, parametric_line t = (x, y) ∧ x + sqrt 3 * y - 6 = 0 := 
sorry

theorem rectangular_eq_curve :
  ∀ x y : ℝ, (polar_curve (real.atan2 y (x - 3)))^2 = (x - 3)^2 + y^2 :=
sorry

theorem max_area_triangle :
  ∀ x y : ℝ, parametric_line x = (y, (6 - sqrt 3 * y - 6) / (sqrt 3)) →
  let A := (3, 0)
  let r := 3
  let d := 3 / 2
  2 * sqrt (r^2 - d^2) = 3 * sqrt 3 →
  max (1 / 2 * (r + d) * 2 * sqrt (r^2 - d^2)) = 27 * sqrt 3 / 4 :=
sorry

end general_eq_line_rectangular_eq_curve_max_area_triangle_l810_810669


namespace max_x_lcm_15_21_105_l810_810766

theorem max_x_lcm_15_21_105 (x : ℕ) : lcm (lcm x 15) 21 = 105 → x = 105 :=
by
  sorry

end max_x_lcm_15_21_105_l810_810766


namespace domain_of_f_l810_810866

def f (x : ℝ) : ℝ := 1 / (x ^ 2 - 5 * x + 6)

theorem domain_of_f :
  ∀ x : ℝ, (x ∈ (-∞, 2) ∨ x ∈ (2, 3) ∨ x ∈ (3, ∞)) ↔ f x ≠ 0 ∧ (x ≠ 2 ∧ x ≠ 3) :=
by
  sorry

end domain_of_f_l810_810866


namespace probability_of_drawing_multiple_of_6_l810_810833

-- Define the set of cards
def cards : set ℕ := { n | 1 ≤ n ∧ n ≤ 100 }

-- Define the subset of multiples of 6 within the set of cards
def multiples_of_6 : set ℕ := { n | n ∈ cards ∧ n % 6 = 0 }

-- Definition for the number of cards
def total_cards : ℕ := set.card cards

-- Definition for the number of multiples of 6
def total_multiples_of_6 : ℕ := set.card multiples_of_6

-- Calculate the probability
def probability : ℚ := total_multiples_of_6 / total_cards

theorem probability_of_drawing_multiple_of_6 : 
  total_cards = 100 ∧ total_multiples_of_6 = 16 → probability = 4 / 25 :=
by 
  sorry

end probability_of_drawing_multiple_of_6_l810_810833


namespace biased_die_sum_is_odd_l810_810873

def biased_die_probabilities : Prop :=
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let scenarios := [
    (1/3) * (2/3)^2,
    (1/3)^3
  ]
  let sum := scenarios.sum
  sum = 13 / 27

theorem biased_die_sum_is_odd :
  biased_die_probabilities := by
    sorry

end biased_die_sum_is_odd_l810_810873


namespace garden_cost_relationship_choosing_strategy_l810_810859

-- Define the conditions and assertions
variable (x : ℝ) (y₁ y₂ : ℝ)

-- Conditions
def condition_x := x > 6
def garden_A_cost := y₁ = 30 * x + 100
def garden_B_cost := y₂ = 25 * x + 150

-- Questions to prove
theorem garden_cost_relationship (hx : condition_x) : garden_A_cost ∧ garden_B_cost :=
by
  split
  sorry

theorem choosing_strategy (hx : condition_x) :
  (y₁ < y₂ ∧ x < 10) ∨ (y₁ = y₂ ∧ x = 10) ∨ (y₂ < y₁ ∧ x > 10) :=
by
  sorry

end garden_cost_relationship_choosing_strategy_l810_810859


namespace not_true_diamond_self_zero_l810_810547

-- Define the operator ⋄
def diamond (x y : ℝ) := |x - 2*y|

-- The problem statement in Lean4
theorem not_true_diamond_self_zero : ¬ (∀ x : ℝ, diamond x x = 0) := by
  sorry

end not_true_diamond_self_zero_l810_810547


namespace count_integer_points_in_region_l810_810567

theorem count_integer_points_in_region :
  ∃ n : ℕ, n = 2551 ∧ ∀ x y : ℤ, y ≤ 3 * x ∧ y ≥ (x : ℚ) / 3 ∧ x + y ≤ 100 → (x, y) ∈ set { (a, b) | a ∈ ℤ ∧ b ∈ ℤ } :=
by
  sorry

end count_integer_points_in_region_l810_810567


namespace sum_of_solutions_l810_810111

theorem sum_of_solutions (y : ℝ) (h : y + 16 / y = 12) : y = 4 ∨ y = 8 → 4 + 8 = 12 :=
by sorry

end sum_of_solutions_l810_810111


namespace combination_eight_choose_five_l810_810138

theorem combination_eight_choose_five : 
  ∀ (n k : ℕ), n = 8 ∧ k = 5 → Nat.choose n k = 56 :=
by 
  intros n k h
  obtain ⟨hn, hk⟩ := h
  rw [hn, hk]
  exact Nat.choose_eq 8 5
  sorry  -- This signifies that the proof needs to be filled in, but we'll skip it as per instructions.

end combination_eight_choose_five_l810_810138


namespace closest_integer_to_cbrt_of_sum_l810_810448

example : (9^3 + 7^3 : ℤ) = 1072 := by
  calc
    (9 : ℤ)^3 + (7 : ℤ)^3 = 729 + 343 := by norm_num
                     ... = 1072 := rfl

theorem closest_integer_to_cbrt_of_sum (x y : ℤ) (hx : x = 9) (hy : y = 7) :
  Int.closest (Real.cbrt (x^3 + y^3)) = 10 :=
by
  -- We will rely on previous example to manage simplifiy
  rw [hx, hy]
  have h : ((9 : ℤ)^3 + (7 : ℤ)^3 : ℤ) = 1072 := by norm_num
  rw h
  -- proof would go here
  sorry

end closest_integer_to_cbrt_of_sum_l810_810448


namespace fat_in_full_cup_of_cream_l810_810375

theorem fat_in_full_cup_of_cream
  (servings_per_recipe : ℕ)
  (half_cup_fat_per_serving : ℕ)
  (half_cup : ℕ)
  (full_cup : ℕ)
  (total_fat_per_half_cup : ℕ)
  (total_fat_per_full_cup : ℕ) :
  servings_per_recipe = 4 →
  half_cup_fat_per_serving = 11 →
  half_cup = 1/2 →
  full_cup = 1 →
  total_fat_per_half_cup = half_cup_fat_per_serving * servings_per_recipe →
  total_fat_per_full_cup = total_fat_per_half_cup * 2 →
  total_fat_per_full_cup = 88 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h2, h1, h5],
  norm_num at *,
  rw [h6],
  norm_num

end fat_in_full_cup_of_cream_l810_810375


namespace zero_points_derivative_midpoint_l810_810191

-- Given function f(x)
def f (a x : Real) : Real := log x - a * x + 1

-- Conditions on a
variable (a : Real)
variable (h_a : a > 0)

-- Theorem about the number of zero points
theorem zero_points (f : Real → Real) (h_f : ∀ x, f x = log x - a x + 1) : 
  (0 < a ∧ a < 1 → ∃ x1 x2, x1 < x2 ∧ f x1 = 0 ∧ f x2 = 0) ∧
  (a = 1 → ∃! x, f x = 0) ∧
  (a > 1 → ∀ x, f x ≠ 0) :=
sorry

-- Theorem about the derivative at the midpoint 
theorem derivative_midpoint (x1 x2 x0 : Real) (h : x1 < x2 ∧ f a x1 = 0 ∧ f a x2 = 0) : 
  let x0 := (x1 + x2) / 2 in
  f a x0 < f a x1 ∧ f a x0 < f a x2 ∧ deriv (f a) x0 < 0 :=
sorry

end zero_points_derivative_midpoint_l810_810191


namespace danny_initial_bottle_caps_l810_810082

theorem danny_initial_bottle_caps :
  ∀ (thrownAway found traded received finalCount initialCount : ℕ),
  thrownAway = 60 →
  found = 58 →
  traded = 15 →
  received = 25 →
  finalCount = 67 →
  initialCount = finalCount - (received - traded) - (found - thrownAway) →
  initialCount = 59 :=
by
  intros thrownAway found traded received finalCount initialCount
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  simp at h6
  linarith
  sorry

end danny_initial_bottle_caps_l810_810082


namespace fraction_simplifies_to_one_l810_810964

theorem fraction_simplifies_to_one :
  (∏ i in (range 16).map (( + ) 1), 1 + 13 / i) / (∏ j in (range 13).map (( + ) 1), 1 + 16 / j) = 1 := by
sorry

end fraction_simplifies_to_one_l810_810964


namespace greatest_x_lcm_105_l810_810792

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l810_810792


namespace boat_travel_distance_l810_810892

variable (x : ℝ) (D : ℝ)

noncomputable def D_equation_1 (x : ℝ) : ℝ := (15.6 - x) * 8
noncomputable def D_equation_2 (x : ℝ) : ℝ := (15.6 + x) * 5

theorem boat_travel_distance (x : ℝ) (h : x = 3.6) : D = 96 := by
  have h1 : D = D_equation_1 x := sorry
  have h2 : D = D_equation_2 x := sorry
  have h3 : D_equation_1 x = D_equation_2 x := by
    rw [h1, h2]
    sorry
  sorry

end boat_travel_distance_l810_810892


namespace discount_rate_2700_l810_810925

def initial_discount (p : ℝ) : ℝ := p * 0.8

def added_discount (p : ℝ) : ℝ := (p / 500).floor * 100

def actual_payment (p : ℝ) : ℝ := initial_discount p - added_discount (initial_discount p)

def actual_discount_rate (marked_price : ℝ) : ℝ := (actual_payment marked_price / marked_price) * 100

theorem discount_rate_2700 :
  actual_discount_rate 2700 ≈ 65 := 
by {
  sorry
}

end discount_rate_2700_l810_810925


namespace expressions_equal_half_l810_810048

theorem expressions_equal_half : 
  (sin (5 * Real.pi / 6) = 1 / 2) ∧ 
  (2 * sin (Real.pi / 12) * cos (Real.pi / 12) = 1 / 2) ∧ 
  (2 * cos (Real.pi / 12) ^ 2 - 1 ≠ 1 / 2) ∧ 
  ((sqrt 3 / 2) * tan (7 * Real.pi / 6) = 1 / 2) := by
  sorry

end expressions_equal_half_l810_810048


namespace ratio_of_height_to_radius_l810_810922

theorem ratio_of_height_to_radius (r h : ℝ)
  (h_cone : r > 0 ∧ h > 0)
  (circumference_cone_base : 20 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2))
  : h / r = Real.sqrt 399 := by
  sorry

end ratio_of_height_to_radius_l810_810922


namespace birthday_friends_count_l810_810369

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l810_810369


namespace car_collision_frequency_l810_810095

theorem car_collision_frequency
  (x : ℝ)
  (h_collision : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * x)
  (h_big_crash : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * 20)
  (h_total_accidents : 240 / x + 240 / 20 = 36) :
  x = 10 :=
by
  sorry

end car_collision_frequency_l810_810095


namespace range_of_k_equation_of_line_existence_of_point_C_l810_810659

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 12 * x + 32 = 0

def line_eq (k x : ℝ) : ℝ := k * x + 2

def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem range_of_k :
  ∀ k : ℝ, 
  ∃ x y : ℝ, 
    circle_eq x y ∧ 
    ∃ (x1 y1 x2 y2 : ℝ), 
    line_eq k x1 = y1 ∧ line_eq k x2 = y2 ∧ 
    x1 ≠ x2 ∧
    -3 / 4 < k ∧ k < 0 :=
sorry

theorem equation_of_line (k : ℝ) :
  ∃ (x1 y1 x2 y2 : ℝ),
    line_eq k x1 = y1 ∧ line_eq k x2 = y2 ∧ 
    dot_product x1 y1 x2 y2 = 28 → 
  l_eq : ∀ x : ℝ, 
    line_eq (-3 + real.sqrt 6) x = l :=
sorry

theorem existence_of_point_C :
  ∃ (C : ℝ × ℝ) (x1 y1 x2 y2 : ℝ),
    C.1 = 0 ∧ 
    circle_eq x1 y1 ∧
    circle_eq x2 y2 ∧
    line_eq (-3 + real.sqrt 6) x1 = y1 ∧ 
    line_eq (-3 + real.sqrt 6) x2 = y2 ∧ 
    dot_product (C.1 - x1) (C.2 - y1) (C.1 - x2) (C.2 - y2) = 36 ∧ 
    C.1 = 0 ∧ C.2 = 2 :=
sorry

end range_of_k_equation_of_line_existence_of_point_C_l810_810659


namespace closest_value_to_division_l810_810980

theorem closest_value_to_division : 
  let x := 805
  let y := 0.410
  let options := [0.4, 4, 40, 400, 4000]
  ∃ (closest : ℕ), closest ∈ options ∧ abs (closest - (x / y)) = list.minimum (options.map (λ opt, abs (opt - (x / y)))) :=
by
  let x := 805
  let y := 0.410
  let options := [0.4, 4, 40, 400, 4000]
  sorry

end closest_value_to_division_l810_810980


namespace find_a_from_conditions_l810_810634

theorem find_a_from_conditions (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + c = 9) 
  (h3 : c = 4) : 
  a = -1 := 
by 
  sorry

end find_a_from_conditions_l810_810634


namespace dagger_evaluation_l810_810227

def dagger (a b : ℚ) : ℚ :=
match a, b with
| ⟨m, n, _, _⟩, ⟨p, q, _, _⟩ => (m * p : ℚ) * (q / n : ℚ)

theorem dagger_evaluation : dagger (3/7) (11/4) = 132/7 := by
  sorry

end dagger_evaluation_l810_810227


namespace prob_both_selected_l810_810470

-- Define the probabilities of selection
def prob_selection_x : ℚ := 1 / 5
def prob_selection_y : ℚ := 2 / 3

-- Prove that the probability that both x and y are selected is 2 / 15
theorem prob_both_selected : prob_selection_x * prob_selection_y = 2 / 15 := 
by
  sorry

end prob_both_selected_l810_810470


namespace cost_of_each_barbell_l810_810288

variables (barbells : ℕ) (money_given money_change : ℝ) (total_cost_per_barbell : ℝ)

-- Given conditions
def conditions := barbells = 3 ∧ money_given = 850 ∧ money_change = 40

-- Theorem statement: Proving the cost of each barbell is $270
theorem cost_of_each_barbell (h : conditions) : total_cost_per_barbell = 270 :=
by
  -- We are using sorry to indicate we are skipping the proof
  sorry

#eval sorry -- Placeholder to verify if the code is syntactically correct

end cost_of_each_barbell_l810_810288


namespace ab_is_square_l810_810379

theorem ab_is_square (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_main : a + b = b * (a - c)) (h_prime : ∃ p : ℕ, Prime p ∧ c + 1 = p^2) :
  ∃ k : ℕ, a + b = k^2 :=
by
  sorry

end ab_is_square_l810_810379


namespace circle_arc_sum_bounds_l810_810435

open Nat

theorem circle_arc_sum_bounds :
  let red_points := 40
  let blue_points := 30
  let green_points := 20
  let total_arcs := 90
  let T := 0 * red_points + 1 * blue_points + 2 * green_points
  let S_min := 6
  let S_max := 140
  (∀ S, (S = 2 * T - A) → (0 ≤ A ∧ A ≤ 134) → (S_min ≤ S ∧ S ≤ S_max))
  → ∃ S_min S_max, S_min = 6 ∧ S_max = 140 :=
by
  intros
  sorry

end circle_arc_sum_bounds_l810_810435


namespace combination_eight_choose_five_l810_810140

theorem combination_eight_choose_five : 
  ∀ (n k : ℕ), n = 8 ∧ k = 5 → Nat.choose n k = 56 :=
by 
  intros n k h
  obtain ⟨hn, hk⟩ := h
  rw [hn, hk]
  exact Nat.choose_eq 8 5
  sorry  -- This signifies that the proof needs to be filled in, but we'll skip it as per instructions.

end combination_eight_choose_five_l810_810140


namespace birthday_friends_count_l810_810347

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l810_810347


namespace area_triangle_ABC_value_a_l810_810002

-- Define the main setup of the problem
variables {a b c A : ℝ}

-- Condition: sides opposite angles A, B, C are respectively a, b, c
-- Given: cos A = (b^2 + c^2 - a^2) / (2bc) = 1 / 2 and bc = 3
def cos_A_condition : Prop := (b^2 + c^2 - a^2) / (2 * b * c) = 1 / 2
def bc_condition : Prop := b * c = 3

-- Prove the area of the triangle ABC
theorem area_triangle_ABC (cos_A : cos_A_condition) (bc : bc_condition) :
  (1/2) * b * c * (sqrt (1 - (1/2)^2)) = sqrt 3 / 2 := by
  sorry

-- Additional condition: c = 1, find the value of a
theorem value_a (cos_A : cos_A_condition) (bc : bc_condition) (h : c = 1) :
  a = 2 * sqrt 5 := by
  sorry

end area_triangle_ABC_value_a_l810_810002


namespace sum_ages_l810_810954

theorem sum_ages (x : ℕ) (h_triple : True) (h_sons_age : ∀ a, a ∈ [16, 16, 16]) (h_beau_age : 42 = 42) :
  3 * (16 - x) = 42 - x → x = 3 := by
  sorry

end sum_ages_l810_810954


namespace greatest_x_lcm_l810_810815

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l810_810815


namespace number_of_friends_l810_810327

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l810_810327


namespace reciprocal_function_passing_point_l810_810493

theorem reciprocal_function_passing_point (n : ℝ) : 
  (∃ (x y : ℝ), y = (n + 5) / x ∧ x = 2 ∧ y = 3) → n = 1 :=
by
  intros h
  cases h with x hx
  cases hx with y hy
  cases hy with hy1 hy2
  cases hy2 with hx2 hy3
  exact sorry

end reciprocal_function_passing_point_l810_810493


namespace simple_interest_principal_l810_810420

theorem simple_interest_principal
  (P_CI : ℝ)
  (r_CI t_CI : ℝ)
  (CI : ℝ)
  (P_SI : ℝ)
  (r_SI t_SI SI : ℝ)
  (h_compound_interest : (CI = P_CI * (1 + r_CI / 100)^t_CI - P_CI))
  (h_simple_interest : SI = (1 / 2) * CI)
  (h_SI_formula : SI = P_SI * r_SI * t_SI / 100) :
  P_SI = 1750 :=
by
  have P_CI := 4000
  have r_CI := 10
  have t_CI := 2
  have r_SI := 8
  have t_SI := 3
  have CI := 840
  have SI := 420
  sorry

end simple_interest_principal_l810_810420


namespace b4_lt_b7_l810_810045

noncomputable def b : ℕ → ℚ
| 1       := 1 + 1 / (1 : ℚ)
| (n + 1) := (1 + 1 / ((b n) + 1 / (1 : ℚ)))

theorem b4_lt_b7 (α : ℕ → ℕ) (hα : ∀ n, α n > 0) : b 4 < b 7 :=
by
  sorry

end b4_lt_b7_l810_810045


namespace number_of_polynomials_correct_l810_810309

noncomputable def number_of_polynomials (n : ℕ) : ℕ :=
  if h : n > 0 then (n / 2) + 1 else 0

theorem number_of_polynomials_correct (n : ℕ) (hn : n > 0) :
  number_of_polynomials n = (n / 2) + 1 :=
by {
  unfold number_of_polynomials,
  split_ifs,
  sorry -- Proof is omitted as specified
}

end number_of_polynomials_correct_l810_810309


namespace tonya_large_lemonade_sales_l810_810846

theorem tonya_large_lemonade_sales 
  (price_small : ℝ)
  (price_medium : ℝ)
  (price_large : ℝ)
  (total_revenue : ℝ)
  (revenue_small : ℝ)
  (revenue_medium : ℝ)
  (n : ℝ)
  (h_price_small : price_small = 1)
  (h_price_medium : price_medium = 2)
  (h_price_large : price_large = 3)
  (h_total_revenue : total_revenue = 50)
  (h_revenue_small : revenue_small = 11)
  (h_revenue_medium : revenue_medium = 24)
  (h_revenue_large : n = (total_revenue - revenue_small - revenue_medium) / price_large) :
  n = 5 :=
sorry

end tonya_large_lemonade_sales_l810_810846


namespace smallest_among_l810_810050

theorem smallest_among {a b c d : ℤ} (h1 : a = -4) (h2 : b = -3) (h3 : c = 0) (h4 : d = 1) :
  a < b ∧ a < c ∧ a < d :=
by
  rw [h1, h2, h3, h4]
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end smallest_among_l810_810050
