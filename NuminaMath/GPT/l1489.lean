import Mathlib

namespace mul_72518_9999_eq_725107482_l1489_148983

theorem mul_72518_9999_eq_725107482 : 72518 * 9999 = 725107482 := by
  sorry

end mul_72518_9999_eq_725107482_l1489_148983


namespace dihedral_angle_equivalence_l1489_148976

namespace CylinderGeometry

variables {α β γ : ℝ} 

-- Given conditions
axiom axial_cross_section : Type
axiom point_on_circumference (C : axial_cross_section) : Prop
axiom dihedral_angle (α: ℝ) : Prop
axiom angle_CAB (β : ℝ) : Prop
axiom angle_CA1B (γ : ℝ) : Prop

-- Proven statement
theorem dihedral_angle_equivalence
    (hx : point_on_circumference C)
    (hα : dihedral_angle α)
    (hβ : angle_CAB β)
    (hγ : angle_CA1B γ):
  α = Real.arcsin (Real.cos β / Real.cos γ) :=
sorry

end CylinderGeometry

end dihedral_angle_equivalence_l1489_148976


namespace methane_needed_l1489_148951

theorem methane_needed (total_benzene_g : ℝ) (molar_mass_benzene : ℝ) (toluene_moles : ℝ) : 
  total_benzene_g = 156 ∧ molar_mass_benzene = 78 ∧ toluene_moles = 2 → 
  toluene_moles = total_benzene_g / molar_mass_benzene := 
by
  intros
  sorry

end methane_needed_l1489_148951


namespace find_n_divisible_by_11_l1489_148955

theorem find_n_divisible_by_11 : ∃ n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 :=
by
  use 1
  -- proof steps would go here, but we're only asked for the statement
  sorry

end find_n_divisible_by_11_l1489_148955


namespace triangle_side_length_l1489_148901

variable (A C : ℝ) (a c b : ℝ)

theorem triangle_side_length (h1 : c = 48) (h2 : a = 27) (h3 : C = 3 * A) : b = 35 := by
  sorry

end triangle_side_length_l1489_148901


namespace cake_angle_between_adjacent_pieces_l1489_148916

theorem cake_angle_between_adjacent_pieces 
  (total_angle : ℝ := 360)
  (total_pieces : ℕ := 10)
  (eaten_pieces : ℕ := 1)
  (angle_per_piece := total_angle / total_pieces)
  (remaining_pieces := total_pieces - eaten_pieces)
  (new_angle_per_piece := total_angle / remaining_pieces) :
  (new_angle_per_piece - angle_per_piece = 4) := 
by
  sorry

end cake_angle_between_adjacent_pieces_l1489_148916


namespace road_completion_l1489_148932

/- 
  The company "Roga and Kopyta" undertook a project to build a road 100 km long. 
  The construction plan is: 
  - In the first month, 1 km of the road will be built.
  - Subsequently, if by the beginning of some month A km is already completed, then during that month an additional 1 / A^10 km of road will be constructed.
  Prove that the road will be completed within 100^11 months.
-/

theorem road_completion (L : ℕ → ℝ) (h1 : L 1 = 1)
  (h2 : ∀ n ≥ 1, L (n + 1) = L n + 1 / (L n) ^ 10) :
  ∃ m ≤ 100 ^ 11, L m ≥ 100 := 
  sorry

end road_completion_l1489_148932


namespace find_d_l1489_148997

-- Definitions for the functions f and g
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

-- Statement to prove the value of d
theorem find_d (c d : ℝ) (h1 : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  -- inserting custom logic for proof
  sorry

end find_d_l1489_148997


namespace inequality_proof_l1489_148950

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_abc : a * b * c = 1)

theorem inequality_proof :
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) 
  ≥ (3 / 2) + (1 / 4) * (a * (c - b) ^ 2 / (c + b) + b * (c - a) ^ 2 / (c + a) + c * (b - a) ^ 2 / (b + a)) :=
by
  sorry

end inequality_proof_l1489_148950


namespace elena_pen_cost_l1489_148936

theorem elena_pen_cost (cost_X : ℝ) (cost_Y : ℝ) (total_pens : ℕ) (brand_X_pens : ℕ) 
    (purchased_X_cost : cost_X = 4.0) (purchased_Y_cost : cost_Y = 2.8)
    (total_pens_condition : total_pens = 12) (brand_X_pens_condition : brand_X_pens = 8) :
    cost_X * brand_X_pens + cost_Y * (total_pens - brand_X_pens) = 43.20 :=
    sorry

end elena_pen_cost_l1489_148936


namespace boat_speed_in_still_water_l1489_148915

variable (b r d v t : ℝ)

theorem boat_speed_in_still_water (hr : r = 3) 
                                 (hd : d = 3.6) 
                                 (ht : t = 1/5) 
                                 (hv : v = b + r) 
                                 (dist_eq : d = v * t) : 
  b = 15 := 
by
  sorry

end boat_speed_in_still_water_l1489_148915


namespace solution_set_max_value_l1489_148993

-- Given function f(x)
def f (x : ℝ) : ℝ := |2 * x - 1| + |x - 1|

-- (I) Prove the solution set of f(x) ≤ 4 is {x | -2/3 ≤ x ≤ 2}
theorem solution_set : {x : ℝ | f x ≤ 4} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 2} :=
sorry

-- (II) Given m is the minimum value of f(x)
def m := 1 / 2

-- Given a, b, c ∈ ℝ^+ and a + b + c = m
variables (a b c : ℝ)
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a + b + c = m)

-- Prove the maximum value of √(2a + 1) + √(2b + 1) + √(2c + 1) is 2√3
theorem max_value : (Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) + Real.sqrt (2 * c + 1)) ≤ 2 * Real.sqrt 3 :=
sorry

end solution_set_max_value_l1489_148993


namespace find_angle_measure_l1489_148988

theorem find_angle_measure (x : ℝ) (h : x = 2 * (90 - x) + 30) : x = 70 :=
by
  exact sorry

end find_angle_measure_l1489_148988


namespace max_product_sum_300_l1489_148987

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l1489_148987


namespace negation_is_false_l1489_148913

-- Define the proposition and its negation
def proposition (x y : ℝ) : Prop := (x > 2 ∧ y > 3) → (x + y > 5)
def negation_proposition (x y : ℝ) : Prop := ¬ proposition x y

-- The proposition and its negation
theorem negation_is_false : ∀ (x y : ℝ), negation_proposition x y = false :=
by sorry

end negation_is_false_l1489_148913


namespace original_number_of_matchsticks_l1489_148974

-- Define the conditions
def matchsticks_per_house : ℕ := 10
def houses_created : ℕ := 30
def total_matchsticks_used := houses_created * matchsticks_per_house

-- Define the question and the proof goal
theorem original_number_of_matchsticks (h : total_matchsticks_used = (Michael's_original_matchsticks / 2)) :
  (Michael's_original_matchsticks = 600) :=
by
  sorry

end original_number_of_matchsticks_l1489_148974


namespace sum_of_roots_eq_two_l1489_148945

theorem sum_of_roots_eq_two {b x1 x2 : ℝ} 
  (h : x1 ^ 2 - 2 * x1 + b = 0) 
  (k : x2 ^ 2 - 2 * x2 + b = 0) 
  (neq : x1 ≠ x2) : 
  x1 + x2 = 2 := 
sorry

end sum_of_roots_eq_two_l1489_148945


namespace find_m_values_l1489_148906

-- Defining the sets and conditions
def A : Set ℝ := { x | x ^ 2 - 9 * x - 10 = 0 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- Stating the proof problem
theorem find_m_values : {m | A ∪ B m = A} = {0, 1, -1 / 10} :=
by
  sorry

end find_m_values_l1489_148906


namespace trapezoid_area_division_l1489_148908

/-- Given a trapezoid where one base is 150 units longer than the other base and the segment joining the midpoints of the legs divides the trapezoid into two regions whose areas are in the ratio 3:4, prove that the greatest integer less than or equal to (x^2 / 150) is 300, where x is the length of the segment that joins the midpoints of the legs and divides the trapezoid into two equal areas. -/
theorem trapezoid_area_division (b h x : ℝ) (h_b : b = 112.5) (h_x : x = 150) :
  ⌊x^2 / 150⌋ = 300 :=
by
  sorry

end trapezoid_area_division_l1489_148908


namespace sum_of_bases_l1489_148931

theorem sum_of_bases (R_1 R_2 : ℕ) 
  (hF1 : (4 * R_1 + 8) / (R_1 ^ 2 - 1) = (3 * R_2 + 6) / (R_2 ^ 2 - 1))
  (hF2 : (8 * R_1 + 4) / (R_1 ^ 2 - 1) = (6 * R_2 + 3) / (R_2 ^ 2 - 1)) : 
  R_1 + R_2 = 21 :=
sorry

end sum_of_bases_l1489_148931


namespace rachel_makes_money_l1489_148930

theorem rachel_makes_money (cost_per_bar total_bars remaining_bars : ℕ) (h_cost : cost_per_bar = 2) (h_total : total_bars = 13) (h_remaining : remaining_bars = 4) :
  cost_per_bar * (total_bars - remaining_bars) = 18 :=
by 
  sorry

end rachel_makes_money_l1489_148930


namespace train_arrival_problem_shooting_problem_l1489_148971

-- Define trials and outcome types
inductive OutcomeTrain : Type
| onTime
| notOnTime

inductive OutcomeShooting : Type
| hitTarget
| missTarget

-- Scenario 1: Train Arrival Problem
def train_arrival_trials_refers_to (n : Nat) : Prop := 
  ∃ trials : List OutcomeTrain, trials.length = 3

-- Scenario 2: Shooting Problem
def shooting_trials_refers_to (n : Nat) : Prop :=
  ∃ trials : List OutcomeShooting, trials.length = 2

theorem train_arrival_problem : train_arrival_trials_refers_to 3 :=
by
  sorry

theorem shooting_problem : shooting_trials_refers_to 2 :=
by
  sorry

end train_arrival_problem_shooting_problem_l1489_148971


namespace trig_expression_value_l1489_148992

theorem trig_expression_value (θ : ℝ) (h1 : Real.tan (2 * θ) = -2 * Real.sqrt 2)
  (h2 : 2 * θ > Real.pi / 2 ∧ 2 * θ < Real.pi) : 
  (2 * Real.cos θ / 2 ^ 2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 2 * Real.sqrt 2 - 3 :=
by
  sorry

end trig_expression_value_l1489_148992


namespace imaginary_part_of_z_l1489_148978

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_of_z : Complex.im z = -2 := by
  sorry

end imaginary_part_of_z_l1489_148978


namespace steve_needs_28_feet_of_wood_l1489_148985

-- Define the required lengths
def lengths_4_feet : Nat := 6
def lengths_2_feet : Nat := 2

-- Define the wood length in feet for each type
def wood_length_4 : Nat := 4
def wood_length_2 : Nat := 2

-- Total feet of wood required
def total_wood : Nat := lengths_4_feet * wood_length_4 + lengths_2_feet * wood_length_2

-- The theorem to prove that the total amount of wood required is 28 feet
theorem steve_needs_28_feet_of_wood : total_wood = 28 :=
by
  sorry

end steve_needs_28_feet_of_wood_l1489_148985


namespace perfect_square_trinomial_l1489_148929

theorem perfect_square_trinomial (m : ℝ) :
  (∃ (a : ℝ), (x^2 + mx + 1) = (x + a)^2) ↔ (m = 2 ∨ m = -2) := sorry

end perfect_square_trinomial_l1489_148929


namespace right_triangle_perimeter_l1489_148917

def right_triangle_circumscribed_perimeter (r c : ℝ) (a b : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter : 
  ∀ (a b : ℝ),
  (4 : ℝ) * (a + b + (26 : ℝ)) = a * b ∧ a^2 + b^2 = (26 : ℝ)^2 →
  right_triangle_circumscribed_perimeter 4 26 a b = 60 := sorry

end right_triangle_perimeter_l1489_148917


namespace Jane_mom_jars_needed_l1489_148952

theorem Jane_mom_jars_needed : 
  ∀ (total_tomatoes jar_capacity : ℕ), 
  total_tomatoes = 550 → 
  jar_capacity = 14 → 
  ⌈(total_tomatoes: ℚ) / jar_capacity⌉ = 40 := 
by 
  intros total_tomatoes jar_capacity htotal hcapacity
  sorry

end Jane_mom_jars_needed_l1489_148952


namespace gross_pay_calculation_l1489_148927

theorem gross_pay_calculation
    (NetPay : ℕ) (Taxes : ℕ) (GrossPay : ℕ) 
    (h1 : NetPay = 315) 
    (h2 : Taxes = 135) 
    (h3 : GrossPay = NetPay + Taxes) : 
    GrossPay = 450 :=
by
    -- We need to prove this part
    sorry

end gross_pay_calculation_l1489_148927


namespace triangle_perimeter_ABC_l1489_148949

noncomputable def perimeter_triangle (AP PB r : ℕ) (hAP : AP = 23) (hPB : PB = 27) (hr : r = 21) : ℕ :=
  2 * (50 + 245 / 2)

theorem triangle_perimeter_ABC (AP PB r : ℕ) 
  (hAP : AP = 23) 
  (hPB : PB = 27) 
  (hr : r = 21) : 
  perimeter_triangle AP PB r hAP hPB hr = 345 :=
by
  sorry

end triangle_perimeter_ABC_l1489_148949


namespace one_quarter_between_l1489_148979

def one_quarter_way (a b : ℚ) : ℚ :=
  a + 1 / 4 * (b - a)

theorem one_quarter_between :
  one_quarter_way (1 / 7) (1 / 4) = 23 / 112 :=
by
  sorry

end one_quarter_between_l1489_148979


namespace find_m_if_f_even_l1489_148956

variable (m : ℝ)

def f (x : ℝ) : ℝ := x^2 + (m + 2) * x + 3

theorem find_m_if_f_even (h : ∀ x, f m x = f m (-x)) : m = -2 :=
by
  sorry

end find_m_if_f_even_l1489_148956


namespace negation_proof_l1489_148946

open Classical

variable {x : ℝ}

theorem negation_proof :
  (∀ x : ℝ, (x + 1) ≥ 0 ∧ (x^2 - x) ≤ 0) ↔ ¬ (∃ x_0 : ℝ, (x_0 + 1) < 0 ∨ (x_0^2 - x_0) > 0) := 
by
  sorry

end negation_proof_l1489_148946


namespace boxes_needed_l1489_148967

-- Define Marilyn's total number of bananas
def num_bananas : Nat := 40

-- Define the number of bananas per box
def bananas_per_box : Nat := 5

-- Calculate the number of boxes required for the given number of bananas and bananas per box
def num_boxes (total_bananas : Nat) (bananas_each_box : Nat) : Nat :=
  total_bananas / bananas_each_box

-- Statement to be proved: given the specific conditions, the result should be 8
theorem boxes_needed : num_boxes num_bananas bananas_per_box = 8 :=
sorry

end boxes_needed_l1489_148967


namespace problem_product_of_areas_eq_3600x6_l1489_148941

theorem problem_product_of_areas_eq_3600x6 
  (x : ℝ) 
  (bottom_area : ℝ) 
  (side_area : ℝ) 
  (front_area : ℝ)
  (bottom_area_eq : bottom_area = 12 * x ^ 2)
  (side_area_eq : side_area = 15 * x ^ 2)
  (front_area_eq : front_area = 20 * x ^ 2)
  (dimensions_proportional : ∃ a b c : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x 
                            ∧ bottom_area = a * b ∧ side_area = a * c ∧ front_area = b * c)
  : bottom_area * side_area * front_area = 3600 * x ^ 6 :=
by 
  -- Proof omitted
  sorry

end problem_product_of_areas_eq_3600x6_l1489_148941


namespace soda_cans_purchasable_l1489_148991

theorem soda_cans_purchasable (S Q : ℕ) (t D : ℝ) (hQ_pos : Q > 0) :
    let quarters_from_dollars := 4 * D
    let total_quarters_with_tax := quarters_from_dollars * (1 + t)
    (total_quarters_with_tax / Q) * S = (4 * D * S * (1 + t)) / Q :=
sorry

end soda_cans_purchasable_l1489_148991


namespace find_k_for_sum_of_cubes_l1489_148996

theorem find_k_for_sum_of_cubes (k : ℝ) (r s : ℝ)
  (h1 : r + s = -2)
  (h2 : r * s = k / 3)
  (h3 : r^3 + s^3 = r + s) : k = 3 :=
by
  -- Sorry will be replaced by the actual proof
  sorry

end find_k_for_sum_of_cubes_l1489_148996


namespace guilt_proof_l1489_148907

theorem guilt_proof (X Y : Prop) (h1 : X ∨ Y) (h2 : ¬X) : Y :=
by
  sorry

end guilt_proof_l1489_148907


namespace probability_genuine_given_equal_weight_l1489_148960

noncomputable def total_coins : ℕ := 15
noncomputable def genuine_coins : ℕ := 12
noncomputable def counterfeit_coins : ℕ := 3

def condition_A : Prop := true
def condition_B (weights : Fin 6 → ℝ) : Prop :=
  weights 0 + weights 1 = weights 2 + weights 3 ∧
  weights 0 + weights 1 = weights 4 + weights 5

noncomputable def P_A_and_B : ℚ := (44 / 70) * (15 / 26) * (28 / 55)
noncomputable def P_B : ℚ := 44 / 70

theorem probability_genuine_given_equal_weight :
  P_A_and_B / P_B = 264 / 443 :=
by
  sorry

end probability_genuine_given_equal_weight_l1489_148960


namespace intervals_of_monotonicity_and_extreme_values_l1489_148989

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < 1 → deriv f x > 0) ∧
  (∀ x : ℝ, x > 1 → deriv f x < 0) ∧
  (∀ x : ℝ, f 1 = 1 / Real.exp 1) :=
by
  sorry

end intervals_of_monotonicity_and_extreme_values_l1489_148989


namespace robin_spent_on_leftover_drinks_l1489_148963

-- Define the number of each type of drink bought and consumed
def sodas_bought : Nat := 30
def sodas_price : Nat := 2
def sodas_consumed : Nat := 10

def energy_drinks_bought : Nat := 20
def energy_drinks_price : Nat := 3
def energy_drinks_consumed : Nat := 14

def smoothies_bought : Nat := 15
def smoothies_price : Nat := 4
def smoothies_consumed : Nat := 5

-- Define the total cost calculation
def total_spent_on_leftover_drinks : Nat :=
  (sodas_bought * sodas_price - sodas_consumed * sodas_price) +
  (energy_drinks_bought * energy_drinks_price - energy_drinks_consumed * energy_drinks_price) +
  (smoothies_bought * smoothies_price - smoothies_consumed * smoothies_price)

theorem robin_spent_on_leftover_drinks : total_spent_on_leftover_drinks = 98 := by
  -- Provide the proof steps here (not required for this task)
  sorry

end robin_spent_on_leftover_drinks_l1489_148963


namespace find_y_l1489_148966

open Real

def vecV (y : ℝ) : ℝ × ℝ := (1, y)
def vecW : ℝ × ℝ := (6, 4)

noncomputable def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dotProduct v w) / (dotProduct w w)
  (scalar * w.1, scalar * w.2)

theorem find_y (y : ℝ) (h : projection (vecV y) vecW = (3, 2)) : y = 5 := by
  sorry

end find_y_l1489_148966


namespace proof_problem_l1489_148912

open Set

variable (U : Set ℕ)
variable (P : Set ℕ)
variable (Q : Set ℕ)

noncomputable def problem_statement : Set ℕ :=
  compl (P ∪ Q) ∩ U

theorem proof_problem :
  U = {1, 2, 3, 4} →
  P = {1, 2} →
  Q = {2, 3} →
  compl (P ∪ Q) ∩ U = {4} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end proof_problem_l1489_148912


namespace speed_of_stream_l1489_148938

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 11) (h2 : upstream_speed = 8) : 
    (downstream_speed - upstream_speed) / 2 = 1.5 :=
by
  rw [h1, h2]
  simp
  norm_num

end speed_of_stream_l1489_148938


namespace symmetry_proof_l1489_148922

-- Define the initial point P and its reflection P' about the x-axis
def P : ℝ × ℝ := (-1, 2)
def P' : ℝ × ℝ := (-1, -2)

-- Define the property of symmetry about the x-axis
def symmetric_about_x_axis (P P' : ℝ × ℝ) : Prop :=
  P'.fst = P.fst ∧ P'.snd = -P.snd

-- The theorem to prove that point P' is symmetric to point P about the x-axis
theorem symmetry_proof : symmetric_about_x_axis P P' :=
  sorry

end symmetry_proof_l1489_148922


namespace prob_at_least_one_multiple_of_4_60_l1489_148911

def num_multiples_of_4 (n : ℕ) : ℕ :=
  n / 4

def total_numbers_in_range (n : ℕ) : ℕ :=
  n

def num_not_multiples_of_4 (n : ℕ) : ℕ :=
  total_numbers_in_range n - num_multiples_of_4 n

def prob_no_multiple_of_4 (n : ℕ) : ℚ :=
  let p := num_not_multiples_of_4 n / total_numbers_in_range n
  p * p

def prob_at_least_one_multiple_of_4 (n : ℕ) : ℚ :=
  1 - prob_no_multiple_of_4 n

theorem prob_at_least_one_multiple_of_4_60 :
  prob_at_least_one_multiple_of_4 60 = 7 / 16 :=
by
  -- Proof is skipped.
  sorry

end prob_at_least_one_multiple_of_4_60_l1489_148911


namespace correct_statements_arithmetic_seq_l1489_148947

/-- For an arithmetic sequence {a_n} with a1 > 0 and common difference d ≠ 0, 
    the correct statements among options A, B, C, and D are B and C. -/
theorem correct_statements_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2) (h_a1_pos : a 1 > 0) (h_d_ne_0 : d ≠ 0) : 
  (S 5 = S 9 → 
   S 7 = (10 * a 4) / 2) ∧ 
  (S 6 > S 7 → S 7 > S 8) := 
sorry

end correct_statements_arithmetic_seq_l1489_148947


namespace rob_travel_time_to_park_l1489_148973

theorem rob_travel_time_to_park : 
  ∃ R : ℝ, 
    (∀ Tm : ℝ, Tm = 3 * R) ∧ -- Mark's travel time is three times Rob's travel time
    (∀ Tr : ℝ, Tm - 2 = R) → -- Considering Mark's head start of 2 hours
    R = 1 :=
sorry

end rob_travel_time_to_park_l1489_148973


namespace rectangle_area_excluding_hole_l1489_148923

theorem rectangle_area_excluding_hole (x : ℝ) (h : x > 5 / 3) :
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  A_large - A_hole = -x^2 + 17 * x + 38 :=
by
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  sorry

end rectangle_area_excluding_hole_l1489_148923


namespace average_age_nine_students_l1489_148935

theorem average_age_nine_students (total_age_15_students : ℕ)
                                (total_age_5_students : ℕ)
                                (age_15th_student : ℕ)
                                (h1 : total_age_15_students = 225)
                                (h2 : total_age_5_students = 65)
                                (h3 : age_15th_student = 16) :
                                (total_age_15_students - total_age_5_students - age_15th_student) / 9 = 16 := by
  sorry

end average_age_nine_students_l1489_148935


namespace shop_length_l1489_148954

def monthly_rent : ℝ := 2244
def width : ℝ := 18
def annual_rent_per_sqft : ℝ := 68

theorem shop_length : 
  (monthly_rent * 12 / annual_rent_per_sqft / width) = 22 := 
by
  -- Proof omitted
  sorry

end shop_length_l1489_148954


namespace cyclists_meet_at_starting_point_l1489_148919

/--
Given a circular track of length 1200 meters, and three cyclists with speeds of 36 kmph, 54 kmph, and 72 kmph,
prove that all three cyclists will meet at the starting point for the first time after 4 minutes.
-/
theorem cyclists_meet_at_starting_point :
  let track_length := 1200
  let speed_a_kmph := 36
  let speed_b_kmph := 54
  let speed_c_kmph := 72
  
  let speed_a_m_per_min := speed_a_kmph * 1000 / 60
  let speed_b_m_per_min := speed_b_kmph * 1000 / 60
  let speed_c_m_per_min := speed_c_kmph * 1000 / 60
  
  let time_a := track_length / speed_a_m_per_min
  let time_b := track_length / speed_b_m_per_min
  let time_c := track_length / speed_c_m_per_min
  
  let lcm := (2 : ℚ)

  (time_a = 2) ∧ (time_b = 4 / 3) ∧ (time_c = 1) → 
  ∀ t, t = lcm * 3 → t = 12 / 3 → t = 4 :=
by
  sorry

end cyclists_meet_at_starting_point_l1489_148919


namespace original_numbers_product_l1489_148953

theorem original_numbers_product (a b c d x : ℕ) 
  (h1 : a + b + c + d = 243)
  (h2 : a + 8 = x)
  (h3 : b - 8 = x)
  (h4 : c * 8 = x)
  (h5 : d / 8 = x) : 
  (min (min a (min b (min c d))) * max a (max b (max c d))) = 576 :=
by 
  sorry

end original_numbers_product_l1489_148953


namespace smallest_possible_value_of_N_l1489_148943

-- Conditions definition:
variable (a b c d : ℕ)
variable (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
variable (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
variable (gcd_ab : Int.gcd a b = 1)
variable (gcd_ac : Int.gcd a c = 2)
variable (gcd_ad : Int.gcd a d = 4)
variable (gcd_bc : Int.gcd b c = 5)
variable (gcd_bd : Int.gcd b d = 3)
variable (gcd_cd : Int.gcd c d = N)
variable (hN : N > 5)

-- Statement to prove:
theorem smallest_possible_value_of_N : N = 14 := sorry

end smallest_possible_value_of_N_l1489_148943


namespace probability_neither_test_l1489_148994

theorem probability_neither_test (P_hist : ℚ) (P_geo : ℚ) (indep : Prop) 
  (H1 : P_hist = 5/9) (H2 : P_geo = 1/3) (H3 : indep) :
  (1 - P_hist) * (1 - P_geo) = 8/27 := by
  sorry

end probability_neither_test_l1489_148994


namespace negation_of_p_is_neg_p_l1489_148984

-- Define the original proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Define what it means for the negation of p to be satisfied
def neg_p := ∀ n : ℕ, 2^n ≤ 100

-- Statement to prove the logical equivalence between the negation of p and neg_p
theorem negation_of_p_is_neg_p : ¬ p ↔ neg_p := by
  sorry

end negation_of_p_is_neg_p_l1489_148984


namespace original_price_of_article_l1489_148995

theorem original_price_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 := 
by 
  sorry

end original_price_of_article_l1489_148995


namespace pete_bus_ride_blocks_l1489_148928

theorem pete_bus_ride_blocks : 
  ∀ (total_walk_blocks bus_blocks total_blocks : ℕ), 
  total_walk_blocks = 10 → 
  total_blocks = 50 → 
  total_walk_blocks + 2 * bus_blocks = total_blocks → 
  bus_blocks = 20 :=
by
  intros total_walk_blocks bus_blocks total_blocks h1 h2 h3
  sorry

end pete_bus_ride_blocks_l1489_148928


namespace bijection_condition_l1489_148937

variable {n m : ℕ}
variable (f : Fin n → Fin n)

theorem bijection_condition (h_even : m % 2 = 0)
(h_prime : Nat.Prime (n + 1))
(h_bij : Function.Bijective f) :
  ∀ x y : Fin n, (n : ℕ) ∣ (m * x - y : ℕ) → (n + 1) ∣ (f x).val ^ m - (f y).val := sorry

end bijection_condition_l1489_148937


namespace fixed_monthly_fee_l1489_148977

theorem fixed_monthly_fee :
  ∀ (x y : ℝ), 
  x + y = 20.00 → 
  x + 2 * y = 30.00 → 
  x + 3 * y = 40.00 → 
  x = 10.00 :=
by
  intros x y H1 H2 H3
  -- Proof can be filled out here
  sorry

end fixed_monthly_fee_l1489_148977


namespace amoeba_growth_after_5_days_l1489_148905

theorem amoeba_growth_after_5_days : (3 : ℕ)^5 = 243 := by
  sorry

end amoeba_growth_after_5_days_l1489_148905


namespace average_headcount_correct_l1489_148959

def avg_headcount_03_04 : ℕ := 11500
def avg_headcount_04_05 : ℕ := 11600
def avg_headcount_05_06 : ℕ := 11300

noncomputable def average_headcount : ℕ :=
  (avg_headcount_03_04 + avg_headcount_04_05 + avg_headcount_05_06) / 3

theorem average_headcount_correct :
  average_headcount = 11467 :=
by
  sorry

end average_headcount_correct_l1489_148959


namespace probability_both_heads_l1489_148975

-- Define the sample space and the probability of each outcome
def sample_space : List (Bool × Bool) := [(true, true), (true, false), (false, true), (false, false)]

-- Define the function to check for both heads
def both_heads (outcome : Bool × Bool) : Bool :=
  outcome = (true, true)

-- Calculate the probability of both heads
theorem probability_both_heads :
  (sample_space.filter both_heads).length / sample_space.length = 1 / 4 := sorry

end probability_both_heads_l1489_148975


namespace find_age_l1489_148903

theorem find_age (x : ℕ) (h : 5 * (x + 5) - 5 * (x - 5) = x) : x = 50 :=
by
  sorry

end find_age_l1489_148903


namespace pastries_and_juices_count_l1489_148902

theorem pastries_and_juices_count 
  (budget : ℕ) 
  (cost_per_pastry : ℕ) 
  (cost_per_juice : ℕ) 
  (total_money : budget = 50)
  (pastry_cost : cost_per_pastry = 7) 
  (juice_cost : cost_per_juice = 2) : 
  ∃ (p j : ℕ), 7 * p + 2 * j ≤ 50 ∧ p + j = 7 :=
by
  sorry

end pastries_and_juices_count_l1489_148902


namespace number_of_members_l1489_148900

theorem number_of_members (n : ℕ) (h1 : n * n = 5929) : n = 77 :=
sorry

end number_of_members_l1489_148900


namespace cost_per_meter_l1489_148910

-- Defining the parameters and their relationships
def length : ℝ := 58
def breadth : ℝ := length - 16
def total_cost : ℝ := 5300
def perimeter : ℝ := 2 * (length + breadth)

-- Proving the cost per meter of fencing
theorem cost_per_meter : total_cost / perimeter = 26.50 := 
by
  sorry

end cost_per_meter_l1489_148910


namespace solve_equation_l1489_148999

theorem solve_equation (Y : ℝ) : (3.242 * 10 * Y) / 100 = 0.3242 * Y := 
by 
  sorry

end solve_equation_l1489_148999


namespace is_hexagonal_number_2016_l1489_148970

theorem is_hexagonal_number_2016 :
  ∃ (n : ℕ), 2 * n^2 - n = 2016 :=
sorry

end is_hexagonal_number_2016_l1489_148970


namespace butterflies_equal_distribution_l1489_148972

theorem butterflies_equal_distribution (N : ℕ) : (∃ t : ℕ, 
    (N - t) % 8 = 0 ∧ (N - t) / 8 > 0) ↔ ∃ k : ℕ, N = 45 * k :=
by sorry

end butterflies_equal_distribution_l1489_148972


namespace express_h_l1489_148942

variable (a b S h : ℝ)
variable (h_formula : S = 1/2 * (a + b) * h)
variable (h_nonzero : a + b ≠ 0)

theorem express_h : h = 2 * S / (a + b) := 
by 
  sorry

end express_h_l1489_148942


namespace express_q_as_polynomial_l1489_148914

def q (x : ℝ) : ℝ := x^3 + 4

theorem express_q_as_polynomial (x : ℝ) : 
  q x + (2 * x^6 + x^5 + 4 * x^4 + 6 * x^2) = (5 * x^4 + 10 * x^3 - x^2 + 8 * x + 15) → 
  q x = -2 * x^6 - x^5 + x^4 + 10 * x^3 - 7 * x^2 + 8 * x + 15 := by
  sorry

end express_q_as_polynomial_l1489_148914


namespace harriet_siblings_product_l1489_148965

-- Definitions based on conditions
def Harry_sisters : ℕ := 6
def Harry_brothers : ℕ := 3
def Harriet_sisters : ℕ := Harry_sisters - 1
def Harriet_brothers : ℕ := Harry_brothers

-- Statement to prove
theorem harriet_siblings_product : Harriet_sisters * Harriet_brothers = 15 := by
  -- Proof is skipped
  sorry

end harriet_siblings_product_l1489_148965


namespace degree_f_x2_g_x3_l1489_148969

open Polynomial

noncomputable def degree_of_composite_polynomials (f g : Polynomial ℝ) : ℕ :=
  let f_degree := Polynomial.degree f
  let g_degree := Polynomial.degree g
  match (f_degree, g_degree) with
  | (some 3, some 6) => 24
  | _ => 0

theorem degree_f_x2_g_x3 (f g : Polynomial ℝ) (h_f : Polynomial.degree f = 3) (h_g : Polynomial.degree g = 6) :
  Polynomial.degree (Polynomial.comp f (X^2) * Polynomial.comp g (X^3)) = 24 := by
  -- content Logic Here
  sorry

end degree_f_x2_g_x3_l1489_148969


namespace inequality_holds_for_all_y_l1489_148925

theorem inequality_holds_for_all_y (x : ℝ) :
  (∀ y : ℝ, y^2 - (5^x - 1) * (y - 1) > 0) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end inequality_holds_for_all_y_l1489_148925


namespace angle_B_l1489_148986

theorem angle_B (a b c A B : ℝ) (h : a * Real.cos B - b * Real.cos A = c) (C : ℝ) (hC : C = Real.pi / 5) (h_triangle : A + B + C = Real.pi) : B = 3 * Real.pi / 10 :=
sorry

end angle_B_l1489_148986


namespace sum_of_primes_product_166_l1489_148957

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m < n → m > 0 → n % m ≠ 0

theorem sum_of_primes_product_166
    (p1 p2 : ℕ)
    (prime_p1 : is_prime p1)
    (prime_p2 : is_prime p2)
    (product_condition : p1 * p2 = 166) :
    p1 + p2 = 85 :=
    sorry

end sum_of_primes_product_166_l1489_148957


namespace sum_of_squares_l1489_148933

theorem sum_of_squares (b j s : ℕ) (h : b + j + s = 34) : b^2 + j^2 + s^2 = 406 :=
sorry

end sum_of_squares_l1489_148933


namespace bobby_total_l1489_148981

-- Define the conditions
def initial_candy : ℕ := 33
def additional_candy : ℕ := 4
def chocolate : ℕ := 14

-- Define the total pieces of candy Bobby ate
def total_candy : ℕ := initial_candy + additional_candy

-- Define the total pieces of candy and chocolate Bobby ate
def total_candy_and_chocolate : ℕ := total_candy + chocolate

-- Theorem to prove the total pieces of candy and chocolate Bobby ate
theorem bobby_total : total_candy_and_chocolate = 51 :=
by sorry

end bobby_total_l1489_148981


namespace sqrt_ac_bd_le_sqrt_ef_l1489_148962

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_ac_bd_le_sqrt_ef
  (a b c d e f : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f)
  (h1 : a + b ≤ e)
  (h2 : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) :=
by
  sorry

end sqrt_ac_bd_le_sqrt_ef_l1489_148962


namespace circles_tangent_l1489_148909

theorem circles_tangent (m : ℝ) :
  (∀ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9 → 
                (x + 1)^2 + (y - m)^2 = 4 →
                ∃ m, m = -1 ∨ m = -2) := 
sorry

end circles_tangent_l1489_148909


namespace golden_ratio_in_range_l1489_148990

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_ratio_in_range : 0.6 < golden_ratio ∧ golden_ratio < 0.7 :=
by
  sorry

end golden_ratio_in_range_l1489_148990


namespace woman_lawyer_probability_l1489_148948

theorem woman_lawyer_probability (total_members women_count lawyer_prob : ℝ) 
  (h1: total_members = 100) 
  (h2: women_count = 0.70 * total_members) 
  (h3: lawyer_prob = 0.40) : 
  (0.40 * 0.70) = 0.28 := by sorry

end woman_lawyer_probability_l1489_148948


namespace common_intersection_implies_cd_l1489_148958

theorem common_intersection_implies_cd (a b c d : ℝ) (h : a ≠ b) (x y : ℝ) 
  (H1 : y = a * x + a) (H2 : y = b * x + b) (H3 : y = c * x + d) : c = d := by
  sorry

end common_intersection_implies_cd_l1489_148958


namespace vacation_cost_correct_l1489_148961

namespace VacationCost

-- Define constants based on conditions
def starting_charge_per_dog : ℝ := 2
def charge_per_block : ℝ := 1.25
def number_of_dogs : ℕ := 20
def total_blocks : ℕ := 128
def family_members : ℕ := 5

-- Define total earnings from walking dogs
def total_earnings : ℝ :=
  (number_of_dogs * starting_charge_per_dog) + (total_blocks * charge_per_block)

-- Define the total cost of the vacation
noncomputable def total_cost_of_vacation : ℝ :=
  total_earnings / family_members * family_members

-- Proof statement: The total cost of the vacation is $200
theorem vacation_cost_correct : total_cost_of_vacation = 200 := by
  sorry

end VacationCost

end vacation_cost_correct_l1489_148961


namespace dave_earnings_l1489_148940

def total_games : Nat := 10
def non_working_games : Nat := 2
def price_per_game : Nat := 4
def working_games : Nat := total_games - non_working_games
def money_earned : Nat := working_games * price_per_game

theorem dave_earnings : money_earned = 32 := by
  sorry

end dave_earnings_l1489_148940


namespace work_completion_time_l1489_148934

noncomputable def work_rate_A : ℚ := 1 / 12
noncomputable def work_rate_B : ℚ := 1 / 14

theorem work_completion_time : 
  (work_rate_A + work_rate_B)⁻¹ = 84 / 13 := by
  sorry

end work_completion_time_l1489_148934


namespace problem1_problem2_problem3_problem4_l1489_148968

theorem problem1 : 
  (3 / 5 : ℚ) - ((2 / 15) + (1 / 3)) = (2 / 15) := 
  by 
  sorry

theorem problem2 : 
  (-2 : ℤ) - 12 * ((1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 2 : ℚ)) = -8 := 
  by 
  sorry

theorem problem3 : 
  (2 : ℤ) * (-3) ^ 2 - (6 / (-2) : ℚ) * (-1 / 3) = 17 := 
  by 
  sorry

theorem problem4 : 
  (-1 ^ 4 : ℤ) + ((abs (2 ^ 3 - 10)) : ℤ) - ((-3 : ℤ) / (-1) ^ 2019) = -2 := 
  by 
  sorry

end problem1_problem2_problem3_problem4_l1489_148968


namespace difference_between_sevens_l1489_148918

-- Define the numeral
def numeral : ℕ := 54179759

-- Define a function to find the place value of a digit at a specific position in a number
def place_value (n : ℕ) (pos : ℕ) : ℕ :=
  let digit := (n / 10^pos) % 10
  digit * 10^pos

-- Define specific place values for the two sevens
def first_seven_place : ℕ := place_value numeral 4  -- Ten-thousands place
def second_seven_place : ℕ := place_value numeral 1 -- Tens place

-- Define their values
def first_seven_value : ℕ := 7 * 10^4  -- 70,000
def second_seven_value : ℕ := 7 * 10^1  -- 70

-- Prove the difference between these place values
theorem difference_between_sevens : first_seven_value - second_seven_value = 69930 := by
  sorry

end difference_between_sevens_l1489_148918


namespace katie_new_games_l1489_148998

theorem katie_new_games (K : ℕ) (h : K + 8 = 92) : K = 84 :=
by
  sorry

end katie_new_games_l1489_148998


namespace adam_tickets_left_l1489_148926

-- Define the initial number of tickets, cost per ticket, and total spending on the ferris wheel
def initial_tickets : ℕ := 13
def cost_per_ticket : ℕ := 9
def total_spent : ℕ := 81

-- Define the number of tickets Adam has after riding the ferris wheel
def tickets_left (initial_tickets cost_per_ticket total_spent : ℕ) : ℕ :=
  initial_tickets - (total_spent / cost_per_ticket)

-- Proposition to prove that Adam has 4 tickets left
theorem adam_tickets_left : tickets_left initial_tickets cost_per_ticket total_spent = 4 :=
by
  sorry

end adam_tickets_left_l1489_148926


namespace Fran_speed_l1489_148924

def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 3.5

theorem Fran_speed :
  ∀ (s : ℝ), (s * Fran_time = Joann_speed * Joann_time) → (s = 120 / 7) :=
by
  intro s h
  sorry

end Fran_speed_l1489_148924


namespace number_of_trees_in_garden_l1489_148944

def total_yard_length : ℕ := 600
def distance_between_trees : ℕ := 24
def tree_at_each_end : ℕ := 1

theorem number_of_trees_in_garden : (total_yard_length / distance_between_trees) + tree_at_each_end = 26 := by
  sorry

end number_of_trees_in_garden_l1489_148944


namespace part_one_part_two_l1489_148980

noncomputable def M := Set.Ioo (-(1 : ℝ)/2) (1/2)

namespace Problem

variables {a b : ℝ}
def in_M (x : ℝ) := x ∈ M

theorem part_one (ha : in_M a) (hb : in_M b) :
  |(1/3 : ℝ) * a + (1/6) * b| < 1/4 :=
sorry

theorem part_two (ha : in_M a) (hb : in_M b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end Problem

end part_one_part_two_l1489_148980


namespace value_of_x_l1489_148982

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := 
by
  sorry

end value_of_x_l1489_148982


namespace max_value_of_x0_l1489_148964

noncomputable def sequence_max_value (seq : Fin 1996 → ℝ) (pos_seq : ∀ i, seq i > 0) : Prop :=
  seq 0 = seq 1995 ∧
  (∀ i : Fin 1995, seq i + 2 / seq i = 2 * seq (i + 1) + 1 / seq (i + 1)) ∧
  (seq 0 ≤ 2^997)

theorem max_value_of_x0 :
  ∃ seq : Fin 1996 → ℝ, ∀ pos_seq : ∀ i, seq i > 0, sequence_max_value seq pos_seq :=
sorry

end max_value_of_x0_l1489_148964


namespace cupcakes_left_at_home_correct_l1489_148920

-- Definitions of the conditions
def total_cupcakes_baked : ℕ := 53
def boxes_given_away : ℕ := 17
def cupcakes_per_box : ℕ := 3

-- Calculate the total number of cupcakes given away
def total_cupcakes_given_away := boxes_given_away * cupcakes_per_box

-- Calculate the number of cupcakes left at home
def cupcakes_left_at_home := total_cupcakes_baked - total_cupcakes_given_away

-- Prove that the number of cupcakes left at home is 2
theorem cupcakes_left_at_home_correct : cupcakes_left_at_home = 2 := by
  sorry

end cupcakes_left_at_home_correct_l1489_148920


namespace solution_value_a_l1489_148939

theorem solution_value_a (x a : ℝ) (h₁ : x = 2) (h₂ : 2 * x + a = 3) : a = -1 :=
by
  -- Proof goes here
  sorry

end solution_value_a_l1489_148939


namespace games_played_in_tournament_l1489_148904

def number_of_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem games_played_in_tournament : number_of_games 18 = 153 :=
  by
    sorry

end games_played_in_tournament_l1489_148904


namespace years_since_marriage_l1489_148921

theorem years_since_marriage (x : ℕ) (ave_age_husband_wife_at_marriage : ℕ)
  (total_family_age_now : ℕ) (child_age : ℕ) (family_members : ℕ) :
  ave_age_husband_wife_at_marriage = 23 →
  total_family_age_now = 19 →
  child_age = 1 →
  family_members = 3 →
  (46 + 2 * x) + child_age = 57 →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end years_since_marriage_l1489_148921
