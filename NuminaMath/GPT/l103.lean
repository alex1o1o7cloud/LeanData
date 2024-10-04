import Mathlib

namespace division_result_l103_103236

theorem division_result (a b : ℕ) (ha : a = 7) (hb : b = 3) :
    ((a^3 + b^3) / (a^2 - a * b + b^2) = 10) := 
by
  sorry

end division_result_l103_103236


namespace intersection_points_count_l103_103039

def parabola := {p : ℝ × ℝ // p.2 ^ 2 = 8 * p.1}
def directrix := {p : ℝ × ℝ // p.1 = -2}
def circle := {p : ℝ × ℝ // (p.1 + 3) ^ 2 + p.2 ^ 2 = 16}
def curve := parabola ∪ directrix

theorem intersection_points_count : 
  ∃ n : ℕ, n = 4 ∧ 
  (∀ p : ℝ × ℝ, p ∈ curve → p ∈ circle) → n = 4 :=
by sorry

end intersection_points_count_l103_103039


namespace initial_birds_l103_103880

theorem initial_birds (B : ℕ) (h1 : B + 21 = 35) : B = 14 :=
by
  sorry

end initial_birds_l103_103880


namespace toy_poodle_height_l103_103915

theorem toy_poodle_height 
  (SP MP TP : ℕ)
  (h1 : SP = MP + 8)
  (h2 : MP = TP + 6)
  (h3 : SP = 28) 
  : TP = 14 := 
    by sorry

end toy_poodle_height_l103_103915


namespace sin_sum_cos_sum_l103_103816

-- Define the triangle and acuity condition
structure Triangle (A B C : ℝ) :=
(acute : A < π/2 ∧ B < π/2 ∧ C < π/2)
(sum_gt_pi_half : A + B > π/2)

-- Define the proposition to be proved
theorem sin_sum_cos_sum {A B C : ℝ} (h : Triangle A B C) :
  sin A + sin B + sin C > 1 + cos A + cos B + cos C :=
by 
  sorry

end sin_sum_cos_sum_l103_103816


namespace larry_minimum_next_test_score_l103_103836

-- Define Larry's test scores
def larry_scores : List ℕ := [75, 65, 85, 95, 60]

-- Define Larry's goal to raise the average by 5 points
def larry_goal_increment := 5

-- Calculate current sum of scores
def current_sum (scores : List ℕ) := scores.sum

-- Calculate the current average given Larry's scores
def current_average (scores : List ℕ) := current_sum scores / scores.length

-- Calculate the required average after the next test
def required_average (current_avg : ℕ) (increment : ℕ) := current_avg + increment

-- Calculate the total required score to achieve the new average
def required_total_score (desired_avg : ℕ) (num_tests : ℕ) := desired_avg * num_tests

-- Prove that the minimum score needed on the next test
theorem larry_minimum_next_test_score : 
  let num_scores := larry_scores.length + 1 in
  let current_sum := current_sum larry_scores in
  let current_avg := current_average larry_scores in
  let desired_avg := required_average current_avg larry_goal_increment in
  let required_total := required_total_score desired_avg num_scores in
  (required_total - current_sum) = 106 := 
by 
  -- Definitions for proof clarity (can be inlined in the future if necessary)
  let current_sum := 380
  let current_avg := 76
  let desired_avg := 81
  let required_total := 486
  -- Calculate required score on next test
  calc
  required_total - current_sum
    = 486 - 380 : by rfl
    ... = 106 : by rfl

end larry_minimum_next_test_score_l103_103836


namespace lowest_possible_students_l103_103289

theorem lowest_possible_students :
  ∃ n : ℕ, (n % 10 = 0 ∧ n % 24 = 0) ∧ n = 120 :=
by
  sorry

end lowest_possible_students_l103_103289


namespace rotate_segment_divisibility_l103_103871

theorem rotate_segment_divisibility (a b c d e A : ℕ) (N : ℕ) 
    (hN : N = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e)
    (div271 : N = 271 * A) : 
  ∀ (k : ℕ), ∃ B : ℕ, rotateN N k = 271 * B :=
by
  sorry

end rotate_segment_divisibility_l103_103871


namespace plane_equation_l103_103660

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - t, 1 - 2 * s, 4 - s + 3 * t)

theorem plane_equation :
  ∃ A B C D : ℤ, A > 0 ∧ Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
  (∀ x y z, (x = 2 + 2 * s - t) ∧ (y = 1 - 2 * s) ∧ (z = 4 - s + 3 * t) →
    A * x + B * y + C * z + D = 0):=
begin
  use [6, 5, 2, -25],
  split,
  { dec_trivial, },
  split,
  { dec_trivial, },
  intros x y z h,
  rcases h with ⟨hx, hy, hz⟩,
  simp [hx, hy, hz],
  linarith,
end

end plane_equation_l103_103660


namespace marble_selection_l103_103129

/-- 
John has 15 marbles, and he is selecting 5 marbles where exactly 2 must be red and yellow. 
The problem is to prove that there are exactly 286 ways for this selection. 
-/
theorem marble_selection (total_marbles : ℕ) (selection_count : ℕ) (remaining_count : ℕ) (fixed_count : ℕ)
  (total_marbles = 15) (selection_count = 5) (remaining_count = 13) (fixed_count = 2) :
  (∃ (remaining_selection : ℕ), remaining_selection = selection_count - fixed_count) ∧ 
  nat.choose remaining_count remaining_selection = 286 :=
by
  sorry

end marble_selection_l103_103129


namespace unique_rectangle_Q_l103_103391

noncomputable def rectangle_Q_count (a : ℝ) :=
  let x := (3 * a) / 2
  let y := a / 2
  if x < 2 * a then 1 else 0

-- The main theorem
theorem unique_rectangle_Q (a : ℝ) (h : a > 0) :
  rectangle_Q_count a = 1 :=
sorry

end unique_rectangle_Q_l103_103391


namespace exterior_angle_of_regular_pentagon_l103_103112

theorem exterior_angle_of_regular_pentagon : 
  ∀ (n : ℕ), (nat.succ (nat.succ (nat.succ (nat.succ n))) = 5) → (180 / 5 = 36) → (180 - 36 = 72):=
  λ n h₁ h₂, sorry

end exterior_angle_of_regular_pentagon_l103_103112


namespace inequality_proof_l103_103546

variable (x y : ℝ)
variable (h1 : x ≥ 0)
variable (h2 : y ≥ 0)
variable (h3 : x + y ≤ 1)

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) : 
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := 
by 
  sorry

end inequality_proof_l103_103546


namespace max_profit_l103_103561

open Real Nat

def trucks := 20
def total_tons := 100
def tons_per_truck_a := 6
def tons_per_truck_b := 5
def tons_per_truck_c := 4
def profit_per_ton_a := 500
def profit_per_ton_b := 600
def profit_per_ton_c := 400

theorem max_profit (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 9) (hy : 2 ≤ y) :
  6 * x + 5 * y + 4 * (trucks - x - y) = total_tons →
  y = -2 * x + 20 →
  (∀ z, 2 ≤ z ∧ z ≤ 9 → let mx := -(profit_per_ton_a - profit_per_ton_b) * z + profit_per_ton_c * total_tons in
                        let my := -(profit_per_ton_b - profit_per_ton_c) * z + profit_per_ton_c * total_tons in
                        -(profit_per_ton_a - profit_per_ton_b) * 2 + profit_per_ton_c * total_tons ≥ mx ∧
                        -(profit_per_ton_b - profit_per_ton_c) * 2 + profit_per_ton_c * total_tons ≥ my) →
  (trag : y = 16)
  truck arrangement : (trucks - x - y = 2) →
  let max_profit := -1400 * 2 + 60000 in
  max_profit = 57200 := sorry

end max_profit_l103_103561


namespace sequence_infinite_negatives_l103_103394

variable (a : ℕ → ℝ) (a1 : ℝ)

-- Sequence definition
def sequence_condition (a : ℕ → ℝ) [Noncomputable]: Prop :=
∀ n : ℕ, 
  a n ≠ 0 → a (n + 1) = (a n ^ 2 - 1) / (2 * a n) ∧ a n = 0 → a (n + 1) = 0

-- The statement to be proved
theorem sequence_infinite_negatives (a : ℕ → ℝ) (a1 : ℝ) (h : sequence_condition a) :
  ∃ infinitely_many (n : ℕ), n ≥ 1 ∧ a n ≤ 0 :=
sorry

end sequence_infinite_negatives_l103_103394


namespace minimal_distance_l103_103203

noncomputable def minimum_distance_travel (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) : ℝ :=
  2 * Real.sqrt 19

theorem minimal_distance (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) :
  minimum_distance_travel a b c ha hb hc = 2 * Real.sqrt 19 :=
by
  -- Proof is omitted
  sorry

end minimal_distance_l103_103203


namespace avg_korean_language_score_l103_103199

theorem avg_korean_language_score (male_avg : ℝ) (female_avg : ℝ) (male_students : ℕ) (female_students : ℕ) 
    (male_avg_given : male_avg = 83.1) (female_avg_given : female_avg = 84) (male_students_given : male_students = 10) (female_students_given : female_students = 8) :
    (male_avg * male_students + female_avg * female_students) / (male_students + female_students) = 83.5 :=
by sorry

end avg_korean_language_score_l103_103199


namespace find_a_2b_3c_l103_103846

noncomputable def a : ℝ := 28
noncomputable def b : ℝ := 32
noncomputable def c : ℝ := -3

def ineq_condition (x : ℝ) : Prop := (x < -3) ∨ (abs (x - 30) ≤ 2)

theorem find_a_2b_3c (a b c : ℝ) (h₁ : a < b)
  (h₂ : ∀ x : ℝ, (x < -3 ∨ abs (x - 30) ≤ 2) ↔ ((x - a)*(x - b)/(x - c) ≤ 0)) :
  a + 2 * b + 3 * c = 83 :=
by
  sorry

end find_a_2b_3c_l103_103846


namespace trees_to_plant_total_l103_103658

def trees_chopped_first_half := 200
def trees_chopped_second_half := 300
def trees_to_plant_per_tree_chopped := 3

theorem trees_to_plant_total : 
  (trees_chopped_first_half + trees_chopped_second_half) * trees_to_plant_per_tree_chopped = 1500 :=
by
  sorry

end trees_to_plant_total_l103_103658


namespace petri_dishes_count_l103_103116

theorem petri_dishes_count :
  let total_germs := 0.036 * 10^5
  let germs_per_dish := 48
  let dishes := total_germs / germs_per_dish
  dishes = 75 :=
by 
  let total_germs := 0.036 * 10^5
  let germs_per_dish := 48
  let dishes := total_germs / germs_per_dish
  have : dishes = 75 := sorry
  exact this

end petri_dishes_count_l103_103116


namespace largest_integer_mod_l103_103716

theorem largest_integer_mod (a : ℕ) (h₁ : a < 100) (h₂ : a % 5 = 2) : a = 97 :=
by sorry

end largest_integer_mod_l103_103716


namespace iggy_pace_l103_103101

theorem iggy_pace 
  (monday_miles : ℕ) (tuesday_miles : ℕ) (wednesday_miles : ℕ)
  (thursday_miles : ℕ) (friday_miles : ℕ) (total_hours : ℕ) 
  (h1 : monday_miles = 3) (h2 : tuesday_miles = 4) 
  (h3 : wednesday_miles = 6) (h4 : thursday_miles = 8) 
  (h5 : friday_miles = 3) (h6 : total_hours = 4) :
  (total_hours * 60) / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles) = 10 :=
sorry

end iggy_pace_l103_103101


namespace ratio_of_capitals_l103_103325

-- Variables for the capitals of Ashok and Pyarelal
variables (A P : ℕ)

-- Given conditions
def total_loss := 670
def pyarelal_loss := 603
def ashok_loss := total_loss - pyarelal_loss

-- Proof statement: the ratio of Ashok's capital to Pyarelal's capital
theorem ratio_of_capitals : ashok_loss * P = total_loss * pyarelal_loss - pyarelal_loss * P → A * pyarelal_loss = P * ashok_loss :=
by
  sorry

end ratio_of_capitals_l103_103325


namespace sum_of_coordinates_of_S_l103_103568

-- Given points
def P : ℝ × ℝ := (5, 5)
def Q : ℝ × ℝ := (1, 1)
def R : ℝ × ℝ := (6, 2)

-- Point S lies in the first quadrant with coordinates (x, y)
variables (S : ℝ × ℝ)

-- S = (x, y)
def x := S.1
def y := S.2

-- Definition of midpoints
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Midpoints of segments
def M1 := midpoint P Q
def M2 := midpoint Q R
def M3 := midpoint R S
def M4 := midpoint S P

-- Square condition
def isSquare (a b c d : ℝ × ℝ) : Prop :=
  let V1 := (b.1 - a.1, b.2 - a.2)
  let V2 := (c.1 - b.1, c.2 - b.2)
  let V3 := (d.1 - c.1, d.2 - c.2)
  let V4 := (a.1 - d.1, a.2 - d.2)
  (V1.1^2 + V1.2^2 = V2.1^2 + V2.2^2) ∧
  (V2.1^2 + V2.2^2 = V3.1^2 + V3.2^2) ∧
  (V3.1^2 + V3.2^2 = V4.1^2 + V4.2^2) ∧
  (V1.1 * V2.1 + V1.2 * V2.2 = 0)

-- The theorem to prove
theorem sum_of_coordinates_of_S (h : isSquare M1 M2 M3 M4) : x + y = 6 :=
sorry

end sum_of_coordinates_of_S_l103_103568


namespace original_cost_eq_l103_103777

-- Definitions
def oz_per_serving : ℕ := 1
def total_oz : ℕ := 40
def coupon : ℝ := 5.00
def cost_per_serving_after_coupon : ℝ := 0.50

-- Hypothesis: After applying the $5.00 coupon, each serving costs $0.50
def cost_after_coupon (total_oz : ℕ) (oz_per_serving : ℕ) (cost_per_serving_after_coupon : ℝ) : ℝ :=
  (total_oz / oz_per_serving) * cost_per_serving_after_coupon

-- Theorem: The original cost of the bag of mixed nuts before applying the coupon
theorem original_cost_eq :
  (cost_after_coupon total_oz oz_per_serving cost_per_serving_after_coupon) + coupon = 25.00 := by
  sorry

end original_cost_eq_l103_103777


namespace william_washed_2_normal_cars_l103_103268

def time_spent_on_one_normal_car : Nat := 4 + 7 + 4 + 9

def time_spent_on_suv : Nat := 2 * time_spent_on_one_normal_car

def total_time_spent : Nat := 96

def time_spent_on_normal_cars : Nat := total_time_spent - time_spent_on_suv

def number_of_normal_cars : Nat := time_spent_on_normal_cars / time_spent_on_one_normal_car

theorem william_washed_2_normal_cars : number_of_normal_cars = 2 := by
  sorry

end william_washed_2_normal_cars_l103_103268


namespace not_possible_one_not_divisible_by_3_l103_103117

theorem not_possible_one_not_divisible_by_3 :
  ∀ (grid : Fin 5 → Fin 5 → ℕ), 
    (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 9) → 
    ¬ (∃ (rows : Fin 5 → ℕ) (cols : Fin 5 → ℕ),
        (∀ i, rows i = (Finset.univ.sum (λ j, grid i j))) ∧
        (∀ j, cols j = (Finset.univ.sum (λ i, grid i j))) ∧
        (∃ k, (rows k % 3 ≠ 0) ∧ 
              (∀ i ≠ k, rows i % 3 = 0) ∧ 
              (∀ j, cols j % 3 = 0))) :=
by
  intro grid cond
  sorry

end not_possible_one_not_divisible_by_3_l103_103117


namespace fraction_is_integer_l103_103526

theorem fraction_is_integer (a b : ℤ) (n : ℕ) (hn : n > 0) :
  ∃ k : ℤ, k = (b^(n-1) * a * ∏ i in Finset.range (n - 1), (a + (i + 1)*b)) / n! :=
by sorry

end fraction_is_integer_l103_103526


namespace g_range_l103_103443

noncomputable section

-- Define the function f(x) = sin(ωx + φ)
def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

-- Conditions
axiom omega_positive : ∀ (ω : ℝ), ω > 0
axiom phi_range : ∀ (φ : ℝ), 0 < φ ∧ φ < Real.pi / 2
axiom passes_through_point : ∀ (φ : ℝ), f 2 φ 0 = 1 / 2
axiom y_difference : ∀ (x1 y1 x2 y2 : ℝ), (f 2 (Real.pi / 6) x1 - f 2 (Real.pi / 6) x2).abs = 2
axiom min_x1_x2_distance : ∀ (x1 x2 : ℝ), (x1 - x2).abs = Real.pi / 2

-- In triangle ABC
axiom triangle_condition : ∀ (A B C a b c : ℝ), 
  2 * Real.sin A * Real.sin C + Real.cos (2 * B) = 1

-- Define the function g(B)
def g (B : ℝ) := sqrt 3 * (f 2 (Real.pi / 6) B) + f 2 (Real.pi / 6) (B + Real.pi / 4)

-- And its range
theorem g_range : ∀ B : ℝ, 0 < B ∧ B ≤ Real.pi / 3 → 0 ≤ g B ∧ g B ≤ 2 := sorry

end g_range_l103_103443


namespace part1_tangent_line_part2_extremum_part3_inequality_l103_103381

variable {a : ℝ} (f g : ℝ → ℝ)
variable {x_0 : ℝ} (x a : ℝ)

def f_def (a : ℝ) (x : ℝ) := a^2 * x^3 - 3 * a * x^2 + 2
def g_def (a : ℝ) (x : ℝ) := -3 * a * x + 3

theorem part1_tangent_line (ha : a = 1) :
  ∃ k b, f = λ x, k * x + b ∧
  (k = -3) ∧ (b = 0) :=
sorry

theorem part2_extremum (ha_pos : a > 0) :
  (a > 2 → f 0 = 2 ∧ f (2 / a) = (2 * a - 4) / a) ∧
  (a = 2 → f 0 = 2 ∧ f 1 = 1) ∧
  (0 < a ∧ a < 2 → f 0 = 2) :=
sorry

theorem part3_inequality (ha_pos : a > 0) (hx0 : x_0 ∈ (0, 1/2])
  (hg : f x_0 > g x_0) :
  a > -3 + sqrt 17 :=
sorry

#eval (part1_tangent_line, part2_extremum, part3_inequality)

end part1_tangent_line_part2_extremum_part3_inequality_l103_103381


namespace old_fridge_cost_l103_103130

-- Define the daily cost of Kurt's old refrigerator
variable (x : ℝ)

-- Define the conditions given in the problem
def new_fridge_cost_per_day : ℝ := 0.45
def savings_per_month : ℝ := 12
def days_in_month : ℝ := 30

-- State the theorem to prove
theorem old_fridge_cost :
  30 * x - 30 * new_fridge_cost_per_day = savings_per_month → x = 0.85 := 
by
  intro h
  sorry

end old_fridge_cost_l103_103130


namespace problemStatement_l103_103500

-- Define the set of values as a type
structure SetOfValues where
  k : ℤ
  b : ℤ

-- The given sets of values
def A : SetOfValues := ⟨2, 2⟩
def B : SetOfValues := ⟨2, -2⟩
def C : SetOfValues := ⟨-2, -2⟩
def D : SetOfValues := ⟨-2, 2⟩

-- Define the conditions for the function
def isValidSet (s : SetOfValues) : Prop :=
  s.k < 0 ∧ s.b > 0

-- The problem statement: Prove that D is a valid set
theorem problemStatement : isValidSet D := by
  sorry

end problemStatement_l103_103500


namespace payment_difference_is_6967_l103_103333

noncomputable def plan1_payment (P : ℝ) (r : ℝ) (n : ℕ) (years : ℕ) : ℝ :=
let A := P * (1 + r/n)^(n*years),
    payment_at_4_years := A / 3,
    new_principal := A - payment_at_4_years,
    final_balance := new_principal * (1 + r/n)^(n*years) in
  payment_at_4_years + final_balance

noncomputable def plan2_payment (P : ℝ) (r : ℝ) (years : ℕ) : ℝ :=
let initial_payment := 5000,
    remaining_balance := P - initial_payment,
    final_balance := remaining_balance * (1 + r)^years in
  initial_payment + final_balance

theorem payment_difference_is_6967 :
  let P := 15000 in
  let r := 0.08 in
  let years := 8 in
  abs (plan1_payment P r 4 years - plan2_payment P r (years/2)) = 6967 :=
by sorry

end payment_difference_is_6967_l103_103333


namespace triangle_area_l103_103674

-- Definitions for the given conditions
def AC : ℝ := 12
def angle_BAC : ℝ := 60
def BK : ℝ := 3
def BC : ℝ := 15
def sqrt3 : ℝ := real.sqrt 3

-- Mathematical theorem based on the given problem
theorem triangle_area (AC_eq : AC = 12) (angle_BAC_eq : angle_BAC = 60)
  (BK_eq : BK = 3) (BC_eq : BC = 15) : 
  let AK := AC * real.sin (angle_BAC * real.pi / 180)
  in 0.5 * BC * AK = 45 * sqrt3 :=
by
  sorry

end triangle_area_l103_103674


namespace twelve_xy_leq_fourx_1_y_9y_1_x_l103_103549

theorem twelve_xy_leq_fourx_1_y_9y_1_x
  (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) :=
  sorry

end twelve_xy_leq_fourx_1_y_9y_1_x_l103_103549


namespace proof_parabola_statements_l103_103068

theorem proof_parabola_statements (b c : ℝ)
  (h1 : 1/2 - b + c < 0)
  (h2 : 2 - 2 * b + c < 0) :
  (b^2 > 2 * c) ∧
  (c > 1 → b > 3/2) ∧
  (∀ (m1 m2 : ℝ), m1 < m2 ∧ m2 < b → ∀ (y : ℝ), y = (1/2)*m1^2 - b*m1 + c → ∀ (y2 : ℝ), y2 = (1/2)*m2^2 - b*m2 + c → y > y2) ∧
  (¬(∃ x1 x2 : ℝ, (1/2) * x1^2 - b * x1 + c = 0 ∧ (1/2) * x2^2 - b * x2 + c = 0 ∧ x1 + x2 > 3)) :=
by sorry

end proof_parabola_statements_l103_103068


namespace find_brick_length_l103_103290

def volume_of_wall (length width height : ℝ) : ℝ := 
  length * width * height

def volume_occupied_by_bricks (total_volume mortar_percentage : ℝ) : ℝ :=
  total_volume * (1 - mortar_percentage)

def volume_of_brick (total_volume number_of_bricks : ℝ) : ℝ :=
  total_volume / number_of_bricks

def length_of_brick (brick_volume width height : ℝ) : ℝ :=
  brick_volume / (width * height)

theorem find_brick_length
  (wall_length wall_width wall_height : ℝ)
  (mortar_percentage : ℝ)
  (brick_length : ℝ) (brick_width brick_height : ℝ)
  (number_of_bricks : ℕ)
  (h1: wall_length = 10)
  (h2: wall_width = 4)
  (h3: wall_height = 5)
  (h4: mortar_percentage = 0.1)
  (h5: brick_width = 15)
  (h6: brick_height = 8)
  (h7: number_of_bricks = 6000)
  : brick_length = 250 :=
by
  let wall_volume := volume_of_wall wall_length wall_width wall_height
  let brick_volume := volume_occupied_by_bricks wall_volume mortar_percentage
  let brick_single_volume := volume_of_brick brick_volume number_of_bricks
  have h8: brick_single_volume = 30000 :=
    calc
      volume_of_wall wall_length wall_width wall_height = 200 : by rw [h1, h2, h3] ; exact rfl
      volume_occupied_by_bricks 200 0.1 = 180 : by rw [h4] ; exact rfl
      180 * 1000000 = 180000000 : rfl
      (volume_of_brick 180000000 6000) = 30000 : by rw [h7] ; exact rfl
  have h9: brick_length = 250 := calc
    length_of_brick 30000 brick_width brick_height = 250 : by rw [h5, h6, h8] ; exact rfl
  exact h9

end find_brick_length_l103_103290


namespace arctan_sum_eq_pi_over_4_l103_103503

theorem arctan_sum_eq_pi_over_4 {A B C D : Type} {a b c m n : ℝ}
  (h1 : ∠A = π / 2)
  (h2 : D ∈ line[BC])
  (h3 : altitude[AD](A, line[BC]))
  (h4 : BD = m)
  (h5 : DC = n)
  (h6 : a = sqrt (b^2 + c^2)) :
  arctan (b / (m + c)) + arctan (c / (n + b)) = π / 4 :=
by
  sorry

end arctan_sum_eq_pi_over_4_l103_103503


namespace volunteers_selection_l103_103584

theorem volunteers_selection : 
  let volunteers := Set.range (1:ℕ) 30, 
      selected_volunteers := {6, 15, 24} in
  ∃ g1 g2 : Finset ℕ,       -- Six volunteers selected and divided into two groups
  g1.card = 3 ∧             -- Group 1 has 3 volunteers
  g2.card = 3 ∧             -- Group 2 has 3 volunteers
  selected_volunteers ⊆ g1 ∨ selected_volunteers ⊆ g2 ∧   -- Ensure 6, 15, 24 are in the same group
  ∑ v in g2, v > 24 ∨       -- Remaining numbers in g1 must be either all > 25
  ∑ v in g1, v < 6 ∧        -- or all < 6
  (Finset.binom 6 3 = 20) ∨ -- Calculate ways to select from 6 volunteers
  (Finset.binom 5 3 = 10) ∧ -- Calculate ways to select from 5 volunteers
  (Finset.card (Finset.range 2)) = 2 → -- Allocate groups to locations
  (20 + 10) * 2 = 60 :=     -- Ensure the solution matches 60 ways
by sorry

end volunteers_selection_l103_103584


namespace max_arithmetic_prog_length_l103_103624

-- Predicate stating that a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Definition of the arithmetic progression sequence
def arithmetic_prog (a d n : ℕ) : ℕ → ℕ 
| 0     := a
| (k+1) := (arithmetic_prog k) + d

-- Main theorem statement
theorem max_arithmetic_prog_length (a d n : ℕ) (h_d2 : d = 2):
  (∀ k < n, is_prime ((arithmetic_prog a d n k)^2 + 1)) → n ≤ 3 := 
sorry

end max_arithmetic_prog_length_l103_103624


namespace f_2016_eq_one_third_l103_103430

noncomputable def f (x : ℕ) : ℝ := sorry

axiom f_one : f 1 = 2
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = (1 + f x) / (1 - f x)

theorem f_2016_eq_one_third : f 2016 = 1 / 3 := sorry

end f_2016_eq_one_third_l103_103430


namespace min_value_f_l103_103533

def f (x : ℝ) (m : ℝ) : ℝ := (1/3) * x^3 - x + m

theorem min_value_f (m : ℝ) (h1 : ∀ x : ℝ, f x m ≤ 1) : ∃ m, f 1 m = -1/3 :=
by
  sorry

end min_value_f_l103_103533


namespace functional_equation_solution_l103_103696

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x) * f(y * f(x) - 1) = x^2 * f(y) - f(x)) →
  (∀ x, f(x) = 0) ∨ (∀ x, f(x) = x) :=
by
  sorry

end functional_equation_solution_l103_103696


namespace cartesian_parametric_eqs_of_curve_C_range_of_AB_l103_103501

-- Configuration of the parametric equations of the line l and curve C
def parametric_line_eq (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, 1 + t * Real.sin α)

def polar_eq_curve (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos θ - 6 * ρ * Real.sin θ + 4 = 0

-- Proving part (I) - Cartesian and parametric equations of curve C
theorem cartesian_parametric_eqs_of_curve_C :
  ∀ (ρ θ: ℝ), polar_eq_curve ρ θ -> 
    let x := ρ * Real.cos θ in
    let y := ρ * Real.sin θ in
    (x - 2)^2 + (y - 3)^2 = 9 ∧
    ∃ φ, x = 2 + 3 * Real.cos φ ∧ y = 3 + 3 * Real.sin φ :=
sorry

-- Proving part (II) - The range of |AB| 
theorem range_of_AB (α : ℝ) :
  let equation_l := λ t, parametric_line_eq t α in
  let curve_eq := λ x y, (x - 2)^2 + (y - 3)^2 = 9 in
  ∀ (t1 t2 : ℝ), curve_eq (fst (equation_l t1)) (snd (equation_l t1)) ∧ curve_eq (fst (equation_l t2)) (snd (equation_l t2)) -> 
  let AB := Real.sqrt (10 * Real.sin (2 * α - Real.acos (4/5)) + 26) in
  4 ≤ AB ∧ AB ≤ 6 :=
sorry

end cartesian_parametric_eqs_of_curve_C_range_of_AB_l103_103501


namespace minimum_total_distance_l103_103205

-- Conditions:
def point (α : Type) := (α × α)
def distance (p1 p2 : point ℝ) : ℝ := 
  float.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def vertex := point ℝ

variables (A B C : vertex)
axiom AB_eq_3 : distance A B = 3
axiom AC_eq_2 : distance A C = 2
axiom BC_eq_sqrt_7 : distance B C = float.sqrt 7
axiom warehouse_pos : vertex -- Assuming existence but not fixing the actual position

-- Question == Answer
theorem minimum_total_distance (warehouse : vertex) 
  (dA := distance warehouse A)
  (dB := distance warehouse B)
  (dC := distance warehouse C) :
  let total_distance := dA + dB + dC 
  in total_distance * 2 = 2 * float.sqrt 19 :=
by sorry

end minimum_total_distance_l103_103205


namespace largest_sum_of_products_l103_103220

variables (x y z w : ℕ) (h : {x, y, z, w} = {7, 8, 9, 10}) (h_order : x < y ∧ y < z ∧ z < w)

theorem largest_sum_of_products : (x * y + y * z + z * w + x * w) = 288 :=
sorry

end largest_sum_of_products_l103_103220


namespace part1_part2_l103_103418

section
variable (k : ℝ)

/-- Part 1: Range of k -/
def discriminant_eqn (k : ℝ) := (2 * k - 1) ^ 2 - 4 * (k ^ 2 - 1)

theorem part1 (h : discriminant_eqn k ≥ 0) : k ≤ 5 / 4 :=
by sorry

/-- Part 2: Value of k when x₁ and x₂ satisfy the given condition -/
def x1_x2_eqn (k x1 x2 : ℝ) := x1 ^ 2 + x2 ^ 2 = 16 + x1 * x2

def vieta (k : ℝ) (x1 x2 : ℝ) :=
  x1 + x2 = 1 - 2 * k ∧ x1 * x2 = k ^ 2 - 1

theorem part2 (x1 x2 : ℝ) (h1 : vieta k x1 x2) (h2 : x1_x2_eqn k x1 x2) : k = -2 :=
by sorry

end

end part1_part2_l103_103418


namespace hyperbola_eccentricity_l103_103888

noncomputable def hyperbola_eq : Prop :=
∃ (a b : ℝ), (a = 1) ∧ (b = 1) ∧ (c = real.sqrt (a^2 + b^2)) ∧ (e = c / a) ∧ (e = real.sqrt 2)

theorem hyperbola_eccentricity : hyperbola_eq :=
by
  -- Proof goes here
  sorry

end hyperbola_eccentricity_l103_103888


namespace valid_triangle_side_l103_103996

theorem valid_triangle_side (x : ℕ) (h_pos : 0 < x) (h1 : x + 6 > 15) (h2 : 21 > x) :
  10 ≤ x ∧ x ≤ 20 :=
by {
  sorry
}

end valid_triangle_side_l103_103996


namespace minimum_m_l103_103745

theorem minimum_m (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 24 * m = n ^ 4) : m ≥ 54 :=
sorry

end minimum_m_l103_103745


namespace find_f_expression_l103_103019

variable {α β : ℝ}

noncomputable def f (x : ℝ) : ℝ := x / (1 + 2 * x^2)

theorem find_f_expression (x y : ℝ) (h1 : sin (2 * α + β) = 3 * sin β) 
                         (h2 : tan α = x) (h3 : tan β = y) :
  y = f x :=
sorry

end find_f_expression_l103_103019


namespace D_subscription_proof_l103_103997

noncomputable def total_capital : ℝ := sorry -- Assume total capital

def A_subs (X : ℝ) : ℝ := (1/3) * X
def B_subs (X : ℝ) : ℝ := (1/4) * X
def C_subs (X : ℝ) : ℝ := (1/5) * X

def D_subs (X : ℝ) : ℝ := X - (A_subs X + B_subs X + C_subs X)

noncomputable def profit_A (total_profit : ℝ) : ℝ := sorry -- Assume A's profit share

theorem D_subscription_proof :
    ∀ (X : ℝ) (total_profit : ℝ) (A_profit : ℝ),
    A_subs X = (1/3) * X →
    B_subs X = (1/4) * X →
    C_subs X = (1/5) * X →
    total_profit = 2415 →
    A_profit = 805 →
    A_profit = (1 / 3) * total_profit →
    D_subs X = (13 / 60) * X := by
  intros X total_profit A_profit hA hB hC hTotal hProfit hShare
  sorry

end D_subscription_proof_l103_103997


namespace find_principal_amount_l103_103717

theorem find_principal_amount :
  let r : ℚ := 0.05 in
  let n : ℕ := 1 in
  let t : ℕ := 6 in
  let inflation_rate : ℚ := 0.02 in
  let final_amount : ℚ := 1120 in
  let real_value : ℚ := final_amount / (1 + inflation_rate)^t in
  let P : ℚ := real_value / (1 + r/n)^(n*t) in
  P ≈ 741.38 := sorry

end find_principal_amount_l103_103717


namespace range_of_a_l103_103731

theorem range_of_a (a x : ℝ) (h_p : a - 4 < x ∧ x < a + 4) (h_q : (x - 2) * (x - 3) > 0) :
  a ≤ -2 ∨ a ≥ 7 :=
sorry

end range_of_a_l103_103731


namespace XYDH_concyclic_l103_103826

noncomputable theory
open_locale classical

variables {A B C D H X Y : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited H] [Inhabited X] [Inhabited Y]

-- Given definitions
variables (triangle_ABC : Triangle A B C)
variables (AD : Line A D) (AH : Line A H)
variables (perp_bisect_AD : PerpendicularBisector AD)
variables (semicircle_AB : Semicircle A B)
variables (semicircle_AC : Semicircle A C)
variables (X_on_semicircle_AB : X ∈ semicircle_AB)
variables (Y_on_semicircle_AC : Y ∈ semicircle_AC)

-- Translating the given problem to a Lean 4 statement
theorem XYDH_concyclic 
  (angle_bisector_AD : AngleBisector AD)
  (altitude_AH : Altitude AH)
  (perp_bisec_intersects_X : perp_bisect_AD.intersects semicircle_AB X)
  (perp_bisec_intersects_Y : perp_bisect_AD.intersects semicircle_AC Y) :
  Concyclic XYDH :=
sorry

end XYDH_concyclic_l103_103826


namespace twelve_xy_leq_fourx_1_y_9y_1_x_l103_103550

theorem twelve_xy_leq_fourx_1_y_9y_1_x
  (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) :=
  sorry

end twelve_xy_leq_fourx_1_y_9y_1_x_l103_103550


namespace sum_of_coeffs_is_neg2_l103_103785

theorem sum_of_coeffs_is_neg2 (a : Fin 12 → ℤ) (x : ℤ) :
  (x^2 + 1) * (2*x + 1)^9 = (a 0 + a 1 * (x + 2) + a 2 * (x + 2)^2 +
                             a 3 * (x + 2)^3 + a 4 * (x + 2)^4 +
                             a 5 * (x + 2)^5 + a 6 * (x + 2)^6 +
                             a 7 * (x + 2)^7 + a 8 * (x + 2)^8 +
                             a 9 * (x + 2)^9 + a 10 * (x + 2)^10 +
                             a 11 * (x + 2)^11) →
  have h : ((1 + 1) * (2*(-1) + 1)^9 = (a 0 + a 1 + ... + a 11)) : sorry :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = -2 := sorry


end sum_of_coeffs_is_neg2_l103_103785


namespace point_on_midline_iff_angle_condition_l103_103838

variables (a b : ℝ) (P : ℝ × ℝ)

-- Define the coordinates of the rectangle vertices
def A := (0, 0) : ℝ × ℝ
def B := (a, 0) : ℝ × ℝ
def C := (a, b) : ℝ × ℝ
def D := (0, b) : ℝ × ℝ

-- Condition: P is between lines AB and CD and satisfies the angle condition
def is_between_lines (P : ℝ × ℝ) : Prop :=
  0 < P.snd ∧ P.snd < b

def angle_condition (P : ℝ × ℝ) : Prop :=
  ∠ A P B = ∠ C P D

theorem point_on_midline_iff_angle_condition :
  is_between_lines a b P → angle_condition a b P → P.snd = b / 2 :=
sorry

end point_on_midline_iff_angle_condition_l103_103838


namespace derivative_of_f_l103_103887

noncomputable def f : ℝ → ℝ := λ x, Real.sin x + 3^x

theorem derivative_of_f (x : ℝ) : deriv f x = Real.cos x + 3^x * Real.log 3 :=
by sorry

end derivative_of_f_l103_103887


namespace sum_function_values_l103_103755

noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

theorem sum_function_values :
  (f 1) + (∑ k in Finset.range 2016, f (k + 2) + f (1 / (k + 2))) = 4033 / 2 :=
by
  sorry

end sum_function_values_l103_103755


namespace solve_quadratic_1_solve_quadratic_2_l103_103583

theorem solve_quadratic_1 :
  ∀ x : ℝ, (3 * x^2 - 1 = 4 * x ↔ x = (2 + Real.sqrt 7) / 3 ∨ x = (2 - Real.sqrt 7) / 3) := 
by
  intros x
  have h := calc 
    3 * x^2 - 1 = 4 * x
  -- continue calculation and add lawful to the placeholder
  sorry

theorem solve_quadratic_2 :
  ∀ x : ℝ, ((x + 4)^2 = 5 * (x + 4) ↔ x = -4 ∨ x = 1) := 
by
  intros x
  have h := calc
    (x + 4)^2 = 5 * (x + 4)
  -- continue calculation and add lawful to the placeholder
  sorry

end solve_quadratic_1_solve_quadratic_2_l103_103583


namespace find_side_b_l103_103479

variable {a b c : ℝ} -- sides of the triangle
variable {A B C : ℝ} -- angles of the triangle
variable {area : ℝ}

axiom sides_form_arithmetic_sequence : 2 * b = a + c
axiom angle_B_is_60_degrees : B = Real.pi / 3
axiom area_is_3sqrt3 : area = 3 * Real.sqrt 3
axiom area_formula : area = 1 / 2 * a * c * Real.sin (B)

theorem find_side_b : b = 2 * Real.sqrt 3 := by
  sorry

end find_side_b_l103_103479


namespace f_range_l103_103465

-- Define the operation a ⊙ b
def odot (a b : ℝ) : ℝ :=
  if a ≥ b then b else a

-- Define the function f
def f (x : ℝ) : ℝ :=
  odot x (2 - x)

-- State the theorem about the range of f
theorem f_range : ∀ y, f y ≤ 1 :=
  sorry

end f_range_l103_103465


namespace board_numbers_become_power_of_2_l103_103863

theorem board_numbers_become_power_of_2 (n : ℕ) (h : n ≥ 3) :
  ∃ (s : ℕ), (2^s) ≥ n ∧ (∀ (p q : ℕ), p ∈ (range (n+1) \ {0}) ∧ q ∈ (range (n+1) \ {0}) → 
  (p + q ∈ pow 2 '' (set.range (λ x:ℕ, x)) ∧ (abs (p - q)) ∈ pow 2 '' (set.range (λ x:ℕ, x)))) :=
sorry

end board_numbers_become_power_of_2_l103_103863


namespace perpendicular_lines_slope_l103_103802

theorem perpendicular_lines_slope {a : ℝ} : 
  (∀ (m₁ m₂ : ℝ), 
    (∀ x y1 y2, 2 * y1 + x + 3 = 0 → y1 = -(1/2) * x - (3/2)) →
    (∀ x y1 y2, 3 * y2 + a * x + 2 = 0 → y2 = -(a/3) * x - (2/3)) →
    (m₁ = -(1/2)) →
    (m₂ = -(a/3)) →
    m₁ * m₂ = -1) →  a = -6 :=
begin
  sorry
end

end perpendicular_lines_slope_l103_103802


namespace part1_part2_l103_103827

open Real

namespace TriangleProblems

-- Define conditions for the problem
variables {A B C a b c : ℝ}
variable h1 : a * tan C = 2 * c * sin A
variable h2 : ∀ (A B C a b c : ℝ), opposite_side A a → opposite_side B b → opposite_side C c → true

-- First part: Determine the size of angle C
theorem part1 (h1 : a * tan C = 2 * c * sin A) : C = π / 3 := sorry

-- Second part: Determine the range of possible values for sin(A) + sin(B)
theorem part2 
  (h1 : a * tan C = 2 * c * sin A) 
  (h3 : C = π / 3)
  (h2 : ∀ {A B C : ℝ} (a b c : ℝ), opposite_side A a → opposite_side B b → opposite_side C c → true) :
  ∃ x, x = sin A + sin (3*π/3 - A) ∧ x ∈ Ioo (sqrt 3 / 2) (sqrt 3 ∪ (Icc 0 (π / 2))) := 
sorry

end TriangleProblems

end part1_part2_l103_103827


namespace time_to_reach_ship_l103_103989

/-- The scuba diver's descent problem -/

def rate_of_descent : ℕ := 35  -- in feet per minute
def depth_of_ship : ℕ := 3500  -- in feet

theorem time_to_reach_ship : depth_of_ship / rate_of_descent = 100 := by
  sorry

end time_to_reach_ship_l103_103989


namespace area_of_enclosed_region_l103_103121

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin(a * x) + Real.cos (a * x)
noncomputable def g (a : ℝ) : ℝ := Real.sqrt(a^2 + 1)
noncomputable def period (a : ℝ) : ℝ := 2 * Real.pi / a
noncomputable def enclosed_area (a : ℝ) : ℝ := (2 * Real.pi / a) * Real.sqrt(a^2 + 1)

theorem area_of_enclosed_region (a : ℝ) (h : a > 0) : 
  ∫ x in 0..period a, (g a - f a x) = enclosed_area a :=
sorry

end area_of_enclosed_region_l103_103121


namespace number_of_common_terms_within_set_l103_103729

theorem number_of_common_terms_within_set:
  let a_n := λ (n : ℕ), 3 * n + 2
  let b_n := λ (n : ℕ), 5 * n + 3
  let M := finset.range 2019
  ∃ k_max : ℕ, (a_n  (5*k_max + 2) ≤ 2018) ∧ ((a_n (5*k_max + 2)) ∈ M) ∧  k_max + 1 = 135 := 
sorry

end number_of_common_terms_within_set_l103_103729


namespace add_B48_57A_eq_5B6_l103_103676

def B48_12 : ℕ := 11 * 12^2 + 4 * 12^1 + 8 * 12^0
def 57A_12 : ℕ :=  5 * 12^2 + 7 * 12^1 + 10 * 12^0
def 5B6_12 : ℕ :=  5 * 12^2 + 11 * 12^1 + 6 * 12^0

theorem add_B48_57A_eq_5B6 : B48_12 + 57A_12 = 5B6_12 := by
  sorry

end add_B48_57A_eq_5B6_l103_103676


namespace running_current_each_unit_l103_103212

theorem running_current_each_unit (I : ℝ) (h1 : ∀i, i = 2 * I) (h2 : ∀i, i * 3 = 6 * I) (h3 : 6 * I = 240) : I = 40 :=
by
  sorry

end running_current_each_unit_l103_103212


namespace min_ratio_l103_103851

theorem min_ratio (x y : ℕ) 
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (mean : (x + y) = 110) :
  x / y = 1 / 9 :=
  sorry

end min_ratio_l103_103851


namespace algebra_expression_value_l103_103732

theorem algebra_expression_value (x y : ℝ)
  (h1 : x + y = 3)
  (h2 : x * y = 1) :
  (5 * x + 3) - (2 * x * y - 5 * y) = 16 :=
by
  sorry

end algebra_expression_value_l103_103732


namespace find_c_l103_103762

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^5 + b * Real.sin x + c

theorem find_c (h : f a b c (-1) + f a b c 1 = 2) : c = 1 :=
sorry

end find_c_l103_103762


namespace area_of_enclosed_shape_l103_103588

noncomputable def enclosed_area : ℝ := 
∫ x in (0 : ℝ)..(2/3 : ℝ), (2 * x - 3 * x^2)

theorem area_of_enclosed_shape : enclosed_area = 4 / 27 := by
  sorry

end area_of_enclosed_shape_l103_103588


namespace quadratic_other_x_intercept_l103_103727

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → (a * x^2 + b * x + c) = -3)
  (h_intercept : ∀ x, x = 1 → (a * x^2 + b * x + c) = 0) : 
  ∃ x : ℝ, x = 9 ∧ (a * x^2 + b * x + c) = 0 :=
sorry

end quadratic_other_x_intercept_l103_103727


namespace max_profit_l103_103562

open Real Nat

def trucks := 20
def total_tons := 100
def tons_per_truck_a := 6
def tons_per_truck_b := 5
def tons_per_truck_c := 4
def profit_per_ton_a := 500
def profit_per_ton_b := 600
def profit_per_ton_c := 400

theorem max_profit (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 9) (hy : 2 ≤ y) :
  6 * x + 5 * y + 4 * (trucks - x - y) = total_tons →
  y = -2 * x + 20 →
  (∀ z, 2 ≤ z ∧ z ≤ 9 → let mx := -(profit_per_ton_a - profit_per_ton_b) * z + profit_per_ton_c * total_tons in
                        let my := -(profit_per_ton_b - profit_per_ton_c) * z + profit_per_ton_c * total_tons in
                        -(profit_per_ton_a - profit_per_ton_b) * 2 + profit_per_ton_c * total_tons ≥ mx ∧
                        -(profit_per_ton_b - profit_per_ton_c) * 2 + profit_per_ton_c * total_tons ≥ my) →
  (trag : y = 16)
  truck arrangement : (trucks - x - y = 2) →
  let max_profit := -1400 * 2 + 60000 in
  max_profit = 57200 := sorry

end max_profit_l103_103562


namespace square_side_length_is_10_l103_103890

-- Define the side lengths of the original squares
def side_length1 : ℝ := 8
def side_length2 : ℝ := 6

-- Define the areas of the original squares
def area1 : ℝ := side_length1^2
def area2 : ℝ := side_length2^2

-- Define the total area of the combined squares
def total_area : ℝ := area1 + area2

-- Define the side length of the new square
def side_length_new_square : ℝ := 10

-- Theorem statement to prove that the side length of the new square is 10 cm
theorem square_side_length_is_10 : side_length_new_square^2 = total_area := by
  sorry

end square_side_length_is_10_l103_103890


namespace infinite_series_correct_l103_103699

noncomputable def infinite_series_sum : ℚ := 
  ∑' n : ℕ, (n+1)^2 * (1/999)^n

theorem infinite_series_correct : infinite_series_sum = 997005 / 996004 :=
  sorry

end infinite_series_correct_l103_103699


namespace yolkino_to_palkino_distance_l103_103556

theorem yolkino_to_palkino_distance 
  (n : ℕ) 
  (digit_sum : ℕ → ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → digit_sum k + digit_sum (n - k) = 13) : 
  n = 49 := 
by 
  sorry

end yolkino_to_palkino_distance_l103_103556


namespace triangle_angles_l103_103912

theorem triangle_angles (C A B : ℝ) (h1 : 3 = 3 ∧ 3 = 3 ∧ (sqrt 7 - sqrt 3) = (sqrt 7 - sqrt 3)) :
  C = real.arccos ((4 + sqrt 21) / 9) ∧ A = B ∧ A = (180 - real.arccos ((4 + sqrt 21) / 9)) / 2 :=
by
  sorry

end triangle_angles_l103_103912


namespace part1_z_axis_equidistant_part2_yOz_plane_equidistant_l103_103968

-- Part 1: Coordinates of a point on the z-axis equidistant from two points A and B
theorem part1_z_axis_equidistant :
  ∃ z : ℚ, (z = 14 / 9) :=
by
  let A := (-4, 1, 7)
  let B := (3, 5, -2)
  let C := (0, 0, z)
  -- define the distances and equate them
  -- solve the resulting equation
  sorry

-- Part 2: Coordinates of a point in the yOz plane equidistant from three points A, B, and C
theorem part2_yOz_plane_equidistant :
  ∃ (y z : ℚ), (y = 1 ∧ z = -2) :=
by
  let A := (3, 1, 2)
  let B := (4, -2, -2)
  let C := (0, 5, 1)
  let D := (0, y, z)
  -- define the distances and equate them
  let dist_eq_1 := 3 * y + 4 * z = -5 -- resulting from |AD| = |BD|
  let dist_eq_2 := 4 * y - z = 6 -- resulting from |AD| = |CD|
  -- solve the resulting system of equations
  sorry

end part1_z_axis_equidistant_part2_yOz_plane_equidistant_l103_103968


namespace sum_of_8_not_equal_fib_l103_103073

def fibonacci : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

theorem sum_of_8_not_equal_fib (k : ℕ) :
  ∀ n : ℕ, (finset.range 8).sum (λ i, fibonacci (k + i)) ≠ fibonacci n :=
sorry

end sum_of_8_not_equal_fib_l103_103073


namespace race_probability_l103_103111

theorem race_probability (Px : ℝ) (Py : ℝ) (Pz : ℝ) 
  (h1 : Px = 1 / 6) 
  (h2 : Pz = 1 / 8) 
  (h3 : Px + Py + Pz = 0.39166666666666666) : Py = 0.1 := 
sorry

end race_probability_l103_103111


namespace value_range_cos_tan_l103_103604

variable (x : ℝ)

def y_def := x ≠ (π / 2) + k * π → (∃ k : ℤ, True)
def y (x : ℝ) : ℝ := cos x * tan x

theorem value_range_cos_tan : ∀ x, y x ∈ Ioo (-1 : ℝ) 1 :=
by {
    sorry
}

end value_range_cos_tan_l103_103604


namespace broken_line_intersects_l103_103833

theorem broken_line_intersects {L : list (ℝ × ℝ)} (hL : ∑ (p : ℝ × ℝ) in L, dist p.1 p.2 > 1000) :
  ∃ l : ℝ, ∃ k < 2, ∀ p : ℝ × ℝ, k ∥ ∃ l > 0 // (intersections_perpendicular l p).length ≥ 501 :=
begin
  sorry
end

end broken_line_intersects_l103_103833


namespace fractional_part_of_sum_l103_103159

theorem fractional_part_of_sum :
  let x := ∑ k in Finset.range (2013), Real.sqrt (1 + 1 / (k + 1)^2 + 1 / (k + 2)^2) in
  x - Real.floor x = 2012 / 2013 := 
sorry

end fractional_part_of_sum_l103_103159


namespace tournament_round_robin_l103_103488

universe u

-- Define the players
inductive Player : Type
| Alesha | Borya | Vitya | Grisha | Dima | Kostya

open Player

-- Define the round-robin matches
structure Match (p1 p2 : Player) : Prop where
  ne : p1 ≠ p2

-- Define match for each day
structure DayMatches :=
  (m1 m2 m3 : Player × Player)
  (distinct : ∀ (p1 p2 : Player) (m : Player × Player), m ∈ [m1, m2, m3] → p1 = m.fst → p2 = m.snd → Match p1 p2)

-- Given conditions
def day1 : DayMatches := 
  { m1 := (Borya, Alesha),
    m2 := (Vitya, Grisha),
    m3 := (Dima, Kostya),
    distinct := by
      intros p1 p2 m hm hp1 hp2
      rcases hm; 
      -- add specific conditions, now it's just a placeholder
      split <|> split <|> split; apply Match.ne h; finish }

def day2 : DayMatches := 
  { m1 := (Borya, Kostya),
    m2 := (Vitya, Alesha),
    m3 := (Grisha, Dima),
    distinct := sorry }

def day3 : DayMatches := 
  { m1 := (Vitya, Borya),
    m2 := (Dima, Alesha),
    m3 := (Grisha, Kostya),
    distinct := sorry }

def day4 : DayMatches := 
  { m1 := (Vitya, Kostya),
    m2 := (Alesha, Grisha),
    m3 := (Borya, Dima),
    distinct := sorry }

def day5 : DayMatches := 
  { m1 := (Vitya, Dima),
    m2 := (Borya, Grisha),
    m3 := (Alesha, Kostya),
    distinct := sorry }

-- Main theorem statement
theorem tournament_round_robin : 
  ∃ day1 day2 day3 day4 day5 : DayMatches,
  List.length [day1, day2, day3, day4, day5] = 5
  ∧ (day1.m1 = (Borya, Alesha) ∧ day1.m2 = (Vitya, Grisha) ∧ day1.m3 = (Dima, Kostya))
  ∧ (day2.m1 = (Borya, Kostya) ∧ day2.m2 = (Vitya, Alesha) ∧ day2.m3 = (Grisha, Dima))
  ∧ (day3.m1 = (Vitya, Borya) ∧ day3.m2 = (Dima, Alesha) ∧ day3.m3 = (Grisha, Kostya))
  ∧ (day4.m1 = (Vitya, Kostya) ∧ day4.m2 = (Alesha, Grisha) ∧ day4.m3 = (Borya, Dima))
  ∧ (day5.m1 = (Vitya, Dima) ∧ day5.m2 = (Borya, Grisha) ∧ day5.m3 = (Alesha, Kostya)) :=
begin
  existsi [day1, day2, day3, day4, day5],
  simp,
  split, sorry,
  repeat {split, simp [day1, day2, day3, day4, day5]},
  sorry
end

end tournament_round_robin_l103_103488


namespace angle_sum_eq_180_l103_103292

theorem angle_sum_eq_180 (A B C D X : Type*)
  (AB BC CD DA : Type*)
  (h1 : AB * CD = BC * DA)
  (h2 : ∠XAB = ∠XCD)
  (h3 : ∠XBC = ∠XDA) :
  ∠AXB + ∠CXD = 180 :=
by
  sorry

end angle_sum_eq_180_l103_103292


namespace combined_tax_rate_approx_l103_103539

noncomputable def combinedTaxRate (M : ℝ) : ℝ :=
  let Mindy := 4 * M
  let Julie := 2 * M
  let totalTax := 0.45 * M + 0.25 * Mindy + 0.35 * Julie
  let totalIncome := M + Mindy + Julie
  totalTax / totalIncome * 100

theorem combined_tax_rate_approx (M : ℝ) :
  combinedTaxRate M ≈ 30.71 :=
by sorry

end combined_tax_rate_approx_l103_103539


namespace choose_program_ways_l103_103671

theorem choose_program_ways :
  let courses := {E, A, G, H, R, L, S}   -- List of courses
  let program : Finset α := {english, algebra, geometry, history, art, latin, science}
  let mandatory_courses := {english}
  let math_courses := {algebra, geometry}
  ∀ p : Finset α,
    (mandatory_course ⊆ p) ∧ (math_courses ⊆ p ∨ ({algebra} ∈ p ∧ {geometry} ∈ p)) ∧
    (p.card = 5) → count_program_combinations = 6 :=
begin
  sorry
end

end choose_program_ways_l103_103671


namespace can_be_divided_into_two_triangles_l103_103508

-- Definitions and properties of geometrical shapes
def is_triangle (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 3 ∧ vertices = 3

def is_pentagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 5 ∧ vertices = 5

def is_hexagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 6 ∧ vertices = 6

def is_heptagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 7 ∧ vertices = 7

-- The theorem we need to prove
theorem can_be_divided_into_two_triangles :
  ∀ sides vertices,
  (is_pentagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_hexagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_heptagon sides vertices → ¬ (is_triangle sides vertices ∧ is_triangle sides vertices)) :=
by sorry

end can_be_divided_into_two_triangles_l103_103508


namespace lcm_prod_eq_factorial_l103_103186

open Nat

theorem lcm_prod_eq_factorial (n : ℕ) : 
  (∏ i in range (n + 1), lcm 1 (⌊ n/i ⌋)) = fact n :=
sorry

end lcm_prod_eq_factorial_l103_103186


namespace perpendicular_parallel_condition_not_nec_suff_l103_103406

-- Definitions for lines m, n, and plane α
variable (m n : ℝ → ℝ → ℝ)
variable (α : ℝ → ℝ)

-- Conditions that m and n are perpendicular lines and α is a plane
variable (perpendicular_m_n : ∀ (x y : ℝ), m x y = 0 ∨ n x y = 0)
variable (plane_α : ∀ (x y z : ℝ), α x y = z)

-- Define what it means for a line to be perpendicular to a plane
def perpendicular (line : ℝ → ℝ → ℝ) (plane : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), plane (line x y) = 0

-- Define what it means for a line to be parallel to a plane
def parallel (line : ℝ → ℝ → ℝ) (plane : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), ∃ c : ℝ, plane (line x y) = c

-- The statement to prove that n parallel α is neither a necessary nor a sufficient condition for m perpendicular α
theorem perpendicular_parallel_condition_not_nec_suff :
  (parallel n α) ↔ ¬ (perpendicular m α) :=
sorry

end perpendicular_parallel_condition_not_nec_suff_l103_103406


namespace a_prime_factor_congruent_1_mod_4_l103_103388

def a : ℕ → ℕ
| 1 := 1
| 2 := 2
| n := 2 * a (n-1) + a (n-2)

theorem a_prime_factor_congruent_1_mod_4 (n : ℕ) : n ≥ 5 → ∃ p : ℕ, p.prime ∧ p ∣ a n ∧ p ≡ 1 [MOD 4] :=
by
  sorry

end a_prime_factor_congruent_1_mod_4_l103_103388


namespace alley_width_l103_103811

theorem alley_width (L w : ℝ) (k h : ℝ)
    (h1 : k = L / 2)
    (h2 : h = L * (Real.sqrt 3) / 2)
    (h3 : w^2 + (L / 2)^2 = L^2)
    (h4 : w^2 + (L * (Real.sqrt 3) / 2)^2 = L^2):
    w = (Real.sqrt 3) * L / 2 := 
sorry

end alley_width_l103_103811


namespace direction_vector_of_line_l103_103982

theorem direction_vector_of_line : 
  ∃ v : ℝ × ℝ, 
  (∀ x y : ℝ, 2 * y + x = 3 → v = (-2, -1)) :=
by
  sorry

end direction_vector_of_line_l103_103982


namespace watches_same_time_after_n_days_l103_103170

theorem watches_same_time_after_n_days :
  ∃ n : ℕ, n = 1440 ∧ ∀ t : ℕ, 
    (t = 12 * n) → -- Glafira's watch deviation in seconds
    (t = -18 * n) → -- Gavrila's watch deviation in seconds
    ∃ k : ℕ, t = 43200 * k :=
by 
  sorry -- Proof omitted

end watches_same_time_after_n_days_l103_103170


namespace quad_root_c_is_3_l103_103347

theorem quad_root_c_is_3 (c : ℝ) : (∀ x : ℝ, 2 * x^2 + 8 * x + c = 0 → x ∈ {(-8 + sqrt 40) / 4, (-8 - sqrt 40) / 4}) → c = 3 :=
by sorry

end quad_root_c_is_3_l103_103347


namespace derived_sum_set_A1_derived_sum_set_A2_min_max_cardinality_of_derived_sum_set_for_n_eq_6_l103_103035

-- Definition of derived sum set
def derived_sum_set (A : Set ℕ) : Set ℕ :=
  {z | ∃ x y ∈ A, x ≠ y ∧ z = x + y}

-- Part (I)
theorem derived_sum_set_A1 :
  derived_sum_set {1, 2, 3, 4} = {3, 4, 5, 6, 7} := sorry

theorem derived_sum_set_A2 :
  derived_sum_set {1, 2, 4, 7} = {3, 5, 6, 8, 9, 11} := sorry

-- Part (II)
noncomputable def cardinality_derived_sum_set (A : Set ℕ) : ℕ :=
  Set.card (derived_sum_set A)

theorem min_max_cardinality_of_derived_sum_set_for_n_eq_6 
  {A : Finset ℕ} (hA : A.card = 6) :
  9 ≤ cardinality_derived_sum_set (↑A : Set ℕ) ∧ cardinality_derived_sum_set (↑A : Set ℕ) ≤ 15 := sorry

end derived_sum_set_A1_derived_sum_set_A2_min_max_cardinality_of_derived_sum_set_for_n_eq_6_l103_103035


namespace reflections_ellie_tall_room_l103_103709

-- Define the variables and constants
variable (E : ℕ)
constant tall_reflection_sarah : ℕ := 10
constant wide_reflection_sarah : ℕ := 5
constant wide_reflection_ellie : ℕ := 3
constant tall_room_passes : ℕ := 3
constant wide_room_passes : ℕ := 5
constant total_reflections : ℕ := 88

-- Define the expressions for total reflections seen by Sarah and Ellie
def sarah_total_reflections :=
  tall_reflection_sarah * tall_room_passes + wide_reflection_sarah * wide_room_passes

def ellie_total_reflections (E : ℕ) :=
  E * tall_room_passes + wide_reflection_ellie * wide_room_passes

-- The main theorem statement to prove
theorem reflections_ellie_tall_room : ellie_total_reflections E = 6 :=
by
  have h1 : sarah_total_reflections = 55 := by
    simp [sarah_total_reflections, tall_reflection_sarah, wide_reflection_sarah,
          tall_room_passes, wide_room_passes]
  rw [sarah_total_reflections] at h1
  have h2 : h1 + ellie_total_reflections E = total_reflections := by simp
  sorry

end reflections_ellie_tall_room_l103_103709


namespace necessary_and_sufficient_condition_l103_103042

theorem necessary_and_sufficient_condition (a : ℝ) (h : 0 < a ∧ a < 1) :
  a < sqrt a ↔ 0 < a ∧ a < 1 :=
sorry

end necessary_and_sufficient_condition_l103_103042


namespace tray_contains_40_brownies_l103_103287

-- Definitions based on conditions
def tray_length : ℝ := 24
def tray_width : ℝ := 15
def brownie_length : ℝ := 3
def brownie_width : ℝ := 3

-- The mathematical statement to prove
theorem tray_contains_40_brownies :
  (tray_length * tray_width) / (brownie_length * brownie_width) = 40 :=
by
  sorry

end tray_contains_40_brownies_l103_103287


namespace find_x_l103_103280

theorem find_x (x : ℝ) (h : (x * 74) / 30 = 1938.8) : x = 786 := by
  sorry

end find_x_l103_103280


namespace point_in_second_quadrant_l103_103213

theorem point_in_second_quadrant (a : ℝ) : 
  ∃ q : ℕ, q = 2 ∧ (-1, a^2 + 1).1 < 0 ∧ 0 < (-1, a^2 + 1).2 :=
by
  sorry

end point_in_second_quadrant_l103_103213


namespace find_x_l103_103401

theorem find_x 
  (x : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (hP : P = (x, 6)) 
  (hcos : Real.cos θ = -4/5) 
  : x = -8 := 
sorry

end find_x_l103_103401


namespace equilateral_triangles_count_l103_103844

def Point3D := (ℝ × ℝ × ℝ)
def T : set Point3D := { p | p.1 ∈ {0, 1, 2, 3} ∧ p.2 ∈ {0, 1, 2, 3} ∧ p.3 ∈ {0, 1, 2, 3} }
def distance3D (p1 p2 : Point3D) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem equilateral_triangles_count : ∃ n, n = 384 ∧ ∀ (a b c : Point3D), 
  a ∈ T → 
  b ∈ T → 
  c ∈ T → 
  distance3D a b = real.sqrt 2 → 
  distance3D b c = real.sqrt 2 → 
  distance3D c a = real.sqrt 2 → 
  true :=
sorry

end equilateral_triangles_count_l103_103844


namespace total_volume_of_four_cubes_is_500_l103_103252

-- Definition of the edge length of each cube
def edge_length : ℝ := 5

-- Definition of the volume of one cube
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the number of cubes
def number_of_cubes : ℕ := 4

-- Definition of the total volume
def total_volume (n : ℕ) (v : ℝ) : ℝ := n * v

-- The proposition we want to prove
theorem total_volume_of_four_cubes_is_500 :
  total_volume number_of_cubes (volume_of_cube edge_length) = 500 :=
by
  sorry

end total_volume_of_four_cubes_is_500_l103_103252


namespace total_volume_of_four_boxes_l103_103259

-- Define the edge length of the cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_one_cube := edge_length ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 4

-- The total volume of the four cubes
def total_volume := number_of_cubes * volume_of_one_cube

-- Statement to prove that the total volume equals 500 cubic feet
theorem total_volume_of_four_boxes :
  total_volume = 500 :=
sorry

end total_volume_of_four_boxes_l103_103259


namespace factor_3x2_minus_3y2_l103_103015

theorem factor_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factor_3x2_minus_3y2_l103_103015


namespace irrational_square_root_of_2_l103_103320

theorem irrational_square_root_of_2 : irrational (real.sqrt 2) :=
sorry

end irrational_square_root_of_2_l103_103320


namespace equal_volume_cubes_l103_103779

noncomputable def volume_box : ℝ := 1 -- volume of the cubical box in cubic meters

noncomputable def edge_length_small_cube : ℝ := 0.04 -- edge length of small cubes in meters

noncomputable def number_of_cubes : ℝ := 15624.999999999998 -- number of small cubes

noncomputable def volume_small_cube : ℝ := edge_length_small_cube^3 -- volume of one small cube

theorem equal_volume_cubes : volume_box = volume_small_cube * number_of_cubes :=
  by
  -- Proof goes here
  sorry

end equal_volume_cubes_l103_103779


namespace team_compositions_count_l103_103185

def num_team_compositions (A B : Prop) (total_sprinters : ℕ := 6) : ℕ :=
  let eligible_first_leg_sprinters := 4 -- 6 total sprinters, excluding A and B for first leg
  let remaining_sprinters := 5 -- After choosing first leg sprinter, 5 remain including A and B
  Nat.choose eligible_first_leg_sprinters 1 * Nat.permutations remaining_sprinters 3

theorem team_compositions_count (A B : Prop) : num_team_compositions A B = 240 := 
by 
  sorry

end team_compositions_count_l103_103185


namespace prism_ratio_l103_103341

theorem prism_ratio (a b c d : ℝ) (h_d : d = 60) (h_c : c = 104) (h_b : b = 78 * Real.pi) (h_a : a = (4 * Real.pi) / 3) :
  b * c / (a * d) = 8112 / 240 := 
by 
  sorry

end prism_ratio_l103_103341


namespace solve_for_x_y_l103_103440

noncomputable def x_y_2018_sum (x y : ℝ) : ℝ := x^2018 + y^2018

theorem solve_for_x_y (A B : Set ℝ) (x y : ℝ)
  (hA : A = {x, x * y, x + y})
  (hB : B = {0, |x|, y}) 
  (h : A = B) :
  x_y_2018_sum x y = 2 := 
by
  sorry

end solve_for_x_y_l103_103440


namespace area_ratio_l103_103499

-- Definitions based on the conditions given
variable {α : Type*} [EuclideanGeometry α]

structure Circle (α : Type*) [EuclideanGeometry α] :=
(center : α)
(radius : ℝ)

structure Point (α : Type*) [EuclideanGeometry α] :=
(x : ℝ)
(y : ℝ)

def diameter (circ: Circle α) (p1 p2: Point α) : Prop :=
circ.radius = EuclidDist p1 p2 / 2

def chord_parallel_diameter (circ: Circle α) (chord diam: Segment α) : Prop :=
chord ∈ circ ∧ diam ∈ circ ∧ parallel chord diam

def diagonal_intersection (circ: Circle α) (seg1 seg2: Segment α) (p: Point α) : Prop :=
intersect seg1 seg2 = some p ∧ p ∈ circ

def angle_bed (E B D: Point α) (beta: ℝ) : Prop :=
angle B E D = beta ∧ EuclidDist B E ≠ circle.radius ∧ EuclidDist E D ≠ circle.radius

-- Proof statement
theorem area_ratio {α : Type*} [EuclideanGeometry α]
  (circ: Circle α) (A B C D E: Point α)
  (h_diam: diameter circ A B)
  (h_chord: chord_parallel_diameter circ ⟨C, D⟩ ⟨A, B⟩)
  (h_diag_intersect: diagonal_intersection circ ⟨A, C⟩ ⟨B, D⟩ E)
  (h_angle_bed: angle_bed E B D β) :
  (area ⟨A, B, E⟩) / (area ⟨C, D, E⟩) = (EuclidDist A B / EuclidDist C D)^2 :=
sorry

end area_ratio_l103_103499


namespace max_elements_T_l103_103525

def T (M : Finset ℕ) : Finset (Finset ℕ) := sorry 

theorem max_elements_T : 
  let M := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} in
  let isValid (s : Finset ℕ) :=
    ∀ {a b x y : ℕ}, a ∈ M → b ∈ M → x ∈ M → y ∈ M →
    (a, b) ≠ (x, y) → 
    ¬ (11 ∣ (a * x + b * y) * (a * y + b * x) ) in
  let T := {s ∈ (Finset.powersetLen 2 M) | isValid s} in
  T.card ≤ 25 :=
begin
  let M := ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ),
  let isValid := 
    λ (s : Finset ℕ), 
      ∀ (a b x y : ℕ), 
        {a, b} ∈ s → {x, y} ∈ s → 
        (a ≠ x ∨ b ≠ y) → 
        ¬ 11 ∣ (a * x + b * y) * (a * y + b * x),
  let T := (Finset.powersetLen 2 M).filter isValid, 
  suffices : T.card ≤ 25,
  sorry,
  sorry
end

end max_elements_T_l103_103525


namespace number_of_ways_three_by_three_even_sums_l103_103817

theorem number_of_ways_three_by_three_even_sums :
  let arrangements := 6 * 7^6 * 6^3 + 9 * 7^4 * 6^5 + 6^9 in
  arrangements = 
  /-
    The number of ways to arrange natural numbers from 1 to 13 in a 3x3 grid 
    such that the sum of the numbers in each row and each column is even.
  -/
sorry

end number_of_ways_three_by_three_even_sums_l103_103817


namespace angle_GAF_degrees_l103_103987

-- Define the main theorem to prove.
theorem angle_GAF_degrees :
  let A B C D F G H : Type -- Points in the plane
    [regular_pentagon A B C D F] -- A, B, C, D, F form a regular pentagon
    [square B C G H] -- B, C, G, H form a square
    (A B C D F G H : ℝ) -- Angle measures in degrees
    (angle_ABC : ℝ) := 108 -- Internal angle of pentagon ABCDF
    (angle_BAG : ℝ) := 90 -- Right angle of square extended

  -- The angle GAF in degrees
  angle_GAF = 54 :=
sorry -- Proof omitted

end angle_GAF_degrees_l103_103987


namespace second_sheet_width_l103_103592

theorem second_sheet_width :
  ∃ w : ℝ, (286 = 22 * w + 100) ∧ w = 8.5 :=
by
  -- Proof goes here
  sorry

end second_sheet_width_l103_103592


namespace sphere_is_only_solid_with_same_views_l103_103634

-- Definitions of viewing properties for the solids
def sphere_has_same_views (S : Type) : Prop :=
  ∀ (angle : ℝ), S = sphere

def cube_views_might_differ (C : Type) : Prop :=
  ∃ (angle : ℝ), ¬ (C = cube)

def tetrahedron_views_differ (T : Type) : Prop :=
  ∃ (view1 view2 : ℕ), view1 ≠ view2 ∧ T = regular_tetrahedron

-- Main proposition
theorem sphere_is_only_solid_with_same_views (S C T : Type) :
  (sphere_has_same_views S ∧ cube_views_might_differ C ∧ tetrahedron_views_differ T) → S = sphere ∧ T ≠ sphere ∧ C ≠ sphere :=
sorry

end sphere_is_only_solid_with_same_views_l103_103634


namespace sqrt_x_minus_2_meaningful_in_reals_l103_103469

theorem sqrt_x_minus_2_meaningful_in_reals (x : ℝ) : (∃ (y : ℝ), y * y = x - 2) → x ≥ 2 :=
by
  sorry

end sqrt_x_minus_2_meaningful_in_reals_l103_103469


namespace total_seats_theater_l103_103964

theorem total_seats_theater (a1 an d n Sn : ℕ) 
    (h1 : a1 = 12) 
    (h2 : d = 2) 
    (h3 : an = 48) 
    (h4 : an = a1 + (n - 1) * d) 
    (h5 : Sn = n * (a1 + an) / 2) : 
    Sn = 570 := 
sorry

end total_seats_theater_l103_103964


namespace factor_expression_l103_103000

theorem factor_expression (m n x y : ℝ) :
  m * (x - y) + n * (y - x) = (x - y) * (m - n) := by
  sorry

end factor_expression_l103_103000


namespace pq_sum_is_38_l103_103044

theorem pq_sum_is_38
  (p q : ℝ)
  (h_root : ∀ x, (2 * x^2) + (p * x) + q = 0 → x = 2 * Complex.I - 3 ∨ x = -2 * Complex.I - 3)
  (h_p_q : ∀ a b : ℂ, a + b = -p / 2 ∧ a * b = q / 2 → p = 12 ∧ q = 26) :
  p + q = 38 :=
sorry

end pq_sum_is_38_l103_103044


namespace area_of_triangle_l103_103277

variables {A B C H O : Type}
variables [HilbertPlane A B C] 

-- Conditions
def acute_triangle (ABC : Type) : Prop :=
  is_acute_angle ∠A ∧ is_acute_angle ∠B ∧ is_acute_angle ∠C

def altitudes_intersect_at (H : Type) (ABC : Type) : Prop :=
  meet_at H (altitude A) (altitude B) (altitude C)

def medians_intersect_at (O : Type) (ABC : Type) : Prop :=
  meet_at O (median A) (median B) (median C)

def bisector_passes_through_midpoint (A : Type) (P mid_OH : Type) : Prop :=
  bisector A intersects middle_point OH

def side_length (BC : ℝ) := BC = 2
def angle_difference (B C : Type) := angle B - angle C = 30

-- Theorem to prove
theorem area_of_triangle {A B C H O : Type} :
  acute_triangle ABC ∧ 
  altitudes_intersect_at H ABC ∧
  medians_intersect_at O ABC ∧
  bisector_passes_through_midpoint A (midpoint OH) ∧
  side_length BC ∧ 
  angle_difference B C → 
  area ABC = (2 * sqrt 3 + 1) / sqrt 15 :=
sorry

end area_of_triangle_l103_103277


namespace sum_of_squares_of_perpendicular_chords_constant_l103_103571

variables {O P : Point}
variables {r OP : ℝ}
variables {h1 h2 d1 d2 : ℝ}
variables [Fact (r > 0)]
variables [Fact (OP ≤ r)]

def perpendicular_chords (h1 h2 : ℝ) := 
  ∃ d1 d2 : ℝ, h1 = 2 * sqrt (r^2 - d1^2) ∧ h2 = 2 * sqrt (r^2 - d2^2) ∧
    (d1^2 + d2^2 = OP^2)

theorem sum_of_squares_of_perpendicular_chords_constant 
  (h1 h2 : ℝ) 
  (h_prod : perpendicular_chords h1 h2) : 
  h1^2 + h2^2 = 8 * r^2 - 4 * OP^2 :=
by sorry

end sum_of_squares_of_perpendicular_chords_constant_l103_103571


namespace probability_letter_in_MATHEMATICS_l103_103795

theorem probability_letter_in_MATHEMATICS :
  let alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  let mathematics := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']
  (mathematics.length : ℚ) / (alphabet.length : ℚ) = 4 / 13 :=
by
  sorry

end probability_letter_in_MATHEMATICS_l103_103795


namespace acute_angle_ACD_l103_103218

theorem acute_angle_ACD (α : ℝ) (h : α ≤ 120) :
  ∃ (ACD : ℝ), ACD = Real.arcsin ((Real.tan (α / 2)) / Real.sqrt 3) :=
sorry

end acute_angle_ACD_l103_103218


namespace simple_interest_correct_l103_103955

theorem simple_interest_correct (P R T : ℝ) (hP : P = 400) (hR : R = 12.5) (hT : T = 2) : 
  (P * R * T) / 100 = 50 :=
by
  sorry -- Proof to be provided

end simple_interest_correct_l103_103955


namespace unique_c_to_have_exactly_3_distinct_real_roots_l103_103518

theorem unique_c_to_have_exactly_3_distinct_real_roots
  (f : ℝ → ℝ) -- Declare the function f
  (c : ℝ) -- Declare the real number c
  (h : ∀ x : ℝ, f x = x^2 + 6 * x + c) -- Conditional definition of f(x)
  : ∃! c, (∃ r s t : ℝ, r ≠ s ∧ s ≠ t ∧ r ≠ t ∧ 
          ∃ x1 x2 x3 : ℝ, f (f x1) = 0 ∧ f (f x2) = 0 ∧ f (f x3) = 0) ↔ 
          c = (11 - Real.sqrt 13) / 2 :=
proof 
  sorry

end unique_c_to_have_exactly_3_distinct_real_roots_l103_103518


namespace find_base_17_digit_l103_103410

theorem find_base_17_digit (a : ℕ) (h1 : 0 ≤ a ∧ a < 17) 
  (h2 : (25 + a) % 16 = 0) : a = 7 :=
sorry

end find_base_17_digit_l103_103410


namespace q1_q2_l103_103020

variable (a b : ℝ)

-- Definition of the conditions
def conditions : Prop := a + b = 7 ∧ a * b = 6

-- Statement of the first question
theorem q1 (h : conditions a b) : a^2 + b^2 = 37 := sorry

-- Statement of the second question
theorem q2 (h : conditions a b) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = 150 := sorry

end q1_q2_l103_103020


namespace lcm_consecutive_impossible_l103_103031

def lcm (a b : Nat) : Nat := 
  (a * b) / (Nat.gcd a b)

theorem lcm_consecutive_impossible (n : Nat) (a : Fin n → Nat) 
  (h : n = 10^1000) :
  ¬∃ (b : Fin n → Nat), (∀ i : Fin n, b i = lcm (a i) (a (i + 1))) ∧ 
  (∀ i : Fin (n - 1), b i + 1 = b (i + 1)) :=
by
  sorry

end lcm_consecutive_impossible_l103_103031


namespace painted_surface_area_of_pyramid_l103_103680

/--
Given 19 unit cubes arranged in a 4-layer pyramid-like structure, where:
- The top layer has 1 cube,
- The second layer has 3 cubes,
- The third layer has 5 cubes,
- The bottom layer has 10 cubes,

Prove that the total painted surface area is 43 square meters.
-/
theorem painted_surface_area_of_pyramid :
  let layer1 := 1 -- top layer
  let layer2 := 3 -- second layer
  let layer3 := 5 -- third layer
  let layer4 := 10 -- bottom layer
  let total_cubes := layer1 + layer2 + layer3 + layer4
  let top_faces := layer1 * 1 + layer2 * 1 + layer3 * 1 + layer4 * 1
  let side_faces_layer1 := layer1 * 5
  let side_faces_layer2 := layer2 * 3
  let side_faces_layer3 := layer3 * 2
  let side_faces := side_faces_layer1 + side_faces_layer2 + side_faces_layer3
  let total_surface_area := top_faces + side_faces
  total_cubes = 19 → total_surface_area = 43 :=
by
  intros
  sorry

end painted_surface_area_of_pyramid_l103_103680


namespace molecular_weight_KBrO3_is_167_l103_103246

def molecular_weight {K Br O : Type} (wK wBr wO : ℝ) (nK nBr nO : ℕ) : ℝ :=
  nK * wK + nBr * wBr + nO * wO

theorem molecular_weight_KBrO3_is_167 :
  molecular_weight 39.10 79.90 16.00 1 1 3 = 167.00 :=
by
  sorry

end molecular_weight_KBrO3_is_167_l103_103246


namespace largest_power_of_two_dividing_50_factorial_l103_103744

def legendre (n p : ℕ) := ∑ k in finset.range (n.log p + 1), n / p^k

theorem largest_power_of_two_dividing_50_factorial : legendre 50 2 = 47 := by
  sorry

end largest_power_of_two_dividing_50_factorial_l103_103744


namespace area_inequality_l103_103818

open Real EuclideanGeometry

variables {A B C D I J K L : Point}

-- Define the problem conditions.
def right_triangle (A B C : Point) :=
  ∃ D : Point, is_right_triangle A B C ∧ is_foot_of_altitude A (line_through B C) D 

def incenters (A B C D I J : Point) :=
  incenter A B D I ∧ incenter A C D J

def intersections (A B C I J K L : Point) :=
  line_through I J ⊓ line_through A B = K ∧ line_through I J ⊓ line_through A C = L

-- Lean statement of the proof problem.
theorem area_inequality (A B C D I J K L : Point)
  (h_right : right_triangle A B C)
  (h_incenters : incenters A B C D I J)
  (h_intersections : intersections A B C I J K L) :
  area A B C ≥ 2 * area A K L := 
begin
  sorry
end

end area_inequality_l103_103818


namespace if_planes_parallel_then_perp_lines_l103_103399

variables {l m : Mathlib.RealLine} {α β : Mathlib.RealPlane}

-- Definitions of the conditions
def perp_line_to_plane (l : Mathlib.RealLine) (α : Mathlib.RealPlane) : Prop := ...
def line_in_plane (m : Mathlib.RealLine) (β : Mathlib.RealPlane) : Prop := ...
def planes_parallel (α β : Mathlib.RealPlane) : Prop := ...
def perp_lines (l m : Mathlib.RealLine) : Prop := ...

-- Conditions given in the problem
axiom H1 : perp_line_to_plane l α
axiom H2 : line_in_plane m β

-- The conlusion we need to prove
theorem if_planes_parallel_then_perp_lines (H3 : planes_parallel α β) : perp_lines l m :=
sorry

end if_planes_parallel_then_perp_lines_l103_103399


namespace expr_eval_l103_103237

theorem expr_eval : 180 / 6 * 2 + 5 = 65 := by
  sorry

end expr_eval_l103_103237


namespace triangle_is_right_l103_103829

theorem triangle_is_right {A B C : ℝ} (h : A + B + C = 180) (h1 : A = B + C) : A = 90 :=
by
  sorry

end triangle_is_right_l103_103829


namespace sqrt_domain_l103_103473

theorem sqrt_domain (x : ℝ) : (∃ y, y * y = x - 2) ↔ (x ≥ 2) :=
by sorry

end sqrt_domain_l103_103473


namespace hexagon_perimeter_l103_103953

-- Condition definitions
def AB := 1
def BC := 1.5
def CD := 1.5
def DE := 1.5
def EF := Real.sqrt 3
def FA := 2

-- Theorem statement
theorem hexagon_perimeter :
  let P := AB + BC + CD + DE + EF + FA
  P = 7.5 + Real.sqrt 3 :=
by {
  sorry
}

end hexagon_perimeter_l103_103953


namespace distance_greater_than_d_l103_103733

theorem distance_greater_than_d (n : ℕ) (points : Fin n → ℝ × ℝ) (d : ℝ) 
  (h_n_ge_4 : 4 ≤ n)
  (h_pairs_d : {p : (Fin n × Fin n) // p.1 < p.2} → Prop)
  (hd_pos : 0 < d)
  (h_more_than_n : ∑ p in Finset.univ.filter (λ p, p.1.dist p.2 = d), 1 > n) :
  ∃ (p : Fin n × Fin n), p.1 < p.2 ∧ p.1.dist p.2 > d := 
sorry

end distance_greater_than_d_l103_103733


namespace locus_of_A_l103_103753

-- Define the problem variables and conditions
def ellipse : set (ℂ) := {z : ℂ | (z.re^2 / 9) + (z.im^2 / 5) = 1}

def F : ℂ := 2 -- Right focal point F(2, 0)

def on_ellipse (B : ℂ) : Prop := B ∈ ellipse

def equilateral_triangle (A B : ℂ) : Prop := 
  ∃ C : ℂ, C = F ∧
  A - F = (B - F) * exp (complex.I * (2 * π / 3)) ∧
  B - F = (A - F) * exp (complex.I * (2 * π / 3))

def counterclockwise (A B : ℂ) : Prop := 
  (F.re - A.re) * (A.im - B.im) - (A.re - B.re) * (F.im - A.im) > 0

-- Proof statement that links all these together
theorem locus_of_A (A : ℂ) (B : ℂ) 
  (h1 : on_ellipse B) 
  (h2 : equilateral_triangle A B) 
  (h3 : counterclockwise A B) : 
  |A - 2| + |A - 2 * complex.I * real.sqrt 3| = 6 :=
sorry

end locus_of_A_l103_103753


namespace exists_member_T_divisible_by_3_l103_103161

-- Define the set T of all numbers which are the sum of the squares of four consecutive integers
def T := { x : ℤ | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 }

-- Theorem to prove that there exists a member in T which is divisible by 3
theorem exists_member_T_divisible_by_3 : ∃ x ∈ T, x % 3 = 0 :=
by
  sorry

end exists_member_T_divisible_by_3_l103_103161


namespace central_angle_of_sector_l103_103736

variable (θ : ℝ)
variable (r : ℝ) := 10
variable (P : ℝ) := 45

theorem central_angle_of_sector : 10 * θ + 20 = 45 → θ = 2.5 :=
by
  intro h
  sorry

end central_angle_of_sector_l103_103736


namespace prove_counterfeit_and_genuine_l103_103978

def coins : Fin 14 → ℝ
def is_counterfeit (c : Fin 14) : Prop := c.val < 7
def is_genuine (c : Fin 14) : Prop := c.val ≥ 7
def is_lighter (c1 c2 : ℝ) : Prop := c1 < c2

theorem prove_counterfeit_and_genuine : 
  (∀ c1 c2, is_counterfeit c1 → is_genuine c2 → is_lighter (coins c1) (coins c2)) →
  (∀ c1 c2, is_counterfeit c1 → is_counterfeit c2 → coins c1 = coins c2) →
  (∀ c1 c2, is_genuine c1 → is_genuine c2 → coins c1 = coins c2) →
  (∃ w1 w2 w3 : list (Fin 14),
    w1 = [0, 7] ∧
    w2 = [1, 2, 7] ∧
    w3 = [3, 4, 5, 6, 7, 8, 9] ∧
    list.sum (w1.map coins) < list.sum (w1.tail.map coins) ∧
    list.sum (w2.take 1.map coins) < list.sum (w2.drop 1.map coins) ∧
    list.sum (w3.take 1.map coins) < list.sum (w3.drop 1.map coins))
    → ∀ c1, (is_counterfeit c1 ∨ is_genuine c1) :=
sorry

end prove_counterfeit_and_genuine_l103_103978


namespace bozan_ball_strategy_exists_l103_103125

def boxes := {x : ℕ // 1 ≤ x ∧ x ≤ 2018}

theorem bozan_ball_strategy_exists :
  ∃ (assign_balls : (fin 4032 → (boxes × boxes)) → (boxes → ℕ)), 
  (∀ i j : boxes, i ≠ j → assign_balls (λ k, some (i,j)) i ≠ assign_balls (λ k, some (i,j)) j) :=
sorry

end bozan_ball_strategy_exists_l103_103125


namespace oc_perp_mn_l103_103506

-- Definitions for points and lines
variables {Point : Type*} [AffineSpace ℝ Point]
variables {A B C D E M N O : Point}

-- Definitions for perpendiculars
def Perpendicular (p1 p2: Point) : Prop := sorry  -- Placeholder definition

-- Problem conditions
variables (ha1 : C ∈ interior (∠ A O B))
variables (ha2 : Perpendicular C D OA)
variables (ha3 : Perpendicular C E OB)
variables (ha4 : Perpendicular E M OA)
variables (ha5 : Perpendicular D N OB)

-- Proof goal
theorem oc_perp_mn : Perpendicular O C MN := sorry

end oc_perp_mn_l103_103506


namespace a10_over_b10_l103_103909

-- Define arithmetic sequences and their sum formulas
variables {a b : ℕ → ℚ}  -- Sequences a_n and b_n

def S (n : ℕ) := (n * (a 1 + a n)) / 2
def T (n : ℕ) := (n * (b 1 + b n)) / 2

-- Conditions given in the problem
axiom arith_seq_S: ∀ n, S n = (7 * n + 2) * T n / (n + 3)

-- Theorem to prove
theorem a10_over_b10 : 
  (a 10) / (b 10) = 135 / 22 :=
by
  -- Steps of the proof would go here
  sorry

end a10_over_b10_l103_103909


namespace number_of_possible_routes_l103_103105

def f (x y : ℕ) : ℕ :=
  if y = 2 then sorry else sorry -- Here you need the exact definition of f(x, y)

theorem number_of_possible_routes (n : ℕ) (h : n > 0) : 
  f n 2 = (1 / 2 : ℚ) * (n^2 + 3 * n + 2) := 
by 
  sorry

end number_of_possible_routes_l103_103105


namespace tan_sum_gt_3_l103_103181

-- Conditions as definitions
def tan_40 : ℝ := Real.tan (40 * Real.pi / 180)
def tan_50 : ℝ := Real.tan (50 * Real.pi / 180)
def tan_45 : ℝ := 1

-- Main statement
theorem tan_sum_gt_3 : tan_40 + tan_45 + tan_50 > 3 := 
by
  -- Placeholder for the proof
  sorry

end tan_sum_gt_3_l103_103181


namespace geometric_sequence_sixth_term_l103_103894

-- Definitions of conditions
def a : ℝ := 512
def r : ℝ := (2 / a)^(1 / 7)

-- The proof statement
theorem geometric_sequence_sixth_term (h : a * r^7 = 2) : 512 * (r^5) = 16 :=
begin
  sorry
end

end geometric_sequence_sixth_term_l103_103894


namespace polynomial_quotient_correct_l103_103954

noncomputable def pol1 : Polynomial ℚ := 6 * Polynomial.X^4 + 5 * Polynomial.X^3 - 4 * Polynomial.X^2 + Polynomial.X + 1
noncomputable def pol2 : Polynomial ℚ := 3 * Polynomial.X + 1
noncomputable def quotient : Polynomial ℚ := 2 * Polynomial.X^3 + Polynomial.X^2 - (7 / 3) * Polynomial.X + (20 / 9)

theorem polynomial_quotient_correct :
  Polynomial.div(pol1, pol2, 0) = quotient :=
by
  sorry

end polynomial_quotient_correct_l103_103954


namespace inequality_not_hold_l103_103380

theorem inequality_not_hold (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) < 1 / a) :=
by
  sorry

end inequality_not_hold_l103_103380


namespace maximum_sine_sum_l103_103365

theorem maximum_sine_sum (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) : 
  ∃ M, (∀ θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ, 
    (sin (θ₁ - θ₂) + sin (θ₂ - θ₃) + sin (θ₃ - θ₄) + sin (θ₄ - θ₅) + sin (θ₅ - θ₁)) ≤ M) 
  ∧ (∀ N, (∀ θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ, 
    (sin (θ₁ - θ₂) + sin (θ₂ - θ₃) + sin (θ₃ - θ₄) + sin (θ₄ - θ₅) + sin (θ₅ - θ₁)) ≤ N) → M ≤ N) 
  ∧ M = 0 := 
sorry

end maximum_sine_sum_l103_103365


namespace imo1984_p25_l103_103359

theorem imo1984_p25 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (a = 18) ∧ (b = 1) →
  ¬ (7 ∣ a * b * (a + b)) ∧ (7^7 ∣ ((a + b)^7 - a^7 - b^7)) :=
by
  intros h ab
  cases h with ha hb
  rw [ha, hb] at ab
  split
  {
    intro hdiv
    simp only [add_right_eq_self, one_mul, mul_pos, Nat.succ_pos', nat.cast_add, nat.cast_one, 
                Nat.not_div_lt_div] at hdiv
    norm_cast at hdiv
    linarith
  },
  {
    use 7
    ring_nf
    norm_num
  }
  sorry

end imo1984_p25_l103_103359


namespace ellipse_eccentricity_is_sqrt2_over_2_ellipse_equation_is_correct_l103_103059

def ellipse_eccentricity (a b : ℝ) (h₀ : a > b) : ℝ := 
  (1 - (b * b) / (a * a))^0.5

theorem ellipse_eccentricity_is_sqrt2_over_2 (a b : ℝ) (ha : a = sqrt 2 * b) (h₀ : a > b) :
  ellipse_eccentricity a b h₀ = (sqrt 2) / 2 :=
by
  sorry

def ellipse_equation (a b : ℝ) : Prop :=
  ∀x y : ℝ, (x * x) / (a * a) + (y * y) / (b * b) = 1

theorem ellipse_equation_is_correct (a b c : ℝ) (ha : a = 6 * sqrt 2) (hb : b = 6) (hc : c = 6) :
  ellipse_equation (sqrt 72) 6 :=
by
  sorry

end ellipse_eccentricity_is_sqrt2_over_2_ellipse_equation_is_correct_l103_103059


namespace probability_of_females_right_of_males_l103_103226

-- Defining the total and favorable outcomes
def total_outcomes : ℕ := Nat.factorial 5
def favorable_outcomes : ℕ := Nat.factorial 3 * Nat.factorial 2

-- Defining the probability as a rational number
def probability_all_females_right : ℚ := favorable_outcomes / total_outcomes

-- Stating the theorem
theorem probability_of_females_right_of_males :
  probability_all_females_right = 1 / 10 :=
by
  -- Proof to be filled in
  sorry

end probability_of_females_right_of_males_l103_103226


namespace proposition_count_l103_103428

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

def central_symmetry (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, g (2 * p.1 - x) = -g x

def axis_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, g (2 * a - x) = g x

def monotonic_increasing (g : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → g x ≤ g y

theorem proposition_count : 
  let p1 := central_symmetry g (ℝ.pi / 6, 0)
  let p2 := axis_symmetry g (ℝ.pi / 6)
  let p3 := monotonic_increasing g (Set.Icc (ℝ.pi / 12) (5 * ℝ.pi / 12))
  (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) = 2 := by
  sorry

end proposition_count_l103_103428


namespace percentage_of_60_eq_15_l103_103905

-- Conditions provided in the problem
def percentage (p : ℚ) : ℚ := p / 100
def num : ℚ := 60
def fraction_of_num (p : ℚ) (n : ℚ) : ℚ := (percentage p) * n

-- Assertion to be proved
theorem percentage_of_60_eq_15 : fraction_of_num 25 num = 15 := 
by 
  show fraction_of_num 25 60 = 15
  sorry

end percentage_of_60_eq_15_l103_103905


namespace line_intersects_y_axis_at_1_l103_103952

theorem line_intersects_y_axis_at_1 :
  ∀ (x y : ℝ), 
  let p1 := (2 : ℝ, 9 : ℝ),
      p2 := (4 : ℝ, 17 : ℝ),
      m := (17 - 9) / (4 - 2 : ℝ),
      b := 9 - m * 2,
      line_eq := λ x, m * x + b
  in (p1, p2) -> x = 0 -> y = line_eq x -> (x, y) = (0, 1) :=
by
  intros x y p1 p2 m b line_eq hP hx hy
  -- skipped proof
  sorry

end line_intersects_y_axis_at_1_l103_103952


namespace inscribed_centers_equidistant_l103_103666

variables {A B C : Type} [Point A] [Point B] [Point C]
variables (triangle_ABC : Triangle A B C) (right_angle_at_C : triangle_ABC.right_angle C)
variables (H_3 : Point) (CH_3_altitude : Altitude triangle_ABC C H_3)
variables (O O_1 O_2 : Point) (incenter_ABC : Incenter triangle_ABC O)
variables (incenter_ACH_3 : Incenter (triangle_ABC.ach_3 A H_3) O_1)
variables (incenter_BCH_3 : Incenter (triangle_ABC.bch_3 B H_3) O_2)
variables (tangency_point : Point) (tangency_condition : Tangency incenter_ABC hypotenuse_A_B tangency_point)

theorem inscribed_centers_equidistant :
  Distance O tangency_point = Distance O_1 tangency_point 
  ∧ Distance O tangency_point = Distance O_2 tangency_point
  :=
sorry

end inscribed_centers_equidistant_l103_103666


namespace power_division_result_l103_103643

theorem power_division_result : (-2)^(2014) / (-2)^(2013) = -2 :=
by
  sorry

end power_division_result_l103_103643


namespace longest_side_quadrilateral_l103_103346

theorem longest_side_quadrilateral :
  (∀ (x y : ℝ), x + y ≤ 5 → 2 * x + y ≥ 3 → x ≥ 1 → y ≥ 0 →
  (∃ a b c d : ℝ, a = (1, 4) ∧ b = (1, 1) ∧ c = (1.5, 0) ∧ d = (1, 0) ∧
   (sqrt ((1 - 1)^2 + (4 - 1)^2) = 3 ∧
    sqrt ((1.5 - 1)^2 + (1 - 0)^2) = (sqrt 5 / 2) ∧
    sqrt ((1.5 - 1)^2 + (0 - 0)^2) = 0.5)) → 3 = 3) :=
by sorry

end longest_side_quadrilateral_l103_103346


namespace horner_value_v3_at_neg4_l103_103232

theorem horner_value_v3_at_neg4 :
  let f : ℤ → ℤ := λ x, 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6
  let x : ℤ := -4
  let v0 : ℤ := 3
  let v1 : ℤ := v0 * x + 5
  let v2 : ℤ := v1 * x + 6
  let v3 : ℤ := v2 * x + 79
  (v3 = -57) :=
by
  have v0_def : v0 = 3 := rfl
  have v1_def : v1 = v0 * x + 5 := rfl
  have v2_def : v2 = v1 * x + 6 := rfl
  have v3_def : v3 = v2 * x + 79 := rfl
  show v3 = -57
  -- Further steps to show the proof would follow, ending in:
  sorry

end horner_value_v3_at_neg4_l103_103232


namespace sum_of_first_n_terms_exists_arithmetic_sequence_consecutive_terms_l103_103123

noncomputable def a : ℕ → ℝ :=
  sorry -- Definition of a_n needs to be provided

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a i

theorem sum_of_first_n_terms (n : ℕ) : 
  S n = n * (n + 1) - 4 + (4 + 2 * n) / 2 ^ n :=
by
  sorry

theorem exists_arithmetic_sequence_consecutive_terms :
  ∃ k ≥ 2, ∀ n, (a (k - 1) + a (k + 1) = 2 * a k) ↔ (k = 3) :=
by
  sorry

end sum_of_first_n_terms_exists_arithmetic_sequence_consecutive_terms_l103_103123


namespace quadratic_other_x_intercept_l103_103726

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → a * x^2 + b * x + c = -3)
  (h_intercept : a * 1^2 + b * 1 + c = 0) : 
  ∃ x0 : ℝ, x0 = 9 ∧ a * x0^2 + b * x0 + c = 0 :=
by
  sorry

end quadratic_other_x_intercept_l103_103726


namespace segments_equal_l103_103291

theorem segments_equal (A B C D A_0 B_0 C_0 D_0 : Point)
  (h_rectangle : rectangle ABCD)
  (h_triangles : four_right_triangles_from_rectangle ABCD A_0 B_0 C_0 D_0) :
  distance A_0 C_0 = distance B_0 D_0 :=
sorry

end segments_equal_l103_103291


namespace sphere_surface_area_of_cube_vertices_l103_103980

-- Given conditions
def cube_edge_length : ℝ := 1
def cube_diameter : ℝ := Real.sqrt 3
def sphere_surface_area (d : ℝ) := 4 * Real.pi * (d / 2)^2

-- Goal to prove
theorem sphere_surface_area_of_cube_vertices :
  sphere_surface_area cube_diameter = 3 * Real.pi :=
by
  unfold sphere_surface_area cube_diameter cube_edge_length
  sorry

end sphere_surface_area_of_cube_vertices_l103_103980


namespace problem_1_problem_2_l103_103061

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem problem_1 :
  (∀ x, 0 ≤ x → x ≤ 1 → a = 2 → f a x = 2^x + log 2 (x + 1)) →
  (a = 2 ∧ ∀ x, 0 ≤ x ∧ x ≤ 1 → f a x ≤ 3) :=
by
  sorry

theorem problem_2 :
  (∀ x, 0 ≤ x → x ≤ 1 → 0 < a → a < 1 → f a x = a^x + Real.logb a (x + 1)) →
  (0 < a ∧ a < 1 ∧ f a 1 = a + Real.logb a 2 → f a 0 = 1 + Real.logb a 1 ∧ (1 + Real.logb a 1) + (a + Real.logb a 2) = a) →
  (a = 1/2) :=
by
  sorry

end problem_1_problem_2_l103_103061


namespace prob1_prob2_l103_103645

-- Definitions and conditions for Problem 1
def U : Set ℝ := {x | x ≤ 4}
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof Problem 1: Equivalent Lean proof statement
theorem prob1 : (U \ A) ∩ B = {-3, -2, 3} := by
  sorry

-- Definitions and conditions for Problem 2
def tan_alpha_eq_3 (α : ℝ) : Prop := Real.tan α = 3

-- Proof Problem 2: Equivalent Lean proof statement
theorem prob2 (α : ℝ) (h : tan_alpha_eq_3 α) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 ∧
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = -4 / 5 := by
  sorry

end prob1_prob2_l103_103645


namespace log_problem_l103_103787

theorem log_problem (x : ℝ) (hx : log 49 (x - 6) = 1 / 2) :
    1 / log x 7 = log 10 13 / log 10 7 :=
sorry

end log_problem_l103_103787


namespace triangle_similarity_l103_103517

-- Definitions for the geometry of the problem
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the excircle tangency points, intersection, and symmetrical points
def isExcircleTangent (C B P Q : Point) := sorry
def isIntersection (P Q M N A_1 : Point) := sorry
def isSymmetricPoint (A A_1 A_2 : Point) := sorry

-- Conditions for the problem
variable (A B C M N P Q A_1 A_2 B_2 C_2 : Point)

-- Triangle ABC is acute and scalene
axiom acute_scalene_triangle (t : Triangle) : true

-- C-excircle tangent points
axiom C_excircle_tangent_to_AB (M A B C N : Point) : isExcircleTangent C A M N
axiom C_excircle_tangent_to_extension_BC_beyond_B (N B C : Point) : isExcircleTangent C B N (Point.mk 0 0)

-- B-excircle tangent points
axiom B_excircle_tangent_to_AC (P A C Q : Point) : isExcircleTangent B A P Q
axiom B_excircle_tangent_to_extension_BC_beyond_C (Q B C : Point) : isExcircleTangent B C Q (Point.mk 0 0)

-- Intersection and symmetric points
axiom intersection_point_A1 (M N P Q A_1 : Point) : isIntersection P Q M N A_1
axiom symmetric_point_A2 (A A_1 : Point) : isSymmetricPoint A A_1 A_2
axiom symmetric_point_B2 (B B_1 : Point) : isSymmetricPoint B B_1 B_2
axiom symmetric_point_C2 (C C_1 : Point) : isSymmetricPoint C C_1 C_2

-- Prove the similarity
theorem triangle_similarity (ABC A_2B_2C_2 : Triangle) : 
  acute_scalene_triangle ABC → 
  (∀(M N P Q A_1 A_2 B_2 C_2 : Point),
   C_excircle_tangent_to_AB M ABC.A ABC.B ABC.C N →
   C_excircle_tangent_to_extension_BC_beyond_B N ABC.B ABC.C →
   B_excircle_tangent_to_AC P ABC.A ABC.C Q →
   B_excircle_tangent_to_extension_BC_beyond_C Q ABC.B ABC.C →
   intersection_point_A1 M N P Q A_1 →
   symmetric_point_A2 ABC.A A_1 A_2 →
   symmetric_point_B2 ABC.B (Point.mk 0 0) B_2 →
   symmetric_point_C2 ABC.C (Point.mk 0 0) C_2 →
   ABC.A = A ∧ ABC.B = B ∧ ABC.C = C → 
   A_2B_2C_2.A = A_2 ∧ A_2B_2C_2.B = B_2 ∧ A_2B_2C_2.C = C_2)
→ (∀ (ABC A_2B_2C_2 : Triangle), ∃ (A_2 B_2 C_2 : Point), 
  (ABC.A = A_2) ∧ (ABC.B = B_2) ∧ (ABC.C = C_2)) :=
sorry

end triangle_similarity_l103_103517


namespace common_tangents_collinear_l103_103885

/-- Let S₁, S₂, and S₃ be three circles.
Let A be the intersection of the common external tangents to the circles S₁ and S₂.
Let B be the intersection of the common external tangents to the circles S₂ and S₃.
Let C be the intersection of the common external tangents to the circles S₃ and S₁.
We need to prove that points A, B, and C are collinear. -/
theorem common_tangents_collinear
  (S₁ S₂ S₃ : Circle)
  (A : Point) (hA : IsIntersectionOfCommonExternalTangents S₁ S₂ A)
  (B : Point) (hB : IsIntersectionOfCommonExternalTangents S₂ S₃ B)
  (C : Point) (hC : IsIntersectionOfCommonExternalTangents S₃ S₁ C) :
  collinear {A, B, C} :=
sorry

end common_tangents_collinear_l103_103885


namespace alice_paid_24_percent_l103_103917

theorem alice_paid_24_percent (P : ℝ) (h1 : P > 0) :
  let MP := 0.60 * P
  let price_paid := 0.40 * MP
  (price_paid / P) * 100 = 24 :=
by
  sorry

end alice_paid_24_percent_l103_103917


namespace central_lit_bulb_all_off_l103_103647

-- Define the size of the grid
def grid_size : ℕ := 5

-- Define the initial condition: only one light bulb is on
def initial_condition (grid : Matrix (Fin grid_size) (Fin grid_size) Bool) : Prop :=
  ∃ i j, grid i j = true ∧ ∀ i' j', (i' ≠ i ∨ j' ≠ j) → grid i' j' = false

-- Define the allowable operation (toggle the state of bulbs in any k x k square, k > 1)
def toggle (grid : Matrix (Fin grid_size) (Fin grid_size) Bool) (k : ℕ) (cond : k > 1)
  (i j : Fin grid_size) : Matrix (Fin grid_size) (Fin grid_size) Bool :=
by sorry  -- Detailed toggle function implementation is skipped

-- The proof problem statement
theorem central_lit_bulb_all_off :
  ∀ grid : Matrix (Fin grid_size) (Fin grid_size) Bool,
    initial_condition grid →
    (∃ i j, grid i j = true) ∧ 
    (∀ (k : ℕ) (hk : k > 1) (i j : Fin grid_size), 
      toggle grid k hk i j = (Matrix.zero (Fin grid_size) (Fin grid_size))) →
    ∃ i j, i = 2 ∧ j = 2 :=
by sorry

end central_lit_bulb_all_off_l103_103647


namespace right_triangle_formation_l103_103957

-- Defining the sets of line segments.
def setA : (ℕ × ℕ × ℕ) := (2, 3, 4)
def setB : (ℕ × ℕ × ℕ) := (3, 4, 5)
def setC : (ℕ × ℕ × ℕ) := (4, 5, 6)
def setD : (ℕ × ℕ × ℕ) := (5, 6, 7)

-- Stating the problem in Lean 4
theorem right_triangle_formation (a b c : ℕ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) :
  ((a = 3 ∧ b = 4 ∧ c = 5) ∧ (a^2 + b^2 = c^2)) :=
by
  sorry

-- Theorem to assert only set B can form a right triangle
example : ¬((setA.1^2 + setA.2^2 = setA.3^2) ∨ (setC.1^2 + setC.2^2 = setC.3^2) ∨ (setD.1^2 + setD.2^2 = setD.3^2)) ∧
  (setB.1^2 + setB.2^2 = setB.3^2) :=
by
  sorry

end right_triangle_formation_l103_103957


namespace intersection_complement_l103_103017

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := Icc 0 2
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, x > 0 ∧ y = 2^x}

-- Complement of B in U (which is ℝ)
def C_U_B : Set ℝ := {y : ℝ | y ≤ 1}

theorem intersection_complement :
  (A ∩ C_U_B) = Icc (0 : ℝ) 1 :=
begin
  sorry
end

end intersection_complement_l103_103017


namespace probability_of_one_red_ball_is_one_third_l103_103490

-- Define the number of red and black balls
def red_balls : Nat := 2
def black_balls : Nat := 4
def total_balls : Nat := red_balls + black_balls

-- Define the probability calculation
def probability_red_ball : ℚ := red_balls / (red_balls + black_balls)

-- State the theorem
theorem probability_of_one_red_ball_is_one_third :
  probability_red_ball = 1 / 3 :=
by
  sorry

end probability_of_one_red_ball_is_one_third_l103_103490


namespace spherical_coordinates_of_neg_z_point_l103_103661

theorem spherical_coordinates_of_neg_z_point 
  (ρ θ φ : ℝ) 
  (h1 : ρ = 2) 
  (h2 : θ = 8 * Real.pi / 7) 
  (h3 : φ = 2 * Real.pi / 9) : 
  ∃ φ' : ℝ, 
    φ' = Real.pi - φ 
    ∧ 
    spherical_coordinates (ρ, θ, φ') (x, y, -z) = (2, 8 * Real.pi / 7, 7 * Real.pi / 9) 
    :=
begin
  sorry
end

end spherical_coordinates_of_neg_z_point_l103_103661


namespace letter_arrangements_proof_l103_103778

noncomputable def arrangements := 
  ∑ k in Finset.range 6, (Nat.choose 5 k) ^ 3

theorem letter_arrangements_proof :
  (∑ k in Finset.range 6, (Nat.choose 5 k) ^ 3) = arrangements := 
  by 
  sorry

end letter_arrangements_proof_l103_103778


namespace water_tank_rise_l103_103172

noncomputable def rise_in_water_level (length width_narrow width_wide avg_displacement num_men) : ℝ :=
  let total_volume_displaced : ℝ :=
    have first_term := 2
    have last_term := 2 + (num_men - 1) * 1
    (num_men / 2) * (first_term + last_term)
  let area_of_tank : ℝ :=
    let a := width_narrow
    let b := width_wide
    let h := length
    ((a + b) / 2) * h
  total_volume_displaced / area_of_tank

theorem water_tank_rise :
  rise_in_water_level 40 10 30 4 50 = 1.65625 :=
by
  -- Here would be the proof of the theorem.
  sorry

end water_tank_rise_l103_103172


namespace triangle_area_l103_103747

theorem triangle_area (F A K : ℝ × ℝ) (p : ℝ) (h1 : F = (2, 0)) 
  (h2 : ∃ x, A = (x^2/(2 * 4), x)) 
  (h3 : K = (0, 0)) 
  (h4 : dist A K = real.sqrt 2 * dist A F)
  (h5 : dist K F = 4) :
  ∃ (ΔAFK_area : ℝ), ΔAFK_area = 8 :=
by
  sorry

end triangle_area_l103_103747


namespace geometric_sequence_sixth_term_l103_103893

theorem geometric_sequence_sixth_term (a : ℝ) (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^(7) = 2) :
  a * r^(5) = 16 :=
by
  sorry

end geometric_sequence_sixth_term_l103_103893


namespace gcd_n3_plus_16_n_plus_4_l103_103375

/-- For a given positive integer n > 2^4, the greatest common divisor of n^3 + 16 and n + 4 is 1. -/
theorem gcd_n3_plus_16_n_plus_4 (n : ℕ) (h : n > 2^4) : Nat.gcd (n^3 + 16) (n + 4) = 1 := by
  sorry

end gcd_n3_plus_16_n_plus_4_l103_103375


namespace length_of_AD_l103_103217

theorem length_of_AD 
  (A B C D : Type) 
  (vertex_angle_equal: ∀ {a b c d : Type}, a = A →
    ∀ (AB AC AD : ℝ), (AB = 24) → (AC = 54) → (AD = 36)) 
  (right_triangles : ∀ {a b : Type}, a = A → ∀ {AB AC : ℝ}, (AB > 0) → (AC > 0) → (AB ^ 2 + AC ^ 2 = AD ^ 2)) :
  ∃ (AD : ℝ), AD = 36 :=
by
  sorry

end length_of_AD_l103_103217


namespace zeros_of_f_in_intervals_l103_103605

open Real

def f (x : ℝ) := (x + 1) * (x - 1) + (x - 1) * (x - 2) + (x - 2) * (x + 1)

theorem zeros_of_f_in_intervals :
  (∃ x ∈ Ioo (-1 : ℝ) (1 : ℝ), f x = 0) ∧ (∃ x ∈ Ioo (1 : ℝ) (2 : ℝ), f x = 0) :=
sorry

end zeros_of_f_in_intervals_l103_103605


namespace veromont_clicked_ads_l103_103944

def ads_on_first_page := 12
def ads_on_second_page := 2 * ads_on_first_page
def ads_on_third_page := ads_on_second_page + 24
def ads_on_fourth_page := (3 / 4) * ads_on_second_page
def total_ads := ads_on_first_page + ads_on_second_page + ads_on_third_page + ads_on_fourth_page
def ads_clicked := (2 / 3) * total_ads

theorem veromont_clicked_ads : ads_clicked = 68 := 
by
  sorry

end veromont_clicked_ads_l103_103944


namespace youth_cup_part1_youth_cup_part2_youth_cup_part3_l103_103928

noncomputable def part1 (P_A2 : ℚ) := P_A2 = 11/20

noncomputable def part2 (E_X : ℚ) := E_X = 249/100

noncomputable def part3 (P_A_given_fouls : ℚ) := P_A_given_fouls = 56/125

theorem youth_cup_part1 {P_B2 P_A2_not P_A2 : ℚ} (H1 : P_B2 = 3/4)
  (H2 : P_A2 = (3/4) * (3/5) + (1 - (3/4)) * (2/5)) :
  part1 P_A2 :=
by {
  unfold part1,
  rw [H2, H1],
  norm_num,
  exact eq.refl _
}

theorem youth_cup_part2 {E_X : ℚ} (P_X2 P_X3 : ℚ) (H1 : P_X2 = (3/5) * (11/20) + (2/5) * (9/20))
  (H2 : P_X3 = 1 - P_X2)
  (H3 : E_X = 2 * P_X2 + 3 * P_X3) :
  part2 E_X :=
by {
  unfold part2,
  rw [H3, H2, H1],
  norm_num,
  exact eq.refl _
}

theorem youth_cup_part3 {P_A_given_fouls : ℚ} (H1 : P_A_given_fouls = (3/5 * 2/5) + (3/5 * 3/5 * 2/5) + (2/5 * 2/5 * 2/5)) :
  part3 P_A_given_fouls :=
by {
  unfold part3,
  rw H1,
  norm_num,
  exact eq.refl _
}

end youth_cup_part1_youth_cup_part2_youth_cup_part3_l103_103928


namespace correct_operation_l103_103318

theorem correct_operation (a b : ℝ) : (a * b) - 2 * (a * b) = - (a * b) :=
sorry

end correct_operation_l103_103318


namespace distance_from_point_to_plane_l103_103048

noncomputable def dist_to_plane (A P : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
(abs (n.1 * (P.1 - A.1) + n.2 * (P.2 - A.2) + n.3 * (P.3 - A.3))) / 
(real.sqrt (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2))

theorem distance_from_point_to_plane :
  let A := (-2, 3, 0) in
  let P := (1, 1, 4) in
  let n := (2, 1, 2) in
  dist_to_plane A P n = 4 :=
by
  sorry

end distance_from_point_to_plane_l103_103048


namespace ratio_of_x_to_y_l103_103099

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 := 
by
  sorry

end ratio_of_x_to_y_l103_103099


namespace simplify_expression_l103_103578

variable (p : ℝ)

theorem simplify_expression (h₁ : p^3 - p^2 + 2p + 16 ≠ 0) (h₂ : p^2 + 2p + 6 ≠ 0) :
  (p^3 + 4 * p^2 + 10 * p + 12) / (p^3 - p^2 + 2p + 16) * (p^3 - 3 * p^2 + 8 * p) / (p^2 + 2 * p + 6) = p :=
by
  sorry

end simplify_expression_l103_103578


namespace simplify_fraction_l103_103579

theorem simplify_fraction :
  ( (5^2010)^2 - (5^2008)^2 ) / ( (5^2009)^2 - (5^2007)^2 ) = 25 := by
  sorry

end simplify_fraction_l103_103579


namespace alphametic_problem_find_S_l103_103821

-- Define the alphametic equation and necessary hypotheses
theorem alphametic_problem_find_S (W E Y S C N: ℕ) 
(h_diff : W ≠ E ∧ W ≠ Y ∧ W ≠ S ∧ W ≠ C ∧ W ≠ N ∧ 
         E ≠ Y ∧ E ≠ S ∧ E ≠ C ≠ N ∧ 
         Y ≠ S ∧ Y ≠ C ∧ Y ≠ N ∧ 
         S ≠ C ∧ S ≠ N ∧ 
         C ≠ N)
(h_valid_digits : ∀ d, d = W ∨ d = E ∨ d = Y ∨ d = S ∨ d = C ∨ d = N → d < 10)
(h_no_leading_zero : W ≠ 0 ∧ E ≠ 0 ∧ S ≠ 0) 
(h_same_W : W = (some value from problem 31)) -- assume some value from problem 31
(h_eq : (10 * W + E) * (100 * E + 10 * Y + E) = 10000 * S + 1000 * C + 100 * E + 10 * N + E):
S = 5 := 
sorry

end alphametic_problem_find_S_l103_103821


namespace find_p_l103_103845

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V) (p : ℝ)
variables (A B D : V)

noncomputable def AB : V := 2 • a + p • b
noncomputable def BC : V := a + b
noncomputable def CD : V := a - 2 • b
noncomputable def BD : V := BC + CD

theorem find_p (h_non_collinear : ¬Collinear ![a, b])
  (h_AB : A = B + AB)
  (h_collinear_ABD : Collinear ![A, B, D]) :
  p = -1 :=
sorry

end find_p_l103_103845


namespace inserted_digit_divisible_by_7_l103_103832

theorem inserted_digit_divisible_by_7 :
  ∃ x : ℕ, x < 10 ∧ (2006 + x * 100) % 7 = 0 := 
by {
  use [0, 7],
  split,
  linarith,
  split,
  exact dec_trivial,
  repeat {
    split; 
    norm_num;
    exact dec_trivial,
  }
}

end inserted_digit_divisible_by_7_l103_103832


namespace neg_rational_is_rational_l103_103794

theorem neg_rational_is_rational (m : ℚ) : -m ∈ ℚ :=
sorry

end neg_rational_is_rational_l103_103794


namespace axis_of_symmetry_y_range_l103_103698

/-- 
The equation of the curve is given by |x| + y^2 - 3y = 0.
We aim to prove two properties:
1. The axis of symmetry of this curve is x = 0.
2. The range of possible values for y is [0, 3].
-/
noncomputable def curve (x y : ℝ) : ℝ := |x| + y^2 - 3*y

theorem axis_of_symmetry : ∀ x y : ℝ, curve x y = 0 → x = 0 :=
sorry

theorem y_range : ∀ y : ℝ, ∃ x : ℝ, curve x y = 0 → (0 ≤ y ∧ y ≤ 3) :=
sorry

end axis_of_symmetry_y_range_l103_103698


namespace circle_chord_length_l103_103201

theorem circle_chord_length
  (r : ℝ)
  (h : ∃ (C : ℝ × ℝ), C = (-4, 5) ∧ (C.1 + 4)^2 + (C.2 - 5)^2 = r^2 )
  (tangent_to_x_axis : r = 5) :
  let y1 := 2
  let y2 := 8
  in abs(y1 - y2) = 6 := by
  sorry

end circle_chord_length_l103_103201


namespace num_grandparents_l103_103128

theorem num_grandparents (h1 : (n : ℕ) → John gets $50 from n grandparents = 100) : n = 2 :=
by
  sorry

end num_grandparents_l103_103128


namespace area_of_circumcircle_of_right_triangle_l103_103654

theorem area_of_circumcircle_of_right_triangle (A B C : Type*) [MetricSpace A]
  [MetricSpace B] [MetricSpace C] (r3 r4 : Real) (π : Real) :
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A) ∧
  (dist A B = 3) ∧ (dist B C = 4) ∧
  (dist A C = Real.sqrt (3^2 + 4^2)) →
  (π > 0) →
  (Real.pi * ((Real.sqrt (3^2 + 4^2)) / 2)^2 = (25 / 4) * π)

sorry

end area_of_circumcircle_of_right_triangle_l103_103654


namespace one_twentieth_of_eighty_l103_103093

/--
Given the conditions, to prove that \(\frac{1}{20}\) of 80 is equal to 4.
-/
theorem one_twentieth_of_eighty : (80 : ℚ) * (1 / 20) = 4 :=
by
  sorry

end one_twentieth_of_eighty_l103_103093


namespace positive_triple_l103_103570

theorem positive_triple
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : ab + bc + ca > 0)
  (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_triple_l103_103570


namespace exists_q_lt_1_l103_103147

variable {a : ℕ → ℝ}

theorem exists_q_lt_1 (h_nonneg : ∀ n, 0 ≤ a n)
  (h_rec : ∀ k m, a (k + m) ≤ a (k + m + 1) + a k * a m)
  (h_large_n : ∃ n₀, ∀ n ≥ n₀, n * a n < 0.2499) :
  ∃ q, 0 < q ∧ q < 1 ∧ (∃ n₀, ∀ n ≥ n₀, a n < q ^ n) :=
by
  sorry

end exists_q_lt_1_l103_103147


namespace dishonest_dealer_profit_l103_103636

theorem dishonest_dealer_profit :
  ∀ (cost_price_per_kg : ℝ) (impurity_percentage : ℝ) (weight_loss_percentage : ℝ),
  cost_price_per_kg = 100 →
  impurity_percentage = 0.35 →
  weight_loss_percentage = 0.20 →
  let counterfeit_weight := (1 - weight_loss_percentage) in
  let actual_product_percentage := (1 - impurity_percentage) in
  let actual_product_weight := (counterfeit_weight * actual_product_percentage) in
  let actual_cost := (cost_price_per_kg * actual_product_weight) in
  let selling_price := cost_price_per_kg in
  let profit := (selling_price - actual_cost) in
  let profit_percentage := (profit / actual_cost) * 100 in
  profit_percentage ≈ 92.31 := 
by
  intros,
  sorry

end dishonest_dealer_profit_l103_103636


namespace lcm_consecutive_impossible_l103_103030

theorem lcm_consecutive_impossible (n : ℕ) (a : fin n → ℕ)
  (h : n = 10 ^ 1000)
  (b : fin n → ℕ)
  (hcirc : ∀ i, b i = nat.lcm (a i) (a ((i + 1) % n))) :
  ¬ (∃ f : fin n → ℕ, bijective f ∧ ∀ i, b i = f i ∧ f (i + 1) % n = f i + 1) :=
sorry

end lcm_consecutive_impossible_l103_103030


namespace probability_of_consecutive_blocks_drawn_l103_103648

theorem probability_of_consecutive_blocks_drawn :
  let total_ways := (Nat.factorial 12)
  let favorable_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5) * (Nat.factorial 3)
  (favorable_ways / total_ways) = 1 / 4620 :=
by
  sorry

end probability_of_consecutive_blocks_drawn_l103_103648


namespace transport_tax_to_be_paid_l103_103514

noncomputable def engine_power : ℕ := 150
noncomputable def tax_rate : ℕ := 20
noncomputable def annual_tax : ℕ := engine_power * tax_rate
noncomputable def months_used : ℕ := 8
noncomputable def prorated_tax : ℕ := (months_used * annual_tax) / 12

theorem transport_tax_to_be_paid : prorated_tax = 2000 := 
by 
  -- sorry is used to skip the proof step
  sorry

end transport_tax_to_be_paid_l103_103514


namespace probability_P_plus_S_mod_7_correct_l103_103936

noncomputable def probability_P_plus_S_mod_7 : ℚ :=
  let n := 60
  let total_ways := (n * (n - 1)) / 2
  let num_special_pairs := total_ways - ((52 * 51) / 2)
  num_special_pairs / total_ways

theorem probability_P_plus_S_mod_7_correct :
  probability_P_plus_S_mod_7 = 148 / 590 :=
by
  rw [probability_P_plus_S_mod_7]
  sorry

end probability_P_plus_S_mod_7_correct_l103_103936


namespace polygon_has_7_sides_l103_103529

open Set

variable (b : ℝ) (x y : ℝ)

-- Conditions defining the set T
def T : Set (ℝ × ℝ) :=
  {p | let (x, y) := p in b ≤ x ∧ x ≤ 3b ∧ b ≤ y ∧ y ≤ 3b ∧ x + y ≥ 2b ∧ x + 2b ≥ 2y ∧ y + 2b ≥ 2x}

-- Theorem stating the polygon defined by the boundary of T has 7 sides
theorem polygon_has_7_sides (b_pos : 0 < b) : 
  ∃ vertices : Finset (ℝ × ℝ), Finset.card vertices = 7 ∧ ∀ p ∈ vertices, p ∈ T := 
sorry

end polygon_has_7_sides_l103_103529


namespace angle_perpendicular_l103_103056

-- Define the vectors a and b
variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Define the magnitude condition
axiom magnitude_condition : 2 * ‖a‖ = ‖b‖

-- State the theorem
theorem angle_perpendicular (a b : EuclideanSpace ℝ (Fin 3)) (h : 2 * ‖a‖ = ‖b‖) :
  let v1 := 2 • a + b,
      v2 := 2 • a - b in
  inner v1 v2 = 0 :=
by {
  sorry
}

end angle_perpendicular_l103_103056


namespace area_ratio_of_octagon_l103_103876

theorem area_ratio_of_octagon (A : ℝ) (hA : 0 < A) :
  let triangle_ABJ_area := A / 8
  let triangle_ACE_area := A / 2
  triangle_ABJ_area / triangle_ACE_area = 1 / 4 := by
  sorry

end area_ratio_of_octagon_l103_103876


namespace smallest_prime_factor_3063_l103_103249

theorem smallest_prime_factor_3063 (h1 : ¬ (3063 % 2 = 0)) (h2 : ((3 + 0 + 6 + 3) = 12)) 
(h3 : 12 % 3 = 0) : ∃ p : ℕ, prime p ∧ p ∣ 3063 ∧ ∀ q : ℕ, prime q ∧ q ∣ 3063 → p ≤ q :=
by
  sorry

end smallest_prime_factor_3063_l103_103249


namespace pamphlet_cost_l103_103864

theorem pamphlet_cost (p : ℝ) 
  (h1 : 9 * p < 10)
  (h2 : 10 * p > 11) : p = 1.11 :=
sorry

end pamphlet_cost_l103_103864


namespace monotonically_increasing_min_value_m_distinct_solutions_k_l103_103065

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x + (2 * k + 1) / x

theorem monotonically_increasing (k : ℝ) (h_k : k ≥ 0) : ∀ x, x ∈ set.Ici (real.sqrt (2 * k + 1)) → has_deriv_at (λ x, f x k) (1 - (2 * k + 1) / (x * x)) x :=
begin
  sorry
end

theorem min_value_m (h_k : 1 ≤ k ∧ k ≤ 7) (m : ℝ) : (∀ x ∈ set.Icc (2 : ℝ) (3 : ℝ), f x k ≥ m) → m ≤ 7 / 2 :=
begin
  sorry
end

theorem distinct_solutions_k : (∃ (f : ℝ → ℝ → ℝ) (k : ℝ), (f (| 2 ^ x - 1 | k) = 3 * k + 2) → (k > 0)) :=
begin
  sorry
end

end monotonically_increasing_min_value_m_distinct_solutions_k_l103_103065


namespace probability_P_plus_S_is_one_less_than_multiple_of_seven_l103_103939

theorem probability_P_plus_S_is_one_less_than_multiple_of_seven :
  ∀ (a b : ℕ), a ∈ finset.range(1, 61) → b ∈ finset.range(1, 61) → a ≠ b →
  let S := a + b in
  let P := a * b in
  (nat.gcd ((P + S + 1), 7) = 1) →
  (finset.filter (λ (a b : ℕ), (a+1) ∣ 7 ∨ (b+1) ∣ 7) (finset.range(1, 61)).product (finset.range(1, 61)).card) / 1770 = 74 / 295 :=
begin
  sorry
end

end probability_P_plus_S_is_one_less_than_multiple_of_seven_l103_103939


namespace triangle_enlargement_invariant_l103_103566

theorem triangle_enlargement_invariant (α β γ : ℝ) (h_sum : α + β + γ = 180) (f : ℝ) :
  (α * f ≠ α) ∧ (β * f ≠ β) ∧ (γ * f ≠ γ) → (α * f + β * f + γ * f = 180 * f) → α + β + γ = 180 :=
by
  sorry

end triangle_enlargement_invariant_l103_103566


namespace sphere_volume_ratio_l103_103436

theorem sphere_volume_ratio (r1 r2 A1 A2 V1 V2 : ℝ) (h1 : A1 = 4 * real.pi * r1^2)
  (h2 : A2 = 4 * real.pi * r2^2) (h3 : V1 = (4 / 3) * real.pi * r1^3) 
  (h4 : V2 = (4 / 3) * real.pi * r2^3) (h5 : A1 / A2 = 1 / 3) : 
  V1 / V2 = 1 / (3 * real.sqrt 3) :=
by
  sorry

end sphere_volume_ratio_l103_103436


namespace problem_statement_l103_103771

open Real

namespace MathProblem

def p₁ := ∃ x : ℝ, x^2 + x + 1 < 0
def p₂ := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_statement : (¬p₁) ∨ (¬p₂) :=
by
  sorry

end MathProblem

end problem_statement_l103_103771


namespace kolya_nuts_division_l103_103641

theorem kolya_nuts_division (n : ℕ) (hn : 2 ≤ n):
  let nuts := 2 * n + 1 in
  let method1 := -- Formalize method 1 picking strategy
    λ (x y : ℕ), x + y = nuts → (x ≥ y → x) + (y ≥ x → y) >= n + 1
  let method2 := -- Formalize method 2 picking strategy
    λ (u v : ℕ), u + v = nuts → (u // 2 + v // 2) >= n
  let method3 := -- Formalize method 3 picking strategy
    λ (a b : ℕ), a + b = nuts - 1 → max ((a // 2) + (b // 2), (a // 2) + (b // 2)) >= n
  in multiply.most_advantageous method1 (by sorry) ∧ multiply.least_advantageous method2 method3 (by sorry).

end kolya_nuts_division_l103_103641


namespace football_team_initial_loss_l103_103983

variables (x : ℝ)

theorem football_team_initial_loss :
  ∃ x, (-x + 11 = 6) ∧ x = 5 :=
begin
  use 5,
  split,
  { -- show that x = 5 satisfies -x + 11 = 6
    show -5 + 11 = 6,
    linarith },
  { -- show that x = 5
    refl }
end

end football_team_initial_loss_l103_103983


namespace dot_product_of_projection_eq_half_l103_103079

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem dot_product_of_projection_eq_half (hb : ∥b∥ = 4) (hproj : ⟪a, b⟫ / ∥b∥ = 1 / 2) : ⟪a, b⟫ = 2 :=
by
  sorry

end dot_product_of_projection_eq_half_l103_103079


namespace sum_intersection_points_eq_m_l103_103034

theorem sum_intersection_points_eq_m
  (f : ℝ → ℝ)
  (h_sym : ∀ x : ℝ, f(-x) = 2 - f(x))
  (intersect_pts : (ℝ × ℝ) → ℕ)
  (x_i : ℕ → ℝ)
  (y_i : ℕ → ℝ)
  (h_intersect : ∀ i, intersect_pts (x_i i, y_i i) = i 
    ∧ y_i i = (x_i i + 1) / x_i i 
    ∧ y_i i = f (x_i i)) :
  (∑ i in finset.range (m+1), (x_i i + y_i i)) = m :=
by
  sorry

end sum_intersection_points_eq_m_l103_103034


namespace minimum_students_needed_l103_103191

theorem minimum_students_needed :
  ∃ (students : ℕ), 
    (∀ i : Fin 6, ∃ (s : Finset ℕ), s.card = 500 ∧ ∀ a b ∈ s, ∃ j : Fin 6, j ≠ i → (a ≠ b → ¬(j = i)))
    ∧ (∀ (a b : ℕ), a ≠ b → ∃ i : Fin 6, ¬(i = a) ∨ ¬(i = b))
    ∧ students = 1000 := 
sorry

end minimum_students_needed_l103_103191


namespace range_of_a_l103_103167

-- Define the function f(x)
def f (x : ℝ) : ℝ := min (x^2 - 1) (min (x + 1) (-x + 1))

-- State the theorem
theorem range_of_a (a : ℝ) : (f (a + 2) > f a) ↔ (a < -2 ∨ (-1 < a ∧ a < 0)) :=
by
  sorry

end range_of_a_l103_103167


namespace perpendicular_theta_magnitude_tan_theta_value_l103_103772

variables (θ : ℝ) (a b : ℝ × ℝ)

def vector_a (θ : ℝ) : ℝ × ℝ := (2 * Real.sin θ, 1)
def vector_b (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, -1)
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Question 1
theorem perpendicular_theta_magnitude (hθ : θ ∈ Set.Ioo 0 (Real.pi / 2)) :
  perpendicular (vector_a θ) (vector_b θ) ↔ (θ = Real.pi / 12 ∨ θ = 5 * Real.pi / 12) := by sorry

-- Question 2
theorem tan_theta_value (hθ : θ ∈ Set.Ioo 0 (Real.pi / 2)) :
  magnitude (vector_a θ - vector_b θ) = 2 * magnitude (vector_b θ) ↔ Real.tan θ = 3 := by sorry

end perpendicular_theta_magnitude_tan_theta_value_l103_103772


namespace part_a_part_b_l103_103281

-- Part (a) statement
theorem part_a (a b n : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < n) (h_eq : a * b = n^2 + 1) : abs (a - b) ≥ Real.sqrt (4 * n - 3) := sorry

-- Part (b) statement
theorem part_b : ∃ᶠ a b n in at_top, 
  0 < a ∧ 0 < b ∧ 0 < n ∧ a * b = n^2 + 1 ∧ abs (a - b) = Real.sqrt (4 * n - 3) := sorry

end part_a_part_b_l103_103281


namespace cab_driver_income_fifth_day_l103_103651

theorem cab_driver_income_fifth_day :
  ∀ (income1 income2 income3 income4 avg : ℤ), 
  income1 = 300 ∧ income2 = 150 ∧ income3 = 750 ∧ income4 = 200 ∧ avg = 400 →
  let total := avg * 5 in
  let first_four_days := income1 + income2 + income3 + income4 in
  let income5 := total - first_four_days in
  income5 = 600
:=
by
  intros income1 income2 income3 income4 avg h,
  cases h with h1 h_rest1,
  cases h_rest1 with h2 h_rest2,
  cases h_rest2 with h3 h_rest3,
  cases h_rest3 with h4 h5,
  rw [h1, h2, h3, h4] at *,
  unfold avg at h5,
  rw [h5] at *,
  let total := 400 * 5,
  let first_four_days := 300 + 150 + 750 + 200,
  let income5 := total - first_four_days,
  sorry

end cab_driver_income_fifth_day_l103_103651


namespace sum_of_ages_l103_103877

theorem sum_of_ages (rose_age mother_age : ℕ) (rose_age_eq : rose_age = 25) (mother_age_eq : mother_age = 75) : 
  rose_age + mother_age = 100 := 
by
  sorry

end sum_of_ages_l103_103877


namespace constants_k1_k2_k3_values_l103_103362

def P1 (k1 k2 k3 : ℝ) := -x^4 - (k1 + 11) * x^3 - k2 * x^2 - 8 * x - k3
def P2 := -(x - 2) * (x^3 - 6 * x^2 + 8 * x - 4)

theorem constants_k1_k2_k3_values :
  ∃ (k1 k2 k3 : ℝ), 
    P1 k1 k2 k3 = P2 ∧ 
    k1 = -19 ∧ 
    k2 = 20 ∧ 
    k3 = 8 := 
sorry

end constants_k1_k2_k3_values_l103_103362


namespace max_kings_on_chessboard_l103_103242

theorem max_kings_on_chessboard : 
  ∀ (board : matrix (fin 12) (fin 12) bool), 
  (∀ i j, board i j = tt → (∃ k l, (abs ((i : int) - k) ≤ 1 ∧ abs ((j : int) - l) ≤ 1) 
  ∧ board k l = tt ∧ (i ≠ k ∨ j ≠ l))) → 
  ∃ (S : finset (fin 12 × fin 12)), S.card = 56 ∧ 
  (∀ (i j) (h1 : (i, j) ∈ S) (h2 : (i', j') ∈ S), (abs ((i : int) - (i' : int)) ≤ 1 ∧ abs ((j : int) - (j' : int)) ≤ 1) 
  → ((i, j) ≠ (i', j')) → (abs ((i : int) - (i' : int)) = 1 ∧ abs ((j : int) - (j' : int)) = 0) 
  ∨ (abs ((i : int) - (i' : int)) = 0 ∧ abs ((j : int) - (j' : int)) = 1))) :=
by
  sorry

end max_kings_on_chessboard_l103_103242


namespace calculate_percentage_l103_103970

theorem calculate_percentage (num : ℕ) (h : num = 4800) : 
  (0.15 * (0.30 * (0.50 * num))) = 108 :=
by
  rw [h]
  sorry

end calculate_percentage_l103_103970


namespace total_distance_of_trail_l103_103572

theorem total_distance_of_trail (a b c d e : ℕ) 
    (h1 : a + b + c = 30) 
    (h2 : b + d = 30) 
    (h3 : d + e = 28) 
    (h4 : a + d = 34) : 
    a + b + c + d + e = 58 := 
sorry

end total_distance_of_trail_l103_103572


namespace geometric_progression_ineq_l103_103038

variable (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ)

-- Condition: \(b_n\) is an increasing positive geometric progression
-- \( q > 1 \) because the progression is increasing
variable (q_pos : q > 1) 

-- Recursive definitions for the geometric progression
variable (geom_b₂ : b₂ = b₁ * q)
variable (geom_b₃ : b₃ = b₁ * q^2)
variable (geom_b₄ : b₄ = b₁ * q^3)
variable (geom_b₅ : b₅ = b₁ * q^4)
variable (geom_b₆ : b₆ = b₁ * q^5)

-- Given condition from the problem
variable (condition : b₄ + b₃ - b₂ - b₁ = 5)

-- Statement to prove
theorem geometric_progression_ineq (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) 
  (q_pos : q > 1) 
  (geom_b₂ : b₂ = b₁ * q)
  (geom_b₃ : b₃ = b₁ * q^2)
  (geom_b₄ : b₄ = b₁ * q^3)
  (geom_b₅ : b₅ = b₁ * q^4)
  (geom_b₆ : b₆ = b₁ * q^5)
  (condition : b₃ + b₄ - b₂ - b₁ = 5) : b₆ + b₅ ≥ 20 := by
    sorry

end geometric_progression_ineq_l103_103038


namespace factor_expression_l103_103060

theorem factor_expression (a b c : ℝ) :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + ab + bc + ca) :=
by
  sorry

end factor_expression_l103_103060


namespace tangent_circle_line_zero_l103_103463

theorem tangent_circle_line_zero (m : ℝ) : (∀ (x y : ℝ), x^2 + y^2 = 4 * m) →
                                           (∀ (x y : ℝ), x - y = 2 * real.sqrt m) →
                                           m = 0 :=
by {
  sorry
}

end tangent_circle_line_zero_l103_103463


namespace rectangular_solid_volume_l103_103303

theorem rectangular_solid_volume (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : a * b * c = 60 :=
by
  sorry

end rectangular_solid_volume_l103_103303


namespace abel_inequality_l103_103140

theorem abel_inequality (ξ ζ : ℝ → ℂ) (hξ1 : ∀ ω, 0 ≤ ξ ω ∧ ξ ω ≤ 1)
  (hξ2 : measurable ξ) (hζ : integrable ζ) :
  |∫ ω, (ξ ω) * (ζ ω) ∂measure_space.volume| ≤ 
  sup (set.Icc 0 1) (λ x, |∫ ω, (ζ ω) * indicator (λ ω', ξ ω' ≥ x) 1 ω ∂measure_space.volume|) := 
sorry

end abel_inequality_l103_103140


namespace first_day_miles_proof_l103_103668

-- Definitions based on conditions
variable (total_road_length : ℕ := 16)
variable (truckloads_per_mile : ℕ := 3)
variable (pitch_to_truckload_ratio : ℕ := 1)
variable (first_day_miles : ℕ)
variable (second_day_miles : ℕ)
variable (pitch_needed_third_day : ℕ := 6)

def paved_miles (x : ℕ) : ℕ :=
  (x + (2 * x - 1))

def remaining_road_length (x : ℕ) : ℕ :=
  total_road_length - (paved_miles x)

def truckloads_needed (pitch_barrels : ℕ) : ℕ :=
  pitch_barrels

def third_day_miles (pitch_barrels : ℕ) : ℕ :=
  truckloads_needed pitch_barrels / truckloads_per_mile

-- Main statement to prove
theorem first_day_miles_proof : 
  ∀ x : ℕ, 
  first_day_miles = x → 
  second_day_miles = 2 * x - 1 → 
  paved_miles x = 3 * x - 1 → 
  remaining_road_length x = 2 → 
  third_day_miles pitch_needed_third_day = 2 → 
  paved_miles x = total_road_length - third_day_miles pitch_needed_third_day → 
  first_day_miles = 5 :=
by {
  intros,
  sorry
}

end first_day_miles_proof_l103_103668


namespace person_A_arrives_before_B_l103_103940

variable {a b S : ℝ}

theorem person_A_arrives_before_B (h : a ≠ b) (a_pos : 0 < a) (b_pos : 0 < b) (S_pos : 0 < S) :
  (2 * S / (a + b)) < ((a + b) * S / (2 * a * b)) :=
by
  sorry

end person_A_arrives_before_B_l103_103940


namespace distance_between_5th_and_23rd_red_light_l103_103580

theorem distance_between_5th_and_23rd_red_light :
  let inch_to_feet (inches : ℕ) : ℝ := inches / 12.0
  let distance_in_inches := 40 * 8
  inch_to_feet distance_in_inches = 26.67 :=
by
  sorry

end distance_between_5th_and_23rd_red_light_l103_103580


namespace length_of_AB_l103_103176

theorem length_of_AB (x1 y1 x2 y2 : ℝ) 
  (h1 : (1:ℝ) = (x1 + x2) / 2)
  (h2 : (1:ℝ) = (y1 + y2) / 2)
  (h3 : (x1^2 / 4) + (y1^2 / 2) = 1)
  (h4 : (x2^2 / 4) + (y2^2 / 2) = 1) : 
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt (30) / 3 := 
sorry

end length_of_AB_l103_103176


namespace proof_problem_l103_103052

open Real

-- Conditions
variables (a : ℝ) (f : ℝ → ℝ) (x : ℝ)
-- Given conditions
def conditions (h : a > 0 ∧ a ≠ 1 ∧ log a 4 = 2 ∧ f = λ x, - log 2 x ∧ f(x-1) > f(5 - x)) : Prop :=
  a = 2 ∧ (1 < x ∧ x < 3)

-- Main statement
theorem proof_problem :
    ∃ a f, (∃ h, conditions a f h) :=
sorry

end proof_problem_l103_103052


namespace trains_crossing_time_l103_103941

noncomputable def length_first_train : ℝ := 140
noncomputable def length_second_train : ℝ := 160
noncomputable def speed_first_train_kmph : ℝ := 60
noncomputable def speed_second_train_kmph : ℝ := 48
noncomputable def relative_speed_kmph := speed_first_train_kmph + speed_second_train_kmph
noncomputable def relative_speed_mps := relative_speed_kmph * (1000 / 3600)
noncomputable def total_distance := length_first_train + length_second_train

theorem trains_crossing_time :
  let time_to_cross := total_distance / relative_speed_mps 
  in time_to_cross = 10 :=
by
  sorry

end trains_crossing_time_l103_103941


namespace catch_up_time_l103_103554

open Real

/-- Object A moves along a straight line with a velocity v_A(t) = 3t^2 + 1 (m/s),
object B is 5 meters ahead of A and moves with velocity v_B(t) = 10t (m/s).
Prove that the time (t in seconds) it takes for object A to catch up with object B
is t = 5.
-/
theorem catch_up_time :
  let v_A := fun t : ℝ => 3 * t^2 + 1,
      v_B := fun t : ℝ => 10 * t,
      dist_A := fun t : ℝ => (∫ s in 0..t, v_A s),
      dist_B := fun t : ℝ => (∫ s in 0..t, v_B s) + 5 in
  ∃ t : ℝ, dist_A t = dist_B t ∧ t = 5 :=
by 
  -- The proof will involve finding that when the distances are equal, then t = 5
  sorry

end catch_up_time_l103_103554


namespace students_suggested_tomatoes_79_l103_103188

theorem students_suggested_tomatoes_79 (T : ℕ)
  (mashed_potatoes : ℕ)
  (h1 : mashed_potatoes = 144)
  (h2 : mashed_potatoes = T + 65) :
  T = 79 :=
by {
  -- Proof steps will go here
  sorry
}

end students_suggested_tomatoes_79_l103_103188


namespace cyclic_quadrilateral_equal_area_l103_103160

variable {A B C D E F G H W X Y Z : Point}

-- Conditions
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry
def is_midpoint (X Y Z W : Point) : Prop := sorry
def is_orthocenter (W X Y Z : Point) : Prop := sorry

-- The theorem statement
theorem cyclic_quadrilateral_equal_area (h_cyclic : is_cyclic_quadrilateral A B C D) 
    (h_midpoints: is_midpoint E F G H) 
    (h_orthocenters : is_orthocenter W X Y Z) : 
    area A B C D = area W X Y Z := 
sorry

end cyclic_quadrilateral_equal_area_l103_103160


namespace find_cos_alpha_find_sin_beta_l103_103386

noncomputable def cos_alpha (α : ℝ) (hα : α ∈ Ioo (π / 2) π) 
  (h_sin_cos : sin (α / 2) + cos (α / 2) = (2 * real.sqrt 3) / 3) : ℝ :=
-sqrt (1 - (1 / 3)^2)

noncomputable def sin_beta (α β : ℝ) (h_alpha : α ∈ Ioo (π / 2) π) 
  (h_beta : β ∈ Ioo 0 (π / 2))
  (h_sin_sum : sin (α + β) = -3 / 5)
  (h_cos_alpha : cos α = -((2 * sqrt 2) / 3)) 
  (h_sin_alpha : sin α = 1 / 3) : ℝ :=
((6 * sqrt 2) + 4) / 15

theorem find_cos_alpha ( 
  α : ℝ 
  (hα : α ∈ Ioo (π / 2) π) 
  (h_sin_cos : sin (α / 2) + cos (α / 2) = (2 * sqrt 3) / 3)
): cos_alpha α hα h_sin_cos = -(2 * sqrt 2) / 3 := by
  sorry

theorem find_sin_beta ( 
  α β : ℝ 
  (hα: α ∈ Ioo (π / 2) π)
  (hβ: β ∈ Ioo 0 (π / 2))
  (h_sin_sum : sin (α + β) = -3 / 5)
  (h_cos_alpha : cos α = -(2 * sqrt 2) / 3) 
  (h_sin_alpha : sin α = 1 / 3)
): sin_beta α β hα hβ h_sin_sum h_cos_alpha h_sin_alpha = (6 * sqrt 2 + 4) / 15 := by
  sorry

end find_cos_alpha_find_sin_beta_l103_103386


namespace power_division_l103_103688

theorem power_division (a : ℝ) (h : a ≠ 0) : ((-a)^6) / (a^3) = a^3 := by
  sorry

end power_division_l103_103688


namespace minimum_distance_tangent_l103_103735

-- Define the circle and the line conditions
def circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def line (x y : ℝ) : Prop := y = x + 2

-- Define the distance function
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Lean statement asserting the minimum distance
theorem minimum_distance_tangent :
  ∃ Mx My Nx Ny,
    line Mx My ∧ circle Nx Ny ∧
    ∀ (x y : ℝ), line x y → 
      distance Mx My Nx Ny = distance x y Nx Ny → 
        distance Mx My Nx Ny = sqrt (7/2) :=
sorry

end minimum_distance_tangent_l103_103735


namespace find_f_neg_t_l103_103848

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x + 1

theorem find_f_neg_t (t : ℝ) (h : f t = 2) : f (-t) = 0 :=
by
  sorry

end find_f_neg_t_l103_103848


namespace min_value_f_proof_l103_103801

noncomputable def min_value_f (a : ℝ) : ℝ :=
if a < 0 then 0
else if 0 ≤ a ∧ a ≤ 1 then -a^2
else 1 - 2 * a

theorem min_value_f_proof (a : ℝ) : 
    (∃ x ∈ (set.Icc (0:ℝ) (1:ℝ)), f(x) = min_value_f(a)) where
  f(x : ℝ) : ℝ := x^2 - 2*a*x :=
sorry

end min_value_f_proof_l103_103801


namespace area_of_rectangle_at_stage_5_l103_103799

def initial_square_area : ℕ := 16

def area_at_stage (n : ℕ) : ℕ :=
  if n = 1 then initial_square_area
  else 2 * area_at_stage (n - 1)

theorem area_of_rectangle_at_stage_5 :
  (area_at_stage 1 + area_at_stage 2 + area_at_stage 3 + area_at_stage 4 + area_at_stage 5) = 496 := by
  sorry

end area_of_rectangle_at_stage_5_l103_103799


namespace outstanding_student_awards_l103_103222

theorem outstanding_student_awards :
  ∃ n : ℕ, 
  (n = Nat.choose 9 7) ∧ 
  (∀ (awards : ℕ) (classes : ℕ), awards = 10 → classes = 8 → n = 36) := 
by
  sorry

end outstanding_student_awards_l103_103222


namespace systematic_sampling_students_l103_103552

def interval := 10
def num_students := 50
def num_selected := 5
def starting_student := 3
def selected_students := {3, 13, 23, 33, 43}

theorem systematic_sampling_students :
  ∃ (interval : ℕ) (starting_student : ℕ) (num_students num_selected : ℕ), 
  interval = num_students / num_selected ∧ 
  interval * 0 + starting_student ∈ selected_students ∧ 
  interval * 1 + starting_student ∈ selected_students ∧ 
  interval * 2 + starting_student ∈ selected_students ∧ 
  interval * 3 + starting_student ∈ selected_students ∧ 
  interval * 4 + starting_student ∈ selected_students :=
by
  use 10, 3, 50, 5
  split
  case h => exact rfl
  all_goals simp [selected_students]
  iterate 5
  { split
    case h => sorry }
  

end systematic_sampling_students_l103_103552


namespace proposition_1_proposition_2_proposition_3_correct_propositions_l103_103999

theorem proposition_1 (A B : ℝ) (a b : ℝ) (h1 : sin A > sin B) : a > b → A > B := sorry

theorem proposition_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  sqrt (a + 3) + sqrt (b + 2) ≤ 3 * sqrt 2 := sorry

theorem proposition_3 (f : ℕ → ℝ) (h : ∀ n, f (n + 1) - f n = f 1 - f 0) :
  ∀ n m, f (n + m) = f n + f m - f 0 := sorry

theorem correct_propositions :
  ∀ (A B : ℝ) (a b : ℝ) (f : ℕ → ℝ),
  (sin A > sin B → a > b → A > B) →
  (a > 0 → b > 0 → a + b = 4 → sqrt (a + 3) + sqrt (b + 2) ≤ 3 * sqrt 2) →
  (∀ n, f (n + 1) - f n = f 1 - f 0 → ∀ n m, f (n + m) = f n + f m - f 0) →
  ∀ q, q ≠ 1 → false := sorry

end proposition_1_proposition_2_proposition_3_correct_propositions_l103_103999


namespace train_passes_platform_in_correct_time_l103_103309

-- Define the conditions
def speed_km_per_hr := 54 -- Speed in km/hr
def speed_m_per_s := speed_km_per_hr * 1000 / 3600 -- Speed in m/s
def time_to_pass_person_s := 20 -- Time in seconds
def platform_length_m := 360.0288 -- Platform length in meters

-- Calculate length of the train
def train_length_m := time_to_pass_person_s * speed_m_per_s

-- Calculate total distance to pass the platform
def total_distance_m := train_length_m + platform_length_m

-- Calculate expected time to pass the platform
def expected_time_s := total_distance_m / speed_m_per_s

-- The proof problem statement
theorem train_passes_platform_in_correct_time : expected_time_s = 44.00192 := by
  sorry

end train_passes_platform_in_correct_time_l103_103309


namespace lcm_consecutive_impossible_l103_103027

noncomputable def n : ℕ := 10^1000

def circle_seq := fin n → ℕ

def lcm {a b : ℕ} : ℕ := Nat.lcm a b

def lcm_seq (a : circle_seq) : fin n → ℕ :=
λ i, lcm (a i) (a (i + 1) % n)

theorem lcm_consecutive_impossible (a : circle_seq) :
  ¬ ∃ (b : fin n → ℕ), (∀ i : fin n, b i = lcm (a i) (a (i + 1) % n)) ∧ (finset.range n).pairwise (λ i j, b i + 1 = b j) :=
sorry

end lcm_consecutive_impossible_l103_103027


namespace evaluate_complex_fraction_l103_103711

theorem evaluate_complex_fraction :
  (⌈20 / 9 - ⌈35 / 21⌉⌉ / ⌈35 / 9 + ⌈(9 * 20) / 35⌉⌉) = 1 / 10 :=
by
  have h1 : ⌈35 / 21⌉ = 2 := by sorry
  have h2 : ⌈(9 * 20) / 35⌉ = 6 := by sorry
  have numerator : ⌈20 / 9 - ⌈35 / 21⌉⌉ = 1 := by sorry
  have denominator : ⌈35 / 9 + ⌈(9 * 20) / 35⌉⌉ = 10 := by sorry
  sorry

end evaluate_complex_fraction_l103_103711


namespace yacht_capacity_l103_103224

theorem yacht_capacity :
  ∀ (x y : ℕ), (3 * x + 2 * y = 68) → (2 * x + 3 * y = 57) → (3 * x + 6 * y = 96) :=
by
  intros x y h1 h2
  sorry

end yacht_capacity_l103_103224


namespace _l103_103840

noncomputable theorem compute_b1c1_plus_b2c2_plus_b3c3 :
  ∀ (b1 b2 b3 c1 c2 c3 : ℝ),
    (∀ x : ℝ, x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
      (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3)) →
    b1 * c1 + b2 * c2 + b3 * c3 = 1 :=
begin
  sorry
end

end _l103_103840


namespace smallest_angle_l103_103719

theorem smallest_angle (θ : ℝ) (hθ : cos θ = sin (45 * real.pi / 180) + cos (30 * real.pi / 180) - sin (18 * real.pi / 180) - cos (10 * real.pi / 180)) : θ = 67 * real.pi / 180 :=
sorry

end smallest_angle_l103_103719


namespace degree_of_polynomial_sum_l103_103150

-- Define polynomials f and g with given conditions
variables {R : Type*} [CommRing R] 
variables (c0 c1 c2 c3 d0 d1 d2 : R)
variables (z : R)

-- Hypotheses based on the given conditions
theorem degree_of_polynomial_sum (hc3 : c3 ≠ 0) (hd2 : d2 ≠ 0) (h_sum : c3 + d2 ≠ 0) :
  degree ((c3 * z^3 + c2 * z^2 + c1 * z + c0) + (d2 * z^2 + d1 * z + d0)) = 3 := by
  sorry

end degree_of_polynomial_sum_l103_103150


namespace polynomial_condition_form_l103_103712

theorem polynomial_condition_form (P : Polynomial ℝ) :
  (∀ a b c : ℝ, ab + bc + ca = 0 → P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)) →
  ∃ α β : ℝ, P = α * Polynomial.X ^ 4 + β * Polynomial.X ^ 2 :=
sorry

end polynomial_condition_form_l103_103712


namespace sin_double_angle_l103_103043

theorem sin_double_angle (α : ℝ) (h1 : α ∈ set.Ioo (Real.pi / 2) Real.pi) (h2 : Real.sin α = 12 / 13) : 
  Real.sin (2 * α) = - 120 / 169 :=
sorry

end sin_double_angle_l103_103043


namespace max_value_ratio_l103_103149

/-- Define the conditions on function f and variables x and y. -/
def conditions (f : ℝ → ℝ) (x y : ℝ) :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x1 x2, x1 < x2 → f x1 < f x2) ∧
  f (x^2 - 6 * x) + f (y^2 - 4 * y + 12) ≤ 0

/-- The maximum value of (y - 2) / x under the given conditions. -/
theorem max_value_ratio (f : ℝ → ℝ) (x y : ℝ) (cond : conditions f x y) :
  (y - 2) / x ≤ (Real.sqrt 2) / 4 :=
sorry

end max_value_ratio_l103_103149


namespace decreasing_intervals_l103_103722

noncomputable def otimes (a b : ℝ) : ℝ :=
if a - b ≤ 2 then a else b

noncomputable def f (x : ℝ) : ℝ := otimes (3^(x+1)) (1-x)
noncomputable def g (x : ℝ) : ℝ := x^2 - 6 * x

theorem decreasing_intervals (m : ℝ) :
  (∀ x ∈ Ioc m (m + 1), deriv f x < 0 ∧ deriv g x < 0) ↔ (0 ≤ m ∧ m ≤ 2) :=
sorry

end decreasing_intervals_l103_103722


namespace compare_values_l103_103050

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

noncomputable def a : ℝ := f 1
noncomputable def b : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def c : ℝ := f ((Real.log 3 / Real.log 2) - 1)

theorem compare_values (h_log1 : Real.log 3 / Real.log 0.5 < -1) 
                       (h_log2 : 0 < (Real.log 3 / Real.log 2) - 1 ∧ (Real.log 3 / Real.log 2) - 1 < 1) : 
  b < a ∧ a < c :=
by
  sorry

end compare_values_l103_103050


namespace combined_tax_rate_correct_l103_103538

variables (mork_income mindy_income julie_income : ℝ)

-- Conditions
def mindy_income_eq_4mork : Prop := mindy_income = 4 * mork_income
def julie_income_eq_2mork : Prop := julie_income = 2 * mork_income
def julie_income_eq_halfmindy : Prop := julie_income = (1 / 2) * mindy_income

-- Tax rates
def mork_tax_rate : ℝ := 0.45
def mindy_tax_rate : ℝ := 0.25
def julie_tax_rate : ℝ := 0.35

-- Total income and total tax calculations
def total_income : ℝ := mork_income + mindy_income + julie_income
def total_tax : ℝ :=
  mork_tax_rate * mork_income + mindy_tax_rate * mindy_income + julie_tax_rate * julie_income

-- Combined tax rate
def combined_tax_rate : ℝ := total_tax / total_income

-- Assertion to prove
theorem combined_tax_rate_correct 
  (h1 : mindy_income_eq_4mork)
  (h2 : julie_income_eq_2mork)
  (h3 : julie_income_eq_halfmindy) :
  combined_tax_rate mork_income mindy_income julie_income = 2.15 / 7 :=
by
  -- Proof would go here
  sorry

end combined_tax_rate_correct_l103_103538


namespace circumcenter_equidistant_l103_103135

-- Define the geometric problem with all necessary conditions

variable {A B C D X Y P M S I : Type}
variable [metric_space A] [metric_space B] [metric_space C]
variable [metric_space D] [metric_space X] [metric_space Y]
variable [metric_space P] [metric_space M] [metric_space S] [metric_space I]
variable [metric_space ℝ]

-- Euler line related perpendicular condition
def foot_of_perpendicular_to_euler_line (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] :=
  -- Definition and properties related to the Euler line

-- Define circle passing through given points
def circle_through_points (A D : Type) [metric_space A] [metric_space D] (S : Type) [metric_space S] :=
  -- Circle omega with center S passing through points A and D

-- Define intersections with sides AB and AC
def circle_intersections (A B C X Y : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space X] [metric_space Y] :=
  -- Intersections of the circle with sides AB and AC at X and Y

-- Definitions for altitude foot and midpoint
def altitude_foot (A B C P : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space P] :=
  -- P is the foot of the altitude from A to BC

def midpoint (B C M : Type) [metric_space B] [metric_space C] [metric_space M] :=
  -- M is the midpoint of BC

-- The main theorem
theorem circumcenter_equidistant (D S X Y A B C P M I : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  [metric_space P] [metric_space M] [metric_space S] [metric_space I] [metric_space ℝ]
  (h1: foot_of_perpendicular_to_euler_line A B C D)
  (h2: circle_through_points A D S)
  (h3: circle_intersections A B C X Y)
  (h4: altitude_foot A B C P)
  (h5: midpoint B C M):
  dist (circumcenter (triangle X S Y)) P = dist (circumcenter (triangle X S Y)) M := sorry

end circumcenter_equidistant_l103_103135


namespace calculate_total_female_students_l103_103585

-- Conditions
def total_students : ℕ := 2000
def sample_size : ℕ := 200
def male_students_in_sample : ℕ := 103

-- Derived Conditions
def female_students_in_sample : ℕ := sample_size - male_students_in_sample
def sample_ratio : ℝ := sample_size / total_students

-- Question translated to Lean
def total_female_students : ℕ := female_students_in_sample * 10

theorem calculate_total_female_students :
  total_female_students = 970 := 
by
  -- Proof omitted
  sorry

end calculate_total_female_students_l103_103585


namespace cricket_initial_overs_l103_103485

-- Definitions based on conditions
def run_rate_initial : ℝ := 3.2
def run_rate_remaining : ℝ := 12.5
def target_runs : ℝ := 282
def remaining_overs : ℕ := 20

-- Mathematical statement to prove
theorem cricket_initial_overs (x : ℝ) (y : ℝ)
    (h1 : y = run_rate_initial * x)
    (h2 : y + run_rate_remaining * remaining_overs = target_runs) :
    x = 10 :=
sorry

end cricket_initial_overs_l103_103485


namespace matrix_mul_correct_l103_103338

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -3], ![2, 4]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![17, -5], ![16, -20]]

theorem matrix_mul_correct : A ⬝ B = C :=
  by
  sorry

end matrix_mul_correct_l103_103338


namespace systematic_sampling_l103_103929
-- importing the entire Mathlib library for necessary definitions and theorems.

-- Define total students, sample size, and known numbers in the sample
def N := 52
def n := 4
def first_in_sample := 7
def second_in_sample := 33
def third_in_sample := 46

-- Calculate the step length
def k := N / n

-- Condition to be proved: the fourth student number in the sequence
theorem systematic_sampling :
  ∃ fourth_in_sample : ℕ, fourth_in_sample = first_in_sample + k :=
begin
  sorry -- proof steps go here
end

end systematic_sampling_l103_103929


namespace melted_mixture_weight_l103_103960

variable (zinc copper total_weight : ℝ)
variable (ratio_zinc ratio_copper : ℝ := 9 / 11)
variable (weight_zinc : ℝ := 31.5)

theorem melted_mixture_weight :
  (zinc / copper = ratio_zinc / ratio_copper) ∧ (zinc = weight_zinc) →
  (total_weight = zinc + copper) →
  total_weight = 70 := 
sorry

end melted_mixture_weight_l103_103960


namespace ratio_male_democrats_l103_103609

theorem ratio_male_democrats (F M : ℕ) (total_participants : ℕ := 720)
  (female_democrats : ℕ := 120)
  (total_democrats : ℕ := total_participants / 3) :
  F + M = 720 →
  F / 2 = 120 →
  1 / 3 * (F + M) = total_democrats →
  female_democrats = 120 →
  (total_democrats - female_democrats) / M = 1 / 4 := 
by
  intros,
  sorry

end ratio_male_democrats_l103_103609


namespace slope_of_tangent_at_4_l103_103602

def f (x : ℝ) : ℝ := x^3 - 7 * x^2 + 1

theorem slope_of_tangent_at_4 : (deriv f 4) = -8 := by
  sorry

end slope_of_tangent_at_4_l103_103602


namespace train_length_is_correct_l103_103673

-- Definitions for conditions
def train_speed_km_hr : ℝ := 450  -- Train speed in km/hr
def crossing_time_sec : ℝ := 25  -- Time to cross a pole in seconds

-- Conversion from km/hr to m/s
def speed_km_hr_to_m_s (speed_km_hr : ℝ) : ℝ :=
  speed_km_hr * (1000 / 3600)

-- Predicted distance in meters
def length_of_train (speed_km_hr : ℝ) (time_sec : ℝ) : ℝ :=
  speed_km_hr_to_m_s speed_km_hr * time_sec

-- Lean statement to prove the length of the train is 3125 meters
theorem train_length_is_correct : length_of_train train_speed_km_hr crossing_time_sec = 3125 := 
by
  -- Placeholder for the actual proof
  sorry

end train_length_is_correct_l103_103673


namespace evaluate_expression_l103_103331

noncomputable def expression := 
  (Real.sqrt 3 * Real.tan (Real.pi / 15) - 3) / 
  (4 * (Real.cos (Real.pi / 15))^2 * Real.sin (Real.pi / 15) - 2 * Real.sin (Real.pi / 15))

theorem evaluate_expression : expression = -4 * Real.sqrt 3 :=
  sorry

end evaluate_expression_l103_103331


namespace triangle_obtuse_l103_103402

theorem triangle_obtuse (a b c : ℝ) (A B C : ℝ) 
  (hBpos : 0 < B) 
  (hBpi : B < Real.pi) 
  (sin_C_lt_cos_A_sin_B : Real.sin C / Real.sin B < Real.cos A) 
  (hC_eq : C = A + B) 
  (ha2 : A + B + C = Real.pi) :
  B > Real.pi / 2 := 
sorry

end triangle_obtuse_l103_103402


namespace tangent_circle_locus_l103_103591

-- Definitions for circle C1 and circle C2
def Circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Definition of being tangent to a circle
def ExternallyTangent (cx cy cr : ℝ) : Prop := (cx - 0)^2 + (cy - 0)^2 = (cr + 1)^2
def InternallyTangent (cx cy cr : ℝ) : Prop := (cx - 3)^2 + (cy - 0)^2 = (3 - cr)^2

-- Definition of locus L where (a,b) are centers of circles tangent to both C1 and C2
def Locus (a b : ℝ) : Prop := 28 * a^2 + 64 * b^2 - 84 * a - 49 = 0

-- The theorem to be proved
theorem tangent_circle_locus (a b r : ℝ) :
  (ExternallyTangent a b r) → (InternallyTangent a b r) → Locus a b :=
by {
  sorry
}

end tangent_circle_locus_l103_103591


namespace prime_power_of_three_l103_103856

theorem prime_power_of_three (p : ℕ) (hp : p > 3) (hp_prime : Prime p) :
  ∃ (a : ℕ → ℤ) (t : ℕ), 
    (∀ i, 1 ≤ i ∧ i ≤ t → p / 2 < a i ∧ a i < p / 2) ∧ 
    ∃ k : ℕ, (∏ i in finset.range(1, t + 1), (p - a i) / (int.natAbs (a i)) = 3^k) :=
sorry

end prime_power_of_three_l103_103856


namespace infinitely_many_negative_elements_l103_103395

def sequence (a : ℕ → ℝ) : ℕ → ℝ
| 0       := a 0
| (n + 1) := if a n ≠ 0 then (a n ^ 2 - 1) / (2 * a n) else 0

theorem infinitely_many_negative_elements (a1 : ℝ) :
  ∃ (S : set ℕ), (∀ n, n ∈ S ↔ (n ≥ 1 ∧ sequence (λ _, a1) n ≤ 0)) ∧ set.infinite S :=
sorry

end infinitely_many_negative_elements_l103_103395


namespace breadth_of_rectangle_is_20_l103_103196

-- Define the conditions
def area := 460
def percentage_increase := 0.15

-- Define the variables and relationships
def breadth (b : ℝ) :=
  ∃ l : ℝ, l = b * (1 + percentage_increase) ∧ l * b = area

-- Statement to prove
theorem breadth_of_rectangle_is_20 :
  ∃ b : ℝ, breadth b ∧ b = 20 :=
by
  sorry

end breadth_of_rectangle_is_20_l103_103196


namespace incorrect_statement_D_l103_103930

theorem incorrect_statement_D 
  (population : Set ℕ)
  (time_spent_sample : ℕ → ℕ)
  (sample_size : ℕ)
  (individual : ℕ)
  (h1 : ∀ s, s ∈ population → s ≤ 24)
  (h2 : ∀ i, i < sample_size → population (time_spent_sample i))
  (h3 : sample_size = 300)
  (h4 : ∀ i, i < 300 → time_spent_sample i = individual):
  ¬ (∀ i, i < 300 → time_spent_sample i = individual) :=
sorry

end incorrect_statement_D_l103_103930


namespace intersection_complement_l103_103770

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x^2 < 1 }
def B : Set ℝ := { x | x^2 - 2 * x > 0 }

theorem intersection_complement (A B : Set ℝ) : 
  (A ∩ (U \ B)) = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_complement_l103_103770


namespace parabola_x_intercepts_incorrect_l103_103011

-- Define the given quadratic function
noncomputable def f (x : ℝ) : ℝ := -1 / 2 * (x - 1)^2 + 2

-- The Lean statement for the problem
theorem parabola_x_intercepts_incorrect :
  ¬ ((f 3 = 0) ∧ (f (-3) = 0)) :=
by
  sorry

end parabola_x_intercepts_incorrect_l103_103011


namespace inverse_proportionality_ratio_l103_103193

variable {x y k x1 x2 y1 y2 : ℝ}

theorem inverse_proportionality_ratio
  (h1 : x * y = k)
  (hx1 : x1 ≠ 0)
  (hx2 : x2 ≠ 0)
  (hy1 : y1 ≠ 0)
  (hy2 : y2 ≠ 0)
  (hx_ratio : x1 / x2 = 3 / 4)
  (hxy1 : x1 * y1 = k)
  (hxy2 : x2 * y2 = k) :
  y1 / y2 = 4 / 3 := by
  sorry

end inverse_proportionality_ratio_l103_103193


namespace max_stamps_l103_103635

theorem max_stamps (n friends extra total: ℕ) (h1: friends = 15) (h2: extra = 5) (h3: total < 150) : total ≤ 140 :=
by
  sorry

end max_stamps_l103_103635


namespace rhombus_min_RS_value_l103_103842

variable (A B C D M R S O : Type)
variable [EuclideanSpace] (M : Points)
variable [R : RealExpose] (S : RealExpos)
variable [AC BD AM AB : Real]

def rhombus_diagonal_lengths (A B C D : Type) [EuclideanSpace] :=
  let AC_len := 24
  let BD_len := 40
  let AO_len := AC_len / 2
  let BO_len := BD_len / 2
  let AM_ratio := 1 / 3
  let AB_len := 2 * BO_len
  
  AO_len = 12 ∧ BO_len = 20 ∧ AM_ratio = 1 / 3 

def perpendicular_feet (M : Points) (AC BD : Real) :=
  let AM_len := (1 / 3) * (2 * (BD / 2))
  let MR_len := (AM_len / (BD / 2)) * (AC / 2)
  let MS_len := (AM_len / (AC / 2)) * (BD / 2)
  
  AM_len = 32 / 3 ∧ MR_len = 64 / 15 ∧ MS_len = 160 / 9

noncomputable def min_RS_value (AC BD : Real) :=
  let RS := sqrt ((64 / 15) ^ 2 + (160 / 9) ^ 2)
  RS = 6.37

theorem rhombus_min_RS_value :
  rhombus_diagonal_lengths A B C D → 
  perpendicular_feet M AC BD →
  min_RS_value AC BD := 
  sorry

end rhombus_min_RS_value_l103_103842


namespace hyperbola_focal_distance_and_asymptotes_l103_103766

-- Define the hyperbola
def hyperbola (y x : ℝ) : Prop := (y^2 / 4) - (x^2 / 3) = 1

-- Prove the properties
theorem hyperbola_focal_distance_and_asymptotes :
  (∀ y x : ℝ, hyperbola y x → ∃ c : ℝ, c = 2 * Real.sqrt 7)
  ∧
  (∀ y x : ℝ, hyperbola y x → (y = (2 * Real.sqrt 3 / 3) * x ∨ y = -(2 * Real.sqrt 3 / 3) * x)) :=
by
  sorry

end hyperbola_focal_distance_and_asymptotes_l103_103766


namespace ratio_PeteHand_to_TracyCartwheel_l103_103564

noncomputable def SusanWalkingSpeed (PeteBackwardSpeed : ℕ) : ℕ :=
  PeteBackwardSpeed / 3

noncomputable def TracyCartwheelSpeed (SusanSpeed : ℕ) : ℕ :=
  SusanSpeed * 2

def PeteHandsWalkingSpeed : ℕ := 2

def PeteBackwardWalkingSpeed : ℕ := 12

theorem ratio_PeteHand_to_TracyCartwheel :
  let SusanSpeed := SusanWalkingSpeed PeteBackwardWalkingSpeed
  let TracySpeed := TracyCartwheelSpeed SusanSpeed
  (PeteHandsWalkingSpeed : ℕ) / (TracySpeed : ℕ) = 1 / 4 :=
by
  sorry

end ratio_PeteHand_to_TracyCartwheel_l103_103564


namespace tan_double_angle_l103_103383

theorem tan_double_angle (α : ℝ) (h₁ : Real.sin α = 4/5) (h₂ : α ∈ Set.Ioc (π / 2) π) :
  Real.tan (2 * α) = 24 / 7 := 
  sorry

end tan_double_angle_l103_103383


namespace total_animals_l103_103223

theorem total_animals : ∀ (D C R : ℕ), 
  C = 5 * D →
  R = D - 12 →
  R = 4 →
  (C + D + R = 100) :=
by
  intros D C R h1 h2 h3
  sorry

end total_animals_l103_103223


namespace actual_distance_traveled_l103_103460

theorem actual_distance_traveled
  (D : ℝ) 
  (H : ∃ T : ℝ, D = 5 * T ∧ D + 20 = 15 * T) : 
  D = 10 :=
by
  sorry

end actual_distance_traveled_l103_103460


namespace extremum_is_not_unique_l103_103889

-- Define the extremum conditionally in terms of unique extremum within an interval for a function
def isExtremum {α : Type*} [Preorder α] (f : α → ℝ) (x : α) :=
  ∀ y, f y ≤ f x ∨ f x ≤ f y

theorem extremum_is_not_unique (α : Type*) [Preorder α] (f : α → ℝ) :
  ¬ ∀ x, isExtremum f x → (∀ y, isExtremum f y → x = y) :=
by
  sorry

end extremum_is_not_unique_l103_103889


namespace max_n_distinct_squares_sum_eq_2100_l103_103244

theorem max_n_distinct_squares_sum_eq_2100 :
  ∃ (n : ℕ), (∀ (k : ℕ → ℕ), (∀ i j, i ≠ j → k i ≠ k j) →
    (∑ i in finset.range n, (k i)^2 = 2100)) → n = 17 :=
sorry

end max_n_distinct_squares_sum_eq_2100_l103_103244


namespace square_perimeter_l103_103587

/-- The area of a square is equal to twice the area of a rectangle with given dimensions,
and we aim to prove the perimeter of the square. -/
theorem square_perimeter (length width : ℝ) (h_length : length = 32) (h_width : width = 64)
  (h_square : ∃ (s : ℝ), s^2 = 2 * (length * width)) :
  ∃ (p : ℝ), p = 4 * 64 :=
by
  -- Establish the rectangle's area
  have h_area_rectangle : length * width = 2048, by
    rw [h_length, h_width]
    simp only [mul_eq_mul_right_iff, eq_self_iff_true, true_or, zero_mul, ne.def, not_false_iff]
    norm_num,
  
  -- Establish the square's area
  rcases h_square with ⟨s, hs⟩,
  have h_square_area : s = 64, by
    have : 2 * 2048 = 4096, by norm_num,
    replace hs : s^2 = 4096, by rwa h_area_rectangle at hs,
    exact eq_of_pow_two_eq_pow_two (by norm_num : 0 <= 4096) hs,

  -- Establish the perimeter of the square
  use 4 * 64,
  norm_num,
  sorry

end square_perimeter_l103_103587


namespace solve_tangent_equation_l103_103961

theorem solve_tangent_equation (k : ℤ) :
  (∀ x : ℝ, cos (5 * x) ≠ 0 ∧ cos (3 * x) ≠ 0 → (tan (5 * x) - 2 * tan (3 * x) = tan (3 * x) ^ 2 * tan (5 * x)) → ∃ k : ℤ, x = k * π) := 
by
  intro x hx_eq
  sorry

end solve_tangent_equation_l103_103961


namespace domain_of_y_min_value_of_f_l103_103763

noncomputable def domainM : Set ℝ := {x | -1 ≤ x ∧ x < 1}

noncomputable def f (x : ℝ) : ℝ := 4 ^ x + 2 ^ (x + 2)

theorem domain_of_y :
  (∀ x : ℝ, y = sqrt ((1 + x) / (1 - x)) + log (3 - 4 * x + x ^ 2) → x ∈ domainM) := by
  sorry

theorem min_value_of_f :
  ∀ x ∈ domainM, f (x) ≥ 9 / 4 := by
  sorry

end domain_of_y_min_value_of_f_l103_103763


namespace circumradius_of_inradius_l103_103734

variable {A B C I : Type}

/-- Given I is the incenter of triangle ABC, 
    and 5 * vector IA = 4 * (vector BI + vector CI).
    Let R and r be the circumradius and inradius of triangle ABC respectively.
    If r = 15, then R = 32.
-/
theorem circumradius_of_inradius (hI : I is_incenter_of_triangle (A, B, C))
    (hvec : 5 • (vector AI) = 4 • (vector BI + vector CI))
    (h_r : r = 15) : R = 32 :=
sorry

end circumradius_of_inradius_l103_103734


namespace toy_poodle_height_l103_103916

theorem toy_poodle_height 
  (SP MP TP : ℕ)
  (h1 : SP = MP + 8)
  (h2 : MP = TP + 6)
  (h3 : SP = 28) 
  : TP = 14 := 
    by sorry

end toy_poodle_height_l103_103916


namespace good_numbers_correct_l103_103110

noncomputable def good_numbers (n : ℕ) : ℝ :=
  1 / 2 * (8^n + 10^n) - 1

theorem good_numbers_correct (n : ℕ) : good_numbers n = 
  1 / 2 * (8^n + 10^n) - 1 := 
sorry

end good_numbers_correct_l103_103110


namespace range_of_a_l103_103062

noncomputable def f (a x : ℝ) : ℝ := a * |x| - 3 * a - 1

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x ∈ set.Icc (-1) 1 ∧ f a x = 0) ↔ (a ∈ set.Icc (-1/2 : ℝ) (-1/3 : ℝ)) :=
by
  sorry

end range_of_a_l103_103062


namespace area_of_triangle_l103_103239

theorem area_of_triangle:
  let line1 := λ x => 3 * x - 6
  let line2 := λ x => -2 * x + 18
  let y_axis: ℝ → ℝ := λ _ => 0
  let intersection := (4.8, line1 4.8)
  let y_intercept1 := (0, -6)
  let y_intercept2 := (0, 18)
  (1/2) * 24 * 4.8 = 57.6 := by
  sorry

end area_of_triangle_l103_103239


namespace sqrt_defined_iff_nonneg_l103_103474

theorem sqrt_defined_iff_nonneg (x : ℝ) : (∃ y, y = sqrt (x - 2)) ↔ x ≥ 2 :=
by
  sorry

end sqrt_defined_iff_nonneg_l103_103474


namespace problem_quadrant_l103_103118

-- Define the complex number given in the problem
def given_complex : ℂ := (Complex.I / (1 + Complex.I)) + ((1 + Complex.sqrt 3 * Complex.I)^2)

-- Define a function to determine the quadrant
def quadrant (z : ℂ) : ℕ :=
  if z.re > 0 then
    if z.im > 0 then 1 else 4
  else
    if z.im > 0 then 2 else 3

-- State the theorem
theorem problem_quadrant : quadrant given_complex = 2 := by
  sorry

end problem_quadrant_l103_103118


namespace power_equation_l103_103448

theorem power_equation (x : ℝ) (h : 128^3 = 16^x) : 2^{-x} = 1 / 2^(21 / 4) :=
sorry

end power_equation_l103_103448


namespace choosing_officers_l103_103977

-- Define the problem conditions
def clubMembers := 20
def positions := 3

-- Define the proof problem
theorem choosing_officers :
  (Finset.range clubMembers).card.Choose positions * factorial positions = 6840 := 
by
  sorry

end choosing_officers_l103_103977


namespace relationship_among_a_b_c_l103_103403

noncomputable def a : ℝ := 3 ^ Real.cos (Real.pi / 6)
noncomputable def b : ℝ := Real.log (Real.sin (Real.pi / 6)) / Real.log (1 / 3)
noncomputable def c : ℝ := Real.log (Real.tan (Real.pi / 6)) / Real.log 2

theorem relationship_among_a_b_c : a > b ∧ b > c := 
by
  sorry

end relationship_among_a_b_c_l103_103403


namespace false_proposition_l103_103317

-- Definitions based on the conditions in the problem
def propositionA (quad : Quadrilateral) : Prop := 
  (quad.oppositeSidesEqual ∧ quad.oppositeAnglesEqual) → quad.isParallelogram

def propositionB (quad : Quadrilateral) : Prop := 
  (quad.diagonalsEqual ∧ quad.diagonalsBisectEachOther) → quad.isRectangle

def propositionC (rhombus : Rhombus) : Prop :=
  rhombus.diagonalsEqual → rhombus.isSquare

def propositionD (rect : Rectangle) : Prop :=
  rect.diagonalsPerpendicular → rect.isSquare

-- The main theorem to be proved
theorem false_proposition : ¬ propositionA ∧ (propositionB ∧ propositionC ∧ propositionD) :=
by
  sorry

end false_proposition_l103_103317


namespace hyperbola_parameters_sum_l103_103108

theorem hyperbola_parameters_sum :
  let h := 1
  let k := 1
  let a := 3
  let c := 7
  let b := real.sqrt 40
  h + k + a + b = (5 : ℝ) + 2 * real.sqrt 10 :=
by
  sorry

end hyperbola_parameters_sum_l103_103108


namespace Alice_wins_optimal_play_l103_103677

/-- Alice (White) wins under optimal play on a 25 x 25 grid chessboard, given the game conditions. -/
theorem Alice_wins_optimal_play:
  ∀ (board : Array (Array (Option Bool))),
  (∀ i < 25, ∀ j < 25, board[i][j] = none) → -- initial condition: all cells are empty
  (∀ (turns : ℕ), even turns → ∀ i < 25, ∀ j < 25, -- Alice's turn
    (board[i][j] ≠ some true) →  -- cell is empty
    (i < 24 ∧ board[i+1][j] ≠ some true ∨ i > 0 ∧ board[i-1][j] ≠ some true ∨ 
      j < 24 ∧ board[i][j+1] ≠ some true ∨ j > 0 ∧ board[i][j-1] ≠ some true) → -- neighboring cells not all white
    ∃ i < 25, ∃ j < 25, board[i][j] = none) → -- Alice has a move
  (∀ (turns : ℕ), odd turns → ∀ i < 25, ∀ j < 25, -- Bob's turn
    (board[i][j] ≠ some false) →  -- cell is empty
    (i < 24 ∧ board[i+1][j] ≠ some false ∨ i > 0 ∧ board[i-1][j] ≠ some false ∨ 
      j < 24 ∧ board[i][j+1] ≠ some false ∨ j > 0 ∧ board[i][j-1] ≠ some false) → -- neighboring cells not all black
    ∃ i < 25, ∃ j < 25, board[i][j] = none) → -- Bob has a move
  False :=
by sorry  -- insert proof here

end Alice_wins_optimal_play_l103_103677


namespace surface_area_increase_l103_103234

-- Define the conditions
def diameter : ℝ := 4 -- Diameter of the onion in cm
def thickness : ℝ := 0.2 -- Thickness of each slice in cm (2 mm)

-- Calculate the original surface area of the spherical onion
def original_surface_area (d : ℝ) : ℝ := 4 * π * (d / 2)^2

-- Calculate the combined surface area of the slices
-- This part essentially includes the summation calculation accounted for 20 slices
def combined_surface_area : ℝ := sorry -- placeholder for the detailed calculation

-- The ratio of combined surface area to original surface area
def surface_area_ratio (d : ℝ) (t : ℝ) : ℝ :=
  combined_surface_area / original_surface_area d

-- The theorem to prove
theorem surface_area_increase :
  surface_area_ratio diameter thickness = 7.65 :=
sorry

end surface_area_increase_l103_103234


namespace number_from_percentage_l103_103657

theorem number_from_percentage (p : ℝ) (b : ℝ) (n : ℝ) : 
  p = 1.50 → b = 80 → n = p * b → n = 120 :=
by
  intros hp hb hn
  rw [hp, hb] at hn
  exact hn

end number_from_percentage_l103_103657


namespace matrix_mul_correct_l103_103339

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -3], ![2, 4]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![17, -5], ![16, -20]]

theorem matrix_mul_correct : A ⬝ B = C :=
  by
  sorry

end matrix_mul_correct_l103_103339


namespace area_of_BEIH_l103_103449

def Point := (ℚ × ℚ)
def Line (p1 p2 : Point) : ℚ → ℚ := 
  λ x, (if p2.1 - p1.1 ≠ 0 then 
          (p2.2 - p1.2) / (p2.1 - p1.1) * (x - p1.1) + p1.2
        else
          p1.2) -- handle vertical line

noncomputable def intersection (l1 l2 : ℚ → ℚ) : Point :=
  let a := l1 0 in 
  let b := l2 0 in
  let x := (b - a) / ((l1 1 - a) - (l2 1 - b)) in
  (x, l1 x)

def shoelace_area (pts : List Point) : ℚ :=
  | ∑ i in range pts.length, pts[i].1 * pts[(i + 1) % pts.length].2 - pts[(i + 1) % pts.length].1 * pts[i].2 | / 2

theorem area_of_BEIH :
  let A : Point := (0, 0)
  let B : Point := (0, 2)
  let C : Point := (3, 2)
  let D : Point := (3, 0)
  let E : Point := (0, 1)
  let F : Point := (1.5, 2)
  let AF := Line A F
  let DE := Line D E
  let BD := Line B D
  let I := intersection AF DE
  let H := intersection BD AF
  shoelace_area [B, E, I, H] = 27 / 20 :=
by 
  sorry

end area_of_BEIH_l103_103449


namespace count_valid_2x2_grids_l103_103085

def is_valid_grid (grid : Matrix (Fin 2) (Fin 2) ℕ) : Prop :=
  ∀ i j : Fin 2, ∀ k l : Fin 2, (abs (grid i j - grid k l) ≤ 2 ∨ (i, j) = (k, l)) ∧ grid i j ∈ {1, 2, 3, 4}

def valid_grid_count (n : ℕ) : Prop :=
  n = 8

theorem count_valid_2x2_grids : ∃ g : Matrix (Fin 2) (Fin 2) ℕ, is_valid_grid g ∧ valid_grid_count 8 :=
  sorry

end count_valid_2x2_grids_l103_103085


namespace percent_voters_for_candidate_A_l103_103482

theorem percent_voters_for_candidate_A (d r i u p_d p_r p_i p_u : ℝ) 
  (hd : d = 0.45) (hr : r = 0.30) (hi : i = 0.20) (hu : u = 0.05)
  (hp_d : p_d = 0.75) (hp_r : p_r = 0.25) (hp_i : p_i = 0.50) (hp_u : p_u = 0.50) :
  d * p_d + r * p_r + i * p_i + u * p_u = 0.5375 :=
by
  sorry

end percent_voters_for_candidate_A_l103_103482


namespace erasers_given_l103_103565

theorem erasers_given (initial final : ℕ) (h1 : initial = 8) (h2 : final = 11) : (final - initial = 3) :=
by
  sorry

end erasers_given_l103_103565


namespace solve_triangle_problem_l103_103441
noncomputable def triangle_problem (A B C a b c : ℝ) (area : ℝ) : Prop :=
  (2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0) ∧
  area = Real.sqrt 3 ∧ 
  b + c = 5 →
  (A = Real.pi / 3) ∧ (a = Real.sqrt 13)

-- Lean statement for the proof problem
theorem solve_triangle_problem 
  (A B C a b c : ℝ) 
  (h1 : 2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0)
  (h2 : 1/2 * b * c * Real.sin A = Real.sqrt 3)
  (h3 : b + c = 5) :
  A = Real.pi / 3 ∧ a = Real.sqrt 13 :=
sorry

end solve_triangle_problem_l103_103441


namespace percent_problem_l103_103100

theorem percent_problem (x y z w : ℝ) 
  (h1 : x = 1.20 * y) 
  (h2 : y = 0.40 * z) 
  (h3 : z = 0.70 * w) : 
  x = 0.336 * w :=
sorry

end percent_problem_l103_103100


namespace find_salary_B_l103_103598

def salary_A : ℕ := 8000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000
def avg_salary : ℕ := 8000

theorem find_salary_B (S_B : ℕ) :
  (salary_A + S_B + salary_C + salary_D + salary_E) / 5 = avg_salary ↔ S_B = 5000 := by
  sorry

end find_salary_B_l103_103598


namespace sum_of_digits_from_1_to_100000_l103_103366

def sumOfDigits (n : Nat) : Nat :=
  if n = 0 then 0
  else (n % 10) + sumOfDigits (n / 10)

def sumOfDigitsInRange (a b : Nat) : Nat :=
  if a > b then 0
  else sumOfDigits a + sumOfDigitsInRange (a + 1) b

theorem sum_of_digits_from_1_to_100000 :
  sumOfDigitsInRange 1 100000 = 2443446 :=
by
  sorry

end sum_of_digits_from_1_to_100000_l103_103366


namespace smallest_N_triangle_ineq_l103_103006

theorem smallest_N_triangle_ineq (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c < a + b) : (a^2 + b^2 + a * b) / c^2 < 1 := 
sorry

end smallest_N_triangle_ineq_l103_103006


namespace value_added_to_half_is_five_l103_103775

theorem value_added_to_half_is_five (n V : ℕ) (h₁ : n = 16) (h₂ : (1 / 2 : ℝ) * n + V = 13) : V = 5 := 
by 
  sorry

end value_added_to_half_is_five_l103_103775


namespace triangle_tan_product_l103_103617

/-- In a triangle ABC with circumcenter O and incircle γ, let ∠BAC = 60°. 
    Given that O lies on the incircle γ, we want to show that tan B * tan C = 4 + 2 * √2,
    and further compute 100a + b for positive integers a and b, such that tan B * tan C = a + √b.
/ -/
theorem triangle_tan_product 
  (ABC : Type) [Triangle ABC]
  (O : Point) [Circumcenter O ABC]
  (γ : Circle) [Incircle γ ABC]
  (hO_on_γ : OnCircle O γ)
  (angle_BAC : ∠BAC = 60°) :
  ∃ (a b : ℕ), a := 4 ∧ b := 8 ∧ tan ∠ABC * tan ∠ACB = a + sqrt b ∧ 100a + b = 408 :=
begin
  sorry,  -- Proof not required
end

end triangle_tan_product_l103_103617


namespace copy_pages_cost_l103_103834

theorem copy_pages_cost :
  (7 : ℕ) * (n : ℕ) = 3500 * 4 / 7 → n = 2000 :=
by
  sorry

end copy_pages_cost_l103_103834


namespace cyclic_quad_equal_angles_l103_103221

-- Definition and conditions of a cyclic quadrilateral
def cyclic_quadrilateral (A B C D : Type*) :=
  ∃ (O : Type*), circle O ∧ points_on_circle [A, B, C, D] O

-- The main theorem statement
theorem cyclic_quad_equal_angles {α : Type*} [circle α] {A B C D : α} (h : cyclic_quadrilateral A B C D) :
  ∠BAC = ∠BDC ∧
  ∠ADB = ∠ACB ∧
  ∠CAD = ∠CBD ∧
  ∠ABD = ∠ACD :=
by
  sorry  -- Proof is not required, skipping with "sorry"


end cyclic_quad_equal_angles_l103_103221


namespace part1_is_perfect_number_29_13_part2_value_of_mn_part3_value_of_k_for_perfect_S_part4_minimum_value_l103_103596

-- Given definitions related to the problem
def is_perfect_number (n : ℤ) : Prop := ∃ a b : ℤ, n = a^2 + b^2

-- Part 1: Identify the "perfect numbers" among given choices
theorem part1_is_perfect_number_29_13 :
  is_perfect_number 29 ∧ is_perfect_number 13 :=
sorry

-- Part 2: Given equation and finding the value of mn
theorem part2_value_of_mn (m n a : ℤ) (h : a^2 - 4a + 8 = (a - m)^2 + n^2) :
  m = 2 ∧ (n = 2 ∨ n = -2) → m * n = ± 4 :=
sorry

-- Part 3: Given expression and finding k for S to be a "perfect number"
theorem part3_value_of_k_for_perfect_S (a b : ℤ) (S : ℤ → ℤ → ℤ) :
  S a b = a^2 + 4*a*b + 5*b^2 - 12*b + 36 →
  is_perfect_number (S a b) :=
sorry

-- Part 4: Finding minimum value of a + b given equation
theorem part4_minimum_value (a b : ℝ) (h : -a^2 + 5*a + b - 7 = 0) :
  a + b = 3 :=
sorry

end part1_is_perfect_number_29_13_part2_value_of_mn_part3_value_of_k_for_perfect_S_part4_minimum_value_l103_103596


namespace solve_set_B_l103_103022

theorem solve_set_B (a b : ℝ) (f : ℝ → ℝ) (A B : set ℝ) (hA : A = {1, -3}) 
                    (hf : ∀ x, f x = x^2 - a * x + b) 
                    (hB : B = {x | f x - a * x = 0}) :
    (B = {-2 - Real.sqrt 7, -2 + Real.sqrt 7}) :=
by
  sorry

end solve_set_B_l103_103022


namespace percentage_growth_of_edge_l103_103260

theorem percentage_growth_of_edge
    (L : ℝ) -- Original edge length
    (P : ℝ) -- Percentage growth of edge
    (A : ℝ := 6 * L^2) -- Original surface area
    (L' : ℝ := L * (1 + P / 100)) -- New edge length
    (A' : ℝ := 6 * L'^2) -- New surface area
    (percentage_increase_area : ℝ := (A' - A) / A * 100) -- Given percentage increase in surface area
    (percentage_increase_value : percentage_increase_area = 156.00000000000006) :
    P = 60 := 
begin
  -- sorry is a placeholder for the proof
  sorry
end

end percentage_growth_of_edge_l103_103260


namespace product_of_possible_values_of_g3_l103_103855

theorem product_of_possible_values_of_g3 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (g x + y) = g (x ^ 2 + y) + 4 * g x * y) :
  let m := 2 in  -- the number of possible values of g(3) (g(3) = 0 and g(3) = 9)
  let t := 0 + 9 in  -- the sum of all possible values of g(3)
  m * t = 18 := by
  let m : ℕ := 2
  let t : ℝ := 0 + 9
  have : m * t = (2 : ℝ) * 9, by rw [t, m]; norm_num
  rw [Nat.cast_bit0, Nat.cast_one, this]
sorry

end product_of_possible_values_of_g3_l103_103855


namespace square_area_in_scientific_notation_l103_103911

theorem square_area_in_scientific_notation:
    (side_length : ℝ) (h : side_length = 5 * 10^2) : 
    (side_length^2) = 2.5 * 10^5 :=
by
  rw h
  sorry

end square_area_in_scientific_notation_l103_103911


namespace triangle_area_sqrt2_div2_find_a_c_l103_103480

  -- Problem 1
  -- Prove the area of triangle ABC is sqrt(2)/2
  theorem triangle_area_sqrt2_div2 {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : Real.arcsin (1) = Real.pi / 2) : 
    (1 / 2) * 1 * Real.sqrt 2 = Real.sqrt 2 / 2 := sorry

  -- Problem 2
  -- Prove a = sqrt(7) and c = 2
  theorem find_a_c {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : (1 / 2) * a * Real.sin (Real.arcsin (Real.sqrt 3 / a)) = Real.sqrt 3 / 2) : 
    a = Real.sqrt 7 ∧ c = 2 := sorry

  
end triangle_area_sqrt2_div2_find_a_c_l103_103480


namespace complementary_three_card_sets_l103_103981

-- Definitions for the problem conditions
inductive Shape | circle | square | triangle | star
inductive Color | red | blue | green | yellow
inductive Shade | light | medium | dark | very_dark

-- Definition of a Card as a combination of shape, color, shade
structure Card :=
(shape : Shape)
(color : Color)
(shade : Shade)

-- Definition of a set being complementary
def is_complementary (c1 c2 c3 : Card) : Prop :=
  ((c1.shape = c2.shape ∧ c2.shape = c3.shape) ∨ (c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape)) ∧
  ((c1.color = c2.color ∧ c2.color = c3.color) ∨ (c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color)) ∧
  ((c1.shade = c2.shade ∧ c2.shade = c3.shade) ∨ (c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade))

-- Definition of the problem statement
def complementary_three_card_sets_count : Nat :=
  360

-- The theorem to be proved
theorem complementary_three_card_sets : ∃ (n : Nat), n = complementary_three_card_sets_count :=
  by
    use 360
    sorry

end complementary_three_card_sets_l103_103981


namespace complement_U_P_l103_103739

def U : Set ℝ := {y | ∃ x > 0, y = logBase 2 x}
def P : Set ℝ := {y | ∃ x > 2, y = 1/x}

theorem complement_U_P : U = Set.univ → P = Set.Ioo 0 (1/2) → (U \ P) = (Set.Iic 0 ∪ Set.Ici (1/2)) :=
by
  intro hU hP
  rw [hU, hP]
  sorry

end complement_U_P_l103_103739


namespace ADQR_cyclic_l103_103492

-- Declare that we have a convex quadrilateral ABCD
variables {A B C D P Q R T : Type} [metric_space A]
variables [metric_space B] [metric_space C] [metric_space D]
variables [metric_space P] [metric_space Q] [metric_space R] [metric_space T]

-- Assume conditions of the problem
axiom AP_PT_TD : ∀ {a b c d : A}, dist a b = dist b c ∧ dist c d = dist a b
axiom QB_BC_CR : ∀ {a b c d : B}, dist a b = dist b c ∧ dist c d = dist a b
axiom PQ_on_AB : ∀ {a b : A}, dist a b > 0
axiom RT_on_CD : ∀ {a b : C}, dist a b > 0
axiom BCTP_cyclic : ∀ {a b c d : Type}, isCyclic a b c d

-- Prove that ADQR is cyclic
theorem ADQR_cyclic : isCyclic A D Q R :=
by {
  sorry
}

end ADQR_cyclic_l103_103492


namespace sum_first_n_terms_l103_103066

def sequence (n : ℕ) : ℝ :=
  1 / (Real.sqrt (n + 1) + Real.sqrt n)

def sum_sequence (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, sequence i

theorem sum_first_n_terms (n : ℕ) :
  sum_sequence n = Real.sqrt (n + 1) - 1 :=
sorry

end sum_first_n_terms_l103_103066


namespace coefficient_x5_in_expansion_of_x_minus_2_pow_36_l103_103498

-- A noncomputable constant for the calculations involving large numbers.
noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- The main theorem statement about the coefficient of x^5 in (x - 2)^36
theorem coefficient_x5_in_expansion_of_x_minus_2_pow_36 :
  (binomial 36 5) * (-2) ^ 31 = -8105545721856 :=
by
  sorry

end coefficient_x5_in_expansion_of_x_minus_2_pow_36_l103_103498


namespace place_spies_l103_103305

/--
A definition of the vision constraints for a spy on a 6x6 board. A spy sees the cells two steps straight ahead 
and one step on each side (left and right) in front of them.
-/
def spy_vision (board : Matrix) (i j : ℕ) : Prop :=
  ∀ i' j', (i' = i ∧ (j' = j + 1 ∨ j' = j + 2)) ∨
  ((i' = i + 1 ∨ i' = i - 1) ∧ j' = j) → board i' j' = "."

/--
A formal statement to place 18 spies on a 6x6 board such that no two spies can see each other.
-/
theorem place_spies : ∃ board : matrix (fin 6) (fin 6) bool,
  (board.enum.filter (λ x, x ≠ 0) = 18) ∧
  ∀ (i₁ j₁ i₂ j₂ : ℕ), board i₁ j₁ = true → board i₂ j₂ = true →
  (i₁ ≠ i₂ ∧ j₁ ≠ j₂) ∧ spy_vision board i₁ j₁ ∧ spy_vision board i₂ j₂ :=
sorry

end place_spies_l103_103305


namespace alice_cannot_win_l103_103679

-- Definitions for the board and coins
inductive Coin
| Penny | Nickel | Dime | Quarter
deriving DecidableEq

structure Board :=
(squares : Fin 20 × Fin 20 → Option Coin)

def initial_board : Board :=
{ squares := fun ⟨x, y⟩ => 
    if (x: ℕ) * 20 + (y: ℕ) < 100 then some Coin.Penny
    else if (x: ℕ) * 20 + (y: ℕ) < 200 then some Coin.Nickel
    else if (x: ℕ) * 20 + (y: ℕ) < 300 then some Coin.Dime
    else some Coin.Quarter }

def is_vacant (b: Board) (pos: Fin 20 × Fin 20) : Prop :=
b.squares pos = none

def can_remove_penny (b: Board) (pos: Fin 20 × Fin 20) : Prop :=
is_vacant b (pos + (0, -1)) ∧ is_vacant b (pos + (0, 1)) ∧
is_vacant b (pos + (-1, 0)) ∧ is_vacant b (pos + (1, 0))

def can_remove_nickel (b: Board) (pos: Fin 20 × Fin 20) : Prop :=
(is_vacant b (pos + (0, -1)) + is_vacant b (pos + (0, 1)) +
is_vacant b (pos + (-1, 0)) + is_vacant b (pos + (1, 0))) ≥ 3

def can_remove_dime (b: Board) (pos: Fin 20 × Fin 20) : Prop :=
(is_vacant b (pos + (0, -1)) + is_vacant b (pos + (0, 1)) +
is_vacant b (pos + (-1, 0)) + is_vacant b (pos + (1, 0))) ≥ 2

def can_remove_quarter (b: Board) (pos: Fin 20 × Fin 20) : Prop :=
(is_vacant b (pos + (0, -1)) + is_vacant b (pos + (0, 1)) +
is_vacant b (pos + (-1, 0)) + is_vacant b (pos + (1, 0))) ≥ 1

-- Main theorem
theorem alice_cannot_win : ∀ (b : Board),
  (∃ pos, b.squares pos ≠ none) ∧ ⟦ ∀ pos,
  (b.squares pos = some Coin.Penny → ¬ can_remove_penny b pos) ∧
  (b.squares pos = some Coin.Nickel → ¬ can_remove_nickel b pos) ∧
  (b.squares pos = some Coin.Dime → ¬ can_remove_dime b pos) ∧
  (b.squares pos = some Coin.Quarter → ¬ can_remove_quarter b pos)) →
  true := sorry

end alice_cannot_win_l103_103679


namespace probability_P_plus_S_is_one_less_than_multiple_of_seven_l103_103938

theorem probability_P_plus_S_is_one_less_than_multiple_of_seven :
  ∀ (a b : ℕ), a ∈ finset.range(1, 61) → b ∈ finset.range(1, 61) → a ≠ b →
  let S := a + b in
  let P := a * b in
  (nat.gcd ((P + S + 1), 7) = 1) →
  (finset.filter (λ (a b : ℕ), (a+1) ∣ 7 ∨ (b+1) ∣ 7) (finset.range(1, 61)).product (finset.range(1, 61)).card) / 1770 = 74 / 295 :=
begin
  sorry
end

end probability_P_plus_S_is_one_less_than_multiple_of_seven_l103_103938


namespace cell_without_arrow_exists_l103_103819

def arrow := ℕ

structure cell :=
(x: ℕ)
(y: ℕ)

def has_arrow (c: cell) : Prop := sorry

def boundary_condition (c: cell) : Prop := sorry

def no_opposite_arrows (c1 c2: cell) : Prop := sorry

theorem cell_without_arrow_exists :
  ∀ (grid : fin 20 × fin 20),
  (∀ c, boundary_condition c → has_arrow c) →
  (∀ c1 c2, (adjacent c1 c2 ∨ diagonal_adjacent c1 c2) → ¬ (opposite_arrow c1 c2)) →
  ∃ c, ¬ has_arrow c :=
begin
  sorry
end

end cell_without_arrow_exists_l103_103819


namespace quadratic_min_value_quadratic_max_value_on_interval_l103_103414

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * (x - 1) ^ 2 + 1

theorem quadratic_min_value :
  (∀ x, f x ≥ 1) ∧ (f 0 = 3) ∧ (f 2 = 3) :=
begin
  split,
  { intro x,
    have h := calc
      f x = 2 * (x - 1) ^ 2 + 1 : rfl
         ... ≥ 1 : by { apply add_nonneg, linarith, apply mul_nonneg, norm_num, apply sq_nonneg },
    exact h },
  split,
  { exact calc
      f 0 = 2 * (0 - 1) ^ 2 + 1 : rfl
         ... = 2 * 1 ^ 2 + 1 : by rw [zero_sub, one_pow]
         ... = 3 : by norm_num },
  { exact calc
      f 2 = 2 * (2 - 1) ^ 2 + 1 : rfl
         ... = 2 * 1 ^ 2 + 1 : by rw [sub_self, sub_add_cancel]
         ... = 3 : by norm_num }
end

theorem quadratic_max_value_on_interval :
  sup {f x | x ∈ set.Icc (-1 / 2 : ℝ) (3 / 2)} = 11 / 2 :=
begin
  sorry -- Proof of maximum value on the interval can be included here.
end

end quadratic_min_value_quadratic_max_value_on_interval_l103_103414


namespace find_interest_rate_l103_103306

def compound_interest_rate (P : ℝ) (r : ℝ) (n : ℝ) : ℝ :=
  P * (1 + r / 100) ^ n

theorem find_interest_rate (A2 : ℝ) (A3 : ℝ) (P : ℝ) (r : ℝ) :
  compound_interest_rate P r 2 = 2420 → 
  compound_interest_rate P r 3 = 3267 → 
  r = 34.96 :=
by
  intros h1 h2
  sorry

end find_interest_rate_l103_103306


namespace sin_alpha_plus_beta_l103_103018

variable (α β : Real)
variable (h1 : Cos (π / 4 - α) = 3 / 5)
variable (h2 : Sin (5 * π / 4 + β) = -12 / 13)
variable (h3 : α ∈ Ioo (π / 4) (3 * π / 4))
variable (h4 : β ∈ Ioo 0 (π / 4))

theorem sin_alpha_plus_beta :
  Sin (α + β) = 56 / 65 := by
  sorry

end sin_alpha_plus_beta_l103_103018


namespace tree_problem_l103_103663

-- Definitions
def distance (a b : ℕ) : ℕ := abs (a - b)

-- Conditions and the final proof statement
theorem tree_problem
(a b c d e : ℕ) -- positions of poplar, willow, locust, birch, and phoenix tree respectively
(h1 : distance a b = distance a c)
(h2 : distance d a = distance d c)
(h3 : distance a b = 1)
(h4 : distance b c = 1)
(h5 : distance c d = 1)
(h6 : distance d e = 1)
(h7 : distance e b = 1) :
distance e d = 2 :=
by
  sorry

end tree_problem_l103_103663


namespace value_of_f_at_2_l103_103413

-- Given the conditions
variable (f : ℝ → ℝ)
variable (h_mono : Monotone f)
variable (h_cond : ∀ x : ℝ, f (f x - 3^x) = 4)

-- Define the proof goal
theorem value_of_f_at_2 : f 2 = 10 := 
sorry

end value_of_f_at_2_l103_103413


namespace factors_divisible_by_3_not_5_l103_103781

theorem factors_divisible_by_3_not_5 (n : ℕ) (h : n = 360) :
  let factors := (3 + 1) * (2 - 1 + 1) * (1 - 1 + 1) in
  factors = 8 :=
  by
    unfold factors
    sorry

end factors_divisible_by_3_not_5_l103_103781


namespace non_neg_solutions_l103_103361

theorem non_neg_solutions (x y z : ℕ) :
  (x^3 = 2 * y^2 - z) →
  (y^3 = 2 * z^2 - x) →
  (z^3 = 2 * x^2 - y) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by {
  sorry
}

end non_neg_solutions_l103_103361


namespace probability_angie_carlos_opposite_l103_103321

noncomputable def probability_opposite_seating (n : ℕ) : ℚ :=
  if n = 5 then 1 / 2 else 0

theorem probability_angie_carlos_opposite :
  probability_opposite_seating 5 = 1 / 2 :=
sorry


end probability_angie_carlos_opposite_l103_103321


namespace transport_problem_l103_103560

theorem transport_problem (x y z : ℕ) (hx : 2 ≤ x) (hxle : x ≤ 9)
    (hy : y = -2 * x + 20) (hz : z = 20 - x - y)
    (H : 6 * x + 5 * y + 4 * z = 100) (Ht : x + y + z = 20)
    (H_min : min_profit = 57200)
    : x = 2 ∧ y = 16 ∧ z = 2 ∧ (-1400 * x + 60000 = 57200) :=
sorry

end transport_problem_l103_103560


namespace solve_quadratic_equation_l103_103874

theorem solve_quadratic_equation :
  let y := x^2 + x in
  (y^2 - 4 * y - 12 = 0) ->
  (x = -3 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l103_103874


namespace locus_of_chord_midpoints_l103_103843

theorem locus_of_chord_midpoints (O P : Point) (K : Circle) (r d : ℝ)
  (h1 : O = K.center)
  (h2 : K.radius = r)
  (h3 : 0 < d)
  (h4 : d = dist O P)
  (h5 : d > r / 2)
  (h6 : d < r) :
  exists (C : Circle), 
  C.radius = d / 2 ∧ (∀ M : Point, (M ∈ C) → (exists (A B : Point), (chord K A B P M))) ∧ (arc_circle C < 360) :=
by
  sorry

end locus_of_chord_midpoints_l103_103843


namespace total_net_gain_computation_l103_103541

noncomputable def house1_initial_value : ℝ := 15000
noncomputable def house2_initial_value : ℝ := 20000

noncomputable def house1_selling_price : ℝ := 1.15 * house1_initial_value
noncomputable def house2_selling_price : ℝ := 1.2 * house2_initial_value

noncomputable def house1_buy_back_price : ℝ := 0.85 * house1_selling_price
noncomputable def house2_buy_back_price : ℝ := 0.8 * house2_selling_price

noncomputable def house1_profit : ℝ := house1_selling_price - house1_buy_back_price
noncomputable def house2_profit : ℝ := house2_selling_price - house2_buy_back_price

noncomputable def total_net_gain : ℝ := house1_profit + house2_profit

theorem total_net_gain_computation : total_net_gain = 7387.5 :=
by
  sorry

end total_net_gain_computation_l103_103541


namespace transport_problem_l103_103559

theorem transport_problem (x y z : ℕ) (hx : 2 ≤ x) (hxle : x ≤ 9)
    (hy : y = -2 * x + 20) (hz : z = 20 - x - y)
    (H : 6 * x + 5 * y + 4 * z = 100) (Ht : x + y + z = 20)
    (H_min : min_profit = 57200)
    : x = 2 ∧ y = 16 ∧ z = 2 ∧ (-1400 * x + 60000 = 57200) :=
sorry

end transport_problem_l103_103559


namespace total_ads_clicked_l103_103945

theorem total_ads_clicked (a1 a2 a3 a4 : ℕ) (clicked_ads : ℕ) :
  a1 = 12 →
  a2 = 2 * a1 →
  a3 = a2 + 24 →
  a4 = (3 * a2) / 4 →
  clicked_ads = (2 * (a1 + a2 + a3 + a4)) / 3 →
  clicked_ads = 68 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end total_ads_clicked_l103_103945


namespace max_value_fraction_l103_103875

open Real

-- Definitions of the conditions
def polynomial (a b c x : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + c

-- Main statement
theorem max_value_fraction (a b c λ x₁ x₂ x₃ : ℝ) (h₁ : λ > 0) 
  (h₂ : x₂ - x₁ = λ) (h₃ : x₃ > (x₁ + x₂) / 2)
  (h₄ : polynomial a b c x₁ = 0) (h₅ : polynomial a b c x₂ = 0)
  (h₆ : polynomial a b c x₃ = 0) :
  (2 * a^3 + 27 * c - 9 * a * b) / λ^3 ≤ 3 * sqrt 3 / 2 := sorry

end max_value_fraction_l103_103875


namespace nolan_monthly_savings_l103_103542

theorem nolan_monthly_savings (m k : ℕ) (H : 12 * m = 36 * k) : m = 3 * k := 
by sorry

end nolan_monthly_savings_l103_103542


namespace product_of_second_largest_and_second_smallest_l103_103610

theorem product_of_second_largest_and_second_smallest (l : List ℕ) (h : l = [10, 11, 12]) :
  l.nthLe 1 (by norm_num [h]) * l.nthLe 1 (by norm_num [h]) = 121 := 
sorry

end product_of_second_largest_and_second_smallest_l103_103610


namespace area_ratio_l103_103619

-- Define the basic setup for the equilateral triangle \(\triangle ABC\)
def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1, 0)
def C : point := (1/2, sqrt 3 / 2)

-- Define a function that describes the movement of a particle along the edges of the triangle over time
def particle_position (start : point) (t : ℝ) : point :=
  if t ≤ 1 then (start.1 * (1 - t) + B.1 * t, start.2 * (1 - t) + B.2 * t)  -- Moving from start to B
  else if t ≤ 2 then (B.1 * (2 - t) + C.1 * (t - 1), B.2 * (2 - t) + C.2 * (t - 1)) -- Moving from B to C
  else (C.1 * (3 - t) + A.1 * (t - 2), C.2 * (3 - t) + A.2 * (t - 2))  -- Moving from C to A

-- Define the midpoint of the two particles' positions
def midpoint (t : ℝ) : point :=
  let p1 := particle_position A t
  let p2 := particle_position B t
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the area of a triangle given its vertices
def triangle_area (p1 p2 p3 : point) : ℝ :=
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Define the vertices of the smaller triangle R formed by midpoints
def R1 : point := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def R2 : point := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def R3 : point := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)

-- Define the area of \( \triangle ABC \)
def area_ABC : ℝ := triangle_area A B C

-- Define the area of the region R
def area_R : ℝ := triangle_area R1 R2 R3

-- Theorem to prove the ratio of the areas is 1/4
theorem area_ratio : (area_R / area_ABC) = 1 / 4 :=
  sorry

end area_ratio_l103_103619


namespace power_of_2_e_q_l103_103152

noncomputable def q : ℝ :=
  ∑ k in Finset.range 8, (k + 1)^2 * Real.log (k + 1)

theorem power_of_2_e_q : Nat.findGreatest (fun n => 2^n ∣ Real.exp q) = 168 :=
by
  sorry

end power_of_2_e_q_l103_103152


namespace inequality_solution_l103_103750

theorem inequality_solution (x : ℝ) (h : ∀ (a b : ℝ) (ha : 0 < a) (hb : 0 < b), x^2 + x < a / b + b / a) : x ∈ Set.Ioo (-2 : ℝ) 1 := 
sorry

end inequality_solution_l103_103750


namespace eccentricity_of_hyperbola_l103_103432

-- Definitions using the given conditions.
variables {a b c : ℝ} (ha : a > 0) (hb : b > 0)
def hyperbola (x y : ℝ) : Prop := (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1
def point_B : ℝ × ℝ := (0, b)
def right_focus := (c, 0)
def perpendicular_slopes := (- b / c) * (b / a) = -1
def hyperbola_property := c ^ 2 = a ^ 2 + b ^ 2

-- Proof goal: Show that the eccentricity e satisfies e = (1 + sqrt(5)) / 2
theorem eccentricity_of_hyperbola : 
  (∃ e : ℝ, e = (1 + Real.sqrt 5) / 2 ∧ 
    (∃ x : ℝ, hyperbola x (e * x)) ∧ 
    perpendicular_slopes ∧ 
    hyperbola_property) := 
sorry

end eccentricity_of_hyperbola_l103_103432


namespace min_value_l103_103743

theorem min_value {x y a b : ℝ} (hx : 0 < x) (hy : 0 < y) (ha : 0 < a) (hb : 0 < b) (hxy : x + y = 1) :
    (∃ k : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 1 → (a / x + b / y) ≥ k) ∧ k = (√a + √b)^2) :=
by
  sorry

end min_value_l103_103743


namespace analytical_expression_monotonicity_on_pos_max_min_values_l103_103064

-- Part 1: Prove the analytical expression of f
theorem analytical_expression (a : ℝ) :
  (∀ x : ℝ, f x = a * x - 1 / x) → f (-2) = -3 / 2 → a = 1 ∧ (∀ x : ℝ, f x = x - 1 / x) := 
sorry

-- Part 2: Prove the monotonicity of f on (0, +∞)
theorem monotonicity_on_pos : 
  (∀ x : ℝ, f x = x - 1 / x) → ∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → f x1 < f x2 := 
sorry

-- Part 3: Prove the max and min values on [1/2, 2]
theorem max_min_values : 
  (∀ x : ℝ, f x = x - 1 / x) → (∀ x : ℝ, 1 / 2 ≤ x ∧ x ≤ 2 → f (1/2) = -3 / 2 ∧ f 2 = 3 / 2) := 
sorry

end analytical_expression_monotonicity_on_pos_max_min_values_l103_103064


namespace additional_toothpicks_to_complete_6_step_staircase_l103_103322

theorem additional_toothpicks_to_complete_6_step_staircase :
  ∀ (toothpicks_3 : ℕ) (toothpicks_4 : ℕ),
    toothpicks_3 = 18 →
    toothpicks_4 = 26 →
    (∀ n m, toothpicks_m - toothpicks_n = 8 + 3 * (m - n - 1)) →
    (let toothpicks_5 := toothpicks_4 + (8 + 3 * 1) in
     let toothpicks_6 := toothpicks_5 + (8 + 3 * 2) in
     toothpicks_6 - toothpicks_4 = 25) := sorry

end additional_toothpicks_to_complete_6_step_staircase_l103_103322


namespace find_angle_A_find_perimeter_l103_103805

-- Definitions extracted from the conditions
def sides (A B C a b c : ℝ) : Prop := 
  a + b + c > 0 ∧ A + B + C = π -- Sum of angles in a triangle is π

def vectors_parallel (a b cosA sinB : ℝ) : Prop := 
  a * sinB = sqrt(3) * b * cosA

def triangle_area_eq (a b c S : ℝ) := 
  1/2 * b * c * sin(π / 3) = S

def law_of_cosines (a b c : ℝ) := 
  a^2 = b^2 + c^2 - b * c

-- Problem statement in Lean
theorem find_angle_A (a b cosA sinB A : ℝ) 
  (h1 : vectors_parallel a b cosA sinB) : 
  A = π / 3 := 
sorry

theorem find_perimeter (a b c A B C S : ℝ) 
  (h1 : a = sqrt(7)) 
  (h2 : triangle_area_eq a b c S) 
  (h3 : S = 3 * sqrt(3) / 2) 
  (h4 : law_of_cosines a b c) 
  (h5 : sides A B C a b c) : 
  a + b + c = sqrt(7) + 5 := 
sorry

end find_angle_A_find_perimeter_l103_103805


namespace vec_min_value_eq_sqrt3div2_l103_103444

noncomputable def vec_min_value (a b : ℝ) (k : ℝ) : ℝ :=
  let dot_prod := -1 / 2
  let a2 := 1
  let b2 := 1
  let norm_sq := a2 - 2 * k * dot_prod + k^2 * b2
  real.sqrt norm_sq

theorem vec_min_value_eq_sqrt3div2 (a b : ℝ) (ha: a = 1) (hb: b = 1) : 
  (∃ k : ℝ, ∀ u v : ℝ, ⟪u, v⟫ = - 1 / 2 → vec_min_value a b k = real.sqrt (3 / 4)) :=
sorry

end vec_min_value_eq_sqrt3div2_l103_103444


namespace angle_of_inclination_of_line_l103_103951

theorem angle_of_inclination_of_line : ∀ (θ : ℝ), 
  (0 ≤ θ ∧ θ < 180) → (tan θ = -√3) → θ = 120 := 
by
  sorry

end angle_of_inclination_of_line_l103_103951


namespace find_f_prime_one_l103_103058

theorem find_f_prime_one (f : ℝ → ℝ) (f_prime_one : ℝ) :
  (∀ x > 0, f x = 2 * x * f_prime_one + Real.log x) →
  (f' 1 = -1) :=
by
  sorry

end find_f_prime_one_l103_103058


namespace radius_of_circle_from_chord_and_line_l103_103495

theorem radius_of_circle_from_chord_and_line (r : ℝ) (t θ : ℝ) 
    (param_line : ℝ × ℝ) (param_circle : ℝ × ℝ)
    (chord_length : ℝ) 
    (h1 : param_line = (3 + 3 * t, 1 - 4 * t))
    (h2 : param_circle = (r * Real.cos θ, r * Real.sin θ))
    (h3 : chord_length = 4) 
    : r = Real.sqrt 13 :=
sorry

end radius_of_circle_from_chord_and_line_l103_103495


namespace max_nonintersecting_diagonals_l103_103138

theorem max_nonintersecting_diagonals (n : ℕ) (h : n ≥ 3) : 
  ∃ m : ℕ, (if n % 2 = 1 then m = n - 3 else m = n - 2) ∧
  m = (finset.univ.image (λ i, (i : fin n → fin n × fin n)).filter (λ p, p.1 ≠ p.2 ∧ intersects_or_perpendicular p.1 p.2)).card := 
sorry

end max_nonintersecting_diagonals_l103_103138


namespace maximize_f_l103_103387

def S (n : ℕ+) := (n * (n + 1)) / 2

def f (n : ℕ+) : ℚ := S n / ((n + 32) * S (n + 1))

theorem maximize_f:
  ∃ n : ℕ+, f n = 1 / 50 :=
by
  use 8
  -- proof steps go here
  sorry

end maximize_f_l103_103387


namespace initial_birds_was_one_l103_103612

def initial_birds (b : Nat) : Prop :=
  b + 4 = 5

theorem initial_birds_was_one : ∃ b, initial_birds b ∧ b = 1 :=
by
  use 1
  unfold initial_birds
  sorry

end initial_birds_was_one_l103_103612


namespace number_of_song_liking_patterns_l103_103575

-- Define the sets and conditions
structure SongLikingPattern where
  AB : Finset ℕ -- Songs liked by Sara and Mia but not Lily
  BC : Finset ℕ -- Songs liked by Mia and Lily but not Sara
  CA : Finset ℕ -- Songs liked by Lily and Sara but not Mia
  A : Finset ℕ  -- Songs liked only by Sara
  B : Finset ℕ  -- Songs liked only by Mia
  C : Finset ℕ  -- Songs liked only by Lily
  N : Finset ℕ  -- Songs liked by none

-- Conditions
def song_liking_conditions : Prop :=
  ∀ (s : SongLikingPattern),
    s.AB ∩ s.BC ∩ s.CA = ∅ ∧
    s.AB ∪ s.BC ∪ s.CA = {0, 1, 2} ∧
    ({s.A, s.B, s.C, s.N}.count Finset.empty = 2)

-- Prove the number of distinct song liking patterns
theorem number_of_song_liking_patterns : ∃ (n : ℕ), song_liking_conditions ∧ n = 30 :=
by
  existsi 30
  sorry -- Proof is omitted as per the instructions.

end number_of_song_liking_patterns_l103_103575


namespace odd_integers_count_l103_103782

theorem odd_integers_count : 
  let lower_bound := (25 : ℚ) / 6
  let upper_bound := (47 : ℚ) / 3
  ∃ (count : ℕ), count = 6 ∧ ∀ (n : ℤ), (lower_bound < (n : ℚ) ∧ (n : ℚ) < upper_bound) → (2 ∣ n = false) :=
by
  let lower_bound := (25 : ℚ) / 6
  let upper_bound := (47 : ℚ) / 3
  have : ∀ (n : ℤ), (lower_bound < (n : ℚ) ∧ (n : ℚ) < upper_bound) → (2 ∣ n = false),
    sorry -- Proof skipped
  existsi 6,
  split,
  sorry, -- Proof for count = 6
  assumption

end odd_integers_count_l103_103782


namespace best_fit_model_l103_103261

-- Definition of the given R^2 values for different models
def R2_A : ℝ := 0.62
def R2_B : ℝ := 0.63
def R2_C : ℝ := 0.68
def R2_D : ℝ := 0.65

-- Theorem statement that model with R2_C has the best fitting effect
theorem best_fit_model : R2_C = max R2_A (max R2_B (max R2_C R2_D)) :=
by
  sorry -- Proof is not required

end best_fit_model_l103_103261


namespace race_order_l103_103812

theorem race_order (h₁ : nat.mod 9 2 = 1)
                   (h₂ : nat.mod 8 2 = 0)
                   (h₃ : ∀ n : ℕ, n % 2 = 1 ∨ n % 2 = 0):
  -- Rewrite the conditions as Lean hypotheses:
  -- h₁: A and B exchanged positions 9 times (odd number)
  -- h₂: B and C exchanged positions 8 times (even number)
  -- h₃: Any number of exchanges between cars results in the relative positional rule
  
  -- Final Order: B A C D
  final_order = ["B", "A", "C", "D"] := 
begin 
  sorry 
end

end race_order_l103_103812


namespace adam_total_figurines_l103_103998

def figurines_created (basswood_blocks butternut_wood_blocks aspen_wood_blocks : ℕ) : ℕ :=
  let basswood_fig := basswood_blocks * 3
  let butternut_fig := butternut_wood_blocks * 4
  let aspen_fig := aspen_wood_blocks * (3 * 2)
  basswood_fig + butternut_fig + aspen_fig

theorem adam_total_figurines : figurines_created 15 20 20 = 245 :=
by
  unfold figurines_created
  norm_num
  sorry

end adam_total_figurines_l103_103998


namespace jonah_fishes_per_day_l103_103106

theorem jonah_fishes_per_day (J G J_total : ℕ) (days : ℕ) (total : ℕ)
  (hJ : J = 6) (hG : G = 8) (hdays : days = 5) (htotal : total = 90) 
  (fish_total : days * J + days * G + days * J_total = total) : 
  J_total = 4 :=
by
  sorry

end jonah_fishes_per_day_l103_103106


namespace total_volume_of_four_cubes_is_500_l103_103254

-- Definitions for the problem assumptions
def edge_length := 5
def volume_of_cube (edge_length : ℕ) := edge_length ^ 3
def number_of_boxes := 4

-- Main statement to prove
theorem total_volume_of_four_cubes_is_500 :
  (volume_of_cube edge_length) * number_of_boxes = 500 :=
by
  -- Proof steps will go here
  sorry

end total_volume_of_four_cubes_is_500_l103_103254


namespace part1_part2_l103_103419

section
variable (k : ℝ)

/-- Part 1: Range of k -/
def discriminant_eqn (k : ℝ) := (2 * k - 1) ^ 2 - 4 * (k ^ 2 - 1)

theorem part1 (h : discriminant_eqn k ≥ 0) : k ≤ 5 / 4 :=
by sorry

/-- Part 2: Value of k when x₁ and x₂ satisfy the given condition -/
def x1_x2_eqn (k x1 x2 : ℝ) := x1 ^ 2 + x2 ^ 2 = 16 + x1 * x2

def vieta (k : ℝ) (x1 x2 : ℝ) :=
  x1 + x2 = 1 - 2 * k ∧ x1 * x2 = k ^ 2 - 1

theorem part2 (x1 x2 : ℝ) (h1 : vieta k x1 x2) (h2 : x1_x2_eqn k x1 x2) : k = -2 :=
by sorry

end

end part1_part2_l103_103419


namespace vertical_asymptotes_at_2_and_3_l103_103798

def y (x : ℝ) : ℝ := (x^2 + 3*x + 9) / (x^2 - 5*x + 6)

theorem vertical_asymptotes_at_2_and_3 : 
  (∀ x : ℝ, x = 2 → (x^2 - 5*x + 6) = 0 ∧ (x^2 + 3*x + 9) ≠ 0)
  ∧ 
  (∀ x : ℝ, x = 3 → (x^2 - 5*x + 6) = 0 ∧ (x^2 + 3*x + 9) ≠ 0) := 
by 
  -- Proof goes here
  sorry

end vertical_asymptotes_at_2_and_3_l103_103798


namespace log2_f4_eq_1_l103_103053

-- Define the power function with unknown exponent α
noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ :=
  x ^ α

-- Given that the function passes through (1/2, sqrt(2)/2)
def condition1 (α : ℝ) : Prop :=
  power_function α (1/2) = sqrt(2)/2

-- Define the function f(x) = sqrt(x)
def f (x : ℝ) : ℝ :=
  sqrt x

-- Our goal is to prove log2 (f 4) = 1
theorem log2_f4_eq_1 (α : ℝ) (h : condition1 α) : 
  Real.logb 2 (f 4) = 1 :=
by 
  sorry

end log2_f4_eq_1_l103_103053


namespace center_of_tangent_circle_l103_103653

theorem center_of_tangent_circle 
  (tangent1 : ∀ x y : ℝ, 3 * x + 4 * y = 24 → False)
  (tangent2 : ∀ x y : ℝ, 3 * x + 4 * y = -16 → False)
  (center_line : ∀ x y : ℝ, x - 3 * y = 0 → False) :
  ∃ (x y : ℝ), x = 12 / 13 ∧ y = 4 / 13 ∧ (3 * x + 4 * y = 4 ∧ x - 3 * y = 0) := 
begin
  sorry
end

end center_of_tangent_circle_l103_103653


namespace bertha_daughters_no_daughters_l103_103327

theorem bertha_daughters_no_daughters (daughters granddaughters: ℕ) (no_great_granddaughters: granddaughters = 5 * daughters) (total_women: 8 + granddaughters = 48) :
  8 + granddaughters = 48 :=
by {
  sorry
}

end bertha_daughters_no_daughters_l103_103327


namespace power_complex_l103_103710

theorem power_complex (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : -64 = (-4)^3) (h3 : (a^b)^((3:ℝ) / 2) = a^(b * ((3:ℝ) / 2))) (h4 : (-4:ℂ)^(1/2) = 2 * i) :
  (↑(-64):ℂ) ^ (3/2) = 512 * i :=
by
  sorry

end power_complex_l103_103710


namespace polynomial_divisibility_l103_103214

theorem polynomial_divisibility (C D : ℝ) (h : ∀ (ω : ℂ), ω^2 + ω + 1 = 0 → (ω^106 + C * ω + D = 0)) : C + D = -1 :=
by
  -- Add proof here
  sorry

end polynomial_divisibility_l103_103214


namespace Alice_min_score_l103_103678

theorem Alice_min_score (score1 score2 score3 : ℕ) (max_score : ℕ) (target_avg : ℚ) (remaining_tests : ℕ) :
  score1 = 85 → score2 = 76 → score3 = 83 → max_score = 100 → target_avg = 80 → remaining_tests = 2 →
  let total_needed := target_avg * (3 + remaining_tests)
  let current_total := score1 + score2 + score3
  let needed_on_remaining := total_needed - current_total
  let max_possible_score := max_score
  needed_on_remaining - max_possible_score = 56 :=
by
  intros
  rw [score1, score2, score3, max_score, target_avg, remaining_tests]
  let total_needed := 80 * 5
  let current_total := 85 + 76 + 83
  let needed_on_remaining := total_needed - current_total
  let max_possible_score := 100
  let min_score_needed := needed_on_remaining - max_possible_score
  have : min_score_needed = 56 := by sorry
  exact this

end Alice_min_score_l103_103678


namespace scientific_notation_1570000000_l103_103958

def scientific_notation (n : ℕ) : ℚ :=
  n / 10^(nat.floor (real.log10 n))

theorem scientific_notation_1570000000 :
  scientific_notation 1570000000 = 1.57 * 10^9 :=
by
  -- Placeholder for the actual proof
  sorry

end scientific_notation_1570000000_l103_103958


namespace sufficient_but_not_necessary_l103_103738

theorem sufficient_but_not_necessary (x : ℝ) : (x > 2) → (x^2 > 4) ∧ ¬(∀ x, x^2 > 4 → x > 2) :=
by
  intro h
  have h₁ : x^2 > 4 := by sorry
  have h₂ : ¬(∀ x, x^2 > 4 → x > 2) := by sorry
  exact ⟨h₁, h₂⟩
  sorry

end sufficient_but_not_necessary_l103_103738


namespace pure_imaginary_complex_number_l103_103462

theorem pure_imaginary_complex_number (m : ℝ) (h : (m^2 - 3*m) = 0) :
  (m^2 - 5*m + 6) ≠ 0 → m = 0 :=
by
  intro h_im
  have h_fact : (m = 0) ∨ (m = 3) := by
    sorry -- This is where the factorization steps would go
  cases h_fact with
  | inl h0 =>
    assumption
  | inr h3 =>
    exfalso
    have : (3^2 - 5*3 + 6) = 0 := by
      sorry -- Simplify to check that m = 3 is not a valid solution
    contradiction

end pure_imaginary_complex_number_l103_103462


namespace total_arrangements_l103_103655

def total_members : ℕ := 6
def days : ℕ := 3
def people_per_day : ℕ := 2

def A_cannot_on_14 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 14 = 1

def B_cannot_on_16 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 16 = 2

theorem total_arrangements (arrangement : ℕ → ℕ) :
  (∀ arrangement, A_cannot_on_14 arrangement ∧ B_cannot_on_16 arrangement) →
  (total_members.choose 2 * (total_members - 2).choose 2 - 
  2 * (total_members - 1).choose 1 * (total_members - 2).choose 2 +
  (total_members - 2).choose 1 * (total_members - 3).choose 1)
  = 42 := 
by
  sorry

end total_arrangements_l103_103655


namespace percentage_increase_is_30_l103_103878

noncomputable def cups_sold_last_week : ℕ := 20
noncomputable def total_cups_sold : ℕ := 46

noncomputable def cups_sold_this_week : ℕ := total_cups_sold - cups_sold_last_week

theorem percentage_increase_is_30 :
  ((cups_sold_this_week - cups_sold_last_week) / cups_sold_last_week.toFloat) * 100 = 30 :=
by 
  sorry

end percentage_increase_is_30_l103_103878


namespace binary_to_decimal_l103_103694

theorem binary_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5) :=
by
  sorry

end binary_to_decimal_l103_103694


namespace PQ_PR_QR_div_l103_103850

theorem PQ_PR_QR_div (p q r : ℝ)
    (midQR : p = 0) (midPR : q = 0) (midPQ : r = 0) :
    (4 * (q ^ 2 + r ^ 2) + 4 * (p ^ 2 + r ^ 2) + 4 * (p ^ 2 + q ^ 2)) / (p ^ 2 + q ^ 2 + r ^ 2) = 8 :=
by {
    sorry
}

end PQ_PR_QR_div_l103_103850


namespace zeros_in_square_of_number_l103_103086

theorem zeros_in_square_of_number (n : ℕ) (h : n = 10^11 - 2) : 
  let m := n^2
  number_of_zeros_at_end m = 10 :=
sorry

end zeros_in_square_of_number_l103_103086


namespace inequality_xy_l103_103544

theorem inequality_xy {x y : ℝ} (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_xy_l103_103544


namespace work_together_days_l103_103971

def total_work : ℝ := 1 -- Let's assume the total work W is normalized to 1

def rate_A : ℝ := total_work / 28 -- A's rate

def rate_AB (D_AB : ℝ) : ℝ := total_work / D_AB -- A and B's combined rate

theorem work_together_days :
  ∃ D_AB : ℝ, (rate_AB(D_AB) * 10) + (rate_A * 21) = total_work ∧ D_AB = 40 := by
  sorry

end work_together_days_l103_103971


namespace solution_set_l103_103451

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x - 1 else sorry -- The exact definition for negative x isn't necessary for the proof.

lemma even_function (x : ℝ) : f (-x) = f x :=
sorry -- Placeholder for property of even functions.

theorem solution_set {x : ℝ} (hx : 0 ≤ x → f x = x - 1) :
  {x : ℝ | f (x - 1) < 0} = set.Ioo 0 2 :=
sorry

end solution_set_l103_103451


namespace number_of_integers_containing_3_and_4_l103_103082

open Nat

/--
Define the conditions for the problem:
1. The number must be between 400 and 900.
2. The number must contain the digit 3.
3. The number must contain the digit 4.
4. Neither 3 nor 4 can be the first digit.
-/
def satisfies_conditions (n : ℕ) : Prop :=
  n >= 400 ∧ n < 900 ∧ 
  ((n / 10) % 10 = 3 ∨ (n / 10) % 10 = 4 ∨ n % 10 = 3 ∨ n % 10 = 4) ∧
  ((n / 100) % 10 ≠ 3 ∧ (n / 100) % 10 ≠ 4) ∧
  (((n / 100) % 10 = 4 ∧ ((n / 10) % 10 = 3 ∨ n % 10 = 3)) ∨
   ((n / 100) % 10 > 4 ∧ (n % 100 = 34 ∨ n % 100 = 43)))

-- The number of such integers is 10.
theorem number_of_integers_containing_3_and_4 : 
  ∃! n, satisfies_conditions n ∧ (card (filter (satisfies_conditions) (range (900 - 400) + 1) = 10) :=
sorry

end number_of_integers_containing_3_and_4_l103_103082


namespace evaluate_expression_l103_103357

section
noncomputable def sum_of_cubes : ℕ → ℤ
| 0       := 0
| (n + 1) := (n + 1 : ℤ)^3 + sum_of_cubes n

noncomputable def sum_of_cubes_neg : ℕ → ℤ
| 0       := 0
| (n + 1) := (-(n + 1 : ℤ))^3 + sum_of_cubes_neg n

theorem evaluate_expression : 
  (sum_of_cubes 51 + sum_of_cubes_neg 51) - (∑ k in finset.range 51, (2 * (k + 1))^2) = -290544 := by 
  sorry 
end

end evaluate_expression_l103_103357


namespace neg_rational_is_rational_l103_103793

theorem neg_rational_is_rational (m : ℚ) : -m ∈ ℚ :=
sorry

end neg_rational_is_rational_l103_103793


namespace twelve_xy_leq_fourx_1_y_9y_1_x_l103_103551

theorem twelve_xy_leq_fourx_1_y_9y_1_x
  (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) :=
  sorry

end twelve_xy_leq_fourx_1_y_9y_1_x_l103_103551


namespace function_form_l103_103360

noncomputable def f : ℝ → ℝ := sorry

theorem function_form (c : ℝ) :
  (∀ x y : ℝ, f(f(x) + y) = x + f(f(y))) →
  (∀ x : ℝ, ∃ c : ℝ, f(x) = x + c) := 
sorry

end function_form_l103_103360


namespace ratio_AD_DC_l103_103478

-- Define the conditions in the problem
variables {A B C D : Type}
variables (AD DC : ℝ)
variable [triangle : Triangle A B C]
variable [segment_AD : Segment A D]
variable [segment_AC : Segment A C]
variable [D_on_AC : OnSegment D A C]
variable [AB_length : SegmentLength A B = 6]
variable [BC_length : SegmentLength B C = 8]
variable [AC_length : SegmentLength A C = 10]
variable [BD_length : SegmentLength B D = 7]

-- Define the proof problem: Prove the ratio of AD to DC is 3/2
theorem ratio_AD_DC : AD / DC = 3 / 2 :=
sorry

end ratio_AD_DC_l103_103478


namespace set_eq_interval_l103_103910

-- Assume the set is defined as all x such that x <= 1
def set_def : Set ℝ := {x | x ≤ 1}

-- Define the interval equivalent
def interval_def : Set ℝ := Iic 1 -- Iic 1 represents the interval (-∞, 1] in Lean

-- Statement of equivalence between the set and the interval
theorem set_eq_interval : set_def = interval_def := by
  sorry

end set_eq_interval_l103_103910


namespace domain_of_f_when_a_is_3_max_value_of_a_for_inequality_l103_103757

noncomputable def f (x : ℝ) (a : ℝ) := log 2 (|x + 1| + |x - 1| - a)

theorem domain_of_f_when_a_is_3 :
  { x : ℝ | x < -3/2 ∨ x > 3/2 } = { x : ℝ | f x 3 ≠ 0 } :=
by
  sorry

theorem max_value_of_a_for_inequality :
  (∀ x : ℝ, f x a ≥ 2) → a ≤ -2 :=
by
  sorry

end domain_of_f_when_a_is_3_max_value_of_a_for_inequality_l103_103757


namespace triangle_alpha_eq_3beta_iff_c_eq_l103_103600

theorem triangle_alpha_eq_3beta_iff_c_eq (a b c : ℝ) (α β γ : ℝ) :
  α = 3 * β ↔ c = (a - b) * real.sqrt(1 + a / b) :=
sorry

end triangle_alpha_eq_3beta_iff_c_eq_l103_103600


namespace problem_I_problem_II_l103_103761

namespace ProofProblems

def f (x a : ℝ) : ℝ := |x - a| + |x + 5|

theorem problem_I (x : ℝ) : (f x 1) ≥ 2 * |x + 5| ↔ x ≤ -2 := 
by sorry

theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, (f x a) ≥ 8) ↔ (a ≥ 3 ∨ a ≤ -13) := 
by sorry

end ProofProblems

end problem_I_problem_II_l103_103761


namespace Tanya_days_correct_l103_103639

-- Definitions based on the given conditions:
def efficiency_factor_tanya : ℝ := 1.25
def days_sakshi : ℕ := 20

-- Calculate the number of days Tanya takes to complete the work
def days_tanya : ℝ := days_sakshi / efficiency_factor_tanya

-- The theorem to be proved
theorem Tanya_days_correct : days_tanya = 16 :=
by
  -- Skipping the proof as requested
  sorry

end Tanya_days_correct_l103_103639


namespace christina_payment_l103_103334

theorem christina_payment :
  let pay_flowers_per_flower := (8 : ℚ) / 3
  let pay_lawn_per_meter := (5 : ℚ) / 2
  let num_flowers := (9 : ℚ) / 4
  let area_lawn := (7 : ℚ) / 3
  let total_payment := pay_flowers_per_flower * num_flowers + pay_lawn_per_meter * area_lawn
  total_payment = 71 / 6 :=
by
  sorry

end christina_payment_l103_103334


namespace scholarship_distribution_l103_103371

theorem scholarship_distribution (total_students : ℕ)
  (perc_full_scholarship perc_half_scholarship : ℝ)
  (students_with_full_scholarship students_with_half_scholarship : ℕ)
  (students_without_scholarship : ℕ)
  (h1 : total_students = 300)
  (h2 : perc_full_scholarship = 0.05)
  (h3 : perc_half_scholarship = 0.10)
  (h4 : students_with_full_scholarship = int.of_nat (perc_full_scholarship * total_students))
  (h5 : students_with_half_scholarship = int.of_nat (perc_half_scholarship * total_students))
  (h6 : students_without_scholarship = total_students - (students_with_full_scholarship + students_with_half_scholarship)) :
  students_without_scholarship = 255 :=
sorry

end scholarship_distribution_l103_103371


namespace target_run_correct_l103_103823

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_10 : ℝ := 10
def run_rate_remaining_22_overs : ℝ := 11.363636363636363
def overs_remaining_22 : ℝ := 22

-- Initialize the target run calculation using the given conditions
def runs_first_10_overs := overs_first_10 * run_rate_first_10_overs
def runs_remaining_22_overs := overs_remaining_22 * run_rate_remaining_22_overs
def target_run := runs_first_10_overs + runs_remaining_22_overs 

-- The goal is to prove that the target run is 282
theorem target_run_correct : target_run = 282 := by
  sorry  -- The proof is not required as per the instructions.

end target_run_correct_l103_103823


namespace bx_in_terms_of_R_phi_psi_l103_103171

theorem bx_in_terms_of_R_phi_psi
  (R : ℝ) (φ ψ : ℝ)
  (h1 : AB_diameter : ℝ) 
  (h2 : C_on_circle : Prop)
  (h3 : D_on_circle : Prop)
  (h4 : CD_intersects_tangent_at_X : Prop) :
  BX = (2 * R * real.sin φ * real.sin ψ) / real.sin (|φ ± ψ|) := 
sorry

end bx_in_terms_of_R_phi_psi_l103_103171


namespace musketeer_statements_triplets_count_l103_103509

-- Definitions based on the conditions
def musketeers : Type := { x : ℕ // x < 3 }

def is_guilty (m : musketeers) : Prop := sorry  -- Placeholder for the property of being guilty

def statement (m1 m2 : musketeers) : Prop := sorry  -- Placeholder for the statement made by one musketeer about another

-- Condition that each musketeer makes one statement
def made_statement (m : musketeers) : Prop := sorry

-- Condition that exactly one musketeer lied
def exactly_one_lied : Prop := sorry

-- The final proof problem statement:
theorem musketeer_statements_triplets_count : ∃ n : ℕ, n = 99 :=
  sorry

end musketeer_statements_triplets_count_l103_103509


namespace squared_difference_l103_103446

theorem squared_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 10) : (x - y)^2 = 24 :=
by
  sorry

end squared_difference_l103_103446


namespace relationship_between_a_b_c_l103_103730

noncomputable def a : ℝ := Real.cos 1
noncomputable def b : ℝ := Real.log (Real.cos 1)
noncomputable def c : ℝ := Real.exp (Real.cos 1)

theorem relationship_between_a_b_c : b < a ∧ a < c :=
by
  -- Use the initial conditions
  have h_a : 0 < a ∧ a < 1 :=
    by sorry -- This corresponds to 0 < Real.cos 1 < 1

  have h_b : b < 0 :=
    by sorry -- This corresponds to Real.log (Real.cos 1) < 0 

  have h_c : 1 < c :=
    by sorry -- This corresponds to Real.exp (Real.cos 1) > 1 

  -- Now we combine these results
  exact ⟨h_b.right_of h_a.left, h_a.right.trans h_c.left⟩

end relationship_between_a_b_c_l103_103730


namespace true_propositions_l103_103211

namespace ProofProblem

variables {α : Type*} [Fintype α] (A B : Finset α)

-- Define the cardinality (number of elements) of a finite set
def card (S : Finset α) : ℕ := S.card

-- Proposition 1: A ∩ B = ∅ ↔ card (A ∪ B) = card A + card B
def prop1 : Prop :=
  A ∩ B = ∅ ↔ card (A ∪ B) = card A + card B

-- Proposition 2: A ⊆ B → card A ≤ card B
def prop2 : Prop :=
  A ⊆ B → card A ≤ card B

-- Proposition 3: A ⊈ B ↔ card A ≤ card B
def prop3 : Prop :=
  A ⊈ B ↔ card A ≤ card B

-- Proposition 4: A = B ↔ card A = card B
def prop4 : Prop :=
  A = B ↔ card A = card B

-- The main theorem stating that prop1 and prop2 are true while prop3 and prop4 are false
theorem true_propositions : (prop1 A B) ∧ (prop2 A B) ∧ ¬(prop3 A B) ∧ ¬(prop4 A B) :=
  sorry

end ProofProblem

end true_propositions_l103_103211


namespace find_k_inverse_proportion_l103_103208

theorem find_k_inverse_proportion :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) → (y = k / x)) ∧ k = 3 :=
by
  sorry

end find_k_inverse_proportion_l103_103208


namespace gcd_n_cube_plus_16_n_plus_4_l103_103373

theorem gcd_n_cube_plus_16_n_plus_4 (n : ℕ) (h1 : n > 16) : 
  Nat.gcd (n^3 + 16) (n + 4) = Nat.gcd 48 (n + 4) :=
by
  sorry

end gcd_n_cube_plus_16_n_plus_4_l103_103373


namespace part1_part2_l103_103166

noncomputable def f_n (n : ℕ+) (x : ℝ) : ℝ := 
  -1 + x + ∑ k in finset.range n, (x^(k+1) / (k+1)^2)

theorem part1 (n : ℕ+) : ∃! x ∈ Icc (2/3 : ℝ) 1, f_n n x = 0 := sorry

theorem part2 (p : ℕ+) (h : ∀ n : ℕ+, ∃! x ∈ Icc (2/3 : ℝ) 1, f_n n x = 0) 
  (x_n : ℕ+ → ℝ) (hx_n : ∀ n, (h n).some = x_n n) : 
  ∀ n : ℕ+, 0 < x_n n - x_n (n + p) ∧ x_n n - x_n (n + p) < 1 / n := sorry

end part1_part2_l103_103166


namespace sqrt_defined_iff_nonneg_l103_103475

theorem sqrt_defined_iff_nonneg (x : ℝ) : (∃ y, y = sqrt (x - 2)) ↔ x ≥ 2 :=
by
  sorry

end sqrt_defined_iff_nonneg_l103_103475


namespace incorrect_description_is_A_l103_103262

-- Definitions for the conditions
def description_A := "Increasing the concentration of reactants increases the percentage of activated molecules, accelerating the reaction rate."
def description_B := "Increasing the pressure of a gaseous reaction system increases the number of activated molecules per unit volume, accelerating the rate of the gas reaction."
def description_C := "Raising the temperature of the reaction increases the percentage of activated molecules, increases the probability of effective collisions, and increases the reaction rate."
def description_D := "Catalysts increase the reaction rate by changing the reaction path and lowering the activation energy required for the reaction."

-- Problem Statement
theorem incorrect_description_is_A :
  description_A ≠ correct :=
  sorry

end incorrect_description_is_A_l103_103262


namespace eleven_pow_2023_mod_eight_l103_103247

theorem eleven_pow_2023_mod_eight (h11 : 11 % 8 = 3) (h3 : 3^2 % 8 = 1) : 11^2023 % 8 = 3 :=
by
  sorry

end eleven_pow_2023_mod_eight_l103_103247


namespace least_positive_integer_divisible_by_primes_gt_5_l103_103240

theorem least_positive_integer_divisible_by_primes_gt_5 : ∃ n : ℕ, n = 7 * 11 * 13 ∧ ∀ k : ℕ, (k > 0 ∧ (k % 7 = 0) ∧ (k % 11 = 0) ∧ (k % 13 = 0)) → k ≥ 1001 := 
sorry

end least_positive_integer_divisible_by_primes_gt_5_l103_103240


namespace inequality_proof_l103_103548

variable (x y : ℝ)
variable (h1 : x ≥ 0)
variable (h2 : y ≥ 0)
variable (h3 : x + y ≤ 1)

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) : 
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := 
by 
  sorry

end inequality_proof_l103_103548


namespace minimum_votes_l103_103705

theorem minimum_votes (n_trainees : ℕ) (n_votes_per_trainee : ℕ) (condition : ∀ (s1 s2 : fin n_trainees), s1 ≠ s2 → ∃ aid, (votes s1 aid) ∧ (votes s2 aid)) (v total_votes) 
  (h1 : n_trainees = 20) 
  (h2 : n_votes_per_trainee = 2)
  (h3 : total_votes = n_trainees * n_votes_per_trainee) 
  (h4 : ∀ (aid : fin total_votes), aid = 14) : 
  ∃ aid, votes ?a aid := 
sorry

end minimum_votes_l103_103705


namespace symmetric_points_sum_l103_103049

-- Define the points and symmetry conditions
def point_A (m : ℝ) : ℝ × ℝ := (m, 1)
def point_B (n : ℝ) : ℝ × ℝ := (2, n)

-- Define the condition for symmetry with respect to the x-axis
def symmetric_about_x (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

-- Assume the given conditions and prove m + n = 1
theorem symmetric_points_sum (m n : ℝ) 
  (h_sym : symmetric_about_x (point_A m) (point_B n)) : 
  m + n = 1 :=
by
  -- Extract the symmetry conditions from the hypothesis
  cases h_sym with h1 h2
  -- Use the conditions m = 2 and n = -1 to conclude
  sorry

end symmetric_points_sum_l103_103049


namespace range_of_m_l103_103074

open Set

def set_A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (3 * m - 2)}

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ m ≤ 4 :=
by sorry

end range_of_m_l103_103074


namespace milk_purchase_l103_103972

theorem milk_purchase : 
  ∀ (milk_cost buy_5_get_1_free total_money : ℕ),
  (milk_cost = 3) →
  (buy_5_get_1_free = 6) →
  (total_money = 20) →
  ∃ (bags_bought remaining_money : ℕ), 
  (bags_bought = 7) ∧ (remaining_money = 2) :=
by
  intros milk_cost buy_5_get_1_free total_money
  intros h1 h2 h3
  use 7
  use 2
  split
  · sorry
  · sorry

end milk_purchase_l103_103972


namespace intersection_product_l103_103067

noncomputable def line (t : ℝ) : ℝ × ℝ :=
(5 + (Real.sqrt 3 / 2) * t, Real.sqrt 3 + (1 / 2) * t)

def curve (θ : ℝ) : ℝ × ℝ :=
(2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)

def point_M : ℝ × ℝ := (5, Real.sqrt 3)

theorem intersection_product :
  ∀ (A B : ℝ × ℝ), A = (5 + (Real.sqrt 3 / 2) * t, Real.sqrt 3 + (1 / 2) * t)
  → B = (5 + (Real.sqrt 3 / 2) * t, Real.sqrt 3 + (1 / 2) * t)
  → ((A.1 - 5)^2 + (A.2 - Real.sqrt 3)^2) * ((B.1 - 5)^2 + (B.2 - Real.sqrt.3)^2) = 18 := 
sorry

end intersection_product_l103_103067


namespace planar_graph_edges_planar_triangulation_edges_l103_103659

theorem planar_graph_edges (n : ℕ) (h : n ≥ 3) :
  ∀ (G : Type) [planar_graph G] (v : G → ℕ) (e : G → ℕ), e G ≤ 3 * v G - 6 :=
sorry

theorem planar_triangulation_edges (n : ℕ) (h : n ≥ 3) :
  ∀ (G : Type) [planar_triangulation G] (v : G → ℕ) (e : G → ℕ), e G = 3 * v G - 6 :=
sorry

end planar_graph_edges_planar_triangulation_edges_l103_103659


namespace solve_equation_l103_103581

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) → x = -4 :=
by
  intro hyp
  sorry

end solve_equation_l103_103581


namespace veromont_clicked_ads_l103_103943

def ads_on_first_page := 12
def ads_on_second_page := 2 * ads_on_first_page
def ads_on_third_page := ads_on_second_page + 24
def ads_on_fourth_page := (3 / 4) * ads_on_second_page
def total_ads := ads_on_first_page + ads_on_second_page + ads_on_third_page + ads_on_fourth_page
def ads_clicked := (2 / 3) * total_ads

theorem veromont_clicked_ads : ads_clicked = 68 := 
by
  sorry

end veromont_clicked_ads_l103_103943


namespace log_arith_example_l103_103279

noncomputable def log10 (x : ℝ) : ℝ := sorry -- Assume the definition of log base 10

theorem log_arith_example : log10 4 + 2 * log10 5 + 8^(2/3) = 6 := 
by
  -- The proof would go here
  sorry

end log_arith_example_l103_103279


namespace abs_z_sq_eq_841_div_9_l103_103530

noncomputable def z (a b : ℝ) : ℂ := a + b * complex.I
noncomputable def abs_z (a b : ℝ) : ℝ := complex.abs (z a b)
noncomputable def abs_z_sq (a b : ℝ) : ℝ := abs_z a b * abs_z a b

theorem abs_z_sq_eq_841_div_9 (a b : ℝ) (h_reim : z a b + abs_z a b = 3 + 7 * complex.I) (ha_re : z a b).re + abs_z a b = 3) 
  (hb_im : (z a b).im = 7) : abs_z_sq a b = 841 / 9 :=
by
  sorry

end abs_z_sq_eq_841_div_9_l103_103530


namespace smallest_positive_period_min_value_interval_l103_103759

-- Definition for the function f(x)
def f (x : ℝ) : ℝ := sin (2 * x) + 2 * cos x ^ 2 - 1

-- Statement for the smallest positive period
theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x :=
by
  sorry

-- Statement for the minimum value in the interval [0, π/2]
theorem min_value_interval : ∃ x ∈ Icc (0 : ℝ) (π / 2), f x = -1 :=
by
  sorry

end smallest_positive_period_min_value_interval_l103_103759


namespace simplify_and_evaluate_expr_l103_103187

noncomputable def a : ℝ := 3 + real.sqrt 3

theorem simplify_and_evaluate_expr : 
  (1 - (1 / (a - 2))) / ((a^2 - 6 * a + 9) / (a^2 - 2 * a)) = real.sqrt 3 + 1 :=
by
  sorry

end simplify_and_evaluate_expr_l103_103187


namespace calculate_green_toad_densities_l103_103589

-- Define the given ratios and densities
structure ToadDensities where
  green_per_brown : ℚ
  green_per_blue : ℚ
  green_per_red : ℚ
  spotted_brown_ratio : ℚ
  plain_brown_ratio : ℚ
  striped_blue_ratio : ℚ
  camouflaged_blue_ratio : ℚ
  star_red_ratio : ℚ
  spotted_red_ratio : ℚ
  wetlands_spotted_brown_density : ℚ
  forests_camouflaged_blue_density : ℚ
  grasslands_star_red_density : ℚ
  marshlands_plain_brown_density : ℚ
  shrublands_striped_blue_density : ℚ

-- Introduce the instance with given values
def given_ratios : ToadDensities := {
  green_per_brown := 1 / 25,
  green_per_blue := 1 / 10,
  green_per_red := 1 / 20,
  spotted_brown_ratio := 1 / 4,
  plain_brown_ratio := 3 / 4,
  striped_blue_ratio := 1 / 3,
  camouflaged_blue_ratio := 2 / 3,
  star_red_ratio := 1 / 2,
  spotted_red_ratio := 1 / 2,
  wetlands_spotted_brown_density := 60,
  forests_camouflaged_blue_density := 45,
  grasslands_star_red_density := 100,
  marshlands_plain_brown_density := 120,
  shrublands_striped_blue_density := 35
}

-- Proposition to prove the densities of green toads
theorem calculate_green_toad_densities (d : ToadDensities) :
  (d.wetlands_spotted_brown_density / d.spotted_brown_ratio) * d.green_per_brown = 9.6 ∧
  (d.forests_camouflaged_blue_density / d.camouflaged_blue_ratio) * d.green_per_blue = 6.75 ∧
  (d.grasslands_star_red_density / d.star_red_ratio) * d.green_per_red = 10 ∧
  (d.marshlands_plain_brown_density / d.plain_brown_ratio) * d.green_per_brown = 6.4 ∧
  (d.shrublands_striped_blue_density / d.striped_blue_ratio) * d.green_per_blue = 10.5 := sorry

#align the given ratios with the calculation
#print calculate_green_toad_densities

end 

end calculate_green_toad_densities_l103_103589


namespace midpoint_of_AB_intersects_ray_and_curve_l103_103114

noncomputable def midpoint_coordinates (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def curve (t : ℝ) : ℝ × ℝ := (t + 1, (t - 1) ^ 2)

def theta_ray : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.2 = p.1}

theorem midpoint_of_AB_intersects_ray_and_curve :
  ∃ (A B M : ℝ × ℝ), 
    A ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = curve t} ∧ 
    B ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = curve t} ∧ 
    A ∈ theta_ray ∧ 
    B ∈ theta_ray ∧ 
    M = midpoint_coordinates A B ∧ 
    M = (2.5, 2.5) :=
by
  sorry

end midpoint_of_AB_intersects_ray_and_curve_l103_103114


namespace transport_tax_to_be_paid_l103_103513

noncomputable def engine_power : ℕ := 150
noncomputable def tax_rate : ℕ := 20
noncomputable def annual_tax : ℕ := engine_power * tax_rate
noncomputable def months_used : ℕ := 8
noncomputable def prorated_tax : ℕ := (months_used * annual_tax) / 12

theorem transport_tax_to_be_paid : prorated_tax = 2000 := 
by 
  -- sorry is used to skip the proof step
  sorry

end transport_tax_to_be_paid_l103_103513


namespace total_volume_of_four_cubes_is_500_l103_103255

-- Definitions for the problem assumptions
def edge_length := 5
def volume_of_cube (edge_length : ℕ) := edge_length ^ 3
def number_of_boxes := 4

-- Main statement to prove
theorem total_volume_of_four_cubes_is_500 :
  (volume_of_cube edge_length) * number_of_boxes = 500 :=
by
  -- Proof steps will go here
  sorry

end total_volume_of_four_cubes_is_500_l103_103255


namespace find_n_l103_103455

theorem find_n (n : ℕ) : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ n → n = 29 :=
by
  sorry

end find_n_l103_103455


namespace max_value_proof_l103_103765

noncomputable def problem (a b x : ℝ) : Prop :=
  (a + b = 1) ∧ (b - a = -7) ∧ ∃ ρ, (4 * Real.cos x - 3 * Real.sin x = 5 * Real.sin (x + ρ))

theorem max_value_proof :
  let a := 4
  let b := -3
  ∀ x, 4 * Real.cos x - 3 * Real.sin x ≤ 5 :=
begin
  sorry
end

example :
  let a := 4
  let b := -3
  (∀ x, ∃ ρ, (4 * Real.cos x - 3 * Real.sin x = 5 * Real.sin (x + ρ))) →
  (∀ x, 4 * Real.cos x - 3 * Real.sin x ≤ 5) :=
begin
  intro h,
  sorry
end

end max_value_proof_l103_103765


namespace correct_statement_l103_103633

theorem correct_statement : 
  ({0} ∉ {0, 1, 2} ∧ ∅ ⊆ {1, 2} ∧ ∅ ∉ {0} ∧ 0 ∉ ∅ ->
  (∀ P, P = ({0} ∈ {0, 1, 2}) ∨ P = (∅ ⊂ {1,2}) ∨ P = (∅ ∈ {0}) ∨ P = (0 ∈ ∅)
  ↔ P = (∅ ⊆ {1,2}))) := 
by 
  sorry 

end correct_statement_l103_103633


namespace fraction_exponentiation_example_l103_103948

theorem fraction_exponentiation_example :
  (5/3)^4 = 625/81 :=
by
  sorry

end fraction_exponentiation_example_l103_103948


namespace even_function_iff_a_eq_1_range_of_m_l103_103531

-- Definition of the function
def f (a : ℝ) (x : ℝ) := a * Real.exp x + Real.exp (-x)

-- Proof statement for Question 1
theorem even_function_iff_a_eq_1 (a : ℝ) : 
  (∀ x, f a x = f a (-x)) ↔ (a = 1) :=
by sorry

-- Proof statement for Question 2
theorem range_of_m (m : ℝ) : 
  let f := λ x : ℝ, Real.exp x + Real.exp (-x) in
  (f (m + 2) ≤ f (2 * m - 3)) ↔ (m ≤ 1/3 ∨ m ≥ 5) :=
by sorry

end even_function_iff_a_eq_1_range_of_m_l103_103531


namespace perpendicular_line_MI_l103_103124

/-- Defining a triangle with side length constraints and incenter --/
def triangle (A B C : EuclideanGeometry.Point) : Prop := 
  ∃ ACmin, ACmin = EuclideanGeometry.dist A C ∧ 
  ACmin < EuclideanGeometry.dist A B ∧ ACmin < EuclideanGeometry.dist B C ∧
  ∃ K L M I, 
    EuclideanGeometry.OnSegment K A B ∧ EuclideanGeometry.OnSegment L C B ∧ 
    EuclideanGeometry.dist K A = ACmin ∧ EuclideanGeometry.dist L C = ACmin ∧
    EuclideanGeometry.MeetAt M (EuclideanGeometry.lineThrough A L) (EuclideanGeometry.lineThrough K C) ∧
    I = EuclideanGeometry.incenter A B C

/-- The goal is to prove MI is perpendicular to AC --/
theorem perpendicular_line_MI (A B C K L M I : EuclideanGeometry.Point) 
  (h_triangle : triangle A B C) : EuclideanGeometry.IsPerpendicular (EuclideanGeometry.lineThrough M I) (EuclideanGeometry.lineThrough A C)
:= sorry

end perpendicular_line_MI_l103_103124


namespace length_of_tangent_l103_103335

/-- 
Let O and O1 be the centers of the larger and smaller circles respectively with radii 8 and 3. 
The circles touch each other internally. Let A be the point of tangency and OM be the tangent from center O to the smaller circle. 
Prove that the length of this tangent is 4.
--/
theorem length_of_tangent {O O1 : Type} (radius_large : ℝ) (radius_small : ℝ) (OO1 : ℝ) 
  (OM O1M : ℝ) (h : 8 - 3 = 5) (h1 : OO1 = 5) (h2 : O1M = 3): OM = 4 :=
by
  sorry

end length_of_tangent_l103_103335


namespace cross_country_winning_scores_l103_103486

/-
Problem Statement:
In a cross country meet between 2 teams of 6 runners each, each runner who finishes in the
n-th position contributes n points to his team's score. The team with the lower score wins.
Assuming there are no ties among the runners, prove that the number of different winning scores
possible is 19.
-/

theorem cross_country_winning_scores :
  let total_positions := (list.range 12).map (+ 1),  -- positions from 1 to 12
      total_score := total_positions.sum,
      lowest_six_positions := [1, 2, 3, 4, 5, 6],
      lowest_six_sum := lowest_six_positions.sum,
      half_total_score_floor := (total_score / 2).floor,
      winning_scores_range := list.range' (lowest_six_sum) (half_total_score_floor - lowest_six_sum + 1)
  in winning_scores_range.length = 19 := 
by
  simp [total_positions, total_score, lowest_six_positions, lowest_six_sum, half_total_score_floor, winning_scores_range]
  sorry

end cross_country_winning_scores_l103_103486


namespace sufficient_but_not_necessary_condition_l103_103593

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 3 → x^2 - 2 * x > 0) ∧ ¬ (x^2 - 2 * x > 0 → x > 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l103_103593


namespace trig_identity_proof_l103_103458

theorem trig_identity_proof
  (x y : ℝ)
  (h1 : sin x / sin y = 3)
  (h2 : cos x / cos y = 1 / 2) :
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 68 / 29 := 
  sorry

end trig_identity_proof_l103_103458


namespace solve_for_x_l103_103789

theorem solve_for_x (x : ℝ) (h : sqrt (3 + sqrt x) = 4) : x = 169 := 
sorry

end solve_for_x_l103_103789


namespace intersection_M_N_l103_103439

def M (x : ℝ) : Prop := (2 - x) / (x + 1) ≥ 0
def N (y : ℝ) : Prop := ∃ x : ℝ, y = Real.log x

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {y : ℝ | N y} = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end intersection_M_N_l103_103439


namespace actual_diameter_correct_l103_103628

-- Define the magnification factor
def magnification_factor : ℝ := 1000

-- Define the magnified diameter
def magnified_diameter : ℝ := 1

-- Define the actual diameter
def actual_diameter : ℝ := magnified_diameter / magnification_factor

-- State the theorem to prove the actual diameter is 0.001 cm
theorem actual_diameter_correct : actual_diameter = 0.001 := 
by
  sorry

end actual_diameter_correct_l103_103628


namespace transport_tax_correct_l103_103511

-- Define the conditions
def car_horsepower : ℕ := 150
def tax_rate : ℕ := 20
def tax_period_months : ℕ := 8

-- Define the function to calculate the annual tax
def annual_transport_tax (horsepower : ℕ) (rate : ℕ) : ℕ :=
  horsepower * rate

-- Define the function to prorate the annual tax
def prorated_tax (annual_tax : ℕ) (months : ℕ) : ℕ :=
  (annual_tax * months) / 12

-- The proof problem: Prove the amount of transport tax Ivan needs to pay
theorem transport_tax_correct :
  let annual_tax := annual_transport_tax car_horsepower tax_rate in
  let prorated_tax := prorated_tax annual_tax tax_period_months in
  prorated_tax = 2000 :=
by 
  sorry

end transport_tax_correct_l103_103511


namespace washing_machine_regular_wash_l103_103311

variable {R : ℕ}

/-- A washing machine uses 20 gallons of water for a heavy wash,
2 gallons of water for a light wash, and an additional light wash
is added when bleach is used. Given conditions:
- Two heavy washes are done.
- Three regular washes are done.
- One light wash is done.
- Two loads are bleached.
- Total water used is 76 gallons.
Prove the washing machine uses 10 gallons of water for a regular wash. -/
theorem washing_machine_regular_wash (h : 2 * 20 + 3 * R + 1 * 2 + 2 * 2 = 76) : R = 10 :=
by
  sorry

end washing_machine_regular_wash_l103_103311


namespace fixed_point_stable_point_condition_l103_103404

-- Definitions
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f(x) = x
def is_stable_point (f : ℝ → ℝ) (x : ℝ) : Prop := f(x) = has_inv.inv f x

-- Main theorem statement
theorem fixed_point_stable_point_condition (f : ℝ → ℝ) (mono_dec : ∀ ⦃x y⦄, x < y → f y < f x) : 
  (∀ x, is_fixed_point f x → is_stable_point f x) ∧ (∃ x, is_stable_point f x ∧ ¬ is_fixed_point f x) := 
by
  sorry

end fixed_point_stable_point_condition_l103_103404


namespace math_problem_l103_103433

-- Define the parabola and its related properties
def parabola (x y : ℝ) : Prop := x^2 = 8 * y
def F : ℝ × ℝ := (0, 2)
def directrix : ℝ → Prop := λ y, y = -2

-- Define points A and E
def A : ℝ × ℝ := (2, 1 / 2)
def E : ℝ × ℝ := (2, 3)

-- Define correct and incorrect choices
def correct_choices : Prop :=
  (dist A F = 5/2) ∧
  (E = (2, 3) → ∀ A', parabola A'.1 A'.2 → min_dist_AE_AF A' E F = 5) ∧
  tangent_circle_diameter_AB_Focus F :=
sorry

-- Distances and minimum value calculation functions
def dist (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

def min_dist_AE_AF (A E F : ℝ × ℝ) : ℝ :=
  sorry -- Placeholder, complex calculation

def tangent_circle_diameter_AB_Focus (F : ℝ × ℝ) : Prop :=
  sorry

theorem math_problem : correct_choices :=
sorry

end math_problem_l103_103433


namespace Q_le_sqrt_2n_P_l103_103723

def P (n : ℕ) : ℕ := -- definition of the number of partitions
sorry

def spread (p : list ℕ) : ℕ := -- definition of spread, the number of distinct elements in a partition
sorry

def Q (n : ℕ) : ℕ := -- definition of Q as the sum of the spreads of all partitions of n
sorry

theorem Q_le_sqrt_2n_P (n : ℕ) : Q(n) ≤ (nat.sqrt (2 * n)) * P(n) :=
sorry

end Q_le_sqrt_2n_P_l103_103723


namespace ellipse_eqn_sum_of_reciprocal_distances_l103_103417

def ellipse_equation (a b : ℝ) := ∀ x y : ℝ, (x, y) ∈ { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 }

def passes_through_point (x y : ℝ) (P : ℝ × ℝ) := ellipse_equation 2 1 (-1) (real.sqrt 2 / 2)

def eccentricity (a b c : ℝ) := c / a

def distance (P Q : ℝ × ℝ) := real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

axiom known_ellipse : ellipse_equation 2 1 (-1) (real.sqrt 2 / 2)
axiom known_eccentricity : eccentricity (real.sqrt 2) (real.sqrt 1) (real.sqrt 2 / 2)

theorem ellipse_eqn : ∀ (a b : ℝ), a^2 = 2 ∧ b^2 = 1 → 
  ∀ (x y : ℝ), ellipse_equation a b x y ↔ ellipse_equation (real.sqrt 2) 1 x y :=
by
  sorry

theorem sum_of_reciprocal_distances : 
  ∀ (A B C D : ℝ × ℝ), A = (-1, 0) ∧ (A = B ∨ A = C ∨ B = D ∨ C = D) →
  (∀ A B C D : ℝ × ℝ, ellipse_equation (real.sqrt 2) 1 A.1 A.2 ∧ ellipse_equation (real.sqrt 2) 1 B.1 B.2 ∧ 
    ellipse_equation (real.sqrt 2) 1 C.1 C.2 ∧ ellipse_equation (real.sqrt 2) 1 D.1 D.2 → 
  ∃ (AB CD : ℝ), distance A B = AB ∧ distance C D = CD ∧ (1 / AB + 1 / CD = (3/4) * real.sqrt 2)) :=
by
  sorry

end ellipse_eqn_sum_of_reciprocal_distances_l103_103417


namespace sqrt_defined_iff_nonneg_l103_103476

theorem sqrt_defined_iff_nonneg (x : ℝ) : (∃ y, y = sqrt (x - 2)) ↔ x ≥ 2 :=
by
  sorry

end sqrt_defined_iff_nonneg_l103_103476


namespace three_digit_integers_count_l103_103925

theorem three_digit_integers_count : 
  let digits := {7, 8, 9}
  ∃ (n : ℕ), n = 27 ∧ ∀ (x : ℕ),
    (100 ≤ x ∧ x < 1000 ∧ 
    (∃ d1 d2 d3, 
      x = d1 * 100 + d2 * 10 + d3 ∧ 
      d1 ∈ digits ∧ 
      d2 ∈ digits ∧ 
      d3 ∈ digits)) ↔ 
    (finset.univ.filter 
      (λ y : ℕ, 
        100 ≤ y ∧ y < 1000 ∧ 
        (y / 100 ∈ digits ∧ 
         (y % 100) / 10 ∈ digits ∧ 
         y % 10 ∈ digits))).card = n :=
by
  sorry

end three_digit_integers_count_l103_103925


namespace cylinder_surface_area_is_4pi_l103_103884

-- Definitions based on conditions
def base_radius : ℝ := 1
def height : ℝ := 1

-- Formula for the surface area of a cylinder
def surface_area_cylinder (r : ℝ) (h : ℝ) : ℝ := 2 * Real.pi * r * (r + h)

-- Theorem statement
theorem cylinder_surface_area_is_4pi : 
  surface_area_cylinder base_radius height = 4 * Real.pi :=
by
  sorry

end cylinder_surface_area_is_4pi_l103_103884


namespace rectangular_segments_length_l103_103483

def verify_lengths (X : ℕ) : Prop :=
  (2 + X + 3 = 4 + 1 + 5) → (X = 5)

theorem rectangular_segments_length : ∀ X : ℕ, verify_lengths X :=
by
  intro X
  unfold verify_lengths
  intro h
  calc
    2 + X + 3 = 4 + 1 + 5  : h
    5 + X = 10 : by ring
  sorry

end rectangular_segments_length_l103_103483


namespace num_elements_satisfying_condition_l103_103861

def set_M : Set ℕ := {0, 1, 2, 3, 4, 5}

def op (i j : ℕ) : ℕ := (i + j) % 4

theorem num_elements_satisfying_condition :
  let M := set_M;
  let A := 0;
  let A_2 := 2;
  let condition (a : ℕ) := op (op a a) A_2 = A;
  fintype.card {a ∈ M | condition a} = 3 := 
  by
  sorry

end num_elements_satisfying_condition_l103_103861


namespace solution_logarithmic_eq_l103_103450

theorem solution_logarithmic_eq (x : ℝ) (h : log 10 (x^2 - 3 * x + 6) = 1) : x = 4 ∨ x = -1 := sorry

end solution_logarithmic_eq_l103_103450


namespace convex_polygon_iff_m_eq_f_l103_103157

noncomputable def a_t (S : set point) (t : ℕ) : ℕ := sorry

noncomputable def m (S : set point) : ℕ :=
  finset.sum (finset.range S.card) (a_t S)

def convex_polygon (S : set point) : Prop := sorry

theorem convex_polygon_iff_m_eq_f (S : set point) (n : ℕ) (h1 : n ≥ 4)
  (h2 : ∀ P1 P2 P3 ∈ S, (collinear P1 P2 P3) → false)
  (h3 : ∀ P1 P2 P3 P4 ∈ S, (concyclic P1 P2 P3 P4) → false) :
  convex_polygon S ↔ m S = 2 * (nat.choose n 4) :=
sorry

end convex_polygon_iff_m_eq_f_l103_103157


namespace kate_average_speed_correct_l103_103516

noncomputable def kate_average_speed : ℝ :=
  let biking_time_hours := 20 / 60
  let walking_time_hours := 60 / 60
  let jogging_time_hours := 40 / 60
  let biking_distance := 20 * biking_time_hours
  let walking_distance := 4 * walking_time_hours
  let jogging_distance := 6 * jogging_time_hours
  let total_distance := biking_distance + walking_distance + jogging_distance
  let total_time_hours := biking_time_hours + walking_time_hours + jogging_time_hours
  total_distance / total_time_hours

theorem kate_average_speed_correct : kate_average_speed = 9 :=
by
  sorry

end kate_average_speed_correct_l103_103516


namespace neg_rational_is_rational_l103_103791

theorem neg_rational_is_rational (m : ℚ) : -m ∈ ℚ := 
by sorry

end neg_rational_is_rational_l103_103791


namespace permutation_divisible_by_7_l103_103510

open Int

theorem permutation_divisible_by_7 (n : ℕ) (h : n ≥ 2) :
  ∃ p : List ℕ, p ~ List.range (n + 1) ∧ (p.foldr (λ (d m : ℕ), d + m * 10) 0) ≡ 0 [MOD 7] :=
  sorry

end permutation_divisible_by_7_l103_103510


namespace total_ads_clicked_l103_103946

theorem total_ads_clicked (a1 a2 a3 a4 : ℕ) (clicked_ads : ℕ) :
  a1 = 12 →
  a2 = 2 * a1 →
  a3 = a2 + 24 →
  a4 = (3 * a2) / 4 →
  clicked_ads = (2 * (a1 + a2 + a3 + a4)) / 3 →
  clicked_ads = 68 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end total_ads_clicked_l103_103946


namespace reduced_price_per_dozen_apples_l103_103304

variables (P R : ℝ) 

theorem reduced_price_per_dozen_apples (h₁ : R = 0.70 * P) 
  (h₂ : (30 / P + 54) * R = 30) :
  12 * R = 2 := 
sorry

end reduced_price_per_dozen_apples_l103_103304


namespace train_length_l103_103994

theorem train_length {s : ℝ} {t : ℝ} (h1 : s = 180) (h2 : t = 9) :
  let speed := s * (1000 / 3600) in
  let length := speed * t in
  length = 450 :=
by
  sorry

end train_length_l103_103994


namespace speed_of_second_train_l103_103672

-- Define the conditions and the given values
def first_train_speed := 35 -- Speed of the first train in km/h
def distance := 1400 -- Distance in km
def departure_time_difference := 5 -- Time difference in hours

-- Define the main theorem to prove the speed of the second train
theorem speed_of_second_train :
  let time_first_train := distance / first_train_speed in
  let time_second_train := time_first_train - departure_time_difference in
  ∃ speed_second_train, speed_second_train = distance / time_second_train ∧ speed_second_train = 40 :=
begin
  sorry
end

end speed_of_second_train_l103_103672


namespace sin_angle_add_pi_over_4_l103_103416

open Real

theorem sin_angle_add_pi_over_4 (α : ℝ) (h1 : (cos α = -3/5) ∧ (sin α = 4/5)) : sin (α + π / 4) = sqrt 2 / 10 :=
by
  sorry

end sin_angle_add_pi_over_4_l103_103416


namespace find_alpha_l103_103741

theorem find_alpha (α : ℝ) (h_cos : Real.cos α = - (Real.sqrt 3 / 2)) (h_range : 0 < α ∧ α < Real.pi) : α = 5 * Real.pi / 6 :=
sorry

end find_alpha_l103_103741


namespace number_of_hens_l103_103637

variables (H C : ℕ)

def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 144

theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 24 :=
sorry

end number_of_hens_l103_103637


namespace sum_floor_terms_eq_314_l103_103278

noncomputable def floor_pi_terms_sum : ℕ :=
  (List.range 100).sum (λ k, (Real.floor (Real.pi + k / 100 : ℝ)).toNat)

theorem sum_floor_terms_eq_314 : floor_pi_terms_sum = 314 := by
  sorry

end sum_floor_terms_eq_314_l103_103278


namespace prob_P_plus_S_one_less_multiple_of_7_l103_103932

theorem prob_P_plus_S_one_less_multiple_of_7 :
  let a b : ℕ := λ x y, x ∈ Finset.range (60+1) ∧ y ∈ Finset.range (60+1) ∧ x ≠ y ∧ 1 ≤ x ∧ x ≤ 60 ∧ 1 ≤ y ∧ y ≤ 60,
      P : ℕ := ∀ a b, a * b,
      S : ℕ := ∀ a b, a + b,
      m : ℕ := (P + S) + 1,
      all_pairs : ℕ := Nat.choose 60 2,
      valid_pairs : ℕ := 444,
      probability : ℚ := valid_pairs / all_pairs
  in probability = 148 / 590 := sorry

end prob_P_plus_S_one_less_multiple_of_7_l103_103932


namespace average_daily_production_correct_l103_103669

-- Defining the monthly production based on given conditions.
def production_in_month (month : ℕ) : ℕ :=
  3000 + (month - 1) * 100

-- Calculating the total annual production by summing the monthly production.
def total_annual_production : ℕ :=
  (List.range 12).sum (λ i, production_in_month (i + 1))

-- Define the number of days in a year.
def days_in_year : ℕ := 365

-- Calculating the average daily production
def average_daily_production : ℕ :=
  total_annual_production / days_in_year

-- The main theorem we need to prove
theorem average_daily_production_correct :
  average_daily_production ≈ 121.1 := sorry

end average_daily_production_correct_l103_103669


namespace fraction_exponentiation_example_l103_103949

theorem fraction_exponentiation_example :
  (5/3)^4 = 625/81 :=
by
  sorry

end fraction_exponentiation_example_l103_103949


namespace part_a_solution_l103_103582

theorem part_a_solution (x y : ℤ) : xy + 3 * x - 5 * y = -3 ↔ 
  (x = 6 ∧ y = -21) ∨ 
  (x = -13 ∧ y = -2) ∨ 
  (x = 4 ∧ y = 15) ∨ 
  (x = 23 ∧ y = -4) ∨ 
  (x = 7 ∧ y = -12) ∨ 
  (x = -4 ∧ y = -1) ∨ 
  (x = 3 ∧ y = 6) ∨ 
  (x = 14 ∧ y = -5) ∨ 
  (x = 8 ∧ y = -9) ∨ 
  (x = -1 ∧ y = 0) ∨ 
  (x = 2 ∧ y = 3) ∨ 
  (x = 11 ∧ y = -6) := 
by sorry

end part_a_solution_l103_103582


namespace continuity_f_at_3_l103_103858

noncomputable def f (x : ℝ) := if x ≤ 3 then 3 * x^2 - 5 else 18 * x - 32

theorem continuity_f_at_3 : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x - f 3) < ε := by
  intro ε ε_pos
  use 1
  simp
  sorry

end continuity_f_at_3_l103_103858


namespace sqrt_3x_minus_5_domain_l103_103800

theorem sqrt_3x_minus_5_domain (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (3 * x - 5)) ↔ x ≥ 5 / 3 :=
by
  sorry

end sqrt_3x_minus_5_domain_l103_103800


namespace find_number_l103_103225

-- Define the conditions
def percentage_less (percent : ℝ) (num : ℝ) : ℝ := num * (1 - percent / 100)
def percentage_more (percent : ℝ) (num : ℝ) : ℝ := num * (1 + percent / 100)

-- Define the known data
def num := 80
def percent_less_num := percentage_less 30 num
def target := 56

-- Main statement to prove
theorem find_number :
  target = percentage_more 40 x → x = 40 :=
by
  sorry

end find_number_l103_103225


namespace sum_of_prime_factors_150280_l103_103250

theorem sum_of_prime_factors_150280 : 
  let prime_factors := [2, 5, 13, 17] in
  prime_factors.sum = 37 :=
begin
  sorry
end

end sum_of_prime_factors_150280_l103_103250


namespace gcd_n3_plus_16_n_plus_4_l103_103374

/-- For a given positive integer n > 2^4, the greatest common divisor of n^3 + 16 and n + 4 is 1. -/
theorem gcd_n3_plus_16_n_plus_4 (n : ℕ) (h : n > 2^4) : Nat.gcd (n^3 + 16) (n + 4) = 1 := by
  sorry

end gcd_n3_plus_16_n_plus_4_l103_103374


namespace smallest_m_l103_103521

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - ⌊x⌋

noncomputable def f (x : ℝ) : ℝ :=
  abs (3 * fractional_part x - 1.5)

theorem smallest_m (m : ℤ) (h1 : ∀ x : ℝ, m^2 * f (x * f x) = x → True) : ∃ m, m = 8 :=
by
  have h2 : ∀ m : ℤ, (∃ (s : ℕ), s ≥ 1008 ∧ (m^2 * abs (3 * fractional_part (s * abs (1.5 - 3 * (fractional_part s) )) - 1.5) = s)) → m = 8
  {
    sorry
  }
  sorry

end smallest_m_l103_103521


namespace area_of_polygon_intersection_l103_103979

-- Definitions of the points
def A := (0, 0, 0)
def B := (24, 0, 0)
def C := (24, 0, 24)
def D := (24, 24, 24)
def P := (4, 0, 0)
def Q := (24, 0, 12)
def R := (24, 8, 24)

-- Theorem that states the area of the polygon formed by the intersection of plane PQR and the cube
theorem area_of_polygon_intersection : 
  let points := PQR_intersection_points (A, B, C, D) (P, Q, R) in
  calculate_polygon_area points = 850 :=
sorry

end area_of_polygon_intersection_l103_103979


namespace ratio_of_areas_l103_103825

variables {A B C D M N : Type} [AffineSpace A] [LinearOrderedField B]

def is_trapezoid (A B C D : A) (BC_parallel_AD : Parallel (line.through B C) (line.through A D)) (AD_eq_3BC : dist A D = 3 * dist B C) : Prop := 
  Parallel (line.through B C) (line.through A D) ∧ dist A D = 3 * dist B C

def ratio_condition (A M B N C D : A) (AM_MB_eq_3_5 : dist A M / dist M B = 3 / 5) (CN_ND_eq_2_7 : dist C N / dist N D = 2 / 7) : Prop := 
  dist A M / dist M B = 3 / 5 ∧ dist C N / dist N D = 2 / 7

theorem ratio_of_areas (A B C D M N : A) 
  (h_trapezoid : is_trapezoid A B C D (parallel_through B C (A D)) (AD_eq_3BC (dist A D))) 
  (h_ratios : ratio_condition A M B N C D (dist A M / dist M B = 3 / 5) (dist C N / dist N D = 2 / 7)) :
  area M B C N / area A M N D = 9 / 23 := 
sorry

end ratio_of_areas_l103_103825


namespace switches_in_position_A_after_steps_l103_103921

-- Definitions for the switches and their properties
def switch_transition (step : ℕ) (x y : ℕ) : ℕ :=
  if (10 - x) * (10 - y) % 5 = 0 then 0 else (step % 5 + (10 - x) * (10 - y)) % 5

-- Main math proof problem
theorem switches_in_position_A_after_steps :
  ∀ n > 0, n ≤ 9,
  ∀ switches : ℕ → ℕ → ℕ, -- Function representing switches labeled (2^x)(3^y)
  let num_positions := 5 in
  let total_switches := 1250 in
  let count_switch : ℕ := 20 in
  -- Determine switches not toggled multiple of 5 times
  ∃ final_position_count : ℕ,
  -- question: prove that 20 switches remain in position A after 1250 steps.
    (final_position_count = count_switch) := by
{
  sorry
}

end switches_in_position_A_after_steps_l103_103921


namespace segment_length_PR_l103_103567

theorem segment_length_PR 
(points_on_circle : Π (P Q : Point) (r : ℝ), (P ∈ Circle r) ∧ (Q ∈ Circle r))
(PQ_length : ∀ (P Q : Point), PQ.distance = 8)
(is_midpoint_minor_arc : ∀ R (P Q : Point), R.is_midpoint_arc PQ)
: segment_length PR = sqrt(98 - 14 * sqrt(33)) :=
sorry

end segment_length_PR_l103_103567


namespace climbing_stairs_time_sum_l103_103336

theorem climbing_stairs_time_sum (n a_1 d : ℕ) (h1 : n = 5) (h2 : a_1 = 20) (h3 : d = 5):
  let a_n := a_1 + (n - 1) * d,
      S_n := n * (a_1 + a_n) / 2
  in S_n = 150 := by
  sorry

end climbing_stairs_time_sum_l103_103336


namespace complex_inequality_l103_103839

variable z : ℂ

theorem complex_inequality (h : |z + 1| > 2) : |z^3 + 1| > 1 := 
by
  sorry

end complex_inequality_l103_103839


namespace circumcenter_BIC_on_circumcircle_ABC_l103_103524

variables {A B C I O : Type}
variable [incircle : is_incenter I (triangle A B C)]
variable [circumcircle : is_circumcircle O (triangle A B C)]

theorem circumcenter_BIC_on_circumcircle_ABC
  (circumcenter_BIC : is_circumcenter (circumcenter (triangle B I C)) (circle O)) :
  circumcenter_BIC ∈ circumcircle (circle O) :=
sorry

end circumcenter_BIC_on_circumcircle_ABC_l103_103524


namespace number_of_students_l103_103685

theorem number_of_students (n : ℕ) 
  (h1 : ∀ (s : Finset ℕ), s.card = n → ∀ (a : ℕ), a ∈ s → (∃ (b : Finset ℕ), b.card = 20 ∧ b ⊆ s ∧ a ∉ b))
  (h2 : ∀ (a b c : ℕ) (s : Finset ℕ), a ≠ b → a > b → a ∈ s → b ∈ s → c ∈ s → (a,b) ∈ c → (b,c) ∈ c → (a,c) ∉ c → (c.card = 20 → (∃ (d: Finset ℕ), d.card = 13 ∧ d ⊆ c ∧ d ≠ a ∧ d ≠ b))) 
  (h3 : ∀ (a b c : ℕ) (s : Finset ℕ), a ≠ b → a > b → a ∈ s → b ∈ s → c ∈ s → (a,b) ∉ c → (b,c) ∈ c → (a,c) ∉ c → (c.card = 20 → (∃ (d: Finset ℕ), d.card = 12 ∧ d ⊆ c ∧ d ≠ a ∧ d ≠ b))) :
  n = 31 :=
  by {
  sorry
}  

end number_of_students_l103_103685


namespace graph_passes_through_fixed_point_l103_103899

theorem graph_passes_through_fixed_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
    ∃ (x y : ℝ), (x = -3) ∧ (y = -1) ∧ (y = a^(x + 3) - 2) :=
by
  sorry

end graph_passes_through_fixed_point_l103_103899


namespace tenfold_largest_two_digit_number_l103_103603

def largest_two_digit_number : ℕ := 99

theorem tenfold_largest_two_digit_number :
  10 * largest_two_digit_number = 990 :=
by
  sorry

end tenfold_largest_two_digit_number_l103_103603


namespace probability_multiple_of_3_l103_103615

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

theorem probability_multiple_of_3 : 
  let total_tickets := 27
  let multiples_of_3 := { n : ℕ | 1 ≤ n ∧ n ≤ total_tickets ∧ is_multiple_of_3 n }
  let count_multiples_of_3 := multiplicative_group.orderOf multiples_of_3
in 
  (count_multiples_of_3 / total_tickets : ℚ) = 1 / 3 :=
sorry

end probability_multiple_of_3_l103_103615


namespace initial_guppies_l103_103343

variable (initial_angelfish : ℕ) (initial_tiger_sharks : ℕ) (initial_oscar_fish : ℕ)
variable (sold_guppies : ℕ) (sold_angelfish : ℕ) (sold_tiger_sharks : ℕ) (sold_oscar_fish : ℕ)
variable (total_remaining_fish : ℕ)

theorem initial_guppies 
  (h1 : initial_angelfish = 76)
  (h2 : initial_tiger_sharks = 89)
  (h3 : initial_oscar_fish = 58)
  (h4 : sold_guppies = 30)
  (h5 : sold_angelfish = 48)
  (h6 : sold_tiger_sharks = 17)
  (h7 : sold_oscar_fish = 24)
  (h8 : total_remaining_fish = 198) :
  initial_guppies = 94 := 
by
  sorry

end initial_guppies_l103_103343


namespace circumcenter_line_perpendicular_to_PQ_l103_103837

open EuclideanGeometry

variables {A B C D P Q : Point}

def is_trapezoid (A B C D : Point) : Prop :=
AD // Here a definition for determining trapezoid nature would go, e.g., AD parallel to BC with AD != BC

axiom AD_gt_BC : (dist A D) > (dist B C)

axiom AB_meets_CD_at_P : meet (line_through A B) (line_through C D) P

axiom Q_on_AD : on (segment A D) Q

axiom BQ_eq_CQ : (dist B Q) = (dist C Q)

theorem circumcenter_line_perpendicular_to_PQ : 
  let O1 := circumcenter (triangle A Q C) in
  let O2 := circumcenter (triangle B Q D) in
  is_perpendicular_to (line_through O1 O2) (line_through P Q) :=
sorry

end circumcenter_line_perpendicular_to_PQ_l103_103837


namespace factorization_l103_103358

-- Define the terms
variables (a x y : ℝ)

-- State the theorem
theorem factorization (a x y : ℝ) : 3 * a * x^2 - 3 * a * y^2 = 3 * a * (x + y) * (x - y) :=
by {
  sorry
}

end factorization_l103_103358


namespace find_value_of_k_l103_103751

-- Define the geometric sequence sum condition.
def geometric_sequence_sum (n : ℕ) (k : ℚ) : ℚ :=
  2 * 3^(n-1) + k

-- Prove the problem's claim
theorem find_value_of_k (k : ℚ) :
  (∀ n : ℕ, geometric_sequence_sum n k = (2 / 3) * 3^n + k)
  → k = -2 / 3 :=
by
  assume h: ∀ n : ℕ, geometric_sequence_sum n k = (2 / 3) * 3^n + k
  -- Proof steps can be filled in to show k = -2 / 3
  sorry

end find_value_of_k_l103_103751


namespace total_income_l103_103985

variable (I : ℝ)

/-- A person distributed 20% of his income to his 3 children each. -/
def distributed_children (I : ℝ) : ℝ := 3 * 0.20 * I

/-- He deposited 30% of his income to his wife's account. -/
def deposited_wife (I : ℝ) : ℝ := 0.30 * I

/-- The total percentage of his income that was given away is 90%. -/
def total_given_away (I : ℝ) : ℝ := distributed_children I + deposited_wife I 

/-- The remaining income after giving away 90%. -/
def remaining_income (I : ℝ) : ℝ := I - total_given_away I

/-- He donated 5% of the remaining income to the orphan house. -/
def donated_orphan_house (remaining : ℝ) : ℝ := 0.05 * remaining

/-- Finally, he has $40,000 left, which is 95% of the remaining income. -/
def final_amount (remaining : ℝ) : ℝ := 0.95 * remaining

theorem total_income (I : ℝ) (h : final_amount (remaining_income I) = 40000) :
  I = 421052.63 := 
  sorry

end total_income_l103_103985


namespace sin_double_angle_cos_sum_angle_tan_sum_angle_l103_103740

variables (θ : ℝ)
hypothesis cos_θ : cos θ = 4 / 5
hypothesis θ_in_first_quadrant : θ ∈ Ioo 0 (π / 2)

theorem sin_double_angle : sin (2 * θ) = 24 / 25 :=
by sorry

theorem cos_sum_angle : cos (θ + π / 4) = √2 / 10 :=
by sorry

theorem tan_sum_angle : tan (θ + π / 4) = 7 :=
by sorry

end sin_double_angle_cos_sum_angle_tan_sum_angle_l103_103740


namespace problem_1_problem_2_l103_103424

def f (a : ℝ) (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 1 then 2^(-x)
  else if 1 ≤ x ∧ x ≤ a then Real.log x
  else 0

theorem problem_1 (x : ℝ) :
  f 2 x = 1 → x = 0 :=
by sorry

theorem problem_2 (a : ℝ) :
  (∀ y ∈ Set.Icc (0 : ℝ) 2, ∃ x : ℝ, -1 ≤ x ∧ x ≤ a ∧ y = f a x) → Real.sqrt Real.exp 1 ≤ a ∧ a ≤ Real.exp 2 :=
by sorry

end problem_1_problem_2_l103_103424


namespace triangle_area_130_l103_103408

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_130 :
  ∀ (a b c : ℝ), a = 28 → b = 26 → c = 10 →
  (a + b > c) → (b + c > a) → (c + a > b) →
  heron_area a b c = 130 :=
by
  intros a b c ha hb hc h_valid1 h_valid2 h_valid3
  rw [ha, hb, hc]
  sorry

end triangle_area_130_l103_103408


namespace total_cost_of_rolls_l103_103835

-- Defining the conditions
def price_per_dozen : ℕ := 5
def total_rolls_bought : ℕ := 36
def rolls_per_dozen : ℕ := 12

-- Prove the total cost calculation
theorem total_cost_of_rolls : (total_rolls_bought / rolls_per_dozen) * price_per_dozen = 15 :=
by
  sorry

end total_cost_of_rolls_l103_103835


namespace watch_cost_price_l103_103271

theorem watch_cost_price (cost_price : ℝ)
  (h1 : SP_loss = 0.90 * cost_price)
  (h2 : SP_gain = 1.08 * cost_price)
  (h3 : SP_gain - SP_loss = 540) :
  cost_price = 3000 := 
sorry

end watch_cost_price_l103_103271


namespace angle_R_l103_103133

variables (A B C P R R' : Type*)
variables [PlaneGeometry A B C P R R']
variables (h1 : Triangle A B C)
variables (h2 : angleBisector C P B R A)
variables (h3 : parallel AC P)
variables (h4 : tangentCircum B R C)
variables (h5 : reflection R' R AB)

theorem angle_R'PB_eq_angle_RPA (h1 : Triangle A B C)
  (h2: angleBisector C P B A)
  (h3: parallel AC P)
  (h4: tangentCircum B R C)
  (h5: reflection R' R AB) : 
  angle R' P B = angle R P A := 
sorry

end angle_R_l103_103133


namespace func_inequality_l103_103760

noncomputable def f (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given function properties
variables {a b c : ℝ} (h_a : a > 0) (symmetry : ∀ x : ℝ, f a b c (2 + x) = f a b c (2 - x))

theorem func_inequality : f a b c 2 < f a b c 1 ∧ f a b c 1 < f a b c 4 :=
by
  sorry

end func_inequality_l103_103760


namespace equilateral_triangles_count_l103_103683

-- Definitions:
def nails : ℕ := 10  -- The number of nails on the board.
def equidistant_nails (n : ℕ) : Prop := ∀ (i j : ℕ), i ≠ j → (i < n → j < n → abs (i - j) = abs (abs ((i + 1) % n - (j + 1) % n)))

-- Theorem stating the problem:
theorem equilateral_triangles_count (n : ℕ) (h : equidistant_nails n) : n = 10 → ∃ t : ℕ, t = 13 := 
sorry

end equilateral_triangles_count_l103_103683


namespace dot_product_calculation_l103_103078

def vector3 := (ℝ, ℝ, ℝ)

def dot_product (v1 v2 : vector3) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Given vectors
def a : vector3 := (4, 2, -4)
def b : vector3 := (6, -3, 2)

-- Intermediate vector operations
def v1 : vector3 := (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2, 2 * a.3 - 3 * b.3)
def v2 : vector3 := (a.1 + 2 * b.1, a.2 + 2 * b.2, a.3 + 2 * b.3)

-- Assertion of the final result
theorem dot_product_calculation : dot_product v1 v2 = -212 := by
  -- By calculations above, we should reach the conclusion
  sorry

end dot_product_calculation_l103_103078


namespace tangent_line_at_2_tangent_lines_through_P_l103_103752

-- Define the curve equation
def curve (x : ℝ) : ℝ := (1 / 3) * x^3 + (4 / 3)

-- Compute the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := x^2

-- Given point (2, 4) on the curve
def P : ℝ × ℝ := (2, 4)

-- Proof of the tangent line at x = 2
theorem tangent_line_at_2 :
  ∃ (k b : ℝ), k = 4 ∧ b = -4 ∧ ∀ x y : ℝ, y = curve_derivative 2 * x + b ↔ 4 * x - y - 4 = 0 := sorry

-- Proof of the tangent lines passing through (2, 4)
theorem tangent_lines_through_P :
  ∃ (x0 : ℝ), (curve 2 = 4) ∧ ((4 * x - y - 4 = 0) ∨ (x - y + 2 = 0)) := sorry

end tangent_line_at_2_tangent_lines_through_P_l103_103752


namespace orange_juice_fraction_correct_l103_103227

def volume (pitcher: ℕ) : ℝ := 800

def orange_juice_first_pitcher : ℝ := (1/4) * volume 1
def orange_juice_second_pitcher : ℝ := (1/3) * volume 2
def apple_juice_third_pitcher : ℝ := (1/2) * volume 3

def total_orange_juice : ℝ := orange_juice_first_pitcher + orange_juice_second_pitcher
def total_volume : ℝ := volume 1 + volume 2 + volume 3

def fraction_orange_juice : ℝ := total_orange_juice / total_volume

theorem orange_juice_fraction_correct :
  fraction_orange_juice = 467 / 2400 := by
  sorry

end orange_juice_fraction_correct_l103_103227


namespace total_volume_of_four_cubes_is_500_l103_103253

-- Definition of the edge length of each cube
def edge_length : ℝ := 5

-- Definition of the volume of one cube
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the number of cubes
def number_of_cubes : ℕ := 4

-- Definition of the total volume
def total_volume (n : ℕ) (v : ℝ) : ℝ := n * v

-- The proposition we want to prove
theorem total_volume_of_four_cubes_is_500 :
  total_volume number_of_cubes (volume_of_cube edge_length) = 500 :=
by
  sorry

end total_volume_of_four_cubes_is_500_l103_103253


namespace combined_tax_rate_approx_l103_103540

noncomputable def combinedTaxRate (M : ℝ) : ℝ :=
  let Mindy := 4 * M
  let Julie := 2 * M
  let totalTax := 0.45 * M + 0.25 * Mindy + 0.35 * Julie
  let totalIncome := M + Mindy + Julie
  totalTax / totalIncome * 100

theorem combined_tax_rate_approx (M : ℝ) :
  combinedTaxRate M ≈ 30.71 :=
by sorry

end combined_tax_rate_approx_l103_103540


namespace count_satisfying_n_l103_103378

theorem count_satisfying_n : 
    (∃! n : ℕ, 1 ≤ n ∧ n ≤ 500 ∧ (∀ t : ℝ, complex.exp (2 * n * complex.I * t) = complex.exp (complex.I * 2 * n * t))) = 125 :=
sorry

end count_satisfying_n_l103_103378


namespace range_of_a_minus_abs_b_l103_103088

theorem range_of_a_minus_abs_b {a b : ℝ} (h1 : 1 < a ∧ a < 3) (h2 : -4 < b ∧ b < 2) :
  -3 < a - |b| ∧ a - |b| < 3 :=
by
  sorry

end range_of_a_minus_abs_b_l103_103088


namespace quadratic_other_x_intercept_l103_103728

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → (a * x^2 + b * x + c) = -3)
  (h_intercept : ∀ x, x = 1 → (a * x^2 + b * x + c) = 0) : 
  ∃ x : ℝ, x = 9 ∧ (a * x^2 + b * x + c) = 0 :=
sorry

end quadratic_other_x_intercept_l103_103728


namespace total_volume_of_four_cubes_is_500_l103_103256

-- Definitions for the problem assumptions
def edge_length := 5
def volume_of_cube (edge_length : ℕ) := edge_length ^ 3
def number_of_boxes := 4

-- Main statement to prove
theorem total_volume_of_four_cubes_is_500 :
  (volume_of_cube edge_length) * number_of_boxes = 500 :=
by
  -- Proof steps will go here
  sorry

end total_volume_of_four_cubes_is_500_l103_103256


namespace intersection_count_l103_103769

def M (x y : ℝ) : Prop := y^2 = x - 1
def N (x y m : ℝ) : Prop := y = 2 * x - 2 * m^2 + m - 2

theorem intersection_count (m x y : ℝ) :
  (M x y ∧ N x y m) → (∃ n : ℕ, n = 1 ∨ n = 2) :=
sorry

end intersection_count_l103_103769


namespace sequence_infinite_negatives_l103_103393

variable (a : ℕ → ℝ) (a1 : ℝ)

-- Sequence definition
def sequence_condition (a : ℕ → ℝ) [Noncomputable]: Prop :=
∀ n : ℕ, 
  a n ≠ 0 → a (n + 1) = (a n ^ 2 - 1) / (2 * a n) ∧ a n = 0 → a (n + 1) = 0

-- The statement to be proved
theorem sequence_infinite_negatives (a : ℕ → ℝ) (a1 : ℝ) (h : sequence_condition a) :
  ∃ infinitely_many (n : ℕ), n ≥ 1 ∧ a n ≤ 0 :=
sorry

end sequence_infinite_negatives_l103_103393


namespace infinite_n_exists_prime_divisors_l103_103376

theorem infinite_n_exists_prime_divisors (P : ℕ → ℕ) (hP : ∀ n > 1, ∃ p, p prime ∧ p | n ∧ (∀ q prime, q | n → q ≤ p)) :
  ∃ᶠ n : ℕ in at_top, 1 < n ∧ P n < P (n + 1) ∧ P (n + 2) :=
by
  sorry

end infinite_n_exists_prime_divisors_l103_103376


namespace area_ratio_of_S_to_T_is_7_over_18_l103_103519

noncomputable def area_ratio := 
  let T := {x : ℝ × ℝ × ℝ | x.1 ≥ 0 ∧ x.2 ≥ 0 ∧ x.3 ≥ 0 ∧ x.1 + x.2 + x.3 = 1}
  let S := {x ∈ T | (x.1 ≥ 1/3 ∧ x.2 ≥ 1/4 ∧ ¬(x.3 ≥ 1/5)) ∨ 
                   (x.1 ≥ 1/3 ∧ ¬(x.2 ≥ 1/4) ∧ x.3 ≥ 1/5) ∨
                   (¬(x.1 ≥ 1/3) ∧ x.2 ≥ 1/4 ∧ x.3 ≥ 1/5)}
  (area S) / (area T)

theorem area_ratio_of_S_to_T_is_7_over_18 : area_ratio = 7 / 18 := 
  sorry

end area_ratio_of_S_to_T_is_7_over_18_l103_103519


namespace total_area_three_plots_l103_103992

variable (x y z A : ℝ)

theorem total_area_three_plots :
  (x = (2 / 5) * A) →
  (z = x - 16) →
  (y = (9 / 8) * z) →
  (A = x + y + z) →
  A = 96 :=
by
  intros h1 h2 h3 h4
  sorry

end total_area_three_plots_l103_103992


namespace max_value_of_C_l103_103010

noncomputable def a (y : ℝ) : ℝ := 1 / y
noncomputable def b (x y : ℝ) : ℝ := y + 1 / x
noncomputable def C(x y : ℝ) : ℝ := min x (min (a y) (b x y))

theorem max_value_of_C (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  C x y ≤ Real.sqrt 2 ∧ 
  (exists x y, 0 < x ∧ 0 < y ∧ C x y = Real.sqrt 2) :=
by
  sorry

end max_value_of_C_l103_103010


namespace sum_inequality_l103_103137

theorem sum_inequality (a : ℕ → ℝ) (h₁ : 1 ≤ a 1) (h₂ : ∀ k, 1 ≤ a (k + 1) - a k) : 
  (∑ k in finset.range 1987, 1 / (a (k + 2) * real.root 1987 (a (k + 1)))) < 1987 :=
sorry

end sum_inequality_l103_103137


namespace rectangle_AB_geq_2_BC_l103_103966

variables {A B C D P Q : Type}
variable [rectangle ABCD]
variable [same_side CD P]
variable [incircle_touch AB Q PAB]

-- Proof that AB >= 2 * BC given conditions
theorem rectangle_AB_geq_2_BC 
  (AB : ℝ) (BC : ℝ) (ABCD_area : AB * BC = 2) : AB ≥ 2 * BC :=
sorry

end rectangle_AB_geq_2_BC_l103_103966


namespace largest_additional_license_plates_l103_103328

theorem largest_additional_license_plates :
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  new_total - original_total = 40 :=
by
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  sorry

end largest_additional_license_plates_l103_103328


namespace not_periodic_f_l103_103178

noncomputable def f : ℝ → ℝ := λ x, Real.sin x + Real.sin (Real.sqrt 2 * x)

theorem not_periodic_f : ¬ ∃ p > 0, ∀ x : ℝ, f (x + p) = f x :=
by
  sorry

end not_periodic_f_l103_103178


namespace probability_xi_eq_12_correct_l103_103807

noncomputable def probability_xi_eq_12 : ℚ := (nat.choose 11 9) * (3 / 8) ^ 9 * (5 / 8) ^ 2 * (3 / 8)

theorem probability_xi_eq_12_correct :
  let p_red := (3 : ℚ) / 8,
      p_white := (5 : ℚ) / 8 in
  P(ξ = 12) = (nat.choose 11 9) * p_red ^ 9 * p_white ^ 2 * p_red := 
begin
  let p_red := 3 / 8,
  let p_white := 5 / 8,
  have : P(ξ = 12) = (nat.choose 11 9) * p_red ^ 9 * p_white ^ 2 * p_red,
  sorry
end

end probability_xi_eq_12_correct_l103_103807


namespace not_solution_of_equation_l103_103754

theorem not_solution_of_equation (a : ℝ) (h : a ≠ 0) : ¬ (a^2 * 1^2 + (a + 1) * 1 + 1 = 0) :=
by {
  sorry
}

end not_solution_of_equation_l103_103754


namespace problem_l103_103274

theorem problem (p q r : ℝ) (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) : r = 2000 :=
by
  sorry

end problem_l103_103274


namespace prob_two_consecutive_heads_is_half_l103_103295

noncomputable def prob_at_least_two_consecutive_heads : ℚ :=
  let total_outcomes := 16 in
  let unfavorable_states := 8 in
  let p_no_consecutive_heads := (unfavorable_states : ℚ) / (total_outcomes : ℚ) in
  1 - p_no_consecutive_heads

theorem prob_two_consecutive_heads_is_half :
  prob_at_least_two_consecutive_heads = 1 / 2 :=
by
  sorry

end prob_two_consecutive_heads_is_half_l103_103295


namespace triangle_exists_among_single_color_sticks_l103_103920

theorem triangle_exists_among_single_color_sticks
  (red yellow green : ℕ)
  (k y g K Y G : ℕ)
  (hk : k + y > G)
  (hy : y + g > K)
  (hg : g + k > Y)
  (hred : red = 100)
  (hyellow : yellow = 100)
  (hgreen : green = 100) :
  ∃ color : string, ∀ a b c : ℕ, (a = k ∨ a = K) → (b = k ∨ b = K) → (c = k ∨ c = K) → a + b > c :=
sorry

end triangle_exists_among_single_color_sticks_l103_103920


namespace incorrect_trigonometric_identity_l103_103319

theorem incorrect_trigonometric_identity 
  (α : ℝ) : 
  (∀ α, tan (π - α) = -tan α) ∧ 
  (∀ α, cos (π / 2 + α) = sin α) ∧ 
  (∀ α, sin (π + α) = -sin α) ∧ 
  (∀ α, cos (π - α) = -cos α) → 
  ¬(∀ α, cos (π / 2 + α) = -sin α) :=
sorry

end incorrect_trigonometric_identity_l103_103319


namespace smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l103_103947

theorem smallest_two_digit_multiple_of_17 : ∃ m, 10 ≤ m ∧ m < 100 ∧ 17 ∣ m ∧ ∀ n, 10 ≤ n ∧ n < 100 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

theorem smallest_four_digit_multiple_of_17 : ∃ m, 1000 ≤ m ∧ m < 10000 ∧ 17 ∣ m ∧ ∀ n, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → m ≤ n :=
by
  sorry

end smallest_two_digit_multiple_of_17_smallest_four_digit_multiple_of_17_l103_103947


namespace find_AD_l103_103119

namespace CyclicQuadrilateral

-- Assuming all values that involve degrees are represented using radians in Lean to work within the system
noncomputable def angle_deg_to_rad (α : ℝ) : ℝ := α * (Real.pi / 180)

def cyclic_quadrilateral (A B C D : Type) (circle_inscribed : ∃ O : Type, O) : Prop :=
  ∃ (AC : ℝ) (BC : ℝ) (angle_ACB : ℝ) (angle_CAD : ℝ),
    AC = 2 * Real.sqrt 3 ∧ 
    BC = Real.sqrt 6 ∧
    angle_ACB = angle_deg_to_rad 15 ∧
    angle_CAD = angle_deg_to_rad 45

theorem find_AD (A B C D : Type) (O : Type) (h : cyclic_quadrilateral A B C D O) : 
  ∃ AD : ℝ, AD = 2 * Real.sqrt 6 :=
begin
  cases h with AC h1,
  cases h1 with BC h2,
  cases h2 with angle_ACB h3,
  cases h3 with angle_CAD h4,
  cases h4 with hAC h4,
  cases h4 with hBC h5,
  cases h5 with hangle_ACB h6,
  cases h6 with hangle_CAD h7,
  use 2 * Real.sqrt 6,
  rw [←hAC, ←hBC, ←hangle_ACB, ←hangle_CAD],
  sorry,
end

end CyclicQuadrilateral

end find_AD_l103_103119


namespace lcm_consecutive_impossible_l103_103026

noncomputable def n : ℕ := 10^1000

def circle_seq := fin n → ℕ

def lcm {a b : ℕ} : ℕ := Nat.lcm a b

def lcm_seq (a : circle_seq) : fin n → ℕ :=
λ i, lcm (a i) (a (i + 1) % n)

theorem lcm_consecutive_impossible (a : circle_seq) :
  ¬ ∃ (b : fin n → ℕ), (∀ i : fin n, b i = lcm (a i) (a (i + 1) % n)) ∧ (finset.range n).pairwise (λ i j, b i + 1 = b j) :=
sorry

end lcm_consecutive_impossible_l103_103026


namespace total_amount_of_money_l103_103973

def one_rupee_note_value := 1
def five_rupee_note_value := 5
def ten_rupee_note_value := 10

theorem total_amount_of_money (n : ℕ) 
  (h : 3 * n = 90) : n * one_rupee_note_value + n * five_rupee_note_value + n * ten_rupee_note_value = 480 :=
by
  sorry

end total_amount_of_money_l103_103973


namespace determine_functions_l103_103344

noncomputable def valid_function (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋

theorem determine_functions :
  ∀ f : ℝ → ℝ,
    valid_function f →
    (f = λ x, 0) ∨ (∃ C, 1 ≤ C ∧ C < 2 ∧ f = λ x, C) :=
by
  sorry

end determine_functions_l103_103344


namespace council_revote_l103_103107

theorem council_revote (x y x' y' m : ℝ) (h1 : x + y = 500)
    (h2 : y - x = m) (h3 : x' - y' = 1.5 * m) (h4 : x' + y' = 500) (h5 : x' = 11 / 10 * y) :
    x' - x = 156.25 := by
  -- Proof goes here
  sorry

end council_revote_l103_103107


namespace combined_tax_rate_correct_l103_103537

variables (mork_income mindy_income julie_income : ℝ)

-- Conditions
def mindy_income_eq_4mork : Prop := mindy_income = 4 * mork_income
def julie_income_eq_2mork : Prop := julie_income = 2 * mork_income
def julie_income_eq_halfmindy : Prop := julie_income = (1 / 2) * mindy_income

-- Tax rates
def mork_tax_rate : ℝ := 0.45
def mindy_tax_rate : ℝ := 0.25
def julie_tax_rate : ℝ := 0.35

-- Total income and total tax calculations
def total_income : ℝ := mork_income + mindy_income + julie_income
def total_tax : ℝ :=
  mork_tax_rate * mork_income + mindy_tax_rate * mindy_income + julie_tax_rate * julie_income

-- Combined tax rate
def combined_tax_rate : ℝ := total_tax / total_income

-- Assertion to prove
theorem combined_tax_rate_correct 
  (h1 : mindy_income_eq_4mork)
  (h2 : julie_income_eq_2mork)
  (h3 : julie_income_eq_halfmindy) :
  combined_tax_rate mork_income mindy_income julie_income = 2.15 / 7 :=
by
  -- Proof would go here
  sorry

end combined_tax_rate_correct_l103_103537


namespace sphere_volume_l103_103803

-- Define the surface area condition
def surface_area_condition (r : ℝ) : Prop :=
  4 * Real.pi * r^2 = 12 * Real.pi

-- Define the volume formula
def volume_formula (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

-- The main theorem stating that given the surface area of 12π, the volume is 4√3π.
theorem sphere_volume (r : ℝ) (h : surface_area_condition r) : volume_formula r = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end sphere_volume_l103_103803


namespace odd_function_monotonic_decreasing_l103_103748

noncomputable def f (a x : ℝ) : ℝ :=
  if x > 0 then -x^2 + a*x + a + 1 else -3*a + 3

theorem odd_function_monotonic_decreasing {a : ℝ} (ha : a ≤ 0) : 
  ∀ x, f a x = (if x = 0 then a + 1 else if x > 0 then -x^2 + a*x + a + 1 else -3*a + 3) ∧
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ ≥ f a x₂) :=
begin
  sorry
end

end odd_function_monotonic_decreasing_l103_103748


namespace geometric_sequence_sixth_term_l103_103892

theorem geometric_sequence_sixth_term (a : ℝ) (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^(7) = 2) :
  a * r^(5) = 16 :=
by
  sorry

end geometric_sequence_sixth_term_l103_103892


namespace total_cost_price_l103_103293

variables (C_table C_chair C_shelf : ℝ)

axiom h1 : 1.24 * C_table = 8091
axiom h2 : 1.18 * C_chair = 5346
axiom h3 : 1.30 * C_shelf = 11700

theorem total_cost_price :
  C_table + C_chair + C_shelf = 20055.51 :=
sorry

end total_cost_price_l103_103293


namespace value_of_x_plus_y_l103_103797

theorem value_of_x_plus_y (x y : ℤ) (h1 : x + 2 = 10) (h2 : y - 1 = 6) : x + y = 15 :=
by
  sorry

end value_of_x_plus_y_l103_103797


namespace minimum_correct_answers_l103_103282

theorem minimum_correct_answers (total_questions correct_percentage : ℕ) (percent_threshold : ℝ) 
  (H1 : total_questions = 50) (H2 : correct_percentage = 70) (H3 : percent_threshold = (70: ℝ) / 100) :
  (total_questions * percent_threshold).ceil.toNat = 35 := by
{ sorry }

end minimum_correct_answers_l103_103282


namespace hyperbola_eq_of_eccentricity_and_foci_l103_103412

noncomputable def hyperbola_equation (a b : ℝ) : AdditiveGroup ℝ := 
(x : ℝ) (y : ℝ) := x^2 / a^2 - y^2 / b^2 - 1

theorem hyperbola_eq_of_eccentricity_and_foci : 
  (eccentricity = 2) ∧ (foci_x = 4) →
  hyperbola_equation 2 (2 * Real.sqrt 3) = 
  hyperbola_equation 2 6 := 
  sorry

end hyperbola_eq_of_eccentricity_and_foci_l103_103412


namespace marys_birthday_l103_103866

theorem marys_birthday (M : ℝ) (h1 : (3 / 4) * M - (3 / 20) * M = 60) : M = 100 := by
  -- Leave the proof as sorry for now
  sorry

end marys_birthday_l103_103866


namespace max_moves_initial_set_l103_103505

def initial_set : Finset ℕ := Finset.range 21

def valid_move (a b : ℕ) : Prop := a < b ∧ b - a ≥ 2

def new_numbers (a b : ℕ) : Finset ℕ := {a + 1, b - 1}

theorem max_moves_initial_set : ∃ moves : ℕ, 
  moves = 9 ∧
  ∀ (S : Finset ℕ), 
    (S = initial_set ∨ ∃ a b, valid_move a b ∧ S = (S.erase a).erase b ∪ new_numbers a b) →
      ∃ moves' : ℕ, 
        moves' ≤ moves :=
begin
  sorry
end

end max_moves_initial_set_l103_103505


namespace min_sample_variance_l103_103814

theorem min_sample_variance
  (a1 a2 a3 : ℝ) (a4 a5 : ℝ) (sum_a4_a5 : a4 + a5 = 5) :
  a1 = 2.5 → a2 = 3.5 → a3 = 4 →
  let mean := (a1 + a2 + a3 + a4 + a5) / 5 in
  let variance := (1 / 5) * ((a1 - mean) ^ 2 + (a2 - mean) ^ 2 + (a3 - mean) ^ 2 + (a4 - mean) ^ 2 + (a5 - mean) ^ 2) in
  mean = 3 ∧ variance = (2 / 5) * (2.5 - mean) ^ 2 + (2 / 5) :=
  a4 = 2.5 ∧ a5 = 2.5 :=
sorry

end min_sample_variance_l103_103814


namespace vector_magnitude_and_angle_l103_103774

variable {Real : Type} [RealField Real]

def magnitude (v : Real × Real) : Real :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

def dot_product (v w : Real × Real) : Real :=
  v.1 * w.1 + v.2 * w.2

def cos_angle (v w : Real × Real) : Real :=
  (dot_product v w) / (magnitude v * magnitude w)

theorem vector_magnitude_and_angle 
  (a b : Real × Real) 
  (ha : a = (Real.sqrt 3, 1))
  (hb : b = (1, -Real.sqrt 3 / 3)) : 
  magnitude a = 2 ∧ Real.arccos (cos_angle a b) = Real.pi / 3 :=
by
  sorry

end vector_magnitude_and_angle_l103_103774


namespace gcd_n_cube_plus_16_n_plus_4_l103_103372

theorem gcd_n_cube_plus_16_n_plus_4 (n : ℕ) (h1 : n > 16) : 
  Nat.gcd (n^3 + 16) (n + 4) = Nat.gcd 48 (n + 4) :=
by
  sorry

end gcd_n_cube_plus_16_n_plus_4_l103_103372


namespace range_of_quadratic_l103_103756

def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem range_of_quadratic (a : ℝ) (f : ℝ → ℝ)
    (h : f = λ x, quadratic (a^2 - 2*a - 3) (a - 3) 1 x) :
    (∃ x : ℝ, ∃ y : ℝ, f x = y) ↔ (a > 3 ∨ a < -1) :=
sorry

end range_of_quadratic_l103_103756


namespace inequality_a_b_cubed_l103_103046

theorem inequality_a_b_cubed (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^3 < b^3 :=
sorry

end inequality_a_b_cubed_l103_103046


namespace minimize_slope_at_one_l103_103047

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * x^2 - (1 / (a * x))

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  4 * a * x - (1 / (a * x^2))

noncomputable def slope_at_one (a : ℝ) : ℝ :=
  f_deriv a 1

theorem minimize_slope_at_one : ∀ a : ℝ, a > 0 → slope_at_one a ≥ 4 ∧ (slope_at_one a = 4 ↔ a = 1 / 2) :=
by 
  sorry

end minimize_slope_at_one_l103_103047


namespace emma_knutt_investment_l103_103354

theorem emma_knutt_investment :
  ∃ (x y : ℝ), x + y = 10000 ∧ 0.09 * x + 0.11 * y = 980 ∧ x = 6000 ∧ y = 4000 :=
by
  use 6000
  use 4000
  split
  -- x + y = 10000
  { exact by norm_num },
  split
  -- 0.09 * x + 0.11 * y = 980
  { exact by norm_num },
  split
  -- x = 6000
  { refl },
  -- y = 4000
  { refl }

end emma_knutt_investment_l103_103354


namespace vector_equality_dot_product_sufficiency_l103_103400

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

theorem vector_equality_dot_product_sufficiency :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → (a = b ↔ a • c = b • c) :=
by sorry

end vector_equality_dot_product_sufficiency_l103_103400


namespace min_radius_cover_four_points_l103_103008

theorem min_radius_cover_four_points (A B C D : Real × Real)
  (hAB : dist A B ≤ 1) (hAC : dist A C ≤ 1) (hAD : dist A D ≤ 1)
  (hBC : dist B C ≤ 1) (hBD : dist B D ≤ 1) (hCD : dist C D ≤ 1) :
  ∃ (R : ℝ), (∀ P ∈ {A, B, C, D}, dist P (circle_center A B C D) ≤ R) ∧ R = sqrt 3 / 3 :=
sorry

end min_radius_cover_four_points_l103_103008


namespace max_value_condition_passes_through_condition_symmetry_condition_find_f_expression_intervals_of_increase_g_l103_103423

noncomputable def f (x : ℝ) : ℝ := 10 * Real.sin (2 * x + Real.pi / 6)

theorem max_value_condition : ∃ (A ω : ℝ), ∀ (x : ℝ), 0 < A ∧ 0 < ω ∧ |Real.sin(ω * x + Real.pi / 6)| ≤ A ∧ is_local_max_on f (Real.sin(ω * x + Real.pi / 6)) (set.univ) := 
sorry

theorem passes_through_condition : f 0 = 5 :=
sorry

theorem symmetry_condition : (∀ x, (f (x + Real.pi / 2) = f(x))) :=
sorry

theorem find_f_expression : f = (λ x, 10 * Real.sin (2*x + Real.pi / 6)) :=
sorry

noncomputable def g (x : ℝ) : ℝ := 10 * Real.sin (2 * x - Real.pi / 6)

theorem intervals_of_increase_g : ∀ k : ℤ, (∀ x : ℝ, -Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi → increasing (g x)) :=
sorry

end max_value_condition_passes_through_condition_symmetry_condition_find_f_expression_intervals_of_increase_g_l103_103423


namespace find_constants_u_v_l103_103072

theorem find_constants_u_v
  (n p r1 r2 : ℝ)
  (h1 : r1 + r2 = n)
  (h2 : r1 * r2 = p) :
  ∃ u v, (r1^4 + r2^4 = -u) ∧ (r1^4 * r2^4 = v) ∧ u = -(n^4 - 4*p*n^2 + 2*p^2) ∧ v = p^4 :=
by
  sorry

end find_constants_u_v_l103_103072


namespace tank_emptying_time_l103_103312

theorem tank_emptying_time
  (initial_volume : ℝ)
  (filling_rate : ℝ)
  (emptying_rate : ℝ)
  (initial_fraction_full : initial_volume = 1 / 5)
  (pipe_a_rate : filling_rate = 1 / 10)
  (pipe_b_rate : emptying_rate = 1 / 6) :
  (initial_volume / (filling_rate - emptying_rate) = 3) :=
by
  sorry

end tank_emptying_time_l103_103312


namespace Lin_finishes_reading_on_Monday_l103_103534

theorem Lin_finishes_reading_on_Monday :
  let start_day := "Tuesday"
  let book_days : ℕ → ℕ := fun n => n
  let total_books := 10
  let total_days := (total_books * (total_books + 1)) / 2
  let days_in_a_week := 7
  let finish_day_offset := total_days % days_in_a_week
  let day_names := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  (day_names.indexOf start_day + finish_day_offset) % days_in_a_week = day_names.indexOf "Monday" :=
by
  sorry

end Lin_finishes_reading_on_Monday_l103_103534


namespace average_of_second_and_fourth_l103_103342

def numbers : List ℤ := [-1, 3, 5, 8, 10]

def satisfiesCondition1 (ls : List ℤ) : Prop :=
ls.index_of 10 ≠ 1 ∧ ls.index_of 10 ∈ [2, 3, 4]

def satisfiesCondition2 (ls : List ℤ) : Prop :=
ls.index_of (-1) ≠ 0 ∧ ls.index_of (-1) ∈ [1, 2]

def satisfiesCondition3 (ls : List ℤ) : Prop :=
ls.index_of 5 ≠ 2

def averageSecondAndFourth (ls : List ℤ) : ℤ :=
(ls.get! 1 + ls.get! 3) / 2

theorem average_of_second_and_fourth :
  satisfiesCondition1 numbers ∧ satisfiesCondition2 numbers ∧ satisfiesCondition3 numbers →
  averageSecondAndFourth numbers = 4.5 := by
sorry

end average_of_second_and_fourth_l103_103342


namespace diego_apples_weight_l103_103703

-- Definitions based on conditions
def bookbag_capacity : ℕ := 20
def weight_watermelon : ℕ := 1
def weight_grapes : ℕ := 1
def weight_oranges : ℕ := 1

-- Lean statement to check
theorem diego_apples_weight : 
  bookbag_capacity - (weight_watermelon + weight_grapes + weight_oranges) = 17 :=
by
  sorry

end diego_apples_weight_l103_103703


namespace pencil_and_pen_choice_count_l103_103607

-- Definitions based on the given conditions
def numPencilTypes : Nat := 4
def numPenTypes : Nat := 6

-- Statement we want to prove
theorem pencil_and_pen_choice_count : (numPencilTypes * numPenTypes) = 24 :=
by
  sorry

end pencil_and_pen_choice_count_l103_103607


namespace Triangle_Equality_l103_103806

noncomputable theory
open EuclideanGeometry

variables {A C E B D F : Point}

-- Conditions
axiom Points_on_lines : OnLine B (Segment A C) ∧ OnLine D (Segment A E)
axiom Intersection_property : ∃ F, IntersectAt CD BE F 
axiom Given_condition : distance A B + distance B F = distance A D + distance D F

-- To prove
theorem Triangle_Equality :
  ∀ (P : Triangle),
  P ⟨A, C, E⟩ → 
  Points_on_lines → 
  Intersection_property → 
  Given_condition → 
  distance A C + distance C F = distance A E + distance E F := 
by
  intros P h₁ h₂ h₃ h₄
  sorry

end Triangle_Equality_l103_103806


namespace min_expression_value_l103_103139

theorem min_expression_value (n : ℕ) (hn : n ≠ 0) 
  (x : Fin n.succ → ℝ) (hx : ∀ i, 0 < x i) 
  (sum_reciprocals : ∑ i, (1 / x i) = n) :
  ∑ i, (x i) + ∑ i in (Finset.range n.succ).erase 0, (x i) ^ i * i⁻¹ = ∑ i in Finset.range n.succ, i⁻¹ :=
sorry

end min_expression_value_l103_103139


namespace arithmetic_sequence_a_common_terms_C_exists_m_n_l103_103146

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) - a n = d

def a_n (n : ℕ) : ℤ := 2 * n - 7
def b_n (n : ℕ) : ℤ := 3 * n - 11
def C_n (n : ℕ) : ℤ := 6 * n + 1

theorem arithmetic_sequence_a (a : ℕ → ℤ) (d : ℤ) (h : is_arithmetic_sequence a d) 
  (h₁ : a 6 = 5) (h₂ : a 2 ^ 2 + a 3 ^ 2 = a 4 ^ 2 + a 5 ^ 2) : 
  ∀ n, a n = 2 * n - 7 := sorry

theorem common_terms_C :
  ∀ n, C_n n = 6 * n + 1 := sorry

theorem exists_m_n (m n : ℕ) (h₁ : m ≠ 5) (h₂ : n ≠ 5) (h₃ : m ≠ n) :
  (m = 11 ∧ n = 1) ∨ (m = 2 ∧ n = 3) ∨ (m = 6 ∧ n = 11) :=
  sorry

end arithmetic_sequence_a_common_terms_C_exists_m_n_l103_103146


namespace number_of_subsets_l103_103906

def M := {1, 2, 3}

theorem number_of_subsets : (M.powerset.card = 8) :=
by
  sorry

end number_of_subsets_l103_103906


namespace earnings_end_of_fourth_month_l103_103557

noncomputable def initial_salary : ℝ := 2000
noncomputable def bonus_first_month : ℝ := 150
noncomputable def salary_increase_percentage (month : ℕ) : ℝ :=
  if month = 1 then 0.05 else 0.05 * 2^(month - 1)
noncomputable def bonus_percentage : ℝ := 0.10

def salary (n : ℕ) : ℝ :=
  if n = 1 then initial_salary + salary_increase_percentage 1 * initial_salary
  else
    let prev_salary := salary (n - 1)
    in prev_salary + (salary_increase_percentage n * prev_salary)

def bonus (n : ℕ) : ℝ :=
  if n = 1 then bonus_first_month
  else
    let prev_bonus := bonus (n - 1)
    in prev_bonus + bonus_percentage * prev_bonus

def total_earnings (n : ℕ) : ℝ :=
  salary n + bonus n

theorem earnings_end_of_fourth_month :
  total_earnings 4 = 4080.45 :=
  by sorry

end earnings_end_of_fourth_month_l103_103557


namespace find_complex_number_l103_103967

open Complex

theorem find_complex_number (z : ℂ) (hz : z + Complex.abs z = Complex.ofReal 2 + 8 * Complex.I) : 
z = -15 + 8 * Complex.I := by sorry

end find_complex_number_l103_103967


namespace inequality_holds_for_all_x_l103_103097

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x)) ↔ -2 < m ∧ m ≤ 2 := 
by
  sorry

end inequality_holds_for_all_x_l103_103097


namespace digit_143_in_decimal_rep_of_3_over_11_l103_103950

theorem digit_143_in_decimal_rep_of_3_over_11 :
  let decimal_rep := "27"
  let period := 2
  let digit_position := 143
  (decimal_rep.toNat digit_position % period) + 1 == 2 :=
by
  sorry

end digit_143_in_decimal_rep_of_3_over_11_l103_103950


namespace cos_identity_proof_l103_103569

noncomputable def cos_eq_half : Prop :=
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7)) = 1 / 2

theorem cos_identity_proof : cos_eq_half :=
  by sorry

end cos_identity_proof_l103_103569


namespace evaluate_f_5_times_l103_103857

def f (x : ℝ) : ℝ :=
if x ≥ 0 then -(x + 3)^2 else x + 5

theorem evaluate_f_5_times : f (f (f (f (f 2)))) = -5 :=
by
  sorry

end evaluate_f_5_times_l103_103857


namespace emily_necklaces_l103_103353

theorem emily_necklaces (total_beads : ℕ) (beads_per_necklace : ℕ) (necklaces_made : ℕ) 
  (h1 : total_beads = 52)
  (h2 : beads_per_necklace = 2)
  (h3 : necklaces_made = total_beads / beads_per_necklace) :
  necklaces_made = 26 :=
by
  rw [h1, h2] at h3
  exact h3

end emily_necklaces_l103_103353


namespace find_c_l103_103437

variable (a b c : ℝ)
variable (a_n b_n : ℕ → ℝ)

def a_n_def (n : ℕ) : ℝ := (a * n^2 + 3) / (b * n^2 - 2 * n + 2)
def b_n_def (n : ℕ) : ℝ := b - a * (1 / 3 : ℝ)^(n-1)

-- Conditions
axiom lim_a_n : filter.tendsto a_n_def filter.at_top (nhds 3)
axiom lim_b_n : filter.tendsto b_n_def filter.at_top (nhds (-1 / 4))
axiom arithmetic_sequence : 2 * b = a + c

-- Proposition to prove
theorem find_c : c = 1 / 4 := sorry

end find_c_l103_103437


namespace sqrt_domain_l103_103472

theorem sqrt_domain (x : ℝ) : (∃ y, y * y = x - 2) ↔ (x ≥ 2) :=
by sorry

end sqrt_domain_l103_103472


namespace min_newspapers_to_earn_back_scooter_cost_l103_103168

theorem min_newspapers_to_earn_back_scooter_cost
  (scooter_cost : ℕ)
  (earn_per_newspaper : ℕ) 
  (cost_per_newspaper : ℕ) :
  (earn_per_newspaper - cost_per_newspaper) * 750 ≥ scooter_cost :=
by
  let net_earning_per_newspaper := earn_per_newspaper - cost_per_newspaper
  have h : net_earning_per_newspaper * 750 = 3000 := sorry
  show net_earning_per_newspaper * 750 ≥ scooter_cost from h.symm ▸ sorry

end min_newspapers_to_earn_back_scooter_cost_l103_103168


namespace three_digit_numbers_count_l103_103621

theorem three_digit_numbers_count : 
  let digits := [1, 2, 3, 4, 5] in
  (∃ n : ℕ, n = 60 ∧ (∃ l : List ℕ, l.length = 3 ∧ (∀ d ∈ l, d ∈ digits) ∧ l.nodup) ∧ 
               (∃ perm : List (List ℕ), perm = l.permutations)) :=
sorry

end three_digit_numbers_count_l103_103621


namespace number_of_valid_arrangements_l103_103276

-- We define our problem as a constant value we aim to prove
constant num_ways_to_place_numbers : ℕ

-- Given conditions: Each four-cell triangle sums to 23 and specific cells must contain certain numbers
axiom sum_in_each_subtriangle_is_23 (numbers : Finset ℕ) (triangle : List (Finset ℕ)) 
  (h_valid_numbers : numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}) : 
  triangle.length = 3 → 
  ∀ t ∈ triangle, t.card = 4 ∧ (t.sum id = 23)

axiom given_numbers_in_specific_cells (specific_cells : List ℕ) (numbers : Finset ℕ)
  (h_valid_numbers : numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}) : 
  ∀ c ∈ specific_cells, c ∈ numbers

-- The answer to the problem
theorem number_of_valid_arrangements : num_ways_to_place_numbers = 4 := 
sorry

end number_of_valid_arrangements_l103_103276


namespace radius_of_circle_l103_103708

theorem radius_of_circle
  (r : ℝ)
  (h1 : ∀ x : ℝ, (x^2 + r = x) → (x^2 - x + r = 0) → ((-1)^2 - 4 * 1 * r = 0)) :
  r = 1/4 :=
sorry

end radius_of_circle_l103_103708


namespace boy_age_half_years_ago_l103_103974

theorem boy_age_half_years_ago (x : ℕ) : (10 - x = 5) → (x = 5) := by
  intro h
  exact eq_of_sub_eq_sub_left h

end boy_age_half_years_ago_l103_103974


namespace distinct_four_digit_integers_digit_product_18_l103_103780

theorem distinct_four_digit_integers_digit_product_18 :
  ∃ n : ℕ, n = 48 ∧ ∀ (a b c d : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9) ∧ (a * b * c * d = 18) →
     multiset.card (multiset.of_list [a, b, c, d]) = 4 :=
sorry

end distinct_four_digit_integers_digit_product_18_l103_103780


namespace part1_part2_l103_103075

-- Define the vectors m and n
def m (x : Real) : Vector Real := ⟨sqrt 3 * sin (2 * x) + 2, cos x⟩
def n (x : Real) : Vector Real := ⟨1, 2 * cos x⟩

-- Define the function f
def f (x : Real) : Real := m x ⬝ n x

-- Statement of the smallest positive period and monotonically increasing interval
theorem part1 : (∃ T, T = π ∧ (∀ k : Int, (k * π - π / 3 ≤ x) ∧ (x ≤ k * π + π / 6))) :=
  sorry

-- Statement for the maximum value of b + c given a and f(A) = 4 in a triangle
theorem part2 (a : Real) (A : Real) (b c : Real) (h₁: a = sqrt 3) (h₂: f A = 4) :
  (b + c ≤ 2 * sqrt 3) :=
  sorry

end part1_part2_l103_103075


namespace solution_l103_103631

noncomputable def problem : Prop :=
  (2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2) ∧
  (1 - 2 * Real.sin (Real.pi / 12) ^ 2 ≠ 1 / 2) ∧
  (Real.cos (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 
   Real.sin (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 2) ∧
  ( (Real.tan (77 * Real.pi / 180) - Real.tan (32 * Real.pi / 180)) /
    (2 * (1 + Real.tan (77 * Real.pi / 180) * Real.tan (32 * Real.pi / 180))) = 1 / 2 )

theorem solution : problem :=
  by 
    sorry

end solution_l103_103631


namespace angle_AC_B₁C₁_is_60_l103_103824

-- Redefine the conditions of the problem using Lean definitions
-- We define a regular triangular prism, equilateral triangle condition,
-- and parallel lines relation.

structure TriangularPrism :=
  (A B C A₁ B₁ C₁ : Type)
  (is_regular : Prop) -- Property stating it is a regular triangular prism
  (base_is_equilateral : Prop) -- Property stating the base is an equilateral triangle
  (B₁C₁_parallel_to_BC : Prop) -- Property stating B₁C₁ is parallel to BC

-- Assume a regular triangular prism with the given properties
variable (prism : TriangularPrism)
axiom isRegularPrism : prism.is_regular
axiom baseEquilateral : prism.base_is_equilateral
axiom parallelLines : prism.B₁C₁_parallel_to_BC

-- Define the angle calculation statement in Lean 4
theorem angle_AC_B₁C₁_is_60 :
  ∃ (angle : ℝ), angle = 60 :=
by
  -- Proof is omitted using sorry
  exact ⟨60, sorry⟩

end angle_AC_B₁C₁_is_60_l103_103824


namespace triangles_have_equal_area_l103_103491

theorem triangles_have_equal_area
  {A B C X Y Z : Type}
  [convex_hexagon AXBYCZ : Prop]
  (h1 : AX ∥ BC) 
  (h2 : BY ∥ XC) 
  (h3 : CZ ∥ XY) :
  (area_of_triangle ABC = area_of_triangle XYZ) :=
sorry

end triangles_have_equal_area_l103_103491


namespace factorization_correct_l103_103206

theorem factorization_correct {c d : ℤ} (h1 : c + 4 * d = 4) (h2 : c * d = -32) :
  c - d = 12 :=
by
  sorry

end factorization_correct_l103_103206


namespace correct_option_l103_103263

-- Define the propositions for each option
def optionA (x : ℝ) : Prop := x^2 + x^2 = 2x
def optionB (a : ℝ) : Prop := a^2 * a^3 = a^5
def optionC (x : ℝ) : Prop := (-2 * x^2)^4 = 16 * x^6
def optionD (x y : ℝ) : Prop := (x + 3 * y) * (x - 3 * y) = x^2 - 3 * y^2

-- The theorem stating that option B is the only correct option.
theorem correct_option : ∃ (a : ℝ), optionB a ∧ 
  (∀ (x : ℝ), ¬optionA x) ∧ 
  (∀ (x : ℝ), ¬optionC x) ∧ 
  (∀ (x y : ℝ), ¬optionD x y) :=
by
  sorry

end correct_option_l103_103263


namespace area_of_quadrilateral_PTXM_l103_103691

section geometry_problem

-- Defining the basic sides and angles of the polygon
def sides (P Q R S T V W X Y Z : Point) : Prop :=
  (dist P Q = 3) ∧ (dist Q R = 3) ∧ (dist R S = 3) ∧ (dist S T = 3) ∧ (dist T V = 3) ∧ 
  (dist V W = 3) ∧ (dist W X = 3) ∧ (dist X Y = 3) ∧ (dist Y Z = 3) ∧ (dist Z P = 3)

def angles (P Q R S T V W X Y Z : Point) : Prop :=
  (angle QPR = π/3) ∧ (angle RQS = π/3) ∧ (angle SQT = π/3) ∧ 
  (angle WXT = π/3) ∧ (angle XWY = π/3) ∧ (angle YWZ = π/3) ∧
  (angle PQV = π/3) ∧ (angle TYV = π/2) ∧ (angle ZYX = π/2)

-- Defining the polygon PQRSTVWXYZ
def polygon (P Q R S T V W X Y Z : Point) : Prop :=
  sides P Q R S T V W X Y Z ∧ angles P Q R S T V W X Y Z

-- Intersection at point M for segments PV and TX
def intersection_at (P T V X M : Point) : Prop :=
  ∃ M, lies_on_line PV M ∧ lies_on_line TX M

-- Using the given conditions and trying to prove the area of PTXM is 18
theorem area_of_quadrilateral_PTXM (P Q R S T V W X Y Z M : Point)
  (hpoly : polygon P Q R S T V W X Y Z) (hint : intersection_at P T V X M) :
  area (quadrilateral P T X M) = 18 :=
sorry

end geometry_problem

end area_of_quadrilateral_PTXM_l103_103691


namespace cubic_roots_number_l103_103847

noncomputable def determinant_cubic (a b c d : ℝ) (x : ℝ) : ℝ :=
  x * (x^2 + a^2) + c * (b * x + a * b) - b * (c * a - b * x)

theorem cubic_roots_number (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ roots : ℕ, (roots = 1 ∨ roots = 3) :=
  sorry

end cubic_roots_number_l103_103847


namespace minimal_distance_l103_103202

noncomputable def minimum_distance_travel (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) : ℝ :=
  2 * Real.sqrt 19

theorem minimal_distance (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) :
  minimum_distance_travel a b c ha hb hc = 2 * Real.sqrt 19 :=
by
  -- Proof is omitted
  sorry

end minimal_distance_l103_103202


namespace range_of_k_and_summation_inequality_l103_103429

noncomputable def f (k x : ℝ) : ℝ := k * x^2 - log x

theorem range_of_k_and_summation_inequality :
  (∀ x : ℝ, x > 0 → f k x ≥ 0) → k ∈ Icc (1 / (2 * Real.exp 1)) ⊤ ∧ 
  ∀ n : ℕ, 2 ≤ n → ∑ i in Finset.range n, (log (1 + i) / (1 + i)^2) < (n - 1) / (2 * Real.exp 1) := sorry

end range_of_k_and_summation_inequality_l103_103429


namespace arithmetic_geometric_sequence_l103_103057

noncomputable def sequence {α : Type*} (a_2 : α) (common_diff : α) : ℕ → α
| 1 := a_2 - common_diff
| 2 := a_2
| 5 := a_2 + 3 * common_diff
| _ := sorry -- Other terms are not defined by our conditions.

theorem arithmetic_geometric_sequence (a_2 : ℝ) (common_diff : ℝ) (h : common_diff = 2) 
  (geom : (a_2 - common_diff) * (a_2 + 3 * common_diff) = a_2^2) : 
  a_2 = 3 :=
by sorry

end arithmetic_geometric_sequence_l103_103057


namespace cosine_shift_left_l103_103927

theorem cosine_shift_left (x : ℝ) : cos x = cos (x - (-π/5)) :=
by
  sorry

end cosine_shift_left_l103_103927


namespace total_volume_of_four_boxes_l103_103258

-- Define the edge length of the cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_one_cube := edge_length ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 4

-- The total volume of the four cubes
def total_volume := number_of_cubes * volume_of_one_cube

-- Statement to prove that the total volume equals 500 cubic feet
theorem total_volume_of_four_boxes :
  total_volume = 500 :=
sorry

end total_volume_of_four_boxes_l103_103258


namespace mutual_fund_overall_change_l103_103515

variable y : ℝ

theorem mutual_fund_overall_change (y : ℝ) : 
  let value_after_first_day := 1.1 * y in
  let value_after_second_day := value_after_first_day * 0.85 in
  let overall_change := (value_after_second_day - y) / y in
  overall_change * 100 = -6.5 :=
by
  sorry

end mutual_fund_overall_change_l103_103515


namespace selling_price_l103_103573

noncomputable def selling_price_in_currency_B :=
  let cost_in_B : ℚ := 9000 / 2 in
  let original_repair_cost : ℚ := 5000 / 0.80 in
  let total_cost : ℚ := cost_in_B + original_repair_cost + 1000 in
  let import_tax : ℚ := 0.05 * total_cost in
  let total_cost_incl_tax : ℚ := total_cost + import_tax in
  let profit : ℚ := 0.50 * total_cost_incl_tax in
  total_cost_incl_tax + profit

theorem selling_price : selling_price_in_currency_B = 18506.25 := by
  sorry

end selling_price_l103_103573


namespace find_A_and_c_l103_103102

noncomputable def triangle_sides (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 + b * c

noncomputable def measures_A (A : ℝ) : Prop :=
  A = 2 * π / 3

noncomputable def measures_c (c : ℝ) : Prop :=
  c = 2

theorem find_A_and_c (a b c A : ℝ) (h1 : triangle_sides a b c) (h2 : a = 2 * real.sqrt(3)) (h3 : b = 2) :
  measures_A A ∧ measures_c c :=
begin
  sorry
end

end find_A_and_c_l103_103102


namespace probability_P_plus_S_mod_7_correct_l103_103935

noncomputable def probability_P_plus_S_mod_7 : ℚ :=
  let n := 60
  let total_ways := (n * (n - 1)) / 2
  let num_special_pairs := total_ways - ((52 * 51) / 2)
  num_special_pairs / total_ways

theorem probability_P_plus_S_mod_7_correct :
  probability_P_plus_S_mod_7 = 148 / 590 :=
by
  rw [probability_P_plus_S_mod_7]
  sorry

end probability_P_plus_S_mod_7_correct_l103_103935


namespace part1_part2_part2_correct_part3_correct_l103_103828

-- Definitions and conditions for the triangle ABC
def sin_ratio_A_B_C : ℝ → ℝ → ℝ → Prop := λ A B C, sin A / sin B = 2 / 1 ∧ sin B / sin C = 1 / Real.sqrt 2
def side_b : ℝ := Real.sqrt 2
def side_a : ℝ := 2 * side_b
def side_c : ℝ := Real.sqrt 2 * side_b

-- Part 1: Proving the value of a
theorem part1 (sinABC : ∀ A B C, sin_ratio_A_B_C A B C): side_a = 2 * Real.sqrt 2 := 
by sorry

-- Part 2: Proving the value of cos C
theorem part2 (a b c : ℝ) (ha : a = side_a) (hb : b = side_b) (hc : c = side_c) :
  cos C = (a^2 + b^2 - c^2) / (2 * a * b) :=
by sorry

theorem part2_correct (ha : side_a = 2 * Real.sqrt 2) :
  cos C = 3 / 4 := 
by sorry

-- Part 3: Proving the value of sin(2C - π / 6)
theorem part3_correct (hcosC : cos C = 3 / 4) :
  sin (2 * C - π / 6) = (3 * Real.sqrt 21 - 1) / 16 :=
by sorry

end part1_part2_part2_correct_part3_correct_l103_103828


namespace total_soaps_total_soaps_proof_l103_103963

def soapParts : Nat := 11
def scrapParts : Nat := 1
def initialScraps : Nat := 251

theorem total_soaps : Nat :=
  let rec count_soaps (scraps : Nat) (total : Nat) : Nat :=
    if scraps < soapParts then total
    else count_soaps (scraps % soapParts + (scraps / soapParts)) (total + (scraps / soapParts))
  count_soaps initialScraps 0

theorem total_soaps_proof : total_soaps = 25 := by
  sorry

end total_soaps_total_soaps_proof_l103_103963


namespace number_of_possible_values_S_l103_103165

theorem number_of_possible_values_S :
  ∀ (S : ℕ) (A : Finset ℕ),
    (A.card = 70) ∧ (A ⊆ Finset.range 121) →
    2485 ≤ S ∧ S ≤ 5985 →
    ∃ P : set ℕ, P.card = 3501 :=
by sorry

end number_of_possible_values_S_l103_103165


namespace area_ratio_l103_103143

variables (A B C D : Point)
variable (convex : ConvexQuad A B C D)
variable (G_A : Point) (G_B : Point) (G_C : Point) (G_D : Point)
variable (ratios : (dist A B / dist G_A B = 4) ∧ (dist B C / dist G_B C = 4) ∧ (dist C D / dist G_C D = 4) ∧ (dist D A / dist G_D A = 4))

theorem area_ratio (h : ConvexQuad A B C D) (r : ratios) :
  area (Quad G_A G_B G_C G_D) / area (Quad A B C D) = 1 / 9 :=
sorry

end area_ratio_l103_103143


namespace sqrt_domain_l103_103471

theorem sqrt_domain (x : ℝ) : (∃ y, y * y = x - 2) ↔ (x ≥ 2) :=
by sorry

end sqrt_domain_l103_103471


namespace two_common_tangents_iff_intersect_l103_103077

theorem two_common_tangents_iff_intersect (r : ℝ) (h : r > 0) :
  (∀ t : ℝ, (t^2 + (sqrt(5)^2 - t^2)^2 = 16) ↔ (t - r)^2 = 4 ∨ (t + r)^2 = 4) ↔
  sqrt(5) - 2 < r ∧ r < sqrt(5) + 2 :=
sorry

end two_common_tangents_iff_intersect_l103_103077


namespace max_min_r_diff_l103_103158

noncomputable theory

open Real

theorem max_min_r_diff (p q r : ℝ) (h1 : p + q + r = 5) (h2 : p^2 + q^2 + r^2 = 27) :
  abs ((5 - (p + q)) + 8 * sqrt 7 / 6 - ((5 - (p + q)) - 8 * sqrt 7 / 6)) = 8 * sqrt 7 / 3 :=
sorry

end max_min_r_diff_l103_103158


namespace product_of_removed_terms_sum_to_one_l103_103686

theorem product_of_removed_terms_sum_to_one :
  let given_sum := [ 1/2, 1/4, 1/6, 1/8, 1/10, 1/12 ],
      removed_terms := [ 1/8, 1/10 ] in
  (sum (given_sum.diff removed_terms) = 1) ∧
  (prod removed_terms = 1/80) := 
by
  sorry

end product_of_removed_terms_sum_to_one_l103_103686


namespace students_problem_count_l103_103613

theorem students_problem_count 
  (x y z q r : ℕ) 
  (H1 : x + y + z + q + r = 30) 
  (H2 : x + 2 * y + 3 * z + 4 * q + 5 * r = 40) 
  (h_y_pos : 1 ≤ y) 
  (h_z_pos : 1 ≤ z) 
  (h_q_pos : 1 ≤ q) 
  (h_r_pos : 1 ≤ r) : 
  x = 26 := 
  sorry

end students_problem_count_l103_103613


namespace brand_tea_ratio_l103_103330

theorem brand_tea_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) :
  let unit_price_A := (0.85 * p) / (1.3 * v)
      unit_price_B := p / v
  in (unit_price_A / unit_price_B) = (17 / 26) :=
by
  -- Calculation steps are omitted according to instructions
  sorry

end brand_tea_ratio_l103_103330


namespace add_to_1_eq_62_l103_103627

theorem add_to_1_eq_62 :
  let y := 5 * 12 / (180 / 3)
  ∃ x, y + x = 62 ∧ x = 61 :=
by
  sorry

end add_to_1_eq_62_l103_103627


namespace remaining_area_fits_9_circles_l103_103484

theorem remaining_area_fits_9_circles :
  ∀ (r_large r_small r_additional : ℝ) (n_small n_additional : ℕ),
    r_large = 18 →
    r_small = 3 →
    r_additional = 1 →
    n_small = 16 →
    n_additional = 9 →
    (n_additional : ℝ) * (π * r_additional * r_additional) ≤ π * (r_large * r_large - n_small * r_small * r_small) :=
begin
  sorry
end

end remaining_area_fits_9_circles_l103_103484


namespace one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l103_103162

theorem one_div_add_one_div_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) := 
sorry

theorem one_div_add_one_div_not_upper_bounded (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M := 
sorry

theorem one_div_add_one_div_in_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (2 ≤ (1 / a + 1 / b) ∧ ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M) := 
sorry

end one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l103_103162


namespace count_multiples_of_7_between_100_and_350_l103_103083

theorem count_multiples_of_7_between_100_and_350 : 
  (nat.count (λ n, 100 < n ∧ n < 350 ∧ n % 7 = 0) {1, 2, ..., 349}) = 35 :=
by
  sorry

end count_multiples_of_7_between_100_and_350_l103_103083


namespace find_y_l103_103092

theorem find_y (x y : ℤ) 
  (h1 : x^2 + 4 = y - 2) 
  (h2 : x = 6) : 
  y = 42 := 
by 
  sorry

end find_y_l103_103092


namespace find_a9_l103_103497

variable {a : ℕ → ℤ}  -- Define a as a sequence of integers
variable (d : ℤ) (a3 : ℤ) (a4 : ℤ)

-- Define the specific conditions given in the problem
def arithmetic_sequence_condition (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ) : Prop :=
  a 3 + a 4 = 12 ∧ d = 2

-- Define the arithmetic sequence relation
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Statement to prove
theorem find_a9 
  (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ)
  (h1 : arithmetic_sequence_condition a d a3 a4)
  (h2 : arithmetic_sequence a d) :
  a 9 = 17 :=
sorry

end find_a9_l103_103497


namespace farm_cows_l103_103487

theorem farm_cows (c h : ℕ) 
  (legs_eq : 5 * c + 2 * h = 20 + 2 * (c + h)) : 
  c = 6 :=
by 
  sorry

end farm_cows_l103_103487


namespace cannot_assemble_shape4_l103_103175

/-- Definition of a rhombus as a figure that can only be rotated. -/
structure Rhombus where
  color : ℕ  -- representing colors, e.g., 1 for white, 2 for grey
  orientation : ℕ -- representing orientation in degrees, e.g., 0, 90, 180, 270

/-- Definition of a larger figure made up of rhombuses. -/
def LargerFigure : Type :=
  List Rhombus

/-- Condition: Petya can rotate rhombuses but not flip them. -/
def isValidFigure (f : LargerFigure) : Prop :=
  ∀ r1 r2 ∈ f, r1.orientation = r2.orientation ∨ r1.orientation = (r2.orientation + 90) % 360 ∨
               r1.orientation = (r2.orientation + 180) % 360 ∨ r1.orientation = (r2.orientation + 270) % 360

/-- Example of the bottom right shape which requires flipping. -/
def shape4 : LargerFigure := [
  Rhombus.mk 1 0, Rhombus.mk 2 90, -- hypothetical configuration of shape4
  Rhombus.mk 2 180, Rhombus.mk 1 270
]

/-- Proof problem: Petya cannot assemble Shape 4 given the rotation-only restriction. -/
theorem cannot_assemble_shape4 : ¬ isValidFigure shape4 := by
  sorry

end cannot_assemble_shape4_l103_103175


namespace infinite_products_of_s_distinct_primes_l103_103183

-- Definitions for the conditions
def coprime (m n : ℕ) : Prop := ∀ d : ℕ, d ∣ m → d ∣ n → d = 1

def arithmetic_sequence (a b : ℕ) (k : ℕ) : ℕ := a * k + b

-- Problem statement
theorem infinite_products_of_s_distinct_primes 
  (a b : ℕ) (h_coprime : coprime a b) (s : ℕ) (h_pos_s : 0 < s) :
  ∃∞ k : ℕ, ∃ p : list ℕ, (∀ q ∈ p, prime q) ∧ p.length = s ∧ arithmetic_sequence a b k = p.prod :=
sorry

end infinite_products_of_s_distinct_primes_l103_103183


namespace trapezoid_area_l103_103995

theorem trapezoid_area 
  (a b c : ℝ)
  (h_a : a = 5)
  (h_b : b = 15)
  (h_c : c = 13)
  : (1 / 2) * (a + b) * (Real.sqrt (c ^ 2 - ((b - a) / 2) ^ 2)) = 120 := by
  sorry

end trapezoid_area_l103_103995


namespace polygon_exterior_angle_l103_103461

theorem polygon_exterior_angle (exterior_angle : ℝ) (h : exterior_angle = 72) :
  ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = 540 :=
by
  have sides := 360 / exterior_angle
  have n := sides.toNat
  use n
  split
  · calc n = 360 / exterior_angle := by sorry
        ... = 5 := by sorry
  · calc (n - 2) * 180 = 3 * 180 := by sorry
                ... = 540 := by sorry

end polygon_exterior_angle_l103_103461


namespace geometric_sequence_sixth_term_l103_103896

-- Definitions of conditions
def a : ℝ := 512
def r : ℝ := (2 / a)^(1 / 7)

-- The proof statement
theorem geometric_sequence_sixth_term (h : a * r^7 = 2) : 512 * (r^5) = 16 :=
begin
  sorry
end

end geometric_sequence_sixth_term_l103_103896


namespace find_heaviest_lightest_coin_l103_103959

theorem find_heaviest_lightest_coin (coins : Fin 68 → ℝ) : ∃ (heaviest lightest : Fin 68), 
  heaviest ≠ lightest ∧ 
  (∀ c, coins c ≤ coins heaviest) ∧ 
  (∀ c, coins lightest ≤ coins c) ∧ 
  100 weighings.

-- Sorry is used as a placeholder to indicate where the proof would go.
sorry

end find_heaviest_lightest_coin_l103_103959


namespace sequence_expression_l103_103145

theorem sequence_expression (S_n : ℕ → ℚ) (a_n : ℕ → ℚ) 
  (h1 : a_2 = 1/2)
  (h2 : ∀ n, a_(n+1) = S_n S_(n+1)) :
  (∀ n, S_n = -1 / n) ∨ (∀ n, S_n = 1 / (3 - n)) :=
by sorry

end sequence_expression_l103_103145


namespace find_c_l103_103865

-- Define the parameters and hypothesis
def meets_conditions (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem find_c (a b : ℕ) (h_eq: meets_conditions a b (11 + 49 + 1)) : 
  meets_conditions 11 b 61 :=
by
  unfold meets_conditions at *
  exact Eq.trans h_eq sorry

end find_c_l103_103865


namespace prob1_max_value_prob2_min_value_prob3_range_values_l103_103969

-- Problem 1
theorem prob1_max_value (x : ℝ) (h : x < -2) : 
  ∃ y, y = 2*x + 1/(x + 2) ∧ y ≤ -2*Real.sqrt 2 - 4 := sorry

-- Problem 2
theorem prob2_min_value : 
  ∃ y, y = (x^2 + 5)/Real.sqrt (x^2 + 4) ∧ y ≥ 5/2 := sorry

-- Problem 3
theorem prob3_range_values (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a*b = a + b + 3) : 
  a + b ≥ 6 := sorry

end prob1_max_value_prob2_min_value_prob3_range_values_l103_103969


namespace geometric_sequence_sum_l103_103120

theorem geometric_sequence_sum (a : ℕ → ℤ) (m n : ℕ) (h1 : a 1 = 2) 
    (h2 : ∀ k, a (k + 1) = 3 * a k) 
    (h3 : m > n)
    (h4 : ∑ k in finset.range (m - n + 1), a (k + n) = 720) : m + n = 9 :=
sorry

end geometric_sequence_sum_l103_103120


namespace part1_part2_l103_103532

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2| + |x - a|

-- Prove part 1: For all x in ℝ, log(f(x, -8)) ≥ 1
theorem part1 : ∀ x : ℝ, Real.log (f x (-8)) ≥ 1 :=
by 
  sorry

-- Prove part 2: For all x in ℝ, if f(x,a) ≥ a, then a ≤ 1
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ a) → a ≤ 1 :=
by
  sorry

end part1_part2_l103_103532


namespace correct_statements_count_l103_103385

theorem correct_statements_count (x : ℝ) :
  let inverse := (x > 0) → (x^2 > 0)
  let converse := (x^2 ≤ 0) → (x ≤ 0)
  let contrapositive := (x ≤ 0) → (x^2 ≤ 0)
  (∃ p : Prop, p = inverse ∨ p = converse ∧ p) ↔ 
  ¬ contrapositive →
  2 = 2 :=
by
  sorry

end correct_statements_count_l103_103385


namespace inverse_sum_l103_103148

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 3 - x else 2 * x - x^2

theorem inverse_sum :
  let f_inv_2 := (1 + Real.sqrt 3)
  let f_inv_1 := 2
  let f_inv_4 := -1
  f_inv_2 + f_inv_1 + f_inv_4 = 2 + Real.sqrt 3 :=
by
  sorry

end inverse_sum_l103_103148


namespace crayons_given_l103_103563

theorem crayons_given (initial lost left given : ℕ)
  (h1 : initial = 1453)
  (h2 : lost = 558)
  (h3 : left = 332)
  (h4 : given = initial - left - lost) :
  given = 563 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end crayons_given_l103_103563


namespace decimal_to_base7_l103_103693

theorem decimal_to_base7 : ∀ n : ℕ, n = 234 → n.base_repr 7 = "453" :=
by
  intro n hn
  rw [hn]
  sorry

end decimal_to_base7_l103_103693


namespace number_of_two_digit_factors_of_2_pow_18_minus_1_l103_103445

theorem number_of_two_digit_factors_of_2_pow_18_minus_1 : 
  let n := 2^18 - 1 in
  (finset.filter (λ x, x > 9 ∧ x < 100) (finset.filter (λ d, n % d = 0) (finset.range 100))).card = 4 :=
by sorry

end number_of_two_digit_factors_of_2_pow_18_minus_1_l103_103445


namespace statement_B_statement_C_l103_103264

variable (a b c : ℝ)

-- Condition: a > b
def condition1 := a > b

-- Condition: a / c^2 > b / c^2
def condition2 := a / c^2 > b / c^2

-- Statement B: If a > b, then a - 1 > b - 2
theorem statement_B (ha_gt_b : condition1 a b) : a - 1 > b - 2 :=
by sorry

-- Statement C: If a / c^2 > b / c^2, then a > b
theorem statement_C (ha_div_csqr_gt_hb_div_csqr : condition2 a b c) : a > b :=
by sorry

end statement_B_statement_C_l103_103264


namespace inequality_and_equality_condition_l103_103965

theorem inequality_and_equality_condition (y : ℝ) (h : 0 < y) : 2 * y ≥ 3 - 1 / y^2 ∧ (2 * y = 3 - 1 / y^2 ↔ y = 1) :=
by
  split
  { -- Proof of the inequality
    sorry },
  { -- Proof of the equality condition
    sorry }

end inequality_and_equality_condition_l103_103965


namespace probability_queen_of_diamonds_l103_103815

/-- 
A standard deck of 52 cards consists of 13 ranks and 4 suits.
We want to prove that the probability the top card is the Queen of Diamonds is 1/52.
-/
theorem probability_queen_of_diamonds 
  (total_cards : ℕ) 
  (queen_of_diamonds : ℕ)
  (h1 : total_cards = 52)
  (h2 : queen_of_diamonds = 1) : 
  (queen_of_diamonds : ℚ) / (total_cards : ℚ) = 1 / 52 := 
by 
  sorry

end probability_queen_of_diamonds_l103_103815


namespace part_I_part_II_l103_103426

variable (f : ℝ → ℝ)
variable (x m k : ℝ)

-- Definitions used in Lean 4 statement
def k_condition : Prop := ∀ x, f x = abs (k * x - 1)

def f_leq_condition (k : ℝ) : Prop := ∀ x, f x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1

def inequality_condition (x : ℝ) : Prop := f (x + 2) - f (2 * x + 1) ≤ 3 - 2 * m

-- The proof statements
theorem part_I (h : ∀ x, k_condition f k) : k = -2 :=
sorry

theorem part_II (h1: k = 1) (h2: ∀ x, inequality_condition f m) : ∀ m, m ≤ 1 :=
sorry

end part_I_part_II_l103_103426


namespace cong_mod_210_l103_103477

theorem cong_mod_210 (x : ℤ) : 
  (x^5 ≡ x [MOD 210]) ↔ ∃ t : ℤ, x = 7 * t ∨ x = 7 * t + 1 ∨ x = 7 * t - 1 :=
by
  sorry

end cong_mod_210_l103_103477


namespace lcm_consecutive_impossible_l103_103025

noncomputable def n : ℕ := 10^1000

def circle_seq := fin n → ℕ

def lcm {a b : ℕ} : ℕ := Nat.lcm a b

def lcm_seq (a : circle_seq) : fin n → ℕ :=
λ i, lcm (a i) (a (i + 1) % n)

theorem lcm_consecutive_impossible (a : circle_seq) :
  ¬ ∃ (b : fin n → ℕ), (∀ i : fin n, b i = lcm (a i) (a (i + 1) % n)) ∧ (finset.range n).pairwise (λ i j, b i + 1 = b j) :=
sorry

end lcm_consecutive_impossible_l103_103025


namespace inequality_proof_l103_103547

variable (x y : ℝ)
variable (h1 : x ≥ 0)
variable (h2 : y ≥ 0)
variable (h3 : x + y ≤ 1)

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) : 
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := 
by 
  sorry

end inequality_proof_l103_103547


namespace abs_neg_2023_l103_103195

theorem abs_neg_2023 : abs (-2023) = 2023 := 
by
  sorry

end abs_neg_2023_l103_103195


namespace area_of_BKLNM_l103_103820

-- Definitions for the problem setup
variables (A B C D O M N L K : Point)
variable (square_ABCD : square A B C D)
variable (area_square : area (square A B C D) = 1)
variable (OD_length : dist O D = 3)
variable (ray1 : ray O intersects line_CD at M and line_AB at N)
variable (ray2 : ray O intersects line_CD at L and line_BC at K)
variable (ON_length : dist O N = a)
variable (angle_BKL : ∠ B K L = α)

-- Statement to be proved
theorem area_of_BKLNM (a : Real) (α : Real) : 
  let polygon_BKLNM := polygon B K L N M in
  area polygon_BKLNM = 1 - (7 / 8) * sqrt(a^2 - 16) + ((1 + 3 * tan(α))^2 / (2 * tan(α))) :=
begin
  sorry
end

end area_of_BKLNM_l103_103820


namespace sequence_geo_is_geometric_sum_inverse_log_l103_103737

open Real

def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 4 / 3 else (1 / 3) * (sequence_a (n - 1)) + (2 / 3)

-- Part 1: Prove that {a_n - 1} is a geometric progression
def sequence_geo (n : ℕ) : ℝ :=
  (sequence_a n) - 1

theorem sequence_geo_is_geometric :
  ∃ r, ∀ n, sequence_geo (n + 1) = r * (sequence_geo n) :=
sorry

-- Part 2: Calculate sum S_n of the sequence {1 / (b_n b_{n+1})} where b_n = log_(1/3)(a_n - 1)
def b (n : ℕ) : ℝ :=
  Real.logBase (1 / 3) (sequence_geo n)

def sequence_inverse_log (m : ℕ) : ℝ :=
  1 / (b m * b (m + 1))

def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, sequence_inverse_log i

theorem sum_inverse_log :
  ∀ n, S n = n / (n + 1) :=
sorry

end sequence_geo_is_geometric_sum_inverse_log_l103_103737


namespace inequality_xy_l103_103543

theorem inequality_xy {x y : ℝ} (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_xy_l103_103543


namespace find_ending_time_l103_103903

-- Let's define the conditions of the problem
def light_glow_interval : ℕ := 30  -- The light glows every 30 seconds
def num_glows : ℚ := 165.63333333333333  -- Maximum number of glows
def start_time : ℕ := 1 * 3600 + 57 * 60 + 58  -- Starting time converted to seconds past 1:00:00
def end_time : ℕ := start_time + (num_glows * light_glow_interval).toNat  -- Ending time in seconds

-- The correct answer in seconds past 1:00:00
def expected_end_time : ℕ := 4 * 3600 + 5 * 60 + 47  -- 4:05:47 converted to seconds past 1:00:00

-- The Lean statement to prove
theorem find_ending_time :
  end_time = expected_end_time :=
by
  sorry

end find_ending_time_l103_103903


namespace no_faces_painted_two_or_three_faces_painted_l103_103611

-- Define the dimensions of the cuboid
def cuboid_length : ℕ := 3
def cuboid_width : ℕ := 4
def cuboid_height : ℕ := 5

-- Define the number of small cubes
def small_cubes_total : ℕ := 60

-- Define the number of small cubes with no faces painted
def small_cubes_no_faces_painted : ℕ := (cuboid_length - 2) * (cuboid_width - 2) * (cuboid_height - 2)

-- Define the number of small cubes with 2 faces painted
def small_cubes_two_faces_painted : ℕ := (cuboid_length - 2) * cuboid_width +
                                          (cuboid_width - 2) * cuboid_length +
                                          (cuboid_height - 2) * cuboid_width

-- Define the number of small cubes with 3 faces painted
def small_cubes_three_faces_painted : ℕ := 8

-- Define the probabilities
def probability_no_faces_painted : ℚ := small_cubes_no_faces_painted / small_cubes_total
def probability_two_or_three_faces_painted : ℚ := (small_cubes_two_faces_painted + small_cubes_three_faces_painted) / small_cubes_total

-- Theorems to prove
theorem no_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                           small_cubes_total = 60 ∧ small_cubes_no_faces_painted = 6) :
  probability_no_faces_painted = 1 / 10 := by
  sorry

theorem two_or_three_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                                    small_cubes_total = 60 ∧ small_cubes_two_faces_painted = 24 ∧
                                    small_cubes_three_faces_painted = 8) :
  probability_two_or_three_faces_painted = 8 / 15 := by
  sorry

end no_faces_painted_two_or_three_faces_painted_l103_103611


namespace sum_of_maximum_and_minimum_of_u_l103_103023

theorem sum_of_maximum_and_minimum_of_u :
  ∀ (x y z : ℝ),
    0 ≤ x → 0 ≤ y → 0 ≤ z →
    3 * x + 2 * y + z = 5 →
    2 * x + y - 3 * z = 1 →
    3 * x + y - 7 * z = 3 * z - 2 →
    (-5 : ℝ) / 7 + (-1 : ℝ) / 11 = -62 / 77 :=
by
  sorry

end sum_of_maximum_and_minimum_of_u_l103_103023


namespace greatest_possible_npm_value_l103_103956

-- Mathematical problem statement

theorem greatest_possible_npm_value :
  ∃ (N M : ℕ), (M = 2 ∨ M = 4 ∨ M = 6 ∨ M = 8) ∧ 
               (N ≥ 1) ∧ 
               (eq_dig := 10 * M + M) ∧ -- MM as equal even digits
               (npm := 100 * N + 10 * (eq_dig / 10 % 10) + M) ∧ -- NPM form
               (eq_dig * M = npm) ∧
               npm = 396 :=
sorry

end greatest_possible_npm_value_l103_103956


namespace smallest_gcd_of_lcm_eq_square_diff_l103_103248

theorem smallest_gcd_of_lcm_eq_square_diff (x y : ℕ) (h : Nat.lcm x y = (x - y) ^ 2) : Nat.gcd x y = 2 :=
sorry

end smallest_gcd_of_lcm_eq_square_diff_l103_103248


namespace find_dried_fruits_kilograms_l103_103675

noncomputable def dried_fruits_kilograms (x : ℝ) : Prop :=
  let nuts_kg := 3 in
  let nuts_cost_per_kg := 12 in
  let fruits_cost_per_kg := 8 in
  let total_cost := 56 in
  (nuts_kg * nuts_cost_per_kg + x * fruits_cost_per_kg = total_cost)

theorem find_dried_fruits_kilograms : ∃ x : ℝ, dried_fruits_kilograms x ∧ x = 2.5 :=
begin
  sorry
end

end find_dried_fruits_kilograms_l103_103675


namespace velocity_at_t4_acceleration_is_constant_l103_103662

noncomputable def s (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 8

def v (t : ℝ) : ℝ := 6 * t - 3

def a : ℝ := 6

theorem velocity_at_t4 : v 4 = 21 := by 
  sorry

theorem acceleration_is_constant : a = 6 := by 
  sorry

end velocity_at_t4_acceleration_is_constant_l103_103662


namespace profit_per_meter_is_25_l103_103993

def sell_price : ℕ := 8925
def cost_price_per_meter : ℕ := 80
def meters_sold : ℕ := 85
def total_cost_price : ℕ := cost_price_per_meter * meters_sold
def total_profit : ℕ := sell_price - total_cost_price
def profit_per_meter : ℕ := total_profit / meters_sold

theorem profit_per_meter_is_25 : profit_per_meter = 25 := by
  sorry

end profit_per_meter_is_25_l103_103993


namespace find_f_cos_10_l103_103021

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (x : ℝ) : f (Real.sin x) = Real.cos (3 * x)

theorem find_f_cos_10 : f (Real.cos (10 * Real.pi / 180)) = -1/2 := by
  sorry

end find_f_cos_10_l103_103021


namespace find_a_l103_103425

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a x

theorem find_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : f a 9 = 2) : a = 3 :=
by
  sorry

end find_a_l103_103425


namespace lcm_consecutive_impossible_l103_103033

def lcm (a b : Nat) : Nat := 
  (a * b) / (Nat.gcd a b)

theorem lcm_consecutive_impossible (n : Nat) (a : Fin n → Nat) 
  (h : n = 10^1000) :
  ¬∃ (b : Fin n → Nat), (∀ i : Fin n, b i = lcm (a i) (a (i + 1))) ∧ 
  (∀ i : Fin (n - 1), b i + 1 = b (i + 1)) :=
by
  sorry

end lcm_consecutive_impossible_l103_103033


namespace triangle_FYH_area_l103_103616

theorem triangle_FYH_area (EF GH : ℝ) (a_trapezoid : EF = 24 ∧ GH = 36) (area_trapezoid : (EF + GH) * h / 2 = 360)
    (intersect_point : diag_intersect E F G H = Y) :
  ∃ h Y : ℝ, triangle_area F Y H = 57.6 :=
by
  sorry

end triangle_FYH_area_l103_103616


namespace conical_surface_radius_l103_103459

theorem conical_surface_radius (r : ℝ) :
  (2 * Real.pi * r = 5 * Real.pi) → r = 2.5 :=
by
  sorry

end conical_surface_radius_l103_103459


namespace compare_y1_y2_l103_103767

theorem compare_y1_y2 
  (y : ℝ → ℝ) (H : ∀ x ≠ 0, y x = -8 / x) 
  (A : y (-2) = 4) 
  (B : ∃ y1, y 1 = y1) 
  (C : ∃ y2, y 3 = y2) :
  ∃ y1 y2, y1 < y2 := 
by
  sorry

end compare_y1_y2_l103_103767


namespace minimum_total_distance_l103_103204

-- Conditions:
def point (α : Type) := (α × α)
def distance (p1 p2 : point ℝ) : ℝ := 
  float.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def vertex := point ℝ

variables (A B C : vertex)
axiom AB_eq_3 : distance A B = 3
axiom AC_eq_2 : distance A C = 2
axiom BC_eq_sqrt_7 : distance B C = float.sqrt 7
axiom warehouse_pos : vertex -- Assuming existence but not fixing the actual position

-- Question == Answer
theorem minimum_total_distance (warehouse : vertex) 
  (dA := distance warehouse A)
  (dB := distance warehouse B)
  (dC := distance warehouse C) :
  let total_distance := dA + dB + dC 
  in total_distance * 2 = 2 * float.sqrt 19 :=
by sorry

end minimum_total_distance_l103_103204


namespace probability_of_stack_height_l103_103370

-- Definitions based on conditions
def num_crates : Nat := 5
def possible_heights : List ℕ := [2, 3, 5]

-- Predicate for valid height summation
def valid_height_comb (a b c : ℕ) : Prop :=
  2 * a + 3 * b + 5 * c = 16 ∧ a + b + c = num_crates

-- Count the number of valid (a, b, c) triplets
def valid_comb_count : ℕ :=
  ([ (0, 2, 3), (2, 3, 0) ] -- The valid solutions identified manually
  |> List.map (λ (abc : ℕ × ℕ × ℕ), Nat.factorial num_crates / (Nat.factorial abc.1 * Nat.factorial abc.2 * Nat.factorial abc.3))
  |> List.sum)

-- Total number of possibilities
def total_possibilities : ℕ := 3 ^ num_crates

-- Probability of the stack height being exactly 16 ft
noncomputable def prob_height_16 : ℚ :=
  valid_comb_count / total_possibilities

-- Statement of the proof problem
theorem probability_of_stack_height :
  prob_height_16 = 20 / 243 := by
  sorry

end probability_of_stack_height_l103_103370


namespace no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l103_103272
-- Bringing in the entirety of Mathlib

-- Problem (a): There are no non-zero integers that increase by 7 or 9 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_7_or_9 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ (10 * X + d = 7 * n ∨ 10 * X + d = 9 * n)) :=
by sorry

-- Problem (b): There are no non-zero integers that increase by 4 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_4 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ 10 * X + d = 4 * n) :=
by sorry

end no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l103_103272


namespace theta_in_fourth_quadrant_l103_103094

-- Define the angle θ
variable (θ : Real.Angle)

-- Conditions
def sin_theta_neg : Prop := Real.sin θ < 0
def tan_theta_neg : Prop := Real.tan θ < 0

-- Theorem statement
theorem theta_in_fourth_quadrant (h1 : sin_theta_neg θ) (h2 : tan_theta_neg θ) : 
    θ ∈ { θ | θ.toReal.rad.toInterval ⟶ list2  (-π/2,0]} :=
    sorry

end theta_in_fourth_quadrant_l103_103094


namespace initial_water_amount_gallons_l103_103267

theorem initial_water_amount_gallons 
  (cup_capacity_oz : ℕ)
  (rows : ℕ)
  (chairs_per_row : ℕ)
  (water_left_oz : ℕ)
  (oz_per_gallon : ℕ)
  (total_gallons : ℕ)
  (h1 : cup_capacity_oz = 6)
  (h2 : rows = 5)
  (h3 : chairs_per_row = 10)
  (h4 : water_left_oz = 84)
  (h5 : oz_per_gallon = 128)
  (h6 : total_gallons = (rows * chairs_per_row * cup_capacity_oz + water_left_oz) / oz_per_gallon) :
  total_gallons = 3 := 
by sorry

end initial_water_amount_gallons_l103_103267


namespace probability_of_winning_pair_l103_103689

/-
The problem involves a total of 10 cards, consisting of 6 blue and 4 purple cards labeled
with letters A-F for blue and A-D for purple. A winning pair is defined as either two cards
of the same color or two with consecutive letters. We need to prove that the probability of
drawing a winning pair is 29/45.
-/

noncomputable def total_cards : ℕ := 10
noncomputable def blue_cards : ℕ := 6
noncomputable def purple_cards : ℕ := 4
def total_ways_to_draw_two_cards : ℕ := (total_cards * (total_cards - 1)) / 2

def ways_to_draw_same_letter : ℕ := 4  -- Four letters A, B, C, D each have blue and purple
def ways_to_draw_same_color : ℕ := ((blue_cards * (blue_cards - 1)) / 2) + ((purple_cards * (purple_cards - 1)) / 2)
def ways_to_draw_consecutive_letters : ℕ := 5 + 3 -- Consecutive pairs in blue and purple

def overlapping_same_letter_and_color : ℕ := 4

def total_favorable_outcomes : ℕ :=
  ways_to_draw_same_letter + ways_to_draw_same_color + ways_to_draw_consecutive_letters - overlapping_same_letter_and_color

theorem probability_of_winning_pair : (total_favorable_outcomes : ℚ) / (total_ways_to_draw_two_cards : ℚ) = 29 / 45 :=
by
  sorry

end probability_of_winning_pair_l103_103689


namespace point_three_units_away_from_A_is_negative_seven_or_negative_one_l103_103867

-- Defining the point A on the number line
def A : ℤ := -4

-- Definition of the condition where a point is 3 units away from A
def three_units_away (x : ℤ) : Prop := (x = A - 3) ∨ (x = A + 3)

-- The statement to be proved
theorem point_three_units_away_from_A_is_negative_seven_or_negative_one (x : ℤ) :
  three_units_away x → (x = -7 ∨ x = -1) :=
sorry

end point_three_units_away_from_A_is_negative_seven_or_negative_one_l103_103867


namespace range_of_k_l103_103054

theorem range_of_k (k : ℝ) (x y : ℝ) : 
  (y = 2 * x - 5 * k + 7) → 
  (y = - (1 / 2) * x + 2) → 
  (x > 0) → 
  (y > 0) → 
  (1 < k ∧ k < 3) :=
by
  sorry

end range_of_k_l103_103054


namespace minimum_triangle_area_l103_103644

theorem minimum_triangle_area :
  ∃ (p q : ℤ), let C := (p, q) in
  let A := (0, 0) in
  let B := (30, 18) in
  let area := abs (30 * q - 18 * p) / 2 in
  area = 3 := 
by
  sorry

end minimum_triangle_area_l103_103644


namespace range_of_t_l103_103398

noncomputable def circle (x y : ℝ) : Prop := (x - √3)^2 + (y - 1)^2 = 1

theorem range_of_t (t : ℝ) (ht : t > 0) :
  (∃ (θ : ℝ), circle (√3 + Real.cos θ) (1 + Real.sin θ) 
              ∧ ∠((√3 + Real.cos θ, 1 + Real.sin θ), (-t, 0), (t, 0)) = π / 2) →
  1 ≤ t ∧ t ≤ 3 := sorry

end range_of_t_l103_103398


namespace diego_can_carry_home_l103_103701

theorem diego_can_carry_home (T W G O A : ℕ) (hT : T = 20) (hW : W = 1) (hG : G = 1) (hO : O = 1) : A = T - (W + G + O) → A = 17 := by
  sorry

end diego_can_carry_home_l103_103701


namespace period_pi_omega_l103_103427

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  3 * (Real.sin (ω * x)) * (Real.cos (ω * x)) - 4 * (Real.cos (ω * x))^2

theorem period_pi_omega (ω : ℝ) (hω : ω > 0) (period_condition : ∀ x, f x ω = f (x + π) ω)
  (theta : ℝ) (h_f_theta : f theta ω = 1 / 2) :
  f (theta + π / 2) ω + f (theta - π / 4) ω = -13 / 2 :=
by
  sorry

end period_pi_omega_l103_103427


namespace tan_2theta_condition_cos_phi_condition_l103_103773

variable {θ φ : ℝ}

-- Define conditions for both questions
def vectors_orthogonal_to_each_other : Prop :=
  θ ∈ (0, π / 2) ∧ (sin θ - 2 * cos θ = 0)

def sin_theta_minus_phi := sin (θ - φ) = √10 / 10
def theta_phi_constraints := θ ∈ (0, π / 2) ∧ 0 < φ < π / 2

-- Proof statements for the given questions

-- Statement for Question 1
theorem tan_2theta_condition (h1 : vectors_orthogonal_to_each_other) : 
  tan (2 * θ) = -4 / 3 :=
sorry

-- Statement for Question 2
theorem cos_phi_condition (h1 : sin_theta_minus_phi) 
  (h2 : theta_phi_constraints) :
  cos φ = √2 / 2 :=
sorry

end tan_2theta_condition_cos_phi_condition_l103_103773


namespace completion_days_for_B_l103_103975

-- Conditions
def A_completion_days := 20
def B_completion_days (x : ℕ) := x
def project_completion_days := 20
def A_work_days := project_completion_days - 10
def B_work_days := project_completion_days
def A_work_rate := 1 / A_completion_days
def B_work_rate (x : ℕ) := 1 / B_completion_days x
def combined_work_rate (x : ℕ) := A_work_rate + B_work_rate x
def A_project_completed := A_work_days * A_work_rate
def B_project_remaining (x : ℕ) := 1 - A_project_completed
def B_project_completion (x : ℕ) := B_work_days * B_work_rate x

-- Proof statement
theorem completion_days_for_B (x : ℕ) 
  (h : B_project_completion x = B_project_remaining x ∧ combined_work_rate x > 0) :
  x = 40 :=
sorry

end completion_days_for_B_l103_103975


namespace point_in_fourth_quadrant_l103_103389

def complex_quadrant : Type :=
  | First
  | Second
  | Third
  | Fourth

open Complex

-- Define the complex number z
noncomputable def z : ℂ := 2 - I

-- Condition given in the problem
axiom condition (z' : ℂ) : (z - 1) * I = I + 1

-- Proving the point corresponds to z lies in the fourth quadrant
theorem point_in_fourth_quadrant (z : ℂ) (h : (z - 1) * I = I + 1) : (2, -1) = (z.re, z.im) → (z.re > 0 ∧ z.im < 0) := by
  sorry

#eval point_in_fourth_quadrant z condition

end point_in_fourth_quadrant_l103_103389


namespace range_of_m_l103_103071

theorem range_of_m (m : ℝ) : (¬ ∃ x : ℝ, 4 ^ x + 2 ^ (x + 1) + m = 0) → m ≥ 0 := 
by
  sorry

end range_of_m_l103_103071


namespace complex_norm_identity_l103_103853

theorem complex_norm_identity (a b : ℂ) : 
  |a + b|^2 - |a - b|^2 = 4 * (Re (a * conj b)) := 
sorry

end complex_norm_identity_l103_103853


namespace cube_coloring_schemes_l103_103174

theorem cube_coloring_schemes (A B C D A₁ B₁ C₁ D₁ : Type) (colors : set (fin 5))
  (colors_at_A : fin 3) 
  (adjacent_faces_diff_colors : ∀ f1 f2 : {f // f ∈ {A, B, C, D, A₁, B₁, C₁, D₁}}, f1 ≠ f2 → colors_at_A f1 ≠ colors_at_A f2) :
  ∃ (color_schemes : fin 3), color_schemes = 13 :=
by
  -- Assume conditions and show the number of valid color schemes is 13
  sorry

end cube_coloring_schemes_l103_103174


namespace shopkeeper_profit_percent_l103_103990

theorem shopkeeper_profit_percent (cost_price profit : ℝ) (h1 : cost_price = 960) (h2 : profit = 40) : 
  (profit / cost_price) * 100 = 4.17 :=
by
  sorry

end shopkeeper_profit_percent_l103_103990


namespace greatest_num_of_coins_l103_103332

-- Define the total amount of money Carlos has in U.S. coins.
def total_value : ℝ := 5.45

-- Define the value of each type of coin.
def quarter_value : ℝ := 0.25
def dime_value : ℝ := 0.10
def nickel_value : ℝ := 0.05

-- Define the number of quarters, dimes, and nickels Carlos has.
def num_coins (q : ℕ) := quarter_value * q + dime_value * q + nickel_value * q

-- The main theorem: Carlos can have at most 13 quarters, dimes, and nickels.
theorem greatest_num_of_coins (q : ℕ) :
  num_coins q = total_value → q ≤ 13 :=
sorry

end greatest_num_of_coins_l103_103332


namespace largest_subset_M_l103_103724

theorem largest_subset_M (n : ℕ) (hn : n > 0) : 
  (if even n then ∀ x : ℝ, x ≥ -1 → n + ∑ i in finset.range n, x ^ (n + 1) ≥ n * finset.prod (finset.range n) (λ i, x) + ∑ i in finset.range n, x
   else ∀ x : ℝ, n + ∑ i in finset.range n, x ^ (n + 1) ≥ n * finset.prod (finset.range n) (λ i, x) + ∑ i in finset.range n, x) :=
begin
  sorry
end

end largest_subset_M_l103_103724


namespace problem_l103_103786

variables (y S : ℝ)

theorem problem (h : 5 * (2 * y + 3 * Real.sqrt 3) = S) : 10 * (4 * y + 6 * Real.sqrt 3) = 4 * S :=
sorry

end problem_l103_103786


namespace inequality_xy_l103_103545

theorem inequality_xy {x y : ℝ} (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_xy_l103_103545


namespace area_of_rectangle_ABCD_l103_103623

-- Define the coordinates of the points
def B : Point := (-1, 2)
def C : Point := (4, 2)
def D : Point := (4, 5)

-- Define the function to calculate the length of a segment given two points
def segment_length (p1 p2 : Point) : ℝ :=
  match p1, p2 with
  | (x1, y1), (x2, y2) => Math.abs (x2 - x1) + Math.abs (y2 - y1)

-- Define the lengths of the sides of the rectangle
def length_BC : ℝ := segment_length B C
def length_DC : ℝ := segment_length D C

-- Define the calculation of the area of the rectangle
def area_of_rectangle (length width : ℝ) : ℝ :=
  length * width

-- The theorem stating the area of the rectangle
theorem area_of_rectangle_ABCD : 
  area_of_rectangle length_BC length_DC = 15 :=
sorry

end area_of_rectangle_ABCD_l103_103623


namespace angle_KPL_either_45_or_135_l103_103173

noncomputable def square_ABCD : Type :=
sorry -- Definition of square ABCD (given and not in standard library)

variables (A B C D K M L P : square_ABCD)
variables (hK : K ∈ side AB) (hM : M ∈ side CD) (hL : L ∈ diag AC)
variables (hML_eq_KL : distance M L = distance K L)
variables (hP : P = intersection_pt segment MK segment BD)

theorem angle_KPL_either_45_or_135 :
  ∃ θ : ℝ, θ = 45 ∨ θ = 135 ∧ θ = angle K P L :=
sorry

end angle_KPL_either_45_or_135_l103_103173


namespace double_entry_proof_l103_103695

-- Define the single component operation
def single_entry (a b c : ℝ) (hc : c ≠ 0) : ℝ := (a + b) / c

-- Define the double entry operation
def double_entry (u v : ℝ × ℝ × ℝ) (hc1 : u.2.2 ≠ 0) (hc2 : v.2.2 ≠ 0) : ℝ :=
  (single_entry (u.1.1 + u.1.2) (v.1.1 + v.1.2) (u.2.2 + v.2.2) hc1 + 
   single_entry (u.2.1 + u.2.2) (v.2.1 + v.2.2) (u.2.2 + v.2.2) hc2) / 2

-- Our main theorem statement
theorem double_entry_proof : double_entry 
  ((single_entry 10 20 30 (by decide), single_entry 40 30 70 (by decide), 30 + 70),
   (single_entry 8 4 12 (by decide), single_entry 18 9 27 (by decide), 12 + 27))
   (by decide) 
   (by decide)
   = 0.142564 := 
sorry

end double_entry_proof_l103_103695


namespace am_gm_inequality_l103_103209

theorem am_gm_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (x + y + z) / 3 ≥ real.cbrt (x * y * z) ∧ ((x + y + z) / 3 = real.cbrt (x * y * z) ↔ x = y ∧ y = z) :=
by
  sorry

end am_gm_inequality_l103_103209


namespace exp_gen_func_eq_exp_l103_103363

noncomputable def exp_generating_function (r : ℝ) (t : ℝ) : ℝ :=
  ∑' n : ℕ, (r ^ n * t ^ n) / n.fact

theorem exp_gen_func_eq_exp (r : ℝ) (h : 0 < r) (t : ℝ) : 
  exp_generating_function r t = Real.exp (r * t) :=
by
  sorry

end exp_gen_func_eq_exp_l103_103363


namespace avg_tech_is_800_l103_103198

-- Define the conditions
variables (avg_all : ℝ) (t: ℝ) (avg_non_tech : ℝ) (total_workers : ℕ)
variables (num_tech : ℕ)

-- Assume the conditions given in the problem
axiom avg_all_def : avg_all = 700
axiom total_workers_def : total_workers = 15
axiom num_tech_def : num_tech = 5
axiom avg_non_tech_def : avg_non_tech = 650

-- Calculate the number of non-technicians
def num_non_tech := total_workers - num_tech

-- Calculate the total salary for all workers
def total_salary := avg_all * total_workers

-- Calculate the total salary for non-technicians
def total_salary_non_tech := avg_non_tech * num_non_tech

-- Calculate the total salary for technicians
def total_salary_tech := total_salary - total_salary_non_tech

-- Calculate the average salary per head for technicians
def avg_tech := total_salary_tech / num_tech

-- Prove that the average salary per head for technicians is Rs. 800
theorem avg_tech_is_800 : avg_tech = 800 :=
by
  rw [avg_tech, total_salary_tech, total_salary, total_salary_non_tech, avg_all_def, total_workers_def, avg_non_tech_def, num_non_tech, num_tech_def],
  norm_num,
  sorry

end avg_tech_is_800_l103_103198


namespace log_equation_solution_l103_103746

theorem log_equation_solution (y : ℝ) (hy : y < 1)
  (h : (log 10 y)^2 - 2 * log 10 (y^2) = 75) :
  (log 10 y)^4 - log 10 (y^4) = (2 - real.sqrt 79)^4 - 4 * (2 - real.sqrt 79) :=
by sorry


end log_equation_solution_l103_103746


namespace no_distinct_primes_p_q_and_positive_integer_n_l103_103180

open Nat

theorem no_distinct_primes_p_q_and_positive_integer_n
    (p q n : ℕ) 
    (hp : Prime p) 
    (hq : Prime q) 
    (h_diff : p ≠ q) 
    (hn_pos : n > 0) : 
  ¬ (p^(q-1) - q^(p-1) = 4 * n^2) :=
by sorry

end no_distinct_primes_p_q_and_positive_integer_n_l103_103180


namespace arithmetic_progression_sum_n_values_l103_103219

theorem arithmetic_progression_sum_n_values :
  let sum_n (a n : ℕ) := n * (2 * a + (n - 1) * 2) / 2
  ∃ a : ℕ, sum_n a n = 186 ∧ ∃ n values, n > 1 ∧ values = {n | ∃ a : ℕ, sum_n a n = 186 ∧ n > 1}.card = 6 := by 
  sorry

end arithmetic_progression_sum_n_values_l103_103219


namespace find_quotient_l103_103810

-- Constants representing the given conditions
def dividend : ℕ := 690
def divisor : ℕ := 36
def remainder : ℕ := 6

-- Theorem statement
theorem find_quotient : ∃ (quotient : ℕ), dividend = (divisor * quotient) + remainder ∧ quotient = 19 := 
by
  sorry

end find_quotient_l103_103810


namespace range_d_l103_103122

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (5, 5)
def trajectory_dist (P : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) := 
  {Q | (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = r^2}

theorem range_d 
  (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (r₁ : ℝ) (r₂ : ℝ) 
  (h₁ : r₁ = 1) 
  (h₂ : (trajectory_dist A r₁) = trajectory_dist A 1) 
  (h₃ : ∀ d: ℝ, (0 < d ∧ d < 4) ↔ (∀ Q ∈ trajectory_dist B d, Q ∈ trajectory_dist A 1)):
  ∀ d: ℝ, 0 < d ∧ d < 4 :=
by 
  sorry

end range_d_l103_103122


namespace triangle_properties_l103_103768

theorem triangle_properties
  (K : ℝ) (α β : ℝ)
  (hK : K = 62.4)
  (hα : α = 70 + 20/60 + 40/3600)
  (hβ : β = 36 + 50/60 + 30/3600) :
  ∃ (a b T : ℝ), 
    a = 16.55 ∧
    b = 30.0 ∧
    T = 260.36 :=
by
  sorry

end triangle_properties_l103_103768


namespace units_digit_factorial_sum_l103_103626

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_factorial_sum : 
  units_digit (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9! + 10!) = 3 := by
  sorry

end units_digit_factorial_sum_l103_103626


namespace sum_f_eq_24136_l103_103382

noncomputable def f (x : ℝ) := x^2 - 2017 * x + 8052 + abs (x^2 - 2017 * x + 8052)

theorem sum_f_eq_24136 : 
  ∑ i in finset.range 2013, f (i + 1) = 24136 :=
by
  sorry

end sum_f_eq_24136_l103_103382


namespace max_angle_in_hexagon_l103_103904

-- Definition of the problem
theorem max_angle_in_hexagon :
  ∃ (a d : ℕ), a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 ∧ 
               a + 5 * d < 180 ∧ 
               (∀ a d : ℕ, a + a + d + a + 2*d + a + 3*d + a + 4*d + a + 5*d = 720 → 
               a + 5*d < 180 → m <= 175) :=
sorry

end max_angle_in_hexagon_l103_103904


namespace domain_of_function_l103_103345

theorem domain_of_function (x : ℝ) : (|x - 2| + |x + 2| ≠ 0) := 
sorry

end domain_of_function_l103_103345


namespace original_quantity_l103_103988

theorem original_quantity (x : ℕ) : 
  (532 * x - 325 * x = 1065430) -> x = 5148 := 
by
  intro h
  sorry

end original_quantity_l103_103988


namespace C2_eqn_proof_AB_distance_proof_l103_103115

section
variables (α θ : ℝ) (x y : ℝ) 

def C1_param_eqn (α : ℝ) := (x = 2 + 2 * Real.cos α ∧ y = 2 * Real.sin α) ∧ (0 < α ∧ α < Real.pi)

def C2_standard_eqn := (x - 1)^2 + y^2 = 1 ∧ 0 < y ∧ y ≤ 1

noncomputable def C1_polar_eqn (θ : ℝ) := ⟨2 * (1 + Real.cos(θ)), 2 * Real.sin(θ)⟩

noncomputable def C2_polar_eqn (θ : ℝ) := ⟨1 + Real.cos(θ), Real.sin(θ)⟩

theorem C2_eqn_proof :
  (∀ (α : ℝ), C1_param_eqn α → ∃ (x y : ℝ), C2_standard_eqn) := sorry

theorem AB_distance_proof :
  (θ = Real.pi/3) → (dist (C1_polar_eqn θ) (C2_polar_eqn θ) = 1) := sorry

end

end C2_eqn_proof_AB_distance_proof_l103_103115


namespace total_agreed_students_is_864_l103_103489

   def pct3 : ℝ := 0.60
   def pct4 : ℝ := 0.45
   def pct5 : ℝ := 0.35
   def pct6 : ℝ := 0.55
   def students3 : ℝ := 256
   def students4 : ℝ := 525
   def students5 : ℝ := 410
   def students6 : ℝ := 600

   noncomputable def total_agreed_students : ℝ :=
     Real.floor (pct3 * students3 + 0.5) +
     Real.floor (pct4 * students4 + 0.5) +
     Real.floor (pct5 * students5 + 0.5) +
     pct6 * students6

   theorem total_agreed_students_is_864 :
     total_agreed_students = 864 := by
     sorry
   
end total_agreed_students_is_864_l103_103489


namespace slope_angle_tangent_line_l103_103601

open Real

-- Define the function 
def f (x : ℝ) := exp x * cos x

-- The proof statement
theorem slope_angle_tangent_line : 
  let α := arctan 1 in
  0 ≤ α ∧ α < π ∧ tan α = 1 ∧ α = π/4 :=
by
  let α := arctan 1
  have hα : α = π / 4, from arctan_eq_pi_div_4,
  have h_tan : tan α = 1, from tan_arctan one_ne_zero,
  have h_bounds : 0 ≤ α ∧ α < π, from ⟨arctan_nonneg 1 (by norm_num), arctan_lt_pi 1⟩,
  exact ⟨h_bounds.left, h_bounds.right, h_tan, hα⟩

end slope_angle_tangent_line_l103_103601


namespace distance_from_A_to_line_l103_103069

/-- Define the polar equation of the line. -/
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = Real.sqrt 2 / 2

/-- Define the point A in polar coordinates. -/
def point_A_polar : ℝ × ℝ := (4, 7 * Real.pi / 4)

/-- Define the distance formula from a point (x₀, y₀) to a line ax + by + c = 0 in Cartesian coordinates. -/
def distance_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a * a + b * b)

/-- Convert a point from polar to Cartesian coordinates. -/
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

/-- The Cartesian form of the point A based on its polar coordinates. -/
def point_A_cartesian : ℝ × ℝ := polar_to_cartesian 4 (7 * Real.pi / 4)

/-- The line equation in Cartesian coordinates converted from the polar line equation ρ * sin(θ + π/4) = sqrt(2)/2 -/
def line_cartesian (x y : ℝ) : Prop := x + y - 1 = 0

/-- Prove that the distance from the point A to the line is sqrt(2)/2. -/
theorem distance_from_A_to_line :
  let a := 1
  let b := 1
  let c := -1
  let (x₀, y₀) := point_A_cartesian
  distance_to_line x₀ y₀ a b c = Real.sqrt 2 / 2 :=
by
  let a := 1
  let b := 1
  let c := -1
  let (x₀, y₀) := polar_to_cartesian 4 (7 * Real.pi / 4)
  show distance_to_line x₀ y₀ a b c = Real.sqrt 2 / 2
  sorry

end distance_from_A_to_line_l103_103069


namespace measure_of_one_interior_angle_of_regular_pentagon_l103_103245

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

def is_regular_polygon (sides : ℕ) (angles : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ sides → 1 ≤ j ∧ j ≤ sides → angles = angles

theorem measure_of_one_interior_angle_of_regular_pentagon :
  ∀ (n : ℕ), n = 5 →
  let S := sum_of_interior_angles n in
  let each_angle := S / n in 
  each_angle = 108 :=
by
  intros n hn
  subst hn
  let S := sum_of_interior_angles n
  let each_angle := S / n
  have h1 : S = 540 := by sorry
  have h2 : each_angle = 108 := by sorry
  exact h2

end measure_of_one_interior_angle_of_regular_pentagon_l103_103245


namespace area_of_triangle_bounded_by_lines_l103_103238

def intersection_point (m₁ m₂ b : ℝ) : ℝ × ℝ :=
  (b / (m₁ - m₂), m₁ * (b / (m₁ - m₂)))

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def area_of_triangle (base height : ℝ) : ℝ :=
  1 / 2 * base * height

theorem area_of_triangle_bounded_by_lines :
  let A := intersection_point 1 0 8 in -- Intersection of y = x and y = 8
  let B := intersection_point (-2) 0 8 in -- Intersection of y = -2x and y = 8
  let O : ℝ × ℝ := (0, 0) in
  distance A B = 12 →
  ∀ (altitude : ℝ), altitude = 8 →
  area_of_triangle 12 8 = 48 :=
by
  intros A B O h₁ h₂
  have base := distance A B
  have height := 8
  have area := area_of_triangle base height
  rw [h₁, h₂] at area
  exact rfl

end area_of_triangle_bounded_by_lines_l103_103238


namespace ce_squared_plus_de_squared_eq_200_l103_103520

theorem ce_squared_plus_de_squared_eq_200 :
  ∀ (O A B C D E : ℝ) (r : ℝ), 
  let AB := 2 * r in
  let BE := 2 * sqrt 10 in
  (A = -r) → (B = r) → (r = 10) → 
  (∠ACD = 30) →
  (CE^2 + DE^2 = 200) :=
by
  intro O A B C D E r
  let AB := 2 * r
  let BE := 2 * sqrt 10
  assume hAeq hBeq hreq hangle
  sorry

end ce_squared_plus_de_squared_eq_200_l103_103520


namespace toy_poodle_height_l103_103914

-- Define the heights of the poodles
variables (S M T : ℝ)

-- Conditions
def std_taller_min : Prop := S = M + 8
def min_taller_toy : Prop := M = T + 6
def std_height : Prop := S = 28

-- Goal: How tall is the toy poodle?
theorem toy_poodle_height (h1 : std_taller_min S M)
                          (h2 : min_taller_toy M T)
                          (h3 : std_height S) : T = 14 :=
by 
  sorry

end toy_poodle_height_l103_103914


namespace total_pie_eaten_l103_103614

theorem total_pie_eaten (s1 s2 s3 : ℚ) (h1 : s1 = 8/9) (h2 : s2 = 5/6) (h3 : s3 = 2/3) :
  s1 + s2 + s3 = 43/18 := by
  sorry

end total_pie_eaten_l103_103614


namespace analytic_expression_of_f_max_min_of_f_on_interval_l103_103758

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem analytic_expression_of_f :
  ∀ A ω φ : ℝ, (∀ x, f x = A * Real.sin (ω * x + φ)) →
  A = 2 ∧ ω = 2 ∧ φ = Real.pi / 6 :=
by
  sorry -- Placeholder for the actual proof

theorem max_min_of_f_on_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≤ Real.sqrt 3) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 12 → f x ≥ 1) :=
by
  sorry -- Placeholder for the actual proof

end analytic_expression_of_f_max_min_of_f_on_interval_l103_103758


namespace diego_can_carry_home_l103_103700

theorem diego_can_carry_home (T W G O A : ℕ) (hT : T = 20) (hW : W = 1) (hG : G = 1) (hO : O = 1) : A = T - (W + G + O) → A = 17 := by
  sorry

end diego_can_carry_home_l103_103700


namespace least_non_lucky_multiple_of_8_l103_103656

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (· + ·) 0

def lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def multiple_of_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem least_non_lucky_multiple_of_8 : ∃ n > 0, multiple_of_8 n ∧ ¬ lucky n ∧ n = 16 :=
by
  -- Proof goes here.
  sorry

end least_non_lucky_multiple_of_8_l103_103656


namespace unique_line_through_point_parallel_to_line_l103_103390

variables (ℝ : Type) [linear_ordered_field ℝ] [topological_space ℝ]
variables (P : point ℝ) (α : plane ℝ) (l : line ℝ)

-- Given conditions
def line_parallel_to_plane (l : line ℝ) (α : plane ℝ) : Prop := is_parallel l α
def point_on_plane (P : point ℝ) (α : plane ℝ) : Prop := α.contains P

-- Problem statement
theorem unique_line_through_point_parallel_to_line (h1 : line_parallel_to_plane l α) (h2 : point_on_plane P α) :
  ∃! l', (α.contains l') ∧ (is_parallel l' l) ∧ (l'.contains P) :=
sorry

end unique_line_through_point_parallel_to_line_l103_103390


namespace omega_in_abc_l103_103881

variables {R : Type*}
variables [LinearOrderedField R]
variables {a b c ω x y z : R} 

theorem omega_in_abc 
  (distinct_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ω ≠ a ∧ ω ≠ b ∧ ω ≠ c)
  (h1 : x + y + z = 1)
  (h2 : a^2 * x + b^2 * y + c^2 * z = ω^2)
  (h3 : a^3 * x + b^3 * y + c^3 * z = ω^3)
  (h4 : a^4 * x + b^4 * y + c^4 * z = ω^4):
  ω = a ∨ ω = b ∨ ω = c :=
sorry

end omega_in_abc_l103_103881


namespace solve_quadratic_l103_103190

-- Problem Definition
def quadratic_equation (x : ℝ) : Prop :=
  2 * x^2 - 6 * x + 3 = 0

-- Solution Definition
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2

-- Lean Theorem Statement
theorem solve_quadratic : ∀ x : ℝ, quadratic_equation x ↔ solution1 x :=
sorry

end solve_quadratic_l103_103190


namespace initial_blocks_l103_103313

variable (x : ℕ)

theorem initial_blocks (h : x + 30 = 65) : x = 35 := by
  sorry

end initial_blocks_l103_103313


namespace number_of_girls_l103_103369

theorem number_of_girls
    (van1 van2 van3 van4 van5 boys : ℕ)
    (h_vans : van1 = 24 ∧ van2 = 30 ∧ van3 = 20 ∧ van4 = 36 ∧ van5 = 29)
    (h_boys : boys = 64)
    : (van1 + van2 + van3 + van4 + van5 - boys = 75) := 
by
  -- Definitions from conditions
  let total_students := van1 + van2 + van3 + van4 + van5
  
  -- Substitution of values based on conditions (summarization skipped for brevity)
  have h1 : total_students = 24 + 30 + 20 + 36 + 29 := by
    rw [h_vans.1, h_vans.2.left, h_vans.2.right.left, h_vans.2.right.right.left, h_vans.2.right.right.right]

  -- Calculate total number of students    
  rw h1 at *
  have h_total : total_students = 139 := by norm_num [h1]
  
  -- Calculate number of girls
  have h_girls : total_students - boys = 139 - 64 := by rw [h_total, h_boys]
  norm_num at h_girls
  
  exact h_girls

end number_of_girls_l103_103369


namespace rectangle_perimeter_l103_103302

theorem rectangle_perimeter 
    (w h : ℕ) 
    (rel_prime : Nat.coprime w h) 
    (w_pos : 0 < w) 
    (h_pos : 0 < h) 
    (squares_partition : ∃ s1 s2 s3 : ℕ, s1 + s2 = w ∧ s3 + w = h) : 
    2 * (w + h) = 64 :=
by 
  sorry

end rectangle_perimeter_l103_103302


namespace range_of_m_l103_103151

open Real

def f (x m: ℝ) : ℝ := x^2 - 2 * x + m^2 + 3 * m - 3

def p (m: ℝ) : Prop := ∃ x, f x m < 0

def q (m: ℝ) : Prop := (5 * m - 1 > 0) ∧ (m - 2 > 0)

theorem range_of_m (m : ℝ) : ¬ (p m ∨ q m) ∧ ¬ (p m ∧ q m) → (m ≤ -4 ∨ m ≥ 2) :=
by
  sorry

end range_of_m_l103_103151


namespace prob_P_plus_S_one_less_multiple_of_7_l103_103933

theorem prob_P_plus_S_one_less_multiple_of_7 :
  let a b : ℕ := λ x y, x ∈ Finset.range (60+1) ∧ y ∈ Finset.range (60+1) ∧ x ≠ y ∧ 1 ≤ x ∧ x ≤ 60 ∧ 1 ≤ y ∧ y ≤ 60,
      P : ℕ := ∀ a b, a * b,
      S : ℕ := ∀ a b, a + b,
      m : ℕ := (P + S) + 1,
      all_pairs : ℕ := Nat.choose 60 2,
      valid_pairs : ℕ := 444,
      probability : ℚ := valid_pairs / all_pairs
  in probability = 148 / 590 := sorry

end prob_P_plus_S_one_less_multiple_of_7_l103_103933


namespace point_P_on_line_l_intersection_curve_C_and_line_l_l103_103496

/-
We define the constants and sets required for the proof problem.
-/

def P : ℝ × ℝ := (0, real.sqrt 3)
def C (φ : ℝ) : ℝ × ℝ := (real.sqrt 2 * real.cos φ, 2 * real.sin φ)

noncomputable def l (θ : ℝ) : ℝ :=
  real.sqrt 3 / (2 * real.cos (θ - real.pi / 6))

def l_cartesian (x y : ℝ) : Prop :=
  real.sqrt 3 * x + y = real.sqrt 3

theorem point_P_on_line_l :
  l_cartesian P.1 P.2 :=
by sorry

theorem intersection_curve_C_and_line_l (t1 t2 : ℝ) (h1 : some_fc_to_find_roots t1 t2) :
  (1 / real.abs (PA t1)) + (1 / real.abs (PB t2)) = real.sqrt 14 :=
by sorry

end point_P_on_line_l_intersection_curve_C_and_line_l_l103_103496


namespace max_min_f_l103_103595

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem max_min_f :
  ∃ (m M : ℝ), (∀ x ∈ set.Icc (-3:ℝ) 0, f x ≤ M ∧ f x ≥ m) ∧ M = 3 ∧ m = -17 :=
by
  sorry

end max_min_f_l103_103595


namespace tan_alpha_implies_fraction_l103_103790

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
sorry

end tan_alpha_implies_fraction_l103_103790


namespace intersection_points_count_l103_103697

-- Define the absolute value functions
def f1 (x : ℝ) : ℝ := |3 * x + 6|
def f2 (x : ℝ) : ℝ := -|4 * x - 4|

-- Prove the number of intersection points is 2
theorem intersection_points_count : 
  (∃ x1 y1, (f1 x1 = y1) ∧ (f2 x1 = y1)) ∧ 
  (∃ x2 y2, (f1 x2 = y2) ∧ (f2 x2 = y2) ∧ x1 ≠ x2) :=
sorry

end intersection_points_count_l103_103697


namespace potatoes_leftover_l103_103976

-- Define the necessary conditions
def fries_per_potato : ℕ := 25
def total_potatoes : ℕ := 15
def fries_needed : ℕ := 200

-- Prove the goal
theorem potatoes_leftover : total_potatoes - (fries_needed / fries_per_potato) = 7 :=
sorry

end potatoes_leftover_l103_103976


namespace min_real_roots_l103_103849

/-- Minimum number of real roots for a polynomial of specific degree and distinct magnitude conditions -/
theorem min_real_roots (f : Polynomial ℝ) (h_deg : f.degree = 2010) (roots : Fin 2010 → ℂ)
  (h_root : ∀ i, f.is_root (roots i)) (h_distinct : (Finset.image (λ i, |roots i|) (Finset.univ : Finset (Fin 2010))).card = 1008) : 
  ∃ r, r.nat_abs = 6 ∧ r ∈ (Finset.image (λ x, x.nat_abs) (Finset.filter Polynomial.is_root f.leading_coeff)).card := 
begin
  sorry
end

end min_real_roots_l103_103849


namespace cost_of_plastering_l103_103638

/-- 
Let's define the problem conditions
Length of the tank (in meters)
-/
def tank_length : ℕ := 25

/--
Width of the tank (in meters)
-/
def tank_width : ℕ := 12

/--
Depth of the tank (in meters)
-/
def tank_depth : ℕ := 6

/--
Cost of plastering per square meter (55 paise converted to rupees)
-/
def cost_per_sq_meter : ℝ := 0.55

/--
Prove that the cost of plastering the walls and bottom of the tank is 409.2 rupees
-/
theorem cost_of_plastering (total_cost : ℝ) : 
  total_cost = 409.2 :=
sorry

end cost_of_plastering_l103_103638


namespace perfect_squares_solutions_l103_103265

noncomputable def isPerfectSquare (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem perfect_squares_solutions :
  ∀ (a b : ℕ),
    0 < a → 0 < b →
    (isPerfectSquare (↑a * ↑a - 4 * ↑b)) →
    (isPerfectSquare (↑b * ↑b - 4 * ↑a)) →
      (a = 4 ∧ b = 4) ∨
      (a = 5 ∧ b = 6) ∨
      (a = 6 ∧ b = 5) :=
by
  -- Proof omitted
  sorry

end perfect_squares_solutions_l103_103265


namespace problem_l103_103764

variable (f : ℝ → ℝ)

-- Condition 1: y = f(x) / e^x is an even function
def even_function : Prop := ∀ x : ℝ, f(-x) / real.exp(-x) = f(x) / real.exp(x)

-- Condition 2: y = f(x) / e^x is monotonically increasing on [0, +∞)
def monotonically_increasing : Prop := ∀ x y : ℝ, 0 ≤ x → x ≤ y → (f(x) / real.exp(x) ≤ f(y) / real.exp(y))

-- Proof problem: Show that given the conditions, ef(1) < f(2)
theorem problem :
  even_function f →
  monotonically_increasing f →
  real.exp(1) * f(1) < f(2) :=
by
  intros h_even h_mono
  -- Proof to be completed
  sorry

end problem_l103_103764


namespace correct_propositions_count_l103_103207

open Real

theorem correct_propositions_count :
  (∀ θ : ℝ, 0 < θ ∧ θ < π / 2 → sin (cos θ) < cos (sin θ)) ∧
  (∀ θ : ℝ, 0 < θ ∧ θ < π / 2 → cos (cos θ) > sin (sin θ)) ∧
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π → sin (cos θ) < cos (sin θ)) →
  (3 = 3) :=
by 
  intros h1 h2 h3
  exact Eq.refl 3

end correct_propositions_count_l103_103207


namespace power_equation_l103_103447

theorem power_equation (x : ℝ) (h : 128^3 = 16^x) : 2^{-x} = 1 / 2^(21 / 4) :=
sorry

end power_equation_l103_103447


namespace pulley_distance_l103_103649

noncomputable def radius_large : ℝ := 15
noncomputable def radius_small : ℝ := 5
noncomputable def distance_contact : ℝ := 30

theorem pulley_distance :
  let AE := distance_contact,
      BE := radius_large - radius_small,
      AB := real.sqrt (AE^2 + BE^2)
  in AB = 10 * real.sqrt 10 :=
by
  sorry

end pulley_distance_l103_103649


namespace prod_mu_is_integer_l103_103131

noncomputable def mu (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if ∃ p : ℕ, p.prime ∧ p^2 ∣ n then 0
  else if ∃ k : ℕ, n = List.prod (List.replicate k {k : ℕ // k.prime}) then (-1 : ℤ)^(List.length (List.replicate k {k : ℕ // k.prime}))
  else 0 

theorem prod_mu_is_integer {a : ℕ → ℕ} (h : ∀ m n : ℕ, m > 0 → n > 0 → a m > 0 → a n > 0 → a (Nat.gcd m n) = Nat.gcd (a m) (a n)) (n : ℕ) (hn : n > 0) :
  ∃ k : ℤ, (∏ d in (Nat.divisors n), (a d : ℤ) ^ (mu (n / d))) = k :=
sorry

end prod_mu_is_integer_l103_103131


namespace part1_identity_part2_identity_l103_103721

theorem part1_identity (n : ℕ) (hn : n ≥ 2) :
  ∏ k in finset.range (n-1), abs (cos (k * real.pi / n)) = (1/2)^n * (1 - (-1)^n) :=
sorry

theorem part2_identity (n : ℕ) (hn : n ≥ 2) :
  ∏ k in finset.range (n-1), real.sin (k * real.pi / n) = n * (1/2)^(n-1) :=
sorry

end part1_identity_part2_identity_l103_103721


namespace x2_term_coefficient_l103_103715

-- Define the polynomials
def poly1 : Polynomial ℝ := 3 * X^2 + 4 * X + 5
def poly2 : Polynomial ℝ := 6 * X^2 + 7 * X + 8

-- State the problem to find the coefficient of x^2 in the product of the polynomials
theorem x2_term_coefficient : (poly1 * poly2).coeff 2 = 82 := 
sorry

end x2_term_coefficient_l103_103715


namespace radius_increase_125_surface_area_l103_103907

theorem radius_increase_125_surface_area (r r' : ℝ) 
(increase_surface_area : 4 * π * (r'^2) = 2.25 * 4 * π * r^2) : r' = 1.5 * r :=
by 
  sorry

end radius_increase_125_surface_area_l103_103907


namespace polynomial_degree_one_l103_103841

noncomputable def is_real : ℂ → Prop := λ x, x.im = 0

theorem polynomial_degree_one 
  (p : ℂ → ℂ) 
  (n : ℕ)
  (h_poly : ∀ x : ℂ, p x = ∑ i in range (n + 1), (coeffs i : ℂ) * (x ^ i))
  (h_deg : degree p = n)
  (h_real : ∀ x : ℝ, is_real (p x))
  (h_inverse_real : ∀ y : ℂ, is_real (p y) → is_real y) :
  n = 1 :=
by
  sorry

end polynomial_degree_one_l103_103841


namespace polynomial_identity_l103_103379

theorem polynomial_identity : 
  ∀ x : ℝ, 
    5 * x^3 - 32 * x^2 + 75 * x - 71 = 
    5 * (x - 2)^3 + (-2) * (x - 2)^2 + 7 * (x - 2) - 9 :=
by 
  sorry

end polynomial_identity_l103_103379


namespace part_A_part_B_centers_of_Γ_locus_l103_103926

variables {A P Q I1 I2 : Point}
variables {S1 S2 Σ Γ : Circle}
variables {l l0 : Line}
variables {M1 M2 N1 N2 : Point}
variables {rotatesAround : Line → Point → Prop}
variables {equilateralTriangle : Point → Point → Point → Prop}

-- Conditions
-- Intersection and points
axiom intersection_point_A : ∃ A : Point, Intersects(A, S1) ∧ Intersects(A, S2)
axiom arbitrary_line_l : Intersects(l, A) ∧ Intersects(l, S1) ∧ Intersects(l, S2)
axiom fixed_line_l0 : Intersects(l0, S1) ∧ Intersects(l0, S2)
axiom points : l ∩ S1 = M1 ∧ l ∩ S2 = N1 ∧ l0 ∩ S1 = M2 ∧ l0 ∩ S2 = N2

-- Definitions
def equilateralOnSegment (x y z : Point) : Prop := equilateralTriangle x y z

-- Proof for Part A
theorem part_A :
  (rotatesAround l A) →
  (equilateralOnSegment M1 M2 P) →
  (∃ Σ : Circle, OnCircumference P Σ) ∧
  (FixedPoint (M1, P) I1) ∧ (FixedPoint (M2, P) I2) :=
sorry

-- Proof for Part B
theorem part_B :
  (rotatesAround l A) →
  (∃ Γ : Circle, OnCircumference Q Γ) :=
sorry

-- Proof for locus of centers of Γ
theorem centers_of_Γ_locus :
  ∀ l0 : Line, centersOfΓFormsCircularDependency :=
sorry

end part_A_part_B_centers_of_Γ_locus_l103_103926


namespace ofelia_ratio_is_two_l103_103169

noncomputable def OfeliaSavingsRatio : ℝ :=
  let january_savings := 10
  let may_savings := 160
  let x := (may_savings / january_savings)^(1/4)
  x

theorem ofelia_ratio_is_two : OfeliaSavingsRatio = 2 := by
  sorry

end ofelia_ratio_is_two_l103_103169


namespace percent_decrease_in_hours_l103_103270

variable {W H : ℝ} (W_nonzero : W ≠ 0) (H_nonzero : H ≠ 0)

theorem percent_decrease_in_hours
  (wage_increase : W' = 1.25 * W)
  (income_unchanged : W * H = W' * H')
  : (H' = 0.8 * H) → H' = H * (1 - 0.2) := by
  sorry

end percent_decrease_in_hours_l103_103270


namespace diego_apples_weight_l103_103702

-- Definitions based on conditions
def bookbag_capacity : ℕ := 20
def weight_watermelon : ℕ := 1
def weight_grapes : ℕ := 1
def weight_oranges : ℕ := 1

-- Lean statement to check
theorem diego_apples_weight : 
  bookbag_capacity - (weight_watermelon + weight_grapes + weight_oranges) = 17 :=
by
  sorry

end diego_apples_weight_l103_103702


namespace parallel_lines_a_eq_neg2_l103_103055

theorem parallel_lines_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 - a = 0) ↔ (x - (1/2) * y = 0)) → a = -2 :=
by sorry

end parallel_lines_a_eq_neg2_l103_103055


namespace range_abs_diff_l103_103690

theorem range_abs_diff (x : ℝ) : 
    ∃ y ∈ Set.range (λ x : ℝ, |x + 3| - |x - 5|), y ∈ Set.Ici (-8) :=
sorry

end range_abs_diff_l103_103690


namespace distance_traveled_l103_103650

def velocity (t : ℝ) : ℝ := t^2 + 1

theorem distance_traveled :
  (∫ t in (0:ℝ)..(3:ℝ), velocity t) = 12 :=
by
  simp [velocity]
  sorry

end distance_traveled_l103_103650


namespace central_angle_nonagon_l103_103590

theorem central_angle_nonagon : (360 / 9 = 40) :=
by
  sorry

end central_angle_nonagon_l103_103590


namespace lateral_surface_area_truncated_pyramid_l103_103113

theorem lateral_surface_area_truncated_pyramid
    (a p q : ℝ) 
    (h1 : p + q = 2 * a) :
    2 * (p + q) * a = 4 * a^2 :=
by
  calc
  2 * (p + q) * a = 2 * (2 * a) * a : by rw [h1]
  ... = 4 * a^2 : by ring

end lateral_surface_area_truncated_pyramid_l103_103113


namespace roger_saves_33_minutes_l103_103184

def time_spent (distance speed : ℕ) : ℝ := distance / speed

def total_time (times : List ℝ) : ℝ := times.sum

def roger_time_saved : ℝ :=
  let monday_time := time_spent 5 10
  let tuesday_time := time_spent 3 4
  let thursday_time := time_spent 6 6
  let saturday_time := time_spent 4 5
  let total_actual_time := total_time [monday_time, tuesday_time, thursday_time, saturday_time]
  let total_walk_time := time_spent 18 5
  (total_actual_time - total_walk_time) * 60

theorem roger_saves_33_minutes :
  roger_time_saved = 33 := by
  sorry

end roger_saves_33_minutes_l103_103184


namespace largest_possible_N_l103_103228

def Point := (ℝ × ℝ)
def Line := { l : Set Point // ∃ a b c : ℝ, (λ p : Point, a * p.1 + b * p.2 + c = 0) = l }

structure ProblemConditions :=
(P1 P2 P3 : Point)
(l1 l2 l3 : Line)
(hP1_not_on_l1 : P1 ∉ l1.1)
(hP1_not_on_l2 : P1 ∉ l2.1)
(hP1_not_on_l3 : P1 ∉ l3.1)
(hP2_not_on_l1 : P2 ∉ l1.1)
(hP2_not_on_l2 : P2 ∉ l2.1)
(hP2_not_on_l3 : P2 ∉ l3.1)
(hP3_not_on_l1 : P3 ∉ l1.1)
(hP3_not_on_l2 : P3 ∉ l2.1)
(hP3_not_on_l3 : P3 ∉ l3.1)

def is_good (P : Point) (l l' : Line) : Prop :=
∃ R : Point, R ∈ l.1 ∧ ∃ M, M ∈ l'.1 ∧ R = ⟨2 * M.1 - P.1, 2 * M.2 - P.2⟩

def is_excellent (l : Line) (conds : ProblemConditions) : Prop :=
∃ (i₁ j₁ i₂ j₂ : ℕ), i₁ ≠ i₂ ∨ j₁ ≠ j₂ ∧
  is_good ([conds.P1, conds.P2, conds.P3].nth i₁) l ([conds.l1.1, conds.l2.1, conds.l3.1].nth j₁) ∧
  is_good ([conds.P1, conds.P2, conds.P3].nth i₂) l ([conds.l1.1, conds.l2.1, conds.l3.1].nth j₂)

theorem largest_possible_N (conds : ProblemConditions) : ∃ N, N = 270 :=
sorry

end largest_possible_N_l103_103228


namespace toy_poodle_height_l103_103913

-- Define the heights of the poodles
variables (S M T : ℝ)

-- Conditions
def std_taller_min : Prop := S = M + 8
def min_taller_toy : Prop := M = T + 6
def std_height : Prop := S = 28

-- Goal: How tall is the toy poodle?
theorem toy_poodle_height (h1 : std_taller_min S M)
                          (h2 : min_taller_toy M T)
                          (h3 : std_height S) : T = 14 :=
by 
  sorry

end toy_poodle_height_l103_103913


namespace starting_player_wins_with_skillful_play_l103_103608

-- Define 6 points A, B, C, D, E, F on the plane
variables (A B C D E F : Point)

-- Condition: None of these points are collinear
axiom non_collinear : ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A B E ∧ ¬ collinear A B F ∧ 
                      ¬ collinear A C D ∧ ¬ collinear A C E ∧ ¬ collinear A C F ∧ ¬ collinear A D E ∧ 
                      ¬ collinear A D F ∧ ¬ collinear A E F ∧ ¬ collinear B C D ∧ ¬ collinear B C E ∧
                      ¬ collinear B C F ∧ ¬ collinear B D E ∧ ¬ collinear B D F ∧ ¬ collinear B E F ∧
                      ¬ collinear C D E ∧ ¬ collinear C D F ∧ ¬ collinear C E F ∧ ¬ collinear D E F

-- Game condition
-- Players take turns drawing line segments to connect any two points. The player who completes a triangle loses.

-- Goal: The starting player can always avoid losing
theorem starting_player_wins_with_skillful_play
  (A B C D E F : Point) (non_collinear : ¬ collinear A B C ∧ ¬ collinear A B D ∧ ¬ collinear A B E ∧ ¬ collinear A B F ∧ 
                                      ¬ collinear A C D ∧ ¬ collinear A C E ∧ ¬ collinear A C F ∧ ¬ collinear A D E ∧ 
                                      ¬ collinear A D F ∧ ¬ collinear A E F ∧ ¬ collinear B C D ∧ ¬ collinear B C E ∧
                                      ¬ collinear B C F ∧ ¬ collinear B D E ∧ ¬ collinear B D F ∧ ¬ collinear B E F ∧
                                      ¬ collinear C D E ∧ ¬ collinear C D F ∧ ¬ collinear C E F ∧ ¬ collinear D E F) :
  ∃ strategy : (list (Point × Point)) → Point × Point → bool, 
    (∀ moves opponent_move, winning_condition strategy moves opponent_move -/ assumming some winning_condition definition /-) := sorry

end starting_player_wins_with_skillful_play_l103_103608


namespace team_selection_count_l103_103599

-- Define the group of players and the specific quadruplets
def players := fin 18
def quadruplets := {x : fin 18 // x.val < 4}

-- Define the team-selection problem conditions
def exactly_two_quadruplets (selected : finset players) : Prop :=
  selected.card = 7 ∧ (selected.filter (λ x, x.val < 4)).card = 2

-- Prove the number of ways to select the team given the conditions
theorem team_selection_count :
  ∃ (count : ℕ), count = (nat.choose 4 2) * (nat.choose 14 5) :=
begin
  use 12012,
  have choose_quadruplets : nat.choose 4 2 = 6 := by norm_num,
  have choose_remaining : nat.choose 14 5 = 2002 := by norm_num,
  rw [choose_quadruplets, choose_remaining],
  norm_num,
end

end team_selection_count_l103_103599


namespace minimum_ships_in_fleet_l103_103692

-- Define the grid size and related conditions
def grid_size : ℕ := 10

def is_ship (grid : ℕ × ℕ → Prop) (i j : ℕ) := grid (i, j) ∧
  ((i + 1 < grid_size ∧ grid (i + 1, j)) ∨
   (j + 1 < grid_size ∧ grid (i, j + 1)) ∨
   (i > 0 ∧ grid (i - 1, j)) ∨
   (j > 0 ∧ grid (i, j - 1)))

def no_shared_vertex (fleet : ℕ × ℕ → Prop) : Prop :=
  ∀ i j, fleet (i, j) →
  (∀ di dj, ((di, dj) ≠ (0, 0) ∧ (di + dj = 1 ∨ di + dj = -1)) →
   (∃ m n, n = j + dj ∧ m = i + di ∧ ¬ fleet (m,n)))

-- Statement of the theorem to prove
theorem minimum_ships_in_fleet :
  ∃ fleet : ℕ × ℕ → Prop, 
  (∀ i j, i < grid_size → j < grid_size → fleet (i, j) ↔ is_ship fleet i j) ∧
  (no_shared_vertex fleet) ∧
  grid_size = 10 ∧ 
  ∀ new_ship, 
    (∀ i j, new_ship (i, j) → is_ship new_ship i j) → 
    (no_shared_vertex (λ x, fleet x ∨ new_ship x)) → 
    ∃ ship_count ≤ 16, 
      (∀ i j, fleet (i, j) ↔ new_ship (i, j) = false). :=
sorry  -- proof to be provided

end minimum_ships_in_fleet_l103_103692


namespace simplify_sqrt_expression_l103_103577

-- Define the expressions under the square root
def expr1 (x : ℝ) : ℝ := sqrt (5 * 2 * x)
def expr2 (x : ℝ) : ℝ := sqrt ((x ^ 3) * (5 ^ 3))

-- Define the goal expression
noncomputable def simplifiedExpr (x : ℝ) : ℝ := 25 * x ^ 2 * sqrt 2

-- Prove the equivalence
theorem simplify_sqrt_expression (x : ℝ) : expr1 x * expr2 x = simplifiedExpr x := 
by
  sorry

end simplify_sqrt_expression_l103_103577


namespace necessary_not_sufficient_condition_l103_103886

variable {α : Type} [LinearOrderedField α]
variable {f : α → α}
variable {x : α}

-- Defining the differentiability at a point
def differentiable_at (f : α → α) (x : α) : Prop :=
∃ f', HasDerivAt f f' x

-- Defining the extremum condition at a point
def has_extremum_at (f : α → α) (x : α) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x', |x' - x| < δ → ¬∃ l, l ≠ f(x) ∧ f(x') = l

-- The Lean statement for the problem
theorem necessary_not_sufficient_condition 
  (h_diff : differentiable_at f x) 
  (h_deriv_zero : deriv f x = 0) : 
  ¬ (has_extremum_at f x ↔ deriv f x = 0) :=
sorry

end necessary_not_sufficient_condition_l103_103886


namespace find_f_99_l103_103860

noncomputable def f : ℝ → ℝ := sorry

axiom f_periodic (x : ℝ) : f(x) * f(x + 2) = 2016
axiom f_at_one : f 1 = 2

theorem find_f_99 : f 99 = 1008 := by
  sorry

end find_f_99_l103_103860


namespace ball_return_to_A_l103_103316

theorem ball_return_to_A (A B C D : Type) :
  let people := [A, B, C, D]
  let pass_ways := λ (p1 p2 : Type), if p1 ≠ p2 then 1 else 0
  let total_ways := 3 * 1 * 3 * 1 + 3 * 2 * 2 * 1
  total_ways = 21 := 
by
  sorry

end ball_return_to_A_l103_103316


namespace cubic_sum_l103_103742

noncomputable def g (x : ℝ) : ℝ := x^3 - 3 * x^2 + 4 * x + 2

theorem cubic_sum : 
  (Finset.range 19).sum (λ i, g ((i + 1 : ℝ) / 10)) = 76 :=
begin
  sorry
end

end cubic_sum_l103_103742


namespace quadratic_other_x_intercept_l103_103725

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → a * x^2 + b * x + c = -3)
  (h_intercept : a * 1^2 + b * 1 + c = 0) : 
  ∃ x0 : ℝ, x0 = 9 ∧ a * x0^2 + b * x0 + c = 0 :=
by
  sorry

end quadratic_other_x_intercept_l103_103725


namespace trains_clear_time_l103_103942

noncomputable def length_train1 : ℝ := 150
noncomputable def length_train2 : ℝ := 165
noncomputable def speed_train1_kmh : ℝ := 80
noncomputable def speed_train2_kmh : ℝ := 65
noncomputable def kmh_to_mps (v : ℝ) : ℝ := v * (5/18)
noncomputable def speed_train1 : ℝ := kmh_to_mps speed_train1_kmh
noncomputable def speed_train2 : ℝ := kmh_to_mps speed_train2_kmh
noncomputable def total_distance : ℝ := length_train1 + length_train2
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_time : time_to_clear = 7.82 := 
sorry

end trains_clear_time_l103_103942


namespace incorrect_statement_A_l103_103194

-- Definitions based on the given conditions
def g (x : ℝ) : ℝ := (x^2 + 6 * x + 5) / (x + 1)

-- Mathematically restate the incorrect statement (A)
theorem incorrect_statement_A : ¬ (g 1 = 8) :=
by
  have h : g 1 = (1 + 5), -- Simplifies to 6 based on given conditions
  calc g 1 = (1^2 + 6 * 1 + 5) / (1 + 1) : rfl
       ... = (1 + 6 + 5) / 2 : by norm_num
       ... = 12 / 2 : rfl
       ... = 6 : rfl
  show ¬ (6 = 8),
  by norm_num
end

end incorrect_statement_A_l103_103194


namespace possible_gcd_values_l103_103629

theorem possible_gcd_values (a b : ℕ) (h : gcd a b * lcm a b = 200) : 
  (∃ g : ℕ, g ∈ (multiset.of_list [1, 2, 5, 10]).nodup) := sorry

end possible_gcd_values_l103_103629


namespace max_determinant_value_l103_103005

theorem max_determinant_value : 
  ∃ θ, 
  let det := Matrix.det ![
      ![1, 1, 1],
      ![1, 1 + Real.sin θ ^ 2, 1],
      ![1 + Real.cos θ ^ 2, 1, 1]
  ]
  in det = 1 := 
by 
  sorry

end max_determinant_value_l103_103005


namespace relationship_between_x_and_y_l103_103457

theorem relationship_between_x_and_y
  (x y : ℝ)
  (h1 : 2 * x - 3 * y > 6 * x)
  (h2 : 3 * x - 4 * y < 2 * y - x) :
  x < y ∧ x < 0 ∧ y < 0 :=
sorry

end relationship_between_x_and_y_l103_103457


namespace cost_of_horse_is_2000_l103_103315

/-- 
Question: What is the cost of a horse?
Conditions: 
1. Albert buys 4 horses and 9 cows for Rs. 13,400.
2. Selling the horses at 10% profit and the cows at 20% profit, Albert earns a total profit of Rs. 1,880.
Expected answer: The cost of a horse (H) is Rs. 2000.
-/

variables (H C : ℝ)

def condition1 : Prop := 4 * H + 9 * C = 13400
def condition2 : Prop := 0.1 * (4 * H) + 0.2 * (9 * C) = 1880

theorem cost_of_horse_is_2000 (h1 : condition1 H C) (h2 : condition2 H C) : H = 2000 :=
sorry

end cost_of_horse_is_2000_l103_103315


namespace all_inequalities_true_l103_103081

variables {x y z : ℝ}
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : x > y)
variable (h4 : z > 0)

theorem all_inequalities_true :
  (x + z > y + z) ∧
  (x - 2z > y - 2z) ∧
  (xz^2 > yz^2) ∧
  (\(\frac{x}{z} > \frac{y}{z}\)) ∧
  (x - z^2 > y - z^2) :=
by
  sorry

end all_inequalities_true_l103_103081


namespace people_after_second_turn_l103_103882

noncomputable def number_of_people_in_front_after_second_turn (formation_size : ℕ) (initial_people : ℕ) (first_turn_people : ℕ) : ℕ := 
  if formation_size = 9 ∧ initial_people = 2 ∧ first_turn_people = 4 then 6 else 0

theorem people_after_second_turn :
  number_of_people_in_front_after_second_turn 9 2 4 = 6 :=
by
  -- Prove the theorem using the conditions and given data
  sorry

end people_after_second_turn_l103_103882


namespace length_of_DE_l103_103883

theorem length_of_DE 
  (base : ℝ)
  (fold_area_ratio : ℝ)
  (fold_area_percentage : ℝ)
  (original_area : ℝ := 1 / 2 * base * height)
  (height : ℝ)
  (DE_length : ℝ)
  (percentage : ℝ)
  (h_base : base = 18) 
  (h_fold_area_percentage : fold_area_percentage = 0.09) 
  (h_fold_area_ratio : fold_area_ratio = 0.3)
  (h_percentage : percentage = 9)
  (h_DE_length : DE_length = fold_area_ratio * base) :
  DE_length = 5.4 :=
  by
  have h1 : fold_area_ratio = real.sqrt fold_area_percentage := by sorry
  have h2 : derivative_area_percentage_correct := by sorry
  have h3 : appliance_area_side_length_ratio_correct := by sorry
  have h4 : correct_folded_area_to_correct_side_length_ratio := by sorry
  calc
    DE_length = fold_area_ratio * base := by sorry
           ... = 0.3 * 18 := by 
                                    rw [h_base, h_fold_area_ratio]
           ... = 5.4 := by norm_num

end length_of_DE_l103_103883


namespace number_of_blue_parrots_l103_103555

-- Defining the known conditions
def total_parrots : ℕ := 120
def fraction_red : ℚ := 2 / 3
def fraction_green : ℚ := 1 / 6

-- Proving the number of blue parrots given the conditions
theorem number_of_blue_parrots : (1 - (fraction_red + fraction_green)) * total_parrots = 20 := by
  sorry

end number_of_blue_parrots_l103_103555


namespace constant_max_value_l103_103096

theorem constant_max_value (n : ℤ) (c : ℝ) (h1 : c * (n^2) ≤ 8100) (h2 : n = 8) :
  c ≤ 126.5625 :=
sorry

end constant_max_value_l103_103096


namespace find_train_length_and_speed_l103_103310

variables {a t1 t2 : ℝ}

noncomputable def length_of_train (a t1 t2 : ℝ) : ℝ :=
  a * t1 / (t2 - t1)

noncomputable def speed_of_train (a t1 t2 : ℝ) : ℝ :=
  a / (t2 - t1)

theorem find_train_length_and_speed (h1 : t1 > 0) (h2 : t2 > t1) :
  ∃ (L V : ℝ), 
    L = length_of_train a t1 t2 ∧ 
    V = speed_of_train a t1 t2 :=
by
  use (length_of_train a t1 t2)
  use (speed_of_train a t1 t2)
  split
  · rfl
  · rfl

end find_train_length_and_speed_l103_103310


namespace part1_part2_part3_l103_103879

def attention_linear (x : ℝ) (h : 0 ≤ x ∧ x ≤ 8) : ℝ := 2 * x + 68

def attention_parabola (x : ℝ) (h : 8 < x) : ℝ := - (1 / 8) * (x - 16)^2 + 92

theorem part1 (x : ℝ) (h : x = 8) : attention_linear x (and.intro (le_refl 8) (le_refl 8)) = 84 := 
by sorry

theorem part2 (t : ℕ) (ht : 0 ≤ t ∧ t ≤ 45) : 
  let ideal_duration := (ite (0 ≤ t ∧ t ≤ 8) (2 * t + 68 ≥ 80) (-(1 / 8) * (t - 16)^2 + 92 ≥ 80)) in
  ∑ t in list.range 46, boolean.to_nat (ideal_duration t) = 20 := 
by sorry

theorem part3 (t : ℕ) (ht : 0 ≤ t ∧ t ≤ 6) : 
  let start_time := (2 * t + 68 = - (1 / 8) * (t + 24 - 16)^2 + 92) in
  start_time = 4 := 
by sorry

end part1_part2_part3_l103_103879


namespace coplanar_vertices_sum_even_l103_103706

theorem coplanar_vertices_sum_even (a b c d e f g h : ℤ) :
  (∃ (a b c d : ℤ), true ∧ (a + b + c + d) % 2 = 0) :=
sorry

end coplanar_vertices_sum_even_l103_103706


namespace count_triangles_hexagon_l103_103084

variable (A B C D E F O : Type)

def is_hexagon_with_center (A B C D E F O : Type) : Prop :=
  -- Here, specific definitions can be set up for points A, B, C, D, E, F forming a hexagon with center O
  sorry

theorem count_triangles_hexagon (A B C D E F O : Type) 
  [is_hexagon_with_center A B C D E F O] : 
  (6 + 3 + 6 + 1 = 16) :=
by
  exact rfl

end count_triangles_hexagon_l103_103084


namespace min_value_of_expr_l103_103163

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  ((x^2 + 1 / y^2 + 1) * (x^2 + 1 / y^2 - 1000)) +
  ((y^2 + 1 / x^2 + 1) * (y^2 + 1 / x^2 - 1000))

theorem min_value_of_expr :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ min_value_expr x y = -498998 :=
by
  sorry

end min_value_of_expr_l103_103163


namespace female_salmon_returned_l103_103707

/-- The number of female salmon that returned to their rivers is 259378,
    given that the total number of salmon that made the trip is 971639 and
    the number of male salmon that returned is 712261. -/
theorem female_salmon_returned :
  let n := 971639
  let m := 712261
  let f := n - m
  f = 259378 :=
by
  rfl

end female_salmon_returned_l103_103707


namespace complex_number_expression_l103_103405

def i : ℂ := complex.I

theorem complex_number_expression : (complex.pow ((1 + i) / (1 - i)) 3) = -i :=
by
  sorry

end complex_number_expression_l103_103405


namespace transport_tax_correct_l103_103512

-- Define the conditions
def car_horsepower : ℕ := 150
def tax_rate : ℕ := 20
def tax_period_months : ℕ := 8

-- Define the function to calculate the annual tax
def annual_transport_tax (horsepower : ℕ) (rate : ℕ) : ℕ :=
  horsepower * rate

-- Define the function to prorate the annual tax
def prorated_tax (annual_tax : ℕ) (months : ℕ) : ℕ :=
  (annual_tax * months) / 12

-- The proof problem: Prove the amount of transport tax Ivan needs to pay
theorem transport_tax_correct :
  let annual_tax := annual_transport_tax car_horsepower tax_rate in
  let prorated_tax := prorated_tax annual_tax tax_period_months in
  prorated_tax = 2000 :=
by 
  sorry

end transport_tax_correct_l103_103512


namespace x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l103_103091

variable {x y : ℝ}

theorem x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13
  (h1 : x + y = 10) 
  (h2 : x * y = 12) : 
  x^3 - y^3 = 176 * Real.sqrt 13 := 
by
  sorry

end x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l103_103091


namespace quadrilateral_cosine_floor_l103_103493

theorem quadrilateral_cosine_floor (A B C D : ℝ) 
  (h_ang_eq : ∠A = ∠C) 
  (h_AB_CD : AB = 200 ∧ CD = 200)
  (h_neq_AD_BC : AD ≠ BC) 
  (h_perimeter : AD + BC + AB + CD = 660) : 
  ⌊1000 * real.cos A⌋ = 650 := 
sorry

end quadrilateral_cosine_floor_l103_103493


namespace line_through_orthocenter_l103_103233

theorem line_through_orthocenter (ABC : Triangle)
  (circum : Circumcircle ABC)
  (D : Point) (K L : Point)
  (hD_diam : D = circumcircle_of_point ABC circum A) -- "D is on the circumcircle and AD is the diameter"
  (hK_on_AB : K ∈ segment AB)
  (hL_on_AC : L ∈ segment AC)
  (tangent1 : tangent_to_circle (line_through D K) (circle_def K L))
  (tangent2 : tangent_to_circle (line_through D L) (circle_def K L)) :
  passes_through (line_through K L) (orthocenter ABC) := 
  sorry

end line_through_orthocenter_l103_103233


namespace maximize_distance_l103_103681

noncomputable def maxTotalDistance (x : ℕ) (y : ℕ) (cityMPG highwayMPG : ℝ) (totalGallons : ℝ) : ℝ :=
  let cityDistance := cityMPG * ((x / 100.0) * totalGallons)
  let highwayDistance := highwayMPG * ((y / 100.0) * totalGallons)
  cityDistance + highwayDistance

theorem maximize_distance (x y : ℕ) (hx : x + y = 100) :
  maxTotalDistance x y 7.6 12.2 24.0 = 7.6 * (x / 100.0 * 24.0) + 12.2 * ((100.0 - x) / 100.0 * 24.0) :=
by
  sorry

end maximize_distance_l103_103681


namespace area_of_region_inside_hexagon_but_outside_circle_l103_103299

theorem area_of_region_inside_hexagon_but_outside_circle :
  ∀ (P : Plane) (cube_volume : ℝ) (hexagon_area : ℝ) (circle_area : ℝ)
  (side_length : ℝ) (hexagon_side_length : ℝ) (circle_radius : ℝ),
    cube_volume = 1 →
    side_length = 1 →
    hexagon_side_length = √2 / 2 →
    circle_radius = 1 / 2 →
    hexagon_area = (3 * √3) / 4 →
    circle_area = π / 4 →
    hexagon_area - circle_area = (3 * √3 - π) / 4 :=
by
  intros P cube_volume hexagon_area circle_area side_length hexagon_side_length circle_radius
  intro h1 h2 h3 h4 h5 h6
  rw [h5, h6]
  sorry

end area_of_region_inside_hexagon_but_outside_circle_l103_103299


namespace soccer_lineup_count_l103_103670

theorem soccer_lineup_count :
  let n_players := 18 in
  let n_defenders := 4 in
  let n_midfielders := 3 in
  let n_attackers := 3 in
  let remaining_after_goalie := n_players - 1 in
  let combinations (n k : ℕ) := Nat.choose n k in
  let choices_goalie := 18 in
  let choices_defenders := combinations remaining_after_goalie n_defenders in
  let choices_midfielders := combinations (remaining_after_goalie - n_defenders) n_midfielders in
  let choices_attackers := combinations ((remaining_after_goalie - n_defenders) - n_midfielders) n_attackers in
  choices_goalie * choices_defenders * choices_midfielders * choices_attackers = 147497760 :=
by
  sorry

end soccer_lineup_count_l103_103670


namespace enclosed_area_correct_l103_103527

def f (x : ℝ) := 1 + real.sqrt(1 - x^2)

noncomputable def enclosed_area : ℝ :=
  let integral_expr := ∫ x in 0..1, (1 + real.sqrt(1 - x^2) - x)
  2 * (integral_expr + 1/2)

theorem enclosed_area_correct : (enclosed_area ≈ 1.57) :=
  sorry

end enclosed_area_correct_l103_103527


namespace expression_sign_l103_103297

noncomputable def calculate_expression (a b c x r : ℝ) (h : a > b ∧ b > c ∧ c > 0 ∧ r > 0) : ℝ :=
  let t := x^2 - r^2 in
  let AP := Real.sqrt (a^2 + t) in
  let BQ := Real.sqrt (b^2 + t) in
  let CR := Real.sqrt (c^2 + t) in
  let AB := a - b in
  let BC := b - c in
  let AC := a - c in
  (AB * CR + BC * AP)^2 - (AC * BQ)^2

theorem expression_sign (a b c x r : ℝ) (h : a > b ∧ b > c ∧ c > 0 ∧ r > 0) :
  let t := x^2 - r^2 in
  (calculate_expression a b c x r h > 0 ↔ t > 0) ∧
  (calculate_expression a b c x r h = 0 ↔ t = 0) ∧
  (calculate_expression a b c x r h < 0 ↔ t < 0) :=
by 
sorry

end expression_sign_l103_103297


namespace trajectory_of_center_of_moving_circle_minimum_area_of_triangle_l103_103494

theorem trajectory_of_center_of_moving_circle :
  ∃ (E : ℝ → ℝ → Prop), (∀ x₀ y₀ : ℝ, E x₀ y₀ ↔ y₀^2 = 2 * x₀) :=
sorry

theorem minimum_area_of_triangle
  (P B C : ℝ × ℝ) (hB : B.1 = 0) (hC : C.1 = 0)
  (inscribed_circle_eqn : ∀ x y, (x-1)^2 + y^2 = 1) :
  ∃ (min_area : ℝ) (P₀ : ℝ × ℝ), 
  min_area = 8 ∧ (P₀ = (4, 2 * real.sqrt 2) ∨ P₀ = (4, -2 * real.sqrt 2)) :=
sorry

end trajectory_of_center_of_moving_circle_minimum_area_of_triangle_l103_103494


namespace perfect_square_trinomial_k_l103_103098

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ a : ℤ, (λ x : ℤ, x^2 + k*x + 9) = (λ x : ℤ, (x + a)^2)) ↔ k = 6 ∨ k = -6 :=
by sorry

end perfect_square_trinomial_k_l103_103098


namespace area_of_circle_l103_103869

structure Point (ℝ : Type) :=
(x : ℝ)
(y : ℝ)

def is_tangent_intersect_x_axis (A B : Point ℝ) (ω : set (Point ℝ)) : Prop :=
∃ T : Point ℝ, T.y = 0 ∧ (∃ lA lB : set (Point ℝ), tangent_line A lA ω ∧ 
tangent_line B lB ω ∧ T ∈ lA ∧ T ∈ lB)

def circle_area (A B : Point ℝ) (ω : set (Point ℝ)) (C: Point ℝ) : ℝ :=
let R := dist C {center_c ∈ ω | 
  ∀ P : Point ℝ, P ∈ ω ↔ dist P center_c = dist A C} in
π * R^2

theorem area_of_circle
  (A B : Point ℝ) (ω : set (Point ℝ)) (h: is_tangent_intersect_x_axis A B ω) :
  ∃C : Point ℝ, circle_area A B ω C = 11.83π :=
sorry

end area_of_circle_l103_103869


namespace students_wearing_other_colors_l103_103273

variable (total_students blue_percentage red_percentage green_percentage : ℕ)
variable (h_total : total_students = 600)
variable (h_blue : blue_percentage = 45)
variable (h_red : red_percentage = 23)
variable (h_green : green_percentage = 15)

theorem students_wearing_other_colors :
  (total_students * (100 - (blue_percentage + red_percentage + green_percentage)) / 100 = 102) :=
by
  sorry

end students_wearing_other_colors_l103_103273


namespace probability_correct_l103_103296

def faces := {-2, -1, 0, 1, 2, 3}

def is_monotonic_increasing (a b : ℤ) : Prop :=
  ∀ x > 0, 2 * a * (x:ℝ) + (b:ℝ) / (x:ℝ) ≥ 0

def favorable_outcomes : ℕ :=
  ∑ a in faces, ∑ b in faces, if is_monotonic_increasing a b then 1 else 0

def total_outcomes : ℕ :=
  (set.size faces) * (set.size faces)

def probability_of_monotone_function : ℚ :=
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_monotone_function = 4 / 9 := sorry

end probability_correct_l103_103296


namespace geometric_sequence_sixth_term_l103_103895

-- Definitions of conditions
def a : ℝ := 512
def r : ℝ := (2 / a)^(1 / 7)

-- The proof statement
theorem geometric_sequence_sixth_term (h : a * r^7 = 2) : 512 * (r^5) = 16 :=
begin
  sorry
end

end geometric_sequence_sixth_term_l103_103895


namespace max_value_m_l103_103215

theorem max_value_m (m n : ℕ) (h : 8 * m + 9 * n = m * n + 6) : m ≤ 75 := 
sorry

end max_value_m_l103_103215


namespace ratio_of_segments_l103_103667

theorem ratio_of_segments (x : ℝ) (x_pos : 0 < x) : 
  let a := 2 * x
      b := 3 * x in
  (2 * x + 3 * x) = x * real.sqrt 13 → 
  let c := x * real.sqrt 13,
  let h := (2 * x) * (3 * x) / (x * real.sqrt 13),
  let AD := h^2 * 4 / 9,
  let CD := h in
      AD / CD = 9 / 4 :=
by
  sorry

end ratio_of_segments_l103_103667


namespace lcm_consecutive_impossible_l103_103029

theorem lcm_consecutive_impossible (n : ℕ) (a : fin n → ℕ)
  (h : n = 10 ^ 1000)
  (b : fin n → ℕ)
  (hcirc : ∀ i, b i = nat.lcm (a i) (a ((i + 1) % n))) :
  ¬ (∃ f : fin n → ℕ, bijective f ∧ ∀ i, b i = f i ∧ f (i + 1) % n = f i + 1) :=
sorry

end lcm_consecutive_impossible_l103_103029


namespace inequality_proof_l103_103873

theorem inequality_proof (a b c d : ℝ) : 
  (a + b + c + d) * (a * b * (c + d) + (a + b) * c * d) - a * b * c * d ≤ 
  (1 / 2) * (a * (b + d) + b * (c + d) + c * (d + a))^2 :=
by
  sorry

end inequality_proof_l103_103873


namespace tan_C_l103_103862

theorem tan_C
    (A B C D E : Type)
    [RightTriangle A B C]
    (hBD : Trisects ∠ A B C B D)
    (hBE : Trisects ∠ A B C B E)
    (hAC : A C = 1)
    (hDE_AE : D E / A E = 3 / 7) :
    tan ∠ B A C = 7 / 10 :=
by 
  sorry

end tan_C_l103_103862


namespace contrapositive_l103_103070

variable (k : ℝ)

theorem contrapositive (h : ¬∃ x : ℝ, x^2 - x - k = 0) : k ≤ 0 :=
sorry

end contrapositive_l103_103070


namespace log_problem_l103_103788

theorem log_problem (x : ℝ) (hx : log 49 (x - 6) = 1 / 2) :
    1 / log x 7 = log 10 13 / log 10 7 :=
sorry

end log_problem_l103_103788


namespace range_a_l103_103063

def f (x : ℝ) : ℝ := abs (x^2 + x - 2)

theorem range_a (a : ℝ) : 
  ((∀ x : ℝ, f x - a * |x - 2| = 0 → x ∈ set_of(λ x, x ≠ -2 ∧ x ≠ 1)) → a ∈ Ioo 0 1)  :=
sorry

end range_a_l103_103063


namespace pave_hall_with_stones_l103_103269

def hall_length_m : ℕ := 36
def hall_breadth_m : ℕ := 15
def stone_length_dm : ℕ := 4
def stone_breadth_dm : ℕ := 5

def to_decimeters (m : ℕ) : ℕ := m * 10

def hall_length_dm : ℕ := to_decimeters hall_length_m
def hall_breadth_dm : ℕ := to_decimeters hall_breadth_m

def hall_area_dm2 : ℕ := hall_length_dm * hall_breadth_dm
def stone_area_dm2 : ℕ := stone_length_dm * stone_breadth_dm

def number_of_stones_required : ℕ := hall_area_dm2 / stone_area_dm2

theorem pave_hall_with_stones :
  number_of_stones_required = 2700 :=
sorry

end pave_hall_with_stones_l103_103269


namespace bananas_to_oranges_cost_l103_103684

noncomputable def cost_equivalence (bananas apples oranges : ℕ) : Prop :=
  (5 * bananas = 3 * apples) ∧
  (8 * apples = 5 * oranges)

theorem bananas_to_oranges_cost (bananas apples oranges : ℕ) 
  (h : cost_equivalence bananas apples oranges) :
  oranges = 9 :=
by sorry

end bananas_to_oranges_cost_l103_103684


namespace area_of_roof_l103_103597

def roof_area (w l : ℕ) : ℕ := l * w

theorem area_of_roof :
  ∃ (w l : ℕ), l = 4 * w ∧ l - w = 45 ∧ roof_area w l = 900 :=
by
  -- Defining witnesses for width and length
  use 15, 60
  -- Splitting the goals for clarity
  apply And.intro
  -- Proving the first condition: l = 4 * w
  · show 60 = 4 * 15
    rfl
  apply And.intro
  -- Proving the second condition: l - w = 45
  · show 60 - 15 = 45
    rfl
  -- Proving the area calculation: roof_area w l = 900
  · show roof_area 15 60 = 900
    rfl

end area_of_roof_l103_103597


namespace percentage_exceed_l103_103804

theorem percentage_exceed (x y : ℝ) (h : y = x + 0.2 * x) :
  (y - x) / x * 100 = 20 :=
by
  -- Proof goes here
  sorry

end percentage_exceed_l103_103804


namespace minimal_distance_proof_profit_proof_l103_103640

noncomputable def minimal_distance (p1 x1 p2 x2 : ℝ) : ℝ :=
  real.sqrt ((x1 - x2)^2 + (p1 - p2)^2)

variables (p1 x1 p2 x2 d : ℝ)

def first_project_condition (p1 x1 : ℝ) : Prop := 3 * x1 - 4 * p1 - 30 = 0
def second_project_condition (p2 x2 : ℝ) : Prop := p2^2 - 12 * p2 + x2^2 - 14 * x2 + 69 = 0
def deal_distance (p1 x1 p2 x2 : ℝ) : ℝ := minimal_distance p1 x1 p2 x2

theorem minimal_distance_proof
  (p1 x1 p2 x2 : ℝ)
  (h1 : first_project_condition p1 x1)
  (h2 : second_project_condition p2 x2) :
  deal_distance p1 x1 p2 x2 = 2.6 :=
sorry

def profit (p1 x1 p2 x2 : ℝ) : ℝ := x1 + x2 - (p1 + p2)

theorem profit_proof
  (p1 x1 p2 x2 : ℝ)
  (h1 : first_project_condition p1 x1)
  (h2 : second_project_condition p2 x2) :
  profit p1 x1 p2 x2 = 16840 :=
sorry

end minimal_distance_proof_profit_proof_l103_103640


namespace seating_arrangement_l103_103352

theorem seating_arrangement (x y z : ℕ) (h1 : z = x + y) (h2 : x*10 + y*9 = 67) : x = 4 :=
by
  sorry

end seating_arrangement_l103_103352


namespace ellipse_equation_dot_product_constant_l103_103397

section EllipseProblem

-- Given conditions in Lean 4
def a : ℝ := 2
def b : ℝ := sqrt 2
def c : ℝ := sqrt 2
def e : ℝ := sqrt 2 / 2
def focal_distance : ℝ := 2 * sqrt 2
def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Questions to be proven
theorem ellipse_equation :
  ∀ x y : ℝ, ellipse_eq x y ↔ (x^2 / 4) + (y^2 / 2) = 1 := by sorry

theorem dot_product_constant :
  ∀ (m : ℝ), let x_P := (2 * m^2 - 4) / (m^2 + 2),
                y_P := (4 * m) / (m^2 + 2),
                x_M := 2,
                y_M := 4 / m,
                OM := (x_M, y_M),
                OP := (x_P, y_P) in
  x_M * x_P + y_M * y_P = 4 := by sorry

end EllipseProblem

end ellipse_equation_dot_product_constant_l103_103397


namespace largest_prime_divisor_base6_l103_103004

theorem largest_prime_divisor_base6 : 
  let n := 1*6^8 + 0*6^7 + 0*6^6 + 1*6^5 + 1*6^4 + 1*6^3 + 0*6^2 + 0*6^1 + 1*6^0
  in is_prime 43 ∧ 43 ∣ n ∧ ∀ p : ℤ, is_prime p → p ∣ n → p ≤ 43 :=
by
  let n := 1*6^8 + 0*6^7 + 0*6^6 + 1*6^5 + 1*6^4 + 1*6^3 + 0*6^2 + 0*6^1 + 1*6^0
  sorry

end largest_prime_divisor_base6_l103_103004


namespace negation_of_exists_l103_103434

theorem negation_of_exists (x : ℝ) : ¬(∃ x_0 : ℝ, |x_0| + x_0^2 < 0) ↔ ∀ x : ℝ, |x| + x^2 ≥ 0 :=
by
  sorry

end negation_of_exists_l103_103434


namespace find_theta_find_alpha_beta_l103_103080

variables (α β : ℝ)
variables (a b : ℝ × ℝ)
variables (c : ℝ × ℝ := (0, 1))

def vector_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vector_b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)

theorem find_theta 
  (h1 : a = vector_a α) 
  (h2 : b = vector_b β) 
  (h3 : 0 < β) 
  (h4 : β < α) 
  (h5 : α < Real.pi) 
  (h6 : (Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)) = Real.sqrt 2) : 
  ∃ θ : ℝ, θ = Real.pi / 2 :=
sorry

theorem find_alpha_beta 
  (h1 : a = vector_a α) 
  (h2 : b = vector_b β) 
  (h3 : 0 < β) 
  (h4 : β < α) 
  (h5 : α < Real.pi) 
  (h6 : (a.1 + b.1, a.2 + b.2) = c) : 
  α = 5 * Real.pi / 6 ∧ β = Real.pi / 6 :=
sorry

end find_theta_find_alpha_beta_l103_103080


namespace stratified_sampling_l103_103809

theorem stratified_sampling
  (a d : ℝ)
  (total_students sample_size : ℕ)
  (total_students = 1500)
  (sample_size = 120)
  (students_A students_B students_C : ℝ)
  (students_A = a - d)
  (students_B = a)
  (students_C = a + d)
  (h : total_students = students_A + students_B + students_C)
  (stratified_sampling_fraction : ℝ := sample_size / total_students) :
  students_B * stratified_sampling_fraction = 40 := 
  sorry

end stratified_sampling_l103_103809


namespace harmonic_sum_exceeds_any_l103_103872

theorem harmonic_sum_exceeds_any (N : ℕ) :
  ∃ n : ℕ, (∑ k in Finset.range (n + 1), 1 / (k + 1 : ℝ)) > N :=
sorry

end harmonic_sum_exceeds_any_l103_103872


namespace solve_for_a_l103_103189

noncomputable def question (a b : ℝ) : Prop := 
  let z := complex.mk a b in
  let z1 := z - complex.mk 0 2 in
  let z2 := z + complex.mk 0 4 in
  z * z1 * z2 = complex.mk 0 4032

theorem solve_for_a : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (question a b) :=
begin
  sorry
end

end solve_for_a_l103_103189


namespace infinitely_many_negative_elements_l103_103396

def sequence (a : ℕ → ℝ) : ℕ → ℝ
| 0       := a 0
| (n + 1) := if a n ≠ 0 then (a n ^ 2 - 1) / (2 * a n) else 0

theorem infinitely_many_negative_elements (a1 : ℝ) :
  ∃ (S : set ℕ), (∀ n, n ∈ S ↔ (n ≥ 1 ∧ sequence (λ _, a1) n ≤ 0)) ∧ set.infinite S :=
sorry

end infinitely_many_negative_elements_l103_103396


namespace equal_circumcircle_radii_l103_103132

def Point := ℝ × ℝ

noncomputable def midpoint (A B : Point) : Point := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def projection (P : Point) (r : Point × Point) : Point :=
  let (x1, y1) := P
  let ((x2, y2), (x3, y3)) := r
  let dx := x3 - x2
  let dy := y3 - y2
  let d := dx * dx + dy * dy
  let a := (dx * (x1 - x2) + dy * (y1 - y2)) / d
  (x2 + a * dx, y2 + a * dy)

def collinear (P1 P2 P3 : Point) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x3, y3) := P3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

theorem equal_circumcircle_radii
  (A B : Point)
  (r : Point × Point)
  (R := projection A r)
  (S := projection B r)
  (M := midpoint A B)
  (h₀ : ¬collinear A M R)
  : (∃ O₁ R₁, circumcircle A M R = (O₁, R₁)) ∧ 
    (∃ O₂ R₂, circumcircle B M S = (O₂, R₂)) ∧
    ((∃ O₁ R₁, circumcircle A M R = (O₁, R₁)) ∧ (∃ O₂ R₂, circumcircle B M S = (O₂, R₂)) → R₁ = R₂) :=
begin
  sorry
end

end equal_circumcircle_radii_l103_103132


namespace ratio_retirement_account_l103_103776

def monthlyIncome : ℝ := 2500
def rent : ℝ := 700
def carPayment : ℝ := 300
def utilities : ℝ := carPayment / 2
def groceries : ℝ := 50
def expenses : ℝ := rent + carPayment + utilities + groceries
def moneyLeftAfterExpenses : ℝ := monthlyIncome - expenses
def remainingMoney : ℝ := 650
def retirementAccount : ℝ := moneyLeftAfterExpenses - remainingMoney

theorem ratio_retirement_account :
  (retirementAccount / remainingMoney) = 1 := by
  sorry

end ratio_retirement_account_l103_103776


namespace solve_eq1_solve_eq2_l103_103368

theorem solve_eq1 (x : ℝ) : 3 * (x - 2) ^ 2 = 27 ↔ (x = 5 ∨ x = -1) :=
by
  sorry

theorem solve_eq2 (x : ℝ) : (x + 5) ^ 3 + 27 = 0 ↔ x = -8 :=
by
  sorry

end solve_eq1_solve_eq2_l103_103368


namespace quadratic_range_and_value_l103_103420

theorem quadratic_range_and_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0)) →
  k ≤ 5 / 4 ∧ (∀ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0) ∧ (x1^2 + x2^2 = 16 + x1 * x2)) → k = -2 :=
by sorry

end quadratic_range_and_value_l103_103420


namespace sum_max_min_values_interval_l103_103898

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem sum_max_min_values_interval :
  let I := Icc 0 (Real.pi / 2),
      max_value := ⨆ x in I, f x,
      min_value := ⨅ x in I, f x
  in max_value + min_value = 1 / 2 :=
by
  sorry

end sum_max_min_values_interval_l103_103898


namespace problem_l103_103136

theorem problem (X : Set ℕ) (h1 : X.Nonempty) 
  (h2 : ∀ x ∈ X, 4 * x ∈ X) 
  (h3 : ∀ x ∈ X, ⌊Real.sqrt x⌋₊ ∈ X) :
  X = Set.univ := 
sorry

end problem_l103_103136


namespace four_consecutive_numbers_l103_103606

theorem four_consecutive_numbers (numbers : List ℝ) (h_distinct : numbers.Nodup) (h_length : numbers.length = 100) :
  ∃ (a b c d : ℝ) (h_seq : ([a, b, c, d] ∈ numbers.cyclicPermutations)), b + c < a + d :=
by
  sorry

end four_consecutive_numbers_l103_103606


namespace problem_solved_prob_l103_103620

theorem problem_solved_prob (pA pB : ℝ) (HA : pA = 1 / 3) (HB : pB = 4 / 5) :
  ((1 - (1 - pA) * (1 - pB)) = 13 / 15) :=
by
  sorry

end problem_solved_prob_l103_103620


namespace quadratic_root_solution_l103_103409

theorem quadratic_root_solution (k : ℤ) (a : ℤ) :
  (∀ x, x^2 + k * x - 10 = 0 → x = 2 ∨ x = a) →
  2 + a = -k →
  2 * a = -10 →
  k = 3 ∧ a = -5 :=
by
  sorry

end quadratic_root_solution_l103_103409


namespace find_AC_l103_103103

-- Given definitions as per the conditions
noncomputable def AB : ℝ := real.sqrt 6
def angle_A : ℝ := 75
def angle_B : ℝ := 45

-- The theorem to prove
theorem find_AC (h1 : AB = real.sqrt 6) 
                (h2 : angle_A = 75) 
                (h3 : angle_B = 45) : 
                ∃ AC : ℝ, AC = 2 := 
by
  -- The proof goes here
  sorry

end find_AC_l103_103103


namespace sasha_wins_l103_103831

-- Define the initial positions and the movement constraints
def initial_red_position := (1, 1)
def initial_blue_position := (2018, 1)

-- Define the movement rule
def valid_move (pos : ℕ × ℕ) (move : ℕ × ℕ) (board_size : ℕ) : Prop :=
  let (x, y) := pos in
  let (dx, dy) := move in
  (x + dx < board_size ∧ y + dy < board_size) ∨
  (x + dy < board_size ∧ y + dx < board_size)

-- Define the game rules
def valid_position (pos1 pos2 : ℕ × ℕ) : Prop :=
  pos1 ≠ pos2

-- Winning strategy for Sasha
def winning_strategy (red_pos blue_pos : ℕ × ℕ) : Prop :=
  -- Sasha wins if there is always a valid move for her starting from the initial positions
  ∃ seq, (seq 0 = initial_blue_position) ∧
         (∀ n, valid_move (seq n) (20, 17) 2018 ∨ valid_move (seq n) (17, 20) 2018) ∧
         (∀ n, ∀ m < n, seq m ≠ seq n) -- No position repeats

theorem sasha_wins : winning_strategy initial_red_position initial_blue_position :=
sorry

end sasha_wins_l103_103831


namespace sqrt_x_minus_2_meaningful_in_reals_l103_103470

theorem sqrt_x_minus_2_meaningful_in_reals (x : ℝ) : (∃ (y : ℝ), y * y = x - 2) → x ≥ 2 :=
by
  sorry

end sqrt_x_minus_2_meaningful_in_reals_l103_103470


namespace copper_pipe_meters_l103_103986

-- Define the variables and the conditions
variables (C P : ℕ)
variables (cost_per_meter : ℕ := 4) -- Each meter costs $4
variables (total_cost : ℕ := 100) -- Total cost is $100

-- Define the equations based on the conditions
def plastic_pipe_eq : Prop := P = C + 5
def total_cost_eq : Prop := 4 * C + 4 * P = total_cost

-- State the proposition to be proven
theorem copper_pipe_meters (h1 : plastic_pipe_eq C P) (h2 : total_cost_eq C P) : C = 10 :=
begin
  sorry
end

end copper_pipe_meters_l103_103986


namespace intersection_diagonals_quadrilateral_l103_103594

theorem intersection_diagonals_quadrilateral
  (a d b c : ℝ)
  (h_quad_eq_1 : ∀ x, y = ax + b)
  (h_quad_eq_2 : ∀ x, y = ax + c)
  (h_quad_eq_3 : ∀ x, y = dx + b)
  (h_quad_eq_4 : ∀ x, y = dx + c) :
  (0, (b + c) / 2) = 
  let p1 := (0, b);
      p2 := (0, c);
      midpoint := (0, (b + c) / 2) in midpoint :=
sorry

end intersection_diagonals_quadrilateral_l103_103594


namespace food_consumption_reduction_l103_103962

theorem food_consumption_reduction :
  ∀ (N P : ℝ), 
  let new_students := 0.85 * N in 
  let new_price := 1.20 * P in 
  let C := N * P / (new_students * new_price) in
  C = 0.98039 :=
by
  intro N P
  let new_students := 0.85 * N
  let new_price := 1.20 * P
  let C := N * P / (new_students * new_price)
  have h : C = 0.98039 := sorry
  exact h

end food_consumption_reduction_l103_103962


namespace no_two_points_same_color_distance_one_l103_103300

/-- Prove that if a plane is colored using seven colors, it is not necessary that there will be two points of the same color exactly 1 unit apart. -/
theorem no_two_points_same_color_distance_one (coloring : ℝ × ℝ → Fin 7) :
  ¬ ∀ (x y : ℝ × ℝ), (dist x y = 1) → (coloring x = coloring y) :=
by
  sorry

end no_two_points_same_color_distance_one_l103_103300


namespace sign_of_f_a_k_l103_103749

noncomputable def bisection_method {f : ℝ → ℝ} (a b : ℝ) (h_cont : Continuous f) (ha : f a < 0) (hb : f b > 0) 
  : ℕ → ℝ × ℝ
| 0 := (a, b)
| (k + 1) := 
  let mid := (bisection_method k).1 + (bisection_method k).2 / 2 in
  if f mid = 0 then 
    (mid, mid) 
  else if f mid > 0 then 
    ((bisection_method k).1, mid) 
  else 
    (mid, (bisection_method k).2)

theorem sign_of_f_a_k (f : ℝ → ℝ) (h_cont : Continuous f) (h_unique_zero : ∃! x, x ∈ set.Ioo a b ∧ f x = 0) 
  (a b : ℝ) (ha : f a < 0) (hb : f b > 0) (n : ℕ) :
  ∀ (a_k b_k : ℝ), (a_k, b_k) = bisection_method f a b h_cont ha hb n → f a_k < 0 :=
sorry

end sign_of_f_a_k_l103_103749


namespace minimum_sheets_needed_for_boats_l103_103558

noncomputable def min_sheets_for_boats (B P H : ℕ) : ℕ :=
  if B + P + H = 250 ∧ B = 9 * (B / 9) ∧ P = 5 * (P / 5) ∧ H = 3 * (H / 3) then
    if P = 0 ∧ H = 0 then B / 9 else 1
  else 
    0 -- This else case handles invalid input situations for consistency.

theorem minimum_sheets_needed_for_boats :
  ∃ (n : ℕ), min_sheets_for_boats 0 1 249 = n ∧ n = 1 :=
by {
  existsi 1,
  unfold min_sheets_for_boats,
  split_ifs,
  try { sorry }
}

end minimum_sheets_needed_for_boats_l103_103558


namespace average_characteristics_l103_103153

def X : Set ℕ := {n | 1 ≤ n ∧ n ≤ 100}

def m (M : Set ℕ) : ℕ :=
  if M = ∅ then 0 else M.sup id + M.inf id

theorem average_characteristics : nat :=
  ∑ i in {M | M ⊆ X ∧ M ≠ ∅}, m M / (2^100 - 1) = 101 := sorry

end average_characteristics_l103_103153


namespace cube_volume_in_pyramid_l103_103301

-- Definition for the conditions and parameters of the problem
def pyramid_condition (base_length : ℝ) (triangle_side : ℝ) : Prop :=
  base_length = 2 ∧ triangle_side = 2 * Real.sqrt 2

-- Definition for the cube's placement and side length condition inside the pyramid
def cube_side_length (s : ℝ) : Prop :=
  s = (Real.sqrt 6 / 3)

-- The final Lean statement proving the volume of the cube
theorem cube_volume_in_pyramid (base_length triangle_side s : ℝ) 
  (h_base_length : base_length = 2)
  (h_triangle_side : triangle_side = 2 * Real.sqrt 2)
  (h_cube_side_length : s = (Real.sqrt 6 / 3)) :
  (s ^ 3) = (2 * Real.sqrt 6 / 9) := 
by
  -- Using the given conditions to assert the conclusion
  rw [h_cube_side_length]
  have : (Real.sqrt 6 / 3) ^ 3 = 2 * Real.sqrt 6 / 9 := sorry
  exact this

end cube_volume_in_pyramid_l103_103301


namespace compute_alternating_squares_sum_l103_103340

theorem compute_alternating_squares_sum :
  let M := (finset.range 50).sum (λ n, if even n then (nat.succ n)^2 else - (nat.succ n)^2)
  M = 1250 :=
by
  let M := (finset.range 50).sum (λ n, if even n then (nat.succ n)^2 else - (nat.succ n)^2)
  sorry

end compute_alternating_squares_sum_l103_103340


namespace find_f_sqrt_10_l103_103090

-- Definitions and conditions provided in the problem
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f_condition (f : ℝ → ℝ) : Prop := ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2 - 8*x + 30

-- The problem specific conditions for f
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_periodic : is_periodic_function f 2)
variable (h_condition : f_condition f)

-- The statement to prove
theorem find_f_sqrt_10 : f (Real.sqrt 10) = -24 :=
by
  sorry

end find_f_sqrt_10_l103_103090


namespace sum_of_squares_is_square_l103_103177

theorem sum_of_squares_is_square (n : ℕ) (a : Fin n → ℕ) (h : Function.Injective a) :
  (∑ j in Finset.univ, a j ^ 2 * ∏ k in (Finset.univ.erase j), (a j + a k) / (a j - a k)) =
  (Finset.univ.sum (λ i, a i)) ^ 2 :=
by 
  sorry

end sum_of_squares_is_square_l103_103177


namespace lcm_consecutive_impossible_l103_103028

theorem lcm_consecutive_impossible (n : ℕ) (a : fin n → ℕ)
  (h : n = 10 ^ 1000)
  (b : fin n → ℕ)
  (hcirc : ∀ i, b i = nat.lcm (a i) (a ((i + 1) % n))) :
  ¬ (∃ f : fin n → ℕ, bijective f ∧ ∀ i, b i = f i ∧ f (i + 1) % n = f i + 1) :=
sorry

end lcm_consecutive_impossible_l103_103028


namespace problem_solution_l103_103314

def problem_conditions : Prop :=
  (∃ (students_total excellent_students: ℕ) 
     (classA_excellent classB_not_excellent: ℕ),
     students_total = 110 ∧
     excellent_students = 30 ∧
     classA_excellent = 10 ∧
     classB_not_excellent = 30)

theorem problem_solution
  (students_total excellent_students: ℕ)
  (classA_excellent classB_not_excellent: ℕ)
  (h : problem_conditions) :
  ∃ classA_not_excellent classB_excellent: ℕ,
    classA_not_excellent = 50 ∧
    classB_excellent = 20 ∧
    ((∃ χ_squared: ℝ, χ_squared = 7.5 ∧ χ_squared > 6.635) → true) ∧
    (∃ selectA selectB: ℕ, selectA = 5 ∧ selectB = 3) :=
by {
  sorry
}

end problem_solution_l103_103314


namespace expected_length_of_string_is_12_l103_103192

noncomputable def expected_length_of_string : ℝ :=
  expected_value_of_length

theorem expected_length_of_string_is_12 :
  expected_length_of_string = 12 :=
sorry

end expected_length_of_string_is_12_l103_103192


namespace find_distance_d_l103_103355

noncomputable def equilateral_triangle_distance (side_length : ℕ) : ℝ :=
  let r := (Math.sqrt 3) / 6 * side_length in
  let R := (Math.sqrt 3) / 3 * side_length in
  let D := side_length in
  -- Assume mathematical setup relating to the problem
  let k := R and θ := 150 in
  let dihedral_angle := Real.pi * θ / 180 in
  let KP := x and KQ := y in
  let OP := OQ in
  let d_eq := (2 * R / Math.sqrt 3 / 2) * Math.sqrt (1 + Math.cos dihedral_angle) in
  let cond1 : (PA = PB = PC ∧ QA = QB = QC) := sorry in
  let cond2 : (d = D) := sorry in
  d_eq

/--
Given:
1. Equilateral triangle \( \triangle ABC \) has side length 720.
2. Points \( P \) and \( Q \) lie outside the plane of \( \triangle ABC \) and are on opposite sides of the plane.
3. \( PA = PB = PC \) and \( QA = QB = QC \).
4. The planes of \( \triangle PAB \) and \( \triangle QAB \) form a 150° dihedral angle.
There exists a point \( O \) whose distance from each of \( A, B, C, P, \) and \( Q \) is \( d \).

We want to prove that \( d = 480 \).
-/
theorem find_distance_d : 
  ∃ O : ℝ, ∀ A B C P Q : ℝ, 
  (equilateral_triangle_distance 720 = 480) := 
sorry

end find_distance_d_l103_103355


namespace sum_of_squared_distances_range_l103_103415

-- Define the curves and points as per given conditions
def C1_parametric (φ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos φ, 3 * Real.sin φ)

def point_A : ℝ × ℝ := (0, 2)
def point_B : ℝ × ℝ := (-2, 0)
def point_C : ℝ × ℝ := (0, -2)
def point_D : ℝ × ℝ := (2, 0)

-- Function to compute the squared distance between two points
def squared_dist (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

-- Function to compute the sum of squared distances
def sum_squares (φ : ℝ) : ℝ :=
  let P := C1_parametric φ
  squared_dist P point_A + squared_dist P point_B +
  squared_dist P point_C + squared_dist P point_D

-- The main theorem to be proved
theorem sum_of_squared_distances_range :
  ∀ φ : ℝ, 32 ≤ sum_squares φ ∧ sum_squares φ ≤ 52 :=
sorry

end sum_of_squared_distances_range_l103_103415


namespace roots_form_parallelogram_l103_103713

theorem roots_form_parallelogram :
  let polynomial := fun (z : ℂ) (a : ℝ) =>
    z^4 - 8*z^3 + 13*a*z^2 - 2*(3*a^2 + 2*a - 4)*z - 2
  let a1 := 7.791
  let a2 := -8.457
  ∀ z1 z2 z3 z4 : ℂ,
    ( (polynomial z1 a1 = 0) ∧ (polynomial z2 a1 = 0) ∧ (polynomial z3 a1 = 0) ∧ (polynomial z4 a1 = 0)
    ∨ (polynomial z1 a2 = 0) ∧ (polynomial z2 a2 = 0) ∧ (polynomial z3 a2 = 0) ∧ (polynomial z4 a2 = 0) )
    → ( (z1 + z2 + z3 + z4) / 4 = 2 )
    → ( Complex.abs (z1 - z2) = Complex.abs (z3 - z4) 
      ∧ Complex.abs (z1 - z3) = Complex.abs (z2 - z4) ) := sorry

end roots_form_parallelogram_l103_103713


namespace area_triangle_eq_l103_103164

variables {a b λ : ℝ} (P P1 P2 : ℝ × ℝ)
variable (h1 : 0 < a)
variable (h2 : 0 < b)

def hyperbola (P : ℝ × ℝ) : Prop := 
  P.1^2 / a^2 - P.2^2 / b^2 = 1

def asymptote1 (P : ℝ × ℝ) : Prop :=
  P.2 = (b / a) * P.1

def asymptote2 (P : ℝ × ℝ) : Prop :=
  P.2 = -(b / a) * P.1

theorem area_triangle_eq :
  hyperbola P →
  asymptote1 P1 →
  asymptote2 P2 →
  (λ : ℝ) → 
  λ = (dist P1 P / dist P P2) →
  (1/2 * abs ((1 + λ)^2 / (4 * abs λ) * a * b)) = (1 + λ)^2 / (4 * abs λ) * a * b :=
sorry

end area_triangle_eq_l103_103164


namespace max_two_colorable_disk_radius_l103_103241

noncomputable def two_colorable_disk (r : ℝ) : Prop :=
  ∃ (coloring : ℝ × ℝ → bool), 
    ∀ (p q : ℝ × ℝ), dist p q = 1 → coloring p ≠ coloring q

theorem max_two_colorable_disk_radius :
  ∀ (r : ℝ), two_colorable_disk r ↔ r <= 1 / 2 := 
by
  sorry

end max_two_colorable_disk_radius_l103_103241


namespace max_surface_area_inscribed_sphere_l103_103502

noncomputable theory
open Real

-- Define the right triangular prism and its geometrical properties
def right_triangular_prism (a b r : ℝ) := 
  (a^2 + b^2 = 25) ∧ (r = a * b / (a + b + 5))

-- maximum surface area of the inscribed sphere
theorem max_surface_area_inscribed_sphere (a b r : ℝ) 
  (h : right_triangular_prism a b r) : 
  ∃ r : ℝ, r = 5 / 2 * (sqrt 2 - 1) → 
  4 * π * r^2 = 25 * (3 - 3 * sqrt 2) * π :=
sorry

end max_surface_area_inscribed_sphere_l103_103502


namespace translate_right_by_pi_over_4_l103_103087

variable (x : ℝ)

-- Define the functions y1 and y2
def y1 : ℝ → ℝ := λ x, sin (2 * x) + cos (2 * x)
def y2 : ℝ → ℝ := λ x, sin (2 * x) - cos (2 * x)

-- Define the transformation function
def transformed_y1 (x : ℝ) : ℝ := y1 (x - π / 4)

-- Theorem statement: translating y1 right by π/4 equals y2
theorem translate_right_by_pi_over_4 : transformed_y1 x = y2 x :=
by
  sorry

end translate_right_by_pi_over_4_l103_103087


namespace fgh_supermarkets_l103_103923

theorem fgh_supermarkets (U C : ℕ) 
  (h1 : U + C = 70) 
  (h2 : U = C + 14) : U = 42 :=
by
  sorry

end fgh_supermarkets_l103_103923


namespace largest_term_in_expansion_of_binomial_l103_103089

theorem largest_term_in_expansion_of_binomial :
  (\sum k in finset.range(n + 1), (1 / (k + 1 : ℚ)) * nat.choose n k) = (31 / (n + 1)) → 
  n = 4 → 
  ∃ k, k * nat.choose 8 k = 70 ∧ k = 4 := 
by
  sorry

end largest_term_in_expansion_of_binomial_l103_103089


namespace instantaneous_speed_at_t1_l103_103298

open Real

noncomputable def s (t : ℝ) : ℝ := 2 * t^3
def velocity (s : ℝ → ℝ) (t : ℝ) : ℝ := deriv s t

theorem instantaneous_speed_at_t1 : velocity s 1 = 6 := by
  apply deriv_eq_slope (λ x, 2 * x^3)
  sorry

end instantaneous_speed_at_t1_l103_103298


namespace smallest_x_in_domain_of_ff_l103_103796

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x - 5)

theorem smallest_x_in_domain_of_ff : ∃ x : ℝ, x = 30 ∧ ∀ y : ℝ, y < 30 → ¬ (y ∈ set_of (λ z, z ∈ (set_of (λ w, w ≥ 5).preimage f))).preimage f :=
by sorry

end smallest_x_in_domain_of_ff_l103_103796


namespace prove_triangle_removal_l103_103323

noncomputable def triangle_removal (initial_matches : ℕ) (initial_triangles : ℕ) (removed_matches : ℕ) : Prop :=
  initial_matches = 16 ∧ initial_triangles = 8 ∧ removed_matches = 4 →
  let remaining_triangles := 4 in
  (initial_matches - removed_matches) = (remaining_triangles * 3) ∧ remaining_triangles = 4

theorem prove_triangle_removal :
  triangle_removal 16 8 4 :=
by {
  unfold triangle_removal,
  intro h,
  cases h with h1 h2,
  cases h2 with h21 h22,
  rw [←h1, ←h21, ←h22],
  exact ⟨refl 12, refl 4⟩,
  sorry
}

end prove_triangle_removal_l103_103323


namespace duty_arrangements_l103_103324

theorem duty_arrangements :
  ∃ (arrangements : ℕ), arrangements = 96 ∧ 
  ∀ (days : fin 6 → option (fin 4)),
    (∀ i, days i ≠ none) ∧
    (∀ j : fin 4, j ≠ 3 → ∃! i : fin 6, days i = some j) ∧
    (∃ i : fin (6 - 2 + 1), ∀ j : fin 3, days (i + j) = some 3) ↔
    arrangements = 96 :=
sorry

end duty_arrangements_l103_103324


namespace continuity_test_l103_103126

noncomputable def f (x : ℝ) : ℝ :=
  if x < -3 then (x^2 + 3*x - 1) / (x + 2)
  else if x <= 4 then (x + 2)^2
  else 9*x + 1

-- Statement: proving the function is continuous at x = -3 and not continuous at x = 4.
theorem continuity_test (x : ℝ) : 
  (continuous_at f (-3) ∧ ¬continuous_at f 4) := sorry

end continuity_test_l103_103126


namespace solve_system_l103_103868

theorem solve_system :
  (∃ x y : ℝ, 4 * x + y = 5 ∧ 2 * x - 3 * y = 13) ↔ (x = 2 ∧ y = -3) :=
by
  sorry

end solve_system_l103_103868


namespace berkeley_students_b_l103_103481

theorem berkeley_students_b (A B M : ℕ) (ratio: Rat):
  A = 12 → B = 20 → M = 30 → ratio = A / B → 5 * M * ratio = 18 * 5 :=
by intros A_eq B_eq M_eq ratio_eq;
   rw [A_eq, B_eq, M_eq, ratio_eq];
   norm_num;
   rw [←nat.cast_mul, ←nat.cast_mul]; 
   norm_num

# Evaluated statement
example : berkeley_students_b 12 20 30 (12 / 20)

end berkeley_students_b_l103_103481


namespace simplify_fraction_l103_103576

theorem simplify_fraction (a b m : ℝ) (h1 : (a / b) ^ m = (a^m) / (b^m)) (h2 : (-1 : ℝ) ^ (0 : ℝ) = 1) :
  ( (81 / 16) ^ (3 / 4) ) - 1 = 19 / 8 :=
by
  sorry

end simplify_fraction_l103_103576


namespace cos_eq_given_sin_l103_103784

theorem cos_eq_given_sin (a : ℝ) (h : sin (π / 3 + a) = 5 / 12) : cos (π / 6 - a) = 5 / 12 :=
sorry

end cos_eq_given_sin_l103_103784


namespace contradiction_proof_l103_103182

theorem contradiction_proof (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  have h1 : a = 0,
    sorry,
  have h2 : b = 0,
    sorry,
  exact ⟨h1, h2⟩

end contradiction_proof_l103_103182


namespace parallelogram_IJKL_l103_103528

variables {A B C D I J K L : Type} [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space D] [euclidean_space I] [euclidean_space J] [euclidean_space K] [euclidean_space L]

-- Define a quadrilateral
def quadrilateral (A B C D : Type) : Prop := True -- Placeholder; refine with exact definition as needed.

-- Define a point being part of an outward equilateral triangle
def outward_equilateral_triangle (A B I : Type) : Prop := True -- Placeholder; refine with exact definition as needed.

-- Define a point being part of an inward equilateral triangle
def inward_equilateral_triangle (B C J : Type) : Prop := True -- Placeholder; refine with exact definition as needed.

-- Given conditions
axiom non_intersecting_quad : quadrilateral A B C D
axiom equilateral_ABI : outward_equilateral_triangle A B I
axiom equilateral_CDK : outward_equilateral_triangle C D K
axiom equilateral_BCJ : inward_equilateral_triangle B C J
axiom equilateral_DAL : inward_equilateral_triangle D A L

-- Main theorem statement
theorem parallelogram_IJKL : parallelogram I J K L :=
by sorry

end parallelogram_IJKL_l103_103528


namespace quadratic_range_and_value_l103_103421

theorem quadratic_range_and_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0)) →
  k ≤ 5 / 4 ∧ (∀ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0) ∧ (x1^2 + x2^2 = 16 + x1 * x2)) → k = -2 :=
by sorry

end quadratic_range_and_value_l103_103421


namespace xyz_inequality_l103_103351

theorem xyz_inequality (x y z : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 1) (hy : 0 ≤ y) (hy' : y ≤ 1) (hz : 0 ≤ z) (hz' : z ≤ 1) :
  (x^2 / (1 + x + x*y*z) + y^2 / (1 + y + x*y*z) + z^2 / (1 + z + x*y*z) ≤ 1) :=
sorry

end xyz_inequality_l103_103351


namespace painting_count_is_nine_l103_103284

noncomputable def count_safe_paintings : ℕ :=
  let grid := fin 3 × fin 3 -- 3x3 grid
  let valid_painting (g: grid → bool) : Prop :=
    ∀ i j, (g (i, j) = true) → (∀ k < i, g (k, j) = true) ∧ (∀ l < j, g (i, l) = true)
  set.count (λ g : grid → bool, valid_painting g)

theorem painting_count_is_nine : count_safe_paintings = 9 := 
sorry

end painting_count_is_nine_l103_103284


namespace companyA_sold_bottles_l103_103230

-- Let CompanyA and CompanyB be the prices per bottle for the respective companies
def CompanyA_price : ℝ := 4
def CompanyB_price : ℝ := 3.5

-- Company B sold 350 bottles
def CompanyB_bottles : ℕ := 350

-- Total revenue of Company B
def CompanyB_revenue : ℝ := CompanyB_price * CompanyB_bottles

-- Additional condition that the revenue difference is $25
def revenue_difference : ℝ := 25

-- Define the total revenue equations for both scenarios
def revenue_scenario1 (x : ℕ) : Prop :=
  CompanyA_price * x = CompanyB_revenue + revenue_difference

def revenue_scenario2 (x : ℕ) : Prop :=
  CompanyA_price * x + revenue_difference = CompanyB_revenue

-- The problem translates to finding x such that either of these conditions hold
theorem companyA_sold_bottles : ∃ x : ℕ, revenue_scenario2 x ∧ x = 300 :=
by
  sorry

end companyA_sold_bottles_l103_103230


namespace probability_is_two_over_nine_l103_103285

-- Define the problem conditions and question
def labels := ["美", "丽", "中", "国"]
def draw_result := [[2, 3, 2], [3, 2, 1], [2, 3, 0], [0, 2, 3], [1, 2, 3], [0, 2, 1], [1, 3, 2], [2, 2, 0], [0, 0, 1],
                    [2, 3, 1], [1, 3, 0], [1, 3, 3], [2, 3, 1], [0, 3, 1], [3, 2, 0], [1, 2, 2], [1, 0, 3], [2, 3, 3]]
def stop_on_third := λ result, result[2] == 0 ∨ result[2] == 1 ∧ (0 ∈ result ∧ 1 ∈ result)

-- Check which results meet the condition of stopping exactly on third draw
def valid_results := List.filter stop_on_third draw_result
def probability_stop_on_third := valid_results.length / draw_result.length

-- Prove that the probability is 2/9
theorem probability_is_two_over_nine : probability_stop_on_third = 2/9 := 
by
  unfold probability_stop_on_third
  have : valid_results.length = 4 := sorry
  have : draw_result.length = 18 := sorry
  show 4 / 18 = 2 / 9 from sorry

end probability_is_two_over_nine_l103_103285


namespace stratified_sampling_vision_test_l103_103229

theorem stratified_sampling_vision_test 
  (n_total : ℕ) (n_HS : ℕ) (n_selected : ℕ)
  (h1 : n_total = 165)
  (h2 : n_HS = 66)
  (h3 : n_selected = 15) :
  (n_HS * n_selected / n_total) = 6 := 
by 
  sorry

end stratified_sampling_vision_test_l103_103229


namespace concyclic_points_l103_103104

open EuclideanGeometry

def midpoint (A B : Point) : Point := sorry

theorem concyclic_points 
  (A B C D M E F: Point) 
  (h₁ : ∠BAC = ∠BAD + ∠DAC) 
  (h₂ : D ∈ Line BC) 
  (h₃ : M = midpoint A D) 
  (Γ₁ : Circle) (Γ₂ : Circle) 
  (h₄ : diameter Γ₁ = AC) 
  (h₅ : diameter Γ₂ = AB) 
  (h₆ : E ∈ (Γ₁ ∩ Line BM)) 
  (h₇ : F ∈ (Γ₂ ∩ Line CM)) : 
  CyclicQuadrilateral B E F C :=
by sorry

end concyclic_points_l103_103104


namespace necessary_but_not_sufficient_l103_103642

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 1 = 0) ↔ (x = -1 ∨ x = 1) ∧ (x - 1 = 0) → (x^2 - 1 = 0) ∧ ¬((x^2 - 1 = 0) → (x - 1 = 0)) := 
by sorry

end necessary_but_not_sufficient_l103_103642


namespace bankers_discount_l103_103200

theorem bankers_discount (td fv bd : ℝ) (h_td : td = 36) (h_fv : fv = 252) :
  let pv := fv - td in
  let bd := (td / pv) * fv in
  bd = 42 :=
by
  intros
  rw [h_td, h_fv]
  have h_pv : 252 - 36 = 216 := by norm_num
  have h_eq : (36 / 216) * 252 = 42 := by norm_num
  rw [h_pv, h_eq]
  exact h_eq

end bankers_discount_l103_103200


namespace find_speed_of_stream_l103_103288

-- Define the conditions
variable boat_speed_in_still_water : ℝ
variable travel_time_downstream : ℝ
variable downstream_distance : ℝ

-- Given conditions
axiom h1 : boat_speed_in_still_water = 24
axiom h2 : travel_time_downstream = 2
axiom h3 : downstream_distance = 56

-- Define the speed of the stream as 'v' and the effective downstream speed
noncomputable def speed_of_stream (v : ℝ) : Prop :=
  downstream_distance = (boat_speed_in_still_water + v) * travel_time_downstream

-- The proof problem statement
theorem find_speed_of_stream : speed_of_stream 4 :=
by
  unfold speed_of_stream
  rw [h1, h2, h3]
  sorry

end find_speed_of_stream_l103_103288


namespace total_points_earned_l103_103266

def defeated_enemies := 15
def points_per_enemy := 12
def level_completion_points := 20
def special_challenges_completed := 5
def points_per_special_challenge := 10

theorem total_points_earned :
  defeated_enemies * points_per_enemy
  + level_completion_points
  + special_challenges_completed * points_per_special_challenge = 250 :=
by
  -- The proof would be developed here.
  sorry

end total_points_earned_l103_103266


namespace beaker_water_division_l103_103630

-- Given conditions
variable (buckets : ℕ) (bucket_capacity : ℕ) (remaining_water : ℝ)
  (total_buckets : ℕ := 2) (capacity : ℕ := 120) (remaining : ℝ := 2.4)

-- Theorem statement
theorem beaker_water_division (h1 : buckets = total_buckets)
                             (h2 : bucket_capacity = capacity)
                             (h3 : remaining_water = remaining) :
                             (total_water : ℝ := buckets * bucket_capacity + remaining_water ) → 
                             (water_per_beaker : ℝ := total_water / 3) →
                             water_per_beaker = 80.8 :=
by
  -- Skipping the proof steps here, will use sorry
  sorry

end beaker_water_division_l103_103630


namespace candles_lighting_time_l103_103618

def length_at_time (initial_length : ℝ) (burn_rate : ℝ) (time : ℝ) : ℝ :=
  initial_length * (1 - time / burn_rate)

theorem candles_lighting_time : 
  ∀ (t : ℝ), 
  let initial_length := 1 in -- length is arbitrary since it cancels out
  let burn_rate1 := 120 in -- first candle burns completely in 120 minutes
  let burn_rate2 := 300 in -- second candle burns completely in 300 minutes
  let time_since_lit := 6 * 60 - t in -- time since lit at 6 PM
  length_at_time initial_length burn_rate1 time_since_lit = 3 * length_at_time initial_length burn_rate2 time_since_lit →
  t = 11 * 60 + 20 :=
sorry 

end candles_lighting_time_l103_103618


namespace final_number_after_operations_l103_103553

theorem final_number_after_operations :
    let numbers := (List.range (2002)).map (λ n, 1 / (n + 1))
    let operation : ℚ → ℚ → ℚ := λ x y, x + y + x * y
    ∃ n : ℚ, by sorry  -- placeholder to represent n derived from 2000 operations
    ∃ final_number : ℚ, (List.foldl operation 0 numbers) = 2002 - 1 ∧ final_number = 2001 := sorry

end final_number_after_operations_l103_103553


namespace length_of_other_train_l103_103231

def speed_first_train := 60 -- km/hr
def speed_second_train := 40 -- km/hr
def length_first_train := 750 -- meters
def time_to_cross := 44.99640028797697 -- seconds

def km_hr_to_m_s (speed : ℕ) : ℝ := (speed * 1000) / 3600
def relative_speed := km_hr_to_m_s (speed_first_train + speed_second_train)
def total_distance := relative_speed * time_to_cross

theorem length_of_other_train : total_distance - length_first_train = 500 :=
by
  sorry

end length_of_other_train_l103_103231


namespace daily_wage_of_man_l103_103275

-- Define the wages for men and women
variables (M W : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 24 * M + 16 * W = 11600
def condition2 : Prop := 12 * M + 37 * W = 11600

-- The theorem we want to prove
theorem daily_wage_of_man (h1 : condition1 M W) (h2 : condition2 M W) : M = 350 :=
by
  sorry

end daily_wage_of_man_l103_103275


namespace tangent_circles_proportion_l103_103442

theorem tangent_circles_proportion
    (circle1 circle2 : Type)
    (A D B C : Point)
    (h1 : tangent_at circle1 A B)
    (h2 : tangent_at circle2 D C)
    (h3 : is_on_circle B circle1)
    (h4 : is_on_circle C circle2)
    (h5 : intersects circle1 circle2 A D) :
    AC / BD = (CD^2) / (AB^2) :=
by
  sorry

end tangent_circles_proportion_l103_103442


namespace area_of_sector_l103_103435

-- Define the conversion from degrees to radians
def degrees_to_radians (deg : ℝ) : ℝ := (deg * Real.pi) / 180

-- Define the sector area formula
def sector_area (r α : ℝ) : ℝ := (1 / 2) * α * r^2

-- Given conditions
def radius : ℝ := 6
def central_angle_degrees : ℝ := 15

-- Conversion of central angle to radians
def central_angle_radians : ℝ := degrees_to_radians central_angle_degrees

-- Proof statement
theorem area_of_sector :
  sector_area radius central_angle_radians = (3 * Real.pi) / 2 :=
by
  sorry

end area_of_sector_l103_103435


namespace problem_l103_103009

theorem problem (m n : ℕ) 
  (m_pos : 0 < m) 
  (n_pos : 0 < n) 
  (h1 : m + 8 < n) 
  (h2 : (m + (m + 3) + (m + 8) + n + (n + 3) + (2 * n - 1)) / 6 = n + 1) 
  (h3 : (m + 8 + n) / 2 = n + 1) : m + n = 16 :=
  sorry

end problem_l103_103009


namespace ff_of_1_l103_103431

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then 2 * x^2 + 1 else x + 5

theorem ff_of_1 : f (f 1) = 8 := by
  sorry

end ff_of_1_l103_103431


namespace max_sinx_sqrt3_cosx_l103_103210

theorem max_sinx_sqrt3_cosx : ∀ x : ℝ, sin x + sqrt 3 * cos x ≤ 2 := by sorry

end max_sinx_sqrt3_cosx_l103_103210


namespace downstream_speed_l103_103984

-- Definitions based on the conditions
def V_m : ℝ := 50 -- speed of the man in still water
def V_upstream : ℝ := 45 -- speed of the man when rowing upstream

-- The statement to prove
theorem downstream_speed : ∃ (V_s V_downstream : ℝ), V_upstream = V_m - V_s ∧ V_downstream = V_m + V_s ∧ V_downstream = 55 := 
by
  sorry

end downstream_speed_l103_103984


namespace knights_wins_33_l103_103808

def sharks_wins : ℕ := sorry
def falcons_wins : ℕ := sorry
def knights_wins : ℕ := sorry
def wolves_wins : ℕ := sorry
def dragons_wins : ℕ := 38 -- Dragons won the most games

-- Condition 1: The Sharks won more games than the Falcons.
axiom sharks_won_more_than_falcons : sharks_wins > falcons_wins

-- Condition 2: The Knights won more games than the Wolves, but fewer than the Dragons.
axiom knights_won_more_than_wolves : knights_wins > wolves_wins
axiom knights_won_less_than_dragons : knights_wins < dragons_wins

-- Condition 3: The Wolves won more than 22 games.
axiom wolves_won_more_than_22 : wolves_wins > 22

-- The possible wins are 24, 27, 33, 36, and 38 and the dragons win 38 (already accounted in dragons_wins)

-- Prove that the Knights won 33 games.
theorem knights_wins_33 : knights_wins = 33 :=
sorry -- proof goes here

end knights_wins_33_l103_103808


namespace monotonic_f_inequality_f_over_h_l103_103422

noncomputable def f (x : ℝ) : ℝ := 1 + (1 / x) + Real.log x + (Real.log x / x)

theorem monotonic_f :
  ∀ x : ℝ, x > 0 → ∃ I : Set ℝ, (I = Set.Ioo 0 x ∨ I = Set.Icc 0 x) ∧ (∀ y ∈ I, y > 0 → f y = f x) :=
by
  sorry

theorem inequality_f_over_h :
  ∀ x : ℝ, x > 1 → (f x) / (Real.exp 1 + 1) > (2 * Real.exp (x - 1)) / (x * Real.exp x + 1) :=
by
  sorry

end monotonic_f_inequality_f_over_h_l103_103422


namespace problem1_problem2_l103_103830

variables {a b c : ℝ} {A B C : ℝ}

-- Conditions
def condition1 : Prop := ∀ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 → sin A * sin A + sin A * sin B - 6 * sin B * sin B = 0
def condition2 : Prop := ∀ (a b : ℝ), cos C = (3 / 4)

-- Proof 1
theorem problem1 (h1 : condition1) : b / a = 2 := sorry

-- Proof 2
theorem problem2 (h1 : condition1) (h2 : condition2) : sin B = sqrt 14 / 8 := sorry

end problem1_problem2_l103_103830


namespace incorrect_games_less_than_three_fourths_l103_103813

/-- In a round-robin chess tournament, each participant plays against every other participant exactly once.
A win earns one point, a draw earns half a point, and a loss earns zero points.
We will call a game incorrect if the player who won the game ends up with fewer total points than the player who lost.

1. Prove that incorrect games make up less than 3/4 of the total number of games in the tournament.
2. Prove that in part (1), the number 3/4 cannot be replaced with a smaller number.
--/
theorem incorrect_games_less_than_three_fourths {n : ℕ} (h : n > 1) :
  ∃ m, (∃ (incorrect_games total_games : ℕ), m = incorrect_games ∧ total_games = (n * (n - 1)) / 2 
    ∧ (incorrect_games : ℚ) / total_games < 3 / 4) 
    ∧ (∀ m' : ℚ, m' ≥ 0 → m = incorrect_games ∧ (incorrect_games : ℚ) / total_games < m' → m' ≥ 3 / 4) :=
sorry

end incorrect_games_less_than_three_fourths_l103_103813


namespace intersection_M_N_eq_M_l103_103859

-- Definitions of M and N
def M : Set ℝ := { x : ℝ | x^2 - x < 0 }
def N : Set ℝ := { x : ℝ | abs x < 2 }

-- Proof statement
theorem intersection_M_N_eq_M : M ∩ N = M := 
  sorry

end intersection_M_N_eq_M_l103_103859


namespace probability_P_plus_S_mod_7_correct_l103_103934

noncomputable def probability_P_plus_S_mod_7 : ℚ :=
  let n := 60
  let total_ways := (n * (n - 1)) / 2
  let num_special_pairs := total_ways - ((52 * 51) / 2)
  num_special_pairs / total_ways

theorem probability_P_plus_S_mod_7_correct :
  probability_P_plus_S_mod_7 = 148 / 590 :=
by
  rw [probability_P_plus_S_mod_7]
  sorry

end probability_P_plus_S_mod_7_correct_l103_103934


namespace students_in_favor_ABC_l103_103326

variables (U A B C : Finset ℕ)

-- Given conditions
axiom total_students : U.card = 300
axiom students_in_favor_A : A.card = 210
axiom students_in_favor_B : B.card = 190
axiom students_in_favor_C : C.card = 160
axiom students_against_all : (U \ (A ∪ B ∪ C)).card = 40

-- Proof goal
theorem students_in_favor_ABC : (A ∩ B ∩ C).card = 80 :=
by {
  sorry
}

end students_in_favor_ABC_l103_103326


namespace mark_deposit_amount_l103_103535

-- Define the conditions
def bryans_deposit (M : ℝ) : ℝ := 5 * M - 40
def total_deposit (M : ℝ) : ℝ := M + bryans_deposit M

-- State the theorem
theorem mark_deposit_amount (M : ℝ) (h1: total_deposit M = 400) : M = 73.33 :=
by
  sorry

end mark_deposit_amount_l103_103535


namespace math_problem_l103_103908

variable (a : ℕ+ → ℝ) (b : ℕ+ → ℝ)

def condition1 : Prop := ∀ n : ℕ+, 1 / a (n + 1) = (1 / a n) + 1
def condition2 : Prop := ∀ n : ℕ+, b n = 1 / a n
def condition3 : Prop := ∑ n in Finset.range 9, b (n + 1) = 45
def question : Prop := b 4 * b 6 = 24

theorem math_problem : condition1 a ∧ condition2 a b ∧ condition3 b → question b :=
by
  sorry

end math_problem_l103_103908


namespace integral_simplification_l103_103364

noncomputable def integral_expression (x : ℝ) : ℝ :=
  ∫ (λ x, ( (1 + x^(4/5))^(3/4) / (x^(12/5)) )) dx

theorem integral_simplification : 
  ∫ (λ x, ( (1 + (x^4)^(1/5))^3 )^(1/4) / ( (x^2) * (x^2)^(1/5) )) dx = ∫ (λ x, ((1 + x^(4/5))^(3/4) * x^(-12/5))) dx :=
begin
  sorry
end

end integral_simplification_l103_103364


namespace equivalent_proof_problem_l103_103051

noncomputable def f : ℝ → ℝ := sorry

def condition1 (x : ℝ) : Prop :=
  f x + Real.sin (π / 6 * x) = f (-x) + Real.sin (π / 6 * x)

def condition2 : Prop :=
  f (Real.log 2 / Real.log (Real.sqrt 2)) = Real.sqrt 3

theorem equivalent_proof_problem : condition1 →
                                   condition2 →
                                   f (Real.log (1 / 4) / Real.log 2) = 2 * Real.sqrt 3 :=
by
  intro h1 h2
  sorry

end equivalent_proof_problem_l103_103051


namespace total_blankets_l103_103356

def initial_blankets := 156
def polka_dot_fraction := 3 / 7
def striped_fraction := 2 / 7
def extra_polka_dot := 9
def extra_striped := 5

theorem total_blankets : 
  let initial_polka_dot := (polka_dot_fraction * initial_blankets).to_nat
  let initial_striped := (striped_fraction * initial_blankets).to_nat
  let total_polka_dot := initial_polka_dot + extra_polka_dot
  let total_striped := initial_striped + extra_striped
  total_polka_dot + total_striped = 124 := 
by
  sorry

end total_blankets_l103_103356


namespace problem_I_problem_II_l103_103522

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := sin x * cos x - cos (x + π / 4) ^ 2

-- Statement of the mathematical proof problems
theorem problem_I : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = π := sorry

theorem problem_II 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : a = 1)
  (h3 : b + c = 2)
  (h4 : f (A / 2) = (sqrt 3 - 1) / 2) :
  ∃ S : ℝ, S = (sqrt 3) / 4 := sorry

end problem_I_problem_II_l103_103522


namespace total_volume_of_four_boxes_l103_103257

-- Define the edge length of the cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_one_cube := edge_length ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 4

-- The total volume of the four cubes
def total_volume := number_of_cubes * volume_of_one_cube

-- Statement to prove that the total volume equals 500 cubic feet
theorem total_volume_of_four_boxes :
  total_volume = 500 :=
sorry

end total_volume_of_four_boxes_l103_103257


namespace find_m_l103_103012

noncomputable def given_hyperbola (x y : ℝ) (m : ℝ) : Prop :=
    x^2 / m - y^2 / 3 = 1

noncomputable def hyperbola_eccentricity (m : ℝ) (e : ℝ) : Prop :=
    e = Real.sqrt (1 + 3 / m)

theorem find_m (m : ℝ) (h1 : given_hyperbola 1 1 m) (h2 : hyperbola_eccentricity m 2) : m = 1 :=
by
  sorry

end find_m_l103_103012


namespace annual_interest_rate_compound_monthly_l103_103991

theorem annual_interest_rate_compound_monthly :
  ∃ r : ℝ, (∀ (P : ℝ), 
    P * (1 + r / 12) ^ (12 * 2) = 2420 ∧ 
    P * (1 + r / 12) ^ (12 * 3) = 3146) → r ≈ 0.2676 :=
by 
  sorry

end annual_interest_rate_compound_monthly_l103_103991


namespace sector_area_is_correct_l103_103197

-- Define parameters and constants
def r : ℝ := 2
def α : ℝ := π / 3

-- Define the expected area for the given sector
def expected_area : ℝ := 2 * π / 3

-- The theorem to prove corresponds to the problem statement
theorem sector_area_is_correct : (1/2) * α * r^2 = expected_area := by
  sorry

end sector_area_is_correct_l103_103197


namespace tourist_tax_calculation_l103_103307

def goods := {total_value : ℝ, electronics_value : ℝ, luxury_value : ℝ, educational_value : ℝ}

noncomputable def calculate_tax (g : goods) (has_student_id : Bool) : ℝ :=
  let excess_value := g.total_value - 600
  let base_tax1 := if excess_value > 400 then 0.12 * 400 else 0.12 * excess_value
  let base_tax2 := if excess_value > 900 then 0.18 * 500 else if excess_value > 400 then 0.18 * (excess_value - 400) else 0
  let base_tax3 := if excess_value > 1400 then 0.25 * (excess_value - 900) else 0
  let electronics_tax := 0.05 * g.electronics_value
  let luxury_tax := 0.10 * g.luxury_value
  let total_tax_before_deduction := base_tax1 + base_tax2 + base_tax3 + electronics_tax + luxury_tax
  let discount := if has_student_id then 0.05 * total_tax_before_deduction else 0
  total_tax_before_deduction - discount

def tourist_goods : goods :=
  { total_value := 2100, electronics_value := 900, luxury_value := 820, educational_value := 380 }

theorem tourist_tax_calculation : calculate_tax tourist_goods True = 304 := sorry

end tourist_tax_calculation_l103_103307


namespace exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l103_103127

-- Define the parabola as a function
def parabola (x : ℝ) : ℝ := x^2

-- N-gon properties
def is_convex_ngon (N : ℕ) (vertices : List (ℝ × ℝ)) : Prop :=
  -- Placeholder for checking properties; actual implementation would validate convexity and equilateral nature.
  sorry 

-- Statement for 2011-gon
theorem exists_convex_2011_gon_on_parabola :
  ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2011 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

-- Statement for 2012-gon
theorem not_exists_convex_2012_gon_on_parabola :
  ¬ ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2012 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

end exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l103_103127


namespace trapezoid_intersects_diagonal_l103_103664

variable {A B C D M N K L : Type*}

-- Define the geometrical constructs and conditions:
def quadrilateral (ABCD : Prop) : Prop := 
  ∃ (A B C D : Point), ABCD = (A, B, C, D)

def trapezoid (MNKL : Prop) : Prop := 
  ∃ (M N K L : Point), MNKL = (M, N, K, L) ∧ parallel M N K L

def parallel (P Q R S : Point) : Prop :=
  -- define the conditions when two line segments PQ and RS are parallel
  sorry

def are_intersect (P Q R S : Line) : Prop :=
  -- define the conditions when two lines intersect at a point
  sorry

def is_diagonal (P Q R S : Point) : Prop :=
  -- define a diagonal in a quadrilateral
  sorry

-- State the problem:
theorem trapezoid_intersects_diagonal (ABCD MNKL : Prop) (MN || LK : Prop) (MN || AC : Prop) : 
  quadrilateral ABCD → 
  trapezoid MNKL →
  parallel MN LK →
  parallel MN AC →
  ∃ (BD_intersection : Point), non_parallel MN LK ∧ point_on_diagonal BD_intersection BD :=
by
  intros ABCD_def MNKL_def MN_par_LK MN_par_AC
  -- We'd normally proceed with the proof
  sorry

end trapezoid_intersects_diagonal_l103_103664


namespace arith_prog_sum_eq_l103_103870

variable (a d : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n / 2) * (2 * a 1 + (n - 1) * d 1)

theorem arith_prog_sum_eq (n : ℕ) : 
  S a d (n + 3) - 3 * S a d (n + 2) + 3 * S a d (n + 1) - S a d n = 0 := 
sorry

end arith_prog_sum_eq_l103_103870


namespace geometric_sequence_sixth_term_l103_103891

theorem geometric_sequence_sixth_term (a : ℝ) (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^(7) = 2) :
  a * r^(5) = 16 :=
by
  sorry

end geometric_sequence_sixth_term_l103_103891


namespace exists_two_lines_cover_l103_103007

def smallest_n_two_lines (A : Finset (ℝ × ℝ)) (n : ℕ) : Prop :=
  ∀ B : Finset (ℝ × ℝ), B ⊆ A → B.card = n → ∃ ℓ₁ ℓ₂ : set (ℝ × ℝ), ∀ p ∈ B, p ∈ ℓ₁ ∪ ℓ₂

theorem exists_two_lines_cover (A : Finset (ℝ × ℝ)) (h : smallest_n_two_lines A 6) :
  ∃ ℓ₁ ℓ₂ : set (ℝ × ℝ), ∀ p ∈ A, p ∈ ℓ₁ ∪ ℓ₂ :=
  sorry

end exists_two_lines_cover_l103_103007


namespace count_divisible_factorial_sum_squares_l103_103377

/-- Prove that the number of positive integers n less than or equal to 15 for which n! 
is evenly divisible by the sum of the squares of the first n integers is 5. -/
theorem count_divisible_factorial_sum_squares :
  {n : ℕ | n > 0 ∧ n ≤ 15 ∧ (n! % (n * (n + 1) * (2 * n + 1) / 6) = 0)}.card = 5 :=
sorry  -- proof of the theorem

end count_divisible_factorial_sum_squares_l103_103377


namespace system_of_linear_eq_l103_103632

theorem system_of_linear_eq :
  ∃ (x y : ℝ), x + y = 5 ∧ y = 2 :=
sorry

end system_of_linear_eq_l103_103632


namespace product_of_roots_l103_103854

-- Define the quadratic function in terms of a, b, c
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the conditions
variables (a b c y : ℝ)

-- Given conditions from the problem
def condition_1 := ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2
def condition_2 := quadratic a b c y = 0
def condition_3 := quadratic a b c (4 * y) = 0

-- The statement to be proved
theorem product_of_roots (a b c y : ℝ) 
  (h1: ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2)
  (h2: quadratic a b c y = 0) 
  (h3: quadratic a b c (4 * y) = 0) :
  ∃ x1 x2, (quadratic a b c x = 0 → (x1 = y ∧ x2 = 4 * y) ∨ (x1 = 4 * y ∧ x2 = y)) ∧ x1 * x2 = 4 * y^2 :=
by
  sorry

end product_of_roots_l103_103854


namespace base_8_subtraction_correct_l103_103002

noncomputable def base_8_subtraction : ℕ :=
let a := 0o5273 in -- Octal 5273
let b := 0o3614 in -- Octal 3614
a - b

theorem base_8_subtraction_correct : base_8_subtraction = 0o1457 := -- Octal 1457
by 
  sorry

end base_8_subtraction_correct_l103_103002


namespace prob_P_plus_S_one_less_multiple_of_7_l103_103931

theorem prob_P_plus_S_one_less_multiple_of_7 :
  let a b : ℕ := λ x y, x ∈ Finset.range (60+1) ∧ y ∈ Finset.range (60+1) ∧ x ≠ y ∧ 1 ≤ x ∧ x ≤ 60 ∧ 1 ≤ y ∧ y ≤ 60,
      P : ℕ := ∀ a b, a * b,
      S : ℕ := ∀ a b, a + b,
      m : ℕ := (P + S) + 1,
      all_pairs : ℕ := Nat.choose 60 2,
      valid_pairs : ℕ := 444,
      probability : ℚ := valid_pairs / all_pairs
  in probability = 148 / 590 := sorry

end prob_P_plus_S_one_less_multiple_of_7_l103_103931


namespace color_nat_two_colors_no_sum_power_of_two_l103_103507

theorem color_nat_two_colors_no_sum_power_of_two :
  ∃ (f : ℕ → ℕ), (∀ a b : ℕ, a ≠ b → f a = f b → ∃ c : ℕ, c > 0 ∧ c ≠ 1 ∧ c ≠ 2 ∧ (a + b ≠ 2 ^ c)) :=
sorry

end color_nat_two_colors_no_sum_power_of_two_l103_103507


namespace angle_EHF_90_degrees_l103_103134

theorem angle_EHF_90_degrees 
  (A B C P E F H: Point)  -- Points
  (circle: Circle)        -- Circle with BC as its diameter
  (H_C': projection P B)  -- C' is the projection of P onto AB
  (H_B': projection P C)  -- B' is the projection of P onto AC
  (orthocenter: Orthocenter) -- H is the orthocenter of triangle AB'C'
  (on_circle: circle.contains P)  -- P is on the circle
  (cut1: circle.intersect AB = F)  -- Circle intersects AB at F
  (cut2: circle.intersect AC = E)  -- Circle intersects AC at E
  : angle E H F = 90 := 
sorry

end angle_EHF_90_degrees_l103_103134


namespace a_n_formula_l103_103013

open Nat

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else n * (n + 1) / 2

theorem a_n_formula (n : ℕ) (h : n > 0) 
  (S_n : ℕ → ℕ)
  (hS : ∀ n, S_n n = (n + 2) / 3 * a_n n) 
  : a_n n = n * (n + 1) / 2 := sorry

end a_n_formula_l103_103013


namespace area_of_triangle_DEF_l103_103822

variable (DE DF : ℝ) (angle_D : ℝ)

-- Given conditions
def given_conditions : Prop :=
  DE = 30 ∧ DF = 24 ∧ angle_D = 90

-- The proof problem in Lean 4
theorem area_of_triangle_DEF : given_conditions DE DF angle_D → 
  let area := (1 / 2) * DE * DF in
  area = 360 :=
by
  intro h
  let area := (1 / 2) * DE * DF
  sorry

end area_of_triangle_DEF_l103_103822


namespace determinant_is_zero_l103_103337

noncomputable def matrix3x3 (a b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  λ i j, match (i, j) with
           | (0, 0) => 1
           | (0, 1) => Real.sin (a + b)
           | (0, 2) => Real.sin a
           | (1, 0) => Real.sin (a + b)
           | (1, 1) => 1
           | (1, 2) => Real.sin b
           | (2, 0) => Real.sin a
           | (2, 1) => Real.sin b
           | (2, 2) => 1
           | _      => 0

theorem determinant_is_zero (a b : ℝ) : Matrix.det (matrix3x3 a b) = 0 :=
  sorry

end determinant_is_zero_l103_103337


namespace insufficient_queries_l103_103922

theorem insufficient_queries (weights : Fin 5 → ℕ) 
(queries : Fin 9 → (Fin 5 × Fin 5 × Fin 5)) 
(responses : Fin 9 → Bool) :
∃ (perms : List (List (Fin 5 → ℕ))), 
  length perms > 1 ∧ 
  (∀ perm ∈ perms, ∀ i : Fin 9, 
     (responses i = true ↔ (perm (queries i).1 < perm (queries i).2 ∧ perm (queries i).2 < perm (queries i).3)) ∧
     (responses i = false ↔ ¬ (perm (queries i).1 < perm (queries i).2 ∧ perm (queries i).2 < perm (queries i).3)) ->
    ∃ perm1 perm2 ∈ perms, perm1 ≠ perm2) :=
sorry

end insufficient_queries_l103_103922


namespace sally_spent_eur_l103_103574

-- Define the given conditions
def coupon_value : ℝ := 3
def peaches_total_usd : ℝ := 12.32
def cherries_original_usd : ℝ := 11.54
def discount_rate : ℝ := 0.1
def conversion_rate : ℝ := 0.85

-- Define the intermediate calculations
def cherries_discount_usd : ℝ := cherries_original_usd * discount_rate
def cherries_final_usd : ℝ := cherries_original_usd - cherries_discount_usd
def total_usd : ℝ := peaches_total_usd + cherries_final_usd
def total_eur : ℝ := total_usd * conversion_rate

-- The final statement to be proven
theorem sally_spent_eur : total_eur = 19.30 := by
  sorry

end sally_spent_eur_l103_103574


namespace faster_walking_speed_l103_103652

theorem faster_walking_speed :
  ∀ (v : ℝ) (t : ℝ), 5 * (t - 6) = 630 → v * (t - 30) = 630 → v = 6.176470588 {
  intros,
  have eq1 : 5 * (t - 6) = 630 := by assumption,
  have eq2 : v * (t - 30) = 630 := by assumption,
  sorry
}

end faster_walking_speed_l103_103652


namespace intersection_M_N_l103_103438

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {x | x ≥ 3}

theorem intersection_M_N : M ∩ N = {3, 4} := 
by
  sorry

end intersection_M_N_l103_103438


namespace eventually_stable_l103_103235

theorem eventually_stable {n : ℕ} (a : Fin n → ℤ) :
  ∃ m : ℕ, ∀ t ≥ m, ∀ i j : Fin n, i ≠ j →
  let b := (a t).update i (Int.gcd (a t).nth i (a t).nth j) in
  let c := b.update j (Int.lcm (a t).nth i (a t).nth j) in
  c = a t :=
sorry

end eventually_stable_l103_103235


namespace table_permutation_count_l103_103037

noncomputable def unique_tables_count : Nat :=
  factorial 7 * factorial 6

theorem table_permutation_count :
  unique_tables_count = factorial 7 * factorial 6 := by
  -- sorry denotes the placeholder for the actual proof
  sorry

end table_permutation_count_l103_103037


namespace non_zero_area_triangles_l103_103783

/--
There are 17 dots in total with 9 dots aligned horizontally and 9 dots aligned vertically.
We are to prove that the number of triangles with non-zero area, formed by choosing 3 vertices out of these 17 dots, is equal to 512.
-/

theorem non_zero_area_triangles (horiz_vert_eq_9 : 9 + 9 = 17) : (number_of_tris_with_non_zero_area Dots_config) = 512 :=
sorry

end non_zero_area_triangles_l103_103783


namespace probability_P_plus_S_is_one_less_than_multiple_of_seven_l103_103937

theorem probability_P_plus_S_is_one_less_than_multiple_of_seven :
  ∀ (a b : ℕ), a ∈ finset.range(1, 61) → b ∈ finset.range(1, 61) → a ≠ b →
  let S := a + b in
  let P := a * b in
  (nat.gcd ((P + S + 1), 7) = 1) →
  (finset.filter (λ (a b : ℕ), (a+1) ∣ 7 ∨ (b+1) ∣ 7) (finset.range(1, 61)).product (finset.range(1, 61)).card) / 1770 = 74 / 295 :=
begin
  sorry
end

end probability_P_plus_S_is_one_less_than_multiple_of_seven_l103_103937


namespace sum_of_x_for_sqrt_eq_eight_l103_103625

theorem sum_of_x_for_sqrt_eq_eight:
  (∑ x in ({x : ℝ | Real.sqrt ((x - 2)^2) = 8}).to_finset, x) = 4 := 
sorry

end sum_of_x_for_sqrt_eq_eight_l103_103625


namespace interval_difference_l103_103367

noncomputable theory

/-- Given the intervals of the inequality and their lengths, prove the difference is equal to 156 --/
theorem interval_difference (c d : ℝ) :
  (∀ x : ℝ, c ≤ x^2 + 2 * x + 1 → x^2 + 2 * x + 1 ≤ d) →
  (sqrt d - sqrt c = 12) →
  (d - c = 156) :=
by
  sorry

end interval_difference_l103_103367


namespace train_speed_conversion_l103_103308

-- Define the speed of the train in meters per second.
def speed_mps : ℝ := 37.503

-- Definition of the conversion factor between m/s and km/h.
def conversion_factor : ℝ := 3.6

-- Define the expected speed of the train in kilometers per hour.
def expected_speed_kmph : ℝ := 135.0108

-- Prove that the speed in km/h is the expected value.
theorem train_speed_conversion :
  (speed_mps * conversion_factor = expected_speed_kmph) :=
by
  sorry

end train_speed_conversion_l103_103308


namespace inclination_angle_l103_103901

theorem inclination_angle (x y : ℝ) : 
  let line := (√3) * x - y + 1 = 0 in
  true :=
sorry

end inclination_angle_l103_103901


namespace triangle_A_pi_over_6_b_plus_c_sqrt_2_l103_103076

variables {A B C : ℝ}
variables {a b c : ℝ}

-- Conditions from the given problem
def condition1 := a * Real.cos B + (√3) * b * Real.sin A = c
def condition2 := a = 1
def condition3 := b * c = 2 - √3

theorem triangle_A_pi_over_6 (h1 : condition1) : A = π / 6 :=
sorry

theorem b_plus_c_sqrt_2 (h1 : condition1) (h2: condition2) (h3 : condition3) : b + c = √2 :=
sorry

end triangle_A_pi_over_6_b_plus_c_sqrt_2_l103_103076


namespace solve_inverse_simplifies_l103_103016

-- Define the context under which the problems reside
def geometric_problem := 
  ∃ (triangle1 triangle2 : Triangle), 
    is_equilateral triangle1 ∧
    is_equilateral triangle2 ∧
    similar triangle1 triangle2

def rope_problem := 
  ∃ (rope : Rope), 
    start_untangle(rope) -> able_to_tie_knot(rope)

-- Define the main theorem about solving inverse problems
theorem solve_inverse_simplifies 
  (gp : geometric_problem)
  (rp : rope_problem) :
  (solve_inverse(gp) → solve_direct(gp)) ∧
  (solve_inverse(rp) → solve_direct(rp)) :=
sorry

end solve_inverse_simplifies_l103_103016


namespace sin_double_angle_l103_103045

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 4) :
  Real.sin (2 * α) = -3 / 4 :=
sorry

end sin_double_angle_l103_103045


namespace neg_rational_is_rational_l103_103792

theorem neg_rational_is_rational (m : ℚ) : -m ∈ ℚ := 
by sorry

end neg_rational_is_rational_l103_103792


namespace spiral_length_is_correct_l103_103294

-- Definitions based on conditions
def circumference : ℝ := 18
def height : ℝ := 8
def turns : ℝ := 2

-- Definition for the problem using given conditions
def length_of_spiral : ℝ :=
  real.sqrt ((circumference * turns)^2 + (height * turns)^2)

-- The theorem statement
theorem spiral_length_is_correct :
  length_of_spiral = real.sqrt 1552 :=
by 
  sorry

end spiral_length_is_correct_l103_103294


namespace sum_of_sums_l103_103142

noncomputable def set_of_non_empty_subsets (n : ℕ) : Set (Set ℕ) :=
  {s | ∃ (t : Set ℕ), t ⊆ (Finset.range (n+1)).toSet ∧ t ≠ ∅ ∧ s = t}

noncomputable def delta (a : Set ℕ) : ℕ :=
  a.sum id

theorem sum_of_sums (n : ℕ) :
  (∑ a in set_of_non_empty_subsets n, delta a) = 2^(n-2) * n * (n+1) := by
  sorry

end sum_of_sums_l103_103142


namespace find_x_l103_103456

theorem find_x (x : ℝ) (h : 0.5 * x = 0.05 * 500 - 20) : x = 10 :=
by
  sorry

end find_x_l103_103456


namespace midpoint_intersection_l103_103155

open EuclideanGeometry Real
namespace TriangleProblem

def Orthocenter {A B C H : Point} : Prop :=
  is_orthocenter H A B C

def Circumcircle {A B C : Point} (H : Point) : Circle :=
  circumcircle H A B

def DiameterCircle {A C : Point} : Circle :=
  circumcircle A C midpoint

theorem midpoint_intersection 
  {A B C H K : Point} 
  (h_orth : Orthocenter A B C H)
  (int_circles : ∃ K, K ∈ Circumcircle A B H ∧ K ∈ DiameterCircle A C) :
  midpoint_line_intersects K C B H :=
begin
  sorry
end

end TriangleProblem

end midpoint_intersection_l103_103155


namespace sqrt_x_minus_2_meaningful_in_reals_l103_103468

theorem sqrt_x_minus_2_meaningful_in_reals (x : ℝ) : (∃ (y : ℝ), y * y = x - 2) → x ≥ 2 :=
by
  sorry

end sqrt_x_minus_2_meaningful_in_reals_l103_103468


namespace total_number_of_balls_l103_103286

theorem total_number_of_balls 
(b : ℕ) (P_blue : ℚ) (h1 : b = 8) (h2 : P_blue = 1/3) : 
  ∃ g : ℕ, b + g = 24 := by
  sorry

end total_number_of_balls_l103_103286


namespace leftmost_three_nonzero_digits_of_ring_arrangements_l103_103041

theorem leftmost_three_nonzero_digits_of_ring_arrangements :
  let num_arrangements := (Nat.choose 10 6) * 6.fact * (Nat.choose 10 4)
  in num_arrangements = 31752000 → (317 : Nat) :=
by
  sorry

end leftmost_three_nonzero_digits_of_ring_arrangements_l103_103041


namespace ratio_of_a_to_c_l103_103216

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 :=
by
  sorry

end ratio_of_a_to_c_l103_103216


namespace password_probability_l103_103329

def prob_odd : ℚ := 5 / 10
def prob_A : ℚ := 1 / 26
def prob_even : ℚ := 5 / 10

def desired_probability : ℚ := prob_odd * prob_A * prob_even

theorem password_probability : desired_probability = 1 / 104 :=
by
  unfold desired_probability prob_odd prob_A prob_even
  norm_num
  sorry

end password_probability_l103_103329


namespace range_of_a_l103_103466

def solution_set_non_empty (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - 3| + |x - 4| < a

theorem range_of_a (a : ℝ) : solution_set_non_empty a ↔ a > 1 := sorry

end range_of_a_l103_103466


namespace total_volume_of_four_cubes_is_500_l103_103251

-- Definition of the edge length of each cube
def edge_length : ℝ := 5

-- Definition of the volume of one cube
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the number of cubes
def number_of_cubes : ℕ := 4

-- Definition of the total volume
def total_volume (n : ℕ) (v : ℝ) : ℝ := n * v

-- The proposition we want to prove
theorem total_volume_of_four_cubes_is_500 :
  total_volume number_of_cubes (volume_of_cube edge_length) = 500 :=
by
  sorry

end total_volume_of_four_cubes_is_500_l103_103251


namespace EF_distance_l103_103144

-- Defining the setup of the problem.
structure trapezoid (A B C D E F : Type) :=
  (is_isosceles : true)
  (AD_parallel_BC : true)
  (angles_at_AD_pi_over_4 : ∀ {a b}, b = A ∨ b = D → a = (π / 4))
  (diagonals_length : ∀ {a}, a = 20 * sqrt 2)
  (dist_EA_20 : E = A → 20)
  (dist_ED_40 : E = D → 40)
  (F_foot_of_altitude_B_to_AD : F = (λ x, x))

-- Defining the problem facts.
abbreviation isosceles_trapezoid (A B C D E F : Type) :=
    ∃ t : trapezoid A B C D E F, t.is_isosceles

noncomputable def distance_EQF (A B C D E F : Type) [isosceles_trapezoid A B C D E F] : ℝ :=
    20 * sqrt 1

-- The theorem we need to prove
theorem EF_distance (A B C D E F : Type)
  [isosceles_trapezoid A B C D E F] :
  distance_EQF A B C D E F = 20 :=
sorry

end EF_distance_l103_103144


namespace football_tournament_max_points_l103_103109

theorem football_tournament_max_points:
  ∃ N : ℕ, let teams := 15,
               matches := teams.choose 2,
               max_points := 105*3 in
    (∀ pts. ∃ six_teams : Finset ℕ, (six_teams.card = 6 ∧ ∀ x ∈ six_teams, pts x ≥ N) → 6 * N ≤ max_points) ∧
    (∀ pts. ∃ six_teams : Finset ℕ, (six_teams.card = 6 ∧ ∀ x ∈ six_teams, pts x ≥ 35) → 6 * 35 > max_points) :=
sorry

end football_tournament_max_points_l103_103109


namespace student_distribution_l103_103704

theorem student_distribution :
  ∃ (distributions : ℕ), distributions = 24 :=
begin
  have h1 : 4 = 4, by norm_num,
  have h2 : 3 = 3, by norm_num,
  -- Define students and classes
  let students := ["A", "B", "C", "D"],
  let classes := ["ClassA", "ClassB", "ClassC"],
  
  -- Main conditions
  have h_class_requirements : ∀ s, s ∈ students → ∃ c, c ∈ classes ∧ c ≠ "ClassA",
  {
    intros s hs,
    cases hs,
    use "ClassB", split; norm_num,
    use "ClassC", split; norm_num,
    use "ClassB", split; norm_num,
    use "ClassC", split; norm_num,
  },
  
  -- Checking the distribution count equals 24 as given by the original solution
  use 24,
  sorry
end

end student_distribution_l103_103704


namespace distribution_count_l103_103348

def num_distributions (novels poetry students : ℕ) : ℕ :=
  -- This is where the formula for counting would go, but we'll just define it as sorry for now
  sorry

theorem distribution_count : num_distributions 3 2 4 = 28 :=
by
  sorry

end distribution_count_l103_103348


namespace s2_sub_c2_range_l103_103523

variable {x y : ℝ}

def r := Real.sqrt (x^2 + y^2)
def s := y / r
def c := x / r

theorem s2_sub_c2_range : -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 := by
  sorry

end s2_sub_c2_range_l103_103523


namespace lcm_consecutive_impossible_l103_103032

def lcm (a b : Nat) : Nat := 
  (a * b) / (Nat.gcd a b)

theorem lcm_consecutive_impossible (n : Nat) (a : Fin n → Nat) 
  (h : n = 10^1000) :
  ¬∃ (b : Fin n → Nat), (∀ i : Fin n, b i = lcm (a i) (a (i + 1))) ∧ 
  (∀ i : Fin (n - 1), b i + 1 = b (i + 1)) :=
by
  sorry

end lcm_consecutive_impossible_l103_103032


namespace conjugate_of_z_l103_103024

-- Define the complex number z and its properties
def z : ℂ := 2 + complex.i

-- State the theorem that proves the conjugate of z is 2 - i
theorem conjugate_of_z : complex.conj z = 2 - complex.i := 
by
  sorry

end conjugate_of_z_l103_103024


namespace shortest_distance_between_PQ_l103_103392

noncomputable def regular_tetrahedron (S A B C D : Point) : Prop :=
  (TetHeight S A B C D = 2) ∧ (SideLength A B = sqrt(2))

def points_on_segments (P Q B D S C : Point) : Prop :=
  OnSegment P B D ∧ OnSegment Q S C

def shortest_distance (P Q : Point) : ℝ := 
  -- Placeholder for the actual distance computation
  sorry

theorem shortest_distance_between_PQ {S A B C D P Q : Point} :
  regular_tetrahedron S A B C D →
  points_on_segments P Q B D S C →
  shortest_distance P Q = 2 * sqrt(5) / 5 :=
by 
  sorry

end shortest_distance_between_PQ_l103_103392


namespace green_and_yellow_peaches_total_is_correct_l103_103283

-- Define the number of red, yellow, and green peaches
def red_peaches : ℕ := 5
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6

-- Definition of the total number of green and yellow peaches
def total_green_and_yellow_peaches : ℕ := green_peaches + yellow_peaches

-- Theorem stating that the total number of green and yellow peaches is 20
theorem green_and_yellow_peaches_total_is_correct : total_green_and_yellow_peaches = 20 :=
by 
  sorry

end green_and_yellow_peaches_total_is_correct_l103_103283


namespace greatest_b_l103_103003

theorem greatest_b (b : ℝ) : (-b^2 + 9 * b - 14 ≥ 0) → b ≤ 7 := sorry

end greatest_b_l103_103003


namespace grasshopper_jump_l103_103900

theorem grasshopper_jump (frog_jump grasshopper_jump : ℕ)
  (h1 : frog_jump = grasshopper_jump + 17)
  (h2 : frog_jump = 53) :
  grasshopper_jump = 36 :=
by
  sorry

end grasshopper_jump_l103_103900


namespace distinct_numbers_in_union_set_l103_103141

def first_seq_term (k : ℕ) : ℤ := 5 * ↑k - 3
def second_seq_term (m : ℕ) : ℤ := 9 * ↑m - 3

def first_seq_set : Finset ℤ := ((Finset.range 1003).image first_seq_term)
def second_seq_set : Finset ℤ := ((Finset.range 1003).image second_seq_term)

def union_set : Finset ℤ := first_seq_set ∪ second_seq_set

theorem distinct_numbers_in_union_set : union_set.card = 1895 := by
  sorry

end distinct_numbers_in_union_set_l103_103141


namespace range_of_a_l103_103464

theorem range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, (4 * x - 3 * y - 2 = 0) → (x^2 + y^2 - 2 * a * x + 4 * y + a^2 - 12 = 0) → x ≠ y) :
  -6 < a ∧ a < 4 :=
by
  sorry

end range_of_a_l103_103464


namespace time_to_empty_pool_l103_103665

-- Definitions based on conditions
def length : ℝ := 30
def width : ℝ := 40
def depth : ℝ := 2
def pump_rate : ℝ := 10
def cubic_foot_to_gallons : ℝ := 7.5

-- Assertion to be proved
theorem time_to_empty_pool : 
  let volume_cubic_feet := length * width * depth;
      total_gallons := volume_cubic_feet * cubic_foot_to_gallons;
      total_pump_rate := 4 * pump_rate in
  (total_gallons / total_pump_rate) = 450 :=
by
  sorry

end time_to_empty_pool_l103_103665


namespace proportion_fourth_number_l103_103453

theorem proportion_fourth_number (x y : ℝ) (h_x : x = 0.6) (h_prop : 0.75 / x = 10 / y) : y = 8 :=
by
  sorry

end proportion_fourth_number_l103_103453


namespace sum_of_coordinates_of_four_points_l103_103924

-- Definitions for conditions
def is_point_on_line_3_units (p : ℝ × ℝ) (y_line : ℝ) (d : ℝ) : Prop :=
  abs(p.snd - y_line) = d

def is_point_distance_from_another (p1 p2 : ℝ × ℝ) (d : ℝ) : Prop :=
  dist p1 p2 = d

noncomputable def sum_of_coordinates (points : List (ℝ × ℝ)) : ℝ :=
  points.sum (λ p, p.fst + p.snd)

-- Given conditions
def points : List (ℝ × ℝ) :=
  [(5 + Real.sqrt 91, 7), (5 - Real.sqrt 91, 7), (5 + Real.sqrt 91, 13), (5 - Real.sqrt 91, 13)]

theorem sum_of_coordinates_of_four_points : 
  sum_of_coordinates points = 60 :=
by
  sorry

end sum_of_coordinates_of_four_points_l103_103924


namespace crucian_vs_bream_l103_103646

-- Define the weights of crucian, bream, and perch
variables (C B P : ℝ)

-- Given conditions
axiom h1 : 6 * C > 10 * B
axiom h2 : 6 * C < 5 * P
axiom h3 : 10 * C > 8 * P

-- Statement to prove
theorem crucian_vs_bream (C B P : ℝ) (h1 : 6 * C > 10 * B) (h2 : 6 * C < 5 * P) (h3 : 10 * C > 8 * P) : 
  2 * C > 3 * B :=
sorry

end crucian_vs_bream_l103_103646


namespace sin_3theta_over_sin_theta_l103_103384

theorem sin_3theta_over_sin_theta (θ : ℝ) (h : Real.tan θ = Real.sqrt 2) : 
  Real.sin (3 * θ) / Real.sin θ = 1 / 3 :=
by
  sorry

end sin_3theta_over_sin_theta_l103_103384


namespace usamo_2003_q3_l103_103156

open Real

theorem usamo_2003_q3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2)
  + (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2)
  + (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2) ) ≤ 8 := 
sorry

end usamo_2003_q3_l103_103156


namespace Vann_total_teeth_cleaned_l103_103622

theorem Vann_total_teeth_cleaned :
  let dogs := 7
  let cats := 12
  let pigs := 9
  let horses := 4
  let rabbits := 15
  let dogs_teeth := 42
  let cats_teeth := 30
  let pigs_teeth := 44
  let horses_teeth := 40
  let rabbits_teeth := 28
  (dogs * dogs_teeth) + (cats * cats_teeth) + (pigs * pigs_teeth) + (horses * horses_teeth) + (rabbits * rabbits_teeth) = 1630 :=
by
  sorry

end Vann_total_teeth_cleaned_l103_103622


namespace find_f_5_l103_103095

-- Definitions of the function f and the given conditions
def f (x y : ℝ) : ℝ := 2 * x^2 + y
-- Given condition f(2) = 100
def condition1 : Prop := f 2 y = 100

-- The goal is to prove f(5) = 142 given the condition1
theorem find_f_5 (y : ℝ) (h : condition1) : f 5 y = 142 :=
by
  sorry

end find_f_5_l103_103095


namespace no_power_of_2_permutation_l103_103349

theorem no_power_of_2_permutation :
  ∀ r s : ℕ, (r > s) →
  (∀ (d : ℕ), digit_of_power_of_2 r d ≠ 0) →
  (∀ (d : ℕ), digit_of_power_of_2 s d ≠ 0) →
  (permuted_digits (power_of_2 r) (power_of_2 s)) →
  (¬∃ r s : ℕ, (power_of_2 r = reorder_power_of_2 s ∨ reorder_power_of_2 r = power_of_2 s)) :=
by
  sorry

def power_of_2 (n: ℕ): ℕ := 2^n

def reorder_power_of_2 (n: ℕ): ℕ := -- function that reorders digits of power_of_2
  sorry

def digit_of_power_of_2 (n d: ℕ): ℕ := -- function that extracts the d-th digit of 2^n
  sorry

def permuted_digits (a b: ℕ): Prop := -- function that checks if digits of a are permutations of digits of b
  sorry

end no_power_of_2_permutation_l103_103349


namespace max_pawns_adj_l103_103243

/-- Define the board size and the adjacency conditions -/
def board_size : ℕ := 12

-- Function to determine adjacency by sides or corners
def adjacent (x1 y1 x2 y2 : ℕ) : Prop :=
  (|x1 - x2| ≤ 1 ∧ |y1 - y2| ≤ 1) ∧ (x1 ≠ x2 ∨ y1 ≠ y2)

/-- Define the maximum number of pawns condition -/
def max_pawns (n : ℕ) : ℕ :=
  if n = board_size then 36 else 0

/-- The theorem we want to prove -/
theorem max_pawns_adj :
  ∃ p : Π (x y : ℕ), x < board_size → y < board_size → Prop,
    (∀ x y x' y', x < board_size → y < board_size → x' < board_size → y' < board_size →
    p x y → p x' y' → (x = x' ∧ y = y' ∨ ¬ adjacent x y x' y')) ∧
    (finset.univ.sum (λ i => finset.univ.sum (λ j => if p i j ↟ sorry then 1 else 0)) = 36) :=
sorry

end max_pawns_adj_l103_103243


namespace and_false_iff_not_both_true_l103_103452

variable (p q : Prop)

theorem and_false_iff_not_both_true (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
    sorry

end and_false_iff_not_both_true_l103_103452


namespace angle_between_tangents_l103_103714

def function (x : ℝ) : ℝ := x^2 * sqrt(3) / 24
def point_M : ℝ × ℝ := (4, -2 * sqrt(3))

theorem angle_between_tangents : 
  let f := function
  let M := point_M
  ∀ (x₀ : ℝ), (x₀ = 12 ∨ x₀ = -4) → 
  let tan_slope1 := sqrt(3)  -- slope of tangent at x₀ = 12
  let tan_slope2 := -1/sqrt(3)  -- slope of tangent at x₀ = -4
  ∃ (angle : ℝ), angle = 90 := 
sorry

end angle_between_tangents_l103_103714


namespace altitudes_of_triangle_roots_of_rational_polynomial_l103_103902

theorem altitudes_of_triangle_roots_of_rational_polynomial 
  (a b c : ℚ) 
  (h_sides_poly : ∃ (q2 q1 q0 : ℚ), ∀ x, (x - a) * (x - b) * (x - c) = x^3 + q2 * x^2 + q1 * x + q0) :
  ∃ (r6 r5 r4 r3 r2 r1 r0 : ℚ), ∀ y, (y - (2 * real.sqrt ((a + b + c) / 2 * ((a + b + c)/2 - a) * ((a + b + c)/2 - b) * ((a + b + c)/2 - c)) / a)^2) * (y - (2 * real.sqrt ((a + b + c) / 2 * ((a + b + c)/2 - a) * ((a + b + c)/2 - b) * ((a + b + c)/2 - c)) / b)^2) * (y - (2 * real.sqrt ((a + b + c) / 2 * ((a + b + c)/2 - a) * ((a + b + c)/2 - b) * ((a + b + c)/2 - c)) / c)^2) = y^6 + r5 * y^5 + r4 * y^4 + r3 * y^3 + r2 * y^2 + r1 * y + r0 :=
sorry

end altitudes_of_triangle_roots_of_rational_polynomial_l103_103902


namespace hall_length_width_difference_l103_103918

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L)
  (h2 : L * W = 450) :
  L - W = 15 :=
sorry

end hall_length_width_difference_l103_103918


namespace focal_length_of_ellipse_l103_103897

theorem focal_length_of_ellipse :
  ∀ (θ : ℝ), (∃ x y : ℝ, x = 3 * Real.cos θ ∧ y = Real.sqrt 6 * Real.sin θ) →
  (2 * Real.sqrt (9 - 6) = 2 * Real.sqrt 3) :=
by
  intro θ h
  have h_cos_sin : Real.cos θ ^ 2 + Real.sin θ ^ 2 = 1 := Real.cos_square_add_sin_square θ
  rcases h with ⟨x, y, hx, hy⟩
  have h_eq : x ^ 2 / 9 + y ^ 2 / 6 = 1 := by
    rw [hx, hy]
    norm_num
    rw [mul_pow, mul_pow]
    apply Eq.trans (congr_arg2 (+) (congr (congr_arg (*) (Real.cos θ ^ 2)) _) (congr_arg (*) (Real.sin θ ^ 2 _)))
    { rw [(congr_arg (_ * _) h_cos_sin)] }
  have h_focal_length : 2 * Real.sqrt (3 ^ 2 - (Real.sqrt 6) ^ 2) = 2 * Real.sqrt 3 := by
    norm_num
    sorry  -- Actual proof steps skipped
  exact h_focal_length

end focal_length_of_ellipse_l103_103897


namespace product_of_bases_l103_103687

noncomputable def binary_to_decimal (b : List ℕ) : ℕ :=
  b.foldr (λ (digit : ℕ) (acc : ℕ), digit + 2 * acc) 0

noncomputable def ternary_to_decimal (t : List ℕ) : ℕ :=
  t.foldr (λ (digit : ℕ) (acc : ℕ), digit + 3 * acc) 0

theorem product_of_bases : binary_to_decimal [1, 0, 1, 0] * ternary_to_decimal [1, 0, 2] = 110 :=
by {
  -- binary_to_decimal [1, 0, 1, 0] = 10
  have h1 : binary_to_decimal [1, 0, 1, 0] = 10, {
    sorry
  },
  -- ternary_to_decimal [1, 0, 2] = 11
  have h2 : ternary_to_decimal [1, 0, 2] = 11, {
    sorry
  },
  -- 10 * 11 = 110
  calc
    binary_to_decimal [1, 0, 1, 0] * ternary_to_decimal [1, 0, 2] = 10 * 11 : by rw [h1, h2]
                                                     ... = 110 : by norm_num
}

end product_of_bases_l103_103687


namespace parallelogram_height_l103_103586

theorem parallelogram_height (area base : ℝ) (h_area : area = 72) (h_base : base = 12) : 
  ∃ height : ℝ, area = base * height ∧ height = 6 :=
by
  use 6
  rw [h_area, h_base]
  norm_num
  sorry

end parallelogram_height_l103_103586


namespace second_box_probability_nth_box_probability_l103_103919

noncomputable def P_A1 : ℚ := 2 / 3
noncomputable def P_A2 : ℚ := 5 / 9
noncomputable def P_An (n : ℕ) : ℚ :=
  1 / 2 * (1 / 3) ^ n + 1 / 2

theorem second_box_probability :
  P_A2 = 5 / 9 := by
  sorry

theorem nth_box_probability (n : ℕ) :
  P_An n = 1 / 2 * (1 / 3) ^ n + 1 / 2 := by
  sorry

end second_box_probability_nth_box_probability_l103_103919


namespace min_value_sq_distance_l103_103040

theorem min_value_sq_distance {x y : ℝ} (h : x^2 + y^2 - 4 * x + 2 = 0) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x^2 + y^2 - 4 * x + 2 = 0 → x^2 + (y - 2)^2 ≥ m) :=
sorry

end min_value_sq_distance_l103_103040


namespace sum_of_squares_not_perfect_square_l103_103179

theorem sum_of_squares_not_perfect_square (n : ℤ) : ¬ (∃ k : ℤ, k^2 = (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2) :=
by
  sorry

end sum_of_squares_not_perfect_square_l103_103179


namespace factorize_expression_l103_103001

theorem factorize_expression (a b x y : ℝ) : 
  a^2 * b * (x - y)^3 - a * b^2 * (y - x)^2 = ab * (x - y)^2 * (a * x - a * y - b) :=
by
  sorry

end factorize_expression_l103_103001


namespace min_h25_l103_103682

noncomputable def h : ℕ → ℤ := sorry

def tenuous (h : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 → y > 0 → h(x) + h(y) > 2 * y^2

theorem min_h25 {h : ℕ → ℤ}
  (H_tenuous : tenuous h)
  (H_sum_min : ∀ k : ℕ, k > 0 → k ≤ 30 → h k = if k < 16 then 62 else 2 * k^2 + 1 - 62):
  h 25 = 1189 :=
by
  sorry

end min_h25_l103_103682


namespace max_product_real_roots_m_zero_l103_103720

theorem max_product_real_roots_m_zero (m : ℝ) :
  (m = 0) ↔ ((∀x ∈ ℝ, 2 * x^2 - m * x + m^2 = 0) ∧ 
              ((2 * x^2 - m * x + m^2).discriminant ≥ 0)) :=
by
  sorry

end max_product_real_roots_m_zero_l103_103720


namespace prob_conditional_B_given_A_l103_103014

-- Define the types for Students and Schools
inductive Student | A | B | C | D
inductive School | A | B | C

-- Define the assignment type as a function from Student to School
def Assignment := Student → School

-- Define the conditions
def valid_assignment (assign : Assignment) : Prop :=
  ∃ s1 s2 s3, s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧
    ({assign Student.A, assign Student.B, assign Student.C, assign Student.D} = {s1, s2, s3})

-- Define the events
def event_M (assign : Assignment) : Prop := assign Student.A = School.A
def event_N (assign : Assignment) : Prop := assign Student.B = School.B

-- Define the theorem to be proved
theorem prob_conditional_B_given_A :
  (∀ (assign : Assignment), valid_assignment assign → event_M assign → event_N assign = (5 / 12)) := 
  sorry

end prob_conditional_B_given_A_l103_103014


namespace LeRoy_should_pay_30_l103_103350

/-- Define the empirical amounts paid by LeRoy and Bernardo, and the total discount. -/
def LeRoy_paid : ℕ := 240
def Bernardo_paid : ℕ := 360
def total_discount : ℕ := 60

/-- Define total expenses pre-discount. -/
def total_expenses : ℕ := LeRoy_paid + Bernardo_paid

/-- Define total expenses post-discount. -/
def adjusted_expenses : ℕ := total_expenses - total_discount

/-- Define each person's adjusted share. -/
def each_adjusted_share : ℕ := adjusted_expenses / 2

/-- Define the amount LeRoy should pay Bernardo. -/
def leroy_to_pay : ℕ := each_adjusted_share - LeRoy_paid

/-- Prove that LeRoy should pay Bernardo $30 to equalize their expenses post-discount. -/
theorem LeRoy_should_pay_30 : leroy_to_pay = 30 :=
by 
  -- Proof goes here...
  sorry

end LeRoy_should_pay_30_l103_103350


namespace range_of_a_for_empty_solution_set_l103_103467

theorem range_of_a_for_empty_solution_set : 
  (∀ a : ℝ, (∀ x : ℝ, |x - 4| + |3 - x| < a → false) ↔ a ≤ 1) := 
sorry

end range_of_a_for_empty_solution_set_l103_103467


namespace HAO_is_38_degrees_l103_103154

noncomputable def ΔABC := {α β γ : ℝ // α + β + γ = 180 ∧ α ≠ 41 ∧ β ≠ 41 ∧ γ = 41}
noncomputable def circumcenter (Δ: ΔABC) : ℝ := sorry
noncomputable def orthocenter (Δ: ΔABC) : ℝ := sorry
noncomputable def midpoint (x y : ℝ) : ℝ := (x + y) / 2
noncomputable def angle_bisector (α : ℝ) : ℝ := sorry

theorem HAO_is_38_degrees
  (Δ : ΔABC)
  (O := circumcenter Δ)
  (H := orthocenter Δ)
  (M := midpoint O H)
  (A := some_point Δ)
  (angle_C : angle_in_triangle Δ γ = 41)
  (bisector_A : angle_bisector α passes_through M) :
  angle_in_triangle Δ HAO = 38 :=
sorry


end HAO_is_38_degrees_l103_103154


namespace equilateral_triangle_l103_103504

-- Given that A, B, and C are angles of triangle ABC
-- and the condition cos(A - B) * cos(B - C) * cos(C - A) = 1 holds
-- We need to prove that triangle ABC is equilateral

theorem equilateral_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : cos (A - B) * cos (B - C) * cos (C - A) = 1) : A = B ∧ B = C :=
by
  sorry

end equilateral_triangle_l103_103504


namespace value_of_expression_l103_103454

variables {x y z w : ℝ}

theorem value_of_expression (h1 : 4 * x * z + y * w = 4) (h2 : x * w + y * z = 8) :
  (2 * x + y) * (2 * z + w) = 20 :=
by
  sorry

end value_of_expression_l103_103454


namespace each_person_ate_2_cakes_l103_103536

def initial_cakes : ℕ := 8
def number_of_friends : ℕ := 4

theorem each_person_ate_2_cakes (h_initial_cakes : initial_cakes = 8)
  (h_number_of_friends : number_of_friends = 4) :
  initial_cakes / number_of_friends = 2 :=
by sorry

end each_person_ate_2_cakes_l103_103536


namespace value_of_a_l103_103411

theorem value_of_a (a : ℝ) (h : ∃ (r : ℕ), r = 3 ∧ -(Nat.choose 6 r).to_real * (-1:ℝ)^r * a^(6 - r) = -160) : a = 2 :=
by 
  sorry

end value_of_a_l103_103411


namespace no_int_k_such_that_P_k_equals_8_l103_103852

theorem no_int_k_such_that_P_k_equals_8
    (P : Polynomial ℤ) 
    (a b c d k : ℤ)
    (h0: a ≠ b)
    (h1: a ≠ c)
    (h2: a ≠ d)
    (h3: b ≠ c)
    (h4: b ≠ d)
    (h5: c ≠ d)
    (h6: P.eval a = 5)
    (h7: P.eval b = 5)
    (h8: P.eval c = 5)
    (h9: P.eval d = 5)
    : P.eval k ≠ 8 := by
  sorry

end no_int_k_such_that_P_k_equals_8_l103_103852


namespace sum_xi_bounds_l103_103407

theorem sum_xi_bounds (n : ℕ) (x : ℕ → ℝ) (h_nonneg : ∀ i, 1 ≤ i → i ≤ n → x i ≥ 0)
  (h_eq : ∑ i in finset.range (n + 1), x i ^ 2 + 2 * ∑ k in finset.range n, ∑ j in finset.Ico (k + 1) (n + 1), (real.sqrt (k / j) * x k * x j) = 1) :
  1 ≤ ∑ i in finset.range (n + 1), x i ∧ ∑ i in finset.range (n + 1), x i ≤ (∑ k in finset.range (n + 1), (real.sqrt k - real.sqrt (k - 1)) ^ 2) ^ (1 / 2) :=
sorry

end sum_xi_bounds_l103_103407


namespace mode_of_data_set_is_6_l103_103036

theorem mode_of_data_set_is_6 : 
  ∀ (x : ℝ), (list := [-1, 0, 4, x, 6, 15]) 
  median list = 5 → list.count 6 ≥ list.count y 6 :=
by
  intros x list h
  sorry

end mode_of_data_set_is_6_l103_103036


namespace delta_ratio_l103_103718

theorem delta_ratio 
  (Δx : ℝ) (Δy : ℝ) 
  (y_new : ℝ := (1 + Δx)^2 + 1)
  (y_old : ℝ := 1^2 + 1)
  (Δy_def : Δy = y_new - y_old) :
  Δy / Δx = 2 + Δx :=
by
  sorry

end delta_ratio_l103_103718
