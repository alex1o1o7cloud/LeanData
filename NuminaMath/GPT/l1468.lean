import Mathlib

namespace population_multiple_of_18_l1468_146892

theorem population_multiple_of_18
  (a b c P : ℕ)
  (ha : P = a^2)
  (hb : P + 200 = b^2 + 1)
  (hc : b^2 + 301 = c^2) :
  ∃ k, P = 18 * k := 
sorry

end population_multiple_of_18_l1468_146892


namespace average_after_15th_inning_l1468_146886

theorem average_after_15th_inning (A : ℝ) 
    (h_avg_increase : (14 * A + 75) = 15 * (A + 3)) : 
    A + 3 = 33 :=
by {
  sorry
}

end average_after_15th_inning_l1468_146886


namespace construct_triangle_l1468_146855

variables (a : ℝ) (α : ℝ) (d : ℝ)

-- Helper definitions
def is_triangle_valid (a α d : ℝ) : Prop := sorry

-- The theorem to be proven
theorem construct_triangle (a α d : ℝ) : is_triangle_valid a α d :=
sorry

end construct_triangle_l1468_146855


namespace my_op_five_four_l1468_146852

-- Define the operation a * b
def my_op (a b : ℤ) := a^2 + a * b - b^2

-- Define the theorem to prove 5 * 4 = 29 given the defined operation my_op
theorem my_op_five_four : my_op 5 4 = 29 := 
by 
sorry

end my_op_five_four_l1468_146852


namespace Carla_is_2_years_older_than_Karen_l1468_146889

-- Define the current age of Karen.
def Karen_age : ℕ := 2

-- Define the current age of Frank given that in 5 years he will be 36 years old.
def Frank_age : ℕ := 36 - 5

-- Define the current age of Ty given that Frank will be 3 times his age in 5 years.
def Ty_age : ℕ := 36 / 3

-- Define Carla's current age given that Ty is currently 4 years more than two times Carla's age.
def Carla_age : ℕ := (Ty_age - 4) / 2

-- Define the difference in age between Carla and Karen.
def Carla_Karen_age_diff : ℕ := Carla_age - Karen_age

-- The statement to be proven.
theorem Carla_is_2_years_older_than_Karen : Carla_Karen_age_diff = 2 := by
  -- The proof is not required, so we use sorry.
  sorry

end Carla_is_2_years_older_than_Karen_l1468_146889


namespace range_of_m_l1468_146868

noncomputable def problem (x m : ℝ) (p q : Prop) : Prop :=
  (¬ p → ¬ q) ∧ (¬ q → ¬ p → False) ∧ (p ↔ |1 - (x - 1) / 3| ≤ 2) ∧ 
  (q ↔ x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0)

theorem range_of_m (m : ℝ) (x : ℝ) (p q : Prop) 
  (h : problem x m p q) : m ≥ 9 :=
sorry

end range_of_m_l1468_146868


namespace jan_total_skips_l1468_146887

def jan_initial_speed : ℕ := 70
def jan_training_factor : ℕ := 2
def jan_skipping_time : ℕ := 5

theorem jan_total_skips :
  (jan_initial_speed * jan_training_factor) * jan_skipping_time = 700 := by
  sorry

end jan_total_skips_l1468_146887


namespace rounded_diff_greater_l1468_146804

variable (x y ε : ℝ)
variable (h1 : x > y)
variable (h2 : y > 0)
variable (h3 : ε > 0)

theorem rounded_diff_greater : (x + ε) - (y - ε) > x - y :=
  by
  sorry

end rounded_diff_greater_l1468_146804


namespace original_number_is_0_2_l1468_146896

theorem original_number_is_0_2 :
  ∃ x : ℝ, (1 / (1 / x - 1) - 1 = -0.75) ∧ x = 0.2 :=
by
  sorry

end original_number_is_0_2_l1468_146896


namespace min_lcm_leq_six_floor_l1468_146824

theorem min_lcm_leq_six_floor (n : ℕ) (h : n ≠ 4) (a : Fin n → ℕ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 2 * n) : 
  ∃ i j, i < j ∧ Nat.lcm (a i) (a j) ≤ 6 * (n / 2 + 1) :=
by
  sorry

end min_lcm_leq_six_floor_l1468_146824


namespace length_of_train_is_correct_l1468_146847

-- Definitions based on conditions
def speed_kmh := 90
def time_sec := 10

-- Convert speed from km/hr to m/s
def speed_ms := speed_kmh * (1000 / 3600)

-- Calculate the length of the train
def length_of_train := speed_ms * time_sec

-- Theorem to prove the length of the train
theorem length_of_train_is_correct : length_of_train = 250 := by
  sorry

end length_of_train_is_correct_l1468_146847


namespace rhombus_area_correct_l1468_146844

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct (x : ℝ) (h1 : rhombus_area 7 (abs (8 - x)) = 56) 
    (h2 : x ≠ 8) : x = -8 ∨ x = 24 :=
by
  sorry

end rhombus_area_correct_l1468_146844


namespace number_of_intersections_is_four_l1468_146865

def LineA (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0
def LineB (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def LineC (x y : ℝ) : Prop := x - y + 1 = 0
def LineD (x y : ℝ) : Prop := y - 2 = 0

def is_intersection (L1 L2 : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := L1 p.1 p.2 ∧ L2 p.1 p.2

theorem number_of_intersections_is_four :
  (∃ p1 : ℝ × ℝ, is_intersection LineA LineB p1) ∧
  (∃ p2 : ℝ × ℝ, is_intersection LineC LineD p2) ∧
  (∃ p3 : ℝ × ℝ, is_intersection LineA LineD p3) ∧
  (∃ p4 : ℝ × ℝ, is_intersection LineB LineD p4) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :=
by
  sorry

end number_of_intersections_is_four_l1468_146865


namespace part_one_part_two_l1468_146834

-- Part (1)
theorem part_one (a : ℝ) (h : a ≤ 2) (x : ℝ) :
  (|x - 1| + |x - a| ≥ 2 ↔ x ≤ 0.5 ∨ x ≥ 2.5) :=
sorry

-- Part (2)
theorem part_two (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, |x - 1| + |x - a| + |x - 1| ≥ 1) :
  a ≥ 2 :=
sorry

end part_one_part_two_l1468_146834


namespace polynomial_horner_form_operations_l1468_146819

noncomputable def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
  coeffs.foldr (fun a acc => a + acc * x) 0

theorem polynomial_horner_form_operations :
  let p := [1, 1, 2, 3, 4, 5]
  let x := 2
  horner_eval p x = ((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 ∧
  (∀ x, x = 2 → (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 =  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + 1 * x + 1)) ∧ 
  (∃ m a, m = 5 ∧ a = 5) := sorry

end polynomial_horner_form_operations_l1468_146819


namespace hajar_score_l1468_146841

variables (F H : ℕ)

theorem hajar_score 
  (h1 : F - H = 21)
  (h2 : F + H = 69)
  (h3 : F > H) :
  H = 24 :=
sorry

end hajar_score_l1468_146841


namespace intersection_A_B_l1468_146857

-- Definition of sets A and B
def A : Set ℤ := {0, 1, 2, 3}
def B : Set ℤ := { x | -1 ≤ x ∧ x < 3 }

-- Statement to prove
theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := 
sorry

end intersection_A_B_l1468_146857


namespace polygons_sides_l1468_146899

def sum_of_angles (x y : ℕ) : ℕ :=
(x - 2) * 180 + (y - 2) * 180

def num_diagonals (x y : ℕ) : ℕ :=
x * (x - 3) / 2 + y * (y - 3) / 2

theorem polygons_sides (x y : ℕ) (hx : x * (x - 3) / 2 + y * (y - 3) / 2 - (x + y) = 99) 
(hs : sum_of_angles x y = 21 * (x + y + num_diagonals x y) - 39) :
x = 17 ∧ y = 3 ∨ x = 3 ∧ y = 17 :=
by
  sorry

end polygons_sides_l1468_146899


namespace complete_square_l1468_146802

theorem complete_square 
  (x : ℝ) : 
  (2 * x^2 - 3 * x - 1 = 0) → 
  ((x - (3/4))^2 = (17/16)) :=
sorry

end complete_square_l1468_146802


namespace maynard_dog_holes_l1468_146861

open Real

theorem maynard_dog_holes (h_filled : ℝ) (h_unfilled : ℝ) (percent_filled : ℝ) 
  (percent_unfilled : ℝ) (total_holes : ℝ) :
  percent_filled = 0.75 →
  percent_unfilled = 0.25 →
  h_unfilled = 2 →
  h_filled = total_holes * percent_filled →
  total_holes = 8 :=
by
  intros hf pu hu hf_total
  sorry

end maynard_dog_holes_l1468_146861


namespace original_price_l1468_146826

theorem original_price (sale_price : ℝ) (discount : ℝ) : 
  sale_price = 55 → discount = 0.45 → 
  ∃ (P : ℝ), 0.55 * P = sale_price ∧ P = 100 :=
by
  sorry

end original_price_l1468_146826


namespace intersect_range_k_l1468_146891

theorem intersect_range_k : 
  ∀ k : ℝ, (∃ x y : ℝ, x^2 - (kx + 2)^2 = 6) ↔ 
  -Real.sqrt (5 / 3) < k ∧ k < Real.sqrt (5 / 3) := 
by sorry

end intersect_range_k_l1468_146891


namespace intersection_of_A_and_B_l1468_146846

def A : Set ℤ := { -3, -1, 0, 1 }
def B : Set ℤ := { x | (-2 < x) ∧ (x < 1) }

theorem intersection_of_A_and_B : A ∩ B = { -1, 0 } := by
  sorry

end intersection_of_A_and_B_l1468_146846


namespace expression_change_l1468_146816

variable (x b : ℝ)

-- The conditions
def expression (x : ℝ) : ℝ := x^3 - 5 * x + 1
def expr_change_plus (x b : ℝ) : ℝ := (x + b)^3 - 5 * (x + b) + 1
def expr_change_minus (x b : ℝ) : ℝ := (x - b)^3 - 5 * (x - b) + 1

-- The Lean statement to prove
theorem expression_change (h_b_pos : 0 < b) :
  expr_change_plus x b - expression x = 3 * b * x^2 + 3 * b^2 * x + b^3 - 5 * b ∨ 
  expr_change_minus x b - expression x = -3 * b * x^2 + 3 * b^2 * x - b^3 + 5 * b := 
by
  sorry

end expression_change_l1468_146816


namespace inscribed_rectangle_sides_l1468_146808

theorem inscribed_rectangle_sides {a b c : ℕ} (h₀ : a = 3) (h₁ : b = 4) (h₂ : c = 5) (ratio : ℚ) (h_ratio : ratio = 1 / 3) :
  ∃ (x y : ℚ), x = 20 / 29 ∧ y = 60 / 29 ∧ x = ratio * y :=
by
  sorry

end inscribed_rectangle_sides_l1468_146808


namespace circles_externally_tangent_l1468_146822

noncomputable def circle1_center : ℝ × ℝ := (-1, 1)
noncomputable def circle1_radius : ℝ := 2
noncomputable def circle2_center : ℝ × ℝ := (2, -3)
noncomputable def circle2_radius : ℝ := 3

noncomputable def distance_centers : ℝ :=
  Real.sqrt ((circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2)

theorem circles_externally_tangent :
  distance_centers = circle1_radius + circle2_radius :=
by
  -- The proof will show that the distance between the centers is equal to the sum of the radii, 
  -- indicating they are externally tangent.
  sorry

end circles_externally_tangent_l1468_146822


namespace length_of_tangent_l1468_146888

/-- 
Let O and O1 be the centers of the larger and smaller circles respectively with radii 8 and 3. 
The circles touch each other internally. Let A be the point of tangency and OM be the tangent from center O to the smaller circle. 
Prove that the length of this tangent is 4.
--/
theorem length_of_tangent {O O1 : Type} (radius_large : ℝ) (radius_small : ℝ) (OO1 : ℝ) 
  (OM O1M : ℝ) (h : 8 - 3 = 5) (h1 : OO1 = 5) (h2 : O1M = 3): OM = 4 :=
by
  sorry

end length_of_tangent_l1468_146888


namespace max_last_digit_of_sequence_l1468_146875

theorem max_last_digit_of_sequence :
  ∀ (s : Fin 1001 → ℕ), 
  (s 0 = 2) →
  (∀ (i : Fin 1000), (s i) * 10 + (s i.succ) ∈ {n | n % 17 = 0 ∨ n % 23 = 0}) →
  ∃ (d : ℕ), (d = s ⟨1000, sorry⟩) ∧ (∀ (d' : ℕ), d' = s ⟨1000, sorry⟩ → d' ≤ d) ∧ (d = 2) :=
by
  intros s h1 h2
  use 2
  sorry

end max_last_digit_of_sequence_l1468_146875


namespace additional_hours_needed_l1468_146882

-- Define the conditions
def speed : ℕ := 5  -- kilometers per hour
def total_distance : ℕ := 30 -- kilometers
def hours_walked : ℕ := 3 -- hours

-- Define the statement to prove
theorem additional_hours_needed : total_distance / speed - hours_walked = 3 := 
by
  sorry

end additional_hours_needed_l1468_146882


namespace cats_to_dogs_ratio_l1468_146858

theorem cats_to_dogs_ratio
    (cats dogs : ℕ)
    (ratio : cats / dogs = 3 / 4)
    (num_cats : cats = 18) :
    dogs = 24 :=
by
    sorry

end cats_to_dogs_ratio_l1468_146858


namespace inequality_solution_l1468_146812

noncomputable def solution_set_inequality : Set ℝ := {x | -2 < x ∧ x < 1 / 3}

theorem inequality_solution :
  {x : ℝ | (2 * x - 1) / (3 * x + 1) > 1} = solution_set_inequality :=
by
  sorry

end inequality_solution_l1468_146812


namespace smallest_integer_in_set_l1468_146801

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 144) (h2 : greatest = 153) : ∃ x : ℤ, x = 135 :=
by
  sorry

end smallest_integer_in_set_l1468_146801


namespace find_angle_l1468_146859

theorem find_angle (r1 r2 : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 2) 
(h_shaded : ∀ α : ℝ, 0 < α ∧ α < 2 * π → 
  (360 / 360 * pi * r1^2 + (α / (2 * π)) * pi * r2^2 - (α / (2 * π)) * pi * r1^2 = (1/3) * (pi * r2^2))) : 
  (∀ α : ℝ, 0 < α ∧ α < 2 * π ↔ 
  α = π / 3 ) :=
by
  sorry

end find_angle_l1468_146859


namespace washer_dryer_cost_diff_l1468_146810

-- conditions
def total_cost : ℕ := 1200
def washer_cost : ℕ := 710
def dryer_cost : ℕ := total_cost - washer_cost

-- proof statement
theorem washer_dryer_cost_diff : (washer_cost - dryer_cost) = 220 :=
by
  sorry

end washer_dryer_cost_diff_l1468_146810


namespace problem_part1_problem_part2_l1468_146863

def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def CR_A : Set ℝ := { x | x < 3 ∨ x > 7 }

theorem problem_part1 : A ∪ B = { x | 3 ≤ x ∧ x ≤ 7 } := by
  sorry

theorem problem_part2 : (CR_A ∩ B) = { x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10) } := by
  sorry

end problem_part1_problem_part2_l1468_146863


namespace prove_x_ge_neg_one_sixth_l1468_146897

variable (x y : ℝ)

theorem prove_x_ge_neg_one_sixth (h : x^4 * y^2 + y^4 + 2 * x^3 * y + 6 * x^2 * y + x^2 + 8 ≤ 0) :
  x ≥ -1 / 6 :=
sorry

end prove_x_ge_neg_one_sixth_l1468_146897


namespace five_year_salary_increase_l1468_146879

noncomputable def salary_growth (S : ℝ) := S * (1.08)^5

theorem five_year_salary_increase (S : ℝ) : 
  salary_growth S = S * 1.4693 := 
sorry

end five_year_salary_increase_l1468_146879


namespace range_of_a_l1468_146818

theorem range_of_a (x a : ℝ) (h₁ : 0 < x) (h₂ : x < 2) (h₃ : a - 1 < x) (h₄ : x ≤ a) :
  1 ≤ a ∧ a < 2 :=
by
  sorry

end range_of_a_l1468_146818


namespace worms_stolen_correct_l1468_146853

-- Given conditions translated into Lean statements
def num_babies : ℕ := 6
def worms_per_baby_per_day : ℕ := 3
def papa_bird_worms : ℕ := 9
def mama_bird_initial_worms : ℕ := 13
def additional_worms_needed : ℕ := 34

-- From the conditions, determine the total number of worms needed for 3 days
def total_worms_needed : ℕ := worms_per_baby_per_day * num_babies * 3

-- Calculate how many worms they will have after catching additional worms
def total_worms_after_catching_more : ℕ := papa_bird_worms + mama_bird_initial_worms + additional_worms_needed

-- Amount suspected to be stolen
def worms_stolen : ℕ := total_worms_after_catching_more - total_worms_needed

theorem worms_stolen_correct : worms_stolen = 2 :=
by sorry

end worms_stolen_correct_l1468_146853


namespace greatest_divisor_of_sum_of_arith_seq_l1468_146878

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l1468_146878


namespace total_copies_l1468_146800

theorem total_copies (rate1 : ℕ) (rate2 : ℕ) (time : ℕ) (total : ℕ) 
  (h1 : rate1 = 25) (h2 : rate2 = 55) (h3 : time = 30) : 
  total = rate1 * time + rate2 * time := 
  sorry

end total_copies_l1468_146800


namespace dot_product_EC_ED_l1468_146854

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l1468_146854


namespace meaningful_expression_range_l1468_146866

theorem meaningful_expression_range (x : ℝ) : (3 * x + 9 ≥ 0) ∧ (x ≠ 2) ↔ (x ≥ -3 ∧ x ≠ 2) := by
  sorry

end meaningful_expression_range_l1468_146866


namespace product_of_slopes_l1468_146807

theorem product_of_slopes (p : ℝ) (hp : 0 < p) :
  let T := (p, 0)
  let parabola := fun x y => y^2 = 2*p*x
  let line := fun x y => y = x - p
  -- Define intersection points A and B on the parabola satisfying the line equation
  ∃ A B : ℝ × ℝ, 
  parabola A.1 A.2 ∧ line A.1 A.2 ∧
  parabola B.1 B.2 ∧ line B.1 B.2 ∧
  -- O is the origin
  let O := (0, 0)
  -- define slope function
  let slope (P Q : ℝ × ℝ) := (Q.2 - P.2) / (Q.1 - P.1)
  -- slopes of OA and OB
  let k_OA := slope O A
  let k_OB := slope O B
  -- product of slopes
  k_OA * k_OB = -2 := sorry

end product_of_slopes_l1468_146807


namespace max_probability_pc_l1468_146895

variables (p1 p2 p3 : ℝ)
variable (h : p3 > p2 ∧ p2 > p1 ∧ p1 > 0)

def PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem max_probability_pc : PC > PA ∧ PC > PB := 
by 
  sorry

end max_probability_pc_l1468_146895


namespace solve_for_x_l1468_146871

theorem solve_for_x : ∃ x : ℝ, 5 * x + 9 * x = 570 - 12 * (x - 5) ∧ x = 315 / 13 :=
by
  sorry

end solve_for_x_l1468_146871


namespace intersect_A_B_complement_l1468_146806

-- Define the sets A and B
def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | x > 1}

-- Find the complement of B in ℝ
def B_complement := {x : ℝ | x ≤ 1}

-- Prove that the intersection of A and the complement of B is equal to (-1, 1]
theorem intersect_A_B_complement : A ∩ B_complement = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  -- Proof is to be provided
  sorry

end intersect_A_B_complement_l1468_146806


namespace least_positive_integer_l1468_146856

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l1468_146856


namespace product_gcd_lcm_15_9_l1468_146876

theorem product_gcd_lcm_15_9 : Nat.gcd 15 9 * Nat.lcm 15 9 = 135 := 
by
  -- skipping proof as instructed
  sorry

end product_gcd_lcm_15_9_l1468_146876


namespace line_intersects_circle_l1468_146881

/-- The positional relationship between the line y = ax + 1 and the circle x^2 + y^2 - 2x - 3 = 0
    is always intersecting for any real number a. -/
theorem line_intersects_circle (a : ℝ) : 
    ∀ a : ℝ, ∃ x y : ℝ, y = a * x + 1 ∧ x^2 + y^2 - 2 * x - 3 = 0 :=
by
    sorry

end line_intersects_circle_l1468_146881


namespace fraction_saved_l1468_146821

variable {P : ℝ} (hP : P > 0)

theorem fraction_saved (f : ℝ) (hf0 : 0 ≤ f) (hf1 : f ≤ 1) (condition : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  sorry

end fraction_saved_l1468_146821


namespace expression_equals_100_l1468_146884

-- Define the terms in the numerator and their squares
def num1 := 0.02
def num2 := 0.52
def num3 := 0.035

def num1_sq := num1^2
def num2_sq := num2^2
def num3_sq := num3^2

-- Define the terms in the denominator and their squares
def denom1 := 0.002
def denom2 := 0.052
def denom3 := 0.0035

def denom1_sq := denom1^2
def denom2_sq := denom2^2
def denom3_sq := denom3^2

-- Define the sums of the squares
def sum_numerator := num1_sq + num2_sq + num3_sq
def sum_denominator := denom1_sq + denom2_sq + denom3_sq

-- Define the final expression
def expression := sum_numerator / sum_denominator

-- Prove the expression equals the correct answer
theorem expression_equals_100 : expression = 100 := by sorry

end expression_equals_100_l1468_146884


namespace work_days_B_works_l1468_146827

theorem work_days_B_works (x : ℕ) (A_work_rate B_work_rate : ℚ) (A_remaining_days : ℕ) (total_work : ℚ) :
  A_work_rate = (1 / 12) ∧
  B_work_rate = (1 / 15) ∧
  A_remaining_days = 4 ∧
  total_work = 1 →
  x * B_work_rate + A_remaining_days * A_work_rate = total_work →
  x = 10 :=
sorry

end work_days_B_works_l1468_146827


namespace vehicle_speed_increase_l1468_146838

/-- Vehicle dynamics details -/
structure Vehicle := 
  (initial_speed : ℝ) 
  (deceleration : ℝ)
  (initial_distance_from_A : ℝ)

/-- Given conditions -/
def conditions (A B C : Vehicle) : Prop :=
  A.initial_speed = 80 ∧
  B.initial_speed = 60 ∧
  C.initial_speed = 70 ∧ 
  C.deceleration = 2 ∧
  B.initial_distance_from_A = 40 ∧
  C.initial_distance_from_A = 260

/-- Prove A needs to increase its speed by 5 mph -/
theorem vehicle_speed_increase (A B C : Vehicle) (h : conditions A B C) : 
  ∃ dA : ℝ, dA = 5 ∧ A.initial_speed + dA > B.initial_speed → 
    (A.initial_distance_from_A / (A.initial_speed + dA - B.initial_speed)) < 
    (C.initial_distance_from_A / (A.initial_speed + dA + C.initial_speed - C.deceleration)) :=
sorry

end vehicle_speed_increase_l1468_146838


namespace probability_at_least_one_red_l1468_146890

def total_balls : ℕ := 6
def red_balls : ℕ := 4
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_at_least_one_red :
  (choose_two red_balls + red_balls * (total_balls - red_balls - 1) / 2) / choose_two total_balls = 14 / 15 :=
sorry

end probability_at_least_one_red_l1468_146890


namespace sum_of_perimeters_l1468_146870

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) : 
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * x + 4 * y := by
  sorry

end sum_of_perimeters_l1468_146870


namespace factorize_expression_l1468_146845

theorem factorize_expression (x a : ℝ) : 4 * x - x * a^2 = x * (2 - a) * (2 + a) :=
by 
  sorry

end factorize_expression_l1468_146845


namespace find_blue_beads_per_row_l1468_146833

-- Given the conditions of the problem:
def number_of_purple_beads : ℕ := 50 * 20
def number_of_gold_beads : ℕ := 80
def total_cost : ℕ := 180

-- Define the main theorem to solve for the number of blue beads per row.
theorem find_blue_beads_per_row (x : ℕ) :
  (number_of_purple_beads + 40 * x + number_of_gold_beads = total_cost) → x = (total_cost - (number_of_purple_beads + number_of_gold_beads)) / 40 := 
by {
  -- Proof steps would go here
  sorry
}

end find_blue_beads_per_row_l1468_146833


namespace max_sum_nonneg_l1468_146874

theorem max_sum_nonneg (a b c d : ℝ) (h : a + b + c + d = 0) : 
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := 
sorry

end max_sum_nonneg_l1468_146874


namespace max_cube_side_length_max_parallelepiped_dimensions_l1468_146880

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l1468_146880


namespace abs_diff_eq_five_l1468_146817

theorem abs_diff_eq_five (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 7) : |a - b| = 5 :=
by
  sorry

end abs_diff_eq_five_l1468_146817


namespace measure_angle_BRC_l1468_146840

inductive Point : Type
| A 
| B 
| C 
| P 
| Q 
| R 

open Point

def is_inside_triangle (P : Point) (A B C : Point) : Prop := sorry

def intersection (a b c : Point) : Point := sorry

def length (a b : Point) : ℝ := sorry

def angle (a b c : Point) : ℝ := sorry

theorem measure_angle_BRC 
  (P : Point) (A B C : Point)
  (h_inside : is_inside_triangle P A B C)
  (hQ : Q = intersection A C P)
  (hR : R = intersection A B P)
  (h_lengths_equal : length A R = length R B ∧ length R B = length C P)
  (h_CQ_PQ : length C Q = length P Q) :
  angle B R C = 120 := 
sorry

end measure_angle_BRC_l1468_146840


namespace arman_hourly_rate_increase_l1468_146893

theorem arman_hourly_rate_increase :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let total_payment := 770
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := total_payment - last_week_earnings
  let this_week_rate := this_week_earnings / this_week_hours
  let rate_increase := this_week_rate - last_week_rate
  rate_increase = 0.50 :=
by {
  sorry
}

end arman_hourly_rate_increase_l1468_146893


namespace sector_central_angle_l1468_146813

theorem sector_central_angle (r l α : ℝ) (h1 : 2 * r + l = 6) (h2 : 1/2 * l * r = 2) :
  α = l / r → (α = 1 ∨ α = 4) :=
by
  sorry

end sector_central_angle_l1468_146813


namespace common_tangent_l1468_146823

-- Definition of the ellipse and hyperbola
def ellipse (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144
def hyperbola (x y : ℝ) : Prop := 7 * x^2 - 32 * y^2 = 224

-- The statement to prove
theorem common_tangent :
  (∀ x y : ℝ, ellipse x y → hyperbola x y → ((x + y + 5 = 0) ∨ (x + y - 5 = 0) ∨ (x - y + 5 = 0) ∨ (x - y - 5 = 0))) := 
sorry

end common_tangent_l1468_146823


namespace bhishma_speed_l1468_146825

-- Given definitions based on conditions
def track_length : ℝ := 600
def bruce_speed : ℝ := 30
def time_meet : ℝ := 90

-- Main theorem we want to prove
theorem bhishma_speed : ∃ v : ℝ, v = 23.33 ∧ (bruce_speed * time_meet) = (v * time_meet + track_length) :=
  by
    sorry

end bhishma_speed_l1468_146825


namespace total_cost_of_tshirts_l1468_146894

theorem total_cost_of_tshirts
  (White_packs : ℕ := 3) (Blue_packs : ℕ := 2) (Red_packs : ℕ := 4) (Green_packs : ℕ := 1) 
  (White_price_per_pack : ℝ := 12) (Blue_price_per_pack : ℝ := 8) (Red_price_per_pack : ℝ := 10) (Green_price_per_pack : ℝ := 6) 
  (White_discount : ℝ := 0.10) (Blue_discount : ℝ := 0.05) (Red_discount : ℝ := 0.15) (Green_discount : ℝ := 0.00) :
  White_packs * White_price_per_pack * (1 - White_discount) +
  Blue_packs * Blue_price_per_pack * (1 - Blue_discount) +
  Red_packs * Red_price_per_pack * (1 - Red_discount) +
  Green_packs * Green_price_per_pack * (1 - Green_discount) = 87.60 := by
    sorry

end total_cost_of_tshirts_l1468_146894


namespace fewest_presses_to_original_l1468_146867

theorem fewest_presses_to_original (x : ℝ) (hx : x = 16) (f : ℝ → ℝ)
    (hf : ∀ y : ℝ, f y = 1 / y) : (f (f x)) = x :=
by
  sorry

end fewest_presses_to_original_l1468_146867


namespace find_non_negative_integers_l1468_146803

def has_exactly_two_distinct_solutions (a : ℕ) (m : ℕ) : Prop :=
  ∃ (x₁ x₂ : ℕ), (x₁ < m) ∧ (x₂ < m) ∧ (x₁ ≠ x₂) ∧ (x₁^2 + a) % m = 0 ∧ (x₂^2 + a) % m = 0

theorem find_non_negative_integers (a : ℕ) (m : ℕ := 2007) : 
  a < m ∧ has_exactly_two_distinct_solutions a m ↔ a = 446 ∨ a = 1115 ∨ a = 1784 :=
sorry

end find_non_negative_integers_l1468_146803


namespace monotonic_decreasing_interval_l1468_146885

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -1 ∧ b = 11 ∧ ∀ x, x > a ∧ x < b → (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l1468_146885


namespace system_of_equations_correct_l1468_146843

theorem system_of_equations_correct (x y : ℝ) (h1 : x + y = 2000) (h2 : y = x * 0.30) :
  x + y = 2000 ∧ y = x * 0.30 :=
by 
  exact ⟨h1, h2⟩

end system_of_equations_correct_l1468_146843


namespace polynomial_divisible_l1468_146851

theorem polynomial_divisible (a b c : ℕ) :
  (X^(3 * a) + X^(3 * b + 1) + X^(3 * c + 2)) % (X^2 + X + 1) = 0 :=
by sorry

end polynomial_divisible_l1468_146851


namespace decreasing_power_function_l1468_146872

theorem decreasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(m^2 + m - 1) < (m^2 - m - 1) * (x + 1) ^ (m^2 + m - 1)) →
  m = -1 :=
sorry

end decreasing_power_function_l1468_146872


namespace ellipse_eq_range_m_l1468_146832

theorem ellipse_eq_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m - 1) + y^2 / (3 - m) = 1)) ↔ (1 < m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end ellipse_eq_range_m_l1468_146832


namespace functional_equation_solution_l1468_146864

noncomputable def f (x : ℚ) : ℚ := sorry

theorem functional_equation_solution (f : ℚ → ℚ) (f_pos_rat : ∀ x : ℚ, 0 < x → 0 < f x) :
  (∀ x y : ℚ, 0 < x → 0 < y → f x + f y + 2 * x * y * f (x * y) = f (x * y) / f (x + y)) →
  (∀ x : ℚ, 0 < x → f x = 1 / x ^ 2) :=
by
  sorry

end functional_equation_solution_l1468_146864


namespace sequence_solution_l1468_146815

theorem sequence_solution 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n / (2 + a n))
  (h2 : a 1 = 1) :
  ∀ n, a n = 1 / (2^n - 1) :=
sorry

end sequence_solution_l1468_146815


namespace beef_weight_after_processing_l1468_146839

noncomputable def initial_weight : ℝ := 840
noncomputable def lost_percentage : ℝ := 35
noncomputable def retained_percentage : ℝ := 100 - lost_percentage
noncomputable def final_weight : ℝ := retained_percentage / 100 * initial_weight

theorem beef_weight_after_processing : final_weight = 546 := by
  sorry

end beef_weight_after_processing_l1468_146839


namespace find_m_plus_n_l1468_146877

theorem find_m_plus_n (m n : ℤ) 
  (H1 : (x^3 + m*x + n) * (x^2 - 3*x + 1) ≠ 1 * x^2 + 1 * x^3) 
  (H2 : (x^3 + m*x + n) * (x^2 - 3*x + 1) ≠ 1 * x^2 + 1 * x^3) : 
  m + n = -4 := 
by
  sorry

end find_m_plus_n_l1468_146877


namespace find_sides_from_diagonals_l1468_146848

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l1468_146848


namespace transform_negation_l1468_146873

-- Define the terms a, b, and c as real numbers
variables (a b c : ℝ)

-- State the theorem we want to prove
theorem transform_negation (a b c : ℝ) : 
  - (a - b + c) = -a + b - c :=
sorry

end transform_negation_l1468_146873


namespace cups_of_rice_morning_l1468_146805

variable (cupsMorning : Nat) -- Number of cups of rice Robbie eats in the morning
variable (cupsAfternoon : Nat := 2) -- Cups of rice in the afternoon
variable (cupsEvening : Nat := 5) -- Cups of rice in the evening
variable (fatPerCup : Nat := 10) -- Fat in grams per cup of rice
variable (weeklyFatIntake : Nat := 700) -- Total fat in grams per week

theorem cups_of_rice_morning :
  ((cupsMorning + cupsAfternoon + cupsEvening) * fatPerCup) = (weeklyFatIntake / 7) → cupsMorning = 3 :=
  by
    sorry

end cups_of_rice_morning_l1468_146805


namespace rainfall_ratio_l1468_146850

theorem rainfall_ratio (R1 R2 : ℕ) (H1 : R2 = 18) (H2 : R1 + R2 = 30) : R2 / R1 = 3 / 2 := by
  sorry

end rainfall_ratio_l1468_146850


namespace solve_eq_solution_l1468_146831

def eq_solution (x y : ℕ) : Prop := 3 ^ x = 2 ^ x * y + 1

theorem solve_eq_solution (x y : ℕ) (h1 : x > 0) (h2 : y > 0) : 
  eq_solution x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) :=
sorry

end solve_eq_solution_l1468_146831


namespace calculator_display_after_50_presses_l1468_146849

theorem calculator_display_after_50_presses :
  let initial_display := 3
  let operation (x : ℚ) := 1 / (1 - x)
  (Nat.iterate operation 50 initial_display) = 2 / 3 :=
by
  sorry

end calculator_display_after_50_presses_l1468_146849


namespace find_root_product_l1468_146842

theorem find_root_product :
  (∃ r s t : ℝ, (∀ x : ℝ, (x - r) * (x - s) * (x - t) = x^3 - 15 * x^2 + 26 * x - 8) ∧
  (1 + r) * (1 + s) * (1 + t) = 50) :=
sorry

end find_root_product_l1468_146842


namespace sugar_for_third_layer_l1468_146862

theorem sugar_for_third_layer (s1 : ℕ) (s2 : ℕ) (s3 : ℕ) 
  (h1 : s1 = 2) 
  (h2 : s2 = 2 * s1) 
  (h3 : s3 = 3 * s2) : 
  s3 = 12 := 
sorry

end sugar_for_third_layer_l1468_146862


namespace teacher_work_months_l1468_146869

variable (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) (total_earnings : ℕ)

def monthly_earnings (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) : ℕ :=
  periods_per_day * pay_per_period * days_per_month

def number_of_months_worked (total_earnings : ℕ) (monthly_earnings : ℕ) : ℕ :=
  total_earnings / monthly_earnings

theorem teacher_work_months :
  let periods_per_day := 5
  let pay_per_period := 5
  let days_per_month := 24
  let total_earnings := 3600
  number_of_months_worked total_earnings (monthly_earnings periods_per_day pay_per_period days_per_month) = 6 :=
by
  sorry

end teacher_work_months_l1468_146869


namespace abs_floor_value_l1468_146837

theorem abs_floor_value : (Int.floor (|(-56.3: Real)|)) = 56 := 
by
  sorry

end abs_floor_value_l1468_146837


namespace difference_of_squares_example_l1468_146820

theorem difference_of_squares_example : 169^2 - 168^2 = 337 :=
by
  -- The proof steps using the difference of squares formula is omitted here.
  sorry

end difference_of_squares_example_l1468_146820


namespace a_divisible_by_11_iff_b_divisible_by_11_l1468_146836

-- Define the relevant functions
def a (n : ℕ) : ℕ := n^5 + 5^n
def b (n : ℕ) : ℕ := n^5 * 5^n + 1

-- State that for a positive integer n, a(n) is divisible by 11 if and only if b(n) is also divisible by 11
theorem a_divisible_by_11_iff_b_divisible_by_11 (n : ℕ) (hn : 0 < n) : 
  (a n % 11 = 0) ↔ (b n % 11 = 0) :=
sorry

end a_divisible_by_11_iff_b_divisible_by_11_l1468_146836


namespace playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l1468_146835

def hasWinningStrategyA (n : ℕ) : Prop :=
  n ≥ 8

def hasWinningStrategyB (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5

def draw (n : ℕ) : Prop :=
  n = 6 ∨ n = 7

theorem playerA_winning_strategy (n : ℕ) : n ≥ 8 → hasWinningStrategyA n :=
by
  sorry

theorem playerB_winning_strategy (n : ℕ) : (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5) → hasWinningStrategyB n :=
by
  sorry

theorem no_winning_strategy (n : ℕ) : n = 6 ∨ n = 7 → draw n :=
by
  sorry

end playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l1468_146835


namespace find_n_l1468_146814

theorem find_n : ∃ (n : ℤ), -150 < n ∧ n < 150 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1600 * Real.pi / 180) :=
sorry

end find_n_l1468_146814


namespace soccer_goal_difference_l1468_146811

theorem soccer_goal_difference (n : ℕ) (h : n = 2020) :
  ¬ ∃ g : Fin n → ℤ,
    (∀ i j : Fin n, i < j → (g i < g j)) ∧ 
    (∀ i : Fin n, ∃ x y : ℕ, x + y = n - 1 ∧ 3 * x = (n - 1 - x) ∧ g i = x - y) :=
by
  sorry

end soccer_goal_difference_l1468_146811


namespace correct_operation_l1468_146809

theorem correct_operation (a b : ℝ) : a * b^2 - b^2 * a = 0 := by
  sorry

end correct_operation_l1468_146809


namespace new_radius_of_circle_l1468_146898

theorem new_radius_of_circle
  (r_1 : ℝ)
  (A_1 : ℝ := π * r_1^2)
  (r_2 : ℝ)
  (A_2 : ℝ := 0.64 * A_1) 
  (h1 : r_1 = 5) 
  (h2 : A_2 = π * r_2^2) : 
  r_2 = 4 :=
by 
  sorry

end new_radius_of_circle_l1468_146898


namespace find_y_l1468_146828

theorem find_y (x : ℤ) (y : ℤ) (h : x = 5) (h1 : 3 * x = (y - x) + 4) : y = 16 :=
by
  sorry

end find_y_l1468_146828


namespace runs_in_last_match_l1468_146860

-- Definitions based on the conditions
def initial_bowling_average : ℝ := 12.4
def wickets_last_match : ℕ := 7
def decrease_average : ℝ := 0.4
def new_average : ℝ := initial_bowling_average - decrease_average
def approximate_wickets_before : ℕ := 145

-- The Lean statement of the problem
theorem runs_in_last_match (R : ℝ) :
  ((initial_bowling_average * approximate_wickets_before + R) / 
   (approximate_wickets_before + wickets_last_match) = new_average) →
   R = 28 :=
by
  sorry

end runs_in_last_match_l1468_146860


namespace joe_two_different_fruits_in_a_day_l1468_146830

def joe_meal_event : Type := {meal : ℕ // meal = 4}
def joe_fruit_choice : Type := {fruit : ℕ // fruit ≤ 4}

noncomputable def prob_all_same_fruit : ℚ := (1 / 4) ^ 4 * 4
noncomputable def prob_at_least_two_diff_fruits : ℚ := 1 - prob_all_same_fruit

theorem joe_two_different_fruits_in_a_day :
  prob_at_least_two_diff_fruits = 63 / 64 :=
by
  sorry

end joe_two_different_fruits_in_a_day_l1468_146830


namespace find_k_l1468_146883

theorem find_k (k : ℝ) (h : (3 : ℝ)^2 - k * (3 : ℝ) - 6 = 0) : k = 1 :=
by
  sorry

end find_k_l1468_146883


namespace spade_evaluation_l1468_146829

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_evaluation : spade 5 (spade 6 7 + 2) = -96 := by
  sorry

end spade_evaluation_l1468_146829
