import Mathlib

namespace NUMINAMATH_GPT_probability_correct_l133_13347

noncomputable def probability_study_group : ℝ :=
  let p_woman : ℝ := 0.5
  let p_man : ℝ := 0.5

  let p_woman_lawyer : ℝ := 0.3
  let p_woman_doctor : ℝ := 0.4
  let p_woman_engineer : ℝ := 0.3

  let p_man_lawyer : ℝ := 0.4
  let p_man_doctor : ℝ := 0.2
  let p_man_engineer : ℝ := 0.4

  (p_woman * p_woman_lawyer + p_woman * p_woman_doctor +
  p_man * p_man_lawyer + p_man * p_man_doctor)

theorem probability_correct : probability_study_group = 0.65 := by
  sorry

end NUMINAMATH_GPT_probability_correct_l133_13347


namespace NUMINAMATH_GPT_alpha_value_l133_13386

theorem alpha_value (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.logb 3 (x + 1)) (h2 : f α = 1) : α = 2 := by
  sorry

end NUMINAMATH_GPT_alpha_value_l133_13386


namespace NUMINAMATH_GPT_total_groups_correct_l133_13363

-- Definitions from conditions
def eggs := 57
def egg_group_size := 7

def bananas := 120
def banana_group_size := 10

def marbles := 248
def marble_group_size := 8

-- Calculate the number of groups for each type of object
def egg_groups := eggs / egg_group_size
def banana_groups := bananas / banana_group_size
def marble_groups := marbles / marble_group_size

-- Total number of groups
def total_groups := egg_groups + banana_groups + marble_groups

-- Proof statement
theorem total_groups_correct : total_groups = 51 := by
  sorry

end NUMINAMATH_GPT_total_groups_correct_l133_13363


namespace NUMINAMATH_GPT_equal_area_condition_l133_13348

variable {θ : ℝ} (h1 : 0 < θ) (h2 : θ < π / 2)

theorem equal_area_condition : 2 * θ = (Real.tan θ) * (Real.tan (2 * θ)) :=
by {
  sorry
}

end NUMINAMATH_GPT_equal_area_condition_l133_13348


namespace NUMINAMATH_GPT_number_of_dogs_l133_13306

def legs_in_pool : ℕ := 24
def human_legs : ℕ := 4
def legs_per_dog : ℕ := 4

theorem number_of_dogs : (legs_in_pool - human_legs) / legs_per_dog = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dogs_l133_13306


namespace NUMINAMATH_GPT_no_solutions_l133_13336

theorem no_solutions (N : ℕ) (d : ℕ) (H : ∀ (i j : ℕ), i ≠ j → d = 6 ∧ d + d = 13) : false :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_l133_13336


namespace NUMINAMATH_GPT_unique_solution_3x_4y_5z_l133_13339

theorem unique_solution_3x_4y_5z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 ^ x + 4 ^ y = 5 ^ z → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_unique_solution_3x_4y_5z_l133_13339


namespace NUMINAMATH_GPT_find_number_l133_13378

theorem find_number (x : ℤ) (h : x - 7 = 9) : x * 3 = 48 :=
by sorry

end NUMINAMATH_GPT_find_number_l133_13378


namespace NUMINAMATH_GPT_smaller_circle_radius_l133_13397

noncomputable def radius_of_smaller_circles (R : ℝ) (r1 r2 r3 : ℝ) (OA OB OC : ℝ) : Prop :=
(OA = R + r1) ∧ (OB = R + 3 * r1) ∧ (OC = R + 5 * r1) ∧ 
((OB = OA + 2 * r1) ∧ (OC = OB + 2 * r1))

theorem smaller_circle_radius (r : ℝ) (R : ℝ := 2) :
  radius_of_smaller_circles R r r r (R + r) (R + 3 * r) (R + 5 * r) → r = 1 :=
by
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l133_13397


namespace NUMINAMATH_GPT_circle_touching_y_axis_radius_5_k_value_l133_13346

theorem circle_touching_y_axis_radius_5_k_value :
  ∃ k : ℝ, ∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) →
    (∃ r : ℝ, r = 5 ∧ (∀ c : ℝ × ℝ, (c.1 + 4)^2 + (c.2 + 2)^2 = r^2) ∧
      (∃ x : ℝ, x + 4 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_circle_touching_y_axis_radius_5_k_value_l133_13346


namespace NUMINAMATH_GPT_reflection_coordinates_l133_13324

-- Define the original coordinates of point M
def original_point : (ℝ × ℝ) := (3, -4)

-- Define the function to reflect a point across the x-axis
def reflect_across_x_axis (p: ℝ × ℝ) : (ℝ × ℝ) :=
  (p.1, -p.2)

-- State the theorem to prove the coordinates after reflection
theorem reflection_coordinates :
  reflect_across_x_axis original_point = (3, 4) :=
by
  sorry

end NUMINAMATH_GPT_reflection_coordinates_l133_13324


namespace NUMINAMATH_GPT_change_positions_of_three_out_of_eight_l133_13340

theorem change_positions_of_three_out_of_eight :
  (Nat.choose 8 3) * (Nat.factorial 3) = (Nat.choose 8 3) * 6 :=
by
  sorry

end NUMINAMATH_GPT_change_positions_of_three_out_of_eight_l133_13340


namespace NUMINAMATH_GPT_largest_x_exists_largest_x_largest_real_number_l133_13388

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end NUMINAMATH_GPT_largest_x_exists_largest_x_largest_real_number_l133_13388


namespace NUMINAMATH_GPT_proof_S_squared_l133_13393

variables {a b c p S r r_a r_b r_c : ℝ}

-- Conditions
axiom cond1 : r * p = r_a * (p - a)
axiom cond2 : r * r_a = (p - b) * (p - c)
axiom cond3 : r_b * r_c = p * (p - a)
axiom heron : S^2 = p * (p - a) * (p - b) * (p - c)

-- Proof statement
theorem proof_S_squared : S^2 = r * r_a * r_b * r_c :=
by sorry

end NUMINAMATH_GPT_proof_S_squared_l133_13393


namespace NUMINAMATH_GPT_Jill_has_5_peaches_l133_13316

-- Define the variables and their relationships
variables (S Jl Jk : ℕ)

-- Declare the conditions as assumptions
axiom Steven_has_14_peaches : S = 14
axiom Jake_has_6_fewer_peaches_than_Steven : Jk = S - 6
axiom Jake_has_3_more_peaches_than_Jill : Jk = Jl + 3

-- Define the theorem to prove Jill has 5 peaches
theorem Jill_has_5_peaches (S Jk Jl : ℕ) 
  (h1 : S = 14) 
  (h2 : Jk = S - 6)
  (h3 : Jk = Jl + 3) : 
  Jl = 5 := 
by
  sorry

end NUMINAMATH_GPT_Jill_has_5_peaches_l133_13316


namespace NUMINAMATH_GPT_all_statements_imply_implication_l133_13358

variables (p q r : Prop)

theorem all_statements_imply_implication :
  (p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (¬ p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (p ∧ ¬ q ∧ ¬ r → ((p → q) → r)) ∧
  (¬ p ∧ q ∧ r → ((p → q) → r)) :=
by { sorry }

end NUMINAMATH_GPT_all_statements_imply_implication_l133_13358


namespace NUMINAMATH_GPT_sqrt_square_eq_17_l133_13323

theorem sqrt_square_eq_17 :
  (Real.sqrt 17) ^ 2 = 17 :=
sorry

end NUMINAMATH_GPT_sqrt_square_eq_17_l133_13323


namespace NUMINAMATH_GPT_part1_part2_l133_13350

-- Define the first part of the problem
theorem part1 (a b : ℝ) :
  (∀ x : ℝ, |x^2 + a * x + b| ≤ 2 * |x - 4| * |x + 2|) → (a = -2 ∧ b = -8) :=
sorry

-- Define the second part of the problem
theorem part2 (a b m : ℝ) :
  (∀ x : ℝ, x > 1 → x^2 + a * x + b ≥ (m + 2) * x - m - 15) → m ≤ 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l133_13350


namespace NUMINAMATH_GPT_arrange_descending_order_l133_13327

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem arrange_descending_order : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_arrange_descending_order_l133_13327


namespace NUMINAMATH_GPT_lines_intersection_l133_13345

def intersection_point_of_lines
  (t u : ℚ)
  (x₁ y₁ x₂ y₂ : ℚ)
  (x y : ℚ) : Prop := 
  ∃ (t u : ℚ),
    (x₁ + 3*t = 7 + 6*u) ∧
    (y₁ - 4*t = -5 + 3*u) ∧
    (x = x₁ + 3 * t) ∧ 
    (y = y₁ - 4 * t)

theorem lines_intersection :
  ∀ (t u : ℚ),
    intersection_point_of_lines t u 3 2 7 (-5) (87/11) (-50/11) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersection_l133_13345


namespace NUMINAMATH_GPT_binom_25_7_l133_13307

theorem binom_25_7 :
  (Nat.choose 23 5 = 33649) →
  (Nat.choose 23 6 = 42504) →
  (Nat.choose 23 7 = 33649) →
  Nat.choose 25 7 = 152306 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_binom_25_7_l133_13307


namespace NUMINAMATH_GPT_find_x_81_9_729_l133_13311

theorem find_x_81_9_729
  (x : ℝ)
  (h : (81 : ℝ)^(x-2) / (9 : ℝ)^(x-2) = (729 : ℝ)^(2*x-1)) :
  x = 1/5 :=
sorry

end NUMINAMATH_GPT_find_x_81_9_729_l133_13311


namespace NUMINAMATH_GPT_union_A_B_l133_13398

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | x^2 - 1 < 0}
def A_union_B := {x : ℝ | (Real.log x ≤ 0) ∨ (x^2 - 1 < 0)}

theorem union_A_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  -- proof to be added
  sorry

end NUMINAMATH_GPT_union_A_B_l133_13398


namespace NUMINAMATH_GPT_parametric_inclination_l133_13312

noncomputable def angle_of_inclination (x y : ℝ) : ℝ := 50

theorem parametric_inclination (t : ℝ) (x y : ℝ) :
  x = t * Real.sin 40 → y = -1 + t * Real.cos 40 → angle_of_inclination x y = 50 :=
by
  intros hx hy
  -- This is where the proof would go, but we skip it.
  sorry

end NUMINAMATH_GPT_parametric_inclination_l133_13312


namespace NUMINAMATH_GPT_perpendicular_vectors_l133_13395

def vec := ℝ × ℝ

def dot_product (a b : vec) : ℝ :=
  a.1 * b.1 + a.2 * b.2

variables (m : ℝ)
def a : vec := (1, 2)
def b : vec := (m, 1)

theorem perpendicular_vectors (h : dot_product a (b m) = 0) : m = -2 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_l133_13395


namespace NUMINAMATH_GPT_range_of_b_no_common_points_l133_13374

theorem range_of_b_no_common_points (b : ℝ) :
  ¬ (∃ x : ℝ, 2 ^ |x| - 1 = b) ↔ b < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_no_common_points_l133_13374


namespace NUMINAMATH_GPT_average_speed_l133_13379

theorem average_speed :
  ∀ (initial_odometer final_odometer total_time : ℕ), 
    initial_odometer = 2332 →
    final_odometer = 2772 →
    total_time = 8 →
    (final_odometer - initial_odometer) / total_time = 55 :=
by
  intros initial_odometer final_odometer total_time h_initial h_final h_time
  sorry

end NUMINAMATH_GPT_average_speed_l133_13379


namespace NUMINAMATH_GPT_all_of_the_above_were_used_as_money_l133_13341

-- Defining the conditions that each type was used as money
def gold_used_as_money : Prop := True
def stones_used_as_money : Prop := True
def horses_used_as_money : Prop := True
def dried_fish_used_as_money : Prop := True
def mollusk_shells_used_as_money : Prop := True

-- The statement to prove
theorem all_of_the_above_were_used_as_money :
  gold_used_as_money ∧
  stones_used_as_money ∧
  horses_used_as_money ∧
  dried_fish_used_as_money ∧
  mollusk_shells_used_as_money ↔
  (∀ x, (x = "gold" ∨ x = "stones" ∨ x = "horses" ∨ x = "dried fish" ∨ x = "mollusk shells") → 
  (x = "gold" ∧ gold_used_as_money) ∨ 
  (x = "stones" ∧ stones_used_as_money) ∨ 
  (x = "horses" ∧ horses_used_as_money) ∨ 
  (x = "dried fish" ∧ dried_fish_used_as_money) ∨ 
  (x = "mollusk shells" ∧ mollusk_shells_used_as_money)) :=
by
  sorry

end NUMINAMATH_GPT_all_of_the_above_were_used_as_money_l133_13341


namespace NUMINAMATH_GPT_shaded_area_l133_13334

/-- Prove that the shaded area of a shape formed by removing four right triangles of legs 2 from each corner of a 6 × 6 square is equal to 28 square units -/
theorem shaded_area (a b c d : ℕ) (square_side_length : ℕ) (triangle_leg_length : ℕ)
  (h1 : square_side_length = 6)
  (h2 : triangle_leg_length = 2)
  (h3 : a = 1)
  (h4 : b = 2)
  (h5 : c = b)
  (h6 : d = 4*a) : 
  a * square_side_length * square_side_length - d * (b * b / 2) = 28 := 
sorry

end NUMINAMATH_GPT_shaded_area_l133_13334


namespace NUMINAMATH_GPT_pentagon_area_l133_13381

variable (a b c d e : ℕ)
variable (r s : ℕ)

-- Given conditions
axiom H₁: a = 14
axiom H₂: b = 35
axiom H₃: c = 42
axiom H₄: d = 14
axiom H₅: e = 35
axiom H₆: r = 21
axiom H₇: s = 28
axiom H₈: r^2 + s^2 = e^2

-- Question: Prove that the area of the pentagon is 1176
theorem pentagon_area : b * c - (1 / 2) * r * s = 1176 := 
by 
  sorry

end NUMINAMATH_GPT_pentagon_area_l133_13381


namespace NUMINAMATH_GPT_evaluate_expression_l133_13390

theorem evaluate_expression : 3002^3 - 3001 * 3002^2 - 3001^2 * 3002 + 3001^3 + 1 = 6004 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l133_13390


namespace NUMINAMATH_GPT_proof_part_a_l133_13377

variable {α : Type} [LinearOrder α]

structure ConvexQuadrilateral (α : Type) :=
(a b c d : α)
(a'b'c'd' : α)
(ab_eq_a'b' : α)
(bc_eq_b'c' : α)
(cd_eq_c'd' : α)
(da_eq_d'a' : α)
(angle_A_gt_angle_A' : Prop)
(angle_B_lt_angle_B' : Prop)
(angle_C_gt_angle_C' : Prop)
(angle_D_lt_angle_D' : Prop)

theorem proof_part_a (Quad : ConvexQuadrilateral ℝ) : 
  Quad.angle_A_gt_angle_A' → 
  Quad.angle_B_lt_angle_B' ∧ Quad.angle_C_gt_angle_C' ∧ Quad.angle_D_lt_angle_D' := sorry

end NUMINAMATH_GPT_proof_part_a_l133_13377


namespace NUMINAMATH_GPT_middle_of_7_consecutive_nat_sum_63_l133_13394

theorem middle_of_7_consecutive_nat_sum_63 (x : ℕ) (h : 7 * x = 63) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_middle_of_7_consecutive_nat_sum_63_l133_13394


namespace NUMINAMATH_GPT_days_vacuuming_l133_13304

theorem days_vacuuming (V : ℕ) (h1 : ∀ V, 130 = 30 * V + 40) : V = 3 :=
by
    have eq1 : 130 = 30 * V + 40 := h1 V
    sorry

end NUMINAMATH_GPT_days_vacuuming_l133_13304


namespace NUMINAMATH_GPT_winning_candidate_percentage_is_57_l133_13305

def candidate_votes : List ℕ := [1136, 7636, 11628]

def total_votes : ℕ := candidate_votes.sum

def winning_votes : ℕ := candidate_votes.maximum?.getD 0

def winning_percentage (votes : ℕ) (total : ℕ) : ℚ :=
  (votes * 100) / total

theorem winning_candidate_percentage_is_57 :
  winning_percentage winning_votes total_votes = 57 := by
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_is_57_l133_13305


namespace NUMINAMATH_GPT_apples_per_sandwich_l133_13372

-- Define the conditions
def sam_sandwiches_per_day : Nat := 10
def days_in_week : Nat := 7
def total_apples_in_week : Nat := 280

-- Calculate total sandwiches in a week
def total_sandwiches_in_week := sam_sandwiches_per_day * days_in_week

-- Prove that Sam eats 4 apples for each sandwich
theorem apples_per_sandwich : total_apples_in_week / total_sandwiches_in_week = 4 :=
  by
    sorry

end NUMINAMATH_GPT_apples_per_sandwich_l133_13372


namespace NUMINAMATH_GPT_greatest_integer_e_minus_5_l133_13314

theorem greatest_integer_e_minus_5 (e : ℝ) (h : 2 < e ∧ e < 3) : ⌊e - 5⌋ = -3 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_e_minus_5_l133_13314


namespace NUMINAMATH_GPT_find_power_of_4_l133_13385

theorem find_power_of_4 (x : Nat) : 
  (2 * x + 5 + 2 = 29) -> 
  (x = 11) :=
by
  sorry

end NUMINAMATH_GPT_find_power_of_4_l133_13385


namespace NUMINAMATH_GPT_minimum_value_l133_13366

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) :
  (∃ (m : ℝ), (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 1 → m ≤ (y / x + 1 / y)) ∧
   m = 3 ∧ (∀ (x : ℝ), 0 < x → 0 < (1 - x) → (1 - x) + x = 1 → (y / x + 1 / y = m) ↔ x = 1 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l133_13366


namespace NUMINAMATH_GPT_find_BD_in_triangle_l133_13313

theorem find_BD_in_triangle (A B C D : Type)
  (distance_AC : Float) (distance_BC : Float)
  (distance_AD : Float) (distance_CD : Float)
  (hAC : distance_AC = 10)
  (hBC : distance_BC = 10)
  (hAD : distance_AD = 12)
  (hCD : distance_CD = 5) :
  ∃ (BD : Float), BD = 6.85435 :=
by 
  sorry

end NUMINAMATH_GPT_find_BD_in_triangle_l133_13313


namespace NUMINAMATH_GPT_number_of_multiples_of_3003_l133_13360

theorem number_of_multiples_of_3003 (i j : ℕ) (h : 0 ≤ i ∧ i < j ∧ j ≤ 199): 
  (∃ n : ℕ, n = 3003 * k ∧ n = 10^j - 10^i) → 
  (number_of_solutions = 1568) :=
sorry

end NUMINAMATH_GPT_number_of_multiples_of_3003_l133_13360


namespace NUMINAMATH_GPT_remainder_when_add_13_l133_13396

theorem remainder_when_add_13 (x : ℤ) (h : x % 82 = 5) : (x + 13) % 41 = 18 :=
sorry

end NUMINAMATH_GPT_remainder_when_add_13_l133_13396


namespace NUMINAMATH_GPT_arlene_hike_distance_l133_13330

-- Define the conditions: Arlene's pace and the time she spent hiking
def arlene_pace : ℝ := 4 -- miles per hour
def arlene_time_hiking : ℝ := 6 -- hours

-- Define the problem statement and provide the mathematical proof
theorem arlene_hike_distance :
  arlene_pace * arlene_time_hiking = 24 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_arlene_hike_distance_l133_13330


namespace NUMINAMATH_GPT_find_a_b_minimum_value_l133_13352

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x^2

/-- Given the function y = f(x) = ax^3 + bx^2, when x = 1, it has a maximum value of 3 -/
def condition1 (a b : ℝ) : Prop :=
  f 1 a b = 3 ∧ (3 * a + 2 * b = 0)

/-- Find the values of the real numbers a and b -/
theorem find_a_b : ∃ (a b : ℝ), condition1 a b :=
sorry

/-- Find the minimum value of the function -/
theorem minimum_value : ∀ (a b : ℝ), condition1 a b → (∃ x_min, ∀ x, f x a b ≥ f x_min a b) :=
sorry

end NUMINAMATH_GPT_find_a_b_minimum_value_l133_13352


namespace NUMINAMATH_GPT_thirty_ml_of_one_liter_is_decimal_fraction_l133_13389

-- We define the known conversion rule between liters and milliliters.
def liter_to_ml := 1000

-- We define the volume in milliliters that we are considering.
def volume_ml := 30

-- We state the main theorem which asserts that 30 ml of a liter is equal to the decimal fraction 0.03.
theorem thirty_ml_of_one_liter_is_decimal_fraction : (volume_ml / (liter_to_ml : ℝ)) = 0.03 := by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_thirty_ml_of_one_liter_is_decimal_fraction_l133_13389


namespace NUMINAMATH_GPT_possible_pairs_copies_each_key_min_drawers_l133_13384

-- Define the number of distinct keys
def num_keys : ℕ := 10

-- Define the function to calculate the number of pairs
def num_pairs (n : ℕ) := n * (n - 1) / 2

-- Theorem for the first question
theorem possible_pairs : num_pairs num_keys = 45 :=
by sorry

-- Define the number of copies needed for each key
def copies_needed (n : ℕ) := n - 1

-- Theorem for the second question
theorem copies_each_key : copies_needed num_keys = 9 :=
by sorry

-- Define the minimum number of drawers Fernando needs to open
def min_drawers_to_open (n : ℕ) := num_pairs n - (n - 1) + 1

-- Theorem for the third question
theorem min_drawers : min_drawers_to_open num_keys = 37 :=
by sorry

end NUMINAMATH_GPT_possible_pairs_copies_each_key_min_drawers_l133_13384


namespace NUMINAMATH_GPT_line_tangent_to_curve_iff_a_zero_l133_13301

noncomputable def f (x : ℝ) := Real.sin (2 * x)
noncomputable def l (x a : ℝ) := 2 * x + a

theorem line_tangent_to_curve_iff_a_zero (a : ℝ) :
  (∃ x₀ : ℝ, deriv f x₀ = 2 ∧ f x₀ = l x₀ a) → a = 0 :=
sorry

end NUMINAMATH_GPT_line_tangent_to_curve_iff_a_zero_l133_13301


namespace NUMINAMATH_GPT_students_in_section_A_l133_13308

theorem students_in_section_A (x : ℕ) (h1 : (40 : ℝ) * x + 44 * 35 = 37.25 * (x + 44)) : x = 36 :=
by
  sorry

end NUMINAMATH_GPT_students_in_section_A_l133_13308


namespace NUMINAMATH_GPT_point_A_outside_circle_l133_13319

noncomputable def circle_radius := 6
noncomputable def distance_OA := 8

theorem point_A_outside_circle : distance_OA > circle_radius :=
by
  -- Solution will go here
  sorry

end NUMINAMATH_GPT_point_A_outside_circle_l133_13319


namespace NUMINAMATH_GPT_pirates_treasure_l133_13325

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end NUMINAMATH_GPT_pirates_treasure_l133_13325


namespace NUMINAMATH_GPT_uv_square_l133_13355

theorem uv_square (u v : ℝ) (h1 : u * (u + v) = 50) (h2 : v * (u + v) = 100) : (u + v)^2 = 150 := by
  sorry

end NUMINAMATH_GPT_uv_square_l133_13355


namespace NUMINAMATH_GPT_polynomial_q_value_l133_13370

theorem polynomial_q_value :
  ∀ (p q d : ℝ),
    (d = 6) →
    (-p / 3 = -d) →
    (1 + p + q + d = - d) →
    q = -31 :=
by sorry

end NUMINAMATH_GPT_polynomial_q_value_l133_13370


namespace NUMINAMATH_GPT_children_ticket_price_difference_l133_13332

noncomputable def regular_ticket_price : ℝ := 9
noncomputable def total_amount_given : ℝ := 2 * 20
noncomputable def total_change_received : ℝ := 1
noncomputable def num_adults : ℕ := 2
noncomputable def num_children : ℕ := 3
noncomputable def total_cost_of_tickets : ℝ := total_amount_given - total_change_received
noncomputable def children_ticket_cost := (total_cost_of_tickets - num_adults * regular_ticket_price) / num_children

theorem children_ticket_price_difference :
  (regular_ticket_price - children_ticket_cost) = 2 := by
  sorry

end NUMINAMATH_GPT_children_ticket_price_difference_l133_13332


namespace NUMINAMATH_GPT_equal_work_women_l133_13353

-- Let W be the amount of work one woman can do in a day.
-- Let M be the amount of work one man can do in a day.
-- Let x be the number of women who do the same amount of work as 5 men.

def numWomenEqualWork (W : ℝ) (M : ℝ) (x : ℝ) : Prop :=
  5 * M = x * W

theorem equal_work_women (W M x : ℝ) 
  (h1 : numWomenEqualWork W M x)
  (h2 : (3 * M + 5 * W) * 10 = (7 * W) * 14) :
  x = 8 :=
sorry

end NUMINAMATH_GPT_equal_work_women_l133_13353


namespace NUMINAMATH_GPT_sophia_fraction_of_pie_l133_13328

theorem sophia_fraction_of_pie
  (weight_fridge : ℕ) (weight_eaten : ℕ)
  (h1 : weight_fridge = 1200)
  (h2 : weight_eaten = 240) :
  (weight_eaten : ℚ) / ((weight_fridge + weight_eaten : ℚ)) = (1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_sophia_fraction_of_pie_l133_13328


namespace NUMINAMATH_GPT_right_triangle_area_l133_13349

theorem right_triangle_area (x y : ℝ) 
  (h1 : x + y = 4) 
  (h2 : x^2 + y^2 = 9) : 
  (1/2) * x * y = 7 / 4 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l133_13349


namespace NUMINAMATH_GPT_parabola_coefficients_l133_13351

theorem parabola_coefficients (a b c : ℝ) 
  (h_vertex : ∀ x, a * (x - 4) * (x - 4) + 3 = a * x * x + b * x + c) 
  (h_pass_point : 1 = a * (2 - 4) * (2 - 4) + 3) :
  (a = -1/2) ∧ (b = 4) ∧ (c = -5) :=
by
  sorry

end NUMINAMATH_GPT_parabola_coefficients_l133_13351


namespace NUMINAMATH_GPT_max_min_diff_c_l133_13320

theorem max_min_diff_c (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  ∃ c_max c_min, 
  (∀ c', (a + b + c' = 3 ∧ a^2 + b^2 + c'^2 = 18) → c_min ≤ c' ∧ c' ≤ c_max) 
  ∧ (c_max - c_min = 6) :=
  sorry

end NUMINAMATH_GPT_max_min_diff_c_l133_13320


namespace NUMINAMATH_GPT_chocolate_cost_l133_13309

def cost_of_chocolates (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

theorem chocolate_cost : cost_of_chocolates 30 8 450 = 120 :=
by
  -- The proof is not needed per the instructions
  sorry

end NUMINAMATH_GPT_chocolate_cost_l133_13309


namespace NUMINAMATH_GPT_polynomial_value_at_minus_two_l133_13315

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem polynomial_value_at_minus_two :
  f (-2) = -1 :=
by sorry

end NUMINAMATH_GPT_polynomial_value_at_minus_two_l133_13315


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l133_13329

-- Define the problem statement in Lean 4
theorem geometric_sequence_third_term :
  ∃ r : ℝ, (a = 1024) ∧ (a_5 = 128) ∧ (a_5 = a * r^4) ∧ 
  (a_3 = a * r^2) ∧ (a_3 = 256) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l133_13329


namespace NUMINAMATH_GPT_angle_P_in_quadrilateral_l133_13321

theorem angle_P_in_quadrilateral : 
  ∀ (P Q R S : ℝ), (P = 3 * Q) → (P = 4 * R) → (P = 6 * S) → (P + Q + R + S = 360) → P = 206 := 
by
  intros P Q R S hP1 hP2 hP3 hSum
  sorry

end NUMINAMATH_GPT_angle_P_in_quadrilateral_l133_13321


namespace NUMINAMATH_GPT_find_third_number_l133_13392

noncomputable def averageFirstSet (x : ℝ) : ℝ := (20 + 40 + x) / 3
noncomputable def averageSecondSet : ℝ := (10 + 70 + 16) / 3

theorem find_third_number (x : ℝ) (h : averageFirstSet x = averageSecondSet + 8) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_third_number_l133_13392


namespace NUMINAMATH_GPT_probability_beautiful_equation_l133_13335

def tetrahedron_faces : Set ℕ := {1, 2, 3, 4}

def is_beautiful_equation (a b : ℕ) : Prop :=
    ∃ m ∈ tetrahedron_faces, a = m + 1 ∨ a = m + 2 ∨ a = m + 3 ∨ a = m + 4 ∧ b = m * (a - m)

theorem probability_beautiful_equation : 
  (∃ a b1 b2, is_beautiful_equation a b1 ∧ is_beautiful_equation a b2) ∧
  (∃ a b1 b2, tetrahedron_faces ⊆ {a} ∧ tetrahedron_faces ⊆ {b1} ∧ tetrahedron_faces ⊆ {b2}) :=
  sorry

end NUMINAMATH_GPT_probability_beautiful_equation_l133_13335


namespace NUMINAMATH_GPT_value_of_x_plus_inv_x_l133_13354

theorem value_of_x_plus_inv_x (x : ℝ) (hx : x ≠ 0) (t : ℝ) (ht : t = x^2 + (1 / x)^2) : x + (1 / x) = 5 :=
by
  have ht_val : t = 23 := by
    rw [ht] -- assuming t = 23 by condition
    sorry -- proof continuation placeholder

  -- introduce y and relate it to t
  let y := x + (1 / x)

  -- express t in terms of y and handle the algebra:
  have t_expr : t = y^2 - 2 := by
    sorry -- proof continuation placeholder

  -- show that y^2 = 25 and therefore y = 5 as the only valid solution:
  have y_val : y = 5 := by
    sorry -- proof continuation placeholder

  -- hence, the required value is found:
  exact y_val

end NUMINAMATH_GPT_value_of_x_plus_inv_x_l133_13354


namespace NUMINAMATH_GPT_simplify_cbrt_8000_eq_21_l133_13310

theorem simplify_cbrt_8000_eq_21 :
  ∃ (a b : ℕ), a * (b^(1/3)) = 20 * (1^(1/3)) ∧ b = 1 ∧ a + b = 21 :=
by
  sorry

end NUMINAMATH_GPT_simplify_cbrt_8000_eq_21_l133_13310


namespace NUMINAMATH_GPT_greatest_integer_is_8_l133_13337

theorem greatest_integer_is_8 {a b : ℤ} (h_sum : a + b + 8 = 21) : max a (max b 8) = 8 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_is_8_l133_13337


namespace NUMINAMATH_GPT_smallest_value_of_Q_l133_13380

def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - 4*x^2 + 2*x - 3

theorem smallest_value_of_Q :
  min (-10) (min 3 (-2)) = -10 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_smallest_value_of_Q_l133_13380


namespace NUMINAMATH_GPT_man_age_twice_son_age_in_2_years_l133_13356

variable (currentAgeSon : ℕ)
variable (currentAgeMan : ℕ)
variable (Y : ℕ)

-- Given conditions
def sonCurrentAge : Prop := currentAgeSon = 23
def manCurrentAge : Prop := currentAgeMan = currentAgeSon + 25
def manAgeTwiceSonAgeInYYears : Prop := currentAgeMan + Y = 2 * (currentAgeSon + Y)

-- Theorem to prove
theorem man_age_twice_son_age_in_2_years :
  sonCurrentAge currentAgeSon →
  manCurrentAge currentAgeSon currentAgeMan →
  manAgeTwiceSonAgeInYYears currentAgeSon currentAgeMan Y →
  Y = 2 :=
by
  intros h_son_age h_man_age h_age_relation
  sorry

end NUMINAMATH_GPT_man_age_twice_son_age_in_2_years_l133_13356


namespace NUMINAMATH_GPT_find_jessica_almonds_l133_13382

-- Definitions for j (Jessica's almonds) and l (Louise's almonds)
variables (j l : ℕ)
-- Conditions
def condition1 : Prop := l = j - 8
def condition2 : Prop := l = j / 3

theorem find_jessica_almonds (h1 : condition1 j l) (h2 : condition2 j l) : j = 12 :=
by sorry

end NUMINAMATH_GPT_find_jessica_almonds_l133_13382


namespace NUMINAMATH_GPT_number_of_teams_l133_13344

-- Define the problem context
variables (n : ℕ)

-- Define the conditions
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The theorem we want to prove
theorem number_of_teams (h : total_games n = 55) : n = 11 :=
sorry

end NUMINAMATH_GPT_number_of_teams_l133_13344


namespace NUMINAMATH_GPT_age_of_B_l133_13326

-- Define the ages based on the conditions
def A (x : ℕ) : ℕ := 2 * x + 2
def B (x : ℕ) : ℕ := 2 * x
def C (x : ℕ) : ℕ := x

-- The main statement to be proved
theorem age_of_B (x : ℕ) (h : A x + B x + C x = 72) : B 14 = 28 :=
by
  -- we need the proof here but we will put sorry for now
  sorry

end NUMINAMATH_GPT_age_of_B_l133_13326


namespace NUMINAMATH_GPT_task_completion_time_l133_13357

noncomputable def john_work_rate := (1: ℚ) / 20
noncomputable def jane_work_rate := (1: ℚ) / 12
noncomputable def combined_work_rate := john_work_rate + jane_work_rate
noncomputable def time_jane_disposed := 4

theorem task_completion_time :
  (∃ x : ℚ, (combined_work_rate * x + john_work_rate * time_jane_disposed = 1) ∧ (x + time_jane_disposed = 10)) :=
by
  use 6  
  sorry

end NUMINAMATH_GPT_task_completion_time_l133_13357


namespace NUMINAMATH_GPT_loss_percentage_remaining_stock_l133_13364

noncomputable def total_worth : ℝ := 9999.999999999998
def overall_loss : ℝ := 200
def profit_percentage_20 : ℝ := 0.1
def sold_20_percentage : ℝ := 0.2
def remaining_percentage : ℝ := 0.8

theorem loss_percentage_remaining_stock :
  ∃ L : ℝ, 0.8 * total_worth * (L / 100) - 0.02 * total_worth = overall_loss ∧ L = 5 :=
by sorry

end NUMINAMATH_GPT_loss_percentage_remaining_stock_l133_13364


namespace NUMINAMATH_GPT_find_four_numbers_proportion_l133_13375

theorem find_four_numbers_proportion :
  ∃ (a b c d : ℝ), 
  a + d = 14 ∧
  b + c = 11 ∧
  a^2 + b^2 + c^2 + d^2 = 221 ∧
  a * d = b * c ∧
  a = 12 ∧
  b = 8 ∧
  c = 3 ∧
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_four_numbers_proportion_l133_13375


namespace NUMINAMATH_GPT_system_of_equations_solution_l133_13317

variable {x y : ℝ}

theorem system_of_equations_solution
  (h1 : x^2 + x * y * Real.sqrt (x * y) + y^2 = 25)
  (h2 : x^2 - x * y * Real.sqrt (x * y) + y^2 = 9) :
  (x, y) = (1, 4) ∨ (x, y) = (4, 1) ∨ (x, y) = (-1, -4) ∨ (x, y) = (-4, -1) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l133_13317


namespace NUMINAMATH_GPT_reduced_price_per_kg_l133_13318

theorem reduced_price_per_kg (P R : ℝ) (Q : ℝ)
  (h1 : R = 0.80 * P)
  (h2 : Q * P = 1500)
  (h3 : (Q + 10) * R = 1500) : R = 30 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_per_kg_l133_13318


namespace NUMINAMATH_GPT_choose_math_class_representative_l133_13383

def number_of_boys : Nat := 26
def number_of_girls : Nat := 24

theorem choose_math_class_representative : number_of_boys + number_of_girls = 50 := 
by
  sorry

end NUMINAMATH_GPT_choose_math_class_representative_l133_13383


namespace NUMINAMATH_GPT_calc_hash_2_5_3_l133_13373

def operation_hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem calc_hash_2_5_3 : operation_hash 2 5 3 = 1 := by
  sorry

end NUMINAMATH_GPT_calc_hash_2_5_3_l133_13373


namespace NUMINAMATH_GPT_range_of_t_l133_13368

variable (f : ℝ → ℝ) (t : ℝ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_t {f : ℝ → ℝ} {t : ℝ} 
  (Hodd : is_odd f) 
  (Hperiodic : ∀ x, f (x + 5 / 2) = -1 / f x) 
  (Hf1 : f 1 ≥ 1) 
  (Hf2014 : f 2014 = (t + 3) / (t - 3)) : 
  0 ≤ t ∧ t < 3 := by
  sorry

end NUMINAMATH_GPT_range_of_t_l133_13368


namespace NUMINAMATH_GPT_ravioli_to_tortellini_ratio_l133_13387

-- Definitions from conditions
def total_students : ℕ := 800
def ravioli_students : ℕ := 300
def tortellini_students : ℕ := 150

-- Ratio calculation as a theorem
theorem ravioli_to_tortellini_ratio : 2 = ravioli_students / Nat.gcd ravioli_students tortellini_students :=
by
  -- Given the defined values
  have gcd_val : Nat.gcd ravioli_students tortellini_students = 150 := by
    sorry
  have ratio_simp : ravioli_students / 150 = 2 := by
    sorry
  exact ratio_simp

end NUMINAMATH_GPT_ravioli_to_tortellini_ratio_l133_13387


namespace NUMINAMATH_GPT_evaluate_expression_l133_13361

theorem evaluate_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l133_13361


namespace NUMINAMATH_GPT_triangle_side_inequality_l133_13338

theorem triangle_side_inequality (y : ℕ) (h : 3 < y^2 ∧ y^2 < 19) : 
  y = 2 ∨ y = 3 ∨ y = 4 :=
sorry

end NUMINAMATH_GPT_triangle_side_inequality_l133_13338


namespace NUMINAMATH_GPT_problem_statement_l133_13371

theorem problem_statement (a b : ℝ) :
  a^2 + b^2 - a - b - a * b + 0.25 ≥ 0 ∧ (a^2 + b^2 - a - b - a * b + 0.25 = 0 ↔ ((a = 0 ∧ b = 0.5) ∨ (a = 0.5 ∧ b = 0))) :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l133_13371


namespace NUMINAMATH_GPT_fuel_tank_capacity_l133_13343

theorem fuel_tank_capacity (x : ℝ) 
  (h1 : (5 / 6) * x - (2 / 3) * x = 15) : x = 90 :=
sorry

end NUMINAMATH_GPT_fuel_tank_capacity_l133_13343


namespace NUMINAMATH_GPT_train_crossing_time_l133_13333

open Real

noncomputable def time_to_cross_bridge 
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000/3600)
  total_distance / speed_train_ms

theorem train_crossing_time
  (length_train : ℝ) (speed_train_kmh : ℝ) (length_bridge : ℝ)
  (h_length_train : length_train = 160)
  (h_speed_train_kmh : speed_train_kmh = 45)
  (h_length_bridge : length_bridge = 215) :
  time_to_cross_bridge length_train speed_train_kmh length_bridge = 30 :=
sorry

end NUMINAMATH_GPT_train_crossing_time_l133_13333


namespace NUMINAMATH_GPT_percentage_equivalence_l133_13376

theorem percentage_equivalence (x : ℝ) : 0.3 * 0.6 * 0.7 * x = 0.126 * x :=
by
  sorry

end NUMINAMATH_GPT_percentage_equivalence_l133_13376


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l133_13302

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l133_13302


namespace NUMINAMATH_GPT_correct_actual_profit_l133_13399

def profit_miscalculation (calculated_profit actual_profit : ℕ) : Prop :=
  let err1 := 5 * 100  -- Error due to mistaking 3 for 8 in the hundreds place
  let err2 := 3 * 10   -- Error due to mistaking 8 for 5 in the tens place
  actual_profit = calculated_profit - err1 + err2

theorem correct_actual_profit : profit_miscalculation 1320 850 :=
by
  sorry

end NUMINAMATH_GPT_correct_actual_profit_l133_13399


namespace NUMINAMATH_GPT_polygon_sides_eq_13_l133_13331

theorem polygon_sides_eq_13 (n : ℕ) (h : n * (n - 3) = 5 * n) : n = 13 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_13_l133_13331


namespace NUMINAMATH_GPT_tournament_games_count_l133_13365

-- We define the conditions
def number_of_players : ℕ := 6

-- Function to calculate the number of games played in a tournament where each player plays twice with each opponent
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Now we state the theorem
theorem tournament_games_count : total_games number_of_players = 60 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tournament_games_count_l133_13365


namespace NUMINAMATH_GPT_sandwich_percentage_not_vegetables_l133_13303

noncomputable def percentage_not_vegetables (total_weight : ℝ) (vegetable_weight : ℝ) : ℝ :=
  (total_weight - vegetable_weight) / total_weight * 100

theorem sandwich_percentage_not_vegetables :
  percentage_not_vegetables 180 50 = 72.22 :=
by
  sorry

end NUMINAMATH_GPT_sandwich_percentage_not_vegetables_l133_13303


namespace NUMINAMATH_GPT_inequality_solution_inequality_proof_l133_13362

def f (x: ℝ) := |x - 5|

theorem inequality_solution : {x : ℝ | f x + f (x + 2) ≤ 3} = {x | 5 / 2 ≤ x ∧ x ≤ 11 / 2} :=
sorry

theorem inequality_proof (a x : ℝ) (h : a < 0) : f (a * x) - f (5 * a) ≥ a * f x :=
sorry

end NUMINAMATH_GPT_inequality_solution_inequality_proof_l133_13362


namespace NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l133_13342

theorem part_a : (4237 * 27925 ≠ 118275855) :=
by sorry

theorem part_b : (42971064 / 8264 ≠ 5201) :=
by sorry

theorem part_c : (1965^2 ≠ 3761225) :=
by sorry

theorem part_d : (23 ^ 5 ≠ 371293) :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_part_d_l133_13342


namespace NUMINAMATH_GPT_total_candies_is_90_l133_13367

-- Defining the conditions
def boxes_chocolate := 6
def boxes_caramel := 4
def pieces_per_box := 9

-- Defining the total number of boxes
def total_boxes := boxes_chocolate + boxes_caramel

-- Defining the total number of candies
def total_candies := total_boxes * pieces_per_box

-- Theorem stating the proof problem
theorem total_candies_is_90 : total_candies = 90 := by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_candies_is_90_l133_13367


namespace NUMINAMATH_GPT_precision_of_21_658_billion_is_hundred_million_l133_13322

theorem precision_of_21_658_billion_is_hundred_million :
  (21.658 : ℝ) * 10^9 % (10^8) = 0 :=
by
  sorry

end NUMINAMATH_GPT_precision_of_21_658_billion_is_hundred_million_l133_13322


namespace NUMINAMATH_GPT_number_of_pairs_l133_13391

theorem number_of_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m^2 + n < 50) : 
  ∃! p : ℕ, p = 203 := 
sorry

end NUMINAMATH_GPT_number_of_pairs_l133_13391


namespace NUMINAMATH_GPT_cone_cylinder_volume_ratio_l133_13359

theorem cone_cylinder_volume_ratio :
  let π := Real.pi
  let Vcylinder := π * (3:ℝ)^2 * (15:ℝ)
  let Vcone := (1/3:ℝ) * π * (2:ℝ)^2 * (5:ℝ)
  (Vcone / Vcylinder) = (4 / 81) :=
by
  let π := Real.pi
  let r_cylinder := (3:ℝ)
  let h_cylinder := (15:ℝ)
  let r_cone := (2:ℝ)
  let h_cone := (5:ℝ)
  let Vcylinder := π * r_cylinder^2 * h_cylinder
  let Vcone := (1/3:ℝ) * π * r_cone^2 * h_cone
  have h1 : Vcylinder = 135 * π := by sorry
  have h2 : Vcone = (20 / 3) * π := by sorry
  have h3 : (Vcone / Vcylinder) = (4 / 81) := by sorry
  exact h3

end NUMINAMATH_GPT_cone_cylinder_volume_ratio_l133_13359


namespace NUMINAMATH_GPT_non_receivers_after_2020_candies_l133_13300

noncomputable def count_non_receivers (k n : ℕ) : ℕ := 
sorry

theorem non_receivers_after_2020_candies :
  count_non_receivers 73 2020 = 36 :=
sorry

end NUMINAMATH_GPT_non_receivers_after_2020_candies_l133_13300


namespace NUMINAMATH_GPT_pentagon_perimeter_l133_13369

-- Define the side length and number of sides for a regular pentagon
def side_length : ℝ := 5
def num_sides : ℕ := 5

-- Define the perimeter calculation as a constant
def perimeter (side_length : ℝ) (num_sides : ℕ) : ℝ := side_length * num_sides

theorem pentagon_perimeter : perimeter side_length num_sides = 25 := by
  sorry

end NUMINAMATH_GPT_pentagon_perimeter_l133_13369
