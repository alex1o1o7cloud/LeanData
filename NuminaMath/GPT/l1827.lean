import Mathlib

namespace NUMINAMATH_GPT_subtraction_identity_l1827_182751

theorem subtraction_identity : 3.57 - 1.14 - 0.23 = 2.20 := sorry

end NUMINAMATH_GPT_subtraction_identity_l1827_182751


namespace NUMINAMATH_GPT_total_dining_bill_before_tip_l1827_182786

-- Define total number of people
def numberOfPeople : ℕ := 6

-- Define the individual payment
def individualShare : ℝ := 25.48

-- Define the total payment
def totalPayment : ℝ := numberOfPeople * individualShare

-- Define the tip percentage
def tipPercentage : ℝ := 0.10

-- Total payment including tip expressed in terms of the original bill B
def totalPaymentWithTip (B : ℝ) : ℝ := B + B * tipPercentage

-- Prove the total dining bill before the tip
theorem total_dining_bill_before_tip : 
    ∃ B : ℝ, totalPayment = totalPaymentWithTip B ∧ B = 139.89 :=
by
    sorry

end NUMINAMATH_GPT_total_dining_bill_before_tip_l1827_182786


namespace NUMINAMATH_GPT_vertex_on_xaxis_l1827_182709

-- Definition of the parabola equation with vertex on the x-axis
def parabola (x m : ℝ) := x^2 - 8 * x + m

-- The problem statement: show that m = 16 given that the vertex of the parabola is on the x-axis
theorem vertex_on_xaxis (m : ℝ) : ∃ x : ℝ, parabola x m = 0 → m = 16 :=
by
  sorry

end NUMINAMATH_GPT_vertex_on_xaxis_l1827_182709


namespace NUMINAMATH_GPT_construct_unit_segment_l1827_182750

-- Definitions of the problem
variable (a b : ℝ)

-- Parabola definition
def parabola (x : ℝ) : ℝ := x^2 + a * x + b

-- Statement of the problem in Lean 4
theorem construct_unit_segment
  (h : ∃ x y : ℝ, parabola a b x = y) :
  ∃ (u v : ℝ), abs (u - v) = 1 :=
sorry

end NUMINAMATH_GPT_construct_unit_segment_l1827_182750


namespace NUMINAMATH_GPT_dots_not_visible_l1827_182738

def total_dots_on_die : Nat := 21
def number_of_dice : Nat := 4
def total_dots : Nat := number_of_dice * total_dots_on_die
def visible_faces : List Nat := [1, 2, 2, 3, 3, 5, 6]
def sum_visible_faces : Nat := visible_faces.sum

theorem dots_not_visible : total_dots - sum_visible_faces = 62 := by
  sorry

end NUMINAMATH_GPT_dots_not_visible_l1827_182738


namespace NUMINAMATH_GPT_min_lcm_value_l1827_182720

-- Definitions
def gcd_77 (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 77

def lcm_n (a b c d n : ℕ) : Prop :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = n

-- Problem statement
theorem min_lcm_value :
  (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d 27720) ∧
  (∀ n : ℕ, (∃ a b c d : ℕ, gcd_77 a b c d ∧ lcm_n a b c d n) → 27720 ≤ n) :=
sorry

end NUMINAMATH_GPT_min_lcm_value_l1827_182720


namespace NUMINAMATH_GPT_red_grapes_more_than_three_times_green_l1827_182710

-- Definitions from conditions
variables (G R B : ℕ)
def condition1 := R = 3 * G + (R - 3 * G)
def condition2 := B = G - 5
def condition3 := R + G + B = 102
def condition4 := R = 67

-- The proof problem
theorem red_grapes_more_than_three_times_green : (R = 67) ∧ (R + G + (G - 5) = 102) ∧ (R = 3 * G + (R - 3 * G)) → R - 3 * G = 7 :=
by sorry

end NUMINAMATH_GPT_red_grapes_more_than_three_times_green_l1827_182710


namespace NUMINAMATH_GPT_ptolemys_inequality_l1827_182761

variable {A B C D : Type} [OrderedRing A]
variable (AB BC CD DA AC BD : A)

/-- Ptolemy's inequality for a quadrilateral -/
theorem ptolemys_inequality 
  (AB_ BC_ CD_ DA_ AC_ BD_ : A) :
  AC * BD ≤ AB * CD + BC * AD :=
  sorry

end NUMINAMATH_GPT_ptolemys_inequality_l1827_182761


namespace NUMINAMATH_GPT_inequality_solution_l1827_182708

theorem inequality_solution (x : ℤ) (h : x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1) : x - 1 ≥ 0 ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1827_182708


namespace NUMINAMATH_GPT_sum_of_possible_values_l1827_182736

theorem sum_of_possible_values (A B : ℕ) 
  (hA1 : A < 10) (hA2 : 0 < A) (hB1 : B < 10) (hB2 : 0 < B)
  (h1 : 3 / 12 < A / 12) (h2 : A / 12 < 7 / 12)
  (h3 : 1 / 10 < 1 / B) (h4 : 1 / B < 1 / 3) :
  3 + 6 = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_l1827_182736


namespace NUMINAMATH_GPT_mark_pond_depth_l1827_182707

def depth_of_Peter_pond := 5

def depth_of_Mark_pond := 3 * depth_of_Peter_pond + 4

theorem mark_pond_depth : depth_of_Mark_pond = 19 := by
  sorry

end NUMINAMATH_GPT_mark_pond_depth_l1827_182707


namespace NUMINAMATH_GPT_enclosure_largest_side_l1827_182755

theorem enclosure_largest_side (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 3600) : l = 60 :=
by
  sorry

end NUMINAMATH_GPT_enclosure_largest_side_l1827_182755


namespace NUMINAMATH_GPT_find_ratio_l1827_182743

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a n > 0 ∧ a (n+1) / a n = a 1 / a 0

def forms_arithmetic_sequence (a1 a3_half a2_times_two : ℝ) : Prop :=
  a3_half = (a1 + a2_times_two) / 2

theorem find_ratio (a : ℕ → ℝ) (h_geom : is_geometric_sequence a)
  (h_arith : forms_arithmetic_sequence (a 1) (1/2 * a 3) (2 * a 2)) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_ratio_l1827_182743


namespace NUMINAMATH_GPT_local_minimum_f_when_k2_l1827_182705

noncomputable def f (k : ℕ) (x : ℝ) : ℝ := (Real.exp x - 1) * (x - 1) ^ k

theorem local_minimum_f_when_k2 : ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f 2 x ≥ f 2 1 :=
by
  -- the question asks to prove that the function attains a local minimum at x = 1 when k = 2
  sorry

end NUMINAMATH_GPT_local_minimum_f_when_k2_l1827_182705


namespace NUMINAMATH_GPT_probability_colors_match_l1827_182767

section ProbabilityJellyBeans

structure JellyBeans where
  green : ℕ
  blue : ℕ
  red : ℕ

def total_jellybeans (jb : JellyBeans) : ℕ :=
  jb.green + jb.blue + jb.red

-- Define the situation using structures
def lila_jellybeans : JellyBeans := { green := 1, blue := 1, red := 1 }
def max_jellybeans : JellyBeans := { green := 2, blue := 1, red := 3 }

-- Define probabilities
noncomputable def probability (count : ℕ) (total : ℕ) : ℚ :=
  if total = 0 then 0 else (count : ℚ) / (total : ℚ)

-- Main theorem
theorem probability_colors_match :
  probability lila_jellybeans.green (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.green (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.blue (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.blue (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.red (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.red (total_jellybeans max_jellybeans) = 1 / 3 :=
by sorry

end ProbabilityJellyBeans

end NUMINAMATH_GPT_probability_colors_match_l1827_182767


namespace NUMINAMATH_GPT_inequality_solution_l1827_182752

open Set

noncomputable def solution_set := { x : ℝ | 5 - x^2 > 4 * x }

theorem inequality_solution :
  solution_set = { x : ℝ | -5 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1827_182752


namespace NUMINAMATH_GPT_inradius_semicircle_relation_l1827_182764

theorem inradius_semicircle_relation 
  (a b c : ℝ)
  (h_acute: a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)
  (S : ℝ)
  (p : ℝ)
  (r : ℝ)
  (ra rb rc : ℝ)
  (h_def_semi_perim : p = (a + b + c) / 2)
  (h_area : S = p * r)
  (h_ra : ra = (2 * S) / (b + c))
  (h_rb : rb = (2 * S) / (a + c))
  (h_rc : rc = (2 * S) / (a + b)) :
  2 / r = 1 / ra + 1 / rb + 1 / rc :=
by
  sorry

end NUMINAMATH_GPT_inradius_semicircle_relation_l1827_182764


namespace NUMINAMATH_GPT_negation_of_proposition_l1827_182770

theorem negation_of_proposition {c : ℝ} (h : ∃ (c : ℝ), c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) :
  ∀ (c : ℝ), c > 0 → ¬ ∃ x : ℝ, x^2 - x + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1827_182770


namespace NUMINAMATH_GPT_complement_union_intersection_l1827_182780

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

theorem complement_union_intersection :
  (compl (A ∪ B) = {x | x ≤ 2 ∨ 9 ≤ x}) ∧
  (compl (A ∩ B) = {x | x < 3 ∨ 5 ≤ x}) :=
by
  sorry

end NUMINAMATH_GPT_complement_union_intersection_l1827_182780


namespace NUMINAMATH_GPT_find_a_b_sum_l1827_182799

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 6 * x - 6

theorem find_a_b_sum (a b : ℝ)
  (h1 : f a = 1)
  (h2 : f b = -5) :
  a + b = 2 :=
  sorry

end NUMINAMATH_GPT_find_a_b_sum_l1827_182799


namespace NUMINAMATH_GPT_km_to_leaps_l1827_182790

theorem km_to_leaps (a b c d e f : ℕ) :
  (2 * a) * strides = (3 * b) * leaps →
  (4 * c) * dashes = (5 * d) * strides →
  (6 * e) * dashes = (7 * f) * kilometers →
  1 * kilometers = (90 * b * d * e) / (56 * a * c * f) * leaps :=
by
  -- Using the given conditions to derive the answer
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_km_to_leaps_l1827_182790


namespace NUMINAMATH_GPT_solution_set_abs_inequality_l1827_182797

theorem solution_set_abs_inequality (x : ℝ) :
  |2 * x + 1| < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_abs_inequality_l1827_182797


namespace NUMINAMATH_GPT_solve_for_m_l1827_182739

theorem solve_for_m (m α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_m_l1827_182739


namespace NUMINAMATH_GPT_frac_pattern_2_11_frac_pattern_general_l1827_182773

theorem frac_pattern_2_11 :
  (2 / 11) = (1 / 6) + (1 / 66) :=
sorry

theorem frac_pattern_general (n : ℕ) (hn : n ≥ 3) :
  (2 / (2 * n - 1)) = (1 / n) + (1 / (n * (2 * n - 1))) :=
sorry

end NUMINAMATH_GPT_frac_pattern_2_11_frac_pattern_general_l1827_182773


namespace NUMINAMATH_GPT_minimum_k_conditions_l1827_182706

theorem minimum_k_conditions (k : ℝ) :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → (|a - b| ≤ k ∨ |1/a - 1/b| ≤ k)) ↔ k = 3/2 :=
sorry

end NUMINAMATH_GPT_minimum_k_conditions_l1827_182706


namespace NUMINAMATH_GPT_rectangle_width_l1827_182744

theorem rectangle_width (side_length_square : ℕ) (length_rectangle : ℕ) (area_equal : side_length_square * side_length_square = length_rectangle * w) : w = 4 := by
  sorry

end NUMINAMATH_GPT_rectangle_width_l1827_182744


namespace NUMINAMATH_GPT_trig_identity_l1827_182784

theorem trig_identity (f : ℝ → ℝ) (ϕ : ℝ) (h₁ : ∀ x, f x = 2 * Real.sin (2 * x + ϕ)) (h₂ : 0 < ϕ) (h₃ : ϕ < π) (h₄ : f 0 = 1) :
  f ϕ = 2 :=
sorry

end NUMINAMATH_GPT_trig_identity_l1827_182784


namespace NUMINAMATH_GPT_parallel_lines_a_values_l1827_182713

theorem parallel_lines_a_values (a : Real) : 
  (∃ k : Real, 2 = k * a ∧ -a = k * (-8)) ↔ (a = 4 ∨ a = -4) := sorry

end NUMINAMATH_GPT_parallel_lines_a_values_l1827_182713


namespace NUMINAMATH_GPT_point_direction_form_eq_l1827_182758

-- Define the conditions
def point := (1, 2)
def direction_vector := (3, -4)

-- Define a function to represent the line equation based on point and direction
def line_equation (x y : ℝ) : Prop :=
  (x - point.1) / direction_vector.1 = (y - point.2) / direction_vector.2

-- State the theorem
theorem point_direction_form_eq (x y : ℝ) :
  (x - 1) / 3 = (y - 2) / -4 →
  line_equation x y :=
sorry

end NUMINAMATH_GPT_point_direction_form_eq_l1827_182758


namespace NUMINAMATH_GPT_add_fractions_l1827_182745

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end NUMINAMATH_GPT_add_fractions_l1827_182745


namespace NUMINAMATH_GPT_percent_savings_per_roll_l1827_182712

theorem percent_savings_per_roll 
  (cost_case : ℕ := 900) -- In cents, equivalent to $9
  (cost_individual : ℕ := 100) -- In cents, equivalent to $1
  (num_rolls : ℕ := 12) :
  (cost_individual - (cost_case / num_rolls)) * 100 / cost_individual = 25 := 
sorry

end NUMINAMATH_GPT_percent_savings_per_roll_l1827_182712


namespace NUMINAMATH_GPT_ice_palace_steps_l1827_182703

theorem ice_palace_steps (time_for_20_steps total_time : ℕ) (h1 : time_for_20_steps = 120) (h2 : total_time = 180) : 
  total_time * 20 / time_for_20_steps = 30 := by
  have time_per_step : ℕ := time_for_20_steps / 20
  have total_steps : ℕ := total_time / time_per_step
  sorry

end NUMINAMATH_GPT_ice_palace_steps_l1827_182703


namespace NUMINAMATH_GPT_original_price_of_stamp_l1827_182796

theorem original_price_of_stamp (original_price : ℕ) (h : original_price * (1 / 5 : ℚ) = 6) : original_price = 30 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_stamp_l1827_182796


namespace NUMINAMATH_GPT_bill_score_l1827_182747

theorem bill_score
  (J B S : ℕ)
  (h1 : B = J + 20)
  (h2 : B = S / 2)
  (h3 : J + B + S = 160) : 
  B = 45 := 
by 
  sorry

end NUMINAMATH_GPT_bill_score_l1827_182747


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l1827_182775

theorem relationship_between_a_and_b 
  (x a b : ℝ)
  (hx : 0 < x)
  (ha : 0 < a)
  (hb : 0 < b)
  (hax : a^x < b^x) 
  (hbx : b^x < 1) : 
  a < b ∧ b < 1 := 
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l1827_182775


namespace NUMINAMATH_GPT_Suresh_meeting_time_l1827_182721

theorem Suresh_meeting_time :
  let C := 726
  let v1 := 75
  let v2 := 62.5
  C / (v1 + v2) = 5.28 := by
  sorry

end NUMINAMATH_GPT_Suresh_meeting_time_l1827_182721


namespace NUMINAMATH_GPT_not_approximately_equal_exp_l1827_182765

noncomputable def multinomial_approximation (n k₁ k₂ k₃ k₄ k₅ : ℕ) : ℝ :=
  (n.factorial : ℝ) / ((k₁.factorial : ℝ) * (k₂.factorial : ℝ) * (k₃.factorial : ℝ) * (k₄.factorial : ℝ) * (k₅.factorial : ℝ))

theorem not_approximately_equal_exp (e : ℝ) (h1 : e > 0) :
  e ^ 2737 ≠ multinomial_approximation 1000 70 270 300 220 140 :=
by 
  sorry  

end NUMINAMATH_GPT_not_approximately_equal_exp_l1827_182765


namespace NUMINAMATH_GPT_find_k_value_l1827_182701

theorem find_k_value (a : ℕ → ℕ) (k : ℕ) (S : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 3 = 5) 
  (h₃ : S (k + 2) - S k = 36) : 
  k = 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_k_value_l1827_182701


namespace NUMINAMATH_GPT_monkey_reaches_top_in_19_minutes_l1827_182792

theorem monkey_reaches_top_in_19_minutes (pole_height : ℕ) (ascend_first_min : ℕ) (slip_every_alternate_min : ℕ) 
    (total_minutes : ℕ) (net_gain_two_min : ℕ) : 
    pole_height = 10 ∧ ascend_first_min = 2 ∧ slip_every_alternate_min = 1 ∧ net_gain_two_min = 1 ∧ total_minutes = 19 →
    (net_gain_two_min * (total_minutes - 1) / 2 + ascend_first_min = pole_height) := 
by
    intros
    sorry

end NUMINAMATH_GPT_monkey_reaches_top_in_19_minutes_l1827_182792


namespace NUMINAMATH_GPT_amount_of_juice_p_in_a_l1827_182789

  def total_p : ℚ := 24
  def total_v : ℚ := 25
  def ratio_a : ℚ := 4 / 1
  def ratio_y : ℚ := 1 / 5

  theorem amount_of_juice_p_in_a :
    ∃ P_a : ℚ, ∃ V_a : ℚ, ∃ P_y : ℚ, ∃ V_y : ℚ,
      P_a / V_a = ratio_a ∧ P_y / V_y = ratio_y ∧
      P_a + P_y = total_p ∧ V_a + V_y = total_v ∧ P_a = 20 :=
  by
    sorry
  
end NUMINAMATH_GPT_amount_of_juice_p_in_a_l1827_182789


namespace NUMINAMATH_GPT_unique_integer_solution_l1827_182719

theorem unique_integer_solution (x y : ℤ) : 
  x^4 + y^4 = 3 * x^3 * y → x = 0 ∧ y = 0 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_unique_integer_solution_l1827_182719


namespace NUMINAMATH_GPT_maximum_students_l1827_182702

theorem maximum_students (x : ℕ) (hx : x / 2 + x / 4 + x / 7 + 6 > x) : x ≤ 28 :=
by sorry

end NUMINAMATH_GPT_maximum_students_l1827_182702


namespace NUMINAMATH_GPT_least_num_subtracted_l1827_182781

theorem least_num_subtracted 
  (n : ℕ) 
  (h1 : n = 642) 
  (rem_cond : ∀ k, (k = 638) → n - k = 4): 
  n - 638 = 4 := 
by sorry

end NUMINAMATH_GPT_least_num_subtracted_l1827_182781


namespace NUMINAMATH_GPT_four_star_three_l1827_182788

def star (a b : ℕ) : ℕ := a^2 - a * b + b^2 + 2 * a * b

theorem four_star_three : star 4 3 = 37 :=
by
  -- here we would normally provide the proof steps
  sorry

end NUMINAMATH_GPT_four_star_three_l1827_182788


namespace NUMINAMATH_GPT_suitable_land_acres_l1827_182728

theorem suitable_land_acres (new_multiplier : ℝ) (previous_acres : ℝ) (pond_acres : ℝ) :
  new_multiplier = 10 ∧ previous_acres = 2 ∧ pond_acres = 1 → 
  (new_multiplier * previous_acres - pond_acres) = 19 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_suitable_land_acres_l1827_182728


namespace NUMINAMATH_GPT_julia_total_spend_l1827_182718

noncomputable def total_cost_julia_puppy : ℝ :=
  let adoption_fee := 20.00
  let dog_food := 20.00
  let treat_cost := 2.50
  let treat_count := 2
  let treats := treat_cost * treat_count
  let toys := 15.00
  let crate := 20.00
  let bed := 20.00
  let collar_leash := 15.00
  let total_supplies := dog_food + treats + toys + crate + bed + collar_leash
  let discount := 0.20 * total_supplies
  let final_supplies := total_supplies - discount
  final_supplies + adoption_fee

theorem julia_total_spend : total_cost_julia_puppy = 96.00 :=
by
  sorry

end NUMINAMATH_GPT_julia_total_spend_l1827_182718


namespace NUMINAMATH_GPT_soft_lenses_more_than_hard_l1827_182794

-- Define the problem conditions as Lean definitions
def total_sales (S H : ℕ) : Prop := 150 * S + 85 * H = 1455
def total_pairs (S H : ℕ) : Prop := S + H = 11

-- The theorem we need to prove
theorem soft_lenses_more_than_hard (S H : ℕ) (h1 : total_sales S H) (h2 : total_pairs S H) : S - H = 5 :=
by
  sorry

end NUMINAMATH_GPT_soft_lenses_more_than_hard_l1827_182794


namespace NUMINAMATH_GPT_nonnegative_integers_with_abs_value_less_than_4_l1827_182771

theorem nonnegative_integers_with_abs_value_less_than_4 :
  {n : ℕ | abs (n : ℤ) < 4} = {0, 1, 2, 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_nonnegative_integers_with_abs_value_less_than_4_l1827_182771


namespace NUMINAMATH_GPT_log_identity_l1827_182704

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_identity : log 2 5 * log 3 2 * log 5 3 = 1 :=
by sorry

end NUMINAMATH_GPT_log_identity_l1827_182704


namespace NUMINAMATH_GPT_binomial_last_three_terms_sum_l1827_182795

theorem binomial_last_three_terms_sum (n : ℕ) :
  (1 + n + (n * (n - 1)) / 2 = 79) → n = 12 :=
by
  sorry

end NUMINAMATH_GPT_binomial_last_three_terms_sum_l1827_182795


namespace NUMINAMATH_GPT_roots_sum_of_quadratic_l1827_182724

theorem roots_sum_of_quadratic :
  ∀ x1 x2 : ℝ, (Polynomial.eval x1 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              (Polynomial.eval x2 (Polynomial.X ^ 2 + 2 * Polynomial.X - 1) = 0) →
              x1 + x2 = -2 :=
by
  intros x1 x2 h1 h2
  sorry

end NUMINAMATH_GPT_roots_sum_of_quadratic_l1827_182724


namespace NUMINAMATH_GPT_marble_cut_percentage_l1827_182772

theorem marble_cut_percentage
  (initial_weight : ℝ)
  (final_weight : ℝ)
  (x : ℝ)
  (first_week_cut : ℝ)
  (second_week_cut : ℝ)
  (third_week_cut : ℝ) :
  initial_weight = 190 →
  final_weight = 109.0125 →
  first_week_cut = (1 - x / 100) →
  second_week_cut = 0.85 →
  third_week_cut = 0.9 →
  (initial_weight * first_week_cut * second_week_cut * third_week_cut = final_weight) →
  x = 24.95 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_marble_cut_percentage_l1827_182772


namespace NUMINAMATH_GPT_count_prime_numbers_in_sequence_l1827_182725

theorem count_prime_numbers_in_sequence : 
  ∀ (k : Nat), (∃ n : Nat, 47 * (10^n * k + (10^(n-1) - 1) / 9) = 47) → k = 0 :=
  sorry

end NUMINAMATH_GPT_count_prime_numbers_in_sequence_l1827_182725


namespace NUMINAMATH_GPT_framed_painting_ratio_l1827_182777

-- Define the conditions and the problem
theorem framed_painting_ratio:
  ∀ (x : ℝ),
    (30 + 2 * x) * (20 + 4 * x) = 1500 →
    (20 + 4 * x) / (30 + 2 * x) = 4 / 5 := 
by sorry

end NUMINAMATH_GPT_framed_painting_ratio_l1827_182777


namespace NUMINAMATH_GPT_solve_for_n_l1827_182723

theorem solve_for_n : ∃ n : ℤ, 3^3 - 5 = 4^2 + n ∧ n = 6 := 
by
  use 6
  sorry

end NUMINAMATH_GPT_solve_for_n_l1827_182723


namespace NUMINAMATH_GPT_area_of_triangle_is_2_l1827_182783

-- Define the conditions of the problem
variable (a b c : ℝ)
variable (A B C : ℝ)  -- Angles in radians

-- Conditions for the triangle ABC
variable (sin_A : ℝ) (sin_C : ℝ)
variable (c2sinA_eq_5sinC : c^2 * sin_A = 5 * sin_C)
variable (a_plus_c_squared_eq_16_plus_b_squared : (a + c)^2 = 16 + b^2)
variable (ac_eq_5 : a * c = 5)
variable (cos_B : ℝ)
variable (sin_B : ℝ)

-- Sine and Cosine law results
variable (cos_B_def : cos_B = (a^2 + c^2 - b^2) / (2 * a * c))
variable (sin_B_def : sin_B = Real.sqrt (1 - cos_B^2))

-- Area of the triangle
noncomputable def area_triangle_ABC := (1/2) * a * c * sin_B

-- Theorem to prove the area
theorem area_of_triangle_is_2 :
  area_triangle_ABC a c sin_B = 2 :=
by
  rw [area_triangle_ABC]
  sorry

end NUMINAMATH_GPT_area_of_triangle_is_2_l1827_182783


namespace NUMINAMATH_GPT_imons_no_entanglements_l1827_182732

-- Define the fundamental structure for imons and their entanglements.
universe u
variable {α : Type u}

-- Define a graph structure to represent imons and their entanglement.
structure Graph (α : Type u) where
  vertices : Finset α
  edges : Finset (α × α)
  edge_sym : ∀ {x y}, (x, y) ∈ edges → (y, x) ∈ edges

-- Define the operations that can be performed on imons.
structure ImonOps (G : Graph α) where
  destroy : {v : α} → G.vertices.card % 2 = 1
  double : Graph α

-- Prove the main theorem
theorem imons_no_entanglements (G : Graph α) (op : ImonOps G) : 
  ∃ seq : List (ImonOps G), ∀ g : Graph α, g ∈ (seq.map (λ h => h.double)) → g.edges = ∅ :=
by
  sorry -- The proof would be constructed here.

end NUMINAMATH_GPT_imons_no_entanglements_l1827_182732


namespace NUMINAMATH_GPT_original_population_l1827_182762

-- Define the initial setup
variable (P : ℝ)

-- The conditions given in the problem
axiom ten_percent_died (P : ℝ) : (1 - 0.1) * P = 0.9 * P
axiom twenty_percent_left (P : ℝ) : (1 - 0.2) * (0.9 * P) = 0.9 * P * 0.8

-- Define the final condition
axiom final_population (P : ℝ) : 0.9 * P * 0.8 = 3240

-- The proof problem
theorem original_population : P = 4500 :=
by
  sorry

end NUMINAMATH_GPT_original_population_l1827_182762


namespace NUMINAMATH_GPT_point_on_curve_l1827_182766

-- Define the equation of the curve
def curve (x y : ℝ) := x^2 - x * y + 2 * y + 1 = 0

-- State that point (3, 10) satisfies the given curve equation
theorem point_on_curve : curve 3 10 :=
by
  -- this is where the proof would go but we will skip it for now
  sorry

end NUMINAMATH_GPT_point_on_curve_l1827_182766


namespace NUMINAMATH_GPT_max_frac_sum_l1827_182742

theorem max_frac_sum (n a b c d : ℕ) (hn : 1 < n) (hab : 0 < a) (hcd : 0 < c)
    (hfrac : (a / b) + (c / d) < 1) (hsum : a + c ≤ n) :
    (∃ (b_val : ℕ), 2 ≤ b_val ∧ b_val ≤ n ∧ 
    1 - 1 / (b_val * (b_val * (n + 1 - b_val) + 1)) = 
    1 - 1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1))) :=
sorry

end NUMINAMATH_GPT_max_frac_sum_l1827_182742


namespace NUMINAMATH_GPT_solve_for_x_l1827_182768

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) (h : (5*x)^10 = (10*x)^5) : x = 2/5 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1827_182768


namespace NUMINAMATH_GPT_smallest_C_inequality_l1827_182749

theorem smallest_C_inequality (x y z : ℝ) (h : x + y + z = -1) : 
  |x^3 + y^3 + z^3 + 1| ≤ (9/10) * |x^5 + y^5 + z^5 + 1| :=
  sorry

end NUMINAMATH_GPT_smallest_C_inequality_l1827_182749


namespace NUMINAMATH_GPT_cricketer_average_after_19_innings_l1827_182711

theorem cricketer_average_after_19_innings
  (runs_19th_inning : ℕ)
  (increase_in_average : ℤ)
  (initial_average : ℤ)
  (new_average : ℤ)
  (h1 : runs_19th_inning = 95)
  (h2 : increase_in_average = 4)
  (eq1 : 18 * initial_average + 95 = 19 * (initial_average + increase_in_average))
  (eq2 : new_average = initial_average + increase_in_average) :
  new_average = 23 :=
by sorry

end NUMINAMATH_GPT_cricketer_average_after_19_innings_l1827_182711


namespace NUMINAMATH_GPT_international_postage_surcharge_l1827_182717

theorem international_postage_surcharge 
  (n_letters : ℕ) 
  (std_postage_per_letter : ℚ) 
  (n_international : ℕ) 
  (total_cost : ℚ) 
  (cents_per_dollar : ℚ) 
  (std_total_cost : ℚ) 
  : 
  n_letters = 4 →
  std_postage_per_letter = 108 / 100 →
  n_international = 2 →
  total_cost = 460 / 100 →
  cents_per_dollar = 100 →
  std_total_cost = n_letters * std_postage_per_letter →
  (total_cost - std_total_cost) / n_international * cents_per_dollar = 14 := 
sorry

end NUMINAMATH_GPT_international_postage_surcharge_l1827_182717


namespace NUMINAMATH_GPT_total_games_played_l1827_182730

theorem total_games_played (games_attended games_missed : ℕ) 
  (h_attended : games_attended = 395) 
  (h_missed : games_missed = 469) : 
  games_attended + games_missed = 864 := 
by
  sorry

end NUMINAMATH_GPT_total_games_played_l1827_182730


namespace NUMINAMATH_GPT_find_m_l1827_182793

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 + 3 * x2 = 5) : m = 7 / 4 :=
  sorry

end NUMINAMATH_GPT_find_m_l1827_182793


namespace NUMINAMATH_GPT_stream_speed_l1827_182774

theorem stream_speed (x : ℝ) (d : ℝ) (v_b : ℝ) (t : ℝ) (h : v_b = 8) (h1 : d = 210) (h2 : t = 56) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_stream_speed_l1827_182774


namespace NUMINAMATH_GPT_count_multiples_4_or_9_but_not_both_l1827_182740

theorem count_multiples_4_or_9_but_not_both (n : ℕ) (h : n = 200) :
  let count_multiples (k : ℕ) := (n / k)
  count_multiples 4 + count_multiples 9 - 2 * count_multiples 36 = 62 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_4_or_9_but_not_both_l1827_182740


namespace NUMINAMATH_GPT_dice_product_composite_probability_l1827_182763

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end NUMINAMATH_GPT_dice_product_composite_probability_l1827_182763


namespace NUMINAMATH_GPT_first_team_odd_is_correct_l1827_182756

noncomputable def odd_for_first_team : Real := 
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let bet_amount := 5.00
  let expected_win := 223.0072
  let total_odds := expected_win / bet_amount
  let denominator := odd2 * odd3 * odd4
  total_odds / denominator

theorem first_team_odd_is_correct : 
  odd_for_first_team = 1.28 := by 
  sorry

end NUMINAMATH_GPT_first_team_odd_is_correct_l1827_182756


namespace NUMINAMATH_GPT_cows_with_no_spot_l1827_182776

theorem cows_with_no_spot (total_cows : ℕ) (percent_red_spot : ℚ) (percent_blue_spot : ℚ) :
  total_cows = 140 ∧ percent_red_spot = 0.40 ∧ percent_blue_spot = 0.25 → 
  ∃ (no_spot_cows : ℕ), no_spot_cows = 63 :=
by 
  sorry

end NUMINAMATH_GPT_cows_with_no_spot_l1827_182776


namespace NUMINAMATH_GPT_solve_equation_l1827_182769

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end NUMINAMATH_GPT_solve_equation_l1827_182769


namespace NUMINAMATH_GPT_symm_y_axis_l1827_182729

noncomputable def f (x : ℝ) : ℝ := abs x

theorem symm_y_axis (x : ℝ) : f (-x) = f (x) := by
  sorry

end NUMINAMATH_GPT_symm_y_axis_l1827_182729


namespace NUMINAMATH_GPT_distinct_digits_solution_l1827_182791

theorem distinct_digits_solution (A B C : ℕ)
  (h1 : A + B = 10)
  (h2 : C + A = 9)
  (h3 : B + C = 9)
  (h4 : A ≠ B)
  (h5 : B ≠ C)
  (h6 : C ≠ A)
  (h7 : 0 < A)
  (h8 : 0 < B)
  (h9 : 0 < C)
  : A = 1 ∧ B = 9 ∧ C = 8 := 
  by sorry

end NUMINAMATH_GPT_distinct_digits_solution_l1827_182791


namespace NUMINAMATH_GPT_dream_star_games_l1827_182753

theorem dream_star_games (x y : ℕ) 
  (h1 : x + y + 2 = 9)
  (h2 : 3 * x + y = 17) : 
  x = 5 ∧ y = 2 := 
by 
  sorry

end NUMINAMATH_GPT_dream_star_games_l1827_182753


namespace NUMINAMATH_GPT_doughnut_machine_completion_time_l1827_182737

-- Define the start time and the time when half the job is completed
def start_time := 8 * 60 -- 8:00 AM in minutes
def half_job_time := 10 * 60 + 30 -- 10:30 AM in minutes

-- Given the machine completes half of the day's job by 10:30 AM
-- Prove that the doughnut machine will complete the entire job by 1:00 PM
theorem doughnut_machine_completion_time :
  half_job_time - start_time = 150 → 
  (start_time + 2 * 150) % (24 * 60) = 13 * 60 :=
by
  sorry

end NUMINAMATH_GPT_doughnut_machine_completion_time_l1827_182737


namespace NUMINAMATH_GPT_cut_scene_length_proof_l1827_182779

noncomputable def original_length : ℕ := 60
noncomputable def final_length : ℕ := 57
noncomputable def cut_scene_length := original_length - final_length

theorem cut_scene_length_proof : cut_scene_length = 3 := by
  sorry

end NUMINAMATH_GPT_cut_scene_length_proof_l1827_182779


namespace NUMINAMATH_GPT_ann_fare_90_miles_l1827_182734

-- Define the conditions as given in the problem
def fare (distance : ℕ) : ℕ := 30 + distance * 2

-- Theorem statement
theorem ann_fare_90_miles : fare 90 = 210 := by
  sorry

end NUMINAMATH_GPT_ann_fare_90_miles_l1827_182734


namespace NUMINAMATH_GPT_distinct_factors_of_product_l1827_182754

theorem distinct_factors_of_product (m a b d : ℕ) (hm : m ≥ 1) (ha : m^2 < a ∧ a < m^2 + m)
  (hb : m^2 < b ∧ b < m^2 + m) (hab : a ≠ b) (hd : d ∣ (a * b)) (hd_range: m^2 < d ∧ d < m^2 + m) :
  d = a ∨ d = b :=
sorry

end NUMINAMATH_GPT_distinct_factors_of_product_l1827_182754


namespace NUMINAMATH_GPT_union_A_B_l1827_182726

noncomputable def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
noncomputable def B : Set ℝ := { x | x^3 = x }

theorem union_A_B : A ∪ B = { -1, 0, 1, 2 } := by
  sorry

end NUMINAMATH_GPT_union_A_B_l1827_182726


namespace NUMINAMATH_GPT_minnie_takes_more_time_l1827_182759

def minnie_speed_flat : ℝ := 25
def minnie_speed_downhill : ℝ := 35
def minnie_speed_uphill : ℝ := 10
def penny_speed_flat : ℝ := 35
def penny_speed_downhill : ℝ := 45
def penny_speed_uphill : ℝ := 15

def distance_A_to_B : ℝ := 15
def distance_B_to_D : ℝ := 20
def distance_D_to_C : ℝ := 25

def distance_C_to_B : ℝ := 20
def distance_D_to_A : ℝ := 25

noncomputable def time_minnie : ℝ :=
(distance_A_to_B / minnie_speed_uphill) + 
(distance_B_to_D / minnie_speed_downhill) + 
(distance_D_to_C / minnie_speed_flat)

noncomputable def time_penny : ℝ :=
(distance_C_to_B / penny_speed_uphill) + 
(distance_B_to_D / penny_speed_downhill) + 
(distance_D_to_A / penny_speed_flat)

noncomputable def time_diff : ℝ := (time_minnie - time_penny) * 60

theorem minnie_takes_more_time : time_diff = 10 := by
  sorry

end NUMINAMATH_GPT_minnie_takes_more_time_l1827_182759


namespace NUMINAMATH_GPT_sum_max_min_a_l1827_182787

theorem sum_max_min_a (a : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x - 20 * a^2 < 0)
  (h2 : ∀ x1 x2 : ℝ, x1^2 - a * x1 - 20 * a^2 = 0 → x2^2 - a * x2 - 20 * a^2 = 0 → |x1 - x2| ≤ 9) :
    -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → (1 + -1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_max_min_a_l1827_182787


namespace NUMINAMATH_GPT_equalChargesAtFour_agencyADecisionWhenTen_l1827_182727

-- Define the conditions as constants
def fullPrice : ℕ := 240
def agencyADiscount : ℕ := 50
def agencyBDiscount : ℕ := 60

-- Define the total charge function for both agencies
def totalChargeAgencyA (students: ℕ) : ℕ :=
  fullPrice * students * agencyADiscount / 100 + fullPrice

def totalChargeAgencyB (students: ℕ) : ℕ :=
  fullPrice * (students + 1) * agencyBDiscount / 100

-- Define the equivalence when the number of students is 4
theorem equalChargesAtFour : totalChargeAgencyA 4 = totalChargeAgencyB 4 := by sorry

-- Define the decision when there are 10 students
theorem agencyADecisionWhenTen : totalChargeAgencyA 10 < totalChargeAgencyB 10 := by sorry

end NUMINAMATH_GPT_equalChargesAtFour_agencyADecisionWhenTen_l1827_182727


namespace NUMINAMATH_GPT_number_of_laborers_l1827_182733

-- Definitions based on conditions in the problem
def hpd := 140   -- Earnings per day for heavy equipment operators
def gpd := 90    -- Earnings per day for general laborers
def totalPeople := 35  -- Total number of people hired
def totalPayroll := 3950  -- Total payroll in dollars

-- Variables H and L for the number of operators and laborers
variables (H L : ℕ)

-- Conditions provided in mathematical problem
axiom equation1 : H + L = totalPeople
axiom equation2 : hpd * H + gpd * L = totalPayroll

-- Theorem statement: we want to prove that L = 19
theorem number_of_laborers : L = 19 :=
sorry

end NUMINAMATH_GPT_number_of_laborers_l1827_182733


namespace NUMINAMATH_GPT_point_a_number_l1827_182778

theorem point_a_number (x : ℝ) (h : abs (x - 2) = 6) : x = 8 ∨ x = -4 :=
sorry

end NUMINAMATH_GPT_point_a_number_l1827_182778


namespace NUMINAMATH_GPT_nasadkas_in_barrel_l1827_182785

def capacity (B N V : ℚ) :=
  (B + 20 * V = 3 * B) ∧ (19 * B + N + 15.5 * V = 20 * B + 8 * V)

theorem nasadkas_in_barrel (B N V : ℚ) (h : capacity B N V) : B / N = 4 :=
by
  sorry

end NUMINAMATH_GPT_nasadkas_in_barrel_l1827_182785


namespace NUMINAMATH_GPT_calculate_base_length_l1827_182741

variable (A b h : ℝ)

def is_parallelogram_base_length (A : ℝ) (b : ℝ) (h : ℝ) : Prop :=
  (A = b * h) ∧ (h = 2 * b)

theorem calculate_base_length (H : is_parallelogram_base_length A b h) : b = 15 := by
  -- H gives us the hypothesis that (A = b * h) and (h = 2 * b)
  have H1 : A = b * h := H.1
  have H2 : h = 2 * b := H.2
  -- Use substitution and algebra to solve for b
  sorry

end NUMINAMATH_GPT_calculate_base_length_l1827_182741


namespace NUMINAMATH_GPT_divisibility_theorem_l1827_182716

theorem divisibility_theorem (a b n : ℕ) (h : a^n ∣ b) : a^(n + 1) ∣ (a + 1)^b - 1 :=
by 
sorry

end NUMINAMATH_GPT_divisibility_theorem_l1827_182716


namespace NUMINAMATH_GPT_remainder_5_to_5_to_5_to_5_mod_1000_l1827_182757

theorem remainder_5_to_5_to_5_to_5_mod_1000 : (5^(5^(5^5))) % 1000 = 125 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_5_to_5_to_5_to_5_mod_1000_l1827_182757


namespace NUMINAMATH_GPT_area_inside_C_outside_A_B_l1827_182700

-- Define the given circles with corresponding radii and positions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the circles A, B, and C with the specific properties given
def CircleA : Circle := { center := (0, 0), radius := 1 }
def CircleB : Circle := { center := (2, 0), radius := 1 }
def CircleC : Circle := { center := (1, 2), radius := 2 }

-- Given that Circle C is tangent to the midpoint M of the line segment AB
-- Prove the area inside Circle C but outside Circle A and B
theorem area_inside_C_outside_A_B : 
  let area_inside_C := π * CircleC.radius ^ 2
  let overlap_area := (π - 2)
  area_inside_C - overlap_area = 3 * π + 2 := by
  sorry

end NUMINAMATH_GPT_area_inside_C_outside_A_B_l1827_182700


namespace NUMINAMATH_GPT_ratio_A_BC_1_to_4_l1827_182782

/-
We will define the conditions and prove the ratio.
-/

def A := 20
def total := 100

-- defining the conditions
variables (B C : ℝ)
def condition1 := A + B + C = total
def condition2 := B = 3 / 5 * (A + C)

-- the theorem to prove
theorem ratio_A_BC_1_to_4 (h1 : condition1 B C) (h2 : condition2 B C) : A / (B + C) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_A_BC_1_to_4_l1827_182782


namespace NUMINAMATH_GPT_symmetric_slope_angle_l1827_182714

-- Define the problem conditions in Lean
def slope_angle (θ : Real) : Prop :=
  0 ≤ θ ∧ θ < Real.pi

-- Statement of the theorem in Lean
theorem symmetric_slope_angle (θ : Real) (h : slope_angle θ) :
  θ = 0 ∨ θ = Real.pi - θ :=
sorry

end NUMINAMATH_GPT_symmetric_slope_angle_l1827_182714


namespace NUMINAMATH_GPT_decrypt_nbui_is_math_l1827_182731

-- Define the sets A and B as the 26 English letters
def A := {c : Char | c ≥ 'a' ∧ c ≤ 'z'}
def B := A

-- Define the mapping f from A to B
def f (c : Char) : Char :=
  if c = 'z' then 'a'
  else Char.ofNat (c.toNat + 1)

-- Define the decryption function g (it reverses the mapping f)
def g (c : Char) : Char :=
  if c = 'a' then 'z'
  else Char.ofNat (c.toNat - 1)

-- Define the decryption of the given ciphertext
def decrypt (ciphertext : String) : String :=
  ciphertext.map g

-- Prove that the decryption of "nbui" is "math"
theorem decrypt_nbui_is_math : decrypt "nbui" = "math" :=
  by
  sorry

end NUMINAMATH_GPT_decrypt_nbui_is_math_l1827_182731


namespace NUMINAMATH_GPT_bug_probability_at_A_after_8_meters_l1827_182760

noncomputable def P : ℕ → ℚ 
| 0 => 1
| (n + 1) => (1 / 3) * (1 - P n)

theorem bug_probability_at_A_after_8_meters :
  P 8 = 547 / 2187 := 
sorry

end NUMINAMATH_GPT_bug_probability_at_A_after_8_meters_l1827_182760


namespace NUMINAMATH_GPT_cos_triple_angle_l1827_182735

theorem cos_triple_angle (x θ : ℝ) (h : x = Real.cos θ) : Real.cos (3 * θ) = 4 * x^3 - 3 * x :=
by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l1827_182735


namespace NUMINAMATH_GPT_booknote_unique_elements_l1827_182748

def booknote_string : String := "booknote"
def booknote_set : Finset Char := { 'b', 'o', 'k', 'n', 't', 'e' }

theorem booknote_unique_elements : booknote_set.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_booknote_unique_elements_l1827_182748


namespace NUMINAMATH_GPT_range_of_f_l1827_182746

noncomputable def f (x : ℕ) : ℤ := x^2 - 3 * x

def domain : Finset ℕ := {1, 2, 3}

def range : Finset ℤ := {-2, 0}

theorem range_of_f :
  Finset.image f domain = range :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1827_182746


namespace NUMINAMATH_GPT_area_of_annulus_l1827_182722

section annulus
variables {R r x : ℝ}
variable (h1 : R > r)
variable (h2 : R^2 - r^2 = x^2)

theorem area_of_annulus (R r x : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = x^2) : 
  π * R^2 - π * r^2 = π * x^2 :=
sorry

end annulus

end NUMINAMATH_GPT_area_of_annulus_l1827_182722


namespace NUMINAMATH_GPT_polygon_sides_eq_eight_l1827_182715

theorem polygon_sides_eq_eight (n : ℕ) 
  (h_diff : (n - 2) * 180 - 360 = 720) :
  n = 8 := 
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_eight_l1827_182715


namespace NUMINAMATH_GPT_problem_statement_l1827_182798

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem problem_statement (x : ℝ) (h : x ≠ 0) : f x > 0 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1827_182798
