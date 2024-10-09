import Mathlib

namespace max_happy_monkeys_l172_17235

-- Definitions for given problem
def pears := 20
def bananas := 30
def peaches := 40
def mandarins := 50
def fruits (x y : Nat) := x + y

-- The theorem to prove
theorem max_happy_monkeys : 
  ∃ (m : Nat), m = (pears + bananas + peaches) / 2 ∧ m ≤ mandarins :=
by
  sorry

end max_happy_monkeys_l172_17235


namespace rectangular_garden_width_l172_17257

-- Define the problem conditions as Lean definitions
def rectangular_garden_length (w : ℝ) : ℝ := 3 * w
def rectangular_garden_area (w : ℝ) : ℝ := rectangular_garden_length w * w

-- This is the theorem we want to prove
theorem rectangular_garden_width : ∃ w : ℝ, rectangular_garden_area w = 432 ∧ w = 12 :=
by
  sorry

end rectangular_garden_width_l172_17257


namespace quadratic_equation_identify_l172_17282

theorem quadratic_equation_identify {a b c x : ℝ} :
  ((3 - 5 * x^2 = x) ↔ true) ∧
  ((3 / x + x^2 - 1 = 0) ↔ false) ∧
  ((a * x^2 + b * x + c = 0) ↔ (a ≠ 0)) ∧
  ((4 * x - 1 = 0) ↔ false) :=
by
  sorry

end quadratic_equation_identify_l172_17282


namespace cheryl_used_material_l172_17201

theorem cheryl_used_material 
  (a b c l : ℚ) 
  (ha : a = 3 / 8) 
  (hb : b = 1 / 3) 
  (hl : l = 15 / 40) 
  (Hc: c = a + b): 
  (c - l = 1 / 3) := 
by 
  -- proof will be deferred to Lean's syntax for user to fill in.
  sorry

end cheryl_used_material_l172_17201


namespace at_least_one_vowel_l172_17296

-- Define the set of letters
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'I'}

-- Define the vowels within the set of letters
def vowels : Finset Char := {'A', 'E', 'I'}

-- Define the consonants within the set of letters
def consonants : Finset Char := {'B', 'C', 'D', 'F'}

-- Function to count the total number of 3-letter words from a given set
def count_words (s : Finset Char) (length : Nat) : Nat :=
  s.card ^ length

-- Define the statement of the problem
theorem at_least_one_vowel : count_words letters 3 - count_words consonants 3 = 279 :=
by
  sorry

end at_least_one_vowel_l172_17296


namespace line_through_point_perpendicular_y_axis_line_through_two_points_l172_17249

-- The first problem
theorem line_through_point_perpendicular_y_axis :
  ∃ (k : ℝ), ∀ (x : ℝ), k = 1 → y = k :=
sorry

-- The second problem
theorem line_through_two_points (x1 y1 x2 y2 : ℝ) (hA : (x1, y1) = (-4, 0)) (hB : (x2, y2) = (0, 6)) :
  ∃ (a b c : ℝ), (a, b, c) = (3, -2, 12) → ∀ (x y : ℝ), a * x + b * y + c = 0 :=
sorry

end line_through_point_perpendicular_y_axis_line_through_two_points_l172_17249


namespace point_on_x_axis_point_on_y_axis_l172_17271

section
-- Definitions for the conditions
def point_A (a : ℝ) : ℝ × ℝ := (a - 3, a ^ 2 - 4)

-- Proof for point A lying on the x-axis
theorem point_on_x_axis (a : ℝ) (h : (point_A a).2 = 0) :
  point_A a = (-1, 0) ∨ point_A a = (-5, 0) :=
sorry

-- Proof for point A lying on the y-axis
theorem point_on_y_axis (a : ℝ) (h : (point_A a).1 = 0) :
  point_A a = (0, 5) :=
sorry
end

end point_on_x_axis_point_on_y_axis_l172_17271


namespace total_days_on_jury_duty_l172_17204

-- Define the conditions
def jury_selection_days : ℕ := 2
def trial_duration_factor : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_per_day : ℕ := 24

-- Calculate the trial duration in days
def trial_days : ℕ := trial_duration_factor * jury_selection_days

-- Calculate the total deliberation time in days
def deliberation_total_hours : ℕ := deliberation_days * deliberation_hours_per_day
def deliberation_days_converted : ℕ := deliberation_total_hours / hours_per_day

-- Statement that John spends a total of 14 days on jury duty
theorem total_days_on_jury_duty : jury_selection_days + trial_days + deliberation_days_converted = 14 :=
sorry

end total_days_on_jury_duty_l172_17204


namespace exponent_on_right_side_l172_17243

theorem exponent_on_right_side (n : ℕ) (h : n = 17) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 :=
by
  sorry

end exponent_on_right_side_l172_17243


namespace func_C_increasing_l172_17260

open Set

noncomputable def func_A (x : ℝ) : ℝ := 3 - x
noncomputable def func_B (x : ℝ) : ℝ := x^2 - x
noncomputable def func_C (x : ℝ) : ℝ := -1 / (x + 1)
noncomputable def func_D (x : ℝ) : ℝ := -abs x

theorem func_C_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → func_C x < func_C y := by
  sorry

end func_C_increasing_l172_17260


namespace range_of_a_l172_17261

noncomputable def satisfies_system (a b c : ℝ) : Prop :=
  (a^2 - b * c - 8 * a + 7 = 0) ∧ (b^2 + c^2 + b * c - 6 * a + 6 = 0)

theorem range_of_a (a b c : ℝ) 
  (h : satisfies_system a b c) : 1 ≤ a ∧ a ≤ 9 :=
by
  sorry

end range_of_a_l172_17261


namespace water_leaked_l172_17227

theorem water_leaked (initial remaining : ℝ) (h_initial : initial = 0.75) (h_remaining : remaining = 0.5) :
  initial - remaining = 0.25 :=
by
  sorry

end water_leaked_l172_17227


namespace kim_paints_fewer_tiles_than_laura_l172_17266

-- Given conditions and definitions
def don_rate : ℕ := 3
def ken_rate : ℕ := don_rate + 2
def laura_rate : ℕ := 2 * ken_rate
def total_tiles_per_15_minutes : ℕ := 375
def total_rate_per_minute : ℕ := total_tiles_per_15_minutes / 15
def kim_rate : ℕ := total_rate_per_minute - (don_rate + ken_rate + laura_rate)

-- Proof goal
theorem kim_paints_fewer_tiles_than_laura :
  laura_rate - kim_rate = 3 :=
by
  sorry

end kim_paints_fewer_tiles_than_laura_l172_17266


namespace sin_cos_theta_l172_17207

-- Define the problem conditions and the question as a Lean statement
theorem sin_cos_theta (θ : ℝ) (h : Real.tan (θ + Real.pi / 2) = 2) : Real.sin θ * Real.cos θ = -2 / 5 := by
  sorry

end sin_cos_theta_l172_17207


namespace divides_seven_l172_17255

theorem divides_seven (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : Nat.gcd x y = 1) (h5 : x^2 + y^2 = z^4) : 7 ∣ x * y :=
by
  sorry

end divides_seven_l172_17255


namespace maximum_value_of_rocks_l172_17290

theorem maximum_value_of_rocks (R6_val R3_val R2_val : ℕ)
  (R6_wt R3_wt R2_wt : ℕ)
  (num6 num3 num2 : ℕ) :
  R6_val = 16 →
  R3_val = 9 →
  R2_val = 3 →
  R6_wt = 6 →
  R3_wt = 3 →
  R2_wt = 2 →
  30 ≤ num6 →
  30 ≤ num3 →
  30 ≤ num2 →
  ∃ (x6 x3 x2 : ℕ),
    x6 ≤ 4 ∧
    x3 ≤ 4 ∧
    x2 ≤ 4 ∧
    (x6 * R6_wt + x3 * R3_wt + x2 * R2_wt ≤ 24) ∧
    (x6 * R6_val + x3 * R3_val + x2 * R2_val = 68) :=
by
  sorry

end maximum_value_of_rocks_l172_17290


namespace int_sol_many_no_int_sol_l172_17254

-- Part 1: If there is one integer solution, there are at least three integer solutions
theorem int_sol_many (n : ℤ) (hn : n > 0) (x y : ℤ) 
  (hxy : x^3 - 3 * x * y^2 + y^3 = n) : 
  ∃ a b c d e f : ℤ, 
    (a, b) ≠ (x, y) ∧ (c, d) ≠ (x, y) ∧ (e, f) ≠ (x, y) ∧ 
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f) ∧ 
    a^3 - 3 * a * b^2 + b^3 = n ∧ 
    c^3 - 3 * c * d^2 + d^3 = n ∧ 
    e^3 - 3 * e * f^2 + f^3 = n :=
sorry

-- Part 2: When n = 2891, the equation has no integer solutions
theorem no_int_sol : ¬ ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end int_sol_many_no_int_sol_l172_17254


namespace roots_negative_condition_l172_17241

theorem roots_negative_condition (a b c r s : ℝ) (h_eqn : a ≠ 0) (h_root : a * r^2 + b * r + c = 0) (h_neg : r = -s) : b = 0 := sorry

end roots_negative_condition_l172_17241


namespace number_of_trees_planted_l172_17280

-- Definition of initial conditions
def initial_trees : ℕ := 22
def final_trees : ℕ := 55

-- Theorem stating the number of trees planted
theorem number_of_trees_planted : final_trees - initial_trees = 33 := by
  sorry

end number_of_trees_planted_l172_17280


namespace range_p_l172_17248

open Set

def p (x : ℝ) : ℝ :=
  x^4 + 6*x^2 + 9

theorem range_p : range p = Ici 9 := by
  sorry

end range_p_l172_17248


namespace sqrt_domain_l172_17270

def inequality_holds (x : ℝ) : Prop := x + 5 ≥ 0

theorem sqrt_domain (x : ℝ) : inequality_holds x ↔ x ≥ -5 := by
  sorry

end sqrt_domain_l172_17270


namespace ruby_siblings_l172_17265

structure Child :=
  (name : String)
  (eye_color : String)
  (hair_color : String)

def children : List Child :=
[
  {name := "Mason", eye_color := "Green", hair_color := "Red"},
  {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"},
  {name := "Fiona", eye_color := "Brown", hair_color := "Red"},
  {name := "Leo", eye_color := "Green", hair_color := "Blonde"},
  {name := "Ivy", eye_color := "Green", hair_color := "Red"},
  {name := "Carlos", eye_color := "Green", hair_color := "Blonde"}
]

def is_sibling_group (c1 c2 c3 : Child) : Prop :=
  (c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color) ∧
  (c2.eye_color = c3.eye_color ∨ c2.hair_color = c3.hair_color) ∧
  (c1.eye_color = c3.eye_color ∨ c1.hair_color = c3.hair_color)

theorem ruby_siblings :
  ∃ (c1 c2 : Child), 
    c1.name ≠ "Ruby" ∧ c2.name ≠ "Ruby" ∧
    c1 ≠ c2 ∧
    is_sibling_group {name := "Ruby", eye_color := "Brown", hair_color := "Blonde"} c1 c2 ∧
    ((c1.name = "Leo" ∧ c2.name = "Carlos") ∨ (c1.name = "Carlos" ∧ c2.name = "Leo")) :=
by
  sorry

end ruby_siblings_l172_17265


namespace stratified_sampling_third_year_students_l172_17245

/-- 
A university's mathematics department has a total of 5000 undergraduate students, 
with the first, second, third, and fourth years having a ratio of their numbers as 4:3:2:1. 
If stratified sampling is employed to select a sample of 200 students from all undergraduates,
prove that the number of third-year students to be sampled is 40.
-/
theorem stratified_sampling_third_year_students :
  let total_students := 5000
  let ratio_first_second_third_fourth := (4, 3, 2, 1)
  let sample_size := 200
  let third_year_ratio := 2
  let total_ratio_units := 4 + 3 + 2 + 1
  let proportion_third_year := third_year_ratio / total_ratio_units
  let expected_third_year_students := sample_size * proportion_third_year
  expected_third_year_students = 40 :=
by
  sorry

end stratified_sampling_third_year_students_l172_17245


namespace kasun_family_children_count_l172_17209

theorem kasun_family_children_count 
    (m : ℝ) (x : ℕ) (y : ℝ)
    (h1 : (m + 50 + x * y + 10) / (3 + x) = 22)
    (h2 : (m + x * y + 10) / (2 + x) = 18) :
    x = 5 :=
by
  sorry

end kasun_family_children_count_l172_17209


namespace tourists_left_l172_17256

theorem tourists_left (initial_tourists eaten_by_anacondas poisoned_fraction recover_fraction : ℕ) 
(h_initial : initial_tourists = 30) 
(h_eaten : eaten_by_anacondas = 2)
(h_poisoned_fraction : poisoned_fraction = 2)
(h_recover_fraction : recover_fraction = 7) :
  initial_tourists - eaten_by_anacondas - (initial_tourists - eaten_by_anacondas) / poisoned_fraction + (initial_tourists - eaten_by_anacondas) / poisoned_fraction / recover_fraction = 16 :=
by
  sorry

end tourists_left_l172_17256


namespace school_year_days_l172_17284

theorem school_year_days :
  ∀ (D : ℕ),
  (9 = 5 * D / 100) →
  D = 180 := by
  intro D
  sorry

end school_year_days_l172_17284


namespace fuel_spending_reduction_l172_17279

-- Define the variables and the conditions
variable (x c : ℝ) -- x for efficiency and c for cost
variable (newEfficiency oldEfficiency newCost oldCost : ℝ)

-- Define the conditions
def conditions := (oldEfficiency = x) ∧ (newEfficiency = 1.75 * oldEfficiency)
                 ∧ (oldCost = c) ∧ (newCost = 1.3 * oldCost)

-- Define the expected reduction in cost
def expectedReduction : ℝ := 25.7142857142857 -- approximately 25 5/7 %

-- Define the assertion that Elmer will reduce his fuel spending by the expected reduction percentage
theorem fuel_spending_reduction : conditions x c oldEfficiency newEfficiency oldCost newCost →
  ((oldCost - (newCost / newEfficiency) * oldEfficiency) / oldCost) * 100 = expectedReduction :=
by
  sorry

end fuel_spending_reduction_l172_17279


namespace wire_length_between_poles_l172_17225

theorem wire_length_between_poles :
  let x_dist := 20
  let y_dist := (18 / 2) - 8
  (x_dist ^ 2 + y_dist ^ 2 = 401) :=
by
  sorry

end wire_length_between_poles_l172_17225


namespace part_a_part_b_l172_17238

theorem part_a (a : ℤ) (k : ℤ) (h : a + 1 = 3 * k) : ∃ m : ℤ, 4 + 7 * a = 3 * m := by
  sorry

theorem part_b (a b : ℤ) (m n : ℤ) (h1 : 2 + a = 11 * m) (h2 : 35 - b = 11 * n) : ∃ p : ℤ, a + b = 11 * p := by
  sorry

end part_a_part_b_l172_17238


namespace hyperbola_equation_chord_length_l172_17239

noncomputable def length_real_axis := 2
noncomputable def eccentricity := Real.sqrt 3
noncomputable def a := 1
noncomputable def b := Real.sqrt 2
noncomputable def hyperbola_eq (x y : ℝ) := x^2 - y^2 / 2 = 1

theorem hyperbola_equation : 
  (∀ x y : ℝ, hyperbola_eq x y ↔ x^2 - (y^2 / 2) = 1) :=
by
  intros x y
  sorry

theorem chord_length (m : ℝ) : 
  ∀ x1 x2 y1 y2 : ℝ, y1 = x1 + m → y2 = x2 + m →
    x1^2 - y1^2 / 2 = 1 → x2^2 - y2^2 / 2 = 1 →
    Real.sqrt (2 * ((x1 + x2)^2 - 4 * x1 * x2)) = 4 * Real.sqrt 2 →
    m = 1 ∨ m = -1 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4 h5
  sorry

end hyperbola_equation_chord_length_l172_17239


namespace total_score_l172_17202

theorem total_score (score_cap : ℝ) (score_val : ℝ) (score_imp : ℝ) (wt_cap : ℝ) (wt_val : ℝ) (wt_imp : ℝ) (total_weight : ℝ) :
  score_cap = 8 → score_val = 9 → score_imp = 7 → wt_cap = 5 → wt_val = 3 → wt_imp = 2 → total_weight = 10 →
  ((score_cap * (wt_cap / total_weight)) + (score_val * (wt_val / total_weight)) + (score_imp * (wt_imp / total_weight))) = 8.1 := 
by
  intros
  sorry

end total_score_l172_17202


namespace find_number_l172_17275

theorem find_number (x : ℕ) (h : 3 * (x + 2) = 24 + x) : x = 9 :=
by 
  sorry

end find_number_l172_17275


namespace sequence_formula_l172_17285

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) (h3 : a 3 = 7) (h4 : a 4 = 15) :
  ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end sequence_formula_l172_17285


namespace nails_per_station_correct_l172_17258

variable (total_nails : ℕ) (total_stations : ℕ) (nails_per_station : ℕ)

theorem nails_per_station_correct :
  total_nails = 140 → total_stations = 20 → nails_per_station = total_nails / total_stations → nails_per_station = 7 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end nails_per_station_correct_l172_17258


namespace smallest_M_inequality_l172_17272

theorem smallest_M_inequality :
  ∃ M : ℝ, 
  M = 9 / (16 * Real.sqrt 2) ∧
  ∀ a b c : ℝ, 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ M * (a^2 + b^2 + c^2)^2 :=
by
  use 9 / (16 * Real.sqrt 2)
  sorry

end smallest_M_inequality_l172_17272


namespace range_of_a_l172_17273

theorem range_of_a (a : ℝ) (h : ∃ x1 x2, x1 ≠ x2 ∧ 3 * x1^2 + a = 0 ∧ 3 * x2^2 + a = 0) : a < 0 :=
sorry

end range_of_a_l172_17273


namespace swimming_pool_time_l172_17289

theorem swimming_pool_time 
  (empty_rate : ℕ) (fill_rate : ℕ) (capacity : ℕ) (final_volume : ℕ) (t : ℕ)
  (h_empty : empty_rate = 120 / 4) 
  (h_fill : fill_rate = 120 / 6) 
  (h_capacity : capacity = 120) 
  (h_final : final_volume = 90) 
  (h_eq : capacity - (empty_rate - fill_rate) * t = final_volume) :
  t = 3 := 
sorry

end swimming_pool_time_l172_17289


namespace LCM_of_8_and_12_l172_17229

-- Definitions based on the provided conditions
def a : ℕ := 8
def x : ℕ := 12

def HCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Conditions
def hcf_condition : HCF a x = 4 := by sorry
def x_condition : x = 12 := rfl

-- The proof statement
theorem LCM_of_8_and_12 : LCM a x = 24 :=
by
  have h1 : HCF a x = 4 := hcf_condition
  have h2 : x = 12 := x_condition
  rw [h2] at h1
  sorry

end LCM_of_8_and_12_l172_17229


namespace eval_g_at_3_l172_17291

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem eval_g_at_3 : g 3 = 10 := by
  -- Proof goes here
  sorry

end eval_g_at_3_l172_17291


namespace intercepts_line_5x_minus_2y_minus_10_eq_0_l172_17267

theorem intercepts_line_5x_minus_2y_minus_10_eq_0 :
  ∃ a b : ℝ, (a = 2 ∧ b = -5) ∧ (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → 
     ((y = 0 ∧ x = a) ∨ (x = 0 ∧ y = b))) :=
by
  sorry

end intercepts_line_5x_minus_2y_minus_10_eq_0_l172_17267


namespace florist_picked_roses_l172_17244

def initial_roses : ℕ := 11
def sold_roses : ℕ := 2
def final_roses : ℕ := 41
def remaining_roses := initial_roses - sold_roses
def picked_roses := final_roses - remaining_roses

theorem florist_picked_roses : picked_roses = 32 :=
by
  -- This is where the proof would go, but we are leaving it empty on purpose
  sorry

end florist_picked_roses_l172_17244


namespace pelican_count_in_shark_bite_cove_l172_17215

theorem pelican_count_in_shark_bite_cove
  (num_sharks_pelican_bay : ℕ)
  (num_pelicans_shark_bite_cove : ℕ)
  (num_pelicans_moved : ℕ) :
  num_sharks_pelican_bay = 60 →
  num_sharks_pelican_bay = 2 * num_pelicans_shark_bite_cove →
  num_pelicans_moved = num_pelicans_shark_bite_cove / 3 →
  num_pelicans_shark_bite_cove - num_pelicans_moved = 20 :=
by
  sorry

end pelican_count_in_shark_bite_cove_l172_17215


namespace train_speed_l172_17242

theorem train_speed (distance : ℝ) (time : ℝ) (distance_eq : distance = 270) (time_eq : time = 9)
  : (distance / time) * (3600 / 1000) = 108 :=
by 
  sorry

end train_speed_l172_17242


namespace smallest_positive_integer_is_53_l172_17232

theorem smallest_positive_integer_is_53 :
  ∃ a : ℕ, a > 0 ∧ a % 3 = 2 ∧ a % 4 = 1 ∧ a % 5 = 3 ∧ a = 53 :=
by
  sorry

end smallest_positive_integer_is_53_l172_17232


namespace problem1_problem2_l172_17218

-- Problem 1: Prove that (a/(a - b)) + (b/(b - a)) = 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2: Prove that (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c
theorem problem2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c :=
sorry

end problem1_problem2_l172_17218


namespace eccentricity_equals_2_l172_17237

variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (A : ℝ × ℝ) (F : ℝ × ℝ) (B : ℝ × ℝ)
variables (eqn_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
variables (focus_F : F = (c, 0)) (imaginary_axis_B : B = (0, b))
variables (intersect_A : A = (c / 3, 2 * b / 3))
variables (vector_eqn : 3 * (A.1, A.2) = (F.1 + 2 * B.1, F.2 + 2 * B.2))
variables (asymptote_eqn : ∀ A1 A2 : ℝ, A2 = (b / a) * A1 → A = (A1, A2))

theorem eccentricity_equals_2 : (c / a = 2) :=
sorry

end eccentricity_equals_2_l172_17237


namespace find_b_l172_17293

theorem find_b (a b : ℝ) (B C : ℝ)
    (h1 : a * b = 60 * Real.sqrt 3)
    (h2 : Real.sin B = Real.sin C)
    (h3 : 15 * Real.sqrt 3 = 1/2 * a * b * Real.sin C) :
  b = 2 * Real.sqrt 15 :=
sorry

end find_b_l172_17293


namespace sufficient_condition_l172_17220

theorem sufficient_condition (A B : Set α) (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
  by
    intro h1
    apply h
    exact h1

end sufficient_condition_l172_17220


namespace evaluate_expression_l172_17264

theorem evaluate_expression :
  let c := (-2 : ℚ)
  let x := (2 : ℚ) / 5
  let y := (3 : ℚ) / 5
  let z := (-3 : ℚ)
  c * x^3 * y^4 * z^2 = (-11664) / 78125 := by
  sorry

end evaluate_expression_l172_17264


namespace problem1_problem2_problem3_problem4_l172_17210

-- Problem 1
theorem problem1 : -9 + 5 - 11 + 16 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : -9 + 5 - (-6) - 18 / (-3) = 8 :=
by
  sorry

-- Problem 3
theorem problem3 : -2^2 - ((-3) * (-4 / 3) - (-2)^3) = -16 :=
by
  sorry

-- Problem 4
theorem problem4 : (59 - (7 / 9 - 11 / 12 + 1 / 6) * (-6)^2) / (-7)^2 = 58 / 49 :=
by
  sorry

end problem1_problem2_problem3_problem4_l172_17210


namespace rahul_matches_played_l172_17205

-- Define the conditions of the problem
variable (m : ℕ) -- number of matches Rahul has played so far
variable (runs_before : ℕ := 51 * m) -- total runs before today's match
variable (runs_today : ℕ := 69) -- runs scored today
variable (new_average : ℕ := 54) -- new batting average after today's match

-- The equation derived from the conditions
def batting_average_equation : Prop :=
  new_average * (m + 1) = runs_before + runs_today

-- The problem: prove that m = 5 given the conditions
theorem rahul_matches_played (h : batting_average_equation m) : m = 5 :=
  sorry

end rahul_matches_played_l172_17205


namespace opposite_of_2023_l172_17217

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 := 
by
  sorry

end opposite_of_2023_l172_17217


namespace quadratic_inequality_solution_set_l172_17277

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 5*x - 14 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 7} :=
by
  -- proof to be filled here
  sorry

end quadratic_inequality_solution_set_l172_17277


namespace total_fat_served_l172_17263

-- Definitions based on conditions
def fat_herring : ℕ := 40
def fat_eel : ℕ := 20
def fat_pike : ℕ := fat_eel + 10
def fish_served_each : ℕ := 40

-- Calculations based on defined conditions
def total_fat_herring : ℕ := fish_served_each * fat_herring
def total_fat_eel : ℕ := fish_served_each * fat_eel
def total_fat_pike : ℕ := fish_served_each * fat_pike

-- Proof statement to show the total fat served
theorem total_fat_served : total_fat_herring + total_fat_eel + total_fat_pike = 3600 := by
  sorry

end total_fat_served_l172_17263


namespace binary_110101_is_53_l172_17213

def binary_to_decimal (n : Nat) : Nat :=
  let digits := [1, 1, 0, 1, 0, 1]  -- Define binary digits from the problem statement
  digits.reverse.foldr (λ d (acc, pow) => (acc + d * (2^pow), pow + 1)) (0, 0) |>.fst

theorem binary_110101_is_53 : binary_to_decimal 110101 = 53 := by
  sorry

end binary_110101_is_53_l172_17213


namespace solve_quadratic_inequality_l172_17224

open Set Real

noncomputable def quadratic_inequality (x : ℝ) : Prop := -9 * x^2 + 6 * x + 8 > 0

theorem solve_quadratic_inequality :
  {x : ℝ | -9 * x^2 + 6 * x + 8 > 0} = {x : ℝ | -2/3 < x ∧ x < 4/3} :=
by
  sorry

end solve_quadratic_inequality_l172_17224


namespace sum_infinite_geometric_series_l172_17287

theorem sum_infinite_geometric_series (a r : ℚ) (h : a = 1) (h2 : r = 1/4) : 
  (∀ S, S = a / (1 - r) → S = 4 / 3) :=
by
  intros S hS
  rw [h, h2] at hS
  simp [hS]
  sorry

end sum_infinite_geometric_series_l172_17287


namespace company_x_installation_charge_l172_17251

theorem company_x_installation_charge:
  let price_X := 575
  let surcharge_X := 0.04 * price_X
  let installation_charge_X := 82.50
  let total_cost_X := price_X + surcharge_X + installation_charge_X
  let price_Y := 530
  let surcharge_Y := 0.03 * price_Y
  let installation_charge_Y := 93.00
  let total_cost_Y := price_Y + surcharge_Y + installation_charge_Y
  let savings := 41.60
  total_cost_X - total_cost_Y = savings → installation_charge_X = 82.50 :=
by
  intros h
  sorry

end company_x_installation_charge_l172_17251


namespace sample_size_9_l172_17223

variable (X : Nat)

theorem sample_size_9 (h : 36 % X = 0 ∧ 36 % (X + 1) ≠ 0) : X = 9 := 
sorry

end sample_size_9_l172_17223


namespace Adam_marbles_l172_17200

variable (Adam Greg : Nat)

theorem Adam_marbles (h1 : Greg = 43) (h2 : Greg = Adam + 14) : Adam = 29 := 
by
  sorry

end Adam_marbles_l172_17200


namespace ratio_of_tax_revenue_to_cost_of_stimulus_l172_17259

-- Definitions based on the identified conditions
def bottom_20_percent_people (total_people : ℕ) : ℕ := (total_people * 20) / 100
def stimulus_per_person : ℕ := 2000
def total_people : ℕ := 1000
def government_profit : ℕ := 1600000

-- Cost of the stimulus
def cost_of_stimulus : ℕ := bottom_20_percent_people total_people * stimulus_per_person

-- Tax revenue returned to the government
def tax_revenue : ℕ := government_profit + cost_of_stimulus

-- The Proposition we need to prove
theorem ratio_of_tax_revenue_to_cost_of_stimulus :
  tax_revenue / cost_of_stimulus = 5 :=
by
  sorry

end ratio_of_tax_revenue_to_cost_of_stimulus_l172_17259


namespace boat_speed_still_water_l172_17247

/-- Proof that the speed of the boat in still water is 10 km/hr given the conditions -/
theorem boat_speed_still_water (V_b V_s : ℝ) 
  (cond1 : V_b + V_s = 15) 
  (cond2 : V_b - V_s = 5) : 
  V_b = 10 :=
by
  sorry

end boat_speed_still_water_l172_17247


namespace sum_of_fractions_l172_17283

theorem sum_of_fractions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/2 + x) + f (1/2 - x) = 2) :
  f (1 / 8) + f (2 / 8) + f (3 / 8) + f (4 / 8) + 
  f (5 / 8) + f (6 / 8) + f (7 / 8) = 7 :=
by 
  sorry

end sum_of_fractions_l172_17283


namespace sum_possible_x_values_in_isosceles_triangle_l172_17230

def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

def valid_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem sum_possible_x_values_in_isosceles_triangle :
  ∃ (x1 x2 x3 : ℝ), isosceles_triangle 80 x1 x1 ∧ isosceles_triangle x2 80 80 ∧ isosceles_triangle 80 x3 x3 ∧ 
  valid_triangle 80 x1 x1 ∧ valid_triangle x2 80 80 ∧ valid_triangle 80 x3 x3 ∧ 
  x1 + x2 + x3 = 150 :=
by
  sorry

end sum_possible_x_values_in_isosceles_triangle_l172_17230


namespace find_first_divisor_l172_17211

theorem find_first_divisor (x : ℕ) (k m : ℕ) (h₁ : 282 = k * x + 3) (h₂ : 282 = 9 * m + 3) : x = 31 :=
sorry

end find_first_divisor_l172_17211


namespace greyson_spent_on_fuel_l172_17246

theorem greyson_spent_on_fuel : ∀ (cost_per_refill times_refilled total_cost : ℕ), 
  cost_per_refill = 10 → 
  times_refilled = 4 → 
  total_cost = cost_per_refill * times_refilled → 
  total_cost = 40 :=
by
  intro cost_per_refill times_refilled total_cost
  intro h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end greyson_spent_on_fuel_l172_17246


namespace solve_for_a_l172_17294

theorem solve_for_a (a : ℝ) (h_pos : a > 0) 
  (h_roots : ∀ x, x^2 - 2*a*x - 3*a^2 = 0 → (x = -a ∨ x = 3*a)) 
  (h_diff : |(-a) - (3*a)| = 8) : a = 2 := 
sorry

end solve_for_a_l172_17294


namespace exists_segment_l172_17295

theorem exists_segment (f : ℚ → ℤ) : 
  ∃ (a b c : ℚ), a ≠ b ∧ c = (a + b) / 2 ∧ f a + f b ≤ 2 * f c :=
by 
  sorry

end exists_segment_l172_17295


namespace find_n_l172_17228

variable (x n : ℕ)
variable (y : ℕ) {h1 : y = 24}

theorem find_n
  (h1 : y = 24) 
  (h2 : x / y = 1 / 4) 
  (h3 : (x + n) / y = 1 / 2) : 
  n = 6 := 
sorry

end find_n_l172_17228


namespace sum_of_coefficients_l172_17252

theorem sum_of_coefficients (a b c d e x : ℝ) (h : 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) :
  a + b + c + d + e = 36 :=
by
  sorry

end sum_of_coefficients_l172_17252


namespace necessary_condition_l172_17262

theorem necessary_condition {x m : ℝ} 
  (p : |1 - (x - 1) / 3| ≤ 2)
  (q : x^2 - 2 * x + 1 - m^2 ≤ 0)
  (hm : m > 0)
  (h_np_nq : ¬(|1 - (x - 1) / 3| ≤ 2) → ¬(x^2 - 2 * x + 1 - m^2 ≤ 0))
  : m ≥ 9 :=
sorry

end necessary_condition_l172_17262


namespace boxes_of_nuts_purchased_l172_17292

theorem boxes_of_nuts_purchased (b : ℕ) (n : ℕ) (bolts_used : ℕ := 7 * 11 - 3) 
    (nuts_used : ℕ := 113 - bolts_used) (total_nuts : ℕ := nuts_used + 6) 
    (nuts_per_box : ℕ := 15) (h_bolts_boxes : b = 7) 
    (h_bolts_per_box : ∀ x, b * x = 77) 
    (h_nuts_boxes : ∃ x, n = x * nuts_per_box)
    : ∃ k, n = k * 15 ∧ k = 3 :=
by
  sorry

end boxes_of_nuts_purchased_l172_17292


namespace different_outcomes_count_l172_17253

-- Define the number of students and competitions
def num_students : ℕ := 4
def num_competitions : ℕ := 3

-- Define the proof statement
theorem different_outcomes_count : (num_competitions ^ num_students) = 81 := 
by
  -- Proof will be here
  sorry

end different_outcomes_count_l172_17253


namespace total_area_of_tickets_is_3_6_m2_l172_17268

def area_of_one_ticket (side_length_cm : ℕ) : ℕ :=
  side_length_cm * side_length_cm

def total_tickets (people : ℕ) (tickets_per_person : ℕ) : ℕ :=
  people * tickets_per_person

def total_area_cm2 (area_per_ticket_cm2 : ℕ) (number_of_tickets : ℕ) : ℕ :=
  area_per_ticket_cm2 * number_of_tickets

def convert_cm2_to_m2 (area_cm2 : ℕ) : ℚ :=
  (area_cm2 : ℚ) / 10000

theorem total_area_of_tickets_is_3_6_m2 :
  let side_length := 30
  let people := 5
  let tickets_per_person := 8
  let one_ticket_area := area_of_one_ticket side_length
  let number_of_tickets := total_tickets people tickets_per_person
  let total_area_cm2 := total_area_cm2 one_ticket_area number_of_tickets
  let total_area_m2 := convert_cm2_to_m2 total_area_cm2
  total_area_m2 = 3.6 := 
by
  sorry

end total_area_of_tickets_is_3_6_m2_l172_17268


namespace maximum_gold_coins_l172_17208

theorem maximum_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 150) : n ≤ 146 :=
by
  sorry

end maximum_gold_coins_l172_17208


namespace probability_multiple_of_3_when_die_rolled_twice_l172_17221

theorem probability_multiple_of_3_when_die_rolled_twice :
  let total_outcomes := 36
  let favorable_outcomes := 12
  (12 / 36 : ℚ) = 1 / 3 :=
by
  sorry

end probability_multiple_of_3_when_die_rolled_twice_l172_17221


namespace find_function_l172_17281

theorem find_function (f : ℝ → ℝ) :
  (∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v)) →
  (∀ u : ℝ, 0 ≤ f u) →
  (∀ x : ℝ, f x = 0) := 
  by
    sorry

end find_function_l172_17281


namespace has_two_roots_l172_17231

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end has_two_roots_l172_17231


namespace vertex_C_path_length_equals_l172_17288

noncomputable def path_length_traversed_by_C (AB BC CA : ℝ) (PQ QR : ℝ) : ℝ :=
  let BC := 3  -- length of side BC is 3 inches
  let AB := 2  -- length of side AB is 2 inches
  let CA := 4  -- length of side CA is 4 inches
  let PQ := 8  -- length of side PQ of the rectangle is 8 inches
  let QR := 6  -- length of side QR of the rectangle is 6 inches
  4 * BC * Real.pi

theorem vertex_C_path_length_equals (AB BC CA PQ QR : ℝ) :
  AB = 2 ∧ BC = 3 ∧ CA = 4 ∧ PQ = 8 ∧ QR = 6 →
  path_length_traversed_by_C AB BC CA PQ QR = 12 * Real.pi :=
by
  intros h
  have hAB : AB = 2 := h.1
  have hBC : BC = 3 := h.2.1
  have hCA : CA = 4 := h.2.2.1
  have hPQ : PQ = 8 := h.2.2.2.1
  have hQR : QR = 6 := h.2.2.2.2
  simp [path_length_traversed_by_C, hAB, hBC, hCA, hPQ, hQR]
  sorry

end vertex_C_path_length_equals_l172_17288


namespace geometric_sequence_fifth_term_l172_17222

variables (a r : ℝ) (h1 : a * r ^ 2 = 12 / 5) (h2 : a * r ^ 6 = 48)

theorem geometric_sequence_fifth_term : a * r ^ 4 = 12 / 5 := by
  sorry

end geometric_sequence_fifth_term_l172_17222


namespace find_ctg_half_l172_17203

noncomputable def ctg (x : ℝ) := 1 / (Real.tan x)

theorem find_ctg_half
  (x : ℝ)
  (h : Real.sin x - Real.cos x = (1 + 2 * Real.sqrt 2) / 3) :
  ctg (x / 2) = Real.sqrt 2 / 2 ∨ ctg (x / 2) = 3 - 2 * Real.sqrt 2 :=
by
  sorry

end find_ctg_half_l172_17203


namespace division_of_fractions_l172_17216

theorem division_of_fractions :
  (5 / 6 : ℚ) / (11 / 12) = 10 / 11 := by
  sorry

end division_of_fractions_l172_17216


namespace spheres_do_not_protrude_l172_17298

-- Define the basic parameters
variables (R r : ℝ) (h_cylinder : ℝ) (h_cone : ℝ)
-- Assume conditions
axiom cylinder_height_diameter : h_cylinder = 2 * R
axiom cone_dimensions : h_cone = h_cylinder ∧ h_cone = R

-- The given radius relationship
axiom radius_relation : R = 3 * r

-- Prove the spheres do not protrude from the container
theorem spheres_do_not_protrude (R r h_cylinder h_cone : ℝ)
  (cylinder_height_diameter : h_cylinder = 2 * R)
  (cone_dimensions : h_cone = h_cylinder ∧ h_cone = R)
  (radius_relation : R = 3 * r) : r ≤ R / 2 :=
sorry

end spheres_do_not_protrude_l172_17298


namespace minimum_value_of_f_maximum_value_of_k_l172_17299

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f : ∃ x : ℝ, 0 < x ∧ f x = -1 / Real.exp 1 :=
sorry

theorem maximum_value_of_k : ∀ x > 2, ∀ k : ℤ, (f x ≥ k * x - 2 * (k + 1)) → k ≤ 3 :=
sorry

end minimum_value_of_f_maximum_value_of_k_l172_17299


namespace intersection_of_sets_l172_17276

noncomputable def setA : Set ℕ := { x : ℕ | x^2 ≤ 4 * x ∧ x > 0 }

noncomputable def setB : Set ℕ := { x : ℕ | 2^x - 4 > 0 ∧ 2^x - 4 ≤ 4 }

theorem intersection_of_sets : { x ∈ setA | x ∈ setB } = {3} :=
by
  sorry

end intersection_of_sets_l172_17276


namespace distance_between_polar_points_l172_17269

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem distance_between_polar_points :
  let A := polar_to_rect 1 (Real.pi / 6)
  let B := polar_to_rect 2 (-Real.pi / 2)
  distance A B = Real.sqrt 7 :=
by
  sorry

end distance_between_polar_points_l172_17269


namespace functional_equation_solution_l172_17297

theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (f (xy - x)) + f (x + y) = y * f (x) + f (y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end functional_equation_solution_l172_17297


namespace sweets_leftover_candies_l172_17236

theorem sweets_leftover_candies (n : ℕ) (h : n % 8 = 5) : (3 * n) % 8 = 7 :=
sorry

end sweets_leftover_candies_l172_17236


namespace oil_already_put_in_engine_l172_17214

def oil_per_cylinder : ℕ := 8
def cylinders : ℕ := 6
def additional_needed_oil : ℕ := 32

theorem oil_already_put_in_engine :
  (oil_per_cylinder * cylinders) - additional_needed_oil = 16 := by
  sorry

end oil_already_put_in_engine_l172_17214


namespace ryan_final_tokens_l172_17219

-- Conditions
def initial_tokens : ℕ := 36
def pacman_fraction : ℚ := 2 / 3
def candy_crush_fraction : ℚ := 1 / 2
def skiball_tokens : ℕ := 7
def friend_borrowed_tokens : ℕ := 5
def friend_returned_tokens : ℕ := 8
def laser_tag_tokens : ℕ := 3
def parents_purchase_factor : ℕ := 10

-- Final Answer
theorem ryan_final_tokens : initial_tokens - 24  - 6 - skiball_tokens + friend_returned_tokens + (parents_purchase_factor * skiball_tokens) - laser_tag_tokens = 75 :=
by sorry

end ryan_final_tokens_l172_17219


namespace deposit_on_Jan_1_2008_l172_17233

-- Let a be the initial deposit amount in yuan.
-- Let x be the annual interest rate.

def compound_interest (a : ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  a * (1 + x) ^ n

theorem deposit_on_Jan_1_2008 (a : ℝ) (x : ℝ) : 
  compound_interest a x 5 = a * (1 + x) ^ 5 := 
by
  sorry

end deposit_on_Jan_1_2008_l172_17233


namespace difference_in_roses_and_orchids_l172_17226

theorem difference_in_roses_and_orchids
    (initial_roses : ℕ) (initial_orchids : ℕ) (initial_tulips : ℕ)
    (final_roses : ℕ) (final_orchids : ℕ) (final_tulips : ℕ)
    (ratio_roses_orchids_num : ℕ) (ratio_roses_orchids_den : ℕ)
    (ratio_roses_tulips_num : ℕ) (ratio_roses_tulips_den : ℕ)
    (h1 : initial_roses = 7)
    (h2 : initial_orchids = 12)
    (h3 : initial_tulips = 5)
    (h4 : final_roses = 11)
    (h5 : final_orchids = 20)
    (h6 : final_tulips = 10)
    (h7 : ratio_roses_orchids_num = 2)
    (h8 : ratio_roses_orchids_den = 5)
    (h9 : ratio_roses_tulips_num = 3)
    (h10 : ratio_roses_tulips_den = 5)
    (h11 : (final_roses : ℚ) / final_orchids = (ratio_roses_orchids_num : ℚ) / ratio_roses_orchids_den)
    (h12 : (final_roses : ℚ) / final_tulips = (ratio_roses_tulips_num : ℚ) / ratio_roses_tulips_den)
    : final_orchids - final_roses = 9 :=
by
  sorry

end difference_in_roses_and_orchids_l172_17226


namespace variance_of_data_l172_17286

def data : List ℝ := [0.7, 1, 0.8, 0.9, 1.1]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.foldr (λ x acc => x + acc) 0) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.foldr (λ x acc => (x - m) ^ 2 + acc) 0) / l.length

theorem variance_of_data :
  variance data = 0.02 :=
by
  sorry

end variance_of_data_l172_17286


namespace nadia_pies_l172_17250

variables (T R B S : ℕ)

theorem nadia_pies (h₁: R = T / 2) 
                   (h₂: B = R - 14) 
                   (h₃: S = (R + B) / 2) 
                   (h₄: T = R + B + S) :
                   R = 21 ∧ B = 7 ∧ S = 14 := 
  sorry

end nadia_pies_l172_17250


namespace least_positive_integer_l172_17278

theorem least_positive_integer (n : ℕ) :
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 6 = 5 ∧
  n % 7 = 2 ↔
  n = 83 :=
by
  sorry

end least_positive_integer_l172_17278


namespace lateral_area_of_given_cone_l172_17206

noncomputable def lateral_area_cone (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r^2 + h^2)
  (Real.pi * r * l)

theorem lateral_area_of_given_cone :
  lateral_area_cone 3 4 = 15 * Real.pi :=
by
  -- sorry to skip the proof
  sorry

end lateral_area_of_given_cone_l172_17206


namespace average_monthly_balance_l172_17240

-- Definitions for the monthly balances
def January_balance : ℝ := 120
def February_balance : ℝ := 240
def March_balance : ℝ := 180
def April_balance : ℝ := 180
def May_balance : ℝ := 160
def June_balance : ℝ := 200

-- The average monthly balance theorem statement
theorem average_monthly_balance : 
    (January_balance + February_balance + March_balance + April_balance + May_balance + June_balance) / 6 = 180 := 
by 
  sorry

end average_monthly_balance_l172_17240


namespace perpendicular_condition_l172_17212

-- Condition definition
def is_perpendicular (a : ℝ) : Prop :=
  let line1_slope := -1
  let line2_slope := - (a / 2)
  (line1_slope * line2_slope = -1)

-- Statement of the theorem
theorem perpendicular_condition (a : ℝ) :
  is_perpendicular a ↔ a = -2 :=
sorry

end perpendicular_condition_l172_17212


namespace count_valid_permutations_eq_X_l172_17234

noncomputable def valid_permutations_count : ℕ :=
sorry

theorem count_valid_permutations_eq_X : valid_permutations_count = X :=
sorry

end count_valid_permutations_eq_X_l172_17234


namespace Marty_paint_combinations_l172_17274

theorem Marty_paint_combinations :
  let colors := 5 -- blue, green, yellow, black, white
  let styles := 3 -- brush, roller, sponge
  let invalid_combinations := 1 * 1 -- white paint with roller
  let total_combinations := (4 * styles) + (1 * (styles - 1))
  total_combinations = 14 :=
by
  -- Define the total number of combinations excluding the invalid one
  let colors := 5
  let styles := 3
  let invalid_combinations := 1 -- number of invalid combinations (white with roller)
  let valid_combinations := (4 * styles) + (1 * (styles - 1))
  show valid_combinations = 14
  {
    exact rfl -- This will assert that the valid_combinations indeed equals 14
  }

end Marty_paint_combinations_l172_17274
