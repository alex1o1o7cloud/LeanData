import Mathlib

namespace range_of_a_l495_49595

theorem range_of_a (a x y : ℝ) (h1 : 77 * a = (2 * x + 2 * y) / 2) (h2 : Real.sqrt (abs a) = Real.sqrt (x * y)) :
  a ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
sorry

end range_of_a_l495_49595


namespace sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l495_49563

-- Proof for Problem 1
theorem sin_of_cos_in_third_quadrant (α : ℝ) 
  (hcos : Real.cos α = -4 / 5)
  (hquad : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3 / 5 :=
by
  sorry

-- Proof for Problem 2
theorem ratio_of_trig_functions (α : ℝ) 
  (htan : Real.tan α = -3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 7 / 2 :=
by
  sorry

end sin_of_cos_in_third_quadrant_ratio_of_trig_functions_l495_49563


namespace largest_n_unique_k_l495_49550

theorem largest_n_unique_k (n : ℕ) (h : ∃ k : ℕ, (9 / 17 : ℚ) < n / (n + k) ∧ n / (n + k) < (8 / 15 : ℚ) ∧ ∀ k' : ℕ, ((9 / 17 : ℚ) < n / (n + k') ∧ n / (n + k') < (8 / 15 : ℚ)) → k' = k) : n = 72 :=
sorry

end largest_n_unique_k_l495_49550


namespace add_and_round_58_29_l495_49537

def add_and_round_to_nearest_ten (a b : ℕ) : ℕ :=
  let sum := a + b
  let rounded_sum := if sum % 10 < 5 then sum - (sum % 10) else sum + (10 - sum % 10)
  rounded_sum

theorem add_and_round_58_29 : add_and_round_to_nearest_ten 58 29 = 90 := by
  sorry

end add_and_round_58_29_l495_49537


namespace scarlet_savings_l495_49577

theorem scarlet_savings : 
  let initial_savings := 80
  let cost_earrings := 23
  let cost_necklace := 48
  let total_spent := cost_earrings + cost_necklace
  initial_savings - total_spent = 9 := 
by 
  sorry

end scarlet_savings_l495_49577


namespace fraction_in_between_l495_49540

variable {r u s v : ℤ}

/-- Assumes r, u, s, v be positive integers such that su - rv = 1 --/
theorem fraction_in_between (h1 : r > 0) (h2 : u > 0) (h3 : s > 0) (h4 : v > 0) (h5 : s * u - r * v = 1) :
  ∀ ⦃x num denom : ℤ⦄, r * denom = num * u → s * denom = (num + 1) * v → r * v ≤ num * denom - 1 / u * v * denom
   ∧ num * denom - 1 / u * v * denom ≤ s * v :=
sorry

end fraction_in_between_l495_49540


namespace parabola_vertex_origin_through_point_l495_49542

theorem parabola_vertex_origin_through_point :
  (∃ p, p > 0 ∧ x^2 = 2 * p * y ∧ (x, y) = (-4, 4) → x^2 = 4 * y) ∨
  (∃ p, p > 0 ∧ y^2 = -2 * p * x ∧ (x, y) = (-4, 4) → y^2 = -4 * x) :=
sorry

end parabola_vertex_origin_through_point_l495_49542


namespace symmetrical_implies_congruent_l495_49578

-- Define a structure to represent figures
structure Figure where
  segments : Set ℕ
  angles : Set ℕ

-- Define symmetry about a line
def is_symmetrical_about_line (f1 f2 : Figure) : Prop :=
  ∀ s ∈ f1.segments, s ∈ f2.segments ∧ ∀ a ∈ f1.angles, a ∈ f2.angles

-- Define congruent figures
def are_congruent (f1 f2 : Figure) : Prop :=
  f1.segments = f2.segments ∧ f1.angles = f2.angles

-- Lean 4 statement of the proof problem
theorem symmetrical_implies_congruent (f1 f2 : Figure) (h : is_symmetrical_about_line f1 f2) : are_congruent f1 f2 :=
by
  sorry

end symmetrical_implies_congruent_l495_49578


namespace complex_sum_cubics_eq_zero_l495_49512

-- Define the hypothesis: omega is a nonreal root of x^3 = 1
def is_nonreal_root_of_cubic (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω ≠ 1

-- Now state the theorem to prove the expression evaluates to 0
theorem complex_sum_cubics_eq_zero (ω : ℂ) (h : is_nonreal_root_of_cubic ω) :
  (2 - 2*ω + 2*ω^2)^3 + (2 + 2*ω - 2*ω^2)^3 = 0 :=
by
  -- This is where the proof would go. 
  sorry

end complex_sum_cubics_eq_zero_l495_49512


namespace one_cubic_foot_is_1728_cubic_inches_l495_49564

-- Define the basic equivalence of feet to inches.
def foot_to_inch : ℝ := 12

-- Define the conversion from cubic feet to cubic inches.
def cubic_foot_to_cubic_inch (cubic_feet : ℝ) : ℝ :=
  (foot_to_inch * cubic_feet) ^ 3

-- State the theorem to prove the equivalence in cubic measurement.
theorem one_cubic_foot_is_1728_cubic_inches : cubic_foot_to_cubic_inch 1 = 1728 :=
  sorry -- Proof skipped.

end one_cubic_foot_is_1728_cubic_inches_l495_49564


namespace sum_of_number_and_reverse_l495_49552

theorem sum_of_number_and_reverse :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → ((10 * a + b) - (10 * b + a) = 7 * (a + b)) → a = 8 * b → 
  (10 * a + b) + (10 * b + a) = 99 :=
by
  intros a b conditions eq diff
  sorry

end sum_of_number_and_reverse_l495_49552


namespace hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l495_49580

theorem hexagon_exists_equal_sides_four_equal_angles : 
  ∃ (A B C D E F : Type) (AB BC CD DE EF FA : ℝ) (angle_A angle_B angle_C angle_D angle_E angle_F : ℝ), 
  (AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ FA = AB) ∧ 
  (angle_A = angle_B ∧ angle_B = angle_E ∧ angle_E = angle_F) ∧ 
  4 * angle_A + angle_C + angle_D = 720 :=
sorry

theorem hexagon_exists_equal_angles_four_equal_sides :
  ∃ (A B C D E F : Type) (AB BC CD DA : ℝ) (angle : ℝ), 
  (angle_A = angle_B ∧ angle_B = angle_C ∧ angle_C = angle_D ∧ angle_D = angle_E ∧ angle_E = angle_F ∧ angle_F = angle_A) ∧ 
  (AB = BC ∧ BC = CD ∧ CD = DA) :=
sorry

end hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l495_49580


namespace symmetric_point_is_correct_l495_49516

/-- A point in 2D Cartesian coordinates -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defining the point P with given coordinates -/
def P : Point := {x := 2, y := 3}

/-- Defining the symmetry of a point with respect to the origin -/
def symmetric_origin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- States that the symmetric point of P (2, 3) with respect to the origin is (-2, -3) -/
theorem symmetric_point_is_correct :
  symmetric_origin P = {x := -2, y := -3} :=
by
  sorry

end symmetric_point_is_correct_l495_49516


namespace min_sum_of_product_2004_l495_49592

theorem min_sum_of_product_2004 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
    (hxyz : x * y * z = 2004) : x + y + z ≥ 174 ∧ ∃ (a b c : ℕ), a * b * c = 2004 ∧ a + b + c = 174 :=
by sorry

end min_sum_of_product_2004_l495_49592


namespace smallest_n_divisible_l495_49562

theorem smallest_n_divisible {n : ℕ} : 
  (∃ n : ℕ, n > 0 ∧ 18 ∣ n^2 ∧ 1152 ∣ n^3 ∧ 
    (∀ m : ℕ, m > 0 → 18 ∣ m^2 → 1152 ∣ m^3 → n ≤ m)) :=
  sorry

end smallest_n_divisible_l495_49562


namespace non_congruent_rectangles_l495_49587

theorem non_congruent_rectangles (h w : ℕ) (hp : 2 * (h + w) = 80) :
  ∃ n, n = 20 := by
  sorry

end non_congruent_rectangles_l495_49587


namespace smallest_possible_b_l495_49584

-- Definitions of conditions
variables {a b c : ℤ}

-- Conditions expressed in Lean
def is_geometric_progression (a b c : ℤ) : Prop := b^2 = a * c
def is_arithmetic_progression (a b c : ℤ) : Prop := a + b = 2 * c

-- The theorem statement
theorem smallest_possible_b (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c) 
  (hg : is_geometric_progression a b c) 
  (ha : is_arithmetic_progression a c b) : b = 2 := sorry

end smallest_possible_b_l495_49584


namespace find_side_a_l495_49588

noncomputable def maximum_area (A b c : ℝ) : Prop :=
  A = 2 * Real.pi / 3 ∧ (b + 2 * c = 8) ∧ 
  ((1 / 2) * b * c * Real.sin (2 * Real.pi / 3) = (Real.sqrt 3 / 2) * c * (4 - c) ∧ 
   (∀ (c' : ℝ), (Real.sqrt 3 / 2) * c' * (4 - c') ≤ 2 * Real.sqrt 3) ∧ 
   c = 2)

theorem find_side_a (A b c a : ℝ) (h : maximum_area A b c) :
  a = 2 * Real.sqrt 7 := 
by
  sorry

end find_side_a_l495_49588


namespace tenth_digit_of_expression_l495_49548

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def tenth_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tenth_digit_of_expression : 
  tenth_digit ((factorial 5 * factorial 5 - factorial 5 * factorial 3) / 5) = 3 :=
by
  -- proof omitted
  sorry

end tenth_digit_of_expression_l495_49548


namespace f_f_4_eq_1_l495_49596

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x

theorem f_f_4_eq_1 : f (f 4) = 1 := by
  sorry

end f_f_4_eq_1_l495_49596


namespace evie_gave_2_shells_to_brother_l495_49566

def daily_shells : ℕ := 10
def days : ℕ := 6
def remaining_shells : ℕ := 58

def total_shells : ℕ := daily_shells * days
def shells_given : ℕ := total_shells - remaining_shells

theorem evie_gave_2_shells_to_brother :
  shells_given = 2 :=
by
  sorry

end evie_gave_2_shells_to_brother_l495_49566


namespace contractor_original_days_l495_49560

noncomputable def original_days (total_laborers absent_laborers working_laborers days_worked : ℝ) : ℝ :=
  (working_laborers * days_worked) / (total_laborers - absent_laborers)

-- Our conditions:
def total_laborers : ℝ := 21.67
def absent_laborers : ℝ := 5
def working_laborers : ℝ := 16.67
def days_worked : ℝ := 13

-- Our main theorem:
theorem contractor_original_days :
  original_days total_laborers absent_laborers working_laborers days_worked = 10 := 
by
  sorry

end contractor_original_days_l495_49560


namespace extra_people_needed_l495_49531

theorem extra_people_needed 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (final_time : ℕ) 
  (work_done : ℕ) 
  (all_paint_same_rate : initial_people * initial_time = work_done) :
  initial_people = 8 →
  initial_time = 3 →
  final_time = 2 →
  work_done = 24 →
  ∃ extra_people : ℕ, extra_people = 4 :=
by
  sorry

end extra_people_needed_l495_49531


namespace max_remainder_is_8_l495_49541

theorem max_remainder_is_8 (d q r : ℕ) (h1 : d = 9) (h2 : q = 6) (h3 : r < d) : 
  r ≤ (d - 1) :=
by 
  sorry

end max_remainder_is_8_l495_49541


namespace find_initial_number_l495_49575

theorem find_initial_number (N : ℝ) (h : ∃ k : ℝ, 330 * k = N + 69.00000000008731) : 
  ∃ m : ℝ, N = 330 * m - 69.00000000008731 :=
by
  sorry

end find_initial_number_l495_49575


namespace Meghan_total_money_l495_49593

theorem Meghan_total_money (h100 : ℕ) (h50 : ℕ) (h10 : ℕ) : 
  h100 = 2 → h50 = 5 → h10 = 10 → 100 * h100 + 50 * h50 + 10 * h10 = 550 :=
by
  sorry

end Meghan_total_money_l495_49593


namespace velvet_needed_for_box_l495_49509

theorem velvet_needed_for_box :
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  long_side_area + short_side_area + top_and_bottom_area = 236 := by
{
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  sorry
}

end velvet_needed_for_box_l495_49509


namespace georgia_carnations_proof_l495_49559

-- Define the conditions
def carnation_cost : ℝ := 0.50
def dozen_cost : ℝ := 4.00
def friends_carnations : ℕ := 14
def total_spent : ℝ := 25.00

-- Define the answer
def teachers_dozen : ℕ := 4

-- Prove the main statement
theorem georgia_carnations_proof : 
  (total_spent - (friends_carnations * carnation_cost)) / dozen_cost = teachers_dozen :=
by
  sorry

end georgia_carnations_proof_l495_49559


namespace min_value_of_a_plus_b_l495_49523

theorem min_value_of_a_plus_b (a b c : ℝ) (C : ℝ) 
  (hC : C = 60) 
  (h : (a + b)^2 - c^2 = 4) : 
  a + b ≥ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end min_value_of_a_plus_b_l495_49523


namespace cost_per_person_l495_49565

theorem cost_per_person (total_cost : ℕ) (num_people : ℕ) (h1 : total_cost = 30000) (h2 : num_people = 300) : total_cost / num_people = 100 := by
  -- No proof provided, only the theorem statement
  sorry

end cost_per_person_l495_49565


namespace slope_at_A_is_7_l495_49501

def curve (x : ℝ) : ℝ := x^2 + 3 * x

def point_A : ℝ × ℝ := (2, 10)

theorem slope_at_A_is_7 : (deriv curve 2) = 7 := 
by
  sorry

end slope_at_A_is_7_l495_49501


namespace gcd_231_154_l495_49546

def find_gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_231_154 : find_gcd 231 154 = 77 := by
  sorry

end gcd_231_154_l495_49546


namespace arithmetic_sequence_count_l495_49525

theorem arithmetic_sequence_count :
  ∃ n : ℕ, 2 + (n-1) * 5 = 2507 ∧ n = 502 :=
by
  sorry

end arithmetic_sequence_count_l495_49525


namespace stop_shooting_after_2nd_scoring_5_points_eq_l495_49590

/-
Define the conditions and problem statement in Lean:
- Each person can shoot up to 10 times.
- Student A's shooting probability for each shot is 2/3.
- If student A stops shooting at the nth consecutive shot, they score 12-n points.
- We need to prove the probability that student A stops shooting right after the 2nd shot and scores 5 points is 8/729.
-/
def student_shoot_probability (shots : List Bool) (p : ℚ) : ℚ :=
  shots.foldr (λ s acc => if s then p * acc else (1 - p) * acc) 1

def stop_shooting_probability : ℚ :=
  let shots : List Bool := [false, true, false, false, false, true, true] -- represents misses and hits
  student_shoot_probability shots (2/3)

theorem stop_shooting_after_2nd_scoring_5_points_eq :
  stop_shooting_probability = (8 / 729) :=
sorry

end stop_shooting_after_2nd_scoring_5_points_eq_l495_49590


namespace ttakjis_count_l495_49545

theorem ttakjis_count (n : ℕ) (initial_residual new_residual total_ttakjis : ℕ) :
  initial_residual = 36 → 
  new_residual = 3 → 
  total_ttakjis = n^2 + initial_residual → 
  total_ttakjis = (n + 1)^2 + new_residual → 
  total_ttakjis = 292 :=
by
  sorry

end ttakjis_count_l495_49545


namespace raking_yard_time_l495_49507

theorem raking_yard_time (your_rate : ℚ) (brother_rate : ℚ) (combined_rate : ℚ) (combined_time : ℚ) :
  your_rate = 1 / 30 ∧ 
  brother_rate = 1 / 45 ∧ 
  combined_rate = your_rate + brother_rate ∧ 
  combined_time = 1 / combined_rate → 
  combined_time = 18 := 
by 
  sorry

end raking_yard_time_l495_49507


namespace cos_double_angle_l495_49576

theorem cos_double_angle (α : ℝ) (h : Real.cos (α + Real.pi / 2) = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
by 
  sorry

end cos_double_angle_l495_49576


namespace security_deposit_correct_l495_49572

-- Definitions (Conditions)
def daily_rate : ℝ := 125
def pet_fee_per_dog : ℝ := 100
def number_of_dogs : ℕ := 2
def tourism_tax_rate : ℝ := 0.10
def service_fee_rate : ℝ := 0.20
def activity_cost_per_person : ℝ := 45
def number_of_activities_per_person : ℕ := 3
def number_of_people : ℕ := 2
def security_deposit_rate : ℝ := 0.50
def usd_to_euro_conversion_rate : ℝ := 0.83

-- Function to calculate total cost
def total_cost_in_euros : ℝ :=
  let rental_cost := daily_rate * 14
  let pet_cost := pet_fee_per_dog * number_of_dogs
  let tourism_tax := tourism_tax_rate * rental_cost
  let service_fee := service_fee_rate * rental_cost
  let cabin_total := rental_cost + pet_cost + tourism_tax + service_fee
  let activities_total := number_of_activities_per_person * activity_cost_per_person * number_of_people
  let total_cost := cabin_total + activities_total
  let security_deposit_usd := security_deposit_rate * total_cost
  security_deposit_usd * usd_to_euro_conversion_rate

-- Theorem to prove
theorem security_deposit_correct :
  total_cost_in_euros = 1139.18 := 
sorry

end security_deposit_correct_l495_49572


namespace log_neq_x_minus_one_l495_49535

theorem log_neq_x_minus_one (x : ℝ) (h₁ : 0 < x) : Real.log x ≠ x - 1 :=
sorry

end log_neq_x_minus_one_l495_49535


namespace solution_set_l495_49551

-- Define the function and the conditions
variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

-- Problem statement
theorem solution_set (hf_even : is_even f)
                     (hf_increasing : increasing_on f (Set.Ioi 0))
                     (hf_value : f (-2013) = 0) :
  {x | x * f x < 0} = {x | x < -2013 ∨ (0 < x ∧ x < 2013)} :=
by
  sorry

end solution_set_l495_49551


namespace number_of_correct_statements_l495_49569

def input_statement (s : String) : Prop :=
  s = "INPUT a; b; c"

def output_statement (s : String) : Prop :=
  s = "A=4"

def assignment_statement1 (s : String) : Prop :=
  s = "3=B"

def assignment_statement2 (s : String) : Prop :=
  s = "A=B=-2"

theorem number_of_correct_statements :
    input_statement "INPUT a; b; c" = false ∧
    output_statement "A=4" = false ∧
    assignment_statement1 "3=B" = false ∧
    assignment_statement2 "A=B=-2" = false :=
sorry

end number_of_correct_statements_l495_49569


namespace abigail_saving_period_l495_49521

-- Define the conditions
def amount_saved_each_month : ℕ := 4000
def total_amount_saved : ℕ := 48000

-- State the theorem
theorem abigail_saving_period : total_amount_saved / amount_saved_each_month = 12 := by
  -- Proof would go here
  sorry

end abigail_saving_period_l495_49521


namespace min_side_b_of_triangle_l495_49583

theorem min_side_b_of_triangle (A B C a b c : ℝ) 
  (h_arith_seq : 2 * B = A + C)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides_opposite : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h_given_eq : 3 * a * c + b^2 = 25) :
  b ≥ 5 / 2 :=
  sorry

end min_side_b_of_triangle_l495_49583


namespace multiply_powers_l495_49581

theorem multiply_powers (a : ℝ) : (a^3) * (a^3) = a^6 := by
  sorry

end multiply_powers_l495_49581


namespace blocks_from_gallery_to_work_l495_49598

theorem blocks_from_gallery_to_work (b_store b_gallery b_already_walked b_more_to_work total_blocks blocks_to_work_from_gallery : ℕ) 
  (h1 : b_store = 11)
  (h2 : b_gallery = 6)
  (h3 : b_already_walked = 5)
  (h4 : b_more_to_work = 20)
  (h5 : total_blocks = b_store + b_gallery + b_more_to_work)
  (h6 : blocks_to_work_from_gallery = total_blocks - b_already_walked - b_store - b_gallery) :
  blocks_to_work_from_gallery = 15 :=
by
  sorry

end blocks_from_gallery_to_work_l495_49598


namespace min_time_proof_l495_49533

/-
  Problem: 
  Given 5 colored lights that each can shine in one of the colors {red, orange, yellow, green, blue},
  and the colors are all different, and the interval between two consecutive flashes is 5 seconds.
  Define the ordered shining of these 5 lights once as a "flash", where each flash lasts 5 seconds.
  We need to show that the minimum time required to achieve all different flashes (120 flashes) is equal to 1195 seconds.
-/

def min_time_required : Nat :=
  let num_flashes := 5 * 4 * 3 * 2 * 1
  let flash_time := 5 * num_flashes
  let interval_time := 5 * (num_flashes - 1)
  flash_time + interval_time

theorem min_time_proof : min_time_required = 1195 := by
  sorry

end min_time_proof_l495_49533


namespace range_of_m_l495_49597

-- Definitions from conditions
def p (m : ℝ) : Prop := (∃ x y : ℝ, 2 * x^2 / m + y^2 / (m - 1) = 1)
def q (m : ℝ) : Prop := ∃ x1 : ℝ, 8 * x1^2 - 8 * m * x1 + 7 * m - 6 = 0
def proposition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬ (p m ∧ q m)

-- Proof statement
theorem range_of_m (m : ℝ) (h : proposition m) : (m ≤ 1 ∨ (3 / 2 < m ∧ m < 2)) :=
by
  sorry

end range_of_m_l495_49597


namespace increase_average_by_3_l495_49530

theorem increase_average_by_3 (x : ℕ) (average_initial : ℕ := 32) (matches_initial : ℕ := 10) (score_11th_match : ℕ := 65) :
  (matches_initial * average_initial + score_11th_match = 11 * (average_initial + x)) → x = 3 := 
sorry

end increase_average_by_3_l495_49530


namespace line_intersects_circle_l495_49571

theorem line_intersects_circle 
  (k : ℝ)
  (x y : ℝ)
  (h_line : x = 0 ∨ y = -2)
  (h_circle : (x - 1)^2 + (y + 2)^2 = 16) :
  (-2 - -2)^2 < 16 := by
  sorry

end line_intersects_circle_l495_49571


namespace slopes_of_intersecting_line_l495_49579

theorem slopes_of_intersecting_line {m : ℝ} :
  (∃ x y : ℝ, y = m * x + 4 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ Set.Iic (-Real.sqrt 0.48) ∪ Set.Ici (Real.sqrt 0.48) :=
by
  sorry

end slopes_of_intersecting_line_l495_49579


namespace remainder_of_sum_is_five_l495_49524

theorem remainder_of_sum_is_five (a b c d : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) (hd : d % 15 = 14) :
  (a + b + c + d) % 15 = 5 :=
by
  sorry

end remainder_of_sum_is_five_l495_49524


namespace dot_product_parallel_vectors_is_minus_ten_l495_49570

-- Definitions from the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -4)
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

theorem dot_product_parallel_vectors_is_minus_ten (x : ℝ) (h : are_parallel vector_a (vector_b x)) : (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2) = -10 :=
by
  sorry

end dot_product_parallel_vectors_is_minus_ten_l495_49570


namespace arccos_one_eq_zero_l495_49561

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l495_49561


namespace parallel_tangents_a3_plus_b2_plus_d_eq_seven_l495_49599

theorem parallel_tangents_a3_plus_b2_plus_d_eq_seven:
  ∃ (a b d : ℝ),
  (1, 1).snd = a * (1:ℝ)^3 + b * (1:ℝ)^2 + d ∧
  (-1, -3).snd = a * (-1:ℝ)^3 + b * (-1:ℝ)^2 + d ∧
  (3 * a * (1:ℝ)^2 + 2 * b * 1 = 3 * a * (-1:ℝ)^2 + 2 * b * -1) ∧
  a^3 + b^2 + d = 7 := 
sorry

end parallel_tangents_a3_plus_b2_plus_d_eq_seven_l495_49599


namespace solution_set_l495_49503

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set (x : ℝ) : 
  ((x > 1 ∧ x < 2 ∨ x > Real.sqrt 10)) ↔ f x > 2 :=
sorry

end solution_set_l495_49503


namespace quadratic_real_roots_l495_49505

theorem quadratic_real_roots (a: ℝ) :
  ∀ x: ℝ, (a-6) * x^2 - 8 * x + 9 = 0 ↔ (a ≤ 70/9 ∧ a ≠ 6) :=
  sorry

end quadratic_real_roots_l495_49505


namespace total_amount_l495_49514

noncomputable def mark_amount : ℝ := 5 / 8

noncomputable def carolyn_amount : ℝ := 7 / 20

theorem total_amount : mark_amount + carolyn_amount = 0.975 := by
  sorry

end total_amount_l495_49514


namespace interval_length_l495_49549

theorem interval_length (x : ℝ) :
  (1/x > 1/2) ∧ (Real.sin x > 1/2) → (2 - Real.pi / 6 = 1.48) :=
by
  sorry

end interval_length_l495_49549


namespace system_of_inequalities_l495_49544

theorem system_of_inequalities :
  ∃ (a b : ℤ), 
  (11 > 2 * a - b) ∧ 
  (25 > 2 * b - a) ∧ 
  (42 < 3 * b - a) ∧ 
  (46 < 2 * a + b) ∧ 
  (a = 14) ∧ 
  (b = 19) := 
sorry

end system_of_inequalities_l495_49544


namespace intersection_is_target_set_l495_49591

-- Define sets A and B
def is_in_A (x : ℝ) : Prop := |x - 1| < 2
def is_in_B (x : ℝ) : Prop := x^2 < 4

-- Define the intersection A ∩ B
def is_in_intersection (x : ℝ) : Prop := is_in_A x ∧ is_in_B x

-- Define the target set
def is_in_target_set (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Statement to prove
theorem intersection_is_target_set : 
  ∀ x : ℝ, is_in_intersection x ↔ is_in_target_set x := sorry

end intersection_is_target_set_l495_49591


namespace fewest_keystrokes_One_to_410_l495_49500

noncomputable def fewest_keystrokes (start : ℕ) (target : ℕ) : ℕ :=
if target = 410 then 10 else sorry

theorem fewest_keystrokes_One_to_410 : fewest_keystrokes 1 410 = 10 :=
by
  sorry

end fewest_keystrokes_One_to_410_l495_49500


namespace committee_selection_count_l495_49502

-- Definition of the problem condition: Club of 12 people, one specific person must always be on the committee.
def club_size : ℕ := 12
def committee_size : ℕ := 4
def specific_person_included : ℕ := 1

-- Number of ways to choose 3 members from the other 11 people
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem committee_selection_count : choose 11 3 = 165 := 
  sorry

end committee_selection_count_l495_49502


namespace value_of_R_l495_49573

theorem value_of_R (R : ℝ) (hR_pos : 0 < R)
  (h_line : ∀ x y : ℝ, x + y = 2 * R)
  (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = R) :
  R = (3 + Real.sqrt 5) / 4 ∨ R = (3 - Real.sqrt 5) / 4 :=
by
  sorry

end value_of_R_l495_49573


namespace annie_overtakes_bonnie_l495_49519

-- Define the conditions
def track_circumference : ℝ := 300
def bonnie_speed (v : ℝ) : ℝ := v
def annie_speed (v : ℝ) : ℝ := 1.5 * v

-- Define the statement for proving the number of laps completed by Annie when she first overtakes Bonnie
theorem annie_overtakes_bonnie (v t : ℝ) : 
  bonnie_speed v * t = track_circumference * 2 → 
  annie_speed v * t = track_circumference * 3 :=
by
  sorry

end annie_overtakes_bonnie_l495_49519


namespace ant_travel_finite_path_exists_l495_49553

theorem ant_travel_finite_path_exists :
  ∃ (x y z t : ℝ), |x| < |y - z + t| ∧ |y| < |x - z + t| ∧ 
                   |z| < |x - y + t| ∧ |t| < |x - y + z| :=
by
  sorry

end ant_travel_finite_path_exists_l495_49553


namespace product_of_two_numbers_l495_49555

theorem product_of_two_numbers :
  ∃ (a b : ℚ), (∀ k : ℚ, a = k + b) ∧ (∀ k : ℚ, a + b = 8 * k) ∧ (∀ k : ℚ, a * b = 40 * k) ∧ (a * b = 6400 / 63) :=
by {
  sorry
}

end product_of_two_numbers_l495_49555


namespace g_of_1986_l495_49554

-- Define the function g and its properties
noncomputable def g : ℕ → ℤ :=
sorry  -- Placeholder for the actual definition according to the conditions

axiom g_is_defined (x : ℕ) : x ≥ 0 → ∃ y : ℤ, g x = y
axiom g_at_1 : g 1 = 1
axiom g_add (a b : ℕ) (h_a : a ≥ 0) (h_b : b ≥ 0) : g (a + b) = g a + g b - 3 * g (a * b) + 1

-- Lean statement for the proof problem
theorem g_of_1986 : g 1986 = 0 :=
sorry

end g_of_1986_l495_49554


namespace emails_in_morning_and_afternoon_l495_49511

-- Conditions
def morning_emails : Nat := 5
def afternoon_emails : Nat := 8

-- Theorem statement
theorem emails_in_morning_and_afternoon : morning_emails + afternoon_emails = 13 := by
  -- Proof goes here, but adding sorry for now
  sorry

end emails_in_morning_and_afternoon_l495_49511


namespace polynomial_evaluation_l495_49539

theorem polynomial_evaluation (P : ℕ → ℝ) (n : ℕ) 
  (h_degree : ∀ k : ℕ, k ≤ n → P k = k / (k + 1)) 
  (h_poly : ∀ k : ℕ, ∃ a : ℝ, P k = a * k ^ n) : 
  P (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) :=
by 
  sorry

end polynomial_evaluation_l495_49539


namespace solve_system_l495_49586

-- Define the conditions of the system of equations
def condition1 (x y : ℤ) := 4 * x - 3 * y = -13
def condition2 (x y : ℤ) := 5 * x + 3 * y = -14

-- Define the proof goal using the conditions
theorem solve_system : ∃ (x y : ℤ), condition1 x y ∧ condition2 x y ∧ x = -3 ∧ y = 1 / 3 :=
by
  sorry

end solve_system_l495_49586


namespace solution_proof_l495_49589

noncomputable def proof_problem : Prop :=
  ∀ (x : ℝ), x ≠ 1 → (1 - 1 / (x - 1) = 2 * x / (1 - x)) → x = 2 / 3

theorem solution_proof : proof_problem := 
by
  sorry

end solution_proof_l495_49589


namespace find_n_l495_49506

noncomputable def positive_geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : positive_geometric_sequence) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def conditions (a : positive_geometric_sequence) :=
  is_geometric_sequence a ∧
  a 0 * a 1 * a 2 = 4 ∧
  a 3 * a 4 * a 5 = 12 ∧
  ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324

theorem find_n (a : positive_geometric_sequence) (h : conditions a) : ∃ n : ℕ, n = 14 :=
by
  sorry

end find_n_l495_49506


namespace probability_is_4_over_5_l495_49567

variable (total_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ)
variable (total_balls_eq : total_balls = 60) (red_balls_eq : red_balls = 5) (purple_balls_eq : purple_balls = 7)

def probability_neither_red_nor_purple : ℚ :=
  let favorable_outcomes := total_balls - (red_balls + purple_balls)
  let total_outcomes := total_balls
  favorable_outcomes / total_outcomes

theorem probability_is_4_over_5 :
  probability_neither_red_nor_purple total_balls red_balls purple_balls = 4 / 5 :=
by
  have h1: total_balls = 60 := total_balls_eq
  have h2: red_balls = 5 := red_balls_eq
  have h3: purple_balls = 7 := purple_balls_eq
  sorry

end probability_is_4_over_5_l495_49567


namespace average_of_solutions_l495_49518

-- Define the quadratic equation condition
def quadratic_eq : Prop := ∃ x : ℂ, 3*x^2 - 4*x + 1 = 0

-- State the theorem
theorem average_of_solutions : quadratic_eq → (∃ avg : ℂ, avg = 2 / 3) :=
by
  sorry

end average_of_solutions_l495_49518


namespace standard_eq_circle_l495_49522

noncomputable def circle_eq (x y : ℝ) (r : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16 ∨ (x - 1)^2 + (y + 1)^2 = 1

theorem standard_eq_circle {x y : ℝ}
  (h1 : 5 * x - 3 * y = 8)
  (h2 : abs x = abs y) :
  ∃ r : ℝ, circle_eq x y r :=
by {
  sorry
}

end standard_eq_circle_l495_49522


namespace circle_symmetric_eq_l495_49526

theorem circle_symmetric_eq :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 2 * y + 1 = 0) → (x - y + 3 = 0) → 
  (∃ (a b : ℝ), (a + 2)^2 + (b - 2)^2 = 1) :=
by
  intros x y hc hl
  sorry

end circle_symmetric_eq_l495_49526


namespace low_card_value_is_one_l495_49557

-- Definitions and setting up the conditions
def num_high_cards : ℕ := 26
def num_low_cards : ℕ := 26
def high_card_points : ℕ := 2
def draw_scenarios : ℕ := 4

-- The point value of a low card L
noncomputable def low_card_points : ℕ :=
  if num_high_cards = 26 ∧ num_low_cards = 26 ∧ high_card_points = 2
     ∧ draw_scenarios = 4
  then 1 else 0 

theorem low_card_value_is_one :
  low_card_points = 1 :=
by
  sorry

end low_card_value_is_one_l495_49557


namespace problem_1_problem_2_l495_49517

open Set -- to work with sets conveniently

noncomputable section -- to allow the use of real numbers and other non-constructive elements

-- Define U as the set of all real numbers
def U : Set ℝ := univ

-- Define M as the set of all x such that y = sqrt(x - 2)
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 2) }

-- Define N as the set of all x such that x < 1 or x > 3
def N : Set ℝ := {x : ℝ | x < 1 ∨ x > 3}

-- Statement to prove (1)
theorem problem_1 : M ∪ N = {x : ℝ | x < 1 ∨ x ≥ 2} := sorry

-- Statement to prove (2)
theorem problem_2 : M ∩ (compl N) = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := sorry

end problem_1_problem_2_l495_49517


namespace arith_seq_sum_of_terms_l495_49504

theorem arith_seq_sum_of_terms 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_pos_diff : 0 < d) 
  (h_first_three_sum : a 0 + a 1 + a 2 = 15) 
  (h_first_three_prod : a 0 * a 1 * a 2 = 80) : 
  a 10 + a 11 + a 12 = 105 := sorry

end arith_seq_sum_of_terms_l495_49504


namespace jay_savings_first_week_l495_49594

theorem jay_savings_first_week :
  ∀ (x : ℕ), (x + (x + 10) + (x + 20) + (x + 30) = 60) → x = 0 :=
by
  intro x h
  sorry

end jay_savings_first_week_l495_49594


namespace probability_at_least_one_woman_selected_l495_49582

open Classical

noncomputable def probability_of_selecting_at_least_one_woman : ℚ :=
  1 - (10 / 15) * (9 / 14) * (8 / 13) * (7 / 12) * (6 / 11)

theorem probability_at_least_one_woman_selected :
  probability_of_selecting_at_least_one_woman = 917 / 1001 :=
sorry

end probability_at_least_one_woman_selected_l495_49582


namespace intersection_of_complement_l495_49515

open Set

variable (U : Set ℤ) (A B : Set ℤ)

def complement (U A : Set ℤ) : Set ℤ := U \ A

theorem intersection_of_complement (hU : U = {-1, 0, 1, 2, 3, 4})
  (hA : A = {1, 2, 3, 4}) (hB : B = {0, 2}) :
  (complement U A) ∩ B = {0} :=
by
  sorry

end intersection_of_complement_l495_49515


namespace sector_area_l495_49585

theorem sector_area (α : ℝ) (l : ℝ) (r : ℝ) (S : ℝ) : 
  α = 1 ∧ l = 6 ∧ l = α * r → S = (1/2) * α * r ^ 2 → S = 18 :=
by
  intros h h' 
  sorry

end sector_area_l495_49585


namespace cash_still_missing_l495_49538

theorem cash_still_missing (c : ℝ) (h : c > 0) :
  (1 : ℝ) - (8 / 9) = (1 / 9 : ℝ) :=
by
  sorry

end cash_still_missing_l495_49538


namespace problem_correct_choice_l495_49534

-- Definitions of the propositions
def p : Prop := ∃ n : ℕ, 3 = 2 * n + 1
def q : Prop := ∃ n : ℕ, 5 = 2 * n

-- The problem statement
theorem problem_correct_choice : p ∨ q :=
sorry

end problem_correct_choice_l495_49534


namespace probability_not_snowing_l495_49558

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 5 / 8) : 1 - p_snow = 3 / 8 :=
by
  rw [h]
  sorry

end probability_not_snowing_l495_49558


namespace nicky_catches_up_time_l495_49508

theorem nicky_catches_up_time
  (head_start : ℕ := 12)
  (cristina_speed : ℕ := 5)
  (nicky_speed : ℕ := 3)
  (head_start_distance : ℕ := nicky_speed * head_start)
  (time_to_catch_up : ℕ := 36 / 2) -- 36 is the head start distance of 36 meters
  (total_time : ℕ := time_to_catch_up + head_start)  -- Total time Nicky runs before Cristina catches up
  : total_time = 30 := sorry

end nicky_catches_up_time_l495_49508


namespace malcolm_red_lights_bought_l495_49536

-- Define the problem's parameters and conditions
variable (R : ℕ) (B : ℕ := 3 * R) (G : ℕ := 6)
variable (initial_white_lights : ℕ := 59) (remaining_colored_lights : ℕ := 5)

-- The total number of colored lights that he still needs to replace the white lights
def total_colored_lights_needed : ℕ := initial_white_lights - remaining_colored_lights

-- Total colored lights bought so far
def total_colored_lights_bought : ℕ := R + B + G

-- The main theorem to prove that Malcolm bought 12 red lights
theorem malcolm_red_lights_bought (h : total_colored_lights_bought = total_colored_lights_needed) :
  R = 12 := by
  sorry

end malcolm_red_lights_bought_l495_49536


namespace coefficient_of_x3y7_in_expansion_l495_49568

-- Definitions based on the conditions in the problem
def a : ℚ := (2 / 3)
def b : ℚ := - (3 / 4)
def n : ℕ := 10
def k1 : ℕ := 3
def k2 : ℕ := 7

-- Statement of the math proof problem
theorem coefficient_of_x3y7_in_expansion :
  (a * x ^ k1 + b * y ^ k2) ^ n = x3y7_coeff * x ^ k1 * y ^ k2  :=
sorry

end coefficient_of_x3y7_in_expansion_l495_49568


namespace pete_ten_dollar_bills_l495_49529

theorem pete_ten_dollar_bills (owes dollars bills: ℕ) (bill_value_per_bottle : ℕ) (num_bottles : ℕ) (ten_dollar_bills : ℕ):
  owes = 90 →
  dollars = 40 →
  bill_value_per_bottle = 5 →
  num_bottles = 20 →
  dollars + (num_bottles * bill_value_per_bottle) + (ten_dollar_bills * 10) = owes →
  ten_dollar_bills = 4 :=
by
  sorry

end pete_ten_dollar_bills_l495_49529


namespace positive_diff_two_largest_prime_factors_l495_49574

theorem positive_diff_two_largest_prime_factors (a b c d : ℕ) (h : 178469 = a * b * c * d) 
  (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) 
  (hle1 : a ≤ b) (hle2 : b ≤ c) (hle3 : c ≤ d):
  d - c = 2 := by sorry

end positive_diff_two_largest_prime_factors_l495_49574


namespace largest_possible_perimeter_l495_49532

noncomputable def max_perimeter_triangle : ℤ :=
  let a : ℤ := 7
  let b : ℤ := 9
  let x : ℤ := 15
  a + b + x

theorem largest_possible_perimeter (x : ℤ) (h1 : 7 + 9 > x) (h2 : 7 + x > 9) (h3 : 9 + x > 7) : max_perimeter_triangle = 31 := by
  sorry

end largest_possible_perimeter_l495_49532


namespace minimum_small_bottles_l495_49556

-- Define the capacities of the bottles
def small_bottle_capacity : ℕ := 35
def large_bottle_capacity : ℕ := 500

-- Define the number of small bottles needed to fill a large bottle
def small_bottles_needed_to_fill_large : ℕ := 
  (large_bottle_capacity + small_bottle_capacity - 1) / small_bottle_capacity

-- Statement of the theorem
theorem minimum_small_bottles : small_bottles_needed_to_fill_large = 15 := by
  sorry

end minimum_small_bottles_l495_49556


namespace arith_seq_problem_l495_49510

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

theorem arith_seq_problem 
  (a : ℕ → ℝ) (a1 d : ℝ)
  (h1 : arithmetic_sequence a a1 d)
  (h2 : a 1 + 3 * a 8 + a 15 = 120) : 
  3 * a 9 - a 11 = 48 :=
by 
  sorry

end arith_seq_problem_l495_49510


namespace smallest_number_greater_than_300_divided_by_25_has_remainder_24_l495_49527

theorem smallest_number_greater_than_300_divided_by_25_has_remainder_24 :
  ∃ x : ℕ, (x > 300) ∧ (x % 25 = 24) ∧ (x = 324) := by
  sorry

end smallest_number_greater_than_300_divided_by_25_has_remainder_24_l495_49527


namespace derivative_y_l495_49528

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sqrt (9 * x^2 - 12 * x + 5)) * Real.arctan (3 * x - 2) - 
  Real.log (3 * x - 2 + Real.sqrt (9 * x^2 - 12 * x + 5))

theorem derivative_y (x : ℝ) :
  ∃ (f' : ℝ → ℝ), deriv y x = f' x ∧ f' x = (9 * x - 6) * Real.arctan (3 * x - 2) / 
  Real.sqrt (9 * x^2 - 12 * x + 5) :=
sorry

end derivative_y_l495_49528


namespace remainder_of_b97_is_52_l495_49547

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem remainder_of_b97_is_52 : (b 97) % 81 = 52 := 
sorry

end remainder_of_b97_is_52_l495_49547


namespace cricket_player_avg_runs_l495_49543

theorem cricket_player_avg_runs (A : ℝ) :
  (13 * A + 92 = 14 * (A + 5)) → A = 22 :=
by
  intro h1
  have h2 : 13 * A + 92 = 14 * A + 70 := by sorry
  have h3 : 92 - 70 = 14 * A - 13 * A := by sorry
  sorry

end cricket_player_avg_runs_l495_49543


namespace find_projection_l495_49513

noncomputable def a : ℝ × ℝ := (-3, 2)
noncomputable def b : ℝ × ℝ := (5, -1)
noncomputable def p : ℝ × ℝ := (21/73, 56/73)
noncomputable def d : ℝ × ℝ := (8, -3)

theorem find_projection :
  ∃ t : ℝ, (t * d.1 - a.1, t * d.2 + a.2) = p ∧
          (p.1 - a.1) * d.1 + (p.2 - a.2) * d.2 = 0 :=
by
  sorry

end find_projection_l495_49513


namespace audit_options_correct_l495_49520

-- Define the initial number of ORs and GTUs
def initial_ORs : ℕ := 13
def initial_GTUs : ℕ := 15

-- Define the number of ORs and GTUs visited in the first week
def visited_ORs : ℕ := 2
def visited_GTUs : ℕ := 3

-- Calculate the remaining ORs and GTUs
def remaining_ORs : ℕ := initial_ORs - visited_ORs
def remaining_GTUs : ℕ := initial_GTUs - visited_GTUs

-- Calculate the number of ways to choose 2 ORs from remaining ORs
def choose_ORs : ℕ := Nat.choose remaining_ORs 2

-- Calculate the number of ways to choose 3 GTUs from remaining GTUs
def choose_GTUs : ℕ := Nat.choose remaining_GTUs 3

-- The final function to calculate the number of options
def number_of_options : ℕ := choose_ORs * choose_GTUs

-- The proof statement asserting the number of options is 12100
theorem audit_options_correct : number_of_options = 12100 := by
    sorry -- Proof will be filled in here

end audit_options_correct_l495_49520
