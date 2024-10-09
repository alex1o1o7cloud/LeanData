import Mathlib

namespace equal_playtime_l601_60199

theorem equal_playtime (children : ℕ) (total_minutes : ℕ) (simultaneous_players : ℕ) (equal_playtime_per_child : ℕ)
  (h1 : children = 12) (h2 : total_minutes = 120) (h3 : simultaneous_players = 2) (h4 : equal_playtime_per_child = (simultaneous_players * total_minutes) / children) :
  equal_playtime_per_child = 20 := 
by sorry

end equal_playtime_l601_60199


namespace last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l601_60189

noncomputable def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_1989_1989:
  last_digit (1989 ^ 1989) = 9 := 
sorry

theorem last_digit_1989_1992:
  last_digit (1989 ^ 1992) = 1 := 
sorry

theorem last_digit_1992_1989:
  last_digit (1992 ^ 1989) = 2 := 
sorry

theorem last_digit_1992_1992:
  last_digit (1992 ^ 1992) = 6 := 
sorry

end last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l601_60189


namespace solution_set_x_l601_60100

theorem solution_set_x (x : ℝ) (h₁ : 33 * 32 ≤ x)
  (h₂ : ⌊x⌋ + ⌈x⌉ = 5) : 2 < x ∧ x < 3 :=
by
  sorry

end solution_set_x_l601_60100


namespace find_k_l601_60176

def condition (k : ℝ) : Prop := 24 / k = 4

theorem find_k (k : ℝ) (h : condition k) : k = 6 :=
sorry

end find_k_l601_60176


namespace international_call_cost_per_minute_l601_60154

theorem international_call_cost_per_minute 
  (local_call_minutes : Nat)
  (international_call_minutes : Nat)
  (local_rate : Nat)
  (total_cost_cents : Nat) 
  (spent_dollars : Nat) 
  (spent_cents : Nat)
  (local_call_cost : Nat)
  (international_call_total_cost : Nat) : 
  local_call_minutes = 45 → 
  international_call_minutes = 31 → 
  local_rate = 5 → 
  total_cost_cents = spent_dollars * 100 → 
  spent_dollars = 10 → 
  local_call_cost = local_call_minutes * local_rate → 
  spent_cents = spent_dollars * 100 → 
  total_cost_cents = spent_cents →  
  international_call_total_cost = total_cost_cents - local_call_cost → 
  international_call_total_cost / international_call_minutes = 25 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end international_call_cost_per_minute_l601_60154


namespace admission_methods_correct_l601_60143

-- Define the number of famous schools.
def famous_schools : ℕ := 8

-- Define the number of students.
def students : ℕ := 3

-- Define the total number of different admission methods:
def admission_methods (schools : ℕ) (students : ℕ) : ℕ :=
  Nat.choose schools 2 * 3

-- The theorem stating the desired result.
theorem admission_methods_correct :
  admission_methods famous_schools students = 84 :=
by
  sorry

end admission_methods_correct_l601_60143


namespace line_slope_is_neg_half_l601_60118

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := x + 2 * y - 4 = 0

-- The main theorem to be proved
theorem line_slope_is_neg_half : ∀ (x y : ℝ), line_eq x y → (∃ m b : ℝ, y = m * x + b ∧ m = -1/2) := by
  sorry

end line_slope_is_neg_half_l601_60118


namespace find_p_l601_60139

-- Conditions: Consider the quadratic equation 2x^2 + px + q = 0 where p and q are integers.
-- Roots of the equation differ by 2.
-- q = 4

theorem find_p (p : ℤ) (q : ℤ) (h1 : q = 4) (h2 : ∃ x₁ x₂ : ℝ, 2 * x₁^2 + p * x₁ + q = 0 ∧ 2 * x₂^2 + p * x₂ + q = 0 ∧ |x₁ - x₂| = 2) :
  p = 7 ∨ p = -7 :=
by
  sorry

end find_p_l601_60139


namespace acute_angle_condition_l601_60177

theorem acute_angle_condition 
  (m : ℝ) 
  (a : ℝ × ℝ := (2,1))
  (b : ℝ × ℝ := (m,6)) 
  (dot_product := a.1 * b.1 + a.2 * b.2)
  (magnitude_a := Real.sqrt (a.1 * a.1 + a.2 * a.2))
  (magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2))
  (cos_angle := dot_product / (magnitude_a * magnitude_b))
  (acute_angle : cos_angle > 0) : -3 < m ∧ m ≠ 12 :=
sorry

end acute_angle_condition_l601_60177


namespace max_value_of_expression_l601_60167

theorem max_value_of_expression {a x1 x2 : ℝ}
  (h1 : x1^2 + a * x1 + a = 2)
  (h2 : x2^2 + a * x2 + a = 2)
  (h1_ne_x2 : x1 ≠ x2) :
  ∃ a : ℝ, (x1 - 2 * x2) * (x2 - 2 * x1) = -63 / 8 :=
by
  sorry

end max_value_of_expression_l601_60167


namespace possible_days_l601_60125

namespace AnyaVanyaProblem

-- Conditions
def AnyaLiesOn (d : String) : Prop := d = "Tuesday" ∨ d = "Wednesday" ∨ d = "Thursday"
def AnyaTellsTruthOn (d : String) : Prop := ¬AnyaLiesOn d

def VanyaLiesOn (d : String) : Prop := d = "Thursday" ∨ d = "Friday" ∨ d = "Saturday"
def VanyaTellsTruthOn (d : String) : Prop := ¬VanyaLiesOn d

-- Statements
def AnyaStatement (d : String) : Prop := d = "Friday"
def VanyaStatement (d : String) : Prop := d = "Tuesday"

-- Proof problem
theorem possible_days (d : String) : 
  (AnyaTellsTruthOn d ↔ AnyaStatement d) ∧ (VanyaTellsTruthOn d ↔ VanyaStatement d)
  → d = "Tuesday" ∨ d = "Thursday" ∨ d = "Friday" := 
sorry

end AnyaVanyaProblem

end possible_days_l601_60125


namespace initial_amount_A_correct_l601_60182

noncomputable def initial_amount_A :=
  let a := 21
  let b := 5
  let c := 9

  -- After A gives B and C
  let b_after_A := b + 5
  let c_after_A := c + 9
  let a_after_A := a - (5 + 9)

  -- After B gives A and C
  let a_after_B := a_after_A + (a_after_A / 2)
  let c_after_B := c_after_A + (c_after_A / 2)
  let b_after_B := b_after_A - (a_after_A / 2 + c_after_A / 2)

  -- After C gives A and B
  let a_final := a_after_B + 3 * a_after_B
  let b_final := b_after_B + 3 * b_after_B
  let c_final := c_after_B - (3 * a_final + b_final)

  (a_final = 24) ∧ (b_final = 16) ∧ (c_final = 8)

theorem initial_amount_A_correct : initial_amount_A := 
by
  -- Skipping proof details
  sorry

end initial_amount_A_correct_l601_60182


namespace frank_can_buy_seven_candies_l601_60115

def tickets_won_whackamole := 33
def tickets_won_skeeball := 9
def cost_per_candy := 6

theorem frank_can_buy_seven_candies : (tickets_won_whackamole + tickets_won_skeeball) / cost_per_candy = 7 :=
by
  sorry

end frank_can_buy_seven_candies_l601_60115


namespace max_total_balls_l601_60173

theorem max_total_balls
  (r₁ : ℕ := 89)
  (t₁ : ℕ := 90)
  (r₂ : ℕ := 8)
  (t₂ : ℕ := 9)
  (y : ℕ)
  (h₁ : t₁ > 0)
  (h₂ : t₂ > 0)
  (h₃ : 92 ≤ (r₁ + r₂ * y) * 100 / (t₁ + t₂ * y))
  : y ≤ 22 → 90 + 9 * y = 288 :=
by sorry

end max_total_balls_l601_60173


namespace pencils_per_student_l601_60129

-- Define the number of pens
def numberOfPens : ℕ := 1001

-- Define the number of pencils
def numberOfPencils : ℕ := 910

-- Define the maximum number of students
def maxNumberOfStudents : ℕ := 91

-- Using the given conditions, prove that each student gets 10 pencils
theorem pencils_per_student :
  (numberOfPencils / maxNumberOfStudents) = 10 :=
by sorry

end pencils_per_student_l601_60129


namespace franks_earnings_l601_60140

/-- Frank's earnings problem statement -/
theorem franks_earnings 
  (total_hours : ℕ) (days : ℕ) (regular_pay_rate : ℝ) (overtime_pay_rate : ℝ)
  (hours_first_day : ℕ) (overtime_first_day : ℕ)
  (hours_second_day : ℕ) (hours_third_day : ℕ)
  (hours_fourth_day : ℕ) (overtime_fourth_day : ℕ)
  (regular_hours_per_day : ℕ) :
  total_hours = 32 →
  days = 4 →
  regular_pay_rate = 15 →
  overtime_pay_rate = 22.50 →
  hours_first_day = 12 →
  overtime_first_day = 4 →
  hours_second_day = 8 →
  hours_third_day = 8 →
  hours_fourth_day = 12 →
  overtime_fourth_day = 4 →
  regular_hours_per_day = 8 →
  (32 * regular_pay_rate + 8 * overtime_pay_rate) = 660 := 
by 
  intros 
  sorry

end franks_earnings_l601_60140


namespace complement_union_A_B_l601_60133

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 5}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}
def U : Set ℝ := A ∪ B
def R : Set ℝ := univ

theorem complement_union_A_B : (R \ U) = {x | -2 < x ∧ x ≤ -1} :=
by
  sorry

end complement_union_A_B_l601_60133


namespace num_marked_cells_at_least_num_cells_in_one_square_l601_60197

-- Defining the total number of squares
def num_squares : ℕ := 2009

-- A square covers a cell if it is within its bounds.
-- A cell is marked if it is covered by an odd number of squares.
-- We have to show that the number of marked cells is at least the number of cells in one square.
theorem num_marked_cells_at_least_num_cells_in_one_square (side_length : ℕ) : 
  side_length * side_length ≤ (num_squares : ℕ) :=
sorry

end num_marked_cells_at_least_num_cells_in_one_square_l601_60197


namespace cube_roots_not_arithmetic_progression_l601_60172

theorem cube_roots_not_arithmetic_progression
  (p q r : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (h_distinct: p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ¬ ∃ (d : ℝ) (m n : ℤ), (n ≠ m) ∧ (↑q)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (m : ℝ) * d ∧ (↑r)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (n : ℝ) * d :=
by sorry

end cube_roots_not_arithmetic_progression_l601_60172


namespace remainder_2abc_mod_7_l601_60131

theorem remainder_2abc_mod_7
  (a b c : ℕ)
  (h₀ : 2 * a + 3 * b + c ≡ 1 [MOD 7])
  (h₁ : 3 * a + b + 2 * c ≡ 2 [MOD 7])
  (h₂ : a + b + c ≡ 3 [MOD 7])
  (ha : a < 7)
  (hb : b < 7)
  (hc : c < 7) :
  2 * a * b * c ≡ 0 [MOD 7] :=
sorry

end remainder_2abc_mod_7_l601_60131


namespace line_intersects_x_axis_at_point_l601_60156

theorem line_intersects_x_axis_at_point : 
  let x1 := 3
  let y1 := 7
  let x2 := -1
  let y2 := 3
  let m := (y2 - y1) / (x2 - x1) -- slope formula
  let b := y1 - m * x1        -- y-intercept formula
  let x_intersect := -b / m  -- x-coordinate where the line intersects x-axis
  (x_intersect, 0) = (-4, 0) :=
by
  sorry

end line_intersects_x_axis_at_point_l601_60156


namespace sum_of_segments_l601_60146

noncomputable def segment_sum (AB_len CB_len FG_len : ℕ) : ℝ :=
  199 * (Real.sqrt (AB_len * AB_len + CB_len * CB_len) +
         Real.sqrt (AB_len * AB_len + FG_len * FG_len))

theorem sum_of_segments : segment_sum 5 6 8 = 199 * (Real.sqrt 61 + Real.sqrt 89) :=
by
  sorry

end sum_of_segments_l601_60146


namespace smallest_points_2016_l601_60109

theorem smallest_points_2016 (n : ℕ) :
  n = 28225 →
  ∀ (points : Fin n → (ℤ × ℤ)),
  ∃ i j : Fin n, i ≠ j ∧
    let dist_sq := (points i).fst - (points j).fst ^ 2 + (points i).snd - (points j).snd ^ 2 
    ∃ k : ℤ, dist_sq = 2016 * k :=
by
  intro h points
  sorry

end smallest_points_2016_l601_60109


namespace seed_mixture_x_percentage_l601_60130

theorem seed_mixture_x_percentage (x y : ℝ) (h : 0.40 * x + 0.25 * y = 0.30 * (x + y)) : 
  (x / (x + y)) * 100 = 33.33 := sorry

end seed_mixture_x_percentage_l601_60130


namespace total_feet_is_140_l601_60151

def total_heads : ℕ := 48
def number_of_hens : ℕ := 26
def number_of_cows : ℕ := total_heads - number_of_hens
def feet_per_hen : ℕ := 2
def feet_per_cow : ℕ := 4

theorem total_feet_is_140 : ((number_of_hens * feet_per_hen) + (number_of_cows * feet_per_cow)) = 140 := by
  sorry

end total_feet_is_140_l601_60151


namespace combined_collectors_edition_dolls_l601_60145

-- Definitions based on given conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def luna_dolls : ℕ := ivy_dolls - 10

-- Additional constraints based on the problem statement
def total_dolls : ℕ := dina_dolls + ivy_dolls + luna_dolls
def ivy_collectors_edition_dolls : ℕ := 2/3 * ivy_dolls
def luna_collectors_edition_dolls : ℕ := 1/2 * luna_dolls

-- Proof statement
theorem combined_collectors_edition_dolls :
  ivy_collectors_edition_dolls + luna_collectors_edition_dolls = 30 :=
sorry

end combined_collectors_edition_dolls_l601_60145


namespace union_complement_U_B_l601_60192

def U : Set ℤ := { x | -3 < x ∧ x < 3 }
def A : Set ℤ := { 1, 2 }
def B : Set ℤ := { -2, -1, 2 }

theorem union_complement_U_B : A ∪ (U \ B) = { 0, 1, 2 } := by
  sorry

end union_complement_U_B_l601_60192


namespace billy_ate_72_cherries_l601_60148

-- Definitions based on conditions:
def initial_cherries : Nat := 74
def remaining_cherries : Nat := 2

-- Problem: How many cherries did Billy eat?
def cherries_eaten := initial_cherries - remaining_cherries

theorem billy_ate_72_cherries : cherries_eaten = 72 :=
by
  -- proof here
  sorry

end billy_ate_72_cherries_l601_60148


namespace exist_indices_eq_l601_60134

theorem exist_indices_eq (p q n : ℕ) (x : ℕ → ℤ) 
    (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_n : 0 < n) 
    (h_pq_n : p + q < n) 
    (h_x0 : x 0 = 0) 
    (h_xn : x n = 0) 
    (h_step : ∀ i, 1 ≤ i ∧ i ≤ n → (x i - x (i - 1) = p ∨ x i - x (i - 1) = -q)) :
    ∃ (i j : ℕ), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
sorry

end exist_indices_eq_l601_60134


namespace base_difference_is_correct_l601_60105

-- Definitions of given conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 324 => 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Lean statement to prove the equivalence
theorem base_difference_is_correct : base9_to_base10 324 - base6_to_base10 231 = 174 :=
by
  sorry

end base_difference_is_correct_l601_60105


namespace brush_length_percentage_increase_l601_60162

-- Define the length of Carla's brush in inches
def carla_brush_length_in_inches : ℝ := 12

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Define the length of Carmen's brush in centimeters
def carmen_brush_length_in_cm : ℝ := 45

-- Noncomputable definition to calculate the percentage increase
noncomputable def percentage_increase : ℝ :=
  let carla_brush_length_in_cm := carla_brush_length_in_inches * inch_to_cm
  (carmen_brush_length_in_cm - carla_brush_length_in_cm) / carla_brush_length_in_cm * 100

-- Statement to prove the percentage increase is 47.6%
theorem brush_length_percentage_increase :
  percentage_increase = 47.6 :=
sorry

end brush_length_percentage_increase_l601_60162


namespace sqrt_continued_fraction_l601_60122

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l601_60122


namespace find_constant_a_l601_60158

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (Real.exp x - 1)

theorem find_constant_a (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = - f a x) : a = -1 := 
by
  sorry

end find_constant_a_l601_60158


namespace calories_burned_per_week_l601_60184

-- Definitions from conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℚ := 1.5
def calories_per_minute : ℕ := 7

-- Prove the total calories burned per week
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * 60) * calories_per_minute) = 1890 := by
    sorry

end calories_burned_per_week_l601_60184


namespace first_pack_weight_l601_60181

variable (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
variable (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ)

theorem first_pack_weight (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
    (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ) :
    hiking_rate = 2.5 →
    hours_per_day = 9 →
    days = 7 →
    pounds_per_mile = 0.6 →
    first_resupply_percentage = 0.30 →
    second_resupply_percentage = 0.20 →
    ∃ first_pack : ℝ, first_pack = 47.25 :=
by
  intro h1 h2 h3 h4 h5 h6
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := pounds_per_mile * total_distance
  let first_resupply := total_supplies * first_resupply_percentage
  let second_resupply := total_supplies * second_resupply_percentage
  let first_pack := total_supplies - (first_resupply + second_resupply)
  use first_pack
  sorry

end first_pack_weight_l601_60181


namespace farmer_total_cows_l601_60150

theorem farmer_total_cows (cows : ℕ) 
  (h1 : 1 / 3 + 1 / 6 + 1 / 8 = 5 / 8) 
  (h2 : (3 / 8) * cows = 15) : 
  cows = 40 := by
  -- Given conditions:
  -- h1: The first three sons receive a total of 5/8 of the cows.
  -- h2: The fourth son receives 3/8 of the cows, which is 15 cows.
  sorry

end farmer_total_cows_l601_60150


namespace intersection_M_N_l601_60113

def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {x | x^2 - 4 * x + 3 = 0}

theorem intersection_M_N : M ∩ N = {1, 3} :=
by sorry

end intersection_M_N_l601_60113


namespace diamond_expression_evaluation_l601_60163

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem diamond_expression_evaluation :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 :=
by {
    sorry
}

end diamond_expression_evaluation_l601_60163


namespace candy_comparison_l601_60175

variable (skittles_bryan : ℕ)
variable (gummy_bears_bryan : ℕ)
variable (chocolate_bars_bryan : ℕ)
variable (mms_ben : ℕ)
variable (jelly_beans_ben : ℕ)
variable (lollipops_ben : ℕ)

def bryan_total_candies := skittles_bryan + gummy_bears_bryan + chocolate_bars_bryan
def ben_total_candies := mms_ben + jelly_beans_ben + lollipops_ben

def difference_skittles_mms := skittles_bryan - mms_ben
def difference_gummy_jelly := jelly_beans_ben - gummy_bears_bryan
def difference_choco_lollipops := chocolate_bars_bryan - lollipops_ben

def sum_of_differences := difference_skittles_mms + difference_gummy_jelly + difference_choco_lollipops

theorem candy_comparison
  (h_bryan_skittles : skittles_bryan = 50)
  (h_bryan_gummy_bears : gummy_bears_bryan = 25)
  (h_bryan_choco_bars : chocolate_bars_bryan = 15)
  (h_ben_mms : mms_ben = 20)
  (h_ben_jelly_beans : jelly_beans_ben = 30)
  (h_ben_lollipops : lollipops_ben = 10) :
  bryan_total_candies = 90 ∧
  ben_total_candies = 60 ∧
  bryan_total_candies > ben_total_candies ∧
  difference_skittles_mms = 30 ∧
  difference_gummy_jelly = 5 ∧
  difference_choco_lollipops = 5 ∧
  sum_of_differences = 40 := by
  sorry

end candy_comparison_l601_60175


namespace ninth_graders_only_math_l601_60180

theorem ninth_graders_only_math 
  (total_students : ℕ)
  (math_students : ℕ)
  (foreign_language_students : ℕ)
  (science_only_students : ℕ)
  (math_and_foreign_language_no_science : ℕ)
  (h1 : total_students = 120)
  (h2 : math_students = 85)
  (h3 : foreign_language_students = 75)
  (h4 : science_only_students = 20)
  (h5 : math_and_foreign_language_no_science = 40) :
  math_students - math_and_foreign_language_no_science = 45 :=
by 
  sorry

end ninth_graders_only_math_l601_60180


namespace wall_building_l601_60142

-- Definitions based on conditions
def total_work (m d : ℕ) : ℕ := m * d

-- Prove that if 30 men including 10 twice as efficient men work for 3 days, they can build the wall
theorem wall_building (m₁ m₂ d₁ d₂ : ℕ) (h₁ : total_work m₁ d₁ = total_work m₂ d₂) (m₁_eq : m₁ = 20) (d₁_eq : d₁ = 6) 
(h₂ : m₂ = 40) : d₂ = 3 :=
  sorry

end wall_building_l601_60142


namespace households_with_car_l601_60188

theorem households_with_car {H_total H_neither H_both H_bike_only : ℕ} 
    (cond1 : H_total = 90)
    (cond2 : H_neither = 11)
    (cond3 : H_both = 22)
    (cond4 : H_bike_only = 35) : 
    H_total - H_neither - (H_bike_only + H_both - H_both) + H_both = 44 := by
  sorry

end households_with_car_l601_60188


namespace ab_cd_eq_one_l601_60135

theorem ab_cd_eq_one (a b c d : ℕ) (p : ℕ) 
  (h_div_a : a % p = 0)
  (h_div_b : b % p = 0)
  (h_div_c : c % p = 0)
  (h_div_d : d % p = 0)
  (h_div_ab_cd : (a * b - c * d) % p = 0) : 
  (a * b - c * d) = 1 :=
sorry

end ab_cd_eq_one_l601_60135


namespace max_popsicles_l601_60116

def popsicles : ℕ := 1
def box_3 : ℕ := 3
def box_5 : ℕ := 5
def box_10 : ℕ := 10
def cost_popsicle : ℕ := 1
def cost_box_3 : ℕ := 2
def cost_box_5 : ℕ := 3
def cost_box_10 : ℕ := 4
def budget : ℕ := 10

theorem max_popsicles : 
  ∀ (popsicle_count : ℕ) (b3_count : ℕ) (b5_count : ℕ) (b10_count : ℕ),
    popsicle_count * cost_popsicle + b3_count * cost_box_3 + b5_count * cost_box_5 + b10_count * cost_box_10 ≤ budget →
    popsicle_count * popsicles + b3_count * box_3 + b5_count * box_5 + b10_count * box_10 ≤ 23 →
    ∃ p b3 b5 b10, popsicle_count = p ∧ b3_count = b3 ∧ b5_count = b5 ∧ b10_count = b10 ∧
    (p * cost_popsicle + b3 * cost_box_3 + b5 * cost_box_5 + b10 * cost_box_10 ≤ budget) ∧
    (p * popsicles + b3 * box_3 + b5 * box_5 + b10 * box_10 = 23) :=
by sorry

end max_popsicles_l601_60116


namespace complex_number_C_l601_60127

-- Define the complex numbers corresponding to points A and B
def A : ℂ := 1 + 2 * Complex.I
def B : ℂ := 3 - 5 * Complex.I

-- Prove the complex number corresponding to point C
theorem complex_number_C :
  ∃ C : ℂ, (C = 10 - 3 * Complex.I) ∧ 
           (A = 1 + 2 * Complex.I) ∧ 
           (B = 3 - 5 * Complex.I) ∧ 
           -- Square with vertices in counterclockwise order
           True := 
sorry

end complex_number_C_l601_60127


namespace num_divisible_by_33_l601_60149

theorem num_divisible_by_33 : ∀ (x y : ℕ), 
  (0 ≤ x ∧ x ≤ 9) → (0 ≤ y ∧ y ≤ 9) →
  (19 + x + y) % 3 = 0 →
  (x - y + 1) % 11 = 0 →
  ∃! (n : ℕ), (20070002008 * 100 + x * 10 + y) = n ∧ n % 33 = 0 :=
by
  intros x y hx hy h3 h11
  sorry

end num_divisible_by_33_l601_60149


namespace compute_expression_l601_60138

theorem compute_expression : 1013^2 - 991^2 - 1007^2 + 997^2 = 24048 := by
  sorry

end compute_expression_l601_60138


namespace fraction_zero_solution_l601_60171

theorem fraction_zero_solution (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := 
by sorry

end fraction_zero_solution_l601_60171


namespace red_balls_count_after_game_l601_60185

structure BagState :=
  (red : Nat)         -- Number of red balls
  (green : Nat)       -- Number of green balls
  (blue : Nat)        -- Number of blue balls
  (yellow : Nat)      -- Number of yellow balls
  (black : Nat)       -- Number of black balls
  (white : Nat)       -- Number of white balls)

def initialBallCount (totalBalls : Nat) : BagState :=
  let totalRatio := 15 + 13 + 17 + 9 + 7 + 23
  { red := totalBalls * 15 / totalRatio
  , green := totalBalls * 13 / totalRatio
  , blue := totalBalls * 17 / totalRatio
  , yellow := totalBalls * 9 / totalRatio
  , black := totalBalls * 7 / totalRatio
  , white := totalBalls * 23 / totalRatio
  }

def finalBallCount (initialState : BagState) : BagState :=
  { red := initialState.red + 400
  , green := initialState.green - 250
  , blue := initialState.blue
  , yellow := initialState.yellow - 100
  , black := initialState.black + 200
  , white := initialState.white - 500
  }

theorem red_balls_count_after_game :
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  final.red = 2185 :=
by
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  sorry

end red_balls_count_after_game_l601_60185


namespace sum_last_two_digits_9_pow_23_plus_11_pow_23_l601_60196

theorem sum_last_two_digits_9_pow_23_plus_11_pow_23 :
  (9^23 + 11^23) % 100 = 60 :=
by
  sorry

end sum_last_two_digits_9_pow_23_plus_11_pow_23_l601_60196


namespace constant_sum_l601_60186

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem constant_sum (a1 d : ℝ) (h : 3 * arithmetic_sequence a1 d 8 = k) :
  ∃ k : ℝ, sum_arithmetic_sequence a1 d 15 = k :=
sorry

end constant_sum_l601_60186


namespace joeys_votes_l601_60187

theorem joeys_votes
  (M B J : ℕ) 
  (h1 : M = 66) 
  (h2 : M = 3 * B) 
  (h3 : B = 2 * (J + 3)) : 
  J = 8 := 
by 
  sorry

end joeys_votes_l601_60187


namespace hundredth_number_is_201_l601_60144

-- Mathematical definition of the sequence
def counting_sequence (n : ℕ) : ℕ :=
  3 + (n - 1) * 2

-- Statement to prove
theorem hundredth_number_is_201 : counting_sequence 100 = 201 :=
by
  sorry

end hundredth_number_is_201_l601_60144


namespace total_oranges_is_correct_l601_60166

-- Definitions based on the problem's conditions
def layer_count : ℕ := 6
def base_length : ℕ := 9
def base_width : ℕ := 6

-- Function to compute the number of oranges in a layer given the current dimensions
def oranges_in_layer (length width : ℕ) : ℕ :=
  length * width

-- Function to compute the total number of oranges in the stack
def total_oranges_in_stack (base_length base_width : ℕ) : ℕ :=
  oranges_in_layer base_length base_width +
  oranges_in_layer (base_length - 1) (base_width - 1) +
  oranges_in_layer (base_length - 2) (base_width - 2) +
  oranges_in_layer (base_length - 3) (base_width - 3) +
  oranges_in_layer (base_length - 4) (base_width - 4) +
  oranges_in_layer (base_length - 5) (base_width - 5)

-- The theorem to be proved
theorem total_oranges_is_correct : total_oranges_in_stack 9 6 = 154 := by
  sorry

end total_oranges_is_correct_l601_60166


namespace length_of_second_train_l601_60124

/-- 
  Given:
  * Speed of train 1 is 60 km/hr.
  * Speed of train 2 is 40 km/hr.
  * Length of train 1 is 500 meters.
  * Time to cross each other is 44.99640028797697 seconds.

  Then the length of train 2 is 750 meters.
-/
theorem length_of_second_train (v1 v2 t : ℝ) (d1 L : ℝ) : 
  v1 = 60 ∧
  v2 = 40 ∧
  t = 44.99640028797697 ∧
  d1 = 500 ∧
  L = ((v1 + v2) * (1000 / 3600) * t - d1) →
  L = 750 :=
by sorry

end length_of_second_train_l601_60124


namespace chord_length_on_parabola_eq_five_l601_60195

theorem chord_length_on_parabola_eq_five
  (A B : ℝ × ℝ)
  (hA : A.snd ^ 2 = 4 * A.fst)
  (hB : B.snd ^ 2 = 4 * B.fst)
  (hM : A.fst + B.fst = 3 ∧ A.snd + B.snd = 2 
     ∧ A.fst - B.fst = 0 ∧ A.snd - B.snd = 0) :
  dist A B = 5 :=
by
  -- Proof goes here
  sorry

end chord_length_on_parabola_eq_five_l601_60195


namespace volume_of_right_square_prism_l601_60194

theorem volume_of_right_square_prism (length width : ℕ) (H1 : length = 12) (H2 : width = 8) :
    ∃ V, (V = 72 ∨ V = 48) :=
by
  sorry

end volume_of_right_square_prism_l601_60194


namespace stream_speed_l601_60168

theorem stream_speed (v : ℝ) : 
  (∀ (speed_boat_in_still_water distance time : ℝ), 
    speed_boat_in_still_water = 25 ∧ distance = 90 ∧ time = 3 →
    distance = (speed_boat_in_still_water + v) * time) →
  v = 5 :=
by
  intro h
  have h1 := h 25 90 3 ⟨rfl, rfl, rfl⟩
  sorry

end stream_speed_l601_60168


namespace total_paint_area_eq_1060_l601_60108

/-- Define the dimensions of the stable and chimney -/
def stable_width := 12
def stable_length := 15
def stable_height := 6
def chimney_width := 2
def chimney_length := 2
def chimney_height := 2

/-- Define the area to be painted computation -/

def wall_area (width length height : ℕ) : ℕ :=
  (width * height * 2) * 2 + (length * height * 2) * 2

def roof_area (width length : ℕ) : ℕ :=
  width * length

def ceiling_area (width length : ℕ) : ℕ :=
  width * length

def chimney_area (width length height : ℕ) : ℕ :=
  (4 * (width * height)) + (width * length)

def total_paint_area : ℕ :=
  wall_area stable_width stable_length stable_height +
  roof_area stable_width stable_length +
  ceiling_area stable_width stable_length +
  chimney_area chimney_width chimney_length chimney_height

/-- Goal: Prove that the total paint area is 1060 sq. yd -/
theorem total_paint_area_eq_1060 : total_paint_area = 1060 := by
  sorry

end total_paint_area_eq_1060_l601_60108


namespace S_eq_Z_l601_60104

noncomputable def set_satisfies_conditions (S : Set ℤ) (a : Fin n → ℤ) :=
  (∀ i : Fin n, a i ∈ S) ∧
  (∀ i j : Fin n, (a i - a j) ∈ S) ∧
  (∀ x y : ℤ, x ∈ S → y ∈ S → x + y ∈ S → x - y ∈ S) ∧
  (Nat.gcd (List.foldr Nat.gcd 0 (Fin.val <$> List.finRange n)) = 1)

theorem S_eq_Z (S : Set ℤ) (a : Fin n → ℤ) (h_cond : set_satisfies_conditions S a) : S = Set.univ :=
  sorry

end S_eq_Z_l601_60104


namespace necessary_but_not_sufficient_l601_60107

-- Definitions used in the conditions
variable (a b : ℝ)

-- The Lean 4 theorem statement for the proof problem
theorem necessary_but_not_sufficient : (a > b - 1) ∧ ¬ (a > b) ↔ a > b := 
sorry

end necessary_but_not_sufficient_l601_60107


namespace problem_1_problem_2_l601_60164

noncomputable def a (k : ℝ) : ℝ × ℝ := (2, k)
noncomputable def b : ℝ × ℝ := (1, 1)
noncomputable def a_minus_3b (k : ℝ) : ℝ × ℝ := (2 - 3 * 1, k - 3 * 1)

-- First problem: Prove that k = 4 given vectors a and b, and the condition that b is perpendicular to (a - 3b)
theorem problem_1 (k : ℝ) (h : b.1 * (a_minus_3b k).1 + b.2 * (a_minus_3b k).2 = 0) : k = 4 :=
sorry

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def cosine (v w : ℝ × ℝ) : ℝ := dot_product v w / (magnitude v * magnitude w)

-- Second problem: Prove that the cosine value of the angle between a and b is 3√10/10 when k is 4
theorem problem_2 (k : ℝ) (hk : k = 4) : cosine (a k) b = 3 * Real.sqrt 10 / 10 :=
sorry

end problem_1_problem_2_l601_60164


namespace nonagon_diagonals_l601_60193

-- Define nonagon and its properties
def is_nonagon (n : ℕ) : Prop := n = 9
def has_parallel_sides (n : ℕ) : Prop := n = 9 ∧ true

-- Define the formula for calculating diagonals in a convex polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The main theorem statement
theorem nonagon_diagonals :
  ∀ (n : ℕ), is_nonagon n → has_parallel_sides n → diagonals n = 27 :=  by 
  intros n hn _ 
  rw [is_nonagon] at hn
  rw [hn]
  sorry

end nonagon_diagonals_l601_60193


namespace ball_count_in_box_eq_57_l601_60106

theorem ball_count_in_box_eq_57 (N : ℕ) (h : N - 44 = 70 - N) : N = 57 :=
sorry

end ball_count_in_box_eq_57_l601_60106


namespace nonagon_area_l601_60119

noncomputable def area_of_nonagon (r : ℝ) : ℝ :=
  (9 / 2) * r^2 * Real.sin (Real.pi * 40 / 180)

theorem nonagon_area (r : ℝ) : 
  area_of_nonagon r = 2.891 * r^2 :=
by
  sorry

end nonagon_area_l601_60119


namespace rightmost_four_digits_of_7_pow_2045_l601_60169

theorem rightmost_four_digits_of_7_pow_2045 : (7^2045 % 10000) = 6807 :=
by
  sorry

end rightmost_four_digits_of_7_pow_2045_l601_60169


namespace max_area_of_cone_l601_60170

noncomputable def max_cross_sectional_area (l θ : ℝ) : ℝ := (1/2) * l^2 * Real.sin θ

theorem max_area_of_cone :
  (∀ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) → max_cross_sectional_area 3 θ ≤ (9 / 2))
  ∧ (∃ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) ∧ max_cross_sectional_area 3 θ = (9 / 2)) := 
by
  sorry

end max_area_of_cone_l601_60170


namespace sum_cyc_geq_one_l601_60165

theorem sum_cyc_geq_one (a b c : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hcond : a * b + b * c + c * a = a * b * c) :
  (a^4 / (b * (b^4 + c^3)) + b^4 / (c * (c^3 + a^4)) + c^4 / (a * (a^4 + b^3))) ≥ 1 :=
sorry

end sum_cyc_geq_one_l601_60165


namespace pave_hall_with_stones_l601_60101

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

end pave_hall_with_stones_l601_60101


namespace flour_baking_soda_ratio_l601_60161

theorem flour_baking_soda_ratio 
  (sugar flour baking_soda : ℕ)
  (h1 : sugar = 2000)
  (h2 : 5 * flour = 6 * sugar)
  (h3 : 8 * (baking_soda + 60) = flour) :
  flour / baking_soda = 10 := by
  sorry

end flour_baking_soda_ratio_l601_60161


namespace final_amount_is_75139_84_l601_60141

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) : ℝ :=
  P * (1 + r/n)^(n * t)

theorem final_amount_is_75139_84 (P : ℝ) (r : ℝ) (t : ℝ) (n : ℕ) :
  P = 64000 → r = 1/12 → t = 2 → n = 12 → compoundInterest P r t n = 75139.84 :=
by
  intros hP hr ht hn
  sorry

end final_amount_is_75139_84_l601_60141


namespace length_of_courtyard_l601_60179

-- Define the dimensions and properties of the courtyard and paving stones
def width := 33 / 2
def numPavingStones := 132
def pavingStoneLength := 5 / 2
def pavingStoneWidth := 2

-- Total area covered by paving stones
def totalArea := numPavingStones * (pavingStoneLength * pavingStoneWidth)

-- To prove: Length of the courtyard
theorem length_of_courtyard : totalArea / width = 40 := by
  sorry

end length_of_courtyard_l601_60179


namespace toucan_count_correct_l601_60155

def initial_toucans : ℕ := 2
def toucans_joined : ℕ := 1
def total_toucans : ℕ := initial_toucans + toucans_joined

theorem toucan_count_correct : total_toucans = 3 := by
  sorry

end toucan_count_correct_l601_60155


namespace sin_neg_600_eq_sqrt_3_div_2_l601_60103

theorem sin_neg_600_eq_sqrt_3_div_2 :
  Real.sin (-(600 * Real.pi / 180)) = Real.sqrt 3 / 2 :=
sorry

end sin_neg_600_eq_sqrt_3_div_2_l601_60103


namespace hens_count_l601_60128

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 144) 
  (h3 : H ≥ 10) (h4 : C ≥ 5) : H = 24 :=
by
  sorry

end hens_count_l601_60128


namespace solve_system_eqns_l601_60153

theorem solve_system_eqns (x y z : ℝ) :
  x^2 - 23 * y + 66 * z + 612 = 0 ∧
  y^2 + 62 * x - 20 * z + 296 = 0 ∧
  z^2 - 22 * x + 67 * y + 505 = 0 ↔
  x = -20 ∧ y = -22 ∧ z = -23 :=
by
  sorry

end solve_system_eqns_l601_60153


namespace passengers_off_in_texas_l601_60112

variable (x : ℕ) -- number of passengers who got off in Texas
variable (initial_passengers : ℕ := 124)
variable (texas_boarding : ℕ := 24)
variable (nc_off : ℕ := 47)
variable (nc_boarding : ℕ := 14)
variable (virginia_passengers : ℕ := 67)

theorem passengers_off_in_texas {x : ℕ} :
  (initial_passengers - x + texas_boarding - nc_off + nc_boarding) = virginia_passengers → 
  x = 48 :=
by
  sorry

end passengers_off_in_texas_l601_60112


namespace bob_more_than_ken_l601_60147

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := by
  -- proof steps to be filled in
  sorry

end bob_more_than_ken_l601_60147


namespace nineteen_times_eight_pow_n_plus_seventeen_is_composite_l601_60110

theorem nineteen_times_eight_pow_n_plus_seventeen_is_composite 
  (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
sorry

end nineteen_times_eight_pow_n_plus_seventeen_is_composite_l601_60110


namespace bert_puzzle_days_l601_60111

noncomputable def words_per_pencil : ℕ := 1050
noncomputable def words_per_puzzle : ℕ := 75

theorem bert_puzzle_days : words_per_pencil / words_per_puzzle = 14 := by
  sorry

end bert_puzzle_days_l601_60111


namespace geometric_sequence_a5_l601_60152

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) (hq : q = 2) (h_a2a6 : a 2 * a 6 = 16) :
  a 5 = 8 :=
sorry

end geometric_sequence_a5_l601_60152


namespace measure_weights_l601_60178

theorem measure_weights (w1 w3 w7 : Nat) (h1 : w1 = 1) (h3 : w3 = 3) (h7 : w7 = 7) :
  ∃ s : Finset Nat, s.card = 7 ∧ 
    (1 ∈ s) ∧ (3 ∈ s) ∧ (7 ∈ s) ∧
    (4 ∈ s) ∧ (8 ∈ s) ∧ (10 ∈ s) ∧ 
    (11 ∈ s) := 
by
  sorry

end measure_weights_l601_60178


namespace total_time_in_range_l601_60136

-- Definitions for the problem conditions
def section1 := 240 -- km
def section2 := 300 -- km
def section3 := 400 -- km

def speed1 := 40 -- km/h
def speed2 := 75 -- km/h
def speed3 := 80 -- km/h

-- The time it takes to cover a section at a certain speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Total time to cover all sections with different speed assignments
def total_time (s1 s2 s3 v1 v2 v3 : ℕ) : ℕ :=
  time s1 v1 + time s2 v2 + time s3 v3

-- Prove that the total time is within the range [15, 17]
theorem total_time_in_range :
  (total_time section1 section2 section3 speed3 speed2 speed1 = 15) ∧
  (total_time section1 section2 section3 speed1 speed2 speed3 = 17) →
  ∃ (T : ℕ), 15 ≤ T ∧ T ≤ 17 :=
by
  intro h
  sorry

end total_time_in_range_l601_60136


namespace problem_solution_l601_60160

theorem problem_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^2 + b^2 + c^2 = 3) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := 
  sorry

end problem_solution_l601_60160


namespace bus_stops_per_hour_l601_60191

theorem bus_stops_per_hour 
  (bus_speed_without_stoppages : Float)
  (bus_speed_with_stoppages : Float)
  (bus_stops_per_hour_in_minutes : Float) :
  bus_speed_without_stoppages = 60 ∧ 
  bus_speed_with_stoppages = 45 → 
  bus_stops_per_hour_in_minutes = 15 := by
  sorry

end bus_stops_per_hour_l601_60191


namespace open_parking_spots_fourth_level_l601_60190

theorem open_parking_spots_fourth_level :
  ∀ (n_first n_total : ℕ)
    (n_second_diff n_third_diff : ℕ),
    n_first = 4 →
    n_second_diff = 7 →
    n_third_diff = 6 →
    n_total = 46 →
    ∃ (n_first n_second n_third n_fourth : ℕ),
      n_second = n_first + n_second_diff ∧
      n_third = n_second + n_third_diff ∧
      n_first + n_second + n_third + n_fourth = n_total ∧
      n_fourth = 14 := by
  sorry

end open_parking_spots_fourth_level_l601_60190


namespace darnell_saves_money_l601_60132

-- Define conditions
def current_plan_cost := 12
def text_cost := 1
def call_cost := 3
def texts_per_month := 60
def calls_per_month := 60
def texts_per_unit := 30
def calls_per_unit := 20

-- Define the costs for the alternative plan
def alternative_texting_cost := (text_cost * (texts_per_month / texts_per_unit))
def alternative_calling_cost := (call_cost * (calls_per_month / calls_per_unit))
def alternative_plan_cost := alternative_texting_cost + alternative_calling_cost

-- Define the problem to prove
theorem darnell_saves_money :
  current_plan_cost - alternative_plan_cost = 1 :=
by
  sorry

end darnell_saves_money_l601_60132


namespace prime_squared_difference_divisible_by_24_l601_60123

theorem prime_squared_difference_divisible_by_24 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) :
  24 ∣ (p^2 - q^2) :=
sorry

end prime_squared_difference_divisible_by_24_l601_60123


namespace A_inter_complement_RB_eq_l601_60117

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x^2)}

def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

def complement_RB : Set ℝ := {x | x ≥ 1}

theorem A_inter_complement_RB_eq : A ∩ complement_RB = {x | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end A_inter_complement_RB_eq_l601_60117


namespace additional_treetags_l601_60120

noncomputable def initial_numerals : Finset ℕ := {1, 2, 3, 4}
noncomputable def initial_letters : Finset Char := {'A', 'E', 'I'}
noncomputable def initial_symbols : Finset Char := {'!', '@', '#', '$'}
noncomputable def added_numeral : Finset ℕ := {5}
noncomputable def added_symbols : Finset Char := {'&'}

theorem additional_treetags : 
  let initial_treetags := initial_numerals.card * initial_letters.card * initial_symbols.card
  let new_numerals := initial_numerals ∪ added_numeral
  let new_symbols := initial_symbols ∪ added_symbols
  let new_treetags := new_numerals.card * initial_letters.card * new_symbols.card
  new_treetags - initial_treetags = 27 := 
by 
  sorry

end additional_treetags_l601_60120


namespace jack_jill_next_in_step_l601_60137

theorem jack_jill_next_in_step (stride_jack : ℕ) (stride_jill : ℕ) : 
  stride_jack = 64 → stride_jill = 56 → Nat.lcm stride_jack stride_jill = 448 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end jack_jill_next_in_step_l601_60137


namespace desired_average_l601_60114

theorem desired_average (P1 P2 P3 : ℝ) (A : ℝ) 
  (hP1 : P1 = 74) 
  (hP2 : P2 = 84) 
  (hP3 : P3 = 67) 
  (hA : A = (P1 + P2 + P3) / 3) : 
  A = 75 :=
  sorry

end desired_average_l601_60114


namespace pages_needed_l601_60174

theorem pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) (total_packs : packs = 60) (cards_in_pack : cards_per_pack = 7) (capacity_per_page : cards_per_page = 10) : (packs * cards_per_pack) / cards_per_page = 42 := 
by
  -- Utilize the conditions
  have H1 : packs = 60 := total_packs
  have H2 : cards_per_pack = 7 := cards_in_pack
  have H3 : cards_per_page = 10 := capacity_per_page
  -- Use these to simplify and prove the target expression 
  sorry

end pages_needed_l601_60174


namespace value_of_a_minus_b_l601_60121

theorem value_of_a_minus_b (a b : ℝ) 
  (h₁ : (a-4)*(a+4) = 28*a - 112) 
  (h₂ : (b-4)*(b+4) = 28*b - 112) 
  (h₃ : a ≠ b)
  (h₄ : a > b) :
  a - b = 20 :=
sorry

end value_of_a_minus_b_l601_60121


namespace reasoning_is_inductive_l601_60159

-- Define conditions
def conducts_electricity (metal : String) : Prop :=
  metal = "copper" ∨ metal = "iron" ∨ metal = "aluminum" ∨ metal = "gold" ∨ metal = "silver"

-- Define the inductive reasoning type
def is_inductive_reasoning : Prop := 
  ∀ metals, conducts_electricity metals → (∀ m : String, conducts_electricity m → conducts_electricity m)

-- The theorem to prove
theorem reasoning_is_inductive : is_inductive_reasoning :=
by
  sorry

end reasoning_is_inductive_l601_60159


namespace fgh_deriv_at_0_l601_60157

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def h : ℝ → ℝ := sorry

-- Function Values at x = 0
axiom f_zero : f 0 = 1
axiom g_zero : g 0 = 2
axiom h_zero : h 0 = 3

-- Derivatives of the pairwise products at x = 0
axiom d_gh_zero : (deriv (λ x => g x * h x)) 0 = 4
axiom d_hf_zero : (deriv (λ x => h x * f x)) 0 = 5
axiom d_fg_zero : (deriv (λ x => f x * g x)) 0 = 6

-- We need to prove that the derivative of the product of f, g, h at x = 0 is 16
theorem fgh_deriv_at_0 : (deriv (λ x => f x * g x * h x)) 0 = 16 := by
  sorry

end fgh_deriv_at_0_l601_60157


namespace base_length_l601_60102

-- Definition: Isosceles triangle
structure IsoscelesTriangle :=
  (perimeter : ℝ)
  (side : ℝ)

-- Conditions: Perimeter and one side of the isosceles triangle
def given_triangle : IsoscelesTriangle := {
  perimeter := 26,
  side := 11
}

-- The problem to solve: length of the base given the perimeter and one side
theorem base_length : 
  (given_triangle.perimeter = 26 ∧ given_triangle.side = 11) →
  (∃ b : ℝ, b = 11 ∨ b = 7.5) :=
by 
  sorry

end base_length_l601_60102


namespace arithmetic_sequence_probability_l601_60126

def favorable_sequences : List (List ℕ) :=
  [[1, 2, 3], [1, 3, 5], [2, 3, 4], [2, 4, 6], [3, 4, 5], [4, 5, 6], 
   [3, 2, 1], [5, 3, 1], [4, 3, 2], [6, 4, 2], [5, 4, 3], [6, 5, 4], 
   [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := favorable_sequences.length

theorem arithmetic_sequence_probability : (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by
  sorry

end arithmetic_sequence_probability_l601_60126


namespace find_a_b_l601_60183

theorem find_a_b (a b : ℝ)
  (h1 : (0 - a)^2 + (-12 - b)^2 = 36)
  (h2 : (0 - a)^2 + (0 - b)^2 = 36) :
  a = 0 ∧ b = -6 :=
by
  sorry

end find_a_b_l601_60183


namespace range_of_a_l601_60198

open Real

noncomputable def A (x : ℝ) : Prop := (x + 1) / (x - 2) ≥ 0
noncomputable def B (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≥ 0

theorem range_of_a :
  (∀ x, A x → B x a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l601_60198
