import Mathlib

namespace carol_has_35_nickels_l1261_126102

def problem_statement : Prop :=
  ∃ (n d : ℕ), 5 * n + 10 * d = 455 ∧ n = d + 7 ∧ n = 35

theorem carol_has_35_nickels : problem_statement := by
  -- Proof goes here
  sorry

end carol_has_35_nickels_l1261_126102


namespace ratio_percent_l1261_126136

theorem ratio_percent (x : ℕ) (h : (15 / x : ℚ) = 60 / 100) : x = 25 := 
sorry

end ratio_percent_l1261_126136


namespace dormouse_stole_flour_l1261_126130

-- Define the suspects
inductive Suspect 
| MarchHare 
| MadHatter 
| Dormouse 

open Suspect 

-- Condition 1: Only one of three suspects stole the flour
def only_one_thief (s : Suspect) : Prop := 
  s = MarchHare ∨ s = MadHatter ∨ s = Dormouse

-- Condition 2: Only the person who stole the flour gave a truthful testimony
def truthful (thief : Suspect) (testimony : Suspect → Prop) : Prop :=
  testimony thief

-- Condition 3: The March Hare testified that the Mad Hatter stole the flour
def marchHare_testimony (s : Suspect) : Prop := 
  s = MadHatter

-- The theorem to prove: Dormouse stole the flour
theorem dormouse_stole_flour : 
  ∃ thief : Suspect, only_one_thief thief ∧ 
    (∀ s : Suspect, (s = thief ↔ truthful s marchHare_testimony) → thief = Dormouse) :=
by
  sorry

end dormouse_stole_flour_l1261_126130


namespace expand_product_l1261_126171

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by sorry

end expand_product_l1261_126171


namespace union_A_B_l1261_126109

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | x > 2}

theorem union_A_B :
  A ∪ B = {x : ℝ | 1 ≤ x} := sorry

end union_A_B_l1261_126109


namespace fraction_cube_l1261_126198

theorem fraction_cube (a b : ℚ) (h : (a / b) ^ 3 = 15625 / 1000000) : a / b = 1 / 4 :=
by
  sorry

end fraction_cube_l1261_126198


namespace juhye_initial_money_l1261_126160

theorem juhye_initial_money
  (M : ℝ)
  (h1 : M - (1 / 4) * M - (2 / 3) * ((3 / 4) * M) = 2500) :
  M = 10000 := by
  sorry

end juhye_initial_money_l1261_126160


namespace pencils_are_left_l1261_126126

-- Define the conditions
def original_pencils : ℕ := 87
def removed_pencils : ℕ := 4

-- Define the expected outcome
def pencils_left : ℕ := original_pencils - removed_pencils

-- Prove that the number of pencils left in the jar is 83
theorem pencils_are_left : pencils_left = 83 := by
  -- Placeholder for the proof
  sorry

end pencils_are_left_l1261_126126


namespace integral_of_2x2_cos3x_l1261_126167

theorem integral_of_2x2_cos3x :
  ∫ x in (0 : ℝ)..(2 * Real.pi), (2 * x ^ 2 - 15) * Real.cos (3 * x) = (8 * Real.pi) / 9 :=
by
  sorry

end integral_of_2x2_cos3x_l1261_126167


namespace dvaneft_shares_percentage_range_l1261_126172

theorem dvaneft_shares_percentage_range :
  ∀ (x y z n m : ℝ),
    (4 * x * n = y * m) →
    (x * n + y * m = z * (m + n)) →
    (16 ≤ y - x ∧ y - x ≤ 20) →
    (42 ≤ z ∧ z ≤ 60) →
    (12.5 ≤ (n / (2 * (n + m)) * 100) ∧ (n / (2 * (n + m)) * 100) ≤ 15) :=
by
  intros x y z n m h1 h2 h3 h4
  sorry

end dvaneft_shares_percentage_range_l1261_126172


namespace price_difference_l1261_126145

noncomputable def originalPriceStrawberries (s : ℝ) (sale_revenue_s : ℝ) := sale_revenue_s / (0.70 * s)
noncomputable def originalPriceBlueberries (b : ℝ) (sale_revenue_b : ℝ) := sale_revenue_b / (0.80 * b)

theorem price_difference
    (s : ℝ) (sale_revenue_s : ℝ)
    (b : ℝ) (sale_revenue_b : ℝ)
    (h1 : sale_revenue_s = 70 * (0.70 * s))
    (h2 : sale_revenue_b = 50 * (0.80 * b)) :
    originalPriceStrawberries (sale_revenue_s / 49) sale_revenue_s - originalPriceBlueberries (sale_revenue_b / 40) sale_revenue_b = 0.71 :=
by
  sorry

end price_difference_l1261_126145


namespace trajectory_of_point_l1261_126184

theorem trajectory_of_point (x y : ℝ)
  (h1 : (x - 1)^2 + (y - 1)^2 = ((3 * x + y - 4)^2) / 10) :
  x - 3 * y + 2 = 0 :=
sorry

end trajectory_of_point_l1261_126184


namespace tan_identity_example_l1261_126162

theorem tan_identity_example (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 :=
by
  sorry

end tan_identity_example_l1261_126162


namespace sum_of_squares_divisibility_l1261_126125

theorem sum_of_squares_divisibility
  (p : ℕ) (hp : Nat.Prime p)
  (x y z : ℕ)
  (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzp : z < p)
  (hmod_eq : ∀ a b c : ℕ, a^3 % p = b^3 % p → b^3 % p = c^3 % p → a^3 % p = c^3 % p) :
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
  sorry

end sum_of_squares_divisibility_l1261_126125


namespace negation_of_prop_l1261_126101

open Classical

theorem negation_of_prop (h : ∀ x : ℝ, x^2 + x + 1 > 0) : ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
sorry

end negation_of_prop_l1261_126101


namespace calculate_pow_zero_l1261_126187

theorem calculate_pow_zero: (2023 - Real.pi) ≠ 0 → (2023 - Real.pi)^0 = 1 := by
  -- Proof
  sorry

end calculate_pow_zero_l1261_126187


namespace family_ate_doughnuts_l1261_126148

variable (box_initial : ℕ) (box_left : ℕ) (dozen : ℕ)

-- Define the initial and remaining conditions
def dozen_value : ℕ := 12
def box_initial_value : ℕ := 2 * dozen_value
def doughnuts_left_value : ℕ := 16

theorem family_ate_doughnuts (h1 : box_initial = box_initial_value) (h2 : box_left = doughnuts_left_value) :
  box_initial - box_left = 8 := by
  -- h1 says the box initially contains 2 dozen, which is 24.
  -- h2 says that there are 16 doughnuts left.
  sorry

end family_ate_doughnuts_l1261_126148


namespace find_number_l1261_126116

theorem find_number (N x : ℕ) (h1 : 3 * x = (N - x) + 26) (h2 : x = 22) : N = 62 :=
by
  sorry

end find_number_l1261_126116


namespace first_part_lending_years_l1261_126169

-- Definitions and conditions from the problem
def total_sum : ℕ := 2691
def second_part : ℕ := 1656
def rate_first_part : ℚ := 3 / 100
def rate_second_part : ℚ := 5 / 100
def time_second_part : ℕ := 3

-- Calculated first part
def first_part : ℕ := total_sum - second_part

-- Prove that the number of years (n) the first part is lent is 8
theorem first_part_lending_years : 
  ∃ n : ℕ, (first_part : ℚ) * rate_first_part * n = (second_part : ℚ) * rate_second_part * time_second_part ∧ n = 8 :=
by
  -- Proof steps would go here
  sorry

end first_part_lending_years_l1261_126169


namespace expected_winnings_is_correct_l1261_126115

variable (prob_1 prob_23 prob_456 : ℚ)
variable (win_1 win_23 loss_456 : ℚ)

theorem expected_winnings_is_correct :
  prob_1 = 1/4 → 
  prob_23 = 1/2 → 
  prob_456 = 1/4 → 
  win_1 = 2 → 
  win_23 = 4 → 
  loss_456 = -3 → 
  (prob_1 * win_1 + prob_23 * win_23 + prob_456 * loss_456 = 1.75) :=
by
  intros
  sorry

end expected_winnings_is_correct_l1261_126115


namespace ratio_of_Victoria_to_Beacon_l1261_126129

def Richmond_population : ℕ := 3000
def Beacon_population : ℕ := 500
def Victoria_population : ℕ := Richmond_population - 1000
def ratio_Victoria_Beacon : ℕ := Victoria_population / Beacon_population

theorem ratio_of_Victoria_to_Beacon : ratio_Victoria_Beacon = 4 := 
by
  unfold ratio_Victoria_Beacon Victoria_population Richmond_population Beacon_population
  sorry

end ratio_of_Victoria_to_Beacon_l1261_126129


namespace noah_total_wattage_l1261_126134

def bedroom_wattage := 6
def office_wattage := 3 * bedroom_wattage
def living_room_wattage := 4 * bedroom_wattage
def hours_on := 2

theorem noah_total_wattage : 
  bedroom_wattage * hours_on + 
  office_wattage * hours_on + 
  living_room_wattage * hours_on = 96 := by
  sorry

end noah_total_wattage_l1261_126134


namespace exists_subset_sum_mod_p_l1261_126151

theorem exists_subset_sum_mod_p (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ)
  (hA_card : A.card = p - 1) (hA : ∀ a ∈ A, a % p ≠ 0) : 
  ∀ n : ℕ, n < p → ∃ B ⊆ A, (B.sum id) % p = n :=
by
  sorry

end exists_subset_sum_mod_p_l1261_126151


namespace initial_action_figures_correct_l1261_126195

def initial_action_figures (x : ℕ) : Prop :=
  x + 11 - 10 = 8

theorem initial_action_figures_correct :
  ∃ x : ℕ, initial_action_figures x ∧ x = 7 :=
by
  sorry

end initial_action_figures_correct_l1261_126195


namespace max_product_of_triangle_sides_l1261_126155

theorem max_product_of_triangle_sides (a c : ℝ) (ha : a ≥ 0) (hc : c ≥ 0) :
  ∃ b : ℝ, b = 4 ∧ ∃ B : ℝ, B = 60 * (π / 180) ∧ a^2 + c^2 - a * c = b^2 ∧ a * c ≤ 16 :=
by
  sorry

end max_product_of_triangle_sides_l1261_126155


namespace tan_half_angle_lt_l1261_126104

theorem tan_half_angle_lt (x : ℝ) (h : 0 < x ∧ x ≤ π / 2) : 
  Real.tan (x / 2) < x := 
by
  sorry

end tan_half_angle_lt_l1261_126104


namespace complex_number_in_first_quadrant_l1261_126103

-- Definition of the imaginary unit
def i : ℂ := Complex.I

-- Definition of the complex number z
def z : ℂ := i * (1 - i)

-- Coordinates of the complex number z
def z_coords : ℝ × ℝ := (z.re, z.im)

-- Statement asserting that the point corresponding to z lies in the first quadrant
theorem complex_number_in_first_quadrant : z_coords.fst > 0 ∧ z_coords.snd > 0 := 
by
  sorry

end complex_number_in_first_quadrant_l1261_126103


namespace remainder_of_product_mod_seven_l1261_126132

-- Definitions derived from the conditions
def seq : List ℕ := [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

-- The main statement to prove
theorem remainder_of_product_mod_seven : 
  (seq.foldl (λ acc x => acc * x) 1) % 7 = 0 := by
  sorry

end remainder_of_product_mod_seven_l1261_126132


namespace equidistant_points_eq_two_l1261_126166

noncomputable def number_of_equidistant_points (O : Point) (r d : ℝ) 
  (h1 : d > r) : ℕ := 
2

theorem equidistant_points_eq_two (O : Point) (r d : ℝ) 
  (h1 : d > r) : number_of_equidistant_points O r d h1 = 2 :=
by
  sorry

end equidistant_points_eq_two_l1261_126166


namespace find_remainder_q_neg2_l1261_126163

-- Define q(x)
def q (x : ℝ) (D E F : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 6

-- The given conditions in the problem
variable {D E F : ℝ}
variable (h_q_2 : q 2 D E F = 14)

-- The statement we aim to prove
theorem find_remainder_q_neg2 (h_q_2 : q 2 D E F = 14) : q (-2) D E F = 14 :=
sorry

end find_remainder_q_neg2_l1261_126163


namespace rectangle_area_l1261_126106

theorem rectangle_area (x y : ℝ) (L W : ℝ) (h_diagonal : (L ^ 2 + W ^ 2) ^ (1 / 2) = x + y) (h_ratio : L / W = 3 / 2) : 
  L * W = (6 * (x + y) ^ 2) / 13 := 
sorry

end rectangle_area_l1261_126106


namespace triangle_area_l1261_126194

-- Define the lines and the x-axis
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 1
noncomputable def line2 (x : ℝ) : ℝ := 1 - 5 * x
noncomputable def x_axis (x : ℝ) : ℝ := 0

-- Define intersection points
noncomputable def intersect_x_axis1 : ℝ × ℝ := (-1 / 2, 0)
noncomputable def intersect_x_axis2 : ℝ × ℝ := (1 / 5, 0)
noncomputable def intersect_lines : ℝ × ℝ := (0, 1)

-- State the theorem for the area of the triangle
theorem triangle_area : 
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  (1 / 2) * d * h = 7 / 20 := 
by
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  sorry

end triangle_area_l1261_126194


namespace ratio_sheep_horses_l1261_126175

theorem ratio_sheep_horses
  (horse_food_per_day : ℕ)
  (total_horse_food : ℕ)
  (number_of_sheep : ℕ)
  (number_of_horses : ℕ)
  (gcd_sheep_horses : ℕ):
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  number_of_sheep = 40 →
  number_of_horses = total_horse_food / horse_food_per_day →
  gcd number_of_sheep number_of_horses = 8 →
  (number_of_sheep / gcd_sheep_horses = 5) ∧ (number_of_horses / gcd_sheep_horses = 7) :=
by
  intros
  sorry

end ratio_sheep_horses_l1261_126175


namespace sum_of_two_numbers_l1261_126179

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 16) (h2 : 1/x = 3 * (1/y)) : 
  x + y = 16 * Real.sqrt 3 / 3 :=
by
  sorry

end sum_of_two_numbers_l1261_126179


namespace poles_inside_base_l1261_126133

theorem poles_inside_base :
  ∃ n : ℕ, 2015 + n ≡ 0 [MOD 36] ∧ n = 1 :=
sorry

end poles_inside_base_l1261_126133


namespace black_lambs_count_l1261_126111

-- Definitions based on the conditions given
def total_lambs : ℕ := 6048
def white_lambs : ℕ := 193

-- Theorem statement
theorem black_lambs_count : total_lambs - white_lambs = 5855 :=
by 
  -- the proof would be provided here
  sorry

end black_lambs_count_l1261_126111


namespace JeremyTotalExpenses_l1261_126153

noncomputable def JeremyExpenses : ℝ :=
  let motherGift := 400
  let fatherGift := 280
  let sisterGift := 100
  let brotherGift := 60
  let friendGift := 50
  let giftWrappingRate := 0.07
  let taxRate := 0.09
  let miscExpenses := 40
  let wrappingCost := motherGift * giftWrappingRate
                  + fatherGift * giftWrappingRate
                  + sisterGift * giftWrappingRate
                  + brotherGift * giftWrappingRate
                  + friendGift * giftWrappingRate
  let totalGiftCost := motherGift + fatherGift + sisterGift + brotherGift + friendGift
  let totalTax := totalGiftCost * taxRate
  wrappingCost + totalTax + miscExpenses

theorem JeremyTotalExpenses : JeremyExpenses = 182.40 := by
  sorry

end JeremyTotalExpenses_l1261_126153


namespace given_condition_l1261_126144

variable (a : ℝ)

theorem given_condition
  (h1 : (a + 1/a)^2 = 5) :
  a^2 + 1/a^2 + a^3 + 1/a^3 = 3 + 2 * Real.sqrt 5 :=
sorry

end given_condition_l1261_126144


namespace inequality_solution_l1261_126197

theorem inequality_solution :
  ∀ x : ℝ, (5 / 24 + |x - 11 / 48| < 5 / 16 ↔ (1 / 8 < x ∧ x < 1 / 3)) :=
by
  intro x
  sorry

end inequality_solution_l1261_126197


namespace range_of_a_l1261_126154

variables (a x : ℝ) -- Define real number variables a and x

-- Define proposition p
def p : Prop := (a - 2) * x * x + 2 * (a - 2) * x - 4 < 0 -- Inequality condition for any real x

-- Define proposition q
def q : Prop := 0 < a ∧ a < 1 -- Condition for logarithmic function to be strictly decreasing

-- Lean 4 statement for the proof problem
theorem range_of_a (Hpq : (p a x ∨ q a) ∧ ¬ (p a x ∧ q a)) :
  (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
sorry

end range_of_a_l1261_126154


namespace line_ellipse_common_point_l1261_126112

theorem line_ellipse_common_point (k : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), y = k * x + 1 →
    (y^2 / m + x^2 / 5 ≤ 1)) ↔ (m ≥ 1 ∧ m ≠ 5) :=
by sorry

end line_ellipse_common_point_l1261_126112


namespace mary_added_peanuts_l1261_126173

-- Defining the initial number of peanuts
def initial_peanuts : ℕ := 4

-- Defining the final number of peanuts
def total_peanuts : ℕ := 10

-- Defining the number of peanuts added by Mary
def peanuts_added : ℕ := total_peanuts - initial_peanuts

-- The proof problem is to show that Mary added 6 peanuts
theorem mary_added_peanuts : peanuts_added = 6 :=
by
  -- We leave the proof part as a sorry as per instruction
  sorry

end mary_added_peanuts_l1261_126173


namespace find_profit_range_l1261_126121

noncomputable def profit_range (x : ℝ) : Prop :=
  0 < x → 0.15 * (1 + 0.25 * x) * (100000 - x) ≥ 0.15 * 100000

theorem find_profit_range (x : ℝ) : profit_range x → 0 < x ∧ x ≤ 6 :=
by
  sorry

end find_profit_range_l1261_126121


namespace rectangle_width_length_ratio_l1261_126143

theorem rectangle_width_length_ratio (w : ℕ) (h : w + 10 = 15) : w / 10 = 1 / 2 :=
by sorry

end rectangle_width_length_ratio_l1261_126143


namespace balls_removal_l1261_126157

theorem balls_removal (total_balls : ℕ) (percent_green initial_green initial_yellow remaining_percent : ℝ)
    (h_percent_green : percent_green = 0.7)
    (h_total_balls : total_balls = 600)
    (h_initial_green : initial_green = percent_green * total_balls)
    (h_initial_yellow : initial_yellow = total_balls - initial_green)
    (h_remaining_percent : remaining_percent = 0.6) :
    ∃ x : ℝ, (initial_green - x) / (total_balls - x) = remaining_percent ∧ x = 150 := 
by 
  sorry

end balls_removal_l1261_126157


namespace nicolai_ate_6_pounds_of_peaches_l1261_126138

noncomputable def total_weight_pounds : ℝ := 8
noncomputable def pound_to_ounce : ℝ := 16
noncomputable def mario_weight_ounces : ℝ := 8
noncomputable def lydia_weight_ounces : ℝ := 24

theorem nicolai_ate_6_pounds_of_peaches :
  (total_weight_pounds * pound_to_ounce - (mario_weight_ounces + lydia_weight_ounces)) / pound_to_ounce = 6 :=
by
  sorry

end nicolai_ate_6_pounds_of_peaches_l1261_126138


namespace volume_of_sphere_l1261_126182

theorem volume_of_sphere (r : ℝ) (h : r = 3) : (4 / 3) * π * r ^ 3 = 36 * π := 
by
  sorry

end volume_of_sphere_l1261_126182


namespace length_chord_AB_l1261_126107

-- Given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Prove the length of the chord AB
theorem length_chord_AB : 
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧ A ≠ B) →
  (∃ (length : ℝ), length = 2*Real.sqrt 2) :=
by
  sorry

end length_chord_AB_l1261_126107


namespace adult_elephant_weekly_bananas_l1261_126174

theorem adult_elephant_weekly_bananas (daily_bananas : Nat) (days_in_week : Nat) (H1 : daily_bananas = 90) (H2 : days_in_week = 7) :
  daily_bananas * days_in_week = 630 :=
by
  sorry

end adult_elephant_weekly_bananas_l1261_126174


namespace intersection_points_l1261_126181

noncomputable def parabola (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 4
noncomputable def line (x : ℝ) : ℝ := -x + 2

theorem intersection_points :
  (parabola (-1 / 3) = line (-1 / 3) ∧ parabola (-2) = line (-2)) ∧
  (parabola (-1 / 3) = 7 / 3) ∧ (parabola (-2) = 4) :=
by
  sorry

end intersection_points_l1261_126181


namespace find_n_l1261_126105

theorem find_n (n : ℤ) (h : 8 + 6 = n + 8) : n = 6 :=
by
  sorry

end find_n_l1261_126105


namespace find_flights_of_stairs_l1261_126149

def t_flight : ℕ := 11
def t_bomb : ℕ := 72
def t_spent : ℕ := 165
def t_diffuse : ℕ := 17

def total_time_running : ℕ := t_spent + (t_bomb - t_diffuse)
def flights_of_stairs (t_run: ℕ) (time_per_flight: ℕ) : ℕ := t_run / time_per_flight

theorem find_flights_of_stairs :
  flights_of_stairs total_time_running t_flight = 20 :=
by
  sorry

end find_flights_of_stairs_l1261_126149


namespace general_equation_of_curve_l1261_126196

variable (θ x y : ℝ)

theorem general_equation_of_curve
  (h1 : x = Real.cos θ - 1)
  (h2 : y = Real.sin θ + 1) :
  (x + 1)^2 + (y - 1)^2 = 1 := sorry

end general_equation_of_curve_l1261_126196


namespace sports_probability_boy_given_sports_probability_l1261_126152

variable (x : ℝ) -- Number of girls

def number_of_boys := 1.5 * x
def boys_liking_sports := 0.4 * number_of_boys x
def girls_liking_sports := 0.2 * x
def total_students := x + number_of_boys x
def total_students_liking_sports := boys_liking_sports x + girls_liking_sports x

theorem sports_probability : (total_students_liking_sports x) / (total_students x) = 8 / 25 := 
sorry

theorem boy_given_sports_probability :
  (boys_liking_sports x) / (total_students_liking_sports x) = 3 / 4 := 
sorry

end sports_probability_boy_given_sports_probability_l1261_126152


namespace fraction_division_l1261_126123

theorem fraction_division:
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end fraction_division_l1261_126123


namespace minimum_detectors_203_l1261_126110

def minimum_detectors (length : ℕ) : ℕ :=
  length / 3 * 2 -- This models the generalization for 1 × (3k + 2)

theorem minimum_detectors_203 : minimum_detectors 203 = 134 :=
by
  -- Length is 203, k = 67 which follows from the floor division
  -- Therefore, minimum detectors = 2 * 67 = 134
  sorry

end minimum_detectors_203_l1261_126110


namespace largest_constant_inequality_l1261_126178

theorem largest_constant_inequality :
  ∃ C, (∀ x y z : ℝ, x^2 + y^2 + z^3 + 1 ≥ C * (x + y + z)) ∧ (C = Real.sqrt 2) :=
sorry

end largest_constant_inequality_l1261_126178


namespace correct_option_for_sentence_completion_l1261_126161

-- Define the mathematical formalization of the problem
def sentence_completion_problem : String × (List String) := 
    ("One of the most important questions they had to consider was _ of public health.", 
     ["what", "this", "that", "which"])

-- Define the correct answer
def correct_answer : String := "that"

-- The formal statement of the problem in Lean 4
theorem correct_option_for_sentence_completion 
    (problem : String × (List String)) (answer : String) :
    answer = "that" :=
by
  sorry  -- Proof to be completed

end correct_option_for_sentence_completion_l1261_126161


namespace value_of_b_l1261_126193

theorem value_of_b :
  (∃ b : ℝ, (1 / Real.log b / Real.log 3 + 1 / Real.log b / Real.log 4 + 1 / Real.log b / Real.log 5 = 1) → b = 60) :=
by
  sorry

end value_of_b_l1261_126193


namespace greg_experienced_less_rain_l1261_126189

theorem greg_experienced_less_rain (rain_day1 rain_day2 rain_day3 rain_house : ℕ) 
  (h1 : rain_day1 = 3) 
  (h2 : rain_day2 = 6) 
  (h3 : rain_day3 = 5) 
  (h4 : rain_house = 26) :
  rain_house - (rain_day1 + rain_day2 + rain_day3) = 12 :=
by
  sorry

end greg_experienced_less_rain_l1261_126189


namespace quadratic_inequality_solution_l1261_126176

theorem quadratic_inequality_solution :
  {x : ℝ | 2*x^2 - 3*x - 2 ≥ 0} = {x : ℝ | x ≤ -1/2 ∨ x ≥ 2} :=
sorry

end quadratic_inequality_solution_l1261_126176


namespace gcd_power_diff_l1261_126141

theorem gcd_power_diff (n m : ℕ) (h₁ : n = 2025) (h₂ : m = 2007) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2^18 - 1 :=
by
  sorry

end gcd_power_diff_l1261_126141


namespace g_of_g_of_2_l1261_126124

def g (x : ℝ) : ℝ := 4 * x^2 - 3

theorem g_of_g_of_2 : g (g 2) = 673 := 
by 
  sorry

end g_of_g_of_2_l1261_126124


namespace find_roots_l1261_126114

theorem find_roots : 
  (∃ x : ℝ, (x-1) * (x-2) * (x+1) * (x-5) = 0) ↔ 
  x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by sorry

end find_roots_l1261_126114


namespace min_value_a4b3c2_l1261_126108

theorem min_value_a4b3c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : (∀ a b c : ℝ, a^4 * b^3 * c^2 ≥ 1/(9^9)) :=
by
  sorry

end min_value_a4b3c2_l1261_126108


namespace linear_function_expression_l1261_126146

theorem linear_function_expression (k b : ℝ) (h : ∀ x : ℝ, (1 ≤ x ∧ x ≤ 4 → 3 ≤ k * x + b ∧ k * x + b ≤ 6)) :
  (k = 1 ∧ b = 2) ∨ (k = -1 ∧ b = 7) :=
by
  sorry

end linear_function_expression_l1261_126146


namespace james_found_bills_l1261_126156

def initial_money : ℝ := 75
def final_money : ℝ := 135
def bill_value : ℝ := 20

theorem james_found_bills :
  (final_money - initial_money) / bill_value = 3 :=
by
  sorry

end james_found_bills_l1261_126156


namespace find_a7_a8_l1261_126158

noncomputable def geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :=
∀ n, a (n + 1) = r * a n

theorem find_a7_a8
  (a : ℕ → ℝ)
  (r : ℝ)
  (hs : geometric_sequence_property a r)
  (h1 : a 1 + a 2 = 40)
  (h2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end find_a7_a8_l1261_126158


namespace find_A_l1261_126185

variable (x A B C : ℝ)

theorem find_A :
  (∃ A B C : ℝ, (∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 → 
  (1 / (x^3 + 2 * x^2 - 19 * x - 30) = 
  (A / (x + 3)) + (B / (x - 2)) + (C / (x - 2)^2)) ∧ 
  A = 1 / 25)) :=
by
  sorry

end find_A_l1261_126185


namespace derivative_of_f_l1261_126122

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_of_f (x : ℝ) (h : 0 < x) :
    deriv f x = (1 - Real.log x) / (x ^ 2) := 
sorry

end derivative_of_f_l1261_126122


namespace top_card_is_5_or_king_l1261_126190

-- Define the number of cards in a deck
def total_cards : ℕ := 52

-- Define the number of 5s in a deck
def number_of_5s : ℕ := 4

-- Define the number of Kings in a deck
def number_of_kings : ℕ := 4

-- Define the number of favorable outcomes (cards that are either 5 or King)
def favorable_outcomes : ℕ := number_of_5s + number_of_kings

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_cards

-- Theorem: The probability that the top card is either a 5 or a King is 2/13
theorem top_card_is_5_or_king (h_total_cards : total_cards = 52)
    (h_number_of_5s : number_of_5s = 4)
    (h_number_of_kings : number_of_kings = 4) :
    probability = 2 / 13 := by
  -- Proof would go here
  sorry

end top_card_is_5_or_king_l1261_126190


namespace decreasing_cubic_function_l1261_126137

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^3 - x

-- Define the condition that f is decreasing on (-∞, ∞)
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

-- The main theorem that needs to be proven
theorem decreasing_cubic_function (m : ℝ) : is_decreasing (f m) → m < 0 := 
by
  sorry

end decreasing_cubic_function_l1261_126137


namespace height_of_taller_tree_l1261_126120

-- Define the conditions as hypotheses:
variables (h₁ h₂ : ℝ)
-- The top of one tree is 24 feet higher than the top of another tree
variables (h_difference : h₁ = h₂ + 24)
-- The heights of the two trees are in the ratio 2:3
variables (h_ratio : h₂ / h₁ = 2 / 3)

theorem height_of_taller_tree : h₁ = 72 :=
by
  -- This is the place where the solution steps would be applied
  sorry

end height_of_taller_tree_l1261_126120


namespace no_real_roots_of_quad_eq_l1261_126150

theorem no_real_roots_of_quad_eq (k : ℝ) :
  ∀ x : ℝ, ¬ (x^2 - 2*x - k = 0) ↔ k < -1 :=
by sorry

end no_real_roots_of_quad_eq_l1261_126150


namespace five_fridays_in_september_l1261_126119

theorem five_fridays_in_september (year : ℕ) :
  (∃ (july_wednesdays : ℕ × ℕ × ℕ × ℕ × ℕ), 
     (july_wednesdays = (1, 8, 15, 22, 29) ∨ 
      july_wednesdays = (2, 9, 16, 23, 30) ∨ 
      july_wednesdays = (3, 10, 17, 24, 31)) ∧ 
      september_days = 30) → 
  ∃ (september_fridays : ℕ × ℕ × ℕ × ℕ × ℕ), 
  (september_fridays = (1, 8, 15, 22, 29)) :=
by
  sorry

end five_fridays_in_september_l1261_126119


namespace line_intersects_ellipse_with_conditions_l1261_126140

theorem line_intersects_ellipse_with_conditions :
  ∃ l : ℝ → ℝ, (∃ A B : ℝ × ℝ, 
  (A.fst^2/6 + A.snd^2/3 = 1 ∧ B.fst^2/6 + B.snd^2/3 = 1) ∧
  A.fst > 0 ∧ A.snd > 0 ∧ B.fst > 0 ∧ B.snd > 0 ∧
  (∃ M N : ℝ × ℝ, 
    M.snd = 0 ∧ N.fst = 0 ∧
    M.fst^2 + N.snd^2 = (2 * Real.sqrt 3)^2 ∧
    (M.snd - A.snd)^2 + (M.fst - A.fst)^2 = (N.fst - B.fst)^2 + (N.snd - B.snd)^2) ∧
    (∀ x, l x + Real.sqrt 2 * x - 2 * Real.sqrt 2 = 0)
) :=
sorry

end line_intersects_ellipse_with_conditions_l1261_126140


namespace longest_side_of_rectangle_l1261_126192

theorem longest_side_of_rectangle (l w : ℕ) 
  (h1 : 2 * l + 2 * w = 240) 
  (h2 : l * w = 1920) : 
  l = 101 ∨ w = 101 :=
sorry

end longest_side_of_rectangle_l1261_126192


namespace binomial_equality_l1261_126142

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l1261_126142


namespace function_relationship_selling_price_for_profit_max_profit_l1261_126131

-- Step (1): Prove the function relationship between y and x
theorem function_relationship (x y: ℝ) (h1 : ∀ x, y = -2*x + 80)
  (h2 : x = 22 ∧ y = 36 ∨ x = 24 ∧ y = 32) :
  y = -2*x + 80 := by
  sorry

-- Step (2): Selling price per book for a 150 yuan profit per week
theorem selling_price_for_profit (x: ℝ) (hx : 20 ≤ x ∧ x ≤ 28) (profit : ℝ)
  (h_profit : profit = (x - 20) * (-2*x + 80)) (h2 : profit = 150) : 
  x = 25 := by
  sorry

-- Step (3): Maximizing the weekly profit
theorem max_profit (x w: ℝ) (hx : 20 ≤ x ∧ x ≤ 28) 
  (profit : ∀ x, w = (x - 20) * (-2*x + 80)) :
  w = 192 ∧ x = 28 := by
  sorry

end function_relationship_selling_price_for_profit_max_profit_l1261_126131


namespace mans_rate_in_still_water_l1261_126170

theorem mans_rate_in_still_water (Vm Vs : ℝ) (h1 : Vm + Vs = 14) (h2 : Vm - Vs = 4) : Vm = 9 :=
by
  sorry

end mans_rate_in_still_water_l1261_126170


namespace find_x_values_l1261_126139

-- Defining the given condition as a function
def equation (x : ℝ) : Prop :=
  (4 / (Real.sqrt (x + 5) - 7)) +
  (3 / (Real.sqrt (x + 5) - 2)) +
  (6 / (Real.sqrt (x + 5) + 2)) +
  (9 / (Real.sqrt (x + 5) + 7)) = 0

-- Statement of the theorem in Lean
theorem find_x_values :
  equation ( -796 / 169) ∨ equation (383 / 22) :=
sorry

end find_x_values_l1261_126139


namespace cupcakes_leftover_l1261_126159

theorem cupcakes_leftover {total_cupcakes nutty_cupcakes gluten_free_cupcakes children children_no_nuts child_only_gf leftover_nutty leftover_regular : Nat} :
  total_cupcakes = 84 →
  children = 7 →
  nutty_cupcakes = 18 →
  gluten_free_cupcakes = 25 →
  children_no_nuts = 2 →
  child_only_gf = 1 →
  leftover_nutty = 3 →
  leftover_regular = 2 →
  leftover_nutty + leftover_regular = 5 :=
by
  sorry

end cupcakes_leftover_l1261_126159


namespace commutative_not_associative_l1261_126180

variable (k : ℝ) (h_k : 0 < k)

noncomputable def star (x y : ℝ) : ℝ := (x * y + k) / (x + y + k)

theorem commutative (x y : ℝ) (h_x : 0 < x) (h_y : 0 < y) :
  star k x y = star k y x :=
by sorry

theorem not_associative (x y z : ℝ) (h_x : 0 < x) (h_y : 0 < y) (h_z : 0 < z) :
  ¬(star k (star k x y) z = star k x (star k y z)) :=
by sorry

end commutative_not_associative_l1261_126180


namespace six_coins_not_sum_to_14_l1261_126165

def coin_values : Set ℕ := {1, 5, 10, 25}

theorem six_coins_not_sum_to_14 (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 ∈ coin_values) (h2 : a2 ∈ coin_values) (h3 : a3 ∈ coin_values) (h4 : a4 ∈ coin_values) (h5 : a5 ∈ coin_values) (h6 : a6 ∈ coin_values) : a1 + a2 + a3 + a4 + a5 + a6 ≠ 14 := 
sorry

end six_coins_not_sum_to_14_l1261_126165


namespace possible_values_of_X_l1261_126164

-- Define the conditions and the problem
def defective_products_total := 3
def total_products := 10
def selected_products := 2

-- Define the random variable X
def X (n : ℕ) : ℕ := n / selected_products

-- Now the statement to prove is that X can only take the values {0, 1, 2}
theorem possible_values_of_X :
  ∀ (X : ℕ → ℕ), ∃ (vals : Set ℕ), (vals = {0, 1, 2} ∧ ∀ (n : ℕ), X n ∈ vals) :=
by
  sorry

end possible_values_of_X_l1261_126164


namespace find_x_l1261_126127

-- Definitions used in conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (4, x)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Main statement of the problem to be proved
theorem find_x (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = -1) : x = -1 / 5 :=
by {
  sorry
}

end find_x_l1261_126127


namespace expression_evaluation_l1261_126177

def evaluate_expression : ℝ := (-1) ^ 51 + 3 ^ (2^3 + 5^2 - 7^2)

theorem expression_evaluation :
  evaluate_expression = -1 + (1 / 43046721) :=
by
  sorry

end expression_evaluation_l1261_126177


namespace terminating_fraction_count_l1261_126191

theorem terminating_fraction_count : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 299 ∧ (∃ k, n = 3 * k)) ∧ 
  (∃ (count : ℕ), count = 99) :=
by
  sorry

end terminating_fraction_count_l1261_126191


namespace find_valid_n_l1261_126199

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_valid_n (n : ℕ) (h1 : n > 0) (h2 : n < 200) (h3 : is_square (n^2 + (n + 1)^2)) :
  n = 3 ∨ n = 20 ∨ n = 119 :=
by
  sorry

end find_valid_n_l1261_126199


namespace trajectory_of_midpoint_l1261_126188

theorem trajectory_of_midpoint (x y : ℝ) (A B : ℝ × ℝ) 
  (hB : B = (4, 0)) (hA_on_circle : (A.1)^2 + (A.2)^2 = 4)
  (hM : ((x, y) = ( (A.1 + B.1)/2, (A.2 + B.2)/2))) :
  (x - 2)^2 + y^2 = 1 :=
sorry

end trajectory_of_midpoint_l1261_126188


namespace fraction_simplification_l1261_126135

theorem fraction_simplification :
  (2/5 + 3/4) / (4/9 + 1/6) = (207/110) := by
  sorry

end fraction_simplification_l1261_126135


namespace remainder_problem_l1261_126147

theorem remainder_problem (d r : ℤ) (h1 : 1237 % d = r)
    (h2 : 1694 % d = r) (h3 : 2791 % d = r) (hd : d > 1) :
    d - r = 134 := sorry

end remainder_problem_l1261_126147


namespace find_f_log_3_54_l1261_126100

noncomputable def f : ℝ → ℝ := sorry  -- Since we have to define a function and we do not need the exact implementation.

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_property : ∀ x : ℝ, f (x + 2) = - 1 / f x
axiom interval_property : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 3 ^ x

theorem find_f_log_3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 :=
by
  sorry


end find_f_log_3_54_l1261_126100


namespace percentage_deposit_paid_l1261_126118

theorem percentage_deposit_paid (D R T : ℝ) (hd : D = 105) (hr : R = 945) (ht : T = D + R) : (D / T) * 100 = 10 := by
  sorry

end percentage_deposit_paid_l1261_126118


namespace poster_distance_from_wall_end_l1261_126168

theorem poster_distance_from_wall_end (w_wall w_poster : ℝ) (h1 : w_wall = 25) (h2 : w_poster = 4) (h3 : 2 * x + w_poster = w_wall) : x = 10.5 :=
by
  sorry

end poster_distance_from_wall_end_l1261_126168


namespace integer_not_in_range_l1261_126183

theorem integer_not_in_range (g : ℝ → ℤ) :
  (∀ x, x > -3 → g x = Int.ceil (2 / (x + 3))) ∧
  (∀ x, x < -3 → g x = Int.floor (2 / (x + 3))) →
  ∀ z : ℤ, (∃ x, g x = z) ↔ z ≠ 0 :=
by
  intros h z
  sorry

end integer_not_in_range_l1261_126183


namespace meaningful_sqrt_range_l1261_126128

theorem meaningful_sqrt_range (x : ℝ) : (3 * x - 6 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end meaningful_sqrt_range_l1261_126128


namespace andrew_game_night_expenses_l1261_126113

theorem andrew_game_night_expenses : 
  let cost_per_game := 9 
  let number_of_games := 5 
  total_money_spent = cost_per_game * number_of_games 
→ total_money_spent = 45 := 
by
  intro cost_per_game number_of_games total_money_spent
  sorry

end andrew_game_night_expenses_l1261_126113


namespace Tom_search_cost_l1261_126186

theorem Tom_search_cost (first_5_days_rate: ℕ) (first_5_days: ℕ) (remaining_days_rate: ℕ) (total_days: ℕ) : 
  first_5_days_rate = 100 → 
  first_5_days = 5 → 
  remaining_days_rate = 60 → 
  total_days = 10 → 
  (first_5_days * first_5_days_rate + (total_days - first_5_days) * remaining_days_rate) = 800 := 
by 
  intros h1 h2 h3 h4 
  sorry

end Tom_search_cost_l1261_126186


namespace impossible_configuration_l1261_126117

-- Define the initial state of stones in boxes
def stones_in_box (n : ℕ) : ℕ :=
  if n ≥ 1 ∧ n ≤ 100 then n else 0

-- Define the condition for moving stones between boxes
def can_move_stones (box1 box2 : ℕ) : Prop :=
  stones_in_box box1 + stones_in_box box2 = 101

-- The proposition: it is impossible to achieve the desired configuration
theorem impossible_configuration :
  ¬ ∃ boxes : ℕ → ℕ, 
    (boxes 70 = 69) ∧ 
    (boxes 50 = 51) ∧ 
    (∀ n, n ≠ 70 → n ≠ 50 → boxes n = stones_in_box n) ∧
    (∀ n1 n2, can_move_stones n1 n2 → (boxes n1 + boxes n2 = 101)) :=
sorry

end impossible_configuration_l1261_126117
