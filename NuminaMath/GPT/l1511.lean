import Mathlib

namespace total_seedlings_transferred_l1511_151192

-- Define the number of seedlings planted on the first day
def seedlings_day_1 : ℕ := 200

-- Define the number of seedlings planted on the second day
def seedlings_day_2 : ℕ := 2 * seedlings_day_1

-- Define the total number of seedlings planted on both days
def total_seedlings : ℕ := seedlings_day_1 + seedlings_day_2

-- The theorem statement
theorem total_seedlings_transferred : total_seedlings = 600 := by
  -- The proof goes here
  sorry

end total_seedlings_transferred_l1511_151192


namespace factorize1_factorize2_factorize3_l1511_151156

theorem factorize1 (x : ℝ) : x^3 + 6 * x^2 + 9 * x = x * (x + 3)^2 := 
  sorry

theorem factorize2 (x y : ℝ) : 16 * x^2 - 9 * y^2 = (4 * x - 3 * y) * (4 * x + 3 * y) := 
  sorry

theorem factorize3 (x y : ℝ) : (3 * x + y)^2 - (x - 3 * y) * (3 * x + y) = 2 * (3 * x + y) * (x + 2 * y) := 
  sorry

end factorize1_factorize2_factorize3_l1511_151156


namespace intersection_point_l1511_151143

theorem intersection_point :
  (∃ (x y : ℝ), 5 * x - 3 * y = 15 ∧ 4 * x + 2 * y = 14)
  → (∃ (x y : ℝ), x = 3 ∧ y = 1) :=
by
  intro h
  sorry

end intersection_point_l1511_151143


namespace reeya_average_score_l1511_151167

theorem reeya_average_score :
  let scores := [50, 60, 70, 80, 80]
  let sum_scores := scores.sum
  let num_scores := scores.length
  sum_scores / num_scores = 68 :=
by
  sorry

end reeya_average_score_l1511_151167


namespace find_tangent_points_l1511_151132

-- Step a: Define the curve and the condition for the tangent line parallel to y = 4x.
def curve (x : ℝ) : ℝ := x^3 + x - 2
def tangent_slope : ℝ := 4

-- Step d: Provide the statement that the coordinates of P₀ are (1, 0) and (-1, -4).
theorem find_tangent_points : 
  ∃ (P₀ : ℝ × ℝ), (curve P₀.1 = P₀.2) ∧ 
                 ((P₀ = (1, 0)) ∨ (P₀ = (-1, -4))) := 
by
  sorry

end find_tangent_points_l1511_151132


namespace boxes_containing_neither_l1511_151159

theorem boxes_containing_neither
  (total_boxes : ℕ := 15)
  (boxes_with_markers : ℕ := 9)
  (boxes_with_crayons : ℕ := 5)
  (boxes_with_both : ℕ := 4) :
  (total_boxes - ((boxes_with_markers - boxes_with_both) + (boxes_with_crayons - boxes_with_both) + boxes_with_both)) = 5 := by
  sorry

end boxes_containing_neither_l1511_151159


namespace range_of_a_l1511_151168

theorem range_of_a (x y a : ℝ) (h1 : 3 * x + y = a + 1) (h2 : x + 3 * y = 3) (h3 : x + y > 5) : a > 16 := 
sorry 

end range_of_a_l1511_151168


namespace min_abs_phi_l1511_151191

open Real

theorem min_abs_phi {k : ℤ} :
  ∃ (φ : ℝ), ∀ (k : ℤ), φ = - (5 * π) / 6 + k * π ∧ |φ| = π / 6 := sorry

end min_abs_phi_l1511_151191


namespace red_car_speed_l1511_151166

/-- Dale owns 4 sports cars where:
1. The red car can travel at twice the speed of the green car.
2. The green car can travel at 8 times the speed of the blue car.
3. The blue car can travel at a speed of 80 miles per hour.
We need to determine the speed of the red car. --/
theorem red_car_speed (r g b: ℕ) (h1: r = 2 * g) (h2: g = 8 * b) (h3: b = 80) : 
  r = 1280 :=
by
  sorry

end red_car_speed_l1511_151166


namespace snow_on_second_day_l1511_151185

-- Definition of conditions as variables in Lean
def snow_on_first_day := 6 -- in inches
def snow_melted := 2 -- in inches
def additional_snow_fifth_day := 12 -- in inches
def total_snow := 24 -- in inches

-- The variable for snow on the second day
variable (x : ℕ)

-- Proof goal
theorem snow_on_second_day : snow_on_first_day + x - snow_melted + additional_snow_fifth_day = total_snow → x = 8 :=
by
  intros h
  sorry

end snow_on_second_day_l1511_151185


namespace smallest_x_solution_l1511_151157

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l1511_151157


namespace arithmetic_sequence_ratios_l1511_151111

noncomputable def a_n : ℕ → ℚ := sorry -- definition of the arithmetic sequence {a_n}
noncomputable def b_n : ℕ → ℚ := sorry -- definition of the arithmetic sequence {b_n}
noncomputable def S_n (n : ℕ) : ℚ := sorry -- definition of the sum of the first n terms of {a_n}
noncomputable def T_n (n : ℕ) : ℚ := sorry -- definition of the sum of the first n terms of {b_n}

theorem arithmetic_sequence_ratios :
  (∀ n : ℕ, 0 < n → S_n n / T_n n = (7 * n + 1) / (4 * n + 27)) →
  (a_n 7 / b_n 7 = 92 / 79) :=
by
  intros h
  sorry

end arithmetic_sequence_ratios_l1511_151111


namespace triangle_is_isosceles_l1511_151104

/-- Given triangle ABC with angles A, B, and C, where C = π - (A + B),
    if 2 * sin A * cos B = sin C, then triangle ABC is an isosceles triangle -/
theorem triangle_is_isosceles
  (A B C : ℝ)
  (hC : C = π - (A + B))
  (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  A = B :=
by
  sorry

end triangle_is_isosceles_l1511_151104


namespace smallest_number_of_ten_consecutive_natural_numbers_l1511_151130

theorem smallest_number_of_ten_consecutive_natural_numbers 
  (x : ℕ) 
  (h : 6 * x + 39 = 2 * (4 * x + 6) + 15) : 
  x = 6 := 
by 
  sorry

end smallest_number_of_ten_consecutive_natural_numbers_l1511_151130


namespace relationship_between_x_and_z_l1511_151142

-- Definitions of the given conditions
variable {x y z : ℝ}

-- Statement of the theorem
theorem relationship_between_x_and_z (h1 : x = 1.027 * y) (h2 : y = 0.45 * z) : x = 0.46215 * z :=
by
  sorry

end relationship_between_x_and_z_l1511_151142


namespace number_of_schools_is_23_l1511_151190

-- Conditions and definitions
noncomputable def number_of_students_per_school : ℕ := 3
def beth_rank : ℕ := 37
def carla_rank : ℕ := 64

-- Statement of the proof problem
theorem number_of_schools_is_23
  (n : ℕ)
  (h1 : ∀ i < n, ∃ r1 r2 r3: ℕ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h2 : ∀ i < n, ∃ A B C: ℕ, A = (2 * B + 1) ∧ C = A ∧ B = 35 ∧ A < beth_rank ∧ beth_rank < carla_rank):
  n = 23 :=
by
  sorry

end number_of_schools_is_23_l1511_151190


namespace car_distance_ratio_l1511_151148

theorem car_distance_ratio (t : ℝ) (h₁ : t > 0)
    (speed_A speed_B : ℝ)
    (h₂ : speed_A = 70)
    (h₃ : speed_B = 35)
    (ratio : ℝ)
    (h₄ : ratio = 2)
    (h_time : ∀ a b : ℝ, a * t = b * t → a = b) :
  (speed_A * t) / (speed_B * t) = ratio := by
  sorry

end car_distance_ratio_l1511_151148


namespace matrix_vector_combination_l1511_151174

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (M : α →ₗ[ℝ] ℝ × ℝ)
variables (u v w : α)
variables (h1 : M u = (-3, 4))
variables (h2 : M v = (2, -7))
variables (h3 : M w = (9, 0))

theorem matrix_vector_combination :
  M (3 • u - 4 • v + 2 • w) = (1, 40) :=
by sorry

end matrix_vector_combination_l1511_151174


namespace sum_of_digits_625_base5_l1511_151116

def sum_of_digits_base_5 (n : ℕ) : ℕ :=
  let rec sum_digits n :=
    if n = 0 then 0
    else (n % 5) + sum_digits (n / 5)
  sum_digits n

theorem sum_of_digits_625_base5 : sum_of_digits_base_5 625 = 5 := by
  sorry

end sum_of_digits_625_base5_l1511_151116


namespace ratio_of_sheep_to_horses_l1511_151107

theorem ratio_of_sheep_to_horses (H : ℕ) (hH : 230 * H = 12880) (n_sheep : ℕ) (h_sheep : n_sheep = 56) :
  (n_sheep / H) = 1 := by
  sorry

end ratio_of_sheep_to_horses_l1511_151107


namespace log_addition_closed_l1511_151196

def is_log_of_nat (n : ℝ) : Prop := ∃ k : ℕ, k > 0 ∧ n = Real.log k

theorem log_addition_closed (a b : ℝ) (ha : is_log_of_nat a) (hb : is_log_of_nat b) : is_log_of_nat (a + b) :=
by
  sorry

end log_addition_closed_l1511_151196


namespace target_run_correct_l1511_151152

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

end target_run_correct_l1511_151152


namespace contrapositive_correct_l1511_151175

-- Conditions and the proposition
def prop1 (a : ℝ) : Prop := a = -1 → a^2 = 1

-- The contrapositive of the proposition
def contrapositive (a : ℝ) : Prop := a^2 ≠ 1 → a ≠ -1

-- The proof problem statement
theorem contrapositive_correct (a : ℝ) : prop1 a ↔ contrapositive a :=
by sorry

end contrapositive_correct_l1511_151175


namespace saree_blue_stripes_l1511_151112

theorem saree_blue_stripes (brown_stripes gold_stripes blue_stripes : ℕ) 
    (h1 : brown_stripes = 4)
    (h2 : gold_stripes = 3 * brown_stripes)
    (h3 : blue_stripes = 5 * gold_stripes) : 
    blue_stripes = 60 := 
by
  sorry

end saree_blue_stripes_l1511_151112


namespace sphere_has_circular_views_l1511_151176

-- Define the geometric shapes
inductive Shape
| cuboid
| cylinder
| cone
| sphere

-- Define a function that describes the views of a shape
def views (s: Shape) : (String × String × String) :=
match s with
| Shape.cuboid   => ("Rectangle", "Rectangle", "Rectangle")
| Shape.cylinder => ("Rectangle", "Rectangle", "Circle")
| Shape.cone     => ("Isosceles Triangle", "Isosceles Triangle", "Circle")
| Shape.sphere   => ("Circle", "Circle", "Circle")

-- Define the property of having circular views in all perspectives
def has_circular_views (s: Shape) : Prop :=
views s = ("Circle", "Circle", "Circle")

-- The theorem to prove
theorem sphere_has_circular_views :
  ∀ (s : Shape), has_circular_views s ↔ s = Shape.sphere :=
by sorry

end sphere_has_circular_views_l1511_151176


namespace problem_l1511_151120

-- Define i as the imaginary unit
def i : ℂ := Complex.I

-- The statement to be proved
theorem problem : i * (1 - i) ^ 2 = 2 := by
  sorry

end problem_l1511_151120


namespace percentage_increase_l1511_151144

variable (S : ℝ) (P : ℝ)
variable (h1 : S + 0.10 * S = 330)
variable (h2 : S + P * S = 324)

theorem percentage_increase : P = 0.08 := sorry

end percentage_increase_l1511_151144


namespace older_brother_catches_up_l1511_151147

theorem older_brother_catches_up (D : ℝ) (t : ℝ) :
  let vy := D / 25
  let vo := D / 15
  let time := 20
  15 * time = 25 * (time - 8) → (15 * time = 25 * (time - 8) → t = 20)
:= by
  sorry

end older_brother_catches_up_l1511_151147


namespace complete_square_to_d_l1511_151171

-- Conditions given in the problem
def quadratic_eq (x : ℝ) : Prop := x^2 + 10 * x + 7 = 0

-- Equivalent Lean 4 statement of the problem
theorem complete_square_to_d (x : ℝ) (c d : ℝ) (h : quadratic_eq x) (hc : c = 5) : (x + c)^2 = d → d = 18 :=
by sorry

end complete_square_to_d_l1511_151171


namespace radius_of_circumscribed_circle_of_right_triangle_l1511_151165

theorem radius_of_circumscribed_circle_of_right_triangle 
  (a b c : ℝ)
  (h_area : (1 / 2) * a * b = 10)
  (h_inradius : (a + b - c) / 2 = 1)
  (h_hypotenuse : c = Real.sqrt (a^2 + b^2)) :
  c / 2 = 4.5 := 
sorry

end radius_of_circumscribed_circle_of_right_triangle_l1511_151165


namespace trapezoid_perimeter_is_183_l1511_151108

-- Declare the lengths of the sides of the trapezoid
def EG : ℕ := 35
def FH : ℕ := 40
def GH : ℕ := 36

-- Declare the relation between the bases EF and GH
def EF : ℕ := 2 * GH

-- The statement of the problem
theorem trapezoid_perimeter_is_183 : EF = 72 ∧ (EG + GH + FH + EF) = 183 := by
  sorry

end trapezoid_perimeter_is_183_l1511_151108


namespace product_of_roots_l1511_151106

theorem product_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 :=
sorry

end product_of_roots_l1511_151106


namespace area_of_rectangle_l1511_151122

theorem area_of_rectangle (w d : ℝ) (h_w : w = 4) (h_d : d = 5) : ∃ l : ℝ, (w^2 + l^2 = d^2) ∧ (w * l = 12) :=
by
  sorry

end area_of_rectangle_l1511_151122


namespace prove_b_eq_d_and_c_eq_e_l1511_151109

variable (a b c d e f : ℕ)

-- Define the expressions for A and B as per the problem statement
def A := 10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f
def B := 10^5 * f + 10^4 * d + 10^3 * e + 10^2 * b + 10 * c + a

-- Define the condition that A - B is divisible by 271
def divisible_by_271 (n : ℕ) : Prop := ∃ k : ℕ, n = 271 * k

-- Define the main theorem to prove b = d and c = e under the given conditions
theorem prove_b_eq_d_and_c_eq_e
    (h1 : divisible_by_271 (A a b c d e f - B a b c d e f)) :
    b = d ∧ c = e :=
sorry

end prove_b_eq_d_and_c_eq_e_l1511_151109


namespace B_can_complete_alone_l1511_151161

-- Define the given conditions
def A_work_rate := 1 / 20
def total_days := 21
def A_quit_days := 15
def B_completion_days := 30

-- Define the problem statement in Lean
theorem B_can_complete_alone (x : ℝ) (h₁ : A_work_rate = 1 / 20) (h₂ : total_days = 21)
  (h₃ : A_quit_days = 15) (h₄ : (21 - A_quit_days) * (1 / 20 + 1 / x) + A_quit_days * (1 / x) = 1) :
  x = B_completion_days :=
  sorry

end B_can_complete_alone_l1511_151161


namespace chess_competition_l1511_151102

theorem chess_competition (W M : ℕ) 
  (hW : W * (W - 1) / 2 = 45) 
  (hM : M * 10 = 200) :
  M * (M - 1) / 2 = 190 :=
by
  sorry

end chess_competition_l1511_151102


namespace xiao_li_more_stable_l1511_151188

def average_xiao_li : ℝ := 95
def average_xiao_zhang : ℝ := 95

def variance_xiao_li : ℝ := 0.55
def variance_xiao_zhang : ℝ := 1.35

theorem xiao_li_more_stable : 
  variance_xiao_li < variance_xiao_zhang :=
by
  sorry

end xiao_li_more_stable_l1511_151188


namespace range_of_a_l1511_151162

theorem range_of_a (x a : ℝ) :
  (∀ x : ℝ, x - 1 < 0 ∧ x < a + 3 → x < 1) → a ≥ -2 :=
by
  sorry

end range_of_a_l1511_151162


namespace triangle_height_l1511_151124

theorem triangle_height (area base : ℝ) (h : ℝ) (h_area : area = 46) (h_base : base = 10) 
  (h_formula : area = (base * h) / 2) : 
  h = 9.2 :=
by
  sorry

end triangle_height_l1511_151124


namespace expression_non_negative_l1511_151135

theorem expression_non_negative (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (4 / (c - a)) ≥ 0 :=
by
  sorry

end expression_non_negative_l1511_151135


namespace sufficient_but_not_necessary_condition_l1511_151100

def M : Set ℝ := {x | 0 < x ∧ x ≤ 2}

def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by
  sorry

end sufficient_but_not_necessary_condition_l1511_151100


namespace area_of_square_not_covered_by_circles_l1511_151184

theorem area_of_square_not_covered_by_circles :
  let side : ℝ := 10
  let radius : ℝ := 5
  (side^2 - 4 * (π * radius^2) + 4 * (π * (radius^2) / 2)) = (100 - 50 * π) := 
sorry

end area_of_square_not_covered_by_circles_l1511_151184


namespace train_speed_correct_l1511_151134

def train_length : ℝ := 250  -- length of the train in meters
def time_to_pass : ℝ := 18  -- time to pass a tree in seconds
def speed_of_train_km_hr : ℝ := 50  -- speed of the train in km/hr

theorem train_speed_correct :
  (train_length / time_to_pass) * (3600 / 1000) = speed_of_train_km_hr :=
by
  sorry

end train_speed_correct_l1511_151134


namespace arithmetic_mean_of_fractions_l1511_151199

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 5
  let b := (5 : ℚ) / 7
  (a + b) / 2 = (23 : ℚ) / 35 := 
by 
  sorry 

end arithmetic_mean_of_fractions_l1511_151199


namespace valid_choice_count_l1511_151182

def is_valid_base_7_digit (n : ℕ) : Prop := n < 7
def is_valid_base_8_digit (n : ℕ) : Prop := n < 8
def to_base_10_base_7 (c3 c2 c1 c0 : ℕ) : ℕ := 2401 * c3 + 343 * c2 + 49 * c1 + 7 * c0
def to_base_10_base_8 (d3 d2 d1 d0 : ℕ) : ℕ := 4096 * d3 + 512 * d2 + 64 * d1 + 8 * d0
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem valid_choice_count :
  ∃ (N : ℕ), is_four_digit_number N →
  ∀ (c3 c2 c1 c0 d3 d2 d1 d0 : ℕ),
    is_valid_base_7_digit c3 → is_valid_base_7_digit c2 → is_valid_base_7_digit c1 → is_valid_base_7_digit c0 →
    is_valid_base_8_digit d3 → is_valid_base_8_digit d2 → is_valid_base_8_digit d1 → is_valid_base_8_digit d0 →
    to_base_10_base_7 c3 c2 c1 c0 = N →
    to_base_10_base_8 d3 d2 d1 d0 = N →
    (to_base_10_base_7 c3 c2 c1 c0 + to_base_10_base_8 d3 d2 d1 d0) % 1000 = (2 * N) % 1000 → N = 20 :=
sorry

end valid_choice_count_l1511_151182


namespace ram_account_balance_increase_l1511_151126

theorem ram_account_balance_increase 
  (initial_deposit : ℕ := 500)
  (first_year_balance : ℕ := 600)
  (second_year_percentage_increase : ℕ := 32)
  (second_year_balance : ℕ := initial_deposit + initial_deposit * second_year_percentage_increase / 100) 
  (second_year_increase : ℕ := second_year_balance - first_year_balance) 
  : (second_year_increase * 100 / first_year_balance) = 10 := 
sorry

end ram_account_balance_increase_l1511_151126


namespace island_perimeter_l1511_151178

-- Defining the properties of the island
def width : ℕ := 4
def length : ℕ := 7

-- The main theorem stating the condition to be proved
theorem island_perimeter : 2 * (length + width) = 22 := by
  sorry

end island_perimeter_l1511_151178


namespace orthocenter_ABC_l1511_151195

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def A : Point2D := ⟨5, -1⟩
def B : Point2D := ⟨4, -8⟩
def C : Point2D := ⟨-4, -4⟩

def isOrthocenter (H : Point2D) (A B C : Point2D) : Prop := sorry  -- Define this properly according to the geometric properties in actual formalization.

theorem orthocenter_ABC : ∃ H : Point2D, isOrthocenter H A B C ∧ H = ⟨3, -5⟩ := 
by 
  sorry  -- Proof omitted

end orthocenter_ABC_l1511_151195


namespace addition_of_decimals_l1511_151110

theorem addition_of_decimals : (0.3 + 0.03 : ℝ) = 0.33 := by
  sorry

end addition_of_decimals_l1511_151110


namespace girl_needs_120_oranges_l1511_151113

-- Define the cost and selling prices per pack
def cost_per_pack : ℤ := 15   -- cents
def oranges_per_pack_cost : ℤ := 4
def sell_per_pack : ℤ := 30   -- cents
def oranges_per_pack_sell : ℤ := 6

-- Define the target profit
def target_profit : ℤ := 150  -- cents

-- Calculate the cost price per orange
def cost_per_orange : ℚ := cost_per_pack / oranges_per_pack_cost

-- Calculate the selling price per orange
def sell_per_orange : ℚ := sell_per_pack / oranges_per_pack_sell

-- Calculate the profit per orange
def profit_per_orange : ℚ := sell_per_orange - cost_per_orange

-- Calculate the number of oranges needed to achieve the target profit
def oranges_needed : ℚ := target_profit / profit_per_orange

-- Lean theorem statement
theorem girl_needs_120_oranges :
  oranges_needed = 120 :=
  sorry

end girl_needs_120_oranges_l1511_151113


namespace James_comics_l1511_151145

theorem James_comics (days_in_year : ℕ) (years : ℕ) (writes_every_other_day : ℕ) (no_leap_years : ℕ) 
  (h1 : days_in_year = 365) (h2 : years = 4) (h3 : writes_every_other_day = 2) : 
  (days_in_year * years) / writes_every_other_day = 730 := 
by
  sorry

end James_comics_l1511_151145


namespace tens_digit_of_binary_result_l1511_151119

def digits_tens_digit_subtraction (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) : ℕ :=
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let difference := original_number - reversed_number
  (difference % 100) / 10

theorem tens_digit_of_binary_result (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) :
  digits_tens_digit_subtraction a b c h1 h2 = 9 :=
sorry

end tens_digit_of_binary_result_l1511_151119


namespace subset_exists_l1511_151154

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {x + 2, 1}

-- Statement of the theorem
theorem subset_exists (x : ℝ) : B 2 ⊆ A 2 :=
by
  sorry

end subset_exists_l1511_151154


namespace chinese_chess_sets_l1511_151121

theorem chinese_chess_sets (x y : ℕ) 
  (h1 : 24 * x + 18 * y = 300) 
  (h2 : x + y = 14) : 
  y = 6 := 
sorry

end chinese_chess_sets_l1511_151121


namespace inequality_proof_l1511_151179

theorem inequality_proof (a b : Real) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end inequality_proof_l1511_151179


namespace factorize_x_squared_minus_sixteen_l1511_151183

theorem factorize_x_squared_minus_sixteen (x : ℝ) : x^2 - 16 = (x + 4) * (x - 4) :=
by
  sorry

end factorize_x_squared_minus_sixteen_l1511_151183


namespace length_of_flat_terrain_l1511_151123

theorem length_of_flat_terrain (total_time : ℚ)
  (total_distance : ℕ)
  (speed_uphill speed_flat speed_downhill : ℚ)
  (distance_uphill distance_flat : ℕ) :
  total_time = 116 / 60 ∧
  total_distance = distance_uphill + distance_flat + (total_distance - distance_uphill - distance_flat) ∧
  speed_uphill = 4 ∧
  speed_flat = 5 ∧
  speed_downhill = 6 ∧
  distance_uphill ≥ 0 ∧
  distance_flat ≥ 0 ∧
  distance_uphill + distance_flat ≤ total_distance →
  distance_flat = 3 := 
by 
  sorry

end length_of_flat_terrain_l1511_151123


namespace eq_system_correct_l1511_151139

theorem eq_system_correct (x y : ℤ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
sorry

end eq_system_correct_l1511_151139


namespace complex_expression_identity_l1511_151128

noncomputable section

variable (x y : ℂ) 

theorem complex_expression_identity (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x^2 + x*y + y^2 = 0) : 
  (x / (x + y)) ^ 1990 + (y / (x + y)) ^ 1990 = -1 := 
by 
  sorry

end complex_expression_identity_l1511_151128


namespace geometric_series_second_term_l1511_151115

theorem geometric_series_second_term (a : ℝ) (r : ℝ) (sum : ℝ) 
  (h1 : r = 1/4) 
  (h2 : sum = 40) 
  (sum_formula : sum = a / (1 - r)) : a * r = 7.5 :=
by {
  -- Proof to be filled in later
  sorry
}

end geometric_series_second_term_l1511_151115


namespace pieces_of_fudge_l1511_151137

def pan_length : ℝ := 27.5
def pan_width : ℝ := 17.5
def pan_height : ℝ := 2.5
def cube_side : ℝ := 2.3

def volume (l w h : ℝ) : ℝ := l * w * h

def V_pan : ℝ := volume pan_length pan_width pan_height
def V_cube : ℝ := volume cube_side cube_side cube_side

theorem pieces_of_fudge : ⌊V_pan / V_cube⌋ = 98 := by
  -- calculation can be filled in here in the actual proof
  sorry

end pieces_of_fudge_l1511_151137


namespace div_eq_eight_fifths_l1511_151155

theorem div_eq_eight_fifths (a b : ℚ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 5) : a / b = 8 / 5 :=
by
  sorry

end div_eq_eight_fifths_l1511_151155


namespace not_minimum_on_l1511_151180

noncomputable def f (x m : ℝ) : ℝ :=
  x * Real.exp x - (m / 2) * x ^ 2 - m * x

theorem not_minimum_on (m : ℝ) : 
  ¬ (∃ x ∈ Set.Icc 1 2, f x m = Real.exp 2 - 2 * m ∧ 
  ∀ y ∈ Set.Icc 1 2, f y m ≥ f x m) :=
sorry

end not_minimum_on_l1511_151180


namespace binary_operation_result_l1511_151187

theorem binary_operation_result :
  let a := 0b1101
  let b := 0b111
  let c := 0b1010
  let d := 0b1001
  a + b - c + d = 0b10011 :=
by {
  sorry
}

end binary_operation_result_l1511_151187


namespace isosceles_trapezoid_area_l1511_151170

-- Defining the problem characteristics
variables {a b c d h θ : ℝ}

-- The area formula for an isosceles trapezoid with given bases and height
theorem isosceles_trapezoid_area (h : ℝ) (c d : ℝ) : 
  (1 / 2) * (c + d) * h = (1 / 2) * (c + d) * h := 
sorry

end isosceles_trapezoid_area_l1511_151170


namespace convert_point_cylindrical_to_rectangular_l1511_151141

noncomputable def cylindrical_to_rectangular_coordinates (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_point_cylindrical_to_rectangular :
  cylindrical_to_rectangular_coordinates 6 (5 * Real.pi / 3) (-3) = (3, -3 * Real.sqrt 3, -3) :=
by
  sorry

end convert_point_cylindrical_to_rectangular_l1511_151141


namespace Tile_in_rectangle_R_l1511_151125

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def X : Tile := ⟨5, 3, 6, 2⟩
def Y : Tile := ⟨3, 6, 2, 5⟩
def Z : Tile := ⟨6, 0, 1, 5⟩
def W : Tile := ⟨2, 5, 3, 0⟩

theorem Tile_in_rectangle_R : 
  X.top = 5 ∧ X.right = 3 ∧ X.bottom = 6 ∧ X.left = 2 ∧ 
  Y.top = 3 ∧ Y.right = 6 ∧ Y.bottom = 2 ∧ Y.left = 5 ∧ 
  Z.top = 6 ∧ Z.right = 0 ∧ Z.bottom = 1 ∧ Z.left = 5 ∧ 
  W.top = 2 ∧ W.right = 5 ∧ W.bottom = 3 ∧ W.left = 0 → 
  (∀ rectangle_R : Tile, rectangle_R = W) :=
by sorry

end Tile_in_rectangle_R_l1511_151125


namespace percentage_increase_is_50_l1511_151163

-- Define the conditions
variables {P : ℝ} {x : ℝ}

-- Define the main statement (goal)
theorem percentage_increase_is_50 (h : 0.80 * P + (0.008 * x * P) = 1.20 * P) : x = 50 :=
sorry  -- Skip the proof as per instruction

end percentage_increase_is_50_l1511_151163


namespace range_of_a_l1511_151197

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) 
  (h2 : log_base a (a^2 + 1) < log_base a (2 * a))
  (h3 : log_base a (2 * a) < 0) : a ∈ Set.Ioo (0.5) 1 := 
sorry

end range_of_a_l1511_151197


namespace value_of_stocks_l1511_151117

def initial_investment (bonus : ℕ) (stocks : ℕ) : ℕ := bonus / stocks
def final_value_stock_A (initial : ℕ) : ℕ := initial * 2
def final_value_stock_B (initial : ℕ) : ℕ := initial * 2
def final_value_stock_C (initial : ℕ) : ℕ := initial / 2

theorem value_of_stocks 
    (bonus : ℕ) (stocks : ℕ) (h_bonus : bonus = 900) (h_stocks : stocks = 3) : 
    initial_investment bonus stocks * 2 + initial_investment bonus stocks * 2 + initial_investment bonus stocks / 2 = 1350 :=
by
    sorry

end value_of_stocks_l1511_151117


namespace tan_triple_angle_l1511_151146

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l1511_151146


namespace find_a_l1511_151136

def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_a (a : ℝ) : 
  are_perpendicular a (a + 2) → 
  a = -1 :=
by
  intro h
  unfold are_perpendicular at h
  have h_eq : a * (a + 2) = -1 := h
  have eq_zero : a * a + 2 * a + 1 = 0 := by linarith
  sorry

end find_a_l1511_151136


namespace negative_three_degrees_below_zero_l1511_151118

-- Definitions based on conditions
def positive_temperature (t : ℤ) : Prop := t > 0
def negative_temperature (t : ℤ) : Prop := t < 0
def above_zero (t : ℤ) : Prop := positive_temperature t
def below_zero (t : ℤ) : Prop := negative_temperature t

-- Example given in conditions
def ten_degrees_above_zero := above_zero 10

-- Lean 4 statement for the proof
theorem negative_three_degrees_below_zero : below_zero (-3) :=
by
  sorry

end negative_three_degrees_below_zero_l1511_151118


namespace exponent_power_identity_l1511_151150

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l1511_151150


namespace remainder_sand_amount_l1511_151149

def total_sand : ℝ := 2548726
def bag_capacity : ℝ := 85741.2
def full_bags : ℝ := 29
def not_full_bag_sand : ℝ := 62231.2

theorem remainder_sand_amount :
  total_sand - (full_bags * bag_capacity) = not_full_bag_sand :=
by
  sorry

end remainder_sand_amount_l1511_151149


namespace average_sum_problem_l1511_151169

theorem average_sum_problem (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 10) : ∃ sum : ℝ, sum = avg * n ∧ sum = 53 :=
by
  sorry

end average_sum_problem_l1511_151169


namespace exponential_decreasing_l1511_151193

theorem exponential_decreasing (a : ℝ) (h : ∀ x y : ℝ, x < y → (a+1)^x > (a+1)^y) : -1 < a ∧ a < 0 :=
sorry

end exponential_decreasing_l1511_151193


namespace distinct_arith_prog_triangles_l1511_151129

theorem distinct_arith_prog_triangles (n : ℕ) (h10 : n % 10 = 0) : 
  (3 * n = 180 → ∃ d : ℕ, ∀ a b c, a = n - d ∧ b = n ∧ c = n + d 
  →  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 60) :=
by
  sorry

end distinct_arith_prog_triangles_l1511_151129


namespace seahawks_field_goals_l1511_151131

-- Defining the conditions as hypotheses
def final_score_seahawks : ℕ := 37
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3
def touchdowns_seahawks : ℕ := 4

-- Stating the goal to prove
theorem seahawks_field_goals : 
  (final_score_seahawks - touchdowns_seahawks * points_per_touchdown) / points_per_fieldgoal = 3 := 
by 
  sorry

end seahawks_field_goals_l1511_151131


namespace order_of_numbers_l1511_151189

variables (a b : ℚ)

theorem order_of_numbers (ha_pos : a > 0) (hb_neg : b < 0) (habs : |a| < |b|) :
  b < -a ∧ -a < a ∧ a < -b :=
by { sorry }

end order_of_numbers_l1511_151189


namespace pizza_topping_combinations_l1511_151127

theorem pizza_topping_combinations :
  (Nat.choose 7 3) = 35 :=
sorry

end pizza_topping_combinations_l1511_151127


namespace g_at_3_eq_19_l1511_151103

def g (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem g_at_3_eq_19 : g 3 = 19 := by
  sorry

end g_at_3_eq_19_l1511_151103


namespace jerry_stickers_l1511_151105

variable (G F J : ℕ)

theorem jerry_stickers (h1 : F = 18) (h2 : G = F - 6) (h3 : J = 3 * G) : J = 36 :=
by {
  sorry
}

end jerry_stickers_l1511_151105


namespace unique_solution_cond_l1511_151114

open Real

theorem unique_solution_cond (a c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + 2) ↔ c ≠ 4 :=
by sorry

end unique_solution_cond_l1511_151114


namespace abs_diff_roots_eq_3_l1511_151198

theorem abs_diff_roots_eq_3 : ∀ (r1 r2 : ℝ), (r1 ≠ r2) → (r1 + r2 = 7) → (r1 * r2 = 10) → |r1 - r2| = 3 :=
by
  intros r1 r2 hneq hsum hprod
  sorry

end abs_diff_roots_eq_3_l1511_151198


namespace non_shaded_perimeter_6_l1511_151186

theorem non_shaded_perimeter_6 
  (area_shaded : ℝ) (area_large_rect : ℝ) (area_extension : ℝ) (total_area : ℝ)
  (non_shaded_area : ℝ) (perimeter : ℝ) :
  area_shaded = 104 → 
  area_large_rect = 12 * 8 → 
  area_extension = 5 * 2 → 
  total_area = area_large_rect + area_extension → 
  non_shaded_area = total_area - area_shaded → 
  non_shaded_area = 2 → 
  perimeter = 2 * (2 + 1) → 
  perimeter = 6 := 
by 
  sorry

end non_shaded_perimeter_6_l1511_151186


namespace total_cost_proof_l1511_151177

def sandwich_cost : ℝ := 2.49
def soda_cost : ℝ := 1.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 12.46

theorem total_cost_proof : (num_sandwiches * sandwich_cost + num_sodas * soda_cost) = total_cost :=
by
  sorry

end total_cost_proof_l1511_151177


namespace square_side_length_l1511_151140

theorem square_side_length(area_sq_cm : ℕ) (h : area_sq_cm = 361) : ∃ side_length : ℕ, side_length ^ 2 = area_sq_cm ∧ side_length = 19 := 
by 
  use 19
  sorry

end square_side_length_l1511_151140


namespace mean_temperature_is_correct_l1511_151133

def temperatures : List ℤ := [-8, -6, -3, -3, 0, 4, -1]
def mean_temperature (temps : List ℤ) : ℚ := (temps.sum : ℚ) / temps.length

theorem mean_temperature_is_correct :
  mean_temperature temperatures = -17 / 7 :=
by
  sorry

end mean_temperature_is_correct_l1511_151133


namespace structure_of_S_l1511_151151

def set_S (x y : ℝ) : Prop :=
  (5 >= x + 1 ∧ 5 >= y - 5) ∨
  (x + 1 >= 5 ∧ x + 1 >= y - 5) ∨
  (y - 5 >= 5 ∧ y - 5 >= x + 1)

theorem structure_of_S :
  ∃ (a b c : ℝ), set_S x y ↔ (y <= x + 6) ∧ (x <= 4) ∧ (y <= 10) 
:= sorry

end structure_of_S_l1511_151151


namespace Rachel_father_age_when_Rachel_is_25_l1511_151164

-- Define the problem conditions:
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Prove the age of Rachel's father when she is 25 years old:
theorem Rachel_father_age_when_Rachel_is_25 : 
  Father_age + (25 - Rachel_age) = 60 := by
    sorry

end Rachel_father_age_when_Rachel_is_25_l1511_151164


namespace mary_total_zoom_time_l1511_151158

noncomputable def timeSpentDownloadingMac : ℝ := 10
noncomputable def timeSpentDownloadingWindows : ℝ := 3 * timeSpentDownloadingMac
noncomputable def audioGlitchesCount : ℝ := 2
noncomputable def audioGlitchDuration : ℝ := 4
noncomputable def totalAudioGlitchTime : ℝ := audioGlitchesCount * audioGlitchDuration
noncomputable def videoGlitchDuration : ℝ := 6
noncomputable def totalGlitchTime : ℝ := totalAudioGlitchTime + videoGlitchDuration
noncomputable def glitchFreeTalkingTime : ℝ := 2 * totalGlitchTime

theorem mary_total_zoom_time : 
  timeSpentDownloadingMac + timeSpentDownloadingWindows + totalGlitchTime + glitchFreeTalkingTime = 82 :=
by sorry

end mary_total_zoom_time_l1511_151158


namespace tim_sarah_age_ratio_l1511_151160

theorem tim_sarah_age_ratio :
  ∀ (x : ℕ), ∃ (t s : ℕ),
    t = 23 ∧ s = 11 ∧
    (23 + x) * 2 = (11 + x) * 3 → x = 13 :=
by
  sorry

end tim_sarah_age_ratio_l1511_151160


namespace sum_of_coefficients_l1511_151172

theorem sum_of_coefficients (s : ℕ → ℝ) (a b c : ℝ) : 
  s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 17 ∧ 
  (∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) → 
  a + b + c = 12 := 
by
  sorry

end sum_of_coefficients_l1511_151172


namespace books_read_l1511_151194

-- Given conditions
def chapters_per_book : ℕ := 17
def total_chapters_read : ℕ := 68

-- Statement to prove
theorem books_read : (total_chapters_read / chapters_per_book) = 4 := 
by sorry

end books_read_l1511_151194


namespace num_perfect_square_factors_l1511_151181

def prime_factors_9600 (n : ℕ) : Prop :=
  n = 9600

theorem num_perfect_square_factors (n : ℕ) (h : prime_factors_9600 n) : 
  let cond := h
  (n = 9600) → 9600 = 2^6 * 5^2 * 3^1 → (∃ factors_count: ℕ, factors_count = 8) := by 
  sorry

end num_perfect_square_factors_l1511_151181


namespace find_height_of_cuboid_l1511_151138

-- Define the cuboid structure and its surface area formula
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

def surface_area (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

-- Given conditions
def given_cuboid : Cuboid := { length := 12, width := 14, height := 7 }
def given_surface_area : ℝ := 700

-- The theorem to prove
theorem find_height_of_cuboid :
  surface_area given_cuboid = given_surface_area :=
by
  sorry

end find_height_of_cuboid_l1511_151138


namespace number_of_members_l1511_151173

theorem number_of_members (n h : ℕ) (h1 : n * n * h = 362525) : n = 5 :=
sorry

end number_of_members_l1511_151173


namespace yearly_feeding_cost_l1511_151153

-- Defining the conditions
def num_geckos := 3
def num_iguanas := 2
def num_snakes := 4

def cost_per_snake_per_month := 10
def cost_per_iguana_per_month := 5
def cost_per_gecko_per_month := 15

-- Statement of the proof problem
theorem yearly_feeding_cost : 
  (num_snakes * cost_per_snake_per_month + num_iguanas * cost_per_iguana_per_month + num_geckos * cost_per_gecko_per_month) * 12 = 1140 := 
  by 
    sorry

end yearly_feeding_cost_l1511_151153


namespace number_of_zero_points_l1511_151101

theorem number_of_zero_points (f : ℝ → ℝ) (h_odd : ∀ x, f x = -f (-x)) (h_period : ∀ x, f (x - π) = f (x + π)) :
  ∃ (points : Finset ℝ), (∀ x ∈ points, 0 ≤ x ∧ x ≤ 8 ∧ f x = 0) ∧ points.card = 7 :=
by
  sorry

end number_of_zero_points_l1511_151101
