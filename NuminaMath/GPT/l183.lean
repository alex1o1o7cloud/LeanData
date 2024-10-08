import Mathlib

namespace total_donation_correct_l183_183897

def carwash_earnings : ℝ := 100
def carwash_donation : ℝ := carwash_earnings * 0.90

def bakesale_earnings : ℝ := 80
def bakesale_donation : ℝ := bakesale_earnings * 0.75

def mowinglawns_earnings : ℝ := 50
def mowinglawns_donation : ℝ := mowinglawns_earnings * 1.00

def total_donation : ℝ := carwash_donation + bakesale_donation + mowinglawns_donation

theorem total_donation_correct : total_donation = 200 := by
  -- the proof will be written here
  sorry

end total_donation_correct_l183_183897


namespace circle_land_represents_30105_l183_183739

-- Definitions based on the problem's conditions
def circleLandNumber (digits : List (ℕ × ℕ)) : ℕ :=
  digits.foldl (λ acc (d_circle : ℕ × ℕ) => acc + d_circle.fst * 10^d_circle.snd) 0

-- Example 207
def number_207 : List (ℕ × ℕ) := [(2, 2), (0, 0), (7, 0)]

-- Example 4520
def number_4520 : List (ℕ × ℕ) := [(4, 3), (5, 1), (2, 0), (0, 0)]

-- The diagram to analyze
def given_diagram : List (ℕ × ℕ) := [(3, 4), (1, 2), (5, 0)]

-- The statement proving the given diagram represents 30105 in Circle Land
theorem circle_land_represents_30105 : circleLandNumber given_diagram = 30105 :=
  sorry

end circle_land_represents_30105_l183_183739


namespace find_x_in_terms_of_N_l183_183222

theorem find_x_in_terms_of_N (N : ℤ) (x y : ℝ) 
(h1 : (⌊x⌋ : ℤ) + 2 * y = N + 2) 
(h2 : (⌊y⌋ : ℤ) + 2 * x = 3 - N) : 
x = (3 / 2) - N := 
by
  sorry

end find_x_in_terms_of_N_l183_183222


namespace midpoint_of_line_segment_l183_183562

theorem midpoint_of_line_segment :
  let z1 := Complex.mk (-7) 5
  let z2 := Complex.mk 5 (-3)
  (z1 + z2) / 2 = Complex.mk (-1) 1 := by sorry

end midpoint_of_line_segment_l183_183562


namespace strictly_increasing_interval_l183_183347

noncomputable def f (x : ℝ) : ℝ := x - x * Real.log x

theorem strictly_increasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x - x * Real.log x → ∀ y : ℝ, (0 < y ∧ y < 1 ∧ y > x) → f y > f x :=
sorry

end strictly_increasing_interval_l183_183347


namespace area_shaded_quad_correct_l183_183829

-- Define the side lengths of the squares
def side_length_small : ℕ := 3
def side_length_middle : ℕ := 5
def side_length_large : ℕ := 7

-- Define the total base length
def total_base_length : ℕ := side_length_small + side_length_middle + side_length_large

-- The height of triangle T3, equal to the side length of the largest square
def height_T3 : ℕ := side_length_large

-- The height-to-base ratio for each triangle
def height_to_base_ratio : ℚ := height_T3 / total_base_length

-- The heights of T1 and T2
def height_T1 : ℚ := side_length_small * height_to_base_ratio
def height_T2 : ℚ := (side_length_small + side_length_middle) * height_to_base_ratio

-- The height of the trapezoid, which is the side length of the middle square
def trapezoid_height : ℕ := side_length_middle

-- The bases of the trapezoid
def base1 : ℚ := height_T1
def base2 : ℚ := height_T2

-- The area of the trapezoid formula
def area_shaded_quad : ℚ := (trapezoid_height * (base1 + base2)) / 2

-- Assertion that the area of the shaded quadrilateral is equal to 77/6
theorem area_shaded_quad_correct : area_shaded_quad = 77 / 6 := by sorry

end area_shaded_quad_correct_l183_183829


namespace total_female_students_l183_183657

def total_students : ℕ := 1600
def sample_size : ℕ := 200
def fewer_girls : ℕ := 10

theorem total_female_students (x : ℕ) (sampled_girls sampled_boys : ℕ) (h_total_sample : sampled_girls + sampled_boys = sample_size)
                             (h_fewer_girls : sampled_girls + fewer_girls = sampled_boys) :
  sampled_girls * 8 = 760 :=
by
  sorry

end total_female_students_l183_183657


namespace odd_function_of_power_l183_183823

noncomputable def f (a b x : ℝ) : ℝ := (a - 1) * x ^ b

theorem odd_function_of_power (a b : ℝ) (h : f a b a = 1/2) : 
  ∀ x : ℝ, f a b (-x) = -f a b x := 
by
  sorry

end odd_function_of_power_l183_183823


namespace problem_l183_183544

noncomputable def roots1 : Set ℝ := { α | α^2 - 2*α + 1 = 0 }
noncomputable def roots2 : Set ℝ := { γ | γ^2 - 3*γ + 1 = 0 }

theorem problem 
  (α β γ δ : ℝ) 
  (hαβ : α ∈ roots1 ∧ β ∈ roots1)
  (hγδ : γ ∈ roots2 ∧ δ ∈ roots2) : 
  (α - γ)^2 * (β - δ)^2 = 1 := 
sorry

end problem_l183_183544


namespace final_image_of_F_is_correct_l183_183809

-- Define the initial F position as a struct
structure Position where
  base : (ℝ × ℝ)
  stem : (ℝ × ℝ)

-- Function to rotate a point 90 degrees counterclockwise around the origin
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Function to reflect a point in the x-axis
def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to rotate a point by 180 degrees around the origin (half turn)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Define the initial state of F
def initialFPosition : Position := {
  base := (-1, 0),  -- Base along the negative x-axis
  stem := (0, -1)   -- Stem along the negative y-axis
}

-- Perform all transformations on the Position of F
def transformFPosition (pos : Position) : Position :=
  let afterRotation90 := Position.mk (rotate90 pos.base) (rotate90 pos.stem)
  let afterReflectionX := Position.mk (reflectX afterRotation90.base) (reflectX afterRotation90.stem)
  let finalPosition := Position.mk (rotate180 afterReflectionX.base) (rotate180 afterReflectionX.stem)
  finalPosition

-- Define the target final position we expect
def finalFPosition : Position := {
  base := (0, 1),   -- Base along the positive y-axis
  stem := (1, 0)    -- Stem along the positive x-axis
}

-- The theorem statement: After the transformations, the position of F
-- should match the final expected position
theorem final_image_of_F_is_correct :
  transformFPosition initialFPosition = finalFPosition := by
  sorry

end final_image_of_F_is_correct_l183_183809


namespace minimum_students_in_class_l183_183278

def min_number_of_students (b g : ℕ) : ℕ :=
  b + g

theorem minimum_students_in_class
  (b g : ℕ)
  (h1 : b = 2 * g / 3)
  (h2 : ∃ k : ℕ, g = 3 * k)
  (h3 : ∃ k : ℕ, 1 / 2 < (2 / 3) * g / b) :
  min_number_of_students b g = 5 :=
sorry

end minimum_students_in_class_l183_183278


namespace triangle_area_base_10_height_10_l183_183273

theorem triangle_area_base_10_height_10 :
  let base := 10
  let height := 10
  (base * height) / 2 = 50 := by
  sorry

end triangle_area_base_10_height_10_l183_183273


namespace dissection_impossible_l183_183356

theorem dissection_impossible :
  ∀ (n m : ℕ), n = 1000 → m = 2016 → ¬(∃ (k l : ℕ), k * (n * m) = 1 * 2015 + l * 3) :=
by
  intros n m hn hm
  sorry

end dissection_impossible_l183_183356


namespace find_a_plus_b_minus_c_l183_183505

theorem find_a_plus_b_minus_c (a b c : ℤ) (h1 : 3 * b = 5 * a) (h2 : 7 * a = 3 * c) (h3 : 3 * a + 2 * b - 4 * c = -9) : a + b - c = 1 :=
by
  sorry

end find_a_plus_b_minus_c_l183_183505


namespace exists_a_b_l183_183699

theorem exists_a_b (n : ℕ) (hn : 0 < n) : ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := by
  sorry

end exists_a_b_l183_183699


namespace complement_of_A_in_U_l183_183787

-- Conditions definitions
def U : Set ℕ := {x | x ≤ 5}
def A : Set ℕ := {x | 2 * x - 5 < 0}

-- Theorem stating the question and the correct answer
theorem complement_of_A_in_U :
  U \ A = {x | 3 ≤ x ∧ x ≤ 5} :=
by
  -- The proof will go here
  sorry

end complement_of_A_in_U_l183_183787


namespace region_midpoint_area_equilateral_triangle_52_36_l183_183667

noncomputable def equilateral_triangle (A B C: ℝ × ℝ) : Prop :=
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

def midpoint_region_area (a b c : ℝ × ℝ) : ℝ := sorry

theorem region_midpoint_area_equilateral_triangle_52_36 (A B C: ℝ × ℝ) (h: equilateral_triangle A B C) :
  let m := (midpoint_region_area A B C)
  100 * m = 52.36 :=
sorry

end region_midpoint_area_equilateral_triangle_52_36_l183_183667


namespace train_length_l183_183196

-- Defining the conditions
def speed_kmh : ℕ := 64
def speed_m_per_s : ℚ := (64 * 1000) / 3600 -- 64 km/h converted to m/s
def time_to_cross_seconds : ℕ := 9 

-- The theorem to prove the length of the train
theorem train_length : speed_m_per_s * time_to_cross_seconds = 160 := 
by 
  unfold speed_m_per_s 
  norm_num
  sorry -- Placeholder for actual proof

end train_length_l183_183196


namespace smallest_perfect_square_greater_l183_183717

theorem smallest_perfect_square_greater (a : ℕ) (h : ∃ n : ℕ, a = n^2) : 
  ∃ m : ℕ, m^2 > a ∧ ∀ k : ℕ, k^2 > a → m^2 ≤ k^2 :=
  sorry

end smallest_perfect_square_greater_l183_183717


namespace smallest_integer_y_l183_183348

theorem smallest_integer_y (y : ℤ) : (∃ (y : ℤ), (y / 4) + (3 / 7) > (4 / 7) ∧ ∀ (z : ℤ), z < y → (z / 4) + (3 / 7) ≤ (4 / 7)) := 
by
  sorry

end smallest_integer_y_l183_183348


namespace smallest_k_for_720_l183_183615

/-- Given a number 720, prove that the smallest positive integer k such that 720 * k is both a perfect square and a perfect cube is 1012500. -/
theorem smallest_k_for_720 (k : ℕ) : (∃ k > 0, 720 * k = (n : ℕ) ^ 6) -> k = 1012500 :=
by sorry

end smallest_k_for_720_l183_183615


namespace linear_func_3_5_l183_183209

def linear_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem linear_func_3_5 (f : ℝ → ℝ) (h_linear: linear_function f) 
  (h_diff: ∀ d : ℝ, f (d + 1) - f d = 3) : f 3 - f 5 = -6 :=
by
  sorry

end linear_func_3_5_l183_183209


namespace train_cross_tunnel_time_l183_183556

noncomputable def train_length : ℝ := 800 -- in meters
noncomputable def train_speed : ℝ := 78 * 1000 / 3600 -- converted to meters per second
noncomputable def tunnel_length : ℝ := 500 -- in meters
noncomputable def total_distance : ℝ := train_length + tunnel_length -- total distance to travel

theorem train_cross_tunnel_time : total_distance / train_speed / 60 = 1 := by
  sorry

end train_cross_tunnel_time_l183_183556


namespace train_length_correct_l183_183046

-- Define the conditions
def bridge_length : ℝ := 180
def train_speed : ℝ := 15
def time_to_cross_bridge : ℝ := 20
def time_to_cross_man : ℝ := 8

-- Define the length of the train
def length_of_train : ℝ := 120

-- Proof statement
theorem train_length_correct :
  (train_speed * time_to_cross_man = length_of_train) ∧
  (train_speed * time_to_cross_bridge = length_of_train + bridge_length) :=
by
  sorry

end train_length_correct_l183_183046


namespace addition_example_l183_183201

theorem addition_example : 36 + 15 = 51 := 
by
  sorry

end addition_example_l183_183201


namespace spring_spending_l183_183228

theorem spring_spending (end_of_feb : ℝ) (end_of_may : ℝ) (h_end_of_feb : end_of_feb = 0.8) (h_end_of_may : end_of_may = 2.5)
  : (end_of_may - end_of_feb) = 1.7 :=
by
  have spending_end_of_feb : end_of_feb = 0.8 := h_end_of_feb
  have spending_end_of_may : end_of_may = 2.5 := h_end_of_may
  sorry

end spring_spending_l183_183228


namespace min_shift_sine_l183_183817

theorem min_shift_sine (φ : ℝ) (hφ : φ > 0) :
    (∃ k : ℤ, 2 * φ + π / 3 = 2 * k * π) → φ = 5 * π / 6 :=
sorry

end min_shift_sine_l183_183817


namespace smallest_base_l183_183582

-- Definitions of the conditions
def condition1 (b : ℕ) : Prop := b > 3
def condition2 (b : ℕ) : Prop := b > 7
def condition3 (b : ℕ) : Prop := b > 6
def condition4 (b : ℕ) : Prop := b > 8

-- Main theorem statement
theorem smallest_base : ∀ b : ℕ, condition1 b ∧ condition2 b ∧ condition3 b ∧ condition4 b → b = 9 := by
  sorry

end smallest_base_l183_183582


namespace hall_length_width_difference_l183_183895

variable (L W : ℕ)

theorem hall_length_width_difference (h₁ : W = 1 / 2 * L) (h₂ : L * W = 800) :
  L - W = 20 :=
sorry

end hall_length_width_difference_l183_183895


namespace factor_expression_l183_183899

theorem factor_expression (y : ℝ) : 49 - 16*y^2 + 8*y = (7 - 4*y)*(7 + 4*y) := 
sorry

end factor_expression_l183_183899


namespace binom_21_10_l183_183313

theorem binom_21_10 :
  (Nat.choose 19 9 = 92378) →
  (Nat.choose 19 10 = 92378) →
  (Nat.choose 19 11 = 75582) →
  Nat.choose 21 10 = 352716 := by
  sorry

end binom_21_10_l183_183313


namespace sum_of_numbers_is_216_l183_183884

-- Define the conditions and what needs to be proved.
theorem sum_of_numbers_is_216 
  (x : ℕ) 
  (h_lcm : Nat.lcm (2 * x) (Nat.lcm (3 * x) (7 * x)) = 126) : 
  2 * x + 3 * x + 7 * x = 216 :=
by
  sorry

end sum_of_numbers_is_216_l183_183884


namespace arithmetic_geometric_value_l183_183254

-- Definitions and annotations
variables {a1 a2 b1 b2 : ℝ}
variable {d : ℝ} -- common difference for the arithmetic sequence
variable {q : ℝ} -- common ratio for the geometric sequence

-- Assuming input values for the initial elements of the sequences
axiom h1 : -9 = -9
axiom h2 : -9 + 3 * d = -1
axiom h3 : b1 = -9 * q
axiom h4 : b2 = -9 * q^2

-- The desired equality to prove
theorem arithmetic_geometric_value :
  b2 * (a2 - a1) = -8 :=
sorry

end arithmetic_geometric_value_l183_183254


namespace find_a_and_max_value_l183_183279

noncomputable def f (x a : ℝ) := 2 * x^3 - 6 * x^2 + a

theorem find_a_and_max_value :
  (∃ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≥ 0) ∧ (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → f x a ≤ 3)) :=
by
  sorry

end find_a_and_max_value_l183_183279


namespace negation_of_universal_proposition_l183_183213

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end negation_of_universal_proposition_l183_183213


namespace monotonicity_and_extreme_values_l183_183504

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem monotonicity_and_extreme_values :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (1 - x)) ∧
  (∀ x : ℝ, x > 1 → f x < f 1) ∧
  f 1 = -1 :=
by 
  sorry

end monotonicity_and_extreme_values_l183_183504


namespace rate_per_meter_for_fencing_l183_183952

/-- The length of a rectangular plot is 10 meters more than its width. 
    The cost of fencing the plot along its perimeter at a certain rate per meter is Rs. 1430. 
    The perimeter of the plot is 220 meters. 
    Prove that the rate per meter for fencing the plot is 6.5 Rs. 
 -/
theorem rate_per_meter_for_fencing (width length perimeter cost : ℝ)
  (h_length : length = width + 10)
  (h_perimeter : perimeter = 2 * (width + length))
  (h_perimeter_value : perimeter = 220)
  (h_cost : cost = 1430) :
  (cost / perimeter) = 6.5 := by
  sorry

end rate_per_meter_for_fencing_l183_183952


namespace journey_time_l183_183983

theorem journey_time
  (t_1 t_2 : ℝ)
  (h1 : t_1 + t_2 = 5)
  (h2 : 40 * t_1 + 60 * t_2 = 240) :
  t_1 = 3 :=
sorry

end journey_time_l183_183983


namespace promotional_pricing_plan_l183_183352

theorem promotional_pricing_plan (n : ℕ) : 
  (8 * 100 = 800) ∧ 
  (∀ n > 100, 6 * n < 640) :=
by
  sorry

end promotional_pricing_plan_l183_183352


namespace relationship_between_a_and_b_l183_183232

def a : ℤ := (-12) * (-23) * (-34) * (-45)
def b : ℤ := (-123) * (-234) * (-345)

theorem relationship_between_a_and_b : a > b := by
  sorry

end relationship_between_a_and_b_l183_183232


namespace share_of_A_eq_70_l183_183725

theorem share_of_A_eq_70 (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 595) : A = 70 :=
sorry

end share_of_A_eq_70_l183_183725


namespace income_expenditure_ratio_l183_183140

theorem income_expenditure_ratio
  (I : ℕ) (E : ℕ) (S : ℕ)
  (h1 : I = 18000)
  (h2 : S = 3600)
  (h3 : S = I - E) : I / E = 5 / 4 :=
by
  -- The actual proof is skipped.
  sorry

end income_expenditure_ratio_l183_183140


namespace number_x_is_divided_by_l183_183767

-- Define the conditions
variable (x y n : ℕ)
variable (cond1 : x = n * y + 4)
variable (cond2 : 2 * x = 8 * 3 * y + 3)
variable (cond3 : 13 * y - x = 1)

-- Define the statement to be proven
theorem number_x_is_divided_by : n = 11 :=
by
  sorry

end number_x_is_divided_by_l183_183767


namespace div_100_by_a8_3a4_minus_4_l183_183620

theorem div_100_by_a8_3a4_minus_4 (a : ℕ) (h : ¬ (5 ∣ a)) : 100 ∣ (a^8 + 3 * a^4 - 4) :=
sorry

end div_100_by_a8_3a4_minus_4_l183_183620


namespace stephanie_quarters_fraction_l183_183733

/-- Stephanie has a collection containing exactly one of the first 25 U.S. state quarters. 
    The quarters are in the order the states joined the union.
    Suppose 8 states joined the union between 1800 and 1809. -/
theorem stephanie_quarters_fraction :
  (8 / 25 : ℚ) = (8 / 25) :=
by
  sorry

end stephanie_quarters_fraction_l183_183733


namespace value_of_t_l183_183703

theorem value_of_t (k : ℤ) (t : ℤ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 68) : t = 20 :=
by
  sorry

end value_of_t_l183_183703


namespace correct_log_conclusions_l183_183115

variables {x₁ x₂ : ℝ} (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (h_diff : x₁ ≠ x₂)
noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem correct_log_conclusions :
  ¬ (f (x₁ + x₂) = f x₁ * f x₂) ∧
  (f (x₁ * x₂) = f x₁ + f x₂) ∧
  ¬ ((f x₁ - f x₂) / (x₁ - x₂) < 0) ∧
  (f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2) :=
by {
  sorry
}

end correct_log_conclusions_l183_183115


namespace margie_drive_distance_l183_183470

-- Conditions
def car_mpg : ℝ := 45  -- miles per gallon
def gas_price : ℝ := 5 -- dollars per gallon
def money_spent : ℝ := 25 -- dollars

-- Question: Prove that Margie can drive 225 miles with $25 worth of gas.
theorem margie_drive_distance (h1 : car_mpg = 45) (h2 : gas_price = 5) (h3 : money_spent = 25) :
  money_spent / gas_price * car_mpg = 225 := by
  sorry

end margie_drive_distance_l183_183470


namespace percentage_25_of_200_l183_183255

def percentage_of (percent : ℝ) (amount : ℝ) : ℝ := percent * amount

theorem percentage_25_of_200 :
  percentage_of 0.25 200 = 50 :=
by sorry

end percentage_25_of_200_l183_183255


namespace merchant_installed_zucchini_l183_183071

theorem merchant_installed_zucchini (Z : ℕ) : 
  (15 + Z + 8) / 2 = 18 → Z = 13 :=
by
 sorry

end merchant_installed_zucchini_l183_183071


namespace football_team_total_players_l183_183924

variable (P : ℕ)
variable (throwers : ℕ := 52)
variable (total_right_handed : ℕ := 64)
variable (remaining := P - throwers)
variable (left_handed := remaining / 3)
variable (right_handed_non_throwers := 2 * remaining / 3)

theorem football_team_total_players:
  right_handed_non_throwers + throwers = total_right_handed →
  P = 70 :=
by
  sorry

end football_team_total_players_l183_183924


namespace force_required_for_bolt_b_20_inch_l183_183905

noncomputable def force_inversely_proportional (F L : ℝ) : ℝ := F * L

theorem force_required_for_bolt_b_20_inch (F L : ℝ) :
  let handle_length_10 := 10
  let force_length_product_bolt_a := 3000
  let force_length_product_bolt_b := 4000
  let new_handle_length := 20
  (F * handle_length_10 = 400)
  ∧ (F * new_handle_length = 200)
  → force_inversely_proportional 400 10 = 4000
  ∧ force_inversely_proportional 200 20 = 4000
:=
by
  sorry

end force_required_for_bolt_b_20_inch_l183_183905


namespace decipher_proof_l183_183520

noncomputable def decipher_message (n : ℕ) (hidden_message : String) :=
  if n = 2211169691162 then hidden_message = "Kiss me, dearest" else false

theorem decipher_proof :
  decipher_message 2211169691162 "Kiss me, dearest" = true :=
by
  -- Proof skipped
  sorry

end decipher_proof_l183_183520


namespace circles_are_externally_tangent_l183_183670

-- Conditions given in the problem
def r1 (r2 : ℝ) : Prop := ∃ r1 : ℝ, r1 * r2 = 10 ∧ r1 + r2 = 7
def distance := 7

-- The positional relationship proof problem statement
theorem circles_are_externally_tangent (r1 r2 : ℝ) (h : r1 * r2 = 10 ∧ r1 + r2 = 7) (d : ℝ) (h_d : d = distance) : 
  d = r1 + r2 :=
sorry

end circles_are_externally_tangent_l183_183670


namespace addition_of_decimals_l183_183113

theorem addition_of_decimals (a b : ℚ) (h1 : a = 7.56) (h2 : b = 4.29) : a + b = 11.85 :=
by
  -- The proof will be provided here
  sorry

end addition_of_decimals_l183_183113


namespace terminal_side_angles_l183_183335

theorem terminal_side_angles (k : ℤ) (β : ℝ) :
  β = (Real.pi / 3) + 2 * k * Real.pi → -2 * Real.pi ≤ β ∧ β < 4 * Real.pi :=
by
  sorry

end terminal_side_angles_l183_183335


namespace smallest_integer_y_l183_183575

theorem smallest_integer_y (y : ℤ) (h : 3 - 5 * y < 23) : -3 ≥ y :=
by {
  sorry
}

end smallest_integer_y_l183_183575


namespace largest_square_side_l183_183821

theorem largest_square_side (width length : ℕ) (h_width : width = 63) (h_length : length = 42) : 
  Nat.gcd width length = 21 :=
by
  rw [h_width, h_length]
  sorry

end largest_square_side_l183_183821


namespace systematic_sampling_fourth_group_number_l183_183105

theorem systematic_sampling_fourth_group_number (n : ℕ) (step_size : ℕ) (first_number : ℕ) : 
  n = 4 → step_size = 6 → first_number = 4 → (first_number + step_size * 3) = 22 :=
by
  intros h_n h_step_size h_first_number
  sorry

end systematic_sampling_fourth_group_number_l183_183105


namespace plate_and_roller_acceleration_l183_183711

noncomputable def m : ℝ := 150
noncomputable def g : ℝ := 10
noncomputable def R : ℝ := 1
noncomputable def r : ℝ := 0.4
noncomputable def alpha : ℝ := Real.arccos 0.68

theorem plate_and_roller_acceleration :
  let sin_alpha_half := Real.sin (alpha / 2)
  sin_alpha_half = 0.4 →
  plate_acceleration == 4 ∧ direction == Real.arcsin 0.4 ∧ rollers_acceleration == 4 :=
by
  sorry

end plate_and_roller_acceleration_l183_183711


namespace problem_l183_183363

theorem problem (n : ℕ) (h : n ∣ (2^n - 2)) : (2^n - 1) ∣ (2^(2^n - 1) - 2) :=
by
  sorry

end problem_l183_183363


namespace correct_answer_l183_183297

-- Statement of the problem
theorem correct_answer :
  ∃ (answer : String),
    (answer = "long before" ∨ answer = "before long" ∨ answer = "soon after" ∨ answer = "shortly after") ∧
    answer = "long before" :=
by
  sorry

end correct_answer_l183_183297


namespace polynomial_value_l183_183433

theorem polynomial_value (x : ℝ) :
  let a := 2009 * x + 2008
  let b := 2009 * x + 2009
  let c := 2009 * x + 2010
  a^2 + b^2 + c^2 - a * b - b * c - a * c = 3 := by
  sorry

end polynomial_value_l183_183433


namespace gcd_72_and_120_l183_183542

theorem gcd_72_and_120 : Nat.gcd 72 120 = 24 := 
by
  sorry

end gcd_72_and_120_l183_183542


namespace present_age_of_son_l183_183132

theorem present_age_of_son (F S : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 := by
  sorry

end present_age_of_son_l183_183132


namespace no_natural_solution_l183_183042

theorem no_natural_solution :
  ¬ (∃ (x y : ℕ), 2 * x + 3 * y = 6) :=
by
sorry

end no_natural_solution_l183_183042


namespace expectation_of_transformed_binomial_l183_183289

def binomial_expectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

def linear_property_of_expectation (a b : ℚ) (E_ξ : ℚ) : ℚ :=
  a * E_ξ + b

theorem expectation_of_transformed_binomial (ξ : ℚ) :
  ξ = binomial_expectation 5 (2/5) →
  linear_property_of_expectation 5 2 ξ = 12 :=
by
  intros h
  rw [h]
  unfold linear_property_of_expectation binomial_expectation
  sorry

end expectation_of_transformed_binomial_l183_183289


namespace sum_of_three_integers_l183_183906

theorem sum_of_three_integers :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = 125 ∧ a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l183_183906


namespace value_of_expression_l183_183423

-- defining the conditions
def in_interval (a : ℝ) : Prop := 1 < a ∧ a < 2

-- defining the algebraic expression
def algebraic_expression (a : ℝ) : ℝ := abs (a - 2) + abs (1 - a)

-- theorem to be proved
theorem value_of_expression (a : ℝ) (h : in_interval a) : algebraic_expression a = 1 :=
by
  -- proof will go here
  sorry

end value_of_expression_l183_183423


namespace square_pyramid_sum_l183_183686

def square_pyramid_faces : Nat := 5
def square_pyramid_edges : Nat := 8
def square_pyramid_vertices : Nat := 5

theorem square_pyramid_sum : square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18 := by
  sorry

end square_pyramid_sum_l183_183686


namespace total_stocks_l183_183979

-- Define the conditions as given in the math problem
def closed_higher : ℕ := 1080
def ratio : ℝ := 1.20

-- Using ℕ for the number of stocks that closed lower
def closed_lower (x : ℕ) : Prop := 1080 = x * ratio ∧ closed_higher = x + x * (1 / 5)

-- Definition to compute the total number of stocks on the stock exchange
def total_number_of_stocks (x : ℕ) : ℕ := closed_higher + x

-- The main theorem to be proved
theorem total_stocks (x : ℕ) (h : closed_lower x) : total_number_of_stocks x = 1980 :=
sorry

end total_stocks_l183_183979


namespace total_number_of_cantelopes_l183_183589

def number_of_cantelopes_fred : ℕ := 38
def number_of_cantelopes_tim : ℕ := 44

theorem total_number_of_cantelopes : number_of_cantelopes_fred + number_of_cantelopes_tim = 82 := by
  sorry

end total_number_of_cantelopes_l183_183589


namespace point_P_in_first_quadrant_l183_183736

def point_P := (3, 2)
def first_quadrant (p : ℕ × ℕ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_P_in_first_quadrant : first_quadrant point_P :=
by
  sorry

end point_P_in_first_quadrant_l183_183736


namespace total_investment_sum_l183_183136

theorem total_investment_sum :
  let R : ℝ := 2200
  let T : ℝ := R - 0.1 * R
  let V : ℝ := T + 0.1 * T
  R + T + V = 6358 := by
  sorry

end total_investment_sum_l183_183136


namespace points_earned_l183_183606

-- Definitions of the types of enemies and their point values
def points_A := 10
def points_B := 15
def points_C := 20

-- Number of each type of enemies in the level
def num_A_total := 3
def num_B_total := 2
def num_C_total := 3

-- Number of each type of enemies defeated
def num_A_defeated := num_A_total -- 3 Type A enemies
def num_B_defeated := 1 -- Half of 2 Type B enemies
def num_C_defeated := 1 -- 1 Type C enemy

-- Calculation of total points earned
def total_points : ℕ :=
  num_A_defeated * points_A + num_B_defeated * points_B + num_C_defeated * points_C

-- Proof that the total points earned is 65
theorem points_earned : total_points = 65 := by
  -- Placeholder for the proof, which calculates the total points
  sorry

end points_earned_l183_183606


namespace xy_value_l183_183923

theorem xy_value (x y : ℝ) (h : (|x| - 1)^2 + (2 * y + 1)^2 = 0) : xy = 1/2 ∨ xy = -1/2 :=
by {
  sorry
}

end xy_value_l183_183923


namespace max_area_garden_l183_183781

/-- Given a rectangular garden with a total perimeter of 480 feet and one side twice as long as another,
    prove that the maximum area of the garden is 12800 square feet. -/
theorem max_area_garden (l w : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 480) : l * w = 12800 := 
sorry

end max_area_garden_l183_183781


namespace necessary_but_not_sufficient_for_lt_l183_183304

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_lt (h : a < b + 1) : a < b := 
sorry

end necessary_but_not_sufficient_for_lt_l183_183304


namespace national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l183_183467

-- Question 5
theorem national_currency_depreciation (term : String) : term = "Devaluation" := 
sorry

-- Question 6
theorem bond_annual_coupon_income 
  (purchase_price face_value annual_yield annual_coupon : ℝ) 
  (h_price : purchase_price = 900)
  (h_face : face_value = 1000)
  (h_yield : annual_yield = 0.15) 
  (h_coupon : annual_coupon = 135) : 
  annual_coupon = annual_yield * purchase_price := 
sorry

-- Question 7
theorem dividend_yield 
  (num_shares price_per_share total_dividends dividend_yield : ℝ)
  (h_shares : num_shares = 1000000)
  (h_price : price_per_share = 400)
  (h_dividends : total_dividends = 60000000)
  (h_yield : dividend_yield = 15) : 
  dividend_yield = (total_dividends / num_shares / price_per_share) * 100 :=
sorry

-- Question 8
theorem tax_deduction 
  (insurance_premium annual_salary tax_return : ℝ)
  (h_premium : insurance_premium = 120000)
  (h_salary : annual_salary = 110000)
  (h_return : tax_return = 14300) : 
  tax_return = 0.13 * min insurance_premium annual_salary := 
sorry

end national_currency_depreciation_bond_annual_coupon_income_dividend_yield_tax_deduction_l183_183467


namespace arithmetic_sequence_sums_l183_183927

variable (a : ℕ → ℕ)

-- Conditions
def condition1 := a 1 + a 4 + a 7 = 39
def condition2 := a 2 + a 5 + a 8 = 33

-- Question and expected answer
def result := a 3 + a 6 + a 9 = 27

theorem arithmetic_sequence_sums (h1 : condition1 a) (h2 : condition2 a) : result a := 
sorry

end arithmetic_sequence_sums_l183_183927


namespace linear_function_graph_not_in_second_quadrant_l183_183281

open Real

theorem linear_function_graph_not_in_second_quadrant 
  (k b : ℝ) (h1 : k > 0) (h2 : b < 0) :
  ¬ ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ y = k * x + b := 
sorry

end linear_function_graph_not_in_second_quadrant_l183_183281


namespace remainder_when_101_divided_by_7_is_3_l183_183327

theorem remainder_when_101_divided_by_7_is_3
    (A : ℤ)
    (h : 9 * A + 1 = 10 * A - 100) : A % 7 = 3 := by
  -- Mathematical steps are omitted as instructed
  sorry

end remainder_when_101_divided_by_7_is_3_l183_183327


namespace problem_l183_183479

def f (x : ℝ) : ℝ := sorry  -- f is a function from ℝ to ℝ

theorem problem (h : ∀ x : ℝ, 3 * f x + f (2 - x) = 4 * x^2 + 1) : f 5 = 133 / 4 := 
by 
  sorry -- the proof is omitted

end problem_l183_183479


namespace find_r_l183_183869

noncomputable def f (r a : ℝ) (x : ℝ) : ℝ := (x - r - 1) * (x - r - 8) * (x - a)
noncomputable def g (r b : ℝ) (x : ℝ) : ℝ := (x - r - 2) * (x - r - 9) * (x - b)

theorem find_r
  (r a b : ℝ)
  (h_condition1 : ∀ x, f r a x - g r b x = r)
  (h_condition2 : f r a (r + 2) = r)
  (h_condition3 : f r a (r + 9) = r)
  : r = -264 / 7 := sorry

end find_r_l183_183869


namespace area_of_union_of_triangle_and_reflection_l183_183949

-- Define points in ℝ²
structure Point where
  x : ℝ
  y : ℝ

-- Define the vertices of the original triangle
def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -1⟩
def C : Point := ⟨7, 0⟩

-- Define the vertices of the reflected triangle
def A' : Point := ⟨-2, 3⟩
def B' : Point := ⟨-4, -1⟩
def C' : Point := ⟨-7, 0⟩

-- Calculate the area of a triangle given three points
def triangleArea (P Q R : Point) : ℝ :=
  0.5 * |P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)|

-- Statement to prove: the area of the union of the original and reflected triangles
theorem area_of_union_of_triangle_and_reflection :
  triangleArea A B C + triangleArea A' B' C' = 14 := 
sorry

end area_of_union_of_triangle_and_reflection_l183_183949


namespace correct_product_l183_183051

theorem correct_product : 
  (0.0063 * 3.85 = 0.024255) :=
sorry

end correct_product_l183_183051


namespace sam_drove_200_miles_l183_183256

-- Define the conditions
def marguerite_distance : ℕ := 150
def marguerite_time : ℕ := 3
def sam_time : ℕ := 4
def marguerite_speed := marguerite_distance / marguerite_time

-- Define the question
def sam_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

-- State the theorem to prove the answer
theorem sam_drove_200_miles :
  sam_distance marguerite_speed sam_time = 200 := by
  sorry

end sam_drove_200_miles_l183_183256


namespace find_number_l183_183192

theorem find_number (x : ℤ) (h : x + x^2 = 342) : x = 18 ∨ x = -19 :=
sorry

end find_number_l183_183192


namespace cos_diff_alpha_beta_l183_183728

theorem cos_diff_alpha_beta (α β : ℝ) (h1 : Real.sin α = 2 / 3) (h2 : Real.cos β = -3 / 4)
    (h3 : α ∈ Set.Ioo (π / 2) π) (h4 : β ∈ Set.Ioo π (3 * π / 2)) :
    Real.cos (α - β) = (3 * Real.sqrt 5 - 2 * Real.sqrt 7) / 12 := 
sorry

end cos_diff_alpha_beta_l183_183728


namespace evaluate_expression_l183_183212

theorem evaluate_expression : 2^(3^2) + 3^(2^3) = 7073 := by
  sorry

end evaluate_expression_l183_183212


namespace flour_more_than_sugar_l183_183919

/-
  Mary is baking a cake. The recipe calls for 6 cups of sugar and 9 cups of flour. 
  She already put in 2 cups of flour. 
  Prove that the number of additional cups of flour Mary needs is 1 more than the number of additional cups of sugar she needs.
-/

theorem flour_more_than_sugar (s f a : ℕ) (h_s : s = 6) (h_f : f = 9) (h_a : a = 2) :
  (f - a) - s = 1 :=
by
  sorry

end flour_more_than_sugar_l183_183919


namespace imaginary_part_of_z_l183_183805

open Complex

theorem imaginary_part_of_z :
  ∃ z: ℂ, (3 - 4 * I) * z = abs (4 + 3 * I) ∧ z.im = 4 / 5 :=
by
  sorry

end imaginary_part_of_z_l183_183805


namespace John_can_finish_work_alone_in_48_days_l183_183223

noncomputable def John_and_Roger_can_finish_together_in_24_days (J R: ℝ) : Prop :=
  1 / J + 1 / R = 1 / 24

noncomputable def John_finished_remaining_work (J: ℝ) : Prop :=
  (1 / 3) / (16 / J) = 1

theorem John_can_finish_work_alone_in_48_days (J R: ℝ) 
  (h1 : John_and_Roger_can_finish_together_in_24_days J R) 
  (h2 : John_finished_remaining_work J):
  J = 48 := 
sorry

end John_can_finish_work_alone_in_48_days_l183_183223


namespace speed_ratio_l183_183881

variables (H D : ℝ)
variables (duck_leaps hen_leaps : ℕ)
-- hen_leaps and duck_leaps denote the leaps taken by hen and duck respectively

-- conditions given
axiom cond1 : hen_leaps = 6 ∧ duck_leaps = 8
axiom cond2 : 4 * D = 3 * H

-- goal to prove
theorem speed_ratio (H D : ℝ) (hen_leaps duck_leaps : ℕ) (cond1 : hen_leaps = 6 ∧ duck_leaps = 8) (cond2 : 4 * D = 3 * H) : 
  (6 * H) = (8 * D) :=
by
  intros
  sorry

end speed_ratio_l183_183881


namespace find_r_l183_183566

theorem find_r (r s : ℝ) (h_quadratic : ∀ y, y^2 - r * y - s = 0) (h_r_pos : r > 0) 
    (h_root_diff : ∀ (y₁ y₂ : ℝ), (y₁ = (r + Real.sqrt (r^2 + 4 * s)) / 2 
        ∧ y₂ = (r - Real.sqrt (r^2 + 4 * s)) / 2) → |y₁ - y₂| = 2) : r = 2 :=
sorry

end find_r_l183_183566


namespace disc_thickness_l183_183010

theorem disc_thickness (r_sphere : ℝ) (r_disc : ℝ) (h : ℝ)
  (h_radius_sphere : r_sphere = 3)
  (h_radius_disc : r_disc = 10)
  (h_volume_constant : (4/3) * Real.pi * r_sphere^3 = Real.pi * r_disc^2 * h) :
  h = 9 / 25 :=
by
  sorry

end disc_thickness_l183_183010


namespace correct_linear_regression_statement_l183_183516

-- Definitions based on the conditions:
def linear_regression (b a e : ℝ) (x : ℝ) : ℝ := b * x + a + e

def statement_A (b a e : ℝ) (x : ℝ) : Prop := linear_regression b a e x = b * x + a + e

def statement_B (b a e : ℝ) (x : ℝ) : Prop := ∀ x1 x2, (linear_regression b a e x1 ≠ linear_regression b a e x2) → (x1 ≠ x2)

def statement_C (b a e : ℝ) (x : ℝ) : Prop := ∃ (other_factors : ℝ), linear_regression b a e x = b * x + a + other_factors + e

def statement_D (b a e : ℝ) (x : ℝ) : Prop := (e ≠ 0) → false

-- The proof statement
theorem correct_linear_regression_statement (b a e : ℝ) (x : ℝ) :
  (statement_C b a e x) :=
sorry

end correct_linear_regression_statement_l183_183516


namespace unique_n0_exists_l183_183429

open Set

theorem unique_n0_exists 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n < a (n + 1)) 
  (h2 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h3 : ∀ n : ℕ, S 0 = a 0) :
  ∃! n_0 : ℕ, (S (n_0 + 1)) / n_0 > a (n_0 + 1)
             ∧ (S (n_0 + 1)) / n_0 ≤ a (n_0 + 2) := 
sorry

end unique_n0_exists_l183_183429


namespace total_selling_price_correct_l183_183249

def original_price : ℝ := 100
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.08

theorem total_selling_price_correct :
  let discount := original_price * discount_percent
  let sale_price := original_price - discount
  let tax := sale_price * tax_percent
  let total_selling_price := sale_price + tax
  total_selling_price = 75.6 := by
sorry

end total_selling_price_correct_l183_183249


namespace more_children_got_off_than_got_on_l183_183420

-- Define the initial number of children on the bus
def initial_children : ℕ := 36

-- Define the number of children who got off the bus
def children_got_off : ℕ := 68

-- Define the total number of children on the bus after changes
def final_children : ℕ := 12

-- Define the unknown number of children who got on the bus
def children_got_on : ℕ := sorry -- We will use the conditions to solve for this in the proof

-- The main proof statement
theorem more_children_got_off_than_got_on : (children_got_off - children_got_on = 24) :=
by
  -- Write the equation describing the total number of children after changes
  have h1 : initial_children - children_got_off + children_got_on = final_children := sorry
  -- Solve for the number of children who got on the bus (children_got_on)
  have h2 : children_got_on = final_children + (children_got_off - initial_children) := sorry
  -- Substitute to find the required difference
  have h3 : children_got_off - final_children - (children_got_off - initial_children) = 24 := sorry
  -- Conclude the proof
  exact sorry


end more_children_got_off_than_got_on_l183_183420


namespace find_xy_l183_183977

theorem find_xy (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : (x - y)^2 = 9) : x * y = 3 :=
sorry

end find_xy_l183_183977


namespace selling_price_when_profit_equals_loss_l183_183810

theorem selling_price_when_profit_equals_loss (CP SP Rs_57 : ℕ) (h1: CP = 50) (h2: Rs_57 = 57) (h3: Rs_57 - CP = CP - SP) : 
  SP = 43 := by
  sorry

end selling_price_when_profit_equals_loss_l183_183810


namespace problem1_problem2_l183_183204

-- Definition of sets A and B
def A : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def B (p : ℝ) : Set ℝ := { x | abs (x - p) > 1 }

-- Statement for the first problem
theorem problem1 : B 0 ∩ A = { x | 1 < x ∧ x < 3 } := 
by
  sorry

-- Statement for the second problem
theorem problem2 (p : ℝ) (h : A ∪ B p = B p) : p ≤ -2 ∨ p ≥ 4 := 
by
  sorry

end problem1_problem2_l183_183204


namespace complement_of_A_in_U_is_2_l183_183321

open Set

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }

theorem complement_of_A_in_U_is_2 : compl A ∩ U = {2} :=
by
  sorry

end complement_of_A_in_U_is_2_l183_183321


namespace isosceles_triangle_third_side_l183_183617

theorem isosceles_triangle_third_side (a b : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h₃ : a = b ∨ ∃ c, c = 9 ∧ (a = c ∨ b = c) ∧ (a + b > c ∧ a + c > b ∧ b + c > a)) :
  a = 9 ∨ b = 9 :=
by
  sorry

end isosceles_triangle_third_side_l183_183617


namespace merchant_should_choose_option2_l183_183745

-- Definitions for the initial price and discounts
def P : ℝ := 20000
def d1_1 : ℝ := 0.25
def d1_2 : ℝ := 0.15
def d1_3 : ℝ := 0.05
def d2_1 : ℝ := 0.35
def d2_2 : ℝ := 0.10
def d2_3 : ℝ := 0.05

-- Define the final prices after applying discount options
def finalPrice1 (P : ℝ) (d1_1 d1_2 d1_3 : ℝ) : ℝ :=
  P * (1 - d1_1) * (1 - d1_2) * (1 - d1_3)

def finalPrice2 (P : ℝ) (d2_1 d2_2 d2_3 : ℝ) : ℝ :=
  P * (1 - d2_1) * (1 - d2_2) * (1 - d2_3)

-- Theorem to state the merchant should choose Option 2
theorem merchant_should_choose_option2 : 
  finalPrice1 P d1_1 d1_2 d1_3 = 12112.50 ∧ 
  finalPrice2 P d2_1 d2_2 d2_3 = 11115 ∧ 
  finalPrice1 P d1_1 d1_2 d1_3 - finalPrice2 P d2_1 d2_2 d2_3 = 997.50 :=
by
  -- Placeholder for the proof
  sorry

end merchant_should_choose_option2_l183_183745


namespace floor_sqrt_20_squared_eq_16_l183_183569

theorem floor_sqrt_20_squared_eq_16 : (Int.floor (Real.sqrt 20))^2 = 16 := by
  sorry

end floor_sqrt_20_squared_eq_16_l183_183569


namespace trader_gain_percentage_l183_183194

theorem trader_gain_percentage (C : ℝ) (h1 : 95 * C = (95 * C - cost_of_95_pens) + (19 * C)) :
  100 * (19 * C / (95 * C)) = 20 := 
by {
  sorry
}

end trader_gain_percentage_l183_183194


namespace red_balls_count_l183_183579

-- Lean 4 statement for proving the number of red balls in the bag is 336
theorem red_balls_count (x : ℕ) (total_balls red_balls : ℕ) 
  (h1 : total_balls = 60 + 18 * x) 
  (h2 : red_balls = 56 + 14 * x) 
  (h3 : (56 + 14 * x : ℚ) / (60 + 18 * x) = 4 / 5) : red_balls = 336 := 
by
  sorry

end red_balls_count_l183_183579


namespace brick_weight_l183_183855

theorem brick_weight (b s : ℕ) (h1 : 5 * b = 4 * s) (h2 : 2 * s = 80) : b = 32 :=
by {
  sorry
}

end brick_weight_l183_183855


namespace solution_set_inequality_l183_183422

-- Definitions of the conditions
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2^x - 3 else - (2^(-x) - 3)

-- Statement to prove
theorem solution_set_inequality :
  is_odd_function f ∧ (∀ x > 0, f x = 2^x - 3)
  → {x : ℝ | f x ≤ -5} = {x : ℝ | x ≤ -3} := by
  sorry

end solution_set_inequality_l183_183422


namespace integral_2x_minus_1_eq_6_l183_183625

noncomputable def definite_integral_example : ℝ :=
  ∫ x in (0:ℝ)..(3:ℝ), (2 * x - 1)

theorem integral_2x_minus_1_eq_6 : definite_integral_example = 6 :=
by
  sorry

end integral_2x_minus_1_eq_6_l183_183625


namespace find_t_of_decreasing_function_l183_183476

theorem find_t_of_decreasing_function 
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_A : f 0 = 4)
  (h_B : f 3 = -2)
  (h_solution_set : ∀ x, |f (x + 1) - 1| < 3 ↔ -1 < x ∧ x < 2) :
  (1 : ℝ) = 1 :=
by
  sorry

end find_t_of_decreasing_function_l183_183476


namespace painted_cubes_l183_183744

/-- 
  Given a cube of side 9 painted red and cut into smaller cubes of side 3,
  prove the number of smaller cubes with paint on exactly 2 sides is 12.
-/
theorem painted_cubes (l : ℕ) (s : ℕ) (n : ℕ) (edges : ℕ) (faces : ℕ)
  (hcube_dimension : l = 9) (hsmaller_cubes_dimension : s = 3) 
  (hedges : edges = 12) (hfaces : faces * edges = 12) 
  (htotal_cubes : n = (l^3) / (s^3)) : 
  n * faces = 12 :=
sorry

end painted_cubes_l183_183744


namespace investor_profits_l183_183040

/-- Problem: Given the total contributions and profit sharing conditions, calculate the amount 
    each investor receives. -/

theorem investor_profits :
  ∀ (A_contribution B_contribution C_contribution D_contribution : ℝ) 
    (A_profit B_profit C_profit D_profit : ℝ) 
    (total_capital total_profit : ℝ),
    total_capital = 100000 → 
    A_contribution = B_contribution + 5000 →
    B_contribution = C_contribution + 10000 →
    C_contribution = D_contribution + 5000 →
    total_profit = 60000 →
    A_profit = (35 / 100) * total_profit * (1 + 10 / 100) →
    B_profit = (30 / 100) * total_profit * (1 + 8 / 100) →
    C_profit = (20 / 100) * total_profit * (1 + 5 / 100) → 
    D_profit = (15 / 100) * total_profit →
    (A_profit = 23100 ∧ B_profit = 19440 ∧ C_profit = 12600 ∧ D_profit = 9000) :=
by
  intros
  sorry

end investor_profits_l183_183040


namespace metallic_sheet_first_dimension_l183_183958

-- Given Conditions
variable (x : ℝ) (height width : ℝ)
def metallic_sheet :=
  (x > 0) ∧ (height = 8) ∧ (width = 36 - 2 * height)

-- Volume of the resulting box should be 5760 m³
def volume_box :=
  (width - 2 * height) * (x - 2 * height) * height = 5760

-- Prove the first dimension of the metallic sheet
theorem metallic_sheet_first_dimension (h1 : metallic_sheet x height width) (h2 : volume_box x height width) : 
  x = 52 :=
  sorry

end metallic_sheet_first_dimension_l183_183958


namespace platform_length_l183_183578

theorem platform_length (speed_km_hr : ℝ) (time_man : ℝ) (time_platform : ℝ) (L : ℝ) (P : ℝ) :
  speed_km_hr = 54 → time_man = 20 → time_platform = 22 → 
  L = (speed_km_hr * (1000 / 3600)) * time_man →
  L + P = (speed_km_hr * (1000 / 3600)) * time_platform → 
  P = 30 := 
by
  intros hs ht1 ht2 hL hLP
  sorry

end platform_length_l183_183578


namespace hostel_initial_plan_l183_183028

variable (x : ℕ) -- representing the initial number of days

-- Define the conditions
def provisions_for_250_men (x : ℕ) : ℕ := 250 * x
def provisions_for_200_men_45_days : ℕ := 200 * 45

-- Prove the statement
theorem hostel_initial_plan (x : ℕ) (h : provisions_for_250_men x = provisions_for_200_men_45_days) :
  x = 36 :=
by
  sorry

end hostel_initial_plan_l183_183028


namespace other_acute_angle_right_triangle_l183_183093

theorem other_acute_angle_right_triangle (A : ℝ) (B : ℝ) (C : ℝ) (h₁ : A + B = 90) (h₂ : B = 54) : A = 36 :=
by
  sorry

end other_acute_angle_right_triangle_l183_183093


namespace anna_spent_more_on_lunch_l183_183202

def bagel_cost : ℝ := 0.95
def cream_cheese_cost : ℝ := 0.50
def orange_juice_cost : ℝ := 1.25
def orange_juice_discount : ℝ := 0.32
def sandwich_cost : ℝ := 4.65
def avocado_cost : ℝ := 0.75
def milk_cost : ℝ := 1.15
def milk_discount : ℝ := 0.10

-- Calculate total cost of breakfast.
def breakfast_cost : ℝ := 
  let bagel_with_cream_cheese := bagel_cost + cream_cheese_cost
  let discounted_orange_juice := orange_juice_cost - (orange_juice_cost * orange_juice_discount)
  bagel_with_cream_cheese + discounted_orange_juice

-- Calculate total cost of lunch.
def lunch_cost : ℝ :=
  let sandwich_with_avocado := sandwich_cost + avocado_cost
  let discounted_milk := milk_cost - (milk_cost * milk_discount)
  sandwich_with_avocado + discounted_milk

-- Calculate the difference between lunch and breakfast costs.
theorem anna_spent_more_on_lunch : lunch_cost - breakfast_cost = 4.14 := by
  sorry

end anna_spent_more_on_lunch_l183_183202


namespace min_x2_y2_z2_given_condition_l183_183513

theorem min_x2_y2_z2_given_condition (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  ∃ (c : ℝ), c = 3 ∧ (∀ x y z : ℝ, x^3 + y^3 + z^3 - 3 * x * y * z = 8 → x^2 + y^2 + z^2 ≥ c) := 
sorry

end min_x2_y2_z2_given_condition_l183_183513


namespace minimum_value_128_l183_183049

theorem minimum_value_128 (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_prod: a * b * c = 8) : 
  (2 * a + b) * (a + 3 * c) * (b * c + 2) ≥ 128 := 
by
  sorry

end minimum_value_128_l183_183049


namespace mrs_awesome_class_l183_183450

def num_students (b g : ℕ) : ℕ := b + g

theorem mrs_awesome_class (b g : ℕ) (h1 : b = g + 3) (h2 : 480 - (b * b + g * g) = 5) : num_students b g = 31 :=
by
  sorry

end mrs_awesome_class_l183_183450


namespace minimum_value_inequality_l183_183545

variable {x y z : ℝ}

theorem minimum_value_inequality (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x + y + z = 4) :
  (1 / x + 4 / y + 9 / z) ≥ 9 :=
sorry

end minimum_value_inequality_l183_183545


namespace base9_minus_base6_l183_183986

-- Definitions from conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 325 => 3 * 9^2 + 2 * 9^1 + 5 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Main theorem statement
theorem base9_minus_base6 : base9_to_base10 325 - base6_to_base10 231 = 175 :=
by
  sorry

end base9_minus_base6_l183_183986


namespace proper_divisors_increased_by_one_l183_183595

theorem proper_divisors_increased_by_one
  (n : ℕ)
  (hn1 : 2 ≤ n)
  (exists_m : ∃ m : ℕ, ∀ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n → d + 1 ∣ m ∧ d + 1 ≠ m)
  : n = 4 ∨ n = 8 :=
  sorry

end proper_divisors_increased_by_one_l183_183595


namespace number_of_birds_is_400_l183_183875

-- Definitions of the problem
def num_stones : ℕ := 40
def num_trees : ℕ := 3 * num_stones + num_stones
def combined_trees_stones : ℕ := num_trees + num_stones
def num_birds : ℕ := 2 * combined_trees_stones

-- Statement to prove
theorem number_of_birds_is_400 : num_birds = 400 := by
  sorry

end number_of_birds_is_400_l183_183875


namespace problem_statement_l183_183138

variable {S R p a b c : ℝ}
variable (τ τ_a τ_b τ_c : ℝ)

theorem problem_statement
  (h1: S = τ * p)
  (h2: S = τ_a * (p - a))
  (h3: S = τ_b * (p - b))
  (h4: S = τ_c * (p - c))
  (h5: τ = S / p)
  (h6: τ_a = S / (p - a))
  (h7: τ_b = S / (p - b))
  (h8: τ_c = S / (p - c))
  (h9: abc / S = 4 * R) :
  1 / τ^3 - 1 / τ_a^3 - 1 / τ_b^3 - 1 / τ_c^3 = 12 * R / S^2 :=
  sorry

end problem_statement_l183_183138


namespace proportion_x_l183_183359

theorem proportion_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
sorry

end proportion_x_l183_183359


namespace cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l183_183552

theorem cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7 (p q r : ℝ)
  (h_roots : 3*p^3 - 4*p^2 + 220*p - 7 = 0 ∧ 3*q^3 - 4*q^2 + 220*q - 7 = 0 ∧ 3*r^3 - 4*r^2 + 220*r - 7 = 0)
  (h_vieta : p + q + r = 4 / 3) :
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 64.556 :=
sorry

end cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l183_183552


namespace sam_catches_alice_in_40_minutes_l183_183626

def sam_speed := 7 -- mph
def alice_speed := 4 -- mph
def initial_distance := 2 -- miles

theorem sam_catches_alice_in_40_minutes : 
  (initial_distance / (sam_speed - alice_speed)) * 60 = 40 :=
by sorry

end sam_catches_alice_in_40_minutes_l183_183626


namespace negation_of_exists_abs_lt_one_l183_183644

theorem negation_of_exists_abs_lt_one :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end negation_of_exists_abs_lt_one_l183_183644


namespace profit_ratio_l183_183940

noncomputable def effective_capital (investment : ℕ) (months : ℕ) : ℕ := investment * months

theorem profit_ratio : 
  let P_investment := 4000
  let P_months := 12
  let Q_investment := 9000
  let Q_months := 8
  let P_effective := effective_capital P_investment P_months
  let Q_effective := effective_capital Q_investment Q_months
  (P_effective / Nat.gcd P_effective Q_effective) = 2 ∧ (Q_effective / Nat.gcd P_effective Q_effective) = 3 :=
sorry

end profit_ratio_l183_183940


namespace solution_set_of_inequality_l183_183637

theorem solution_set_of_inequality :
  { x : ℝ | x > 0 ∧ x < 1 } = { x : ℝ | 1 / x > 1 } :=
by
  sorry

end solution_set_of_inequality_l183_183637


namespace solve_fraction_eq_zero_l183_183729

theorem solve_fraction_eq_zero (a : ℝ) (h : a ≠ -1) : (a^2 - 1) / (a + 1) = 0 ↔ a = 1 :=
by {
  sorry
}

end solve_fraction_eq_zero_l183_183729


namespace circles_intersect_l183_183853

theorem circles_intersect :
  ∀ (x y : ℝ),
    ((x^2 + y^2 - 2 * x + 4 * y + 1 = 0) →
    (x^2 + y^2 - 6 * x + 2 * y + 9 = 0) →
    (∃ c1 c2 r1 r2 d : ℝ,
      (x - 1)^2 + (y + 2)^2 = r1 ∧ r1 = 4 ∧
      (x - 3)^2 + (y + 1)^2 = r2 ∧ r2 = 1 ∧
      d = Real.sqrt ((3 - 1)^2 + (-1 + 2)^2) ∧
      d > abs (r1 - r2) ∧ d < (r1 + r2))) :=
sorry

end circles_intersect_l183_183853


namespace original_sequence_polynomial_of_degree_3_l183_183840

def is_polynomial_of_degree (u : ℕ → ℤ) (n : ℕ) :=
  ∃ a b c d : ℤ, u n = a * n^3 + b * n^2 + c * n + d

def fourth_difference_is_zero (u : ℕ → ℤ) :=
  ∀ n : ℕ, (u (n + 4) - 4 * u (n + 3) + 6 * u (n + 2) - 4 * u (n + 1) + u n) = 0

theorem original_sequence_polynomial_of_degree_3 (u : ℕ → ℤ)
  (h : fourth_difference_is_zero u) : 
  ∃ (a b c d : ℤ), ∀ n : ℕ, u n = a * n^3 + b * n^2 + c * n + d := sorry

end original_sequence_polynomial_of_degree_3_l183_183840


namespace minimum_number_of_guests_l183_183843

theorem minimum_number_of_guests :
  ∀ (total_food : ℝ) (max_food_per_guest : ℝ), total_food = 411 → max_food_per_guest = 2.5 →
  ⌈total_food / max_food_per_guest⌉ = 165 :=
by
  intros total_food max_food_per_guest h1 h2
  rw [h1, h2]
  norm_num
  sorry

end minimum_number_of_guests_l183_183843


namespace machine_output_l183_183210

theorem machine_output (input : ℕ) (output : ℕ) (h : input = 26) (h_out : output = input + 15 - 6) : output = 35 := 
by 
  sorry

end machine_output_l183_183210


namespace find_k_l183_183955

theorem find_k (k : ℚ) : (∀ x y : ℚ, (x, y) = (2, 1) → 3 * k * x - k = -4 * y - 2) → k = -(6 / 5) :=
by
  intro h
  have key := h 2 1 rfl
  have : 3 * k * 2 - k = -4 * 1 - 2 := key
  linarith

end find_k_l183_183955


namespace sum_c_2017_l183_183775

def a (n : ℕ) : ℕ := 3 * n + 1

def b (n : ℕ) : ℕ := 4^(n-1)

def c (n : ℕ) : ℕ := if n = 1 then 7 else 3 * 4^(n-1)

theorem sum_c_2017 : (Finset.range 2017).sum c = 4^2017 + 3 :=
by
  -- definitions and required assumptions
  sorry

end sum_c_2017_l183_183775


namespace abigail_total_savings_l183_183743

def monthly_savings : ℕ := 4000
def months_in_year : ℕ := 12

theorem abigail_total_savings : monthly_savings * months_in_year = 48000 := by
  sorry

end abigail_total_savings_l183_183743


namespace colten_chickens_l183_183294

theorem colten_chickens (x : ℕ) (Quentin Skylar Colten : ℕ) 
  (h1 : Quentin + Skylar + Colten = 383)
  (h2 : Quentin = 25 + 2 * Skylar)
  (h3 : Skylar = 3 * Colten - 4) : 
  Colten = 37 := 
  sorry

end colten_chickens_l183_183294


namespace sum_first_10_log_a_l183_183360

-- Given sum of the first n terms of the sequence
def S (n : ℕ) : ℕ := 2^n - 1

-- Function to get general term log_2 a_n
def log_a (n : ℕ) : ℕ := n - 1

-- The statement to prove
theorem sum_first_10_log_a : (List.range 10).sum = 45 := by 
  sorry

end sum_first_10_log_a_l183_183360


namespace ratio_apples_pie_to_total_is_one_to_two_l183_183023

variable (x : ℕ) -- number of apples Paul put aside for pie
variable (total_apples : ℕ := 62) 
variable (fridge_apples : ℕ := 25)
variable (muffin_apples : ℕ := 6)

def apples_pie_ratio (x total_apples : ℕ) : ℕ := x / gcd x total_apples

theorem ratio_apples_pie_to_total_is_one_to_two :
  x + fridge_apples + muffin_apples = total_apples -> apples_pie_ratio x total_apples = 1 / 2 :=
by
  sorry

end ratio_apples_pie_to_total_is_one_to_two_l183_183023


namespace expression_value_as_fraction_l183_183478

theorem expression_value_as_fraction (x y : ℕ) (hx : x = 3) (hy : y = 5) : 
  ( ( (1 / (y : ℚ)) / (1 / (x : ℚ)) ) ^ 2 ) = 9 / 25 := 
by
  sorry

end expression_value_as_fraction_l183_183478


namespace infinite_not_expressible_as_sum_of_three_squares_l183_183012

theorem infinite_not_expressible_as_sum_of_three_squares :
  ∃ (n : ℕ), ∃ (infinitely_many_n : ℕ → Prop), (∀ m:ℕ, (infinitely_many_n m ↔ m ≡ 7 [MOD 8])) ∧ ∀ a b c : ℕ, n ≠ a^2 + b^2 + c^2 := 
by
  sorry

end infinite_not_expressible_as_sum_of_three_squares_l183_183012


namespace avg_salary_l183_183682

-- Conditions as definitions
def number_of_technicians : Nat := 7
def salary_per_technician : Nat := 10000
def number_of_workers : Nat := 14
def salary_per_non_technician : Nat := 6000

-- Total salary of technicians
def total_salary_technicians : Nat := number_of_technicians * salary_per_technician

-- Number of non-technicians
def number_of_non_technicians : Nat := number_of_workers - number_of_technicians

-- Total salary of non-technicians
def total_salary_non_technicians : Nat := number_of_non_technicians * salary_per_non_technician

-- Total salary
def total_salary_all_workers : Nat := total_salary_technicians + total_salary_non_technicians

-- Average salary of all workers
def avg_salary_all_workers : Nat := total_salary_all_workers / number_of_workers

-- Theorem to prove
theorem avg_salary (A : Nat) (h : A = avg_salary_all_workers) : A = 8000 := by
  sorry

end avg_salary_l183_183682


namespace expression_multiple_of_five_l183_183719

theorem expression_multiple_of_five (n : ℕ) (h : n ≥ 10) : 
  (∃ k : ℕ, (n + 2) * (n + 1) = 5 * k) :=
sorry

end expression_multiple_of_five_l183_183719


namespace find_speeds_of_A_and_B_l183_183584

noncomputable def speed_A_and_B (x y : ℕ) : Prop :=
  30 * x - 30 * y = 300 ∧ 2 * x + 2 * y = 300

theorem find_speeds_of_A_and_B : ∃ (x y : ℕ), speed_A_and_B x y ∧ x = 80 ∧ y = 70 :=
by
  sorry

end find_speeds_of_A_and_B_l183_183584


namespace cannot_form_square_with_sticks_l183_183546

theorem cannot_form_square_with_sticks
    (num_1cm_sticks : ℕ)
    (num_2cm_sticks : ℕ)
    (num_3cm_sticks : ℕ)
    (num_4cm_sticks : ℕ)
    (len_1cm_stick : ℕ)
    (len_2cm_stick : ℕ)
    (len_3cm_stick : ℕ)
    (len_4cm_stick : ℕ)
    (sum_lengths : ℕ) :
    num_1cm_sticks = 6 →
    num_2cm_sticks = 3 →
    num_3cm_sticks = 6 →
    num_4cm_sticks = 5 →
    len_1cm_stick = 1 →
    len_2cm_stick = 2 →
    len_3cm_stick = 3 →
    len_4cm_stick = 4 →
    sum_lengths = num_1cm_sticks * len_1cm_stick + 
                  num_2cm_sticks * len_2cm_stick + 
                  num_3cm_sticks * len_3cm_stick + 
                  num_4cm_sticks * len_4cm_stick →
    ∃ (s : ℕ), sum_lengths = 4 * s → False := 
by
  intros num_1cm_sticks_eq num_2cm_sticks_eq num_3cm_sticks_eq num_4cm_sticks_eq
         len_1cm_stick_eq len_2cm_stick_eq len_3cm_stick_eq len_4cm_stick_eq
         sum_lengths_def

  sorry

end cannot_form_square_with_sticks_l183_183546


namespace six_digits_sum_l183_183481

theorem six_digits_sum 
  (a b c d e f g : ℕ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e) (h5 : a ≠ f) (h6 : a ≠ g)
  (h7 : b ≠ c) (h8 : b ≠ d) (h9 : b ≠ e) (h10 : b ≠ f) (h11 : b ≠ g)
  (h12 : c ≠ d) (h13 : c ≠ e) (h14 : c ≠ f) (h15 : c ≠ g)
  (h16 : d ≠ e) (h17 : d ≠ f) (h18 : d ≠ g)
  (h19 : e ≠ f) (h20 : e ≠ g)
  (h21 : f ≠ g)
  (h22 : 2 ≤ a) (h23 : a ≤ 9) 
  (h24 : 2 ≤ b) (h25 : b ≤ 9) 
  (h26 : 2 ≤ c) (h27 : c ≤ 9)
  (h28 : 2 ≤ d) (h29 : d ≤ 9)
  (h30 : 2 ≤ e) (h31 : e ≤ 9)
  (h32 : 2 ≤ f) (h33 : f ≤ 9)
  (h34 : 2 ≤ g) (h35 : g ≤ 9)
  (h36 : a + b + c = 25)
  (h37 : d + e + f + g = 15)
  (h38 : b = e) :
  a + b + c + d + f + g = 31 := 
sorry

end six_digits_sum_l183_183481


namespace luisa_mpg_l183_183508

theorem luisa_mpg
  (d_grocery d_mall d_pet d_home : ℕ)
  (cost_per_gal total_cost : ℚ)
  (total_miles : ℕ )
  (total_gallons : ℚ)
  (mpg : ℚ):
  d_grocery = 10 →
  d_mall = 6 →
  d_pet = 5 →
  d_home = 9 →
  cost_per_gal = 3.5 →
  total_cost = 7 →
  total_miles = d_grocery + d_mall + d_pet + d_home →
  total_gallons = total_cost / cost_per_gal →
  mpg = total_miles / total_gallons →
  mpg = 15 :=
by
  intros
  sorry

end luisa_mpg_l183_183508


namespace arithmetic_sequence_sum_and_mean_l183_183925

theorem arithmetic_sequence_sum_and_mean :
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  S = 121 ∧ (S / n) = 11 :=
by
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  have h1 : S = 121 := sorry
  have h2 : (S / n) = 11 := by
    rw [h1]
    exact sorry
  exact ⟨h1, h2⟩

end arithmetic_sequence_sum_and_mean_l183_183925


namespace trapezoid_shorter_base_length_l183_183135

theorem trapezoid_shorter_base_length 
  (a b : ℕ) 
  (mid_segment_length longer_base : ℕ) 
  (h1 : mid_segment_length = 5) 
  (h2 : longer_base = 103) 
  (trapezoid_property : mid_segment_length = (longer_base - a) / 2) : 
  a = 93 := 
sorry

end trapezoid_shorter_base_length_l183_183135


namespace jane_crayon_count_l183_183250

def billy_crayons : ℝ := 62.0
def total_crayons : ℝ := 114
def jane_crayons : ℝ := total_crayons - billy_crayons

theorem jane_crayon_count : jane_crayons = 52 := by
  unfold jane_crayons
  show total_crayons - billy_crayons = 52
  sorry

end jane_crayon_count_l183_183250


namespace negation_all_swans_white_l183_183597

variables {α : Type} (swan white : α → Prop)

theorem negation_all_swans_white :
  (¬ ∀ x, swan x → white x) ↔ (∃ x, swan x ∧ ¬ white x) :=
by {
  sorry
}

end negation_all_swans_white_l183_183597


namespace complement_intersection_l183_183786

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {2, 4}
def N : Set ℕ := {3, 5}

theorem complement_intersection (hU: U = {1, 2, 3, 4, 5}) (hM: M = {2, 4}) (hN: N = {3, 5}) : 
  (U \ M) ∩ N = {3, 5} := 
by 
  sorry

end complement_intersection_l183_183786


namespace find_f_2013_l183_183237

noncomputable def f : ℝ → ℝ := sorry
axiom functional_eq : ∀ (m n : ℝ), f (m + n^2) = f m + 2 * (f n)^2
axiom f_1_ne_0 : f 1 ≠ 0

theorem find_f_2013 : f 2013 = 4024 * (f 1)^2 + f 1 :=
sorry

end find_f_2013_l183_183237


namespace min_rectilinear_distance_to_parabola_l183_183291

theorem min_rectilinear_distance_to_parabola :
  ∃ t : ℝ, ∀ t', (|t' + 1| + t'^2) ≥ (|t + 1| + t^2) ∧ (|t + 1| + t^2) = 3 / 4 := sorry

end min_rectilinear_distance_to_parabola_l183_183291


namespace value_of_e_over_f_l183_183673

theorem value_of_e_over_f 
    (a b c d e f : ℝ) 
    (h1 : a * b * c = 1.875 * d * e * f)
    (h2 : a / b = 5 / 2)
    (h3 : b / c = 1 / 2)
    (h4 : c / d = 1)
    (h5 : d / e = 3 / 2) : 
    e / f = 1 / 3 :=
by
  sorry

end value_of_e_over_f_l183_183673


namespace bacteria_growth_l183_183317

theorem bacteria_growth (d : ℕ) (t : ℕ) (initial final : ℕ) 
  (h_doubling : d = 4) 
  (h_initial : initial = 500) 
  (h_final : final = 32000) 
  (h_ratio : final / initial = 2^6) :
  t = d * 6 → t = 24 :=
by
  sorry

end bacteria_growth_l183_183317


namespace venus_hall_meal_cost_l183_183064

theorem venus_hall_meal_cost (V : ℕ) :
  let caesars_total_cost := 800 + 30 * 60;
  let venus_hall_total_cost := 500 + V * 60;
  caesars_total_cost = venus_hall_total_cost → V = 35 :=
by
  let caesars_total_cost := 800 + 30 * 60
  let venus_hall_total_cost := 500 + V * 60
  intros h
  sorry

end venus_hall_meal_cost_l183_183064


namespace units_digit_G1000_l183_183846

def units_digit (n : ℕ) : ℕ :=
  n % 10

def power_cycle : List ℕ := [3, 9, 7, 1]

def G (n : ℕ) : ℕ :=
  3^(2^n) + 2

theorem units_digit_G1000 : units_digit (G 1000) = 3 :=
by
  sorry

end units_digit_G1000_l183_183846


namespace smallest_value_of_expression_l183_183694

noncomputable def f (x : ℝ) : ℝ := x^4 + 14*x^3 + 52*x^2 + 56*x + 16

theorem smallest_value_of_expression :
  ∀ z : Fin 4 → ℝ, (∀ i, f (z i) = 0) → 
  ∃ (a b c d : Fin 4), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ d ≠ b ∧ a ≠ c ∧ 
  |(z a * z b) + (z c * z d)| = 8 :=
by
  sorry

end smallest_value_of_expression_l183_183694


namespace matt_minus_sara_l183_183005

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25

def matt_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
def sara_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

theorem matt_minus_sara : matt_total - sara_total = 0 :=
by
  sorry

end matt_minus_sara_l183_183005


namespace price_per_glass_second_day_l183_183604

-- Given conditions
variables {O P : ℝ}
axiom condition1 : 0.82 * 2 * O = P * 3 * O

-- Problem statement
theorem price_per_glass_second_day : 
  P = 0.55 :=
by
  -- This is where the actual proof would go
  sorry

end price_per_glass_second_day_l183_183604


namespace range_of_x_for_odd_function_l183_183472

theorem range_of_x_for_odd_function (f : ℝ → ℝ) (domain : Set ℝ)
  (h_odd : ∀ x ∈ domain, f (-x) = -f x)
  (h_mono : ∀ x y, 0 < x -> x < y -> f x < f y)
  (h_f3 : f 3 = 0)
  (h_ineq : ∀ x, x ∈ domain -> x * (f x - f (-x)) < 0) : 
  ∀ x, x * f x < 0 ↔ -3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3 :=
by sorry

end range_of_x_for_odd_function_l183_183472


namespace smallest_n_boxes_cookies_l183_183117

theorem smallest_n_boxes_cookies (n : ℕ) (h : (17 * n - 1) % 12 = 0) : n = 5 :=
sorry

end smallest_n_boxes_cookies_l183_183117


namespace tank_capacity_l183_183967

theorem tank_capacity (c w : ℝ) 
  (h1 : w / c = 1 / 7) 
  (h2 : (w + 5) / c = 1 / 5) : 
  c = 87.5 := 
by
  sorry

end tank_capacity_l183_183967


namespace find_weight_first_dog_l183_183642

noncomputable def weight_first_dog (x : ℕ) (y : ℕ) : Prop :=
  (x + 31 + 35 + 33) / 4 = (x + 31 + 35 + 33 + y) / 5

theorem find_weight_first_dog (x : ℕ) : weight_first_dog x 31 → x = 25 := by
  sorry

end find_weight_first_dog_l183_183642


namespace inequality_a_b_c_d_l183_183503

theorem inequality_a_b_c_d 
  (a b c d : ℝ) 
  (h0 : 0 ≤ a) 
  (h1 : a ≤ b) 
  (h2 : b ≤ c) 
  (h3 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
by
  sorry

end inequality_a_b_c_d_l183_183503


namespace total_pennies_l183_183088

-- Definitions based on conditions
def initial_pennies_per_compartment := 2
def additional_pennies_per_compartment := 6
def compartments := 12

-- Mathematically equivalent proof statement
theorem total_pennies (initial_pennies_per_compartment : Nat) 
                      (additional_pennies_per_compartment : Nat)
                      (compartments : Nat) : 
                      initial_pennies_per_compartment = 2 → 
                      additional_pennies_per_compartment = 6 → 
                      compartments = 12 → 
                      compartments * (initial_pennies_per_compartment + additional_pennies_per_compartment) = 96 := 
by
  intros
  sorry

end total_pennies_l183_183088


namespace complement_union_equals_l183_183351

def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3, 4, 5}
def A : Set ℤ := {-1, 0, 1, 2, 3}
def B : Set ℤ := {-2, 0, 2}

def C_I (I : Set ℤ) (s : Set ℤ) : Set ℤ := I \ s

theorem complement_union_equals :
  C_I universal_set (A ∪ B) = {4, 5} :=
by
  sorry

end complement_union_equals_l183_183351


namespace vector_field_lines_l183_183750

noncomputable def vector_lines : Prop :=
  ∃ (C_1 C_2 : ℝ), ∀ (x y z : ℝ), (9 * z^2 + 4 * y^2 = C_1) ∧ (x = C_2)

-- We state the proof goal as follows:
theorem vector_field_lines :
  ∀ (a : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ), 
    (∀ (x y z : ℝ), a (x, y, z) = (0, 9 * z, -4 * y)) →
    vector_lines :=
by
  intro a ha
  sorry

end vector_field_lines_l183_183750


namespace diet_soda_bottles_l183_183522

/-- Define variables for the number of bottles. -/
def total_bottles : ℕ := 38
def regular_soda : ℕ := 30

/-- Define the problem of finding the number of diet soda bottles -/
def diet_soda := total_bottles - regular_soda

/-- Claim that the number of diet soda bottles is 8 -/
theorem diet_soda_bottles : diet_soda = 8 :=
by
  sorry

end diet_soda_bottles_l183_183522


namespace iPhone_savings_l183_183446

theorem iPhone_savings
  (costX costY : ℕ)
  (discount_same_model discount_mixed : ℝ)
  (h1 : costX = 600)
  (h2 : costY = 800)
  (h3 : discount_same_model = 0.05)
  (h4 : discount_mixed = 0.03) :
  (costX + costX + costY) - ((costX * (1 - discount_same_model)) * 2 + costY * (1 - discount_mixed)) = 84 :=
by
  sorry

end iPhone_savings_l183_183446


namespace find_b_value_l183_183045

theorem find_b_value (b : ℕ) 
  (h1 : 5 ^ 5 * b = 3 * 15 ^ 5) 
  (h2 : b = 9 ^ 3) : b = 729 :=
by
  sorry

end find_b_value_l183_183045


namespace remaining_surface_area_l183_183749

def edge_length_original : ℝ := 9
def edge_length_small : ℝ := 2
def surface_area (a : ℝ) : ℝ := 6 * a^2

theorem remaining_surface_area :
  surface_area edge_length_original - 3 * (edge_length_small ^ 2) + 3 * (edge_length_small ^ 2) = 486 :=
by
  sorry

end remaining_surface_area_l183_183749


namespace count_three_digit_values_with_double_sum_eq_six_l183_183784

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_three_digit (x : ℕ) : Prop := 
  100 ≤ x ∧ x < 1000

theorem count_three_digit_values_with_double_sum_eq_six :
  ∃ count : ℕ, is_three_digit count ∧ (
    (∀ x, is_three_digit x → sum_of_digits (sum_of_digits x) = 6) ↔ count = 30
  ) :=
sorry

end count_three_digit_values_with_double_sum_eq_six_l183_183784


namespace quadratic_b_value_l183_183778
open Real

theorem quadratic_b_value (b n : ℝ) 
  (h1: b < 0) 
  (h2: ∀ x, x^2 + b * x + (1 / 4) = (x + n)^2 + (1 / 16)) :
  b = - (sqrt 3 / 2) :=
by
  -- sorry is used to skip the proof
  sorry

end quadratic_b_value_l183_183778


namespace primes_between_30_and_50_l183_183886

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l183_183886


namespace yura_picture_dimensions_l183_183629

theorem yura_picture_dimensions (l w : ℕ) (h_frame : (l + 2) * (w + 2) - l * w = l * w) :
    (l = 3 ∧ w = 10) ∨ (l = 4 ∧ w = 6) :=
by {
  sorry
}

end yura_picture_dimensions_l183_183629


namespace faye_homework_problems_l183_183755

----- Definitions based on the conditions given -----

def total_math_problems : ℕ := 46
def total_science_problems : ℕ := 9
def problems_finished_at_school : ℕ := 40

----- Theorem statement -----

theorem faye_homework_problems : total_math_problems + total_science_problems - problems_finished_at_school = 15 := by
  -- Sorry is used here to skip the proof
  sorry

end faye_homework_problems_l183_183755


namespace neg_p_false_sufficient_but_not_necessary_for_p_or_q_l183_183945

variable (p q : Prop)

theorem neg_p_false_sufficient_but_not_necessary_for_p_or_q :
  (¬ p = false) → (p ∨ q) ∧ ¬((p ∨ q) → (¬ p = false)) :=
by
  sorry

end neg_p_false_sufficient_but_not_necessary_for_p_or_q_l183_183945


namespace decipher_rebus_l183_183592

theorem decipher_rebus (a b c d : ℕ) :
  (a = 10 ∧ b = 14 ∧ c = 12 ∧ d = 13) ↔
  (∀ (x y z w: ℕ), 
    (x = 10 → 5 + 5 * 7 = 49) ∧
    (y = 14 → 2 - 4 * 3 = 9) ∧
    (z = 12 → 12 - 1 - 1 * 2 = 20) ∧
    (w = 13 → 13 - 1 + 10 - 5 = 17) ∧
    (49 + 9 + 20 + 17 = 95)) :=
by sorry

end decipher_rebus_l183_183592


namespace find_ab_integer_l183_183439

theorem find_ab_integer (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a ≠ b) :
    ∃ n : ℤ, (a^b + b^a) = n * (a^a - b^b) ↔ (a = 2 ∧ b = 1) ∨ (a = 1 ∧ b = 2) := 
sorry

end find_ab_integer_l183_183439


namespace sum_of_midpoints_l183_183231

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l183_183231


namespace cost_of_5kg_l183_183035

def cost_of_seeds (x : ℕ) : ℕ :=
  if x ≤ 2 then 5 * x else 4 * x + 2

theorem cost_of_5kg : cost_of_seeds 5 = 22 := by
  sorry

end cost_of_5kg_l183_183035


namespace tan_of_angle_l183_183266

open Real

-- Given conditions in the problem
variables {α : ℝ}

-- Define the given conditions
def sinα_condition (α : ℝ) : Prop := sin α = 3 / 5
def α_in_quadrant_2 (α : ℝ) : Prop := π / 2 < α ∧ α < π

-- Define the Lean statement
theorem tan_of_angle {α : ℝ} (h1 : sinα_condition α) (h2 : α_in_quadrant_2 α) :
  tan α = -3 / 4 :=
sorry

end tan_of_angle_l183_183266


namespace shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l183_183097

theorem shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder
  (c : ℝ)
  (r : ℝ)
  (θ : ℝ)
  (hr : r ≥ 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  ∃ (x y z : ℝ), (z = c) ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ :=
by
  sorry

end shape_descibed_by_z_eq_c_in_cylindrical_coords_is_cylinder_l183_183097


namespace team_A_wins_series_4_1_probability_l183_183974

noncomputable def probability_team_A_wins_series_4_1 : ℝ :=
  let home_win_prob : ℝ := 0.6
  let away_win_prob : ℝ := 0.5
  let home_loss_prob : ℝ := 1 - home_win_prob
  let away_loss_prob : ℝ := 1 - away_win_prob
  -- Scenario 1: L W W W W
  let p1 := home_loss_prob * home_win_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 2: W L W W W
  let p2 := home_win_prob * home_loss_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 3: W W L W W
  let p3 := home_win_prob * home_win_prob * away_loss_prob * away_win_prob * home_win_prob
  -- Scenario 4: W W W L W
  let p4 := home_win_prob * home_win_prob * away_win_prob * away_loss_prob * home_win_prob
  p1 + p2 + p3 + p4

theorem team_A_wins_series_4_1_probability : 
  probability_team_A_wins_series_4_1 = 0.18 :=
by
  -- This where the proof would go
  sorry

end team_A_wins_series_4_1_probability_l183_183974


namespace simplify_expression_l183_183612

open Nat

theorem simplify_expression (x : ℤ) : 2 - (3 - (2 - (5 - (3 - x)))) = -1 - x :=
by
  sorry

end simplify_expression_l183_183612


namespace average_age_of_remaining_people_l183_183065

theorem average_age_of_remaining_people:
  ∀ (ages : List ℕ), 
  (List.length ages = 8) →
  (List.sum ages = 224) →
  (24 ∈ ages) →
  ((List.sum ages - 24) / 7 = 28 + 4/7) := 
by
  intro ages
  intro h_len
  intro h_sum
  intro h_24
  sorry

end average_age_of_remaining_people_l183_183065


namespace ram_balance_speed_l183_183454

theorem ram_balance_speed
  (part_speed : ℝ)
  (balance_distance : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (part_time : ℝ)
  (balance_speed : ℝ)
  (h1 : part_speed = 20)
  (h2 : total_distance = 400)
  (h3 : total_time = 8)
  (h4 : part_time = 3.2)
  (h5 : balance_distance = total_distance - part_speed * part_time)
  (h6 : balance_speed = balance_distance / (total_time - part_time)) :
  balance_speed = 70 :=
by
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end ram_balance_speed_l183_183454


namespace remainder_product_div_17_l183_183879

theorem remainder_product_div_17 :
  (2357 ≡ 6 [MOD 17]) → (2369 ≡ 4 [MOD 17]) → (2384 ≡ 0 [MOD 17]) →
  (2391 ≡ 9 [MOD 17]) → (3017 ≡ 9 [MOD 17]) → (3079 ≡ 0 [MOD 17]) →
  (3082 ≡ 3 [MOD 17]) →
  ((2357 * 2369 * 2384 * 2391) * (3017 * 3079 * 3082) ≡ 0 [MOD 17]) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end remainder_product_div_17_l183_183879


namespace nth_term_closed_form_arithmetic_sequence_l183_183873

open Nat

noncomputable def S (n : ℕ) : ℕ := 3 * n^2 + 4 * n
noncomputable def a (n : ℕ) : ℕ := if h : n > 0 then S n - S (n-1) else S n

theorem nth_term_closed_form (n : ℕ) (h : n > 0) : a n = 6 * n + 1 :=
by
  sorry

theorem arithmetic_sequence (n : ℕ) (h : n > 1) : a n - a (n - 1) = 6 :=
by
  sorry

end nth_term_closed_form_arithmetic_sequence_l183_183873


namespace geometric_sequence_b_value_l183_183063

theorem geometric_sequence_b_value (b : ℝ) (r : ℝ) (h1 : 210 * r = b) (h2 : b * r = 35 / 36) (hb : b > 0) : 
  b = Real.sqrt (7350 / 36) :=
by
  sorry

end geometric_sequence_b_value_l183_183063


namespace find_theta_perpendicular_l183_183052

theorem find_theta_perpendicular (θ : ℝ) (hθ : 0 < θ ∧ θ < π)
  (a b : ℝ × ℝ) (ha : a = (Real.sin θ, 1)) (hb : b = (2 * Real.cos θ, -1))
  (hperp : a.fst * b.fst + a.snd * b.snd = 0) : θ = π / 4 :=
by
  -- Proof would be written here
  sorry

end find_theta_perpendicular_l183_183052


namespace product_of_consecutive_nat_is_divisible_by_2_l183_183073

theorem product_of_consecutive_nat_is_divisible_by_2 (n : ℕ) : 2 ∣ n * (n + 1) :=
sorry

end product_of_consecutive_nat_is_divisible_by_2_l183_183073


namespace find_m_l183_183265

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A based on the condition in the problem
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5 * x + m = 0}

-- Define the complement of A in the universal set U
def complementA (m : ℕ) : Set ℕ := U \ A m

-- Given condition that the complement of A in U is {2, 3}
def complementA_condition : Set ℕ := {2, 3}

-- The proof problem statement: Prove that m = 4 given the conditions
theorem find_m (m : ℕ) (h : complementA m = complementA_condition) : m = 4 :=
sorry

end find_m_l183_183265


namespace number_of_days_worked_l183_183791

theorem number_of_days_worked (total_toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : total_toys_per_week = 6000) (h₂ : toys_per_day = 1500) : (total_toys_per_week / toys_per_day) = 4 :=
by
  sorry

end number_of_days_worked_l183_183791


namespace tanA_tanB_eq_thirteen_div_four_l183_183872

-- Define the triangle and its properties
variables {A B C : Type}
variables (a b c : ℝ)  -- sides BC, AC, AB
variables (HF HC : ℝ)  -- segments of altitude CF
variables (tanA tanB : ℝ)

-- Given conditions
def orthocenter_divide_altitude (HF HC : ℝ) : Prop :=
  HF = 8 ∧ HC = 18

-- The result we want to prove
theorem tanA_tanB_eq_thirteen_div_four (h : orthocenter_divide_altitude HF HC) : 
  tanA * tanB = 13 / 4 :=
  sorry

end tanA_tanB_eq_thirteen_div_four_l183_183872


namespace greg_pages_per_day_l183_183572

variable (greg_pages : ℕ)
variable (brad_pages : ℕ)

theorem greg_pages_per_day :
  brad_pages = 26 → brad_pages = greg_pages + 8 → greg_pages = 18 :=
by
  intros h1 h2
  rw [h1, add_comm] at h2
  linarith

end greg_pages_per_day_l183_183572


namespace find_first_term_and_common_difference_l183_183057

variable (a d : ℕ)
variable (S_even S_odd S_total : ℕ)

-- Conditions
axiom condition1 : S_total = 354
axiom condition2 : S_even = 192
axiom condition3 : S_odd = 162
axiom condition4 : 12*(2*a + 11*d) = 2*S_total
axiom condition5 : 6*(a + 6*d) = S_even
axiom condition6 : 6*(a + 5*d) = S_odd

-- Theorem to prove
theorem find_first_term_and_common_difference (a d S_even S_odd S_total : ℕ)
  (h1 : S_total = 354)
  (h2 : S_even = 192)
  (h3 : S_odd = 162)
  (h4 : 12*(2*a + 11*d) = 2*S_total)
  (h5 : 6*(a + 6*d) = S_even)
  (h6 : 6*(a + 5*d) = S_odd) : a = 2 ∧ d = 5 := by
  sorry

end find_first_term_and_common_difference_l183_183057


namespace ternary_to_decimal_l183_183680

theorem ternary_to_decimal (k : ℕ) (hk : k > 0) : (1 * 3^3 + k * 3^1 + 2 = 35) → k = 2 :=
by
  sorry

end ternary_to_decimal_l183_183680


namespace find_cos_squared_y_l183_183301

noncomputable def α : ℝ := Real.arccos (-3 / 7)

def arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

def transformed_arithmetic_progression (a b c : ℝ) : Prop :=
  14 / Real.cos b = 1 / Real.cos a + 1 / Real.cos c

theorem find_cos_squared_y (x y z : ℝ)
  (h1 : arithmetic_progression x y z)
  (h2 : transformed_arithmetic_progression x y z)
  (hα : 2 * α = z - x) : Real.cos y ^ 2 = 10 / 13 :=
by
  sorry

end find_cos_squared_y_l183_183301


namespace positive_integer_conditions_l183_183296

theorem positive_integer_conditions (p : ℕ) (hp : p > 0) : 
  (∃ k : ℕ, k > 0 ∧ 4 * p + 28 = k * (3 * p - 7)) ↔ (p = 6 ∨ p = 28) :=
by
  sorry

end positive_integer_conditions_l183_183296


namespace curve_intersects_itself_l183_183021

theorem curve_intersects_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ (t₁^2 - 3, t₁^3 - 6 * t₁ + 4) = (3, 4) ∧ (t₂^2 - 3, t₂^3 - 6 * t₂ + 4) = (3, 4) :=
sorry

end curve_intersects_itself_l183_183021


namespace MattRate_l183_183510

variable (M : ℝ) (t : ℝ)

def MattRateCondition : Prop := M * t = 220
def TomRateCondition : Prop := (M + 5) * t = 275

theorem MattRate (h1 : MattRateCondition M t) (h2 : TomRateCondition M t) : M = 20 := by
  sorry

end MattRate_l183_183510


namespace repeating_decimal_sum_l183_183008

theorem repeating_decimal_sum :
  let a := (2 : ℚ) / 3
  let b := (2 : ℚ) / 9
  let c := (4 : ℚ) / 9
  a + b - c = (4 : ℚ) / 9 :=
by
  sorry

end repeating_decimal_sum_l183_183008


namespace units_digit_product_first_four_composite_numbers_l183_183985

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l183_183985


namespace fraction_greater_than_decimal_l183_183234

/-- 
  Prove that the fraction 1/3 is greater than the decimal 0.333 by the amount 1/(3 * 10^3)
-/
theorem fraction_greater_than_decimal :
  (1 / 3 : ℚ) = (333 / 1000 : ℚ) + (1 / (3 * 1000) : ℚ) :=
by
  sorry

end fraction_greater_than_decimal_l183_183234


namespace function_relationship_area_60_maximum_area_l183_183951

-- Definitions and conditions
def perimeter := 32
def side_length (x : ℝ) : ℝ := 16 - x  -- One side of the rectangle
def area (x : ℝ) : ℝ := x * (16 - x)

-- Theorem 1: Function relationship between y and x
theorem function_relationship (x : ℝ) (hx : 0 < x ∧ x < 16) : area x = -x^2 + 16 * x :=
by
  sorry

-- Theorem 2: Values of x when the area is 60 square meters
theorem area_60 (x : ℝ) (hx1 : area x = 60) : x = 6 ∨ x = 10 :=
by
  sorry

-- Theorem 3: Maximum area
theorem maximum_area : ∃ x, area x = 64 ∧ x = 8 :=
by
  sorry

end function_relationship_area_60_maximum_area_l183_183951


namespace find_n_l183_183628

theorem find_n (n : ℤ) (h : n * 1296 / 432 = 36) : n = 12 :=
sorry

end find_n_l183_183628


namespace more_birds_than_storks_l183_183811

-- Defining the initial number of birds
def initial_birds : ℕ := 2

-- Defining the number of birds that joined
def additional_birds : ℕ := 5

-- Defining the number of storks that joined
def storks : ℕ := 4

-- Defining the total number of birds
def total_birds : ℕ := initial_birds + additional_birds

-- Defining the problem statement in Lean 4
theorem more_birds_than_storks : (total_birds - storks) = 3 := by
  sorry

end more_birds_than_storks_l183_183811


namespace range_of_mn_l183_183892

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x

theorem range_of_mn (m n : ℝ)
  (h₁ : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4)
  (h₂ : ∀ z, -5 ≤ z ∧ z ≤ 4 → ∃ x, f x = z ∧ m ≤ x ∧ x ≤ n) :
  1 ≤ m + n ∧ m + n ≤ 7 :=
by
  sorry

end range_of_mn_l183_183892


namespace three_digit_problem_l183_183425

theorem three_digit_problem :
  ∃ (M Γ U : ℕ), 
    M ≠ Γ ∧ M ≠ U ∧ Γ ≠ U ∧
    M ≤ 9 ∧ Γ ≤ 9 ∧ U ≤ 9 ∧
    100 * M + 10 * Γ + U = (M + Γ + U) * (M + Γ + U - 2) ∧
    100 * M + 10 * Γ + U = 195 :=
by
  sorry

end three_digit_problem_l183_183425


namespace monthly_income_of_p_l183_183933

theorem monthly_income_of_p (P Q R : ℕ) 
    (h1 : (P + Q) / 2 = 5050)
    (h2 : (Q + R) / 2 = 6250)
    (h3 : (P + R) / 2 = 5200) :
    P = 4000 :=
by
  -- proof would go here
  sorry

end monthly_income_of_p_l183_183933


namespace intersection_of_A_and_B_range_of_a_l183_183662

open Set

namespace ProofProblem

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | x ≥ 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 ≤ x ∧ x < 3} := 
sorry

theorem range_of_a (a : ℝ) :
  (B ∪ C a) = C a → a ≤ 3 :=
sorry

end ProofProblem

end intersection_of_A_and_B_range_of_a_l183_183662


namespace original_number_is_80_l183_183133

theorem original_number_is_80 (t : ℝ) (h : t * 1.125 - t * 0.75 = 30) : t = 80 := by
  sorry

end original_number_is_80_l183_183133


namespace isosceles_triangle_base_angle_l183_183440

theorem isosceles_triangle_base_angle (A B C : ℝ) (h_sum : A + B + C = 180) (h_iso : B = C) (h_one_angle : A = 80) : B = 50 :=
sorry

end isosceles_triangle_base_angle_l183_183440


namespace work_duration_B_l183_183731

theorem work_duration_B (x : ℕ) (h : x = 10) : 
  (x * (1 / 15 : ℚ)) + (2 * (1 / 6 : ℚ)) = 1 := 
by 
  rw [h]
  sorry

end work_duration_B_l183_183731


namespace george_and_hannah_received_A_grades_l183_183916

-- Define students as propositions
variables (Elena Fred George Hannah : Prop)

-- Define the conditions
def condition1 : Prop := Elena → Fred
def condition2 : Prop := Fred → George
def condition3 : Prop := George → Hannah
def condition4 : Prop := ∃ A1 A2 : Prop, A1 ∧ A2 ∧ (A1 ≠ A2) ∧ (A1 = George ∨ A1 = Hannah) ∧ (A2 = George ∨ A2 = Hannah)

-- The theorem to be proven: George and Hannah received A grades
theorem george_and_hannah_received_A_grades :
  condition1 Elena Fred →
  condition2 Fred George →
  condition3 George Hannah →
  condition4 George Hannah :=
by
  sorry

end george_and_hannah_received_A_grades_l183_183916


namespace compute_expression_l183_183693

theorem compute_expression : (5 + 9)^2 + Real.sqrt (5^2 + 9^2) = 196 + Real.sqrt 106 := 
by sorry

end compute_expression_l183_183693


namespace midline_equation_l183_183708

theorem midline_equation (a b : ℝ) (K1 K2 : ℝ)
  (h1 : K1^2 = (a^2) / 4 + b^2)
  (h2 : K2^2 = a^2 + (b^2) / 4) :
  16 * K2^2 - 4 * K1^2 = 15 * a^2 :=
by
  sorry

end midline_equation_l183_183708


namespace problem_statement_l183_183361

def diamond (x y : ℝ) : ℝ := (x + y) ^ 2 * (x - y) ^ 2

theorem problem_statement : diamond 2 (diamond 3 4) = 5745329 := by
  sorry

end problem_statement_l183_183361


namespace scientific_notation_l183_183336

def z := 10374 * 10^9

theorem scientific_notation (a : ℝ) (n : ℤ) (h₁ : 1 ≤ |a|) (h₂ : |a| < 10) (h₃ : a * 10^n = z) : a = 1.04 ∧ n = 13 := sorry

end scientific_notation_l183_183336


namespace polynomial_divisibility_l183_183679

theorem polynomial_divisibility (r s : ℝ) :
  (∀ x, 10 * x^4 - 15 * x^3 - 55 * x^2 + 85 * x - 51 = 10 * (x - r)^2 * (x - s)) →
  r = 3 / 2 ∧ s = -5 / 2 :=
by
  intros h
  sorry

end polynomial_divisibility_l183_183679


namespace mean_first_set_l183_183698

noncomputable def mean (s : List ℚ) : ℚ := s.sum / s.length

theorem mean_first_set (x : ℚ) (h : mean [128, 255, 511, 1023, x] = 423) :
  mean [28, x, 42, 78, 104] = 90 :=
sorry

end mean_first_set_l183_183698


namespace repeating_decimals_sum_is_fraction_l183_183932

-- Define the repeating decimals as fractions
def x : ℚ := 1 / 3
def y : ℚ := 2 / 99

-- Define the sum of the repeating decimals
def sum := x + y

-- State the theorem
theorem repeating_decimals_sum_is_fraction :
  sum = 35 / 99 := sorry

end repeating_decimals_sum_is_fraction_l183_183932


namespace radian_measure_of_240_degrees_l183_183832

theorem radian_measure_of_240_degrees : (240 * (π / 180) = 4 * π / 3) := by
  sorry

end radian_measure_of_240_degrees_l183_183832


namespace radius_of_circle_nearest_integer_l183_183107

theorem radius_of_circle_nearest_integer (θ L : ℝ) (hθ : θ = 300) (hL : L = 2000) : 
  abs ((1200 / (Real.pi)) - 382) < 1 := 
by {
  sorry
}

end radius_of_circle_nearest_integer_l183_183107


namespace g_is_even_l183_183236

noncomputable def g (x : ℝ) : ℝ := 4^(x^2 - 3) - 2 * |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l183_183236


namespace smallest_b_l183_183808

theorem smallest_b (b : ℕ) : 
  (b % 3 = 2) ∧ (b % 4 = 3) ∧ (b % 5 = 4) ∧ (b % 7 = 6) ↔ b = 419 :=
by sorry

end smallest_b_l183_183808


namespace gcd_of_17934_23526_51774_l183_183834

-- Define the three integers
def a : ℕ := 17934
def b : ℕ := 23526
def c : ℕ := 51774

-- State the theorem
theorem gcd_of_17934_23526_51774 : Int.gcd a (Int.gcd b c) = 2 := by
  sorry

end gcd_of_17934_23526_51774_l183_183834


namespace inequality_solution_set_l183_183048

theorem inequality_solution_set (x : ℝ) :
  (4 * x - 2 ≥ 3 * (x - 1)) ∧ ((x - 5) / 2 > x - 4) ↔ (-1 ≤ x ∧ x < 3) := 
by sorry

end inequality_solution_set_l183_183048


namespace circular_garden_radius_l183_183903

theorem circular_garden_radius (r : ℝ) (h1 : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) : r = 12 :=
by sorry

end circular_garden_radius_l183_183903


namespace largest_integer_of_four_l183_183506

theorem largest_integer_of_four (A B C D : ℤ)
  (h_diff: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_order: A < B ∧ B < C ∧ C < D)
  (h_avg: (A + B + C + D) / 4 = 74)
  (h_A_min: A ≥ 29) : D = 206 :=
by
  sorry

end largest_integer_of_four_l183_183506


namespace one_liter_fills_five_cups_l183_183436

-- Define the problem conditions and question in Lean 4
def one_liter_milliliters : ℕ := 1000
def cup_volume_milliliters : ℕ := 200

theorem one_liter_fills_five_cups : one_liter_milliliters / cup_volume_milliliters = 5 := 
by 
  sorry -- proof skipped

end one_liter_fills_five_cups_l183_183436


namespace train_cross_time_l183_183491

def length_of_train : Float := 135.0 -- in meters
def speed_of_train_kmh : Float := 45.0 -- in kilometers per hour
def length_of_bridge : Float := 240.03 -- in meters

def speed_of_train_ms : Float := speed_of_train_kmh * 1000.0 / 3600.0

def total_distance : Float := length_of_train + length_of_bridge

def time_to_cross : Float := total_distance / speed_of_train_ms

theorem train_cross_time : time_to_cross = 30.0024 :=
by
  sorry

end train_cross_time_l183_183491


namespace part1_part2_part3_l183_183826

section Part1

variables (a b : Real)

theorem part1 : 2 * (a + b)^2 - 8 * (a + b)^2 + 3 * (a + b)^2 = -3 * (a + b)^2 :=
by
  sorry

end Part1

section Part2

variables (x y : Real)

theorem part2 (h : x^2 + 2 * y = 4) : -3 * x^2 - 6 * y + 17 = 5 :=
by
  sorry

end Part2

section Part3

variables (a b c d : Real)

theorem part3 (h1 : a - 3 * b = 3) (h2 : 2 * b - c = -5) (h3 : c - d = 9) :
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by
  sorry

end Part3

end part1_part2_part3_l183_183826


namespace sin_225_cos_225_l183_183801

noncomputable def sin_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2

noncomputable def cos_225_eq_neg_sqrt2_div_2 : Prop :=
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2

theorem sin_225 : sin_225_eq_neg_sqrt2_div_2 := by
  sorry

theorem cos_225 : cos_225_eq_neg_sqrt2_div_2 := by
  sorry

end sin_225_cos_225_l183_183801


namespace ab_power_2023_l183_183964

theorem ab_power_2023 (a b : ℤ) (h : |a + 2| + (b - 1) ^ 2 = 0) : (a + b) ^ 2023 = -1 :=
by
  sorry

end ab_power_2023_l183_183964


namespace evaluate_expression_l183_183326

noncomputable def g (x : ℝ) : ℝ := x^3 + 3*x + 2*Real.sqrt x

theorem evaluate_expression : 
  3 * g 3 - 2 * g 9 = -1416 + 6 * Real.sqrt 3 :=
by
  sorry

end evaluate_expression_l183_183326


namespace oliver_final_amount_l183_183353

variable (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ)

def final_amount_after_transactions (initial_amount saved_amount spent_on_frisbee spent_on_puzzle gift : ℕ) : ℕ :=
  initial_amount + saved_amount - (spent_on_frisbee + spent_on_puzzle) + gift

theorem oliver_final_amount :
  final_amount_after_transactions 9 5 4 3 8 = 15 :=
by
  -- We can fill in the exact calculations here to provide the proof.
  sorry

end oliver_final_amount_l183_183353


namespace sum_a1_a4_l183_183659

variables (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := n^2 + n + 1

-- Define the individual terms of the sequence
def term_seq (n : ℕ) : ℕ :=
if n = 1 then sum_seq 1 else sum_seq n - sum_seq (n - 1)

-- Prove that the sum of the first and fourth terms equals 11
theorem sum_a1_a4 : 
  (term_seq 1) + (term_seq 4) = 11 :=
by
  -- to be completed with proof steps
  sorry

end sum_a1_a4_l183_183659


namespace range_of_x_l183_183343

noncomputable def A (x : ℝ) : ℤ := Int.ceil x

theorem range_of_x (x : ℝ) (h₁ : 0 < x) (h₂ : A (2 * x * A x) = 5) : 1 < x ∧ x ≤ 5 / 4 := 
sorry

end range_of_x_l183_183343


namespace fraction_simplification_l183_183092

theorem fraction_simplification (a : ℝ) (h1 : a > 1) (h2 : a ≠ 2 / Real.sqrt 3) : 
  (a^3 - 3 * a^2 + 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) / 
  (a^3 + 3 * a^2 - 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) = 
  ((a - 2) * Real.sqrt (a + 1)) / ((a + 2) * Real.sqrt (a - 1)) :=
by
  sorry

end fraction_simplification_l183_183092


namespace area_of_right_triangle_with_incircle_l183_183337

theorem area_of_right_triangle_with_incircle (a b c r : ℝ) :
  (a = 6 + r) → 
  (b = 7 + r) → 
  (c = 13) → 
  (a^2 + b^2 = c^2) →
  (2 * r^2 + 26 * r = 84) →
  (area = 1/2 * ((6 + r) * (7 + r))) →
  area = 42 := 
by 
  sorry

end area_of_right_triangle_with_incircle_l183_183337


namespace rationalize_denominator_l183_183911

theorem rationalize_denominator : (7 / Real.sqrt 147) = (Real.sqrt 3 / 3) :=
by
  sorry

end rationalize_denominator_l183_183911


namespace barium_oxide_moles_l183_183224

noncomputable def moles_of_bao_needed (mass_H2O : ℝ) (molar_mass_H2O : ℝ) : ℝ :=
  mass_H2O / molar_mass_H2O

theorem barium_oxide_moles :
  moles_of_bao_needed 54 18.015 = 3 :=
by
  unfold moles_of_bao_needed
  norm_num
  sorry

end barium_oxide_moles_l183_183224


namespace remainder_div_power10_l183_183458

theorem remainder_div_power10 (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, (10^n - 1) % 37 = k^2 := by
  sorry

end remainder_div_power10_l183_183458


namespace perimeter_of_unshaded_rectangle_l183_183718

theorem perimeter_of_unshaded_rectangle (length width height base area shaded_area perimeter : ℝ)
  (h1 : length = 12)
  (h2 : width = 9)
  (h3 : height = 3)
  (h4 : base = (2 * shaded_area) / height)
  (h5 : shaded_area = 18)
  (h6 : perimeter = 2 * ((length - base) + width))
  : perimeter = 24 := by
  sorry

end perimeter_of_unshaded_rectangle_l183_183718


namespace probability_divisible_by_8_l183_183069

-- Define the problem conditions
def is_8_sided_die (n : ℕ) : Prop := n = 6
def roll_dice (m : ℕ) : Prop := m = 8

-- Define the main proof statement
theorem probability_divisible_by_8 (n m : ℕ) (hn : is_8_sided_die n) (hm : roll_dice m) :  
  (35 : ℚ) / 36 = 
  (1 - ((1/2) ^ m + 28 * ((1/n) ^ 2 * ((1/2) ^ 6))) : ℚ) :=
by
  sorry

end probability_divisible_by_8_l183_183069


namespace scientific_notation_of_number_l183_183509

theorem scientific_notation_of_number (num : ℝ) (a b: ℝ) : 
  num = 0.0000046 ∧ 
  (a = 46 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -6 ∨ 
   a = 0.46 ∧ b = -5) → 
  a = 4.6 ∧ b = -6 :=
by
  sorry

end scientific_notation_of_number_l183_183509


namespace solved_just_B_is_six_l183_183609

variables (a b c d e f g : ℕ)

-- Conditions given
axiom total_competitors : a + b + c + d + e + f + g = 25
axiom twice_as_many_solved_B : b + d = 2 * (c + d)
axiom only_A_one_more : a = 1 + (e + f + g)
axiom A_equals_B_plus_C : a = b + c

-- Prove that the number of competitors solving just problem B is 6.
theorem solved_just_B_is_six : b = 6 :=
by
  sorry

end solved_just_B_is_six_l183_183609


namespace cost_50_jasmines_discounted_l183_183287

variable (cost_per_8_jasmines : ℝ) (num_jasmines : ℕ) (discount : ℝ)
variable (proportional : Prop) (c_50_jasmines : ℝ)

-- Given the cost of a bouquet with 8 jasmines
def cost_of_8_jasmines : ℝ := 24

-- Given the price is directly proportional to the number of jasmines
def price_proportional := ∀ (n : ℕ), num_jasmines = 8 → proportional

-- Given the bouquet with 50 jasmines
def num_jasmines_50 : ℕ := 50

-- Applying a 10% discount
def ten_percent_discount : ℝ := 0.9

-- Prove the cost of the bouquet with 50 jasmines after a 10% discount
theorem cost_50_jasmines_discounted :
  proportional ∧ (c_50_jasmines = (cost_of_8_jasmines / 8) * num_jasmines_50) →
  (c_50_jasmines * ten_percent_discount) = 135 :=
by
  sorry

end cost_50_jasmines_discounted_l183_183287


namespace cost_price_of_each_watch_l183_183887

-- Define the given conditions.
def sold_at_loss (C : ℝ) := 0.925 * C
def total_transaction_price (C : ℝ) := 3 * C * 1.053
def sold_for_more (C : ℝ) := 0.925 * C + 265

-- State the theorem to prove the cost price of each watch.
theorem cost_price_of_each_watch (C : ℝ) :
  3 * sold_for_more C = total_transaction_price C → C = 2070.31 :=
by
  intros h
  sorry

end cost_price_of_each_watch_l183_183887


namespace sand_cake_probability_is_12_percent_l183_183730

def total_days : ℕ := 5
def ham_days : ℕ := 3
def cake_days : ℕ := 1

-- Probability of packing a ham sandwich on any given day
def prob_ham_sandwich : ℚ := ham_days / total_days

-- Probability of packing a piece of cake on any given day
def prob_cake : ℚ := cake_days / total_days

-- Calculate the combined probability that Karen packs a ham sandwich and cake on the same day
def combined_probability : ℚ := prob_ham_sandwich * prob_cake

-- Convert the combined probability to a percentage
def combined_probability_as_percentage : ℚ := combined_probability * 100

-- The proof problem to show that the probability that Karen packs a ham sandwich and cake on the same day is 12%
theorem sand_cake_probability_is_12_percent : combined_probability_as_percentage = 12 := 
  by sorry

end sand_cake_probability_is_12_percent_l183_183730


namespace problem_statement_l183_183768

noncomputable def M (x y : ℝ) : ℝ := max x y
noncomputable def m (x y : ℝ) : ℝ := min x y

theorem problem_statement {p q r s t : ℝ} (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) :
  M (M p (m q r)) (m s (m p t)) = q :=
by
  sorry

end problem_statement_l183_183768


namespace ratio_proof_l183_183631

theorem ratio_proof (a b c d : ℝ) (h1 : b = 3 * a) (h2 : c = 4 * b) (h3 : d = 2 * b - a) :
  (a + b + d) / (b + c + d) = 9 / 20 :=
by sorry

end ratio_proof_l183_183631


namespace reciprocal_sum_of_roots_l183_183754

theorem reciprocal_sum_of_roots :
  (∃ m n : ℝ, (m^2 + 2 * m - 3 = 0) ∧ (n^2 + 2 * n - 3 = 0) ∧ m ≠ n) →
  (∃ m n : ℝ, (1/m + 1/n = 2/3)) :=
by
  sorry

end reciprocal_sum_of_roots_l183_183754


namespace range_of_a_l183_183424

variable (a : ℝ)

theorem range_of_a
  (h : ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) :
  a < -1 ∨ a > 1 :=
by {
  sorry
}

end range_of_a_l183_183424


namespace p_is_necessary_but_not_sufficient_for_q_l183_183828

variable (x : ℝ)
def p := |x| ≤ 2
def q := 0 ≤ x ∧ x ≤ 2

theorem p_is_necessary_but_not_sufficient_for_q : (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬ q x := by
  sorry

end p_is_necessary_but_not_sufficient_for_q_l183_183828


namespace simplify_expression_l183_183034

theorem simplify_expression :
  (144 / 12) * (5 / 90) * (9 / 3) * 2 = 4 := by
  sorry

end simplify_expression_l183_183034


namespace ratio_of_x_to_y_l183_183984

theorem ratio_of_x_to_y (x y : ℚ) (h : (2 * x - 3 * y) / (x + 2 * y) = 5 / 4) : x / y = 22 / 3 := by
  sorry

end ratio_of_x_to_y_l183_183984


namespace gravity_anomaly_l183_183800

noncomputable def gravity_anomaly_acceleration
  (α : ℝ) (v₀ : ℝ) (g : ℝ) (S : ℝ) (g_a : ℝ) : Prop :=
  α = 30 ∧ v₀ = 10 ∧ g = 10 ∧ S = 3 * Real.sqrt 3 → g_a = 250

theorem gravity_anomaly (α v₀ g S g_a : ℝ) : gravity_anomaly_acceleration α v₀ g S g_a :=
by
  intro h
  sorry

end gravity_anomaly_l183_183800


namespace gcd_15_70_l183_183752

theorem gcd_15_70 : Int.gcd 15 70 = 5 := by
  sorry

end gcd_15_70_l183_183752


namespace cylinder_h_over_r_equals_one_l183_183822

theorem cylinder_h_over_r_equals_one
  (A : ℝ) (r h : ℝ)
  (h_surface_area : A = 2 * π * r^2 + 2 * π * r * h)
  (V : ℝ := π * r^2 * h)
  (max_V : ∀ r' h', (A = 2 * π * r'^2 + 2 * π * r' * h') → (π * r'^2 * h' ≤ V) → (r' = r ∧ h' = h)) :
  h / r = 1 := by
sorry

end cylinder_h_over_r_equals_one_l183_183822


namespace terminal_side_in_second_quadrant_l183_183614

theorem terminal_side_in_second_quadrant (α : ℝ) (h : (Real.tan α < 0) ∧ (Real.cos α < 0)) :
  (2 < α / (π / 2)) ∧ (α / (π / 2) < 3) :=
by
  sorry

end terminal_side_in_second_quadrant_l183_183614


namespace height_of_parallelogram_l183_183943

theorem height_of_parallelogram (A B h : ℝ) (hA : A = 72) (hB : B = 12) (h_area : A = B * h) : h = 6 := by
  sorry

end height_of_parallelogram_l183_183943


namespace find_people_who_own_only_cats_l183_183414

variable (C : ℕ)

theorem find_people_who_own_only_cats
  (ownsOnlyDogs : ℕ)
  (ownsCatsAndDogs : ℕ)
  (ownsCatsDogsSnakes : ℕ)
  (totalPetOwners : ℕ)
  (h1 : ownsOnlyDogs = 15)
  (h2 : ownsCatsAndDogs = 5)
  (h3 : ownsCatsDogsSnakes = 3)
  (h4 : totalPetOwners = 59) :
  C = 36 :=
by
  sorry

end find_people_who_own_only_cats_l183_183414


namespace roots_transformation_l183_183262

-- Given polynomial
def poly1 (x : ℝ) : ℝ := x^3 - 3*x^2 + 8

-- Polynomial with roots 3*r1, 3*r2, 3*r3
def poly2 (x : ℝ) : ℝ := x^3 - 9*x^2 + 216

-- Theorem stating the equivalence
theorem roots_transformation (r1 r2 r3 : ℝ) 
  (h : ∀ x, poly1 x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) :
  ∀ x, poly2 x = 0 ↔ x = 3*r1 ∨ x = 3*r2 ∨ x = 3*r3 :=
sorry

end roots_transformation_l183_183262


namespace find_n_l183_183428

theorem find_n (x n : ℝ) (h_x : x = 0.5) : (9 / (1 + n / x) = 1) → n = 4 := 
by
  intro h
  have h_x_eq : x = 0.5 := h_x
  -- Proof content here covering the intermediary steps
  sorry

end find_n_l183_183428


namespace function_odd_on_domain_l183_183619

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

theorem function_odd_on_domain :
  ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x :=
by
  intros x h
  sorry

end function_odd_on_domain_l183_183619


namespace verify_a_eq_x0_verify_p_squared_ge_4x0q_l183_183227

theorem verify_a_eq_x0 (p q x0 a b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + a * x + b)) : 
  a = x0 :=
by
  sorry

theorem verify_p_squared_ge_4x0q (p q x0 b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + x0 * x + b)) : 
  p^2 ≥ 4 * x0 * q :=
by
  sorry

end verify_a_eq_x0_verify_p_squared_ge_4x0q_l183_183227


namespace minimum_value_fraction_l183_183240

theorem minimum_value_fraction (m n : ℝ) (h_line : 2 * m * 2 + n * 2 - 4 = 0) (h_pos_m : m > 0) (h_pos_n : n > 0) :
  (m + n / 2 = 1) -> ∃ (m n : ℝ), (m > 0 ∧ n > 0) ∧ (3 + 2 * Real.sqrt 2 ≤ (1 / m + 4 / n)) :=
by
  sorry

end minimum_value_fraction_l183_183240


namespace Gabrielle_sells_8_crates_on_Wednesday_l183_183948

-- Definitions based on conditions from part a)
def crates_sold_on_Monday := 5
def crates_sold_on_Tuesday := 2 * crates_sold_on_Monday
def crates_sold_on_Thursday := crates_sold_on_Tuesday / 2
def total_crates_sold := 28
def crates_sold_on_Wednesday := total_crates_sold - (crates_sold_on_Monday + crates_sold_on_Tuesday + crates_sold_on_Thursday)

-- The theorem to prove the question == answer given conditions
theorem Gabrielle_sells_8_crates_on_Wednesday : crates_sold_on_Wednesday = 8 := by
  sorry

end Gabrielle_sells_8_crates_on_Wednesday_l183_183948


namespace evaluate_expr_correct_l183_183941

def evaluate_expr : Prop :=
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25)

theorem evaluate_expr_correct : evaluate_expr :=
by
  sorry

end evaluate_expr_correct_l183_183941


namespace oranges_in_bin_l183_183380

theorem oranges_in_bin (initial_oranges thrown_out new_oranges : ℕ) (h1 : initial_oranges = 34) (h2 : thrown_out = 20) (h3 : new_oranges = 13) :
  (initial_oranges - thrown_out + new_oranges = 27) :=
by
  sorry

end oranges_in_bin_l183_183380


namespace smallest_s_for_F_l183_183139

def F (a b c d : ℕ) : ℕ := a * b^(c^d)

theorem smallest_s_for_F :
  ∃ s : ℕ, F s s 2 2 = 65536 ∧ ∀ t : ℕ, F t t 2 2 = 65536 → s ≤ t :=
sorry

end smallest_s_for_F_l183_183139


namespace value_of_a_l183_183650

theorem value_of_a (a : ℚ) (h : 2 * a + a / 2 = 9 / 2) : a = 9 / 5 :=
by
  sorry

end value_of_a_l183_183650


namespace greatest_x_for_4x_in_factorial_21_l183_183332

-- Definition and theorem to state the problem mathematically
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_x_for_4x_in_factorial_21 : ∃ x : ℕ, (4^x ∣ factorial 21) ∧ ∀ y : ℕ, (4^y ∣ factorial 21) → y ≤ 9 :=
by
  sorry

end greatest_x_for_4x_in_factorial_21_l183_183332


namespace percent_of_x_is_y_l183_183907

theorem percent_of_x_is_y (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y = 0.3333 * x :=
by
  sorry

end percent_of_x_is_y_l183_183907


namespace grandson_age_l183_183861

-- Define the ages of Markus, his son, and his grandson
variables (M S G : ℕ)

-- Conditions given in the problem
axiom h1 : M = 2 * S
axiom h2 : S = 2 * G
axiom h3 : M + S + G = 140

-- Theorem to prove that the age of Markus's grandson is 20 years
theorem grandson_age : G = 20 :=
by
  sorry

end grandson_age_l183_183861


namespace largest_of_A_B_C_l183_183390

noncomputable def A : ℝ := (2010 / 2009) + (2010 / 2011)
noncomputable def B : ℝ := (2010 / 2011) + (2012 / 2011)
noncomputable def C : ℝ := (2011 / 2010) + (2011 / 2012)

theorem largest_of_A_B_C : B > A ∧ B > C := by
  sorry

end largest_of_A_B_C_l183_183390


namespace problem_statement_l183_183976

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 8) * x

theorem problem_statement
  (f : ℝ → ℝ → ℝ)
  (sol_set : Set ℝ)
  (cond1 : ∀ a : ℝ, sol_set = {x : ℝ | -1 ≤ x ∧ x ≤ 5} → ∀ x : ℝ, f x a ≤ 5 ↔ x ∈ sol_set)
  (cond2 : ∀ x : ℝ, ∀ m : ℝ, f x 2 ≥ m^2 - 4 * m - 9) :
  (∃ a : ℝ, a = 2) ∧ (∀ m : ℝ, -1 ≤ m ∧ m ≤ 5) :=
by
  sorry

end problem_statement_l183_183976


namespace find_constants_l183_183561

noncomputable def f (a b x : ℝ) : ℝ :=
(a * x + b) / (x + 1)

theorem find_constants (a b : ℝ) (x : ℝ) (h : x ≠ -1) : 
  (f a b (f a b x) = x) → (a = -1 ∧ ∀ b, ∃ c : ℝ, b = c) :=
by 
  sorry

end find_constants_l183_183561


namespace power_expression_l183_183435

theorem power_expression (a b : ℕ) (h1 : a = 12) (h2 : b = 18) : (3^a * 3^b) = (243^6) :=
by
  let c := 3
  have h3 : a + b = 30 := by simp [h1, h2]
  have h4 : 3^(a + b) = 3^30 := by rw [h3]
  have h5 : 3^30 = 243^6 := by norm_num
  sorry  -- skip other detailed steps

end power_expression_l183_183435


namespace distance_between_parallel_lines_l183_183946

theorem distance_between_parallel_lines (A B C1 C2 : ℝ) (hA : A = 2) (hB : B = 4)
  (hC1 : C1 = -8) (hC2 : C2 = 7) : 
  (|C2 - C1| / (Real.sqrt (A^2 + B^2)) = 3 * Real.sqrt 5 / 2) :=
by
  rw [hA, hB, hC1, hC2]
  sorry

end distance_between_parallel_lines_l183_183946


namespace definite_integral_value_l183_183030

theorem definite_integral_value :
  (∫ x in (0 : ℝ)..Real.arctan (1/3), (8 + Real.tan x) / (18 * Real.sin x^2 + 2 * Real.cos x^2)) 
  = (Real.pi / 3) + (Real.log 2 / 36) :=
by
  -- Proof to be provided
  sorry

end definite_integral_value_l183_183030


namespace tree_heights_l183_183630

theorem tree_heights (h : ℕ) (ratio : 5 / 7 = (h - 20) / h) : h = 70 :=
sorry

end tree_heights_l183_183630


namespace original_number_l183_183994

theorem original_number (h : 2.04 / 1.275 = 1.6) : 204 / 12.75 = 16 := 
by
  sorry

end original_number_l183_183994


namespace selling_price_is_1260_l183_183188

-- Definitions based on conditions
def purchase_price : ℕ := 900
def repair_cost : ℕ := 300
def gain_percent : ℕ := 5 -- percentage as a natural number

-- Known variables
def total_cost : ℕ := purchase_price + repair_cost
def gain_amount : ℕ := (gain_percent * total_cost) / 100
def selling_price : ℕ := total_cost + gain_amount

-- The theorem we want to prove
theorem selling_price_is_1260 : selling_price = 1260 := by
  sorry

end selling_price_is_1260_l183_183188


namespace number_of_terms_in_sequence_l183_183538

def arithmetic_sequence_terms (a d l : ℕ) : ℕ :=
  (l - a) / d + 1

theorem number_of_terms_in_sequence : arithmetic_sequence_terms 1 4 57 = 15 :=
by {
  sorry
}

end number_of_terms_in_sequence_l183_183538


namespace age_difference_l183_183109

variable (A B C : ℕ)

def condition1 := C = B / 2
def condition2 := A + B + C = 22
def condition3 := B = 8

theorem age_difference (h1 : condition1 C B)
                       (h2 : condition2 A B C) 
                       (h3 : condition3 B) : A - B = 2 := by
  sorry

end age_difference_l183_183109


namespace range_of_m_l183_183026

theorem range_of_m (m : ℝ) :
  let A := {x : ℝ | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
  let B := {x : ℝ | 1 ≤ x ∧ x ≤ 10}
  (A ⊆ B) ↔ (m ≤ (11:ℝ)/3) :=
by
  sorry

end range_of_m_l183_183026


namespace face_opposite_one_is_three_l183_183746

def faces : List ℕ := [1, 2, 3, 4, 5, 6]

theorem face_opposite_one_is_three (x : ℕ) (h1 : x ∈ faces) (h2 : x ≠ 1) : x = 3 :=
by
  sorry

end face_opposite_one_is_three_l183_183746


namespace length_more_than_breadth_by_200_percent_l183_183060

noncomputable def length: ℝ := 19.595917942265423
noncomputable def total_cost: ℝ := 640
noncomputable def rate_per_sq_meter: ℝ := 5

theorem length_more_than_breadth_by_200_percent
  (area : ℝ := total_cost / rate_per_sq_meter)
  (breadth : ℝ := area / length) :
  ((length - breadth) / breadth) * 100 = 200 := by
  have h1 : area = 128 := by sorry
  have h2 : breadth = 128 / 19.595917942265423 := by sorry
  rw [h1, h2]
  sorry

end length_more_than_breadth_by_200_percent_l183_183060


namespace train_length_55_meters_l183_183389

noncomputable def V_f := 47 * 1000 / 3600 -- Speed of the faster train in m/s
noncomputable def V_s := 36 * 1000 / 3600 -- Speed of the slower train in m/s
noncomputable def t := 36 -- Time in seconds

theorem train_length_55_meters (L : ℝ) (Vf : ℝ := V_f) (Vs : ℝ := V_s) (time : ℝ := t) :
  (2 * L = (Vf - Vs) * time) → L = 55 :=
by
  sorry

end train_length_55_meters_l183_183389


namespace cycling_race_difference_l183_183525

-- Define the speeds and time
def s_Chloe : ℝ := 18
def s_David : ℝ := 15
def t : ℝ := 5

-- Define the distances based on the speeds and time
def d_Chloe : ℝ := s_Chloe * t
def d_David : ℝ := s_David * t
def distance_difference : ℝ := d_Chloe - d_David

-- The theorem to prove
theorem cycling_race_difference :
  distance_difference = 15 := by
  sorry

end cycling_race_difference_l183_183525


namespace intersection_of_A_and_B_l183_183041

def setA : Set ℤ := {x | abs x < 4}
def setB : Set ℤ := {x | x - 1 ≥ 0}
def setIntersection : Set ℤ := {1, 2, 3}

theorem intersection_of_A_and_B : setA ∩ setB = setIntersection :=
by
  sorry

end intersection_of_A_and_B_l183_183041


namespace unique_solution_h_l183_183847

theorem unique_solution_h (h : ℝ) (hne_zero : h ≠ 0) :
  (∃! x : ℝ, (x - 3) / (h * x + 2) = x) ↔ h = 1 / 12 :=
by
  sorry

end unique_solution_h_l183_183847


namespace tan_theta_value_l183_183070

open Real

theorem tan_theta_value (θ : ℝ) (h : sin (θ / 2) - 2 * cos (θ / 2) = 0) : tan θ = -4 / 3 :=
sorry

end tan_theta_value_l183_183070


namespace average_of_w_x_z_eq_one_sixth_l183_183776

open Real

variable {w x y z t : ℝ}

theorem average_of_w_x_z_eq_one_sixth
  (h1 : 3 / w + 3 / x + 3 / z = 3 / (y + t))
  (h2 : w * x * z = y + t)
  (h3 : w * z + x * t + y * z = 3 * w + 3 * x + 3 * z) :
  (w + x + z) / 3 = 1 / 6 :=
by 
  sorry

end average_of_w_x_z_eq_one_sixth_l183_183776


namespace ratio_a_over_3_to_b_over_2_l183_183461

theorem ratio_a_over_3_to_b_over_2 (a b c : ℝ) (h1 : 2 * a = 3 * b) (h2 : c ≠ 0) (h3 : 3 * a + 2 * b = c) :
  (a / 3) / (b / 2) = 1 :=
sorry

end ratio_a_over_3_to_b_over_2_l183_183461


namespace systematic_sampling_starts_with_srs_l183_183293

-- Define the concept of systematic sampling
def systematically_sampled (initial_sampled: Bool) : Bool :=
  initial_sampled

-- Initial sample is determined by simple random sampling
def simple_random_sampling : Bool :=
  True

-- We need to prove that systematic sampling uses simple random sampling at the start
theorem systematic_sampling_starts_with_srs : systematically_sampled simple_random_sampling = True :=
by 
  sorry

end systematic_sampling_starts_with_srs_l183_183293


namespace trigonometric_identity_l183_183997

theorem trigonometric_identity (α : ℝ) (h : Real.tan (π + α) = 2) :
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π + α) - Real.cos (π + α)) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l183_183997


namespace geometric_sequence_solution_l183_183605

theorem geometric_sequence_solution (x : ℝ) (h : ∃ r : ℝ, 12 * r = x ∧ x * r = 3) : x = 6 ∨ x = -6 := 
by
  sorry

end geometric_sequence_solution_l183_183605


namespace B_is_criminal_l183_183368

-- Introduce the conditions
variable (A B C : Prop)  -- A, B, and C represent whether each individual is the criminal.

-- A says they did not commit the crime
axiom A_says_innocent : ¬A

-- Exactly one of A_says_innocent must hold true (A says ¬A, so B or C must be true)
axiom exactly_one_assertion_true : (¬A ∨ B ∨ C)

-- Problem Statement: Prove that B is the criminal
theorem B_is_criminal : B :=
by
  -- Solution steps would go here
  sorry

end B_is_criminal_l183_183368


namespace ratio_proof_l183_183081

variables (x y m n : ℝ)

def ratio_equation1 (x y m n : ℝ) : Prop :=
  (5 * x + 7 * y) / (3 * x + 2 * y) = m / n

def target_equation (x y m n : ℝ) : Prop :=
  (13 * x + 16 * y) / (2 * x + 5 * y) = (2 * m + n) / (m - n)

theorem ratio_proof (x y m n : ℝ) (h: ratio_equation1 x y m n) :
  target_equation x y m n :=
by
  sorry

end ratio_proof_l183_183081


namespace average_age_of_omi_kimiko_arlette_l183_183127

theorem average_age_of_omi_kimiko_arlette (Kimiko Omi Arlette : ℕ) (hK : Kimiko = 28) (hO : Omi = 2 * Kimiko) (hA : Arlette = (3 * Kimiko) / 4) : 
  (Omi + Kimiko + Arlette) / 3 = 35 := 
by
  sorry

end average_age_of_omi_kimiko_arlette_l183_183127


namespace crayons_in_drawer_before_l183_183824

theorem crayons_in_drawer_before (m c : ℕ) (h1 : m = 3) (h2 : c = 10) : c - m = 7 := 
  sorry

end crayons_in_drawer_before_l183_183824


namespace necessary_but_not_sufficient_condition_l183_183528

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ^ 2 = a n * a (n + 2)

theorem necessary_but_not_sufficient_condition
  (a : ℕ → ℝ) :
  condition a → ¬ is_geometric_sequence a :=
sorry

end necessary_but_not_sufficient_condition_l183_183528


namespace probability_of_all_girls_chosen_is_1_over_11_l183_183288

-- Defining parameters and conditions
def total_members : ℕ := 12
def boys : ℕ := 6
def girls : ℕ := 6
def chosen_members : ℕ := 3

-- Number of combinations to choose 3 members from 12
def total_combinations : ℕ := Nat.choose total_members chosen_members

-- Number of combinations to choose 3 girls from 6
def girl_combinations : ℕ := Nat.choose girls chosen_members

-- Probability is defined as the ratio of these combinations
def probability_all_girls_chosen : ℚ := girl_combinations / total_combinations

-- Proof Statement
theorem probability_of_all_girls_chosen_is_1_over_11 : probability_all_girls_chosen = 1 / 11 := by
  sorry -- Proof to be completed

end probability_of_all_girls_chosen_is_1_over_11_l183_183288


namespace jean_to_shirt_ratio_l183_183707

theorem jean_to_shirt_ratio (shirts_sold jeans_sold shirt_cost total_revenue: ℕ) (h1: shirts_sold = 20) (h2: jeans_sold = 10) (h3: shirt_cost = 10) (h4: total_revenue = 400) : 
(shirt_cost * shirts_sold + jeans_sold * ((total_revenue - (shirt_cost * shirts_sold)) / jeans_sold)) / (total_revenue - (shirt_cost * shirts_sold)) / jeans_sold = 2 := 
sorry

end jean_to_shirt_ratio_l183_183707


namespace midpoint_of_diagonal_l183_183758

-- Definition of the points
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (14, 9)

-- Statement about the midpoint of a diagonal in a rectangle
theorem midpoint_of_diagonal : 
  ∀ (x1 y1 x2 y2 : ℝ), (x1, y1) = point1 → (x2, y2) = point2 →
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  (midpoint_x, midpoint_y) = (8, 3) :=
by
  intros
  sorry

end midpoint_of_diagonal_l183_183758


namespace hats_in_shipment_l183_183833

theorem hats_in_shipment (H : ℝ) (h_condition : 0.75 * H = 90) : H = 120 :=
sorry

end hats_in_shipment_l183_183833


namespace kylie_coins_l183_183541

open Nat

theorem kylie_coins :
  ∀ (coins_from_piggy_bank coins_from_brother coins_from_father coins_given_to_friend total_coins_left : ℕ),
  coins_from_piggy_bank = 15 →
  coins_from_brother = 13 →
  coins_from_father = 8 →
  coins_given_to_friend = 21 →
  total_coins_left = coins_from_piggy_bank + coins_from_brother + coins_from_father - coins_given_to_friend →
  total_coins_left = 15 :=
by
  intros
  sorry

end kylie_coins_l183_183541


namespace number_of_full_rows_in_first_field_l183_183059

-- Define the conditions
def total_corn_cobs : ℕ := 116
def rows_in_second_field : ℕ := 16
def cobs_per_row : ℕ := 4
def cobs_in_second_field : ℕ := rows_in_second_field * cobs_per_row
def cobs_in_first_field : ℕ := total_corn_cobs - cobs_in_second_field

-- Define the theorem to be proven
theorem number_of_full_rows_in_first_field : 
  cobs_in_first_field / cobs_per_row = 13 :=
by
  sorry

end number_of_full_rows_in_first_field_l183_183059


namespace tree_height_by_time_boy_is_36_inches_l183_183243

noncomputable def final_tree_height : ℕ :=
  let T₀ := 16
  let B₀ := 24
  let Bₓ := 36
  let boy_growth := Bₓ - B₀
  let tree_growth := 2 * boy_growth
  T₀ + tree_growth

theorem tree_height_by_time_boy_is_36_inches :
  final_tree_height = 40 :=
by
  sorry

end tree_height_by_time_boy_is_36_inches_l183_183243


namespace range_of_x_div_y_l183_183362

theorem range_of_x_div_y {x y : ℝ} (hx : 1 < x ∧ x < 6) (hy : 2 < y ∧ y < 8) : 
  (1/8 < x / y) ∧ (x / y < 3) :=
sorry

end range_of_x_div_y_l183_183362


namespace proportion_x_l183_183789

theorem proportion_x (x : ℝ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end proportion_x_l183_183789


namespace pentagon_area_l183_183837

/-- Given a convex pentagon ABCDE where BE and CE are angle bisectors at vertices B and C 
respectively, with ∠A = 35 degrees, ∠D = 145 degrees, and the area of triangle BCE is 11, 
prove that the area of the pentagon ABCDE is 22. -/
theorem pentagon_area (ABCDE : Type) (angle_A : ℝ) (angle_D : ℝ) (area_BCE : ℝ)
  (h_A : angle_A = 35) (h_D : angle_D = 145) (h_area_BCE : area_BCE = 11) :
  ∃ (area_ABCDE : ℝ), area_ABCDE = 22 :=
by
  sorry

end pentagon_area_l183_183837


namespace find_a2_plus_b2_l183_183157

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 15) : a^2 + b^2 = 39 :=
by
  sorry

end find_a2_plus_b2_l183_183157


namespace find_n_in_arithmetic_sequence_l183_183588

noncomputable def arithmetic_sequence (n : ℕ) (a_n S_n d : ℕ) :=
  ∀ (a₁ : ℕ), 
    a₁ + d * (n - 1) = a_n →
    n * a₁ + d * n * (n - 1) / 2 = S_n

theorem find_n_in_arithmetic_sequence 
   (a_n S_n d n : ℕ) 
   (h_a_n : a_n = 44) 
   (h_S_n : S_n = 158) 
   (h_d : d = 3) :
   arithmetic_sequence n a_n S_n d → 
   n = 4 := 
by 
  sorry

end find_n_in_arithmetic_sequence_l183_183588


namespace find_n_l183_183271

theorem find_n (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) = 14) : n = 2 :=
sorry

end find_n_l183_183271


namespace percentage_of_rotten_oranges_l183_183726

-- Define the conditions
def total_oranges : ℕ := 600
def total_bananas : ℕ := 400
def rotten_bananas_percentage : ℝ := 0.08
def good_fruits_percentage : ℝ := 0.878

-- Define the proof problem
theorem percentage_of_rotten_oranges :
  let total_fruits := total_oranges + total_bananas
  let number_of_rotten_bananas := rotten_bananas_percentage * total_bananas
  let number_of_good_fruits := good_fruits_percentage * total_fruits
  let number_of_rotten_fruits := total_fruits - number_of_good_fruits
  let number_of_rotten_oranges := number_of_rotten_fruits - number_of_rotten_bananas
  let percentage_of_rotten_oranges := (number_of_rotten_oranges / total_oranges) * 100
  percentage_of_rotten_oranges = 15 := 
by
  sorry

end percentage_of_rotten_oranges_l183_183726


namespace pyramids_from_cuboid_l183_183938

-- Define the vertices of a cuboid
def vertices_of_cuboid : ℕ := 8

-- Define the edges of a cuboid
def edges_of_cuboid : ℕ := 12

-- Define the faces of a cuboid
def faces_of_cuboid : ℕ := 6

-- Define the combinatoric calculation
def combinations (n k : ℕ) : ℕ := (n.choose k)

-- Define the total number of tetrahedrons formed
def total_tetrahedrons : ℕ := combinations 7 3 - faces_of_cuboid * combinations 4 3

-- Define the expected result
def expected_tetrahedrons : ℕ := 106

-- The theorem statement to prove that the total number of tetrahedrons is 106
theorem pyramids_from_cuboid : total_tetrahedrons = expected_tetrahedrons :=
by
  sorry

end pyramids_from_cuboid_l183_183938


namespace value_at_17pi_over_6_l183_183357

variable (f : Real → Real)

-- Defining the conditions
def period (f : Real → Real) (T : Real) := ∀ x, f (x + T) = f x
def specific_value (f : Real → Real) (x : Real) (v : Real) := f x = v

-- The main theorem statement
theorem value_at_17pi_over_6 : 
  period f (π / 2) →
  specific_value f (π / 3) 1 →
  specific_value f (17 * π / 6) 1 :=
by
  intros h_period h_value
  sorry

end value_at_17pi_over_6_l183_183357


namespace suki_bags_l183_183264

theorem suki_bags (bag_weight_suki : ℕ) (bag_weight_jimmy : ℕ) (containers : ℕ) 
  (container_weight : ℕ) (num_bags_jimmy : ℝ) (num_containers : ℕ)
  (h1 : bag_weight_suki = 22) 
  (h2 : bag_weight_jimmy = 18) 
  (h3 : container_weight = 8) 
  (h4 : num_bags_jimmy = 4.5)
  (h5 : num_containers = 28) : 
  6 = ⌊(num_containers * container_weight - num_bags_jimmy * bag_weight_jimmy) / bag_weight_suki⌋ :=
by
  sorry

end suki_bags_l183_183264


namespace total_end_of_year_students_l183_183020

theorem total_end_of_year_students :
  let start_fourth := 33
  let start_fifth := 45
  let start_sixth := 28
  let left_fourth := 18
  let joined_fourth := 14
  let left_fifth := 12
  let joined_fifth := 20
  let left_sixth := 10
  let joined_sixth := 16

  let end_fourth := start_fourth - left_fourth + joined_fourth
  let end_fifth := start_fifth - left_fifth + joined_fifth
  let end_sixth := start_sixth - left_sixth + joined_sixth
  
  end_fourth + end_fifth + end_sixth = 116 := by
    sorry

end total_end_of_year_students_l183_183020


namespace problem_arith_sequences_l183_183396

theorem problem_arith_sequences (a b : ℕ → ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = b n + e)
  (h1 : a 1 = 25)
  (h2 : b 1 = 75)
  (h3 : a 2 + b 2 = 100) : 
  a 37 + b 37 = 100 := 
sorry

end problem_arith_sequences_l183_183396


namespace class_size_count_l183_183616

theorem class_size_count : 
  ∃ (n : ℕ), 
  n = 6 ∧ 
  (∀ (b g : ℕ), (2 < b ∧ b < 10) → (14 < g ∧ g < 23) → b + g > 25 → 
    ∃ (sizes : Finset ℕ), sizes.card = n ∧ 
    ∀ (s : ℕ), s ∈ sizes → (∃ (b' g' : ℕ), s = b' + g' ∧ s > 25)) :=
sorry

end class_size_count_l183_183616


namespace polynomial_g_l183_183016

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g_l183_183016


namespace closest_point_is_correct_l183_183756

def line_eq (x : ℝ) : ℝ := -3 * x + 5

def closest_point_on_line_to_given_point : Prop :=
  ∃ (x y : ℝ), y = line_eq x ∧ (x, y) = (17 / 10, -1 / 10) ∧
  (∀ (x' y' : ℝ), y' = line_eq x' → (x' - -4)^2 + (y' - -2)^2 ≥ (x - -4)^2 + (y - -2)^2)
  
theorem closest_point_is_correct : closest_point_on_line_to_given_point :=
sorry

end closest_point_is_correct_l183_183756


namespace megan_markers_l183_183152

def initial_markers : ℕ := 217
def roberts_gift : ℕ := 109
def sarah_took : ℕ := 35

def final_markers : ℕ := initial_markers + roberts_gift - sarah_took

theorem megan_markers : final_markers = 291 := by
  sorry

end megan_markers_l183_183152


namespace remaining_movies_to_watch_l183_183688

theorem remaining_movies_to_watch (total_movies watched_movies remaining_movies : ℕ) 
  (h1 : total_movies = 8) 
  (h2 : watched_movies = 4) 
  (h3 : remaining_movies = total_movies - watched_movies) : 
  remaining_movies = 4 := 
by
  sorry

end remaining_movies_to_watch_l183_183688


namespace cannot_divide_m_l183_183191

/-
  A proof that for the real number m = 2009^3 - 2009, 
  the number 2007 does not divide m.
-/

theorem cannot_divide_m (m : ℤ) (h : m = 2009^3 - 2009) : ¬ (2007 ∣ m) := 
by sorry

end cannot_divide_m_l183_183191


namespace correct_division_result_l183_183593

-- Define the conditions
def incorrect_divisor : ℕ := 48
def correct_divisor : ℕ := 36
def incorrect_quotient : ℕ := 24
def dividend : ℕ := incorrect_divisor * incorrect_quotient

-- Theorem statement
theorem correct_division_result : (dividend / correct_divisor) = 32 := by
  -- proof to be filled later
  sorry

end correct_division_result_l183_183593


namespace train_scheduled_speed_l183_183819

theorem train_scheduled_speed (a v : ℝ) (hv : 0 < v)
  (h1 : a / v - a / (v + 5) = 1 / 3)
  (h2 : a / (v - 5) - a / v = 5 / 12) : v = 45 :=
by
  sorry

end train_scheduled_speed_l183_183819


namespace necessary_and_sufficient_condition_l183_183077

variable (a b : ℝ)

theorem necessary_and_sufficient_condition:
  (ab + 1 ≠ a + b) ↔ (a ≠ 1 ∧ b ≠ 1) :=
sorry

end necessary_and_sufficient_condition_l183_183077


namespace diameter_increase_l183_183014

theorem diameter_increase (h : 0.628 = π * d) : d = 0.2 := 
sorry

end diameter_increase_l183_183014


namespace exercise_l183_183607

noncomputable def f : ℝ → ℝ := sorry

theorem exercise
  (h_even : ∀ x : ℝ, f (x + 1) = f (-(x + 1)))
  (h_increasing : ∀ ⦃a b : ℝ⦄, 1 ≤ a → a ≤ b → f a ≤ f b)
  (x1 x2 : ℝ)
  (h_x1_neg : x1 < 0)
  (h_x2_pos : x2 > 0)
  (h_sum_neg : x1 + x2 < -2) :
  f (-x1) > f (-x2) :=
sorry

end exercise_l183_183607


namespace total_length_proof_l183_183272

def length_of_first_tape : ℝ := 25
def overlap : ℝ := 3
def number_of_tapes : ℝ := 64

def total_tape_length : ℝ :=
  let effective_length_per_subsequent_tape := length_of_first_tape - overlap
  let length_of_remaining_tapes := effective_length_per_subsequent_tape * (number_of_tapes - 1)
  length_of_first_tape + length_of_remaining_tapes

theorem total_length_proof : total_tape_length = 1411 := by
  sorry

end total_length_proof_l183_183272


namespace total_birds_count_l183_183530

def blackbirds_per_tree : ℕ := 3
def tree_count : ℕ := 7
def magpies : ℕ := 13

theorem total_birds_count : (blackbirds_per_tree * tree_count) + magpies = 34 := by
  sorry

end total_birds_count_l183_183530


namespace distance_eq_l183_183658

open Real

variables (a b c d p q: ℝ)

-- Conditions from step a)
def onLine1 : Prop := b = (p-1)*a + q
def onLine2 : Prop := d = (p-1)*c + q

-- Theorem about the distance between points (a, b) and (c, d)
theorem distance_eq : 
  onLine1 a b p q → 
  onLine2 c d p q → 
  dist (a, b) (c, d) = abs (a - c) * sqrt (1 + (p - 1)^2) := 
by
  intros h1 h2
  sorry

end distance_eq_l183_183658


namespace find_a3_l183_183795

noncomputable def S (n : ℕ) : ℤ := 2 * n^2 - 1
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_a3 : a 3 = 10 := by
  sorry

end find_a3_l183_183795


namespace change_from_15_dollars_l183_183241

theorem change_from_15_dollars :
  let cost_eggs := 3
  let cost_pancakes := 2
  let cost_mugs_of_cocoa := 2 * 2
  let tax := 1
  let initial_cost := cost_eggs + cost_pancakes + cost_mugs_of_cocoa + tax
  let additional_pancakes := 2
  let additional_mug_of_cocoa := 2
  let additional_cost := additional_pancakes + additional_mug_of_cocoa
  let new_total_cost := initial_cost + additional_cost
  let payment := 15
  let change := payment - new_total_cost
  change = 1 :=
by
  sorry

end change_from_15_dollars_l183_183241


namespace elise_spent_on_puzzle_l183_183741

-- Definitions based on the problem conditions:
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def remaining_money : ℕ := 1

-- Prove that the amount spent on the puzzle is $18.
theorem elise_spent_on_puzzle : initial_money + saved_money - spent_on_comic - remaining_money = 18 := by
  sorry

end elise_spent_on_puzzle_l183_183741


namespace increasing_arithmetic_sequence_l183_183573

theorem increasing_arithmetic_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) = a n + 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end increasing_arithmetic_sequence_l183_183573


namespace range_of_a_for_inequality_l183_183533

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_for_inequality_l183_183533


namespace remaining_distance_l183_183445

-- Definitions for the given conditions
def total_distance : ℕ := 436
def first_stopover_distance : ℕ := 132
def second_stopover_distance : ℕ := 236

-- Prove that the remaining distance from the second stopover to the island is 68 miles.
theorem remaining_distance : total_distance - (first_stopover_distance + second_stopover_distance) = 68 := by
  -- The proof (details) will go here
  sorry

end remaining_distance_l183_183445


namespace range_of_a_l183_183184

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.exp x / x) - a * (x ^ 2)

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → (f a x1 / x2) - (f a x2 / x1) < 0) ↔ (a ≤ Real.exp 2 / 12) := by
  sorry

end range_of_a_l183_183184


namespace min_abs_sum_l183_183403

theorem min_abs_sum (x y : ℝ) : (|x - 1| + |x| + |y - 1| + |y + 1|) ≥ 3 :=
sorry

end min_abs_sum_l183_183403


namespace tangent_line_eq_at_P_tangent_lines_through_P_l183_183142

-- Define the function and point of interest
def f (x : ℝ) : ℝ := x^3
def P : ℝ × ℝ := (1, 1)

-- State the first part: equation of the tangent line at (1, 1)
theorem tangent_line_eq_at_P : 
  (∀ x : ℝ, x = P.1 → (f x) = P.2) → 
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ y = f x ∧ x = 1 → y = 3 * x - 2) :=
sorry

-- State the second part: equations of tangent lines passing through (1, 1)
theorem tangent_lines_through_P :
  (∀ x : ℝ, x = P.1 → (f x) = P.2) → 
  (∀ (x₀ y₀ : ℝ), y₀ = x₀^3 → 
  (x₀ ≠ 1 → ∃ k : ℝ,  k = 3 * (x₀)^2 → 
  (∀ x y : ℝ, y = k * (x - 1) + 1 ∧ y = f x₀ → y = y₀))) → 
  (∃ m b m' b' : ℝ, 
    (¬ ∀ x : ℝ, ∀ y : ℝ, (y = m *x + b ∧ y = 3 * x - 2) → y = m' * x + b') ∧ 
    ((m = 3 ∧ b = -2) ∧ (m' = 3/4 ∧ b' = 1/4))) :=
sorry

end tangent_line_eq_at_P_tangent_lines_through_P_l183_183142


namespace profit_percent_is_25_l183_183798

noncomputable def SP : ℝ := sorry
noncomputable def CP : ℝ := 0.80 * SP
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercent : ℝ := (Profit / CP) * 100

theorem profit_percent_is_25 :
  ProfitPercent = 25 :=
by
  sorry

end profit_percent_is_25_l183_183798


namespace number_of_quadruplets_l183_183330

variables (a b c : ℕ)

theorem number_of_quadruplets (h1 : 2 * a + 3 * b + 4 * c = 1200)
                             (h2 : b = 3 * c)
                             (h3 : a = 2 * b) :
  4 * c = 192 :=
by
  sorry

end number_of_quadruplets_l183_183330


namespace aquarium_height_l183_183796

theorem aquarium_height (h : ℝ) (V : ℝ) (final_volume : ℝ) :
  let length := 4
  let width := 6
  let halfway_volume := (length * width * h) / 2
  let spilled_volume := halfway_volume / 2
  let tripled_volume := 3 * spilled_volume
  tripled_volume = final_volume →
  final_volume = 54 →
  h = 3 := by
  intros
  sorry

end aquarium_height_l183_183796


namespace find_B_value_l183_183519

-- Define the polynomial and conditions
def polynomial (A B : ℤ) (z : ℤ) : ℤ := z^4 - 12 * z^3 + A * z^2 + B * z + 36

-- Define roots and their properties according to the conditions
def roots_sum_to_twelve (r1 r2 r3 r4 : ℕ) : Prop := r1 + r2 + r3 + r4 = 12

-- The final statement to prove
theorem find_B_value (r1 r2 r3 r4 : ℕ) (A B : ℤ) (h_sum : roots_sum_to_twelve r1 r2 r3 r4)
    (h_pos : r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0) 
    (h_poly : polynomial A B = (z^4 - 12*z^3 + Az^2 + Bz + 36)) :
    B = -96 :=
    sorry

end find_B_value_l183_183519


namespace fiona_reaches_pad_thirteen_without_predators_l183_183154

noncomputable def probability_reach_pad_thirteen : ℚ := sorry

theorem fiona_reaches_pad_thirteen_without_predators :
  probability_reach_pad_thirteen = 3 / 2048 :=
sorry

end fiona_reaches_pad_thirteen_without_predators_l183_183154


namespace union_complement_eq_universal_l183_183716

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5}

-- The proof problem
theorem union_complement_eq_universal :
  U = A ∪ (U \ B) :=
by
  sorry

end union_complement_eq_universal_l183_183716


namespace calculate_sum_and_double_l183_183409

theorem calculate_sum_and_double :
  2 * (1324 + 4231 + 3124 + 2413) = 22184 :=
by
  sorry

end calculate_sum_and_double_l183_183409


namespace smallest_angle_WYZ_l183_183863

-- Define the given angle measures.
def angle_XYZ : ℝ := 40
def angle_XYW : ℝ := 15

-- The theorem statement proving the smallest possible degree measure for ∠WYZ
theorem smallest_angle_WYZ : angle_XYZ - angle_XYW = 25 :=
by
  -- Add the proof here
  sorry

end smallest_angle_WYZ_l183_183863


namespace find_value_of_m_l183_183126

-- Define the quadratic function and the values in the given table
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
variables (a b c m : ℝ)
variables (h1 : quadratic_function a b c (-1) = m)
variables (h2 : quadratic_function a b c 0 = 2)
variables (h3 : quadratic_function a b c 1 = 1)
variables (h4 : quadratic_function a b c 2 = 2)
variables (h5 : quadratic_function a b c 3 = 5)
variables (h6 : quadratic_function a b c 4 = 10)

-- Theorem stating that the value of m is 5
theorem find_value_of_m : m = 5 :=
by
  sorry

end find_value_of_m_l183_183126


namespace preimage_of_3_1_l183_183123

theorem preimage_of_3_1 (a b : ℝ) (f : ℝ × ℝ → ℝ × ℝ) (h : ∀ (a b : ℝ), f (a, b) = (a + 2 * b, 2 * a - b)) :
  f (1, 1) = (3, 1) :=
by {
  sorry
}

end preimage_of_3_1_l183_183123


namespace smallest_common_multiple_of_10_11_18_l183_183700

theorem smallest_common_multiple_of_10_11_18 : 
  ∃ (n : ℕ), (n % 10 = 0) ∧ (n % 11 = 0) ∧ (n % 18 = 0) ∧ (n = 990) :=
by
  sorry

end smallest_common_multiple_of_10_11_18_l183_183700


namespace minor_axis_length_is_2sqrt3_l183_183169

-- Define the points given in the problem
def points : List (ℝ × ℝ) := [(1, 1), (0, 0), (0, 3), (4, 0), (4, 3)]

-- Define a function that checks if an ellipse with axes parallel to the coordinate axes
-- passes through given points, and returns the length of its minor axis if it does.
noncomputable def minor_axis_length (pts : List (ℝ × ℝ)) : ℝ :=
  if h : (0,0) ∈ pts ∧ (0,3) ∈ pts ∧ (4,0) ∈ pts ∧ (4,3) ∈ pts ∧ (1,1) ∈ pts then
    let a := (4 - 0) / 2 -- half the width of the rectangle
    let b_sq := 3 -- derived from solving the ellipse equation
    2 * Real.sqrt b_sq
  else 0

-- The theorem statement:
theorem minor_axis_length_is_2sqrt3 : minor_axis_length points = 2 * Real.sqrt 3 := by
  sorry

end minor_axis_length_is_2sqrt3_l183_183169


namespace solution_l183_183170

def p : Prop := ∀ x > 0, Real.log (x + 1) > 0
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

theorem solution : p ∧ ¬ q := by
  sorry

end solution_l183_183170


namespace Gloria_pine_tree_price_l183_183381

theorem Gloria_pine_tree_price :
  ∀ (cabin_cost cash cypress_count pine_count maple_count cypress_price maple_price left_over_price : ℕ)
  (cypress_total maple_total total_required total_from_cypress_and_maple total_needed amount_per_pine : ℕ),
    cabin_cost = 129000 →
    cash = 150 →
    cypress_count = 20 →
    pine_count = 600 →
    maple_count = 24 →
    cypress_price = 100 →
    maple_price = 300 →
    left_over_price = 350 →
    cypress_total = cypress_count * cypress_price →
    maple_total = maple_count * maple_price →
    total_required = cabin_cost - cash + left_over_price →
    total_from_cypress_and_maple = cypress_total + maple_total →
    total_needed = total_required - total_from_cypress_and_maple →
    amount_per_pine = total_needed / pine_count →
    amount_per_pine = 200 :=
by
  intros
  sorry

end Gloria_pine_tree_price_l183_183381


namespace relationship_between_abc_l183_183239

theorem relationship_between_abc 
  (a b c : ℝ) 
  (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) 
  (ha : Real.exp a = 9 * a * Real.log 11)
  (hb : Real.exp b = 10 * b * Real.log 10)
  (hc : Real.exp c = 11 * c * Real.log 9) : 
  a < b ∧ b < c :=
sorry

end relationship_between_abc_l183_183239


namespace functional_equation_odd_l183_183818

   variable {R : Type*} [AddCommGroup R] [Module ℝ R]

   def isOdd (f : ℝ → ℝ) : Prop :=
     ∀ x : ℝ, f (-x) = -f x

   theorem functional_equation_odd (f : ℝ → ℝ)
       (h_fun : ∀ x y : ℝ, f (x + y) = f x + f y) : isOdd f :=
   by
     sorry
   
end functional_equation_odd_l183_183818


namespace shaded_shape_area_l183_183171

/-- Define the coordinates and the conditions for the central square and triangles in the grid -/
def grid_size := 10
def central_square_side := 2
def central_square_area := central_square_side * central_square_side

def triangle_base := 5
def triangle_height := 5
def triangle_area := (1 / 2) * triangle_base * triangle_height

def number_of_triangles := 4
def total_triangle_area := number_of_triangles * triangle_area

def total_shaded_area := total_triangle_area + central_square_area

theorem shaded_shape_area : total_shaded_area = 54 :=
by
  -- We have defined each area component and summed them to the total shaded area.
  -- The statement ensures that the area of the shaded shape is equal to 54.
  sorry

end shaded_shape_area_l183_183171


namespace mechanical_pencils_fraction_l183_183490

theorem mechanical_pencils_fraction (total_pencils : ℕ) (frac_mechanical : ℚ)
    (mechanical_pencils : ℕ) (standard_pencils : ℕ) (new_total_pencils : ℕ) 
    (new_standard_pencils : ℕ) (new_frac_mechanical : ℚ):
  total_pencils = 120 →
  frac_mechanical = 1 / 4 →
  mechanical_pencils = frac_mechanical * total_pencils →
  standard_pencils = total_pencils - mechanical_pencils →
  new_standard_pencils = 3 * standard_pencils →
  new_total_pencils = mechanical_pencils + new_standard_pencils →
  new_frac_mechanical = mechanical_pencils / new_total_pencils →
  new_frac_mechanical = 1 / 10 :=
by
  sorry

end mechanical_pencils_fraction_l183_183490


namespace Thursday_total_rainfall_correct_l183_183722

def Monday_rainfall : ℝ := 0.9
def Tuesday_rainfall : ℝ := Monday_rainfall - 0.7
def Wednesday_rainfall : ℝ := Tuesday_rainfall + 0.5 * Tuesday_rainfall
def additional_rain : ℝ := 0.3
def decrease_factor : ℝ := 0.2
def Thursday_rainfall_before_addition : ℝ := Wednesday_rainfall - decrease_factor * Wednesday_rainfall
def Thursday_total_rainfall : ℝ := Thursday_rainfall_before_addition + additional_rain

theorem Thursday_total_rainfall_correct :
  Thursday_total_rainfall = 0.54 :=
by
  sorry

end Thursday_total_rainfall_correct_l183_183722


namespace ab_square_l183_183621

theorem ab_square (x y : ℝ) (hx : y = 4 * x^2 + 7 * x - 1) (hy : y = -4 * x^2 + 7 * x + 1) :
  (2 * x)^2 + (2 * y)^2 = 50 :=
by
  sorry

end ab_square_l183_183621


namespace factor_poly_l183_183305

theorem factor_poly (P Q : ℝ) (h1 : ∃ b c : ℝ, 
  (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + Q)
  : P + Q = 50 :=
sorry

end factor_poly_l183_183305


namespace solve_for_a_l183_183211

theorem solve_for_a (a : ℤ) : -2 - a = 0 → a = -2 :=
by
  sorry

end solve_for_a_l183_183211


namespace mom_buys_tshirts_l183_183473

theorem mom_buys_tshirts 
  (tshirts_per_package : ℕ := 3) 
  (num_packages : ℕ := 17) :
  tshirts_per_package * num_packages = 51 :=
by
  sorry

end mom_buys_tshirts_l183_183473


namespace probability_of_less_than_5_is_one_half_l183_183926

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l183_183926


namespace range_of_a_l183_183783

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (y / 4 - (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) ↔ (-3 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l183_183783


namespace exists_large_cube_construction_l183_183815

theorem exists_large_cube_construction (n : ℕ) :
  ∃ N : ℕ, ∀ n > N, ∃ k : ℕ, k^3 = n :=
sorry

end exists_large_cube_construction_l183_183815


namespace f_is_odd_l183_183416

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (α - 2) * x ^ α

theorem f_is_odd (α : ℝ) (hα : α = 3) : ∀ x : ℝ, f α (-x) = -f α x :=
by sorry

end f_is_odd_l183_183416


namespace greatest_possible_value_of_x_l183_183408

-- Define the function based on the given equation
noncomputable def f (x : ℝ) : ℝ := (4 * x - 16) / (3 * x - 4)

-- Statement to be proved
theorem greatest_possible_value_of_x : 
  (∀ x : ℝ, (f x)^2 + (f x) = 20) → 
  ∃ x : ℝ, (f x)^2 + (f x) = 20 ∧ x = 36 / 19 :=
by
  sorry

end greatest_possible_value_of_x_l183_183408


namespace largest_m_dividing_factorials_l183_183038

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

theorem largest_m_dividing_factorials (m : ℕ) :
  (∀ k : ℕ, k ≤ m → factorial k ∣ (factorial 100 + factorial 99 + factorial 98)) ↔ m = 98 :=
by
  sorry

end largest_m_dividing_factorials_l183_183038


namespace quadratic_roots_r12_s12_l183_183410

theorem quadratic_roots_r12_s12 (r s : ℝ) (h1 : r + s = 2 * Real.sqrt 3) (h2 : r * s = 1) :
  r^12 + s^12 = 940802 :=
sorry

end quadratic_roots_r12_s12_l183_183410


namespace complement_of_log_set_l183_183920

-- Define the set A based on the logarithmic inequality condition
def A : Set ℝ := { x : ℝ | Real.log x / Real.log (1 / 2) ≥ 2 }

-- Define the complement of A in the real numbers
noncomputable def complement_A : Set ℝ := { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 }

-- The goal is to prove the equivalence
theorem complement_of_log_set :
  complement_A = { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 } :=
by
  sorry

end complement_of_log_set_l183_183920


namespace min_value_of_a_l183_183037

theorem min_value_of_a 
  (a b x1 x2 : ℕ) 
  (h1 : a = b - 2005) 
  (h2 : (x1 + x2) = a) 
  (h3 : (x1 * x2) = b) 
  (h4 : x1 > 0 ∧ x2 > 0) : 
  a ≥ 95 :=
sorry

end min_value_of_a_l183_183037


namespace selling_price_is_80000_l183_183160

-- Given the conditions of the problem
def purchasePrice : ℕ := 45000
def repairCosts : ℕ := 12000
def profitPercent : ℚ := 40.35 / 100

-- Total cost calculation
def totalCost := purchasePrice + repairCosts

-- Profit calculation
def profit := profitPercent * totalCost

-- Selling price calculation
def sellingPrice := totalCost + profit

-- Statement of the proof problem
theorem selling_price_is_80000 : round sellingPrice = 80000 := by
  sorry

end selling_price_is_80000_l183_183160


namespace percentage_of_l183_183050

theorem percentage_of (part whole : ℕ) (h_part : part = 120) (h_whole : whole = 80) : 
  ((part : ℚ) / (whole : ℚ)) * 100 = 150 := 
by
  sorry

end percentage_of_l183_183050


namespace original_price_of_shoes_l183_183917

-- Define the conditions.
def discount_rate : ℝ := 0.20
def amount_paid : ℝ := 480

-- Statement of the theorem.
theorem original_price_of_shoes (P : ℝ) (h₀ : P * (1 - discount_rate) = amount_paid) : 
  P = 600 :=
by
  sorry

end original_price_of_shoes_l183_183917


namespace correct_word_for_blank_l183_183904

theorem correct_word_for_blank :
  (∀ (word : String), word = "that" ↔ word = "whoever" ∨ word = "someone" ∨ word = "that" ∨ word = "any") :=
by
  sorry

end correct_word_for_blank_l183_183904


namespace return_trip_time_l183_183108

variable (d p w_1 w_2 : ℝ)
variable (t t' : ℝ)
variable (h1 : d / (p - w_1) = 120)
variable (h2 : d / (p + w_2) = t - 10)
variable (h3 : t = d / p)

theorem return_trip_time :
  t' = 72 :=
by
  sorry

end return_trip_time_l183_183108


namespace total_books_l183_183909

theorem total_books (b1 b2 b3 b4 b5 b6 b7 b8 b9 : ℕ) :
  b1 = 56 →
  b2 = b1 + 2 →
  b3 = b2 + 2 →
  b4 = b3 + 2 →
  b5 = b4 + 2 →
  b6 = b5 + 2 →
  b7 = b6 - 4 →
  b8 = b7 - 4 →
  b9 = b8 - 4 →
  b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 = 490 :=
by
  sorry

end total_books_l183_183909


namespace gcd_98_140_245_l183_183936

theorem gcd_98_140_245 : Nat.gcd (Nat.gcd 98 140) 245 = 7 := 
by 
  sorry

end gcd_98_140_245_l183_183936


namespace sum_five_smallest_primes_l183_183534

theorem sum_five_smallest_primes : (2 + 3 + 5 + 7 + 11) = 28 := by
  -- We state the sum of the known five smallest prime numbers.
  sorry

end sum_five_smallest_primes_l183_183534


namespace inequality_proof_l183_183417

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y) :=
sorry

end inequality_proof_l183_183417


namespace polynomial_solution_l183_183735

theorem polynomial_solution (P : ℝ → ℝ) (hP : ∀ x : ℝ, (x + 1) * P (x - 1) + (x - 1) * P (x + 1) = 2 * x * P x) :
  ∃ (a d : ℝ), ∀ x : ℝ, P x = a * x^3 - a * x + d := 
sorry

end polynomial_solution_l183_183735


namespace max_value_of_squares_l183_183090

theorem max_value_of_squares (a b c d : ℝ) 
  (h1 : a + b = 18) 
  (h2 : ab + c + d = 91) 
  (h3 : ad + bc = 187) 
  (h4 : cd = 105) : 
  a^2 + b^2 + c^2 + d^2 ≤ 107 :=
sorry

end max_value_of_squares_l183_183090


namespace min_2a_b_c_l183_183455

theorem min_2a_b_c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : (a + b) * b * c = 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 := sorry

end min_2a_b_c_l183_183455


namespace billboard_shorter_side_length_l183_183539

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 120)
  (h2 : 2 * L + 2 * W = 46) :
  min L W = 8 :=
by
  sorry

end billboard_shorter_side_length_l183_183539


namespace subtraction_to_nearest_thousandth_l183_183280

theorem subtraction_to_nearest_thousandth : 
  (456.789 : ℝ) - (234.567 : ℝ) = 222.222 :=
by
  sorry

end subtraction_to_nearest_thousandth_l183_183280


namespace exterior_angle_of_triangle_cond_40_degree_l183_183890

theorem exterior_angle_of_triangle_cond_40_degree (A B C : ℝ)
  (h1 : (A = 40 ∨ B = 40 ∨ C = 40))
  (h2 : A = B)
  (h3 : A + B + C = 180) :
  ((180 - C) = 80 ∨ (180 - C) = 140) :=
by
  sorry

end exterior_angle_of_triangle_cond_40_degree_l183_183890


namespace find_d_l183_183082

-- Defining the basic points and their corresponding conditions
structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def a : Point ℝ := ⟨1, 0, 1⟩
def b : Point ℝ := ⟨0, 1, 0⟩
def c : Point ℝ := ⟨0, 1, 1⟩

-- introducing k as a positive integer
variables (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1)

def d (k : ℤ) : Point ℝ := ⟨k*d, k*d, -d⟩ where d := -(k / (k-1))

-- The proof statement
theorem find_d (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1) :
∃ d: ℝ, d = - (k / (k-1)) :=
sorry

end find_d_l183_183082


namespace trig_identity_one_trig_identity_two_l183_183286

theorem trig_identity_one :
  2 * (Real.cos (45 * Real.pi / 180)) - (3 / 2) * (Real.tan (30 * Real.pi / 180)) * (Real.cos (30 * Real.pi / 180)) + (Real.sin (60 * Real.pi / 180))^2 = Real.sqrt 2 :=
sorry

theorem trig_identity_two :
  (Real.sin (30 * Real.pi / 180))⁻¹ * (Real.sin (60 * Real.pi / 180) - Real.cos (45 * Real.pi / 180)) - Real.sqrt ((1 - Real.tan (60 * Real.pi / 180))^2) = 1 - Real.sqrt 2 :=
sorry

end trig_identity_one_trig_identity_two_l183_183286


namespace emily_did_not_sell_bars_l183_183144

-- Definitions based on conditions
def cost_per_bar : ℕ := 4
def total_bars : ℕ := 8
def total_earnings : ℕ := 20

-- The statement to be proved
theorem emily_did_not_sell_bars :
  (total_bars - (total_earnings / cost_per_bar)) = 3 :=
by
  sorry

end emily_did_not_sell_bars_l183_183144


namespace percentage_to_decimal_l183_183554

theorem percentage_to_decimal : (5 / 100 : ℚ) = 0.05 := by
  sorry

end percentage_to_decimal_l183_183554


namespace product_of_intersection_coordinates_l183_183089

noncomputable def circle1 := {P : ℝ×ℝ | (P.1^2 - 4*P.1 + P.2^2 - 8*P.2 + 20) = 0}
noncomputable def circle2 := {P : ℝ×ℝ | (P.1^2 - 6*P.1 + P.2^2 - 8*P.2 + 25) = 0}

theorem product_of_intersection_coordinates :
  ∀ P ∈ circle1 ∩ circle2, P = (2, 4) → (P.1 * P.2) = 8 :=
by
  sorry

end product_of_intersection_coordinates_l183_183089


namespace secant_length_problem_l183_183452

theorem secant_length_problem (tangent_length : ℝ) (internal_segment_length : ℝ) (external_segment_length : ℝ) 
    (h1 : tangent_length = 18) (h2 : internal_segment_length = 27) : external_segment_length = 9 :=
by
  sorry

end secant_length_problem_l183_183452


namespace inequality_for_positive_reals_l183_183987

variable {a b c : ℝ}
variable {k : ℕ}

theorem inequality_for_positive_reals 
  (hab : a > 0) 
  (hbc : b > 0) 
  (hac : c > 0) 
  (hprod : a * b * c = 1) 
  (hk : k ≥ 2) 
  : (a ^ k) / (a + b) + (b ^ k) / (b + c) + (c ^ k) / (c + a) ≥ 3 / 2 := 
sorry

end inequality_for_positive_reals_l183_183987


namespace maxim_birth_probability_l183_183959

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l183_183959


namespace holidays_per_month_l183_183721

theorem holidays_per_month (total_holidays : ℕ) (months_in_year : ℕ) (holidays_in_month : ℕ) 
    (h1 : total_holidays = 48) (h2 : months_in_year = 12) : holidays_in_month = 4 := 
by
  sorry

end holidays_per_month_l183_183721


namespace solve_inequality_l183_183426

theorem solve_inequality {a x : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ (x : ℝ), (a > 1 ∧ (a^(2/3) ≤ x ∧ x < a^(3/4) ∨ x > a)) ∨ (0 < a ∧ a < 1 ∧ (a^(3/4) < x ∧ x ≤ a^(2/3) ∨ 0 < x ∧ x < a))) :=
sorry

end solve_inequality_l183_183426


namespace tan_pi_minus_alpha_l183_183358

theorem tan_pi_minus_alpha 
  (α : ℝ) 
  (h1 : Real.sin α = 1 / 3) 
  (h2 : π / 2 < α) 
  (h3 : α < π) :
  Real.tan (π - α) = Real.sqrt 2 / 4 :=
by
  sorry

end tan_pi_minus_alpha_l183_183358


namespace second_alloy_amount_l183_183121

theorem second_alloy_amount (x : ℝ) :
  (0.12 * 15 + 0.08 * x = 0.092 * (15 + x)) → x = 35 :=
by
  sorry

end second_alloy_amount_l183_183121


namespace part_I_part_II_l183_183462

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (x a : ℝ) : ℝ := (f x a) + (g x)

theorem part_I (a : ℝ) :
  (∀ x > 0, f x a ≥ g x) → a ≤ 0.5 :=
by
  sorry

theorem part_II (a x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) 
  (hx1_lt_half : x1 < 0.5) :
  (h x1 a = 2 * x1^2 + Real.log x1) →
  (h x2 a = 2 * x2^2 + Real.log x2) →
  (x1 * x2 = 0.5) →
  h x1 a - h x2 a > (3 / 4) - Real.log 2 :=
by
  sorry

end part_I_part_II_l183_183462


namespace abi_suji_age_ratio_l183_183372

theorem abi_suji_age_ratio (A S : ℕ) (h1 : S = 24) 
  (h2 : (A + 3) / (S + 3) = 11 / 9) : A / S = 5 / 4 := 
by 
  sorry

end abi_suji_age_ratio_l183_183372


namespace range_of_a_l183_183214

noncomputable def A : Set ℝ := {x : ℝ | ((x^2) - x - 2) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x ∈ A, (x^2 - a*x - a - 2) ≤ 0) → a ≥ (2/3) :=
by
  intro h
  sorry

end range_of_a_l183_183214


namespace Nancy_shelved_biographies_l183_183153

def NancyBooks.shelved_books_from_top : Nat := 12 + 8 + 4 -- history + romance + poetry
def NancyBooks.total_books_on_cart : Nat := 46
def NancyBooks.bottom_books_after_top_shelved : Nat := 46 - 24
def NancyBooks.mystery_books_on_bottom : Nat := NancyBooks.bottom_books_after_top_shelved / 2
def NancyBooks.western_novels_on_bottom : Nat := 5
def NancyBooks.biographies : Nat := NancyBooks.bottom_books_after_top_shelved - NancyBooks.mystery_books_on_bottom - NancyBooks.western_novels_on_bottom

theorem Nancy_shelved_biographies : NancyBooks.biographies = 6 := by
  sorry

end Nancy_shelved_biographies_l183_183153


namespace simplify_expression_l183_183543

variable (x y : ℝ)

-- Define the proposition
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := 
by
  sorry

end simplify_expression_l183_183543


namespace sixteenth_term_l183_183877

theorem sixteenth_term :
  (-1)^(16+1) * Real.sqrt (3 * (16 - 1)) = -3 * Real.sqrt 5 :=
by sorry

end sixteenth_term_l183_183877


namespace percentage_of_female_officers_on_duty_l183_183325

theorem percentage_of_female_officers_on_duty
    (on_duty : ℕ) (half_on_duty_female : on_duty / 2 = 100)
    (total_female_officers : ℕ)
    (total_female_officers_value : total_female_officers = 1000)
    : (100 / total_female_officers : ℝ) * 100 = 10 :=
by sorry

end percentage_of_female_officers_on_duty_l183_183325


namespace ticket_sales_total_cost_l183_183655

noncomputable def total_ticket_cost (O B : ℕ) : ℕ :=
  12 * O + 8 * B

theorem ticket_sales_total_cost (O B : ℕ) (h1 : O + B = 350) (h2 : B = O + 90) :
  total_ticket_cost O B = 3320 :=
by
  -- the proof steps calculating the total cost will go here
  sorry

end ticket_sales_total_cost_l183_183655


namespace power_of_negative_125_l183_183099

theorem power_of_negative_125 : (-125 : ℝ)^(4/3) = 625 := by
  sorry

end power_of_negative_125_l183_183099


namespace park_area_is_120000_l183_183598

noncomputable def area_of_park : ℕ :=
  let speed_km_hr := 12
  let speed_m_min := speed_km_hr * 1000 / 60
  let time_min := 8
  let perimeter := speed_m_min * time_min
  let ratio_l_b := (1, 3)
  let length := perimeter / (2 * (ratio_l_b.1 + ratio_l_b.2))
  let breadth := ratio_l_b.2 * length
  length * breadth

theorem park_area_is_120000 :
  area_of_park = 120000 :=
by
  sorry

end park_area_is_120000_l183_183598


namespace no_negative_roots_l183_183901

theorem no_negative_roots (x : ℝ) :
  x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 = 0 → 0 ≤ x :=
by
  sorry

end no_negative_roots_l183_183901


namespace min_value_expression_l183_183375

noncomputable def expression (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_expression : ∃ x : ℝ, expression x = -6480.25 :=
sorry

end min_value_expression_l183_183375


namespace no_such_function_exists_l183_183197

noncomputable def func_a (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ ∀ n : ℕ, a n = n - a (a n)

theorem no_such_function_exists : ¬ ∃ a : ℕ → ℕ, func_a a :=
by
  sorry

end no_such_function_exists_l183_183197


namespace evaluate_at_points_l183_183558

noncomputable def f (x : ℝ) : ℝ :=
if x > 3 then x^2 - 3*x + 2
else if -2 ≤ x ∧ x ≤ 3 then -3*x + 5
else 9

theorem evaluate_at_points : f (-3) + f (0) + f (4) = 20 := by
  sorry

end evaluate_at_points_l183_183558


namespace shortest_side_of_right_triangle_l183_183465

theorem shortest_side_of_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) : 
  ∀ c, (c = 5 ∨ c = 12 ∨ c = (Real.sqrt (a^2 + b^2))) → c = 5 :=
by
  intros c h
  sorry

end shortest_side_of_right_triangle_l183_183465


namespace convex_polyhedron_property_l183_183039

-- Given conditions as definitions
def num_faces : ℕ := 40
def num_hexagons : ℕ := 8
def num_triangles_eq_twice_pentagons (P : ℕ) (T : ℕ) : Prop := T = 2 * P
def num_pentagons_eq_twice_hexagons (P : ℕ) (H : ℕ) : Prop := P = 2 * H

-- Main statement for the proof problem
theorem convex_polyhedron_property (P T V : ℕ) :
  num_triangles_eq_twice_pentagons P T ∧ num_pentagons_eq_twice_hexagons P num_hexagons ∧ 
  num_faces = T + P + num_hexagons ∧ V = (T * 3 + P * 5 + num_hexagons * 6) / 2 + num_faces - 2 →
  100 * P + 10 * T + V = 535 :=
by
  sorry

end convex_polyhedron_property_l183_183039


namespace eric_has_9306_erasers_l183_183762

-- Define the conditions as constants
def number_of_friends := 99
def erasers_per_friend := 94

-- Define the total number of erasers based on the conditions
def total_erasers := number_of_friends * erasers_per_friend

-- Theorem stating the total number of erasers Eric has
theorem eric_has_9306_erasers : total_erasers = 9306 := by
  -- Proof to be filled in
  sorry

end eric_has_9306_erasers_l183_183762


namespace candy_bar_cost_l183_183156

theorem candy_bar_cost :
  ∃ C : ℕ, (C + 1 = 3) → (C = 2) :=
by
  use 2
  intros h
  linarith

end candy_bar_cost_l183_183156


namespace largest_rectangle_area_l183_183567

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end largest_rectangle_area_l183_183567


namespace part_a_part_b_l183_183207

-- Define the problem as described
noncomputable def can_transform_to_square (figure : Type) (parts : ℕ) (all_triangles : Bool) : Bool :=
sorry  -- This is a placeholder for the actual implementation

-- The figure satisfies the condition to cut into four parts and rearrange into a square
theorem part_a (figure : Type) : can_transform_to_square figure 4 false = true :=
sorry

-- The figure satisfies the condition to cut into five triangular parts and rearrange into a square
theorem part_b (figure : Type) : can_transform_to_square figure 5 true = true :=
sorry

end part_a_part_b_l183_183207


namespace max_possible_score_l183_183338

theorem max_possible_score (s : ℝ) (h : 80 = s * 2) : s * 5 ≥ 100 :=
by
  -- sorry placeholder for the proof
  sorry

end max_possible_score_l183_183338


namespace real_solutions_of_polynomial_l183_183004

theorem real_solutions_of_polynomial :
  ∀ x : ℝ, x^4 - 3 * x^3 + x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end real_solutions_of_polynomial_l183_183004


namespace factorize_polynomial_l183_183148

noncomputable def zeta : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem factorize_polynomial :
  (zeta^3 = 1) ∧ (zeta^2 + zeta + 1 = 0) → (x : ℂ) → (x^15 + x^10 + x) = (x^3 - 1) * (x^12 + x^9 + x^6 + x^3 + 1)
:= sorry

end factorize_polynomial_l183_183148


namespace how_much_does_c_have_l183_183740

theorem how_much_does_c_have (A B C : ℝ) (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : B + C = 150) : C = 50 :=
by
  sorry

end how_much_does_c_have_l183_183740


namespace bologna_sandwiches_l183_183779

variable (C B P : ℕ)

theorem bologna_sandwiches (h1 : C = 1) (h2 : B = 7) (h3 : P = 8)
                          (h4 : C + B + P = 16) (h5 : 80 / 16 = 5) :
                          B * 5 = 35 :=
by
  -- omit the proof part
  sorry

end bologna_sandwiches_l183_183779


namespace Toby_second_part_distance_l183_183412

noncomputable def total_time_journey (distance_unloaded_second: ℝ) : ℝ :=
  18 + (distance_unloaded_second / 20) + 8 + 7

theorem Toby_second_part_distance:
  ∃ d : ℝ, total_time_journey d = 39 ∧ d = 120 :=
by
  use 120
  unfold total_time_journey
  sorry

end Toby_second_part_distance_l183_183412


namespace topsoil_cost_l183_183999

theorem topsoil_cost :
  let cubic_yard_to_cubic_foot := 27
  let cubic_feet_in_5_cubic_yards := 5 * cubic_yard_to_cubic_foot
  let cost_per_cubic_foot := 6
  let total_cost := cubic_feet_in_5_cubic_yards * cost_per_cubic_foot
  total_cost = 810 :=
by
  sorry

end topsoil_cost_l183_183999


namespace range_of_a_l183_183471

noncomputable def e := Real.exp 1

theorem range_of_a (a : Real) 
  (h : ∀ x : Real, 1 ≤ x ∧ x ≤ 2 → Real.exp x - a ≥ 0) : 
  a ≤ e :=
by
  sorry

end range_of_a_l183_183471


namespace find_f_lg_lg_2_l183_183323

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * (Real.sin x) + 4

theorem find_f_lg_lg_2 (a b : ℝ) (m : ℝ) 
  (h1 : f a b (Real.logb 10 2) = 5) 
  (h2 : m = Real.logb 10 2) : 
  f a b (Real.logb 2 m) = 3 :=
sorry

end find_f_lg_lg_2_l183_183323


namespace one_fourth_of_6_8_is_fraction_l183_183145

theorem one_fourth_of_6_8_is_fraction :
  (6.8 / 4 : ℚ) = 17 / 10 :=
sorry

end one_fourth_of_6_8_is_fraction_l183_183145


namespace find_a_l183_183482

theorem find_a (a x : ℝ) (h1: a - 2 ≤ x) (h2: x ≤ a + 1) (h3 : -x^2 + 2 * x + 3 = 3) :
  a = 2 := sorry

end find_a_l183_183482


namespace lines_intersection_l183_183177

/-- Two lines are defined by the equations y = 2x + c and y = 4x + d.
These lines intersect at the point (8, 12).
Prove that c + d = -24. -/
theorem lines_intersection (c d : ℝ) (h1 : 12 = 2 * 8 + c) (h2 : 12 = 4 * 8 + d) :
    c + d = -24 :=
by
  sorry

end lines_intersection_l183_183177


namespace rickshaw_distance_l183_183306

theorem rickshaw_distance :
  ∃ (distance : ℝ), 
  (13.5 + (distance - 1) * (2.50 / (1 / 3))) = 103.5 ∧ distance = 13 :=
by
  sorry

end rickshaw_distance_l183_183306


namespace income_of_first_member_l183_183334

-- Define the number of family members
def num_members : ℕ := 4

-- Define the average income per member
def avg_income : ℕ := 10000

-- Define the known incomes of the other three members
def income2 : ℕ := 15000
def income3 : ℕ := 6000
def income4 : ℕ := 11000

-- Define the total income of the family
def total_income : ℕ := avg_income * num_members

-- Define the total income of the other three members
def total_other_incomes : ℕ := income2 + income3 + income4

-- Define the income of the first member
def income1 : ℕ := total_income - total_other_incomes

-- The theorem to prove
theorem income_of_first_member : income1 = 8000 := by
  sorry

end income_of_first_member_l183_183334


namespace book_total_pages_l183_183003

theorem book_total_pages (num_chapters pages_per_chapter : ℕ) (h1 : num_chapters = 31) (h2 : pages_per_chapter = 61) :
  num_chapters * pages_per_chapter = 1891 := sorry

end book_total_pages_l183_183003


namespace necessarily_positive_y_plus_z_l183_183437

-- Given conditions
variables {x y z : ℝ}

-- Assert the conditions
axiom hx : 0 < x ∧ x < 1
axiom hy : -1 < y ∧ y < 0
axiom hz : 1 < z ∧ z < 2

-- Prove that y + z is necessarily positive
theorem necessarily_positive_y_plus_z : y + z > 0 :=
by
  sorry

end necessarily_positive_y_plus_z_l183_183437


namespace fraction_of_fritz_money_l183_183451

theorem fraction_of_fritz_money
  (Fritz_money : ℕ)
  (total_amount : ℕ)
  (fraction : ℚ)
  (Sean_money : ℚ)
  (Rick_money : ℚ)
  (h1 : Fritz_money = 40)
  (h2 : total_amount = 96)
  (h3 : Sean_money = fraction * Fritz_money + 4)
  (h4 : Rick_money = 3 * Sean_money)
  (h5 : Rick_money + Sean_money = total_amount) :
  fraction = 1 / 2 :=
by
  sorry

end fraction_of_fritz_money_l183_183451


namespace totalCroissants_is_18_l183_183780

def jorgeCroissants : ℕ := 7
def giulianaCroissants : ℕ := 5
def matteoCroissants : ℕ := 6

def totalCroissants : ℕ := jorgeCroissants + giulianaCroissants + matteoCroissants

theorem totalCroissants_is_18 : totalCroissants = 18 := by
  -- Proof will be provided here
  sorry

end totalCroissants_is_18_l183_183780


namespace union_set_l183_183759

def M : Set ℝ := {x | -2 < x ∧ x < 1}
def P : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem union_set : M ∪ P = {x : ℝ | -2 ≤ x ∧ x < 2} := by
  sorry

end union_set_l183_183759


namespace exists_pythagorean_number_in_range_l183_183066

def is_pythagorean_area (a : ℕ) : Prop :=
  ∃ (x y z : ℕ), x^2 + y^2 = z^2 ∧ a = (x * y) / 2

theorem exists_pythagorean_number_in_range (n : ℕ) (hn : n > 12) : 
  ∃ (m : ℕ), is_pythagorean_area m ∧ n < m ∧ m < 2 * n :=
sorry

end exists_pythagorean_number_in_range_l183_183066


namespace exist_distinct_indices_l183_183386

theorem exist_distinct_indices (n : ℕ) (h1 : n > 3)
  (a : Fin n.succ → ℕ) 
  (h2 : StrictMono a) 
  (h3 : a n ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n.succ), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ 
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
    k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ 
    a i + a j = a k + a l ∧ 
    a k + a l = a m := 
sorry

end exist_distinct_indices_l183_183386


namespace cover_points_with_two_disks_l183_183492

theorem cover_points_with_two_disks :
  ∀ (points : Fin 2014 → ℝ × ℝ),
    (∀ (i j k : Fin 2014), i ≠ j → j ≠ k → i ≠ k → 
      dist (points i) (points j) ≤ 1 ∨ dist (points j) (points k) ≤ 1 ∨ dist (points i) (points k) ≤ 1) →
    ∃ (A B : ℝ × ℝ), ∀ (p : Fin 2014),
      dist (points p) A ≤ 1 ∨ dist (points p) B ≤ 1 :=
by
  sorry

end cover_points_with_two_disks_l183_183492


namespace M_sufficient_not_necessary_for_N_l183_183246

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 2}

theorem M_sufficient_not_necessary_for_N (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → ¬ (a ∈ M)) :=
sorry

end M_sufficient_not_necessary_for_N_l183_183246


namespace complement_union_l183_183002

-- Definitions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3}

-- Theorem Statement
theorem complement_union (hU: U = {0, 1, 2, 3, 4}) (hA: A = {0, 1, 3}) (hB: B = {2, 3}) :
  (U \ (A ∪ B)) = {4} :=
sorry

end complement_union_l183_183002


namespace graded_worksheets_before_l183_183087

-- Definitions based on conditions
def initial_worksheets : ℕ := 34
def additional_worksheets : ℕ := 36
def total_worksheets : ℕ := 63

-- Equivalent proof problem statement
theorem graded_worksheets_before (x : ℕ) (h₁ : initial_worksheets - x + additional_worksheets = total_worksheets) : x = 7 :=
by sorry

end graded_worksheets_before_l183_183087


namespace solve_quadratics_l183_183695

theorem solve_quadratics (p q u v : ℤ)
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ p ≠ q)
  (h2 : u ≠ 0 ∧ v ≠ 0 ∧ u ≠ v)
  (h3 : p + q = -u)
  (h4 : pq = -v)
  (h5 : u + v = -p)
  (h6 : uv = -q) :
  p = -1 ∧ q = 2 ∧ u = -1 ∧ v = 2 :=
by {
  sorry
}

end solve_quadratics_l183_183695


namespace range_of_x_range_of_a_l183_183489

-- Definitions of propositions p and q
def p (a x : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- Question 1
theorem range_of_x (a x : ℝ) : a = 1 → p a x ∧ q x → 2 < x ∧ x < 3 := by
  sorry

-- Question 2
theorem range_of_a (a : ℝ) : (∀ x, ¬p a x → ¬q x) → (∀ x, q x → p a x) → 1 < a ∧ a ≤ 2 := by
  sorry

end range_of_x_range_of_a_l183_183489


namespace billy_free_time_l183_183102

theorem billy_free_time
  (play_time_percentage : ℝ := 0.75)
  (read_pages_per_hour : ℝ := 60)
  (book_pages : ℝ := 80)
  (number_of_books : ℝ := 3)
  (read_percentage : ℝ := 1 - play_time_percentage)
  (total_pages : ℝ := number_of_books * book_pages)
  (read_time_hours : ℝ := total_pages / read_pages_per_hour)
  (free_time_hours : ℝ := read_time_hours / read_percentage) :
  free_time_hours = 16 := 
sorry

end billy_free_time_l183_183102


namespace number_from_division_l183_183942

theorem number_from_division (number : ℝ) (h : number / 2000 = 0.012625) : number = 25.25 :=
by
  sorry

end number_from_division_l183_183942


namespace areasEqualForHexagonAndOctagon_l183_183027

noncomputable def areaHexagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 6) -- Circumscribed radius
  let a := s / (2 * Real.tan (Real.pi / 6)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

noncomputable def areaOctagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 8) -- Circumscribed radius
  let a := s / (2 * Real.tan (3 * Real.pi / 8)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

theorem areasEqualForHexagonAndOctagon :
  let s := 3
  areaHexagon s = areaOctagon s := sorry

end areasEqualForHexagonAndOctagon_l183_183027


namespace simplify_expression_l183_183836

variable (q : ℚ)

theorem simplify_expression :
  (2 * q^3 - 7 * q^2 + 3 * q - 4) + (5 * q^2 - 4 * q + 8) = 2 * q^3 - 2 * q^2 - q + 4 :=
by
  sorry

end simplify_expression_l183_183836


namespace bus_fare_with_train_change_in_total_passengers_l183_183963

variables (p : ℝ) (q : ℝ) (TC : ℝ → ℝ)
variables (p_train : ℝ) (train_capacity : ℝ)

-- Demand function
def demand_function (p : ℝ) : ℝ := 4200 - 100 * p

-- Train fare is fixed
def train_fare : ℝ := 4

-- Train capacity
def train_cap : ℝ := 800

-- Bus total cost function
def total_cost (y : ℝ) : ℝ := 10 * y + 225

-- Case when there is competition (train available)
def optimal_bus_fare_with_train : ℝ := 22

-- Case when there is no competition (train service is closed)
def optimal_bus_fare_without_train : ℝ := 26

-- Change in the number of passengers when the train service closes
def change_in_passengers : ℝ := 400

-- Theorems to prove
theorem bus_fare_with_train : optimal_bus_fare_with_train = 22 := sorry
theorem change_in_total_passengers : change_in_passengers = 400 := sorry

end bus_fare_with_train_change_in_total_passengers_l183_183963


namespace blithe_toy_count_l183_183550

-- Define the initial number of toys, the number lost, and the number found.
def initial_toys := 40
def toys_lost := 6
def toys_found := 9

-- Define the total number of toys after the changes.
def total_toys_after_changes := initial_toys - toys_lost + toys_found

-- The proof statement.
theorem blithe_toy_count : total_toys_after_changes = 43 :=
by
  -- Placeholder for the proof
  sorry

end blithe_toy_count_l183_183550


namespace probability_of_other_note_being_counterfeit_l183_183665

def total_notes := 20
def counterfeit_notes := 5

-- Binomial coefficient (n choose k)
noncomputable def binom (n k : ℕ) : ℚ := n.choose k

-- Probability of event A: both notes are counterfeit
noncomputable def P_A : ℚ :=
  binom counterfeit_notes 2 / binom total_notes 2

-- Probability of event B: at least one note is counterfeit
noncomputable def P_B : ℚ :=
  (binom counterfeit_notes 2 + binom counterfeit_notes 1 * binom (total_notes - counterfeit_notes) 1) / binom total_notes 2

-- Conditional probability P(A|B)
noncomputable def P_A_given_B : ℚ :=
  P_A / P_B

theorem probability_of_other_note_being_counterfeit :
  P_A_given_B = 2/17 :=
by
  sorry

end probability_of_other_note_being_counterfeit_l183_183665


namespace cube_volume_in_pyramid_l183_183908

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

end cube_volume_in_pyramid_l183_183908


namespace triangle_ABC_area_l183_183218

def point : Type := ℚ × ℚ

def triangle_area (A B C : point) : ℚ :=
  let (x1, y1) := A;
  let (x2, y2) := B;
  let (x3, y3) := C;
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_ABC_area :
  let A : point := (-5, 4)
  let B : point := (1, 7)
  let C : point := (4, -3)
  triangle_area A B C = 34.5 :=
by
  sorry

end triangle_ABC_area_l183_183218


namespace algebraic_expression_evaluation_l183_183268

theorem algebraic_expression_evaluation (x y : ℝ) (h : 2 * x - y + 1 = 3) : 4 * x - 2 * y + 5 = 9 := 
by
  sorry

end algebraic_expression_evaluation_l183_183268


namespace relationship_of_sets_l183_183981

def set_A : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 6 + 1}
def set_B : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 3 + 1 / 2}
def set_C : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k : ℝ) / 3 + 1 / 2}

theorem relationship_of_sets : set_C ⊆ set_B ∧ set_B ⊆ set_A := by
  sorry

end relationship_of_sets_l183_183981


namespace twenty_three_percent_of_number_is_forty_six_l183_183937

theorem twenty_three_percent_of_number_is_forty_six (x : ℝ) (h : (23 / 100) * x = 46) : x = 200 :=
sorry

end twenty_three_percent_of_number_is_forty_six_l183_183937


namespace positive_solution_range_l183_183468

theorem positive_solution_range (a : ℝ) (h : a > 0) (x : ℝ) : (∃ x, (a / (x + 3) = 1 / 2) ∧ x > 0) ↔ a > 3 / 2 := by
  sorry

end positive_solution_range_l183_183468


namespace part_one_part_two_range_l183_183400

/-
Definitions based on conditions from the problem:
- Given vectors ax = (\cos x, \sin x), bx = (3, - sqrt(3))
- Domain for x is [0, π]
--
- Prove if a + b is parallel to b, then x = 5π / 6
- Definition of function f(x), and g(x) based on problem requirements.
- Prove the range of g(x) is [-3, sqrt(3)]
-/

/-
Part (1):
Given ax + bx = (cos x + 3, sin x - sqrt(3)) is parallel to bx =  (3, - sqrt(3));
Prove that x = 5π / 6 under x ∈ [0, π].
-/
noncomputable def vector_ax (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_bx : ℝ × ℝ := (3, - Real.sqrt 3)

theorem part_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) 
  (h_parallel : (vector_ax x).1 + vector_bx.1 = (vector_ax x).2 + vector_bx.2) :
  x = 5 * Real.pi / 6 :=
  sorry

/-
Part (2):
Let f(x) = 3 cos x - sqrt(3) sin x.
The function g(x) = -2 sqrt(3) sin(1/2 x - 2π/3) is defined by shifting f(x) right by π/3 and doubling the horizontal coordinate.
Prove the range of g(x) is [-3, sqrt(3)].
-/
noncomputable def f (x : ℝ) := 3 * Real.cos x - Real.sqrt 3 * Real.sin x
noncomputable def g (x : ℝ) := -2 * Real.sqrt 3 * Real.sin (0.5 * x - 2 * Real.pi / 3)

theorem part_two_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  -3 ≤ g x ∧ g x ≤ Real.sqrt 3 :=
  sorry

end part_one_part_two_range_l183_183400


namespace solve_equation_l183_183079

theorem solve_equation : ∀ (x : ℝ), -2 * x + 3 - 2 * x + 3 = 3 * x - 6 → x = 12 / 7 :=
by 
  intro x
  intro h
  sorry

end solve_equation_l183_183079


namespace share_difference_l183_183660

variables {x : ℕ}

theorem share_difference (h1: 12 * x - 7 * x = 5000) : 7 * x - 3 * x = 4000 :=
by
  sorry

end share_difference_l183_183660


namespace sum_a_c_eq_l183_183355

theorem sum_a_c_eq
  (a b c d : ℝ)
  (h1 : a * b + a * c + b * c + b * d + c * d + a * d = 40)
  (h2 : b^2 + d^2 = 29) :
  a + c = 8.4 :=
by
  sorry

end sum_a_c_eq_l183_183355


namespace probability_same_color_pair_l183_183802

theorem probability_same_color_pair : 
  let total_shoes := 28
  let black_pairs := 8
  let brown_pairs := 4
  let gray_pairs := 2
  total_shoes = 2 * (black_pairs + brown_pairs + gray_pairs) → 
  ∃ (prob : ℚ), prob = 7 / 32 := by
  sorry

end probability_same_color_pair_l183_183802


namespace lucy_final_balance_l183_183199

def initial_balance : ℝ := 65
def deposit : ℝ := 15
def withdrawal : ℝ := 4

theorem lucy_final_balance : initial_balance + deposit - withdrawal = 76 :=
by
  sorry

end lucy_final_balance_l183_183199


namespace ring_revolutions_before_stopping_l183_183404

variable (R ω μ m g : ℝ) -- Declare the variables as real numbers

-- Statement of the theorem
theorem ring_revolutions_before_stopping
  (h_positive_R : 0 < R)
  (h_positive_ω : 0 < ω)
  (h_positive_μ : 0 < μ)
  (h_positive_m : 0 < m)
  (h_positive_g : 0 < g) :
  let N1 := m * g / (1 + μ^2)
  let N2 := μ * m * g / (1 + μ^2)
  let K_initial := (1 / 2) * m * R^2 * ω^2
  let A_friction := -2 * π * R * n * μ * (N1 + N2)
  ∃ n : ℝ, n = ω^2 * R * (1 + μ^2) / (4 * π * g * μ * (1 + μ)) :=
by sorry

end ring_revolutions_before_stopping_l183_183404


namespace Q_no_negative_roots_and_at_least_one_positive_root_l183_183540

def Q (x : ℝ) : ℝ := x^7 - 2 * x^6 - 6 * x^4 - 4 * x + 16

theorem Q_no_negative_roots_and_at_least_one_positive_root :
  (∀ x, x < 0 → Q x > 0) ∧ (∃ x, x > 0 ∧ Q x = 0) := 
sorry

end Q_no_negative_roots_and_at_least_one_positive_root_l183_183540


namespace how_many_peaches_l183_183766

-- Define the main problem statement and conditions.
theorem how_many_peaches (A P J_A J_P : ℕ) (h_person_apples: A = 16) (h_person_peaches: P = A + 1) (h_jake_apples: J_A = A + 8) (h_jake_peaches: J_P = P - 6) : P = 17 :=
by
  -- Since the proof is not required, we use sorry to skip it.
  sorry

end how_many_peaches_l183_183766


namespace find_number_l183_183215

noncomputable def question (x : ℝ) : Prop :=
  (2 * x^2 + Real.sqrt 6)^3 = 19683

theorem find_number : ∃ x : ℝ, question x ∧ (x = Real.sqrt ((27 - Real.sqrt 6) / 2) ∨ x = -Real.sqrt ((27 - Real.sqrt 6) / 2)) :=
  sorry

end find_number_l183_183215


namespace hyperbola_triangle_area_l183_183889

/-- The relationship between the hyperbola's asymptotes, tangent, and area proportion -/
theorem hyperbola_triangle_area (a b x0 y0 : ℝ) 
  (h_asymptote1 : ∀ x, y = (b / a) * x)
  (h_asymptote2 : ∀ x, y = -(b / a) * x)
  (h_tangent    : ∀ x y, (x0 * x) / (a ^ 2) - (y0 * y) / (b ^ 2) = 1)
  (h_condition  : (x0 ^ 2) * (a ^ 2) - (y0 ^ 2) * (b ^ 2) = (a ^ 2) * (b ^ 2)) :
  ∃ k : ℝ, k = a ^ 4 :=
sorry

end hyperbola_triangle_area_l183_183889


namespace quadratic_roots_ratio_l183_183812

theorem quadratic_roots_ratio (a b c : ℝ) (h1 : ∀ (s1 s2 : ℝ), s1 * s2 = a → s1 + s2 = -c → 3 * s1 + 3 * s2 = -a → 9 * s1 * s2 = b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  b / c = 27 := sorry

end quadratic_roots_ratio_l183_183812


namespace slope_of_line_AB_is_pm_4_3_l183_183143

noncomputable def slope_of_line_AB : ℝ := sorry

theorem slope_of_line_AB_is_pm_4_3 (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁^2 = 4 * x₁)
  (h₂ : y₂^2 = 4 * x₂)
  (h₃ : (x₁, y₁) ≠ (x₂, y₂))
  (h₄ : (x₁ - 1, y₁) = -4 * (x₂ - 1, y₂)) :
  slope_of_line_AB = 4 / 3 ∨ slope_of_line_AB = -4 / 3 :=
sorry

end slope_of_line_AB_is_pm_4_3_l183_183143


namespace mean_of_combined_sets_l183_183391

theorem mean_of_combined_sets 
  (mean1 mean2 mean3 : ℚ)
  (count1 count2 count3 : ℕ)
  (h1 : mean1 = 15)
  (h2 : mean2 = 20)
  (h3 : mean3 = 12)
  (hc1 : count1 = 7)
  (hc2 : count2 = 8)
  (hc3 : count3 = 5) :
  ((count1 * mean1 + count2 * mean2 + count3 * mean3) / (count1 + count2 + count3)) = 16.25 :=
by
  sorry

end mean_of_combined_sets_l183_183391


namespace paul_spent_374_43_l183_183764

noncomputable def paul_total_cost_after_discounts : ℝ :=
  let dress_shirts := 4 * 15.00
  let discount_dress_shirts := dress_shirts * 0.20
  let cost_dress_shirts := dress_shirts - discount_dress_shirts
  
  let pants := 2 * 40.00
  let discount_pants := pants * 0.30
  let cost_pants := pants - discount_pants
  
  let suit := 150.00
  
  let sweaters := 2 * 30.00
  
  let ties := 3 * 20.00
  let discount_tie := 20.00 * 0.50
  let cost_ties := 20.00 + (20.00 - discount_tie) + 20.00

  let shoes := 80.00
  let discount_shoes := shoes * 0.25
  let cost_shoes := shoes - discount_shoes

  let total_after_discounts := cost_dress_shirts + cost_pants + suit + sweaters + cost_ties + cost_shoes
  
  let total_after_coupon := total_after_discounts * 0.90
  
  let total_after_rewards := total_after_coupon - (500 * 0.05)
  
  let total_after_tax := total_after_rewards * 1.05
  
  total_after_tax

theorem paul_spent_374_43 :
  paul_total_cost_after_discounts = 374.43 :=
by
  sorry

end paul_spent_374_43_l183_183764


namespace grayson_time_per_answer_l183_183244

variable (totalQuestions : ℕ) (unansweredQuestions : ℕ) (totalTimeHours : ℕ)

def timePerAnswer (totalQuestions : ℕ) (unansweredQuestions : ℕ) (totalTimeHours : ℕ) : ℕ :=
  let answeredQuestions := totalQuestions - unansweredQuestions
  let totalTimeMinutes := totalTimeHours * 60
  totalTimeMinutes / answeredQuestions

theorem grayson_time_per_answer :
  totalQuestions = 100 →
  unansweredQuestions = 40 →
  totalTimeHours = 2 →
  timePerAnswer totalQuestions unansweredQuestions totalTimeHours = 2 :=
by
  intros hTotal hUnanswered hTime
  rw [hTotal, hUnanswered, hTime]
  sorry

end grayson_time_per_answer_l183_183244


namespace part_a_part_b_l183_183198

noncomputable def probability_Peter_satisfied : ℚ :=
  let total_people := 100
  let men := 50
  let women := 50
  let P_both_men := (men - 1 : ℚ)/ (total_people - 1 : ℚ) * (men - 2 : ℚ)/ (total_people - 2 : ℚ)
  1 - P_both_men

theorem part_a : probability_Peter_satisfied = 25 / 33 := 
  sorry

noncomputable def expected_satisfied_men : ℚ :=
  let men := 50
  probability_Peter_satisfied * men

theorem part_b : expected_satisfied_men = 1250 / 33 := 
  sorry

end part_a_part_b_l183_183198


namespace fraction_zero_iff_l183_183285

theorem fraction_zero_iff (x : ℝ) (h₁ : (x - 1) / (2 * x - 4) = 0) (h₂ : 2 * x - 4 ≠ 0) : x = 1 := sorry

end fraction_zero_iff_l183_183285


namespace simplify_expression_l183_183684

variable (x : ℝ)

theorem simplify_expression :
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 = -x^2 + 23 * x - 3 :=
sorry

end simplify_expression_l183_183684


namespace complement_intersection_l183_183499

open Set

variable {R : Type} [LinearOrderedField R]

def P : Set R := {x | x^2 - 2*x ≥ 0}
def Q : Set R := {x | 1 < x ∧ x ≤ 3}

theorem complement_intersection : (compl P ∩ Q) = {x : R | 1 < x ∧ x < 2} := by
  sorry

end complement_intersection_l183_183499


namespace votes_cast_l183_183200

theorem votes_cast (total_votes : ℕ) 
  (h1 : (3/8 : ℚ) * total_votes = 45)
  (h2 : (1/4 : ℚ) * total_votes = (1/4 : ℚ) * 120) : 
  total_votes = 120 := 
by
  sorry

end votes_cast_l183_183200


namespace find_some_number_l183_183992

theorem find_some_number (x : ℤ) (h : 45 - (28 - (x - (15 - 20))) = 59) : x = 37 :=
by
  sorry

end find_some_number_l183_183992


namespace remainder_of_product_divided_by_10_l183_183669

theorem remainder_of_product_divided_by_10 :
  let a := 2457
  let b := 6273
  let c := 91409
  (a * b * c) % 10 = 9 :=
by
  sorry

end remainder_of_product_divided_by_10_l183_183669


namespace quadratic_root_in_interval_l183_183067

theorem quadratic_root_in_interval 
  (a b c : ℝ) 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end quadratic_root_in_interval_l183_183067


namespace solve_inequality_l183_183017

theorem solve_inequality (x : ℝ) : -1/3 * x + 1 ≤ -5 → x ≥ 18 := 
  sorry

end solve_inequality_l183_183017


namespace greatest_m_value_l183_183900

theorem greatest_m_value (x y z u : ℕ) (hx : x ≥ y) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ m ≤ x / y :=
sorry

end greatest_m_value_l183_183900


namespace x12_is_1_l183_183364

noncomputable def compute_x12 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : ℝ :=
  x ^ 12

theorem x12_is_1 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : compute_x12 x h = 1 :=
  sorry

end x12_is_1_l183_183364


namespace evaluate_expression_l183_183727

variable (b : ℝ)

theorem evaluate_expression : ( ( (b^(16/8))^(1/4) )^3 * ( (b^(16/4))^(1/8) )^3 ) = b^3 := by
  sorry

end evaluate_expression_l183_183727


namespace total_pay_is_186_l183_183878

-- Define the conditions
def regular_rate : ℕ := 3 -- dollars per hour
def regular_hours : ℕ := 40 -- hours
def overtime_rate_multiplier : ℕ := 2
def overtime_hours : ℕ := 11

-- Calculate the regular pay
def regular_pay : ℕ := regular_hours * regular_rate

-- Calculate the overtime pay
def overtime_rate : ℕ := regular_rate * overtime_rate_multiplier
def overtime_pay : ℕ := overtime_hours * overtime_rate

-- Calculate the total pay
def total_pay : ℕ := regular_pay + overtime_pay

-- The statement to be proved
theorem total_pay_is_186 : total_pay = 186 :=
by 
  sorry

end total_pay_is_186_l183_183878


namespace pow_mod_cycle_l183_183151

theorem pow_mod_cycle (n : ℕ) : 3^250 % 13 = 3 := 
by
  sorry

end pow_mod_cycle_l183_183151


namespace selection_plans_count_l183_183226

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the number of subjects
def num_subjects : ℕ := 3

-- Prove that the number of selection plans is 120
theorem selection_plans_count :
  (Nat.choose total_students num_subjects) * (num_subjects.factorial) = 120 := 
by
  sorry

end selection_plans_count_l183_183226


namespace solve_for_x_l183_183131

theorem solve_for_x (x : ℝ) (h : x ≠ 2) : (7 * x) / (x - 2) - 5 / (x - 2) = 3 / (x - 2) → x = 8 / 7 :=
by
  sorry

end solve_for_x_l183_183131


namespace triangular_array_sum_digits_l183_183891

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 3780) : (N / 10 + N % 10) = 15 :=
sorry

end triangular_array_sum_digits_l183_183891


namespace cistern_emptying_time_l183_183056

noncomputable def cistern_time_without_tap (tap_rate : ℕ) (empty_time_with_tap : ℕ) (cistern_volume : ℕ) : ℕ := 
  let tap_total := tap_rate * empty_time_with_tap
  let leaked_volume := cistern_volume - tap_total
  let leak_rate := leaked_volume / empty_time_with_tap
  cistern_volume / leak_rate

theorem cistern_emptying_time :
  cistern_time_without_tap 4 24 480 = 30 := 
by
  unfold cistern_time_without_tap
  norm_num

end cistern_emptying_time_l183_183056


namespace bill_toys_l183_183247

variable (B H : ℕ)

theorem bill_toys (h1 : H = B / 2 + 9) (h2 : B + H = 99) : B = 60 := by
  sorry

end bill_toys_l183_183247


namespace find_multiple_l183_183996
-- Importing Mathlib to access any necessary math definitions.

-- Define the constants based on the given conditions.
def Darwin_money : ℝ := 45
def Mia_money : ℝ := 110
def additional_amount : ℝ := 20

-- The Lean theorem which encapsulates the proof problem.
theorem find_multiple (x : ℝ) : 
  Mia_money = x * Darwin_money + additional_amount → x = 2 :=
by
  sorry

end find_multiple_l183_183996


namespace find_expression_value_l183_183888

theorem find_expression_value : 1 + 2 * 3 - 4 + 5 = 8 :=
by
  sorry

end find_expression_value_l183_183888


namespace solve_system_l183_183270

def system_of_equations : Prop :=
  ∃ (x y : ℝ), 2 * x - y = 6 ∧ x + 2 * y = -2 ∧ x = 2 ∧ y = -2

theorem solve_system : system_of_equations := by
  sorry

end solve_system_l183_183270


namespace josie_total_animals_is_correct_l183_183608

noncomputable def totalAnimals : Nat :=
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  let giraffes := antelopes + 15
  let lions := leopards + giraffes
  let elephants := 3 * lions
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants

theorem josie_total_animals_is_correct : totalAnimals = 1308 := by
  sorry

end josie_total_animals_is_correct_l183_183608


namespace max_roses_purchasable_l183_183880

theorem max_roses_purchasable 
  (price_individual : ℝ) (price_dozen : ℝ) (price_two_dozen : ℝ) (price_five_dozen : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) (total_money : ℝ) : 
  (price_individual = 4.50) →
  (price_dozen = 36) →
  (price_two_dozen = 50) →
  (price_five_dozen = 110) →
  (discount_threshold = 36) →
  (discount_rate = 0.10) →
  (total_money = 680) →
  ∃ (roses : ℕ), roses = 364 :=
by
  -- Definitions based on conditions
  intros
  -- The proof steps have been omitted for brevity
  sorry

end max_roses_purchasable_l183_183880


namespace min_digits_fraction_l183_183760

def minDigitsToRightOfDecimal (n : ℕ) : ℕ :=
  -- This represents the minimum number of digits needed to express n / (2^15 * 5^7)
  -- as a decimal.
  -- The actual function body is hypothetical and not implemented here.
  15

theorem min_digits_fraction :
  minDigitsToRightOfDecimal 987654321 = 15 :=
by
  sorry

end min_digits_fraction_l183_183760


namespace num_ways_arrange_passengers_l183_183324

theorem num_ways_arrange_passengers 
  (seats : ℕ) (passengers : ℕ) (consecutive_empty : ℕ)
  (h1 : seats = 10) (h2 : passengers = 4) (h3 : consecutive_empty = 5) :
  ∃ ways, ways = 480 := by
  sorry

end num_ways_arrange_passengers_l183_183324


namespace same_terminal_side_l183_183600

theorem same_terminal_side
  (k : ℤ)
  (angle1 := (π / 5))
  (angle2 := (21 * π / 5)) :
  ∃ k : ℤ, angle2 = 2 * k * π + angle1 := by
  sorry

end same_terminal_side_l183_183600


namespace cos_double_angle_l183_183931

variable (α : ℝ)

theorem cos_double_angle (h1 : 0 < α ∧ α < π / 2) 
                         (h2 : Real.cos ( α + π / 4) = 3 / 5) : 
    Real.cos (2 * α) = 24 / 25 :=
by
  sorry

end cos_double_angle_l183_183931


namespace find_values_of_a_and_b_l183_183310

theorem find_values_of_a_and_b (a b x y : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < x) (h4: 0 < y) 
  (h5 : a + b = 10) (h6 : a / x + b / y = 1) (h7 : x + y = 18) : 
  (a = 2 ∧ b = 8) ∨ (a = 8 ∧ b = 2) := 
sorry

end find_values_of_a_and_b_l183_183310


namespace book_pricing_and_min_cost_l183_183988

-- Define the conditions
def price_relation (a : ℝ) (ps_price : ℝ) : Prop :=
  ps_price = 1.2 * a

def book_count_relation (a : ℝ) (lit_count ps_count : ℕ) : Prop :=
  lit_count = 1200 / a ∧ ps_count = 1200 / (1.2 * a) ∧ lit_count - ps_count = 10

def min_cost_condition (x : ℕ) : Prop :=
  x ≤ 600

def total_cost (x : ℕ) : ℝ :=
  20 * x + 24 * (1000 - x)

-- The theorem combining all parts
theorem book_pricing_and_min_cost:
  ∃ (a : ℝ) (ps_price : ℝ) (lit_count ps_count : ℕ),
    price_relation a ps_price ∧
    book_count_relation a lit_count ps_count ∧
    a = 20 ∧ ps_price = 24 ∧
    (∀ (x : ℕ), min_cost_condition x → total_cost x ≥ 21600) ∧
    (total_cost 600 = 21600) :=
by
  sorry

end book_pricing_and_min_cost_l183_183988


namespace second_tray_holds_l183_183769

-- The conditions and the given constants
variables (x : ℕ) (h1 : 2 * x - 20 = 500)

-- The theorem proving the number of cups the second tray holds is 240 
theorem second_tray_holds (h2 : x = 260) : x - 20 = 240 := by
  sorry

end second_tray_holds_l183_183769


namespace chives_planted_l183_183553

theorem chives_planted (total_rows : ℕ) (plants_per_row : ℕ)
  (parsley_rows : ℕ) (rosemary_rows : ℕ) :
  total_rows = 20 →
  plants_per_row = 10 →
  parsley_rows = 3 →
  rosemary_rows = 2 →
  (plants_per_row * (total_rows - (parsley_rows + rosemary_rows))) = 150 :=
by
  intro h1 h2 h3 h4
  sorry

end chives_planted_l183_183553


namespace find_y_l183_183298

theorem find_y (y : ℕ) (h : (2 * y) / 5 = 10) : y = 25 :=
sorry

end find_y_l183_183298


namespace max_xy_value_l183_183664

theorem max_xy_value (x y : ℝ) (h : x^2 + y^2 + 3 * x * y = 2015) : xy <= 403 :=
sorry

end max_xy_value_l183_183664


namespace freight_cost_minimization_l183_183661

-- Define the main parameters: tonnage and costs for the trucks.
def freight_cost (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  65 * num_seven_ton_trucks + 50 * num_five_ton_trucks

-- Define the total transported capacity by the two types of trucks.
def total_capacity (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  7 * num_seven_ton_trucks + 5 * num_five_ton_trucks

-- Define the minimum freight cost given the conditions.
def minimum_freight_cost := 685

-- The theorem we want to prove.
theorem freight_cost_minimization : ∃ x y : ℕ, total_capacity x y ≥ 73 ∧
  (freight_cost x y = minimum_freight_cost) :=
by
  sorry

end freight_cost_minimization_l183_183661


namespace student_B_speed_l183_183340

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end student_B_speed_l183_183340


namespace no_solutions_l183_183370

theorem no_solutions (x : ℝ) (h : x ≠ 0) : 4 * Real.sin x - 3 * Real.cos x ≠ 5 + 1 / |x| := 
by
  sorry

end no_solutions_l183_183370


namespace mabel_counts_sharks_l183_183299

theorem mabel_counts_sharks 
    (fish_day1 : ℕ) 
    (fish_day2 : ℕ) 
    (shark_percentage : ℚ) 
    (total_fish : ℕ) 
    (total_sharks : ℕ) 
    (h1 : fish_day1 = 15) 
    (h2 : fish_day2 = 3 * fish_day1) 
    (h3 : shark_percentage = 0.25) 
    (h4 : total_fish = fish_day1 + fish_day2) 
    (h5 : total_sharks = total_fish * shark_percentage) : 
    total_sharks = 15 := 
by {
  sorry
}

end mabel_counts_sharks_l183_183299


namespace nonneg_reals_sum_to_one_implies_ineq_l183_183384

theorem nonneg_reals_sum_to_one_implies_ineq
  (x y z : ℝ)
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 :=
sorry

end nonneg_reals_sum_to_one_implies_ineq_l183_183384


namespace apple_price_33_kgs_l183_183526

theorem apple_price_33_kgs (l q : ℕ) (h1 : 30 * l + 6 * q = 366) (h2 : 15 * l = 150) : 
  30 * l + 3 * q = 333 :=
by
  sorry

end apple_price_33_kgs_l183_183526


namespace next_perfect_cube_l183_183929

theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^3) : 
  ∃ m : ℕ, m^3 = x + 3 * (x^(1/3))^2 + 3 * x^(1/3) + 1 :=
by
  sorry

end next_perfect_cube_l183_183929


namespace prob_fourth_black_ball_is_half_l183_183101

-- Define the conditions
def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 4
def total_balls : ℕ := num_red_balls + num_black_balls

-- The theorem stating that the probability of drawing a black ball on the fourth draw is 1/2
theorem prob_fourth_black_ball_is_half : 
  (num_black_balls : ℚ) / (total_balls : ℚ) = 1 / 2 :=
by
  sorry

end prob_fourth_black_ball_is_half_l183_183101


namespace carla_water_drank_l183_183431

theorem carla_water_drank (W S : ℝ) (h1 : W + S = 54) (h2 : S = 3 * W - 6) : W = 15 :=
by
  sorry

end carla_water_drank_l183_183431


namespace lim_sup_eq_Union_lim_inf_l183_183535

open Set

theorem lim_sup_eq_Union_lim_inf
  (Ω : Type*)
  (A : ℕ → Set Ω) :
  (⋂ n, ⋃ k ≥ n, A k) = ⋃ (n_infty : ℕ → ℕ) (hn : StrictMono n_infty), ⋃ n, ⋂ k ≥ n, A (n_infty k) :=
by
  sorry

end lim_sup_eq_Union_lim_inf_l183_183535


namespace Xiaoli_estimate_is_larger_l183_183269

variables {x y x' y' : ℝ}

theorem Xiaoli_estimate_is_larger (h1 : x > y) (h2 : y > 0) (h3 : x' = 1.01 * x) (h4 : y' = 0.99 * y) : x' - y' > x - y :=
by sorry

end Xiaoli_estimate_is_larger_l183_183269


namespace anika_more_than_twice_reeta_l183_183586

theorem anika_more_than_twice_reeta (R A M : ℕ) (h1 : R = 20) (h2 : A + R = 64) (h3 : A = 2 * R + M) : M = 4 :=
by
  sorry

end anika_more_than_twice_reeta_l183_183586


namespace total_movies_attended_l183_183308

-- Defining the conditions for Timothy's movie attendance
def Timothy_2009 := 24
def Timothy_2010 := Timothy_2009 + 7

-- Defining the conditions for Theresa's movie attendance
def Theresa_2009 := Timothy_2009 / 2
def Theresa_2010 := Timothy_2010 * 2

-- Prove that the total number of movies Timothy and Theresa went to in both years is 129
theorem total_movies_attended :
  (Timothy_2009 + Timothy_2010 + Theresa_2009 + Theresa_2010) = 129 :=
by
  -- proof goes here
  sorry

end total_movies_attended_l183_183308


namespace inequality_solution_l183_183842

theorem inequality_solution (x : ℝ) (h : 3 * x - 5 > 11 - 2 * x) : x > 16 / 5 := 
sorry

end inequality_solution_l183_183842


namespace friend_balloons_count_l183_183930

-- Definitions of the conditions
def balloons_you_have : ℕ := 7
def balloons_difference : ℕ := 2

-- Proof problem statement
theorem friend_balloons_count : (balloons_you_have - balloons_difference) = 5 :=
by
  sorry

end friend_balloons_count_l183_183930


namespace problem_statement_l183_183459

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2004 = 2005 :=
sorry

end problem_statement_l183_183459


namespace solution_set_inequality_l183_183965

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 :=
by {
  sorry -- proof omitted
}

end solution_set_inequality_l183_183965


namespace sufficient_but_not_necessary_l183_183137

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 1) → (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 2 ∧ b > 1)) :=
by
  sorry

end sufficient_but_not_necessary_l183_183137


namespace combined_prism_volume_is_66_l183_183158

noncomputable def volume_of_combined_prisms
  (length_rect : ℝ) (width_rect : ℝ) (height_rect : ℝ)
  (base_tri : ℝ) (height_tri : ℝ) (length_tri : ℝ) : ℝ :=
  let volume_rect := length_rect * width_rect * height_rect
  let area_tri := (1 / 2) * base_tri * height_tri
  let volume_tri := area_tri * length_tri
  volume_rect + volume_tri

theorem combined_prism_volume_is_66 :
  volume_of_combined_prisms 6 4 2 3 3 4 = 66 := by
  sorry

end combined_prism_volume_is_66_l183_183158


namespace triangle_third_side_l183_183095

noncomputable def length_of_third_side
  (a b : ℝ) (θ : ℝ) (cosθ : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2 * a * b * cosθ)

theorem triangle_third_side : 
  length_of_third_side 8 15 (Real.pi / 6) (Real.cos (Real.pi / 6)) = Real.sqrt (289 - 120 * Real.sqrt 3) :=
by
  sorry

end triangle_third_side_l183_183095


namespace complement_union_l183_183315

def is_pos_int_less_than_9 (x : ℕ) : Prop := x > 0 ∧ x < 9

def U : Set ℕ := {x | is_pos_int_less_than_9 x}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union :
  (U \ (M ∪ N)) = {2, 4, 8} :=
by
  sorry

end complement_union_l183_183315


namespace output_increase_percentage_l183_183570

theorem output_increase_percentage (O : ℝ) (P : ℝ) (h : (O * (1 + P / 100) * 1.60) * 0.5682 = O) : P = 10.09 :=
by 
  sorry

end output_increase_percentage_l183_183570


namespace negation_of_proposition_l183_183100

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∃ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_of_proposition_l183_183100


namespace arithmetic_sequence_a5_l183_183464

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- The terms of the arithmetic sequence

theorem arithmetic_sequence_a5 :
  (∀ (n : ℕ), a_n n = a_n 0 + n * (a_n 1 - a_n 0)) →
  a_n 1 = 1 →
  a_n 1 + a_n 3 = 16 →
  a_n 4 = 15 :=
by {
  -- Proof omission, ensure these statements are correct with sorry
  sorry
}

end arithmetic_sequence_a5_l183_183464


namespace value_of_expression_l183_183738

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end value_of_expression_l183_183738


namespace circumference_given_area_l183_183720

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def circumference_of_circle (r : ℝ) : ℝ := 2 * Real.pi * r

theorem circumference_given_area :
  (∃ r : ℝ, area_of_circle r = 616) →
  circumference_of_circle 14 = 2 * Real.pi * 14 :=
by
  sorry

end circumference_given_area_l183_183720


namespace num_ways_product_72_l183_183303

def num_ways_product (n : ℕ) : ℕ := sorry  -- Definition for D(n), the number of ways to write n as a product of integers greater than 1

def example_integer := 72  -- Given integer n

theorem num_ways_product_72 : num_ways_product example_integer = 67 := by 
  sorry

end num_ways_product_72_l183_183303


namespace range_of_a_l183_183563

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l183_183563


namespace prob_A_wins_match_is_correct_l183_183486

/-- Definitions -/

def prob_A_wins_game : ℝ := 0.6

def prob_B_wins_game : ℝ := 1 - prob_A_wins_game

def prob_A_wins_match (p: ℝ) : ℝ :=
  p * p * (1 - p) + p * (1 - p) * p + p * p

/-- Theorem -/

theorem prob_A_wins_match_is_correct : 
  prob_A_wins_match prob_A_wins_game = 0.648 :=
by
  sorry

end prob_A_wins_match_is_correct_l183_183486


namespace negation_of_universal_l183_183590

variable (P : ℝ → Prop)
def pos (x : ℝ) : Prop := x > 0
def gte_zero (x : ℝ) : Prop := x^2 - x ≥ 0
def lt_zero (x : ℝ) : Prop := x^2 - x < 0

theorem negation_of_universal :
  ¬ (∀ x, pos x → gte_zero x) ↔ ∃ x, pos x ∧ lt_zero x := by
  sorry

end negation_of_universal_l183_183590


namespace inequality_proof_l183_183047

theorem inequality_proof {x y z : ℝ} (hxy : 0 < x) (hyz : 0 < y) (hzx : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (x + z) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
sorry

end inequality_proof_l183_183047


namespace product_correlation_function_l183_183443

open ProbabilityTheory

/-
Theorem: Given two centered and uncorrelated random functions \( \dot{X}(t) \) and \( \dot{Y}(t) \),
the correlation function of their product \( Z(t) = \dot{X}(t) \dot{Y}(t) \) is the product of their correlation functions.
-/
theorem product_correlation_function 
  (X Y : ℝ → ℝ)
  (hX_centered : ∀ t, (∫ x, X t ∂x) = 0) 
  (hY_centered : ∀ t, (∫ y, Y t ∂y) = 0)
  (h_uncorrelated : ∀ t1 t2, ∫ x, X t1 * Y t2 ∂x = (∫ x, X t1 ∂x) * (∫ y, Y t2 ∂y)) :
  ∀ t1 t2, 
  (∫ x, (X t1 * Y t1) * (X t2 * Y t2) ∂x) = 
  (∫ x, X t1 * X t2 ∂x) * (∫ y, Y t1 * Y t2 ∂y) :=
by
  sorry

end product_correlation_function_l183_183443


namespace volume_calc_l183_183883

noncomputable
def volume_of_open_box {l w : ℕ} (sheet_length : l = 48) (sheet_width : w = 38) (cut_length : ℕ) (cut_length_eq : cut_length = 8) : ℕ :=
  let new_length := l - 2 * cut_length
  let new_width := w - 2 * cut_length
  let height := cut_length
  new_length * new_width * height

theorem volume_calc : volume_of_open_box (sheet_length := rfl) (sheet_width := rfl) (cut_length := 8) (cut_length_eq := rfl) = 5632 :=
sorry

end volume_calc_l183_183883


namespace square_side_length_l183_183639

theorem square_side_length (s : ℝ) (h : s^2 = 9/16) : s = 3/4 :=
sorry

end square_side_length_l183_183639


namespace speed_of_A_l183_183112

theorem speed_of_A (B_speed : ℕ) (crossings : ℕ) (H : B_speed = 3 ∧ crossings = 5 ∧ 5 * (1 / (x + B_speed)) = 1) : x = 2 :=
by
  sorry

end speed_of_A_l183_183112


namespace price_of_first_variety_l183_183238

theorem price_of_first_variety
  (p2 : ℝ) (p3 : ℝ) (r : ℝ) (w : ℝ)
  (h1 : p2 = 135)
  (h2 : p3 = 177.5)
  (h3 : r = 154)
  (h4 : w = 4) :
  ∃ p1 : ℝ, 1 * p1 + 1 * p2 + 2 * p3 = w * r ∧ p1 = 126 :=
by {
  sorry
}

end price_of_first_variety_l183_183238


namespace sum_of_arithmetic_sequence_l183_183898

-- Given conditions in the problem
axiom arithmetic_sequence (a : ℕ → ℤ): Prop
axiom are_roots (a b : ℤ): ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a = p ∨ a = q) ∧ (b = p ∨ b = q)

-- The equivalent proof problem statement
theorem sum_of_arithmetic_sequence (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a)
  (h2 : ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a 2 = p ∨ a 2 = q) ∧ (a 11 = p ∨ a 11 = q)):
  a 5 + a 8 = 3 :=
sorry

end sum_of_arithmetic_sequence_l183_183898


namespace complement_of_union_l183_183015

open Set

variable (U A B : Set ℕ)
variable (u_def : U = {0, 1, 2, 3, 4, 5, 6})
variable (a_def : A = {1, 3})
variable (b_def : B = {3, 5})

theorem complement_of_union :
  (U \ (A ∪ B)) = {0, 2, 4, 6} :=
by
  sorry

end complement_of_union_l183_183015


namespace expression_evaluation_l183_183536

theorem expression_evaluation (x : ℝ) (h : 2 * x - 7 = 8 * x - 1) : 5 * (x - 3) = -20 :=
by
  sorry

end expression_evaluation_l183_183536


namespace basketball_players_l183_183502

theorem basketball_players {total : ℕ} (total_boys : total = 22) 
                           (football_boys : ℕ) (football_boys_count : football_boys = 15) 
                           (neither_boys : ℕ) (neither_boys_count : neither_boys = 3) 
                           (both_boys : ℕ) (both_boys_count : both_boys = 18) : 
                           (total - neither_boys = 19) := 
by
  sorry

end basketball_players_l183_183502


namespace compound_interest_comparison_l183_183083

theorem compound_interest_comparison :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  (P * (1 + r_monthly)^((12 * t)) > P * (1 + r_annual)^t) :=
by
  sorry

end compound_interest_comparison_l183_183083


namespace five_ab4_is_perfect_square_l183_183253

theorem five_ab4_is_perfect_square (a b : ℕ) (h : 5000 ≤ 5000 + 100 * a + 10 * b + 4 ∧ 5000 + 100 * a + 10 * b + 4 ≤ 5999) :
    ∃ n, n^2 = 5000 + 100 * a + 10 * b + 4 → a + b = 9 :=
by
  sorry

end five_ab4_is_perfect_square_l183_183253


namespace luke_can_see_silvia_for_22_point_5_minutes_l183_183973

/--
Luke is initially 0.75 miles behind Silvia. Luke rollerblades at 10 mph and Silvia cycles 
at 6 mph. Luke can see Silvia until she is 0.75 miles behind him. Prove that Luke can see 
Silvia for a total of 22.5 minutes.
-/
theorem luke_can_see_silvia_for_22_point_5_minutes :
    let distance := (3 / 4 : ℝ)
    let luke_speed := (10 : ℝ)
    let silvia_speed := (6 : ℝ)
    let relative_speed := luke_speed - silvia_speed
    let time_to_reach := distance / relative_speed
    let total_time := 2 * time_to_reach * 60 
    total_time = 22.5 :=
by
    sorry

end luke_can_see_silvia_for_22_point_5_minutes_l183_183973


namespace product_squared_inequality_l183_183401

theorem product_squared_inequality (n : ℕ) (a : Fin n → ℝ) (h : (Finset.univ.prod (λ i => a i)) = 1) :
    (Finset.univ.prod (λ i => (1 + (a i)^2))) ≥ 2^n := 
sorry

end product_squared_inequality_l183_183401


namespace decimal_to_fraction_l183_183978

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l183_183978


namespace primes_with_consecutives_l183_183646

-- Define what it means for a number to be prime
def is_prime (n : Nat) := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬ (n % m = 0)

-- Define the main theorem to prove
theorem primes_with_consecutives (p : Nat) : is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  sorry

end primes_with_consecutives_l183_183646


namespace factor_expression_l183_183835

theorem factor_expression (y : ℝ) : 
  3 * y * (2 * y + 5) + 4 * (2 * y + 5) = (3 * y + 4) * (2 * y + 5) :=
by
  sorry

end factor_expression_l183_183835


namespace angle_diff_complement_supplement_l183_183706

theorem angle_diff_complement_supplement (α : ℝ) : (180 - α) - (90 - α) = 90 := by
  sorry

end angle_diff_complement_supplement_l183_183706


namespace test_scores_order_l183_183263

def kaleana_score : ℕ := 75

variable (M Q S : ℕ)

-- Assuming conditions from the problem
axiom h1 : Q = kaleana_score
axiom h2 : M < max Q S
axiom h3 : S > min Q M
axiom h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S

-- Theorem statement
theorem test_scores_order (M Q S : ℕ) (h1 : Q = kaleana_score) (h2 : M < max Q S) (h3 : S > min Q M) (h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S) :
  M < Q ∧ Q < S :=
sorry

end test_scores_order_l183_183263


namespace fourth_person_height_l183_183921

theorem fourth_person_height 
  (H : ℕ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) : 
  H + 10 = 85 :=
by
  sorry

end fourth_person_height_l183_183921


namespace FashionDesignNotInServiceAreas_l183_183623

-- Define the service areas of Digital China
def ServiceAreas (x : String) : Prop :=
  x = "Understanding the situation of soil and water loss in the Yangtze River Basin" ∨
  x = "Understanding stock market trends" ∨
  x = "Wanted criminals"

-- Prove that "Fashion design" is not in the service areas of Digital China
theorem FashionDesignNotInServiceAreas : ¬ ServiceAreas "Fashion design" :=
sorry

end FashionDesignNotInServiceAreas_l183_183623


namespace area_of_flowerbed_l183_183488

theorem area_of_flowerbed :
  ∀ (a b : ℕ), 2 * (a + b) = 24 → b + 1 = 3 * (a + 1) → 
  let shorter_side := 3 * a
  let longer_side := 3 * b
  shorter_side * longer_side = 144 :=
by
  sorry

end area_of_flowerbed_l183_183488


namespace translation_coordinates_l183_183813

theorem translation_coordinates
  (a b : ℝ)
  (h₁ : 4 = a + 2)
  (h₂ : -3 = b - 6) :
  (a, b) = (2, 3) :=
by
  sorry

end translation_coordinates_l183_183813


namespace Taehyung_walked_distance_l183_183009

variable (step_distance : ℝ) (steps_per_set : ℕ) (num_sets : ℕ)
variable (h1 : step_distance = 0.45)
variable (h2 : steps_per_set = 90)
variable (h3 : num_sets = 13)

theorem Taehyung_walked_distance :
  (steps_per_set * step_distance) * num_sets = 526.5 :=
by 
  rw [h1, h2, h3]
  sorry

end Taehyung_walked_distance_l183_183009


namespace arithmetic_sequence_eightieth_term_l183_183928

open BigOperators

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_eightieth_term :
  ∀ (d : ℝ),
  arithmetic_sequence 3 d 21 = 41 →
  arithmetic_sequence 3 d 80 = 153.1 :=
by
  intros
  sorry

end arithmetic_sequence_eightieth_term_l183_183928


namespace a_range_iff_l183_183671

theorem a_range_iff (a x : ℝ) (h1 : x < 3) (h2 : (a - 1) * x < a + 3) : 
  1 ≤ a ∧ a < 3 := 
by
  sorry

end a_range_iff_l183_183671


namespace least_multiple_36_sum_digits_l183_183463

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem least_multiple_36_sum_digits :
  ∃ n : ℕ, n = 36 ∧ (36 ∣ n) ∧ (9 ∣ digit_sum n) ∧ (∀ m : ℕ, (36 ∣ m) ∧ (9 ∣ digit_sum m) → 36 ≤ m) :=
by sorry

end least_multiple_36_sum_digits_l183_183463


namespace quadratic_completion_l183_183500

theorem quadratic_completion :
  (∀ x : ℝ, (∃ a h k : ℝ, (x ^ 2 - 2 * x - 1 = a * (x - h) ^ 2 + k) ∧ (a = 1) ∧ (h = 1) ∧ (k = -2))) :=
sorry

end quadratic_completion_l183_183500


namespace minimum_value_of_f_l183_183624

variable (a k : ℝ)
variable (k_gt_1 : k > 1)
variable (a_gt_0 : a > 0)

noncomputable def f (x : ℝ) : ℝ := k * Real.sqrt (a^2 + x^2) - x

theorem minimum_value_of_f : ∃ x_0, ∀ x, f a k x ≥ f a k x_0 ∧ f a k x_0 = a * Real.sqrt (k^2 - 1) :=
by
  sorry

end minimum_value_of_f_l183_183624


namespace find_m_value_l183_183217

-- Definitions based on conditions
variables {a b m : ℝ} (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1)

-- Lean 4 statement of the problem
theorem find_m_value (ha : 2 ^ a = m) (hb : 5 ^ b = m) (h : 1 / a + 1 / b = 1) : m = 10 := sorry

end find_m_value_l183_183217


namespace quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l183_183242

-- Prove the range of k for distinct real roots
theorem quadratic_distinct_real_roots (k: ℝ) (h: k ≠ 0) : 
  (40 * k + 16 > 0) ↔ (k > -2/5) := 
by sorry

-- Prove the solutions for the quadratic equation when k = 1
theorem quadratic_solutions_k_eq_1 (x: ℝ) : 
  (x^2 - 6*x - 5 = 0) ↔ 
  (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14) := 
by sorry

end quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l183_183242


namespace Peter_bought_5_kilos_of_cucumbers_l183_183852

/-- 
Peter carried $500 to the market. 
He bought 6 kilos of potatoes for $2 per kilo, 
9 kilos of tomato for $3 per kilo, 
some kilos of cucumbers for $4 per kilo, 
and 3 kilos of bananas for $5 per kilo. 
After buying all these items, Peter has $426 remaining. 
How many kilos of cucumbers did Peter buy? 
-/
theorem Peter_bought_5_kilos_of_cucumbers : 
   ∃ (kilos_cucumbers : ℕ),
   (500 - (6 * 2 + 9 * 3 + 3 * 5 + kilos_cucumbers * 4) = 426) →
   kilos_cucumbers = 5 :=
sorry

end Peter_bought_5_kilos_of_cucumbers_l183_183852


namespace minimum_value_y_l183_183585

theorem minimum_value_y (x : ℝ) (h : x ≥ 1) : 5*x^2 - 8*x + 20 ≥ 13 :=
by {
  sorry
}

end minimum_value_y_l183_183585


namespace rtl_to_conventional_notation_l183_183868

theorem rtl_to_conventional_notation (a b c d e : ℚ) :
  (a / (b - (c * (d + e)))) = a / (b - c * (d + e)) := by
  sorry

end rtl_to_conventional_notation_l183_183868


namespace car_speed_after_modifications_l183_183309

theorem car_speed_after_modifications (s : ℕ) (p : ℝ) (w : ℕ) :
  s = 150 →
  p = 0.30 →
  w = 10 →
  s + (p * s) + w = 205 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  done

end car_speed_after_modifications_l183_183309


namespace dogs_left_l183_183914

-- Define the conditions
def total_dogs : ℕ := 50
def dog_houses : ℕ := 17

-- Statement to prove the number of dogs left
theorem dogs_left : (total_dogs % dog_houses) = 16 :=
by sorry

end dogs_left_l183_183914


namespace total_students_l183_183173

-- Definitions based on conditions
variable (T M Z : ℕ)  -- T for Tina's students, M for Maura's students, Z for Zack's students

-- Conditions as hypotheses
axiom h1 : T = M  -- Tina's classroom has the same amount of students as Maura's
axiom h2 : Z = (T + M) / 2  -- Zack's classroom has half the amount of total students between Tina and Maura's classrooms
axiom h3 : Z = 23  -- There are 23 students in Zack's class when present

-- Proof statement
theorem total_students : T + M + Z = 69 :=
  sorry

end total_students_l183_183173


namespace equivalence_statements_l183_183571

variables (P Q : Prop)

theorem equivalence_statements :
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end equivalence_statements_l183_183571


namespace pow_mod_1000_of_6_eq_296_l183_183599

theorem pow_mod_1000_of_6_eq_296 : (6 ^ 1993) % 1000 = 296 := by
  sorry

end pow_mod_1000_of_6_eq_296_l183_183599


namespace exists_x_nat_l183_183557

theorem exists_x_nat (a c : ℕ) (b : ℤ) : ∃ x : ℕ, (a^x + x) % c = b % c :=
by
  sorry

end exists_x_nat_l183_183557


namespace sqrt_sum_odds_l183_183748

theorem sqrt_sum_odds : 
  (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9) + Real.sqrt (1+3+5+7+9+11)) = 21 := 
by
  sorry

end sqrt_sum_odds_l183_183748


namespace inequality_holds_iff_b_lt_a_l183_183915

theorem inequality_holds_iff_b_lt_a (a b : ℝ) :
  (∀ x : ℝ, (a + 1) * x^2 + a * x + a > b * (x^2 + x + 1)) ↔ b < a :=
by
  sorry

end inequality_holds_iff_b_lt_a_l183_183915


namespace no_values_of_expression_l183_183993

theorem no_values_of_expression (x : ℝ) (h : x^2 - 4 * x + 4 < 0) :
  ¬ ∃ y, y = x^2 + 4 * x + 5 :=
by
  sorry

end no_values_of_expression_l183_183993


namespace sum_of_consecutive_numbers_mod_13_l183_183203

theorem sum_of_consecutive_numbers_mod_13 :
  ((8930 + 8931 + 8932 + 8933 + 8934) % 13) = 5 :=
by
  sorry

end sum_of_consecutive_numbers_mod_13_l183_183203


namespace value_of_x_l183_183116

noncomputable def k := 9

theorem value_of_x (y : ℝ) (h1 : y = 3) (h2 : ∀ (x : ℝ), x = 2.25 → x = k / (2 : ℝ)^2) : 
  ∃ (x : ℝ), x = 1 := by
  sorry

end value_of_x_l183_183116


namespace expected_turns_formula_l183_183086

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n +  1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1)))

theorem expected_turns_formula (n : ℕ) (h : n ≥ 1) :
  expected_turns n = n + 1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1))) :=
by
  sorry

end expected_turns_formula_l183_183086


namespace num_sides_of_length4_eq_4_l183_183031

-- Definitions of the variables and conditions
def total_sides : ℕ := 6
def total_perimeter : ℕ := 30
def side_length1 : ℕ := 7
def side_length2 : ℕ := 4

-- The conditions imposed by the problem
def is_hexagon (x y : ℕ) : Prop := x + y = total_sides
def perimeter_condition (x y : ℕ) : Prop := side_length1 * x + side_length2 * y = total_perimeter

-- The proof problem: Prove that the number of sides of length 4 is 4
theorem num_sides_of_length4_eq_4 (x y : ℕ) 
    (h1 : is_hexagon x y) 
    (h2 : perimeter_condition x y) : y = 4 :=
sorry

end num_sides_of_length4_eq_4_l183_183031


namespace perpendicular_bisector_l183_183991

theorem perpendicular_bisector (x y : ℝ) :
  (x - 2 * y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 3) → (2 * x + y - 3 = 0) :=
by
  sorry

end perpendicular_bisector_l183_183991


namespace three_digit_solutions_modulo_l183_183794

def three_digit_positive_integers (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999

theorem three_digit_solutions_modulo (h : ∃ x : ℕ, three_digit_positive_integers x ∧ 
  (2597 * x + 763) % 17 = 1459 % 17) : 
  ∃ (count : ℕ), count = 53 :=
by sorry

end three_digit_solutions_modulo_l183_183794


namespace custom_op_1_neg3_l183_183565

-- Define the custom operation as per the condition
def custom_op (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2

-- The theorem to prove that 1 * (-3) = -14 using the defined operation
theorem custom_op_1_neg3 : custom_op 1 (-3) = -14 := sorry

end custom_op_1_neg3_l183_183565


namespace gym_hours_per_week_l183_183329

-- Definitions for conditions
def timesAtGymEachWeek : ℕ := 3
def weightliftingTimeEachDay : ℕ := 1
def warmupCardioFraction : ℚ := 1 / 3

-- The theorem to prove
theorem gym_hours_per_week : (timesAtGymEachWeek * (weightliftingTimeEachDay + weightliftingTimeEachDay * warmupCardioFraction) = 4) := 
by
  sorry

end gym_hours_per_week_l183_183329


namespace arithmetic_geometric_inequality_l183_183258

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := a1 + n * d

noncomputable def geometric_sequence (b1 r : ℝ) (n : ℕ) : ℝ := b1 * r^n

theorem arithmetic_geometric_inequality
  (a1 b1 : ℝ) (d r : ℝ) (n : ℕ)
  (h_pos : 0 < a1) 
  (ha1_eq_b1 : a1 = b1) 
  (h_eq_2np1 : arithmetic_sequence a1 d (2*n+1) = geometric_sequence b1 r (2*n+1)) :
  arithmetic_sequence a1 d (n+1) ≥ geometric_sequence b1 r (n+1) :=
sorry

end arithmetic_geometric_inequality_l183_183258


namespace four_sin_t_plus_cos_2t_bounds_l183_183518

theorem four_sin_t_plus_cos_2t_bounds (t : ℝ) : -5 ≤ 4 * Real.sin t + Real.cos (2 * t) ∧ 4 * Real.sin t + Real.cos (2 * t) ≤ 3 := by
  sorry

end four_sin_t_plus_cos_2t_bounds_l183_183518


namespace percentage_of_seeds_germinated_l183_183638

theorem percentage_of_seeds_germinated (P1 P2 : ℕ) (GP1 GP2 : ℕ) (SP1 SP2 TotalGerminated TotalPlanted : ℕ) (PG : ℕ) 
  (h1 : P1 = 300) (h2 : P2 = 200) (h3 : GP1 = 60) (h4 : GP2 = 70) (h5 : SP1 = P1) (h6 : SP2 = P2)
  (h7 : TotalGerminated = GP1 + GP2) (h8 : TotalPlanted = SP1 + SP2) : 
  PG = (TotalGerminated * 100) / TotalPlanted :=
sorry

end percentage_of_seeds_germinated_l183_183638


namespace conversion_proofs_l183_183697

-- Define the necessary constants for unit conversion
def cm_to_dm2 (cm2: ℚ) : ℚ := cm2 / 100
def m3_to_dm3 (m3: ℚ) : ℚ := m3 * 1000
def dm3_to_liters (dm3: ℚ) : ℚ := dm3
def liters_to_ml (liters: ℚ) : ℚ := liters * 1000

theorem conversion_proofs :
  (cm_to_dm2 628 = 6.28) ∧
  (m3_to_dm3 4.5 = 4500) ∧
  (dm3_to_liters 3.6 = 3.6) ∧
  (liters_to_ml 0.6 = 600) :=
by
  sorry

end conversion_proofs_l183_183697


namespace example_problem_l183_183548

-- Define the numbers of students in each grade
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300

-- Define the total number of spots for the trip
def total_spots : ℕ := 40

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + juniors

-- Define the fraction of sophomores relative to the total number of students
def fraction_sophomores : ℚ := sophomores / total_students

-- Define the number of spots allocated to sophomores
def spots_sophomores : ℚ := fraction_sophomores * total_spots

-- The theorem we need to prove
theorem example_problem : spots_sophomores = 13 :=
by 
  sorry

end example_problem_l183_183548


namespace positive_number_property_l183_183792

theorem positive_number_property (x : ℝ) (h : x > 0) (hx : (x / 100) * x = 9) : x = 30 := by
  sorry

end positive_number_property_l183_183792


namespace problem_statement_l183_183122

theorem problem_statement (x : ℝ) (h : 8 * x = 4) : 150 * (1 / x) = 300 :=
by
  sorry

end problem_statement_l183_183122


namespace completing_square_solution_l183_183394

theorem completing_square_solution (x : ℝ) : x^2 - 4 * x - 22 = 0 ↔ (x - 2)^2 = 26 := sorry

end completing_square_solution_l183_183394


namespace students_passed_both_tests_l183_183517

theorem students_passed_both_tests
  (n : ℕ) (A : ℕ) (B : ℕ) (C : ℕ)
  (h1 : n = 100) 
  (h2 : A = 60) 
  (h3 : B = 40) 
  (h4 : C = 20) :
  A + B - ((n - C) - (A + B - n)) = 20 :=
by
  sorry

end students_passed_both_tests_l183_183517


namespace alice_unanswered_questions_l183_183259

theorem alice_unanswered_questions 
    (c w u : ℕ)
    (h1 : 6 * c - 2 * w + 3 * u = 120)
    (h2 : 3 * c - w = 70)
    (h3 : c + w + u = 40) :
    u = 10 :=
sorry

end alice_unanswered_questions_l183_183259


namespace cistern_fill_time_l183_183501

theorem cistern_fill_time (hA : ℝ) (hB : ℝ) (hC : ℝ) : hA = 12 → hB = 18 → hC = 15 → 
  1 / ((1 / hA) + (1 / hB) - (1 / hC)) = 180 / 13 :=
by
  intros hA_eq hB_eq hC_eq
  rw [hA_eq, hB_eq, hC_eq]
  sorry

end cistern_fill_time_l183_183501


namespace total_volume_of_5_cubes_is_135_l183_183068

-- Define the edge length of a single cube
def edge_length : ℕ := 3

-- Define the volume of a single cube
def volume_single_cube (s : ℕ) : ℕ := s^3

-- State the total volume for a given number of cubes
def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_single_cube s

-- Prove that for 5 cubes with an edge length of 3 meters, the total volume is 135 cubic meters
theorem total_volume_of_5_cubes_is_135 :
    total_volume 5 edge_length = 135 :=
by
  sorry

end total_volume_of_5_cubes_is_135_l183_183068


namespace sum_p_q_eq_21_l183_183954

theorem sum_p_q_eq_21 (p q : ℤ) :
  {x | x^2 + 6 * x - q = 0} ∩ {x | x^2 - p * x + 6 = 0} = {2} → p + q = 21 :=
by
  sorry

end sum_p_q_eq_21_l183_183954


namespace solve_for_x_l183_183257

theorem solve_for_x (x : ℝ) (h : (5 * x - 3) / (6 * x - 6) = (4 / 3)) : x = 5 / 3 :=
sorry

end solve_for_x_l183_183257


namespace quadratic_real_roots_l183_183507

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, (k + 1) * x^2 + 4 * x - 1 = 0) ↔ k ≥ -5 ∧ k ≠ -1 :=
by
  sorry

end quadratic_real_roots_l183_183507


namespace find_pairs_l183_183710

theorem find_pairs :
  ∀ (x y : ℕ), 0 < x → 0 < y → 7 ^ x - 3 * 2 ^ y = 1 → (x, y) = (1, 1) ∨ (x, y) = (2, 4) :=
by
  intros x y hx hy h
  -- Proof would go here
  sorry

end find_pairs_l183_183710


namespace number_of_adults_l183_183110

-- Define the constants and conditions of the problem.
def children : ℕ := 52
def total_seats : ℕ := 95
def empty_seats : ℕ := 14

-- Define the number of adults and prove it equals 29 given the conditions.
theorem number_of_adults : total_seats - empty_seats - children = 29 :=
by {
  sorry
}

end number_of_adults_l183_183110


namespace perfect_square_trinomial_k_l183_183032

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ a : ℤ, x^2 + k*x + 25 = (x + a)^2 ∧ a^2 = 25) → (k = 10 ∨ k = -10) :=
by
  sorry

end perfect_square_trinomial_k_l183_183032


namespace power_equation_value_l183_183770

theorem power_equation_value (n : ℕ) (h : n = 20) : n ^ (n / 2) = 102400000000000000000 := by
  sorry

end power_equation_value_l183_183770


namespace triangle_external_angle_properties_l183_183434

theorem triangle_external_angle_properties (A B C : ℝ) (hA : 0 < A ∧ A < 180) (hB : 0 < B ∧ B < 180) (hC : 0 < C ∧ C < 180) (h_sum : A + B + C = 180) :
  (∃ E1 E2 E3, E1 + E2 + E3 = 360 ∧ E1 > 90 ∧ E2 > 90 ∧ E3 <= 90) :=
by
  sorry

end triangle_external_angle_properties_l183_183434


namespace store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l183_183084

-- Definitions and conditions
def cost_per_soccer : ℕ := 200
def cost_per_basketball : ℕ := 80
def discount_A_soccer (n : ℕ) : ℕ := n * cost_per_soccer
def discount_A_basketball (n : ℕ) : ℕ := if n > 100 then (n - 100) * cost_per_basketball else 0
def discount_B_soccer (n : ℕ) : ℕ := n * cost_per_soccer * 8 / 10
def discount_B_basketball (n : ℕ) : ℕ := n * cost_per_basketball * 8 / 10

-- For x = 100
def total_cost_A_100 : ℕ := discount_A_soccer 100 + discount_A_basketball 100
def total_cost_B_100 : ℕ := discount_B_soccer 100 + discount_B_basketball 100

-- Prove that for x = 100, Store A is more cost-effective
theorem store_A_more_cost_effective_100 : total_cost_A_100 < total_cost_B_100 :=
by sorry

-- For x > 100, express costs in terms of x
def total_cost_A (x : ℕ) : ℕ := 80 * x + 12000
def total_cost_B (x : ℕ) : ℕ := 64 * x + 16000

-- Prove the expressions for costs
theorem cost_expressions_for_x (x : ℕ) (h : x > 100) : 
  total_cost_A x = 80 * x + 12000 ∧ total_cost_B x = 64 * x + 16000 :=
by sorry

-- For x = 300, most cost-effective plan
def combined_A_100_B_200 : ℕ := (discount_A_soccer 100 + cost_per_soccer * 100) + (200 * cost_per_basketball * 8 / 10)
def only_A_300 : ℕ := discount_A_soccer 100 + (300 - 100) * cost_per_basketball
def only_B_300 : ℕ := discount_B_soccer 100 + 300 * cost_per_basketball * 8 / 10

-- Prove the most cost-effective plan for x = 300
theorem most_cost_effective_plan : combined_A_100_B_200 < only_B_300 ∧ combined_A_100_B_200 < only_A_300 :=
by sorry

end store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l183_183084


namespace sqrt_product_l183_183774

open Real

theorem sqrt_product :
  sqrt 54 * sqrt 48 * sqrt 6 = 72 * sqrt 3 := by
  sorry

end sqrt_product_l183_183774


namespace polygon_area_leq_17_point_5_l183_183581

theorem polygon_area_leq_17_point_5 (proj_OX proj_bisector_13 proj_OY proj_bisector_24 : ℝ)
  (h1: proj_OX = 4)
  (h2: proj_bisector_13 = 3 * Real.sqrt 2)
  (h3: proj_OY = 5)
  (h4: proj_bisector_24 = 4 * Real.sqrt 2)
  (S : ℝ) :
  S ≤ 17.5 := sorry

end polygon_area_leq_17_point_5_l183_183581


namespace find_t_l183_183864

-- Definitions of the vectors involved
def vector_AB : ℝ × ℝ := (2, 3)
def vector_AC (t : ℝ) : ℝ × ℝ := (3, t)
def vector_BC (t : ℝ) : ℝ × ℝ := ((vector_AC t).1 - (vector_AB).1, (vector_AC t).2 - (vector_AB).2)

-- Condition for orthogonality
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- Main statement to be proved
theorem find_t : ∃ t : ℝ, is_perpendicular vector_AB (vector_BC t) ∧ t = 7 / 3 :=
by
  sorry

end find_t_l183_183864


namespace hexagon_perimeter_l183_183314

-- Define the side length 's' based on the given area condition
def side_length (s : ℝ) : Prop :=
  (3 * Real.sqrt 2 + Real.sqrt 3) / 4 * s^2 = 12

-- The theorem to prove
theorem hexagon_perimeter (s : ℝ) (h : side_length s) : 
  6 * s = 6 * Real.sqrt (48 / (3 * Real.sqrt 2 + Real.sqrt 3)) :=
by
  sorry

end hexagon_perimeter_l183_183314


namespace existence_of_solution_values_continuous_solution_value_l183_183185

noncomputable def functional_equation_has_solution (a : ℝ) (f : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ f 1 = 1 ∧ ∀ x y, (x ≤ y → f ((x + y) / 2) = (1 - a) * f x + a * f y)

theorem existence_of_solution_values :
  {a : ℝ | ∃ f : ℝ → ℝ, functional_equation_has_solution a f} = {0, 1/2, 1} :=
sorry

theorem continuous_solution_value :
  {a : ℝ | ∃ (f : ℝ → ℝ) (hf : Continuous f), functional_equation_has_solution a f} = {1/2} :=
sorry

end existence_of_solution_values_continuous_solution_value_l183_183185


namespace find_number_l183_183457

theorem find_number (x : ℝ) (h : 0.30 * x = 90 + 120) : x = 700 :=
by 
  sorry

end find_number_l183_183457


namespace number_of_students_at_end_of_year_l183_183591

def students_at_start_of_year : ℕ := 35
def students_left_during_year : ℕ := 10
def students_joined_during_year : ℕ := 10

theorem number_of_students_at_end_of_year : students_at_start_of_year - students_left_during_year + students_joined_during_year = 35 :=
by
  sorry -- Proof goes here

end number_of_students_at_end_of_year_l183_183591


namespace no_solution_natural_p_q_r_l183_183687

theorem no_solution_natural_p_q_r :
  ¬ ∃ (p q r : ℕ), 2^p + 5^q = 19^r := sorry

end no_solution_natural_p_q_r_l183_183687


namespace simplify_expression_l183_183724

open Real

theorem simplify_expression (x : ℝ) (hx : 0 < x) : Real.sqrt (Real.sqrt (x^3 * sqrt (x^5))) = x^(11/8) :=
sorry

end simplify_expression_l183_183724


namespace solve_equation_l183_183788

theorem solve_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) : (2 / x = 1 / (x + 1)) ↔ (x = -2) :=
by {
  sorry
}

end solve_equation_l183_183788


namespace find_y_l183_183854

theorem find_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hrem : x % y = 5) (hdiv : (x : ℝ) / y = 96.2) : y = 25 := by
  sorry

end find_y_l183_183854


namespace vector_parallel_sum_l183_183574

theorem vector_parallel_sum (m n : ℝ) (a b : ℝ × ℝ × ℝ)
  (h_a : a = (2, -1, 3))
  (h_b : b = (4, m, n))
  (h_parallel : ∃ k : ℝ, a = k • b) :
  m + n = 4 :=
sorry

end vector_parallel_sum_l183_183574


namespace binomial_coeff_sum_l183_183814

-- Define the problem: compute the numerical sum of the binomial coefficients
theorem binomial_coeff_sum (a b : ℕ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  (a + b) ^ 8 = 256 :=
by
  -- Therefore, the sum must be 256
  sorry

end binomial_coeff_sum_l183_183814


namespace expected_value_of_flipped_coins_l183_183318

theorem expected_value_of_flipped_coins :
  let p := 1
  let n := 5
  let d := 10
  let q := 25
  let f := 50
  let prob := (1:ℝ) / 2
  let V := prob * p + prob * n + prob * d + prob * q + prob * f
  V = 45.5 :=
by
  sorry

end expected_value_of_flipped_coins_l183_183318


namespace parabola_intersection_at_1_2003_l183_183399

theorem parabola_intersection_at_1_2003 (p q : ℝ) (h : p + q = 2002) :
  (1, (1 : ℝ)^2 + p * 1 + q) = (1, 2003) :=
by
  sorry

end parabola_intersection_at_1_2003_l183_183399


namespace johnnys_age_l183_183681

theorem johnnys_age (x : ℤ) (h : x + 2 = 2 * (x - 3)) : x = 8 := sorry

end johnnys_age_l183_183681


namespace plane_speed_with_tailwind_l183_183007

theorem plane_speed_with_tailwind (V : ℝ) (tailwind_speed : ℝ) (ground_speed_against_tailwind : ℝ) 
  (H1 : tailwind_speed = 75) (H2 : ground_speed_against_tailwind = 310) (H3 : V - tailwind_speed = ground_speed_against_tailwind) :
  V + tailwind_speed = 460 :=
by
  sorry

end plane_speed_with_tailwind_l183_183007


namespace capacity_of_new_bucket_l183_183771

def number_of_old_buckets : ℕ := 26
def capacity_of_old_bucket : ℝ := 13.5
def total_volume : ℝ := number_of_old_buckets * capacity_of_old_bucket
def number_of_new_buckets : ℕ := 39

theorem capacity_of_new_bucket :
  total_volume / number_of_new_buckets = 9 :=
sorry

end capacity_of_new_bucket_l183_183771


namespace problem_sequence_sum_l183_183354

theorem problem_sequence_sum (a : ℤ) (h : 14 * a^2 + 7 * a = 135) : 7 * a + (a - 1) = 23 :=
by {
  sorry
}

end problem_sequence_sum_l183_183354


namespace length_of_qr_l183_183405

theorem length_of_qr (Q : ℝ) (PQ QR : ℝ) 
  (h1 : Real.sin Q = 0.6)
  (h2 : PQ = 15) :
  QR = 18.75 :=
by
  sorry

end length_of_qr_l183_183405


namespace necessary_but_not_sufficient_l183_183104

noncomputable def represents_ellipse (m : ℝ) : Prop :=
  2 < m ∧ m < 6 ∧ m ≠ 4

theorem necessary_but_not_sufficient (m : ℝ) :
  represents_ellipse (m) ↔ (2 < m ∧ m < 6) :=
by
  sorry

end necessary_but_not_sufficient_l183_183104


namespace limit_at_2_l183_183696

noncomputable def delta (ε : ℝ) : ℝ := ε / 3

theorem limit_at_2 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x : ℝ, (0 < |x - 2| ∧ |x - 2| < δ) → |(3 * x^2 - 5 * x - 2) / (x - 2) - 7| < ε :=
by
  let δ := delta ε
  have hδ : δ > 0 := by
    sorry
  use δ, hδ
  intros x hx
  sorry

end limit_at_2_l183_183696


namespace units_digit_a2019_l183_183969

theorem units_digit_a2019 (a : ℕ → ℝ) (h₁ : ∀ n, a n > 0)
  (h₂ : a 2 ^ 2 + a 4 ^ 2 = 900 - 2 * a 1 * a 5)
  (h₃ : a 5 = 9 * a 3) : (3^(2018) % 10) = 9 := by
  sorry

end units_digit_a2019_l183_183969


namespace sum_a1_to_a14_equals_zero_l183_183806

theorem sum_a1_to_a14_equals_zero 
  (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 : ℝ) 
  (h1 : (1 + x - x^2)^3 * (1 - 2 * x^2)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 + a13 * x^13 + a14 * x^14) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 1) 
  (h3 : a = 1) : 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 0 := by
  sorry

end sum_a1_to_a14_equals_zero_l183_183806


namespace arrangement_count_l183_183643

-- Define the sets of books
def italian_books : Finset String := { "I1", "I2", "I3" }
def german_books : Finset String := { "G1", "G2", "G3" }
def french_books : Finset String := { "F1", "F2", "F3", "F4", "F5" }

-- Define the arrangement count as a noncomputable definition, because we are going to use factorial which involves an infinite structure
noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Prove the required arrangement
theorem arrangement_count : 
  (factorial 3) * ((factorial 3) * (factorial 3) * (factorial 5)) = 25920 := 
by
  -- Provide the solution steps here (omitted for now)
  sorry

end arrangement_count_l183_183643


namespace rectangle_longer_side_length_l183_183576

theorem rectangle_longer_side_length (r : ℝ) (h1 : r = 4) 
  (h2 : ∃ w l, w * l = 2 * (π * r^2) ∧ w = 2 * r) : 
  ∃ l, l = 4 * π :=
by 
  obtain ⟨w, l, h_area, h_shorter_side⟩ := h2
  sorry

end rectangle_longer_side_length_l183_183576


namespace trigonometric_identity_l183_183124

theorem trigonometric_identity (x : ℝ) (h : Real.tan (3 * π - x) = 2) :
    (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end trigonometric_identity_l183_183124


namespace sum_one_to_twenty_nine_l183_183956

theorem sum_one_to_twenty_nine : (29 / 2) * (1 + 29) = 435 := by
  -- proof
  sorry

end sum_one_to_twenty_nine_l183_183956


namespace parabola_equation_l183_183166

theorem parabola_equation (a b c : ℝ)
  (h_p : (a + b + c = 1))
  (h_q : (4 * a + 2 * b + c = -1))
  (h_tangent : (4 * a + b = 1)) :
  y = 3 * x^2 - 11 * x + 9 :=
by {
  sorry
}

end parabola_equation_l183_183166


namespace number_decomposition_l183_183365

theorem number_decomposition (n : ℕ) : n = 6058 → (n / 1000 = 6) ∧ ((n % 100) / 10 = 5) ∧ (n % 10 = 8) :=
by
  -- Actual proof will go here
  sorry

end number_decomposition_l183_183365


namespace max_min_values_l183_183961

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values :
  (∀ x ∈ (Set.Icc 0 2), f x ≤ 5) ∧ (∃ x ∈ (Set.Icc 0 2), f x = 5) ∧
  (∀ x ∈ (Set.Icc 0 2), f x ≥ -15) ∧ (∃ x ∈ (Set.Icc 0 2), f x = -15) :=
by
  sorry

end max_min_values_l183_183961


namespace molecular_weight_of_compound_l183_183220

def num_atoms_C : ℕ := 6
def num_atoms_H : ℕ := 8
def num_atoms_O : ℕ := 7

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_weight (nC nH nO : ℕ) (wC wH wO : ℝ) : ℝ :=
  nC * wC + nH * wH + nO * wO

theorem molecular_weight_of_compound :
  molecular_weight num_atoms_C num_atoms_H num_atoms_O atomic_weight_C atomic_weight_H atomic_weight_O = 192.124 :=
by
  sorry

end molecular_weight_of_compound_l183_183220


namespace movie_ticket_ratio_l183_183172

-- Definitions based on the conditions
def monday_cost : ℕ := 5
def wednesday_cost : ℕ := 2 * monday_cost

theorem movie_ticket_ratio (S : ℕ) (h1 : wednesday_cost + S = 35) :
  S / monday_cost = 5 :=
by
  -- Placeholder for proof
  sorry

end movie_ticket_ratio_l183_183172


namespace sum_of_digits_of_power_eight_2010_l183_183980

theorem sum_of_digits_of_power_eight_2010 :
  let n := 2010
  let a := 8
  let tens_digit := (a ^ n / 10) % 10
  let units_digit := a ^ n % 10
  tens_digit + units_digit = 1 :=
by
  sorry

end sum_of_digits_of_power_eight_2010_l183_183980


namespace isabella_paint_area_l183_183061

-- Lean 4 statement for the proof problem based on given conditions and question:
theorem isabella_paint_area :
  let length := 15
  let width := 12
  let height := 9
  let door_and_window_area := 80
  let number_of_bedrooms := 4
  (2 * (length * height) + 2 * (width * height) - door_and_window_area) * number_of_bedrooms = 1624 :=
by
  sorry

end isabella_paint_area_l183_183061


namespace range_of_m_l183_183747

noncomputable def quadratic_expr_never_equal (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2 * x^2 + 4 * x + m ≠ 3 * x^2 - 2 * x + 6

theorem range_of_m (m : ℝ) : quadratic_expr_never_equal m ↔ m < -3 := 
by
  sorry

end range_of_m_l183_183747


namespace power_sums_equal_l183_183385

theorem power_sums_equal (x y a b : ℝ)
  (h1 : x + y = a + b)
  (h2 : x^2 + y^2 = a^2 + b^2) :
  ∀ n : ℕ, x^n + y^n = a^n + b^n :=
by
  sorry

end power_sums_equal_l183_183385


namespace not_snowing_next_five_days_l183_183968

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l183_183968


namespace final_cards_l183_183161

def initial_cards : ℝ := 47.0
def lost_cards : ℝ := 7.0

theorem final_cards : (initial_cards - lost_cards) = 40.0 :=
by
  sorry

end final_cards_l183_183161


namespace plane_eq_passing_A_perpendicular_BC_l183_183610

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def subtract_points (P Q : Point3D) : Point3D :=
  { x := P.x - Q.x, y := P.y - Q.y, z := P.z - Q.z }

-- Points A, B, and C given in the conditions
def A : Point3D := { x := 1, y := -5, z := -2 }
def B : Point3D := { x := 6, y := -2, z := 1 }
def C : Point3D := { x := 2, y := -2, z := -2 }

-- Vector BC
def BC : Point3D := subtract_points C B

theorem plane_eq_passing_A_perpendicular_BC :
  (-4 : ℝ) * (A.x - 1) + (0 : ℝ) * (A.y + 5) + (-3 : ℝ) * (A.z + 2) = 0 :=
  sorry

end plane_eq_passing_A_perpendicular_BC_l183_183610


namespace evaluate_expression_l183_183383

theorem evaluate_expression : (16 ^ 24) / (64 ^ 8) = 16 ^ 12 :=
by sorry

end evaluate_expression_l183_183383


namespace boat_breadth_is_two_l183_183000

noncomputable def breadth_of_boat (L h m g ρ : ℝ) : ℝ :=
  let W := m * g
  let V := W / (ρ * g)
  V / (L * h)

theorem boat_breadth_is_two :
  breadth_of_boat 7 0.01 140 9.81 1000 = 2 := 
by
  unfold breadth_of_boat
  simp
  sorry

end boat_breadth_is_two_l183_183000


namespace extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l183_183514

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)

-- Problem 1: Prove that the extreme value of f(x) when a = 1 is \frac{3}{4} + \ln 2
theorem extreme_value_f_at_a_eq_1 : 
  f (3/2) 1 = 3/4 + Real.log 2 :=
sorry

-- Problem 2: Prove the monotonic intervals of f(x) based on the value of a
theorem monotonic_intervals_f :
  ∀ a : ℝ, 
    (if a ≤ 0 then 
      ∀ x, 1 < x → f x' a > 0
     else
      ∀ x, 1 < x ∧ x ≤ (a + 2) / 2 → f x a ≤ 0 ∧ ∀ x, x ≥ (a + 2) / 2 → f x a > 0) :=
sorry

-- Problem 3: Prove that for a ≥ 1, there exists an a such that f(x) has no common points with y = \frac{5}{8} + \ln 2
theorem exists_no_common_points (h : 1 ≤ a) :
  ∃ x : ℝ, f x a ≠ 5/8 + Real.log 2 :=
sorry

end extreme_value_f_at_a_eq_1_monotonic_intervals_f_exists_no_common_points_l183_183514


namespace correct_equation_l183_183062

theorem correct_equation (a b : ℝ) : (a - b) ^ 3 * (b - a) ^ 4 = (a - b) ^ 7 :=
sorry

end correct_equation_l183_183062


namespace equation_of_curve_t_circle_through_fixed_point_l183_183523

noncomputable def problem (x y : ℝ) : Prop :=
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (0, -1)
  let O : ℝ × ℝ := (0, 0)
  let M : ℝ × ℝ := (x, y)
  let N : ℝ × ℝ := (0, y)
  (x + 1) * (x - 1) + y * y = y * (y + 1)

noncomputable def curve_t_equation (x : ℝ) : ℝ :=
  x^2 - 1

theorem equation_of_curve_t (x y : ℝ) 
  (h : problem x y) :
  y = curve_t_equation x := 
sorry

noncomputable def passing_through_fixed_point (x y : ℝ) : Prop :=
  let y := x^2 - 1
  let y' := 2 * x
  let P : ℝ × ℝ := (x, y)
  let Q_x := (4 * x^2 - 1) / (8 * x)
  let Q : ℝ × ℝ := (Q_x, -5 / 4)
  let H : ℝ × ℝ := (0, -3 / 4)
  (x * Q_x + (-3 / 4 - y) * ( -3 / 4 + 5 / 4)) = 0

theorem circle_through_fixed_point (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y = curve_t_equation x)
  (h : passing_through_fixed_point x y) :
  ∃ t : ℝ, passing_through_fixed_point x t ∧ t = -3 / 4 :=
sorry

end equation_of_curve_t_circle_through_fixed_point_l183_183523


namespace nesbitts_inequality_l183_183229

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

end nesbitts_inequality_l183_183229


namespace combined_area_of_triangles_l183_183022

noncomputable def area_of_rectangle (length width : ℝ) : ℝ :=
  length * width

noncomputable def first_triangle_area (x : ℝ) : ℝ :=
  5 * x

noncomputable def second_triangle_area (base height : ℝ) : ℝ :=
  (base * height) / 2

theorem combined_area_of_triangles (length width x base height : ℝ)
  (h1 : area_of_rectangle length width / first_triangle_area x = 2 / 5)
  (h2 : base + height = 20)
  (h3 : second_triangle_area base height / first_triangle_area x = 3 / 5)
  (length_value : length = 6)
  (width_value : width = 4)
  (base_value : base = 8) :
  first_triangle_area x + second_triangle_area base height = 108 := 
by
  sorry

end combined_area_of_triangles_l183_183022


namespace quadratic_general_form_l183_183827

theorem quadratic_general_form (x : ℝ) :
    (x + 3)^2 = x * (3 * x - 1) →
    2 * x^2 - 7 * x - 9 = 0 :=
by
  intros h
  sorry

end quadratic_general_form_l183_183827


namespace find_x_l183_183339

noncomputable def x : ℝ :=
  sorry

theorem find_x (h : ∃ x : ℝ, x > 0 ∧ ⌊x⌋ * x = 48) : x = 8 :=
  sorry

end find_x_l183_183339


namespace find_missing_number_l183_183559

theorem find_missing_number
  (x : ℝ)
  (h1 : (12 + x + y + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) : 
  y = 42 :=
  sorry

end find_missing_number_l183_183559


namespace smallest_x_exists_l183_183918

theorem smallest_x_exists {M : ℤ} (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ x : ℕ, 2520 * x = M^3 ∧ x = 3675 := 
by {
  sorry
}

end smallest_x_exists_l183_183918


namespace Christine_savings_l183_183850

theorem Christine_savings 
  (commission_rate: ℝ) 
  (total_sales: ℝ) 
  (personal_needs_percentage: ℝ) 
  (savings: ℝ) 
  (h1: commission_rate = 0.12) 
  (h2: total_sales = 24000) 
  (h3: personal_needs_percentage = 0.60) 
  (h4: savings = total_sales * commission_rate * (1 - personal_needs_percentage)) : 
  savings = 1152 := by 
  sorry

end Christine_savings_l183_183850


namespace expression_evaluation_l183_183982

variable (a b : ℝ)

theorem expression_evaluation (h : a + b = 1) :
  a^3 + b^3 + 3 * (a^3 * b + a * b^3) + 6 * (a^3 * b^2 + a^2 * b^3) = 1 :=
by
  sorry

end expression_evaluation_l183_183982


namespace sin_theta_correct_l183_183524

noncomputable def sin_theta : ℝ :=
  let d := (4, 5, 7)
  let n := (3, -4, 5)
  let d_dot_n := 4 * 3 + 5 * (-4) + 7 * 5
  let norm_d := Real.sqrt (4^2 + 5^2 + 7^2)
  let norm_n := Real.sqrt (3^2 + (-4)^2 + 5^2)
  let cos_theta := d_dot_n / (norm_d * norm_n)
  cos_theta

theorem sin_theta_correct :
  sin_theta = 27 / Real.sqrt 4500 :=
by
  sorry

end sin_theta_correct_l183_183524


namespace total_spent_on_burgers_l183_183998

def days_in_june := 30
def burgers_per_day := 4
def cost_per_burger := 13

theorem total_spent_on_burgers (total_spent : Nat) :
  total_spent = days_in_june * burgers_per_day * cost_per_burger :=
sorry

end total_spent_on_burgers_l183_183998


namespace trapezium_area_l183_183618

variables {A B C D O : Type}
variables (P Q : ℕ)

-- Conditions
def trapezium (ABCD : Type) : Prop := true
def parallel_lines (AB DC : Type) : Prop := true
def intersection (AC BD O : Type) : Prop := true
def area_AOB (P : ℕ) : Prop := P = 16
def area_COD : ℕ := 25

theorem trapezium_area (ABCD AC BD AB DC O : Type) (P Q : ℕ)
  (h1 : trapezium ABCD)
  (h2 : parallel_lines AB DC)
  (h3 : intersection AC BD O)
  (h4 : area_AOB P) 
  (h5 : area_COD = 25) :
  Q = 81 :=
sorry

end trapezium_area_l183_183618


namespace sin_theta_value_l183_183430

theorem sin_theta_value (θ : ℝ) (h₁ : θ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) 
  (h₂ : Real.sin (2 * θ) = (3 * Real.sqrt 7) / 8) : Real.sin θ = 3 / 4 :=
sorry

end sin_theta_value_l183_183430


namespace graphs_symmetric_respect_to_x_equals_1_l183_183672

-- Define the function f
variable (f : ℝ → ℝ)

-- Define g(x) = f(x-1)
def g (x : ℝ) : ℝ := f (x - 1)

-- Define h(x) = f(1 - x)
def h (x : ℝ) : ℝ := f (1 - x)

-- The theorem that their graphs are symmetric with respect to the line x = 1
theorem graphs_symmetric_respect_to_x_equals_1 :
  ∀ x : ℝ, g f x = h f x ↔ f x = f (2 - x) :=
sorry

end graphs_symmetric_respect_to_x_equals_1_l183_183672


namespace jonessa_total_pay_l183_183076

theorem jonessa_total_pay (total_pay : ℝ) (take_home_pay : ℝ) (h1 : take_home_pay = 450) (h2 : 0.90 * total_pay = take_home_pay) : total_pay = 500 :=
by
  sorry

end jonessa_total_pay_l183_183076


namespace vector_dot_product_l183_183870

-- Definitions
def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, -1)

-- Theorem to prove
theorem vector_dot_product : 
  ((vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) : ℝ × ℝ) • (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2) = 10 :=
by
  sorry

end vector_dot_product_l183_183870


namespace hungarian_math_olympiad_1927_l183_183176

-- Definitions
def is_coprime (a b : ℤ) : Prop :=
  Int.gcd a b = 1

-- The main statement
theorem hungarian_math_olympiad_1927
  (a b c d x y k m : ℤ) 
  (h_coprime : is_coprime a b)
  (h_m : m = a * d - b * c)
  (h_divides : m ∣ (a * x + b * y)) :
  m ∣ (c * x + d * y) :=
sorry

end hungarian_math_olympiad_1927_l183_183176


namespace find_number_l183_183763

-- Define the hypothesis/condition
def condition (x : ℤ) : Prop := 2 * x + 20 = 8 * x - 4

-- Define the statement to prove
theorem find_number (x : ℤ) (h : condition x) : x = 4 := 
by
  sorry

end find_number_l183_183763


namespace cake_has_more_calories_l183_183537

-- Define the conditions
def cake_slices : Nat := 8
def cake_calories_per_slice : Nat := 347
def brownie_count : Nat := 6
def brownie_calories_per_brownie : Nat := 375

-- Define the total calories for the cake and the brownies
def total_cake_calories : Nat := cake_slices * cake_calories_per_slice
def total_brownie_calories : Nat := brownie_count * brownie_calories_per_brownie

-- Prove the difference in calories
theorem cake_has_more_calories : 
  total_cake_calories - total_brownie_calories = 526 :=
by
  sorry

end cake_has_more_calories_l183_183537


namespace find_a1_in_geometric_sequence_l183_183851

noncomputable def geometric_sequence_first_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n : ℕ, a (n + 1) = a n * r) : ℝ :=
  a 0

theorem find_a1_in_geometric_sequence (a : ℕ → ℝ) (h_geo : ∀ n : ℕ, a (n + 1) = a n * (1 / 2)) :
  a 2 = 16 → a 3 = 8 → geometric_sequence_first_term a (1 / 2) h_geo = 64 :=
by
  intros h2 h3
  -- Proof would go here
  sorry

end find_a1_in_geometric_sequence_l183_183851


namespace factorize_expr_l183_183493

theorem factorize_expr (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) :=
  sorry

end factorize_expr_l183_183493


namespace solve_equation_l183_183322

noncomputable def f (x : ℝ) : ℝ :=
  2 * x + 1 + Real.arctan x * Real.sqrt (x^2 + 1)

theorem solve_equation : ∃ x : ℝ, f x + f (x + 1) = 0 ∧ x = -1/2 :=
  by
    use -1/2
    simp [f]
    sorry

end solve_equation_l183_183322


namespace solve_real_number_pairs_l183_183970

theorem solve_real_number_pairs (x y : ℝ) :
  (x^2 + y^2 - 48 * x - 29 * y + 714 = 0 ∧ 2 * x * y - 29 * x - 48 * y + 756 = 0) ↔
  (x = 31.5 ∧ y = 10.5) ∨ (x = 20 ∧ y = 22) ∨ (x = 28 ∧ y = 7) ∨ (x = 16.5 ∧ y = 18.5) :=
by
  sorry

end solve_real_number_pairs_l183_183970


namespace utilities_cost_l183_183398

theorem utilities_cost
    (rent1 : ℝ) (utility1 : ℝ) (rent2 : ℝ) (utility2 : ℝ)
    (distance1 : ℝ) (distance2 : ℝ) 
    (cost_per_mile : ℝ) 
    (drive_days : ℝ) (cost_difference : ℝ)
    (h1 : rent1 = 800)
    (h2 : rent2 = 900)
    (h3 : utility2 = 200)
    (h4 : distance1 = 31)
    (h5 : distance2 = 21)
    (h6 : cost_per_mile = 0.58)
    (h7 : drive_days = 20)
    (h8 : cost_difference = 76)
    : utility1 = 259.60 := 
by
  sorry

end utilities_cost_l183_183398


namespace problem1_problem2_l183_183141

-- Define the universe U
def U : Set ℝ := Set.univ

-- Define the sets A and B
def A : Set ℝ := { x | -4 < x ∧ x < 4 }
def B : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }

-- Statement of the first proof problem: Prove A ∩ B is equal to the given set
theorem problem1 : A ∩ B = { x | -4 < x ∧ x ≤ 1 ∨ 4 > x ∧ x ≥ 3 } :=
by
  sorry

-- Statement of the second proof problem: Prove the complement of (A ∪ B) in the universe U is ∅
theorem problem2 : Set.compl (A ∪ B) = ∅ :=
by
  sorry

end problem1_problem2_l183_183141


namespace fraction_left_handed_non_throwers_is_one_third_l183_183495

theorem fraction_left_handed_non_throwers_is_one_third :
  let total_players := 70
  let throwers := 31
  let right_handed := 57
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed_non_throwers := non_throwers - right_handed_non_throwers
  (left_handed_non_throwers : ℝ) / non_throwers = 1 / 3 := by
  sorry

end fraction_left_handed_non_throwers_is_one_third_l183_183495


namespace tens_digit_of_large_power_l183_183190

theorem tens_digit_of_large_power : ∃ a : ℕ, a = 2 ∧ ∀ n ≥ 2, (5 ^ n) % 100 = 25 :=
by
  sorry

end tens_digit_of_large_power_l183_183190


namespace prob_even_sum_is_one_third_l183_183245

def is_even_sum_first_last (d1 d2 d3 d4 : Nat) : Prop :=
  (d1 + d4) % 2 = 0

def num_unique_arrangements : Nat := 12

def num_favorable_arrangements : Nat := 4

def prob_even_sum_first_last : Rat :=
  num_favorable_arrangements / num_unique_arrangements

theorem prob_even_sum_is_one_third :
  prob_even_sum_first_last = 1 / 3 := 
  sorry

end prob_even_sum_is_one_third_l183_183245


namespace completion_time_l183_183913

theorem completion_time (total_work : ℕ) (initial_num_men : ℕ) (initial_efficiency : ℝ)
  (new_num_men : ℕ) (new_efficiency : ℝ) :
  total_work = 12 ∧ initial_num_men = 4 ∧ initial_efficiency = 1.5 ∧
  new_num_men = 6 ∧ new_efficiency = 2.0 →
  total_work / (new_num_men * new_efficiency) = 1 :=
by
  sorry

end completion_time_l183_183913


namespace billy_distance_l183_183893

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem billy_distance :
  distance 0 0 (7 + 4 * Real.sqrt 2) (4 * (Real.sqrt 2 + 1)) = Real.sqrt (129 + 88 * Real.sqrt 2) :=
by
  -- proof goes here
  sorry

end billy_distance_l183_183893


namespace sum_of_first_10_terms_l183_183882

noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def S (n : ℕ) : ℕ := sorry

variable {n : ℕ}

-- Conditions
axiom h1 : ∀ n, S (n + 1) = S n + a n + 3
axiom h2 : a 5 + a 6 = 29

-- Statement to prove
theorem sum_of_first_10_terms : S 10 = 145 := 
sorry

end sum_of_first_10_terms_l183_183882


namespace melissa_games_played_l183_183349

-- Define the conditions mentioned:
def points_per_game := 12
def total_points := 36

-- State the proof problem:
theorem melissa_games_played : total_points / points_per_game = 3 :=
by sorry

end melissa_games_played_l183_183349


namespace ratio_change_factor_is_5_l183_183125

-- Definitions based on problem conditions
def original_bleach : ℕ := 4
def original_detergent : ℕ := 40
def original_water : ℕ := 100

-- Simplified original ratio
def original_bleach_ratio : ℕ := original_bleach / 4
def original_detergent_ratio : ℕ := original_detergent / 4
def original_water_ratio : ℕ := original_water / 4

-- Altered conditions
def altered_detergent : ℕ := 60
def altered_water : ℕ := 300

-- Simplified altered ratio of detergent to water
def altered_detergent_ratio : ℕ := altered_detergent / 60
def altered_water_ratio : ℕ := altered_water / 60

-- Proof that the ratio change factor is 5
theorem ratio_change_factor_is_5 : 
  (original_water_ratio / altered_water_ratio) = 5
  := by
    have original_detergent_ratio : ℕ := 10
    have original_water_ratio : ℕ := 25
    have altered_detergent_ratio : ℕ := 1
    have altered_water_ratio : ℕ := 5
    sorry

end ratio_change_factor_is_5_l183_183125


namespace maximum_diagonal_intersections_l183_183284

theorem maximum_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
  ∃ k, k = (n * (n - 1) * (n - 2) * (n - 3)) / 24 :=
by sorry

end maximum_diagonal_intersections_l183_183284


namespace find_x_l183_183175

variable (P T S : Point)
variable (angle_PTS angle_TSR x : ℝ)
variable (reflector : Point)

-- Given conditions
axiom angle_PTS_is_90 : angle_PTS = 90
axiom angle_TSR_is_26 : angle_TSR = 26

-- Proof problem
theorem find_x : x = 32 := by
  sorry

end find_x_l183_183175


namespace function_decomposition_l183_183912

open Real

noncomputable def f (x : ℝ) : ℝ := log (10^x + 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
noncomputable def h (x : ℝ) : ℝ := log (10^x + 1) - x / 2

theorem function_decomposition :
  ∀ x : ℝ, f x = g x + h x ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, h (-x) = h x) :=
by
  intro x
  sorry

end function_decomposition_l183_183912


namespace tan_of_cos_l183_183799

theorem tan_of_cos (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_alpha : 0 < α ∧ α < Real.pi) : 
  Real.tan α = -3 / 4 :=
sorry

end tan_of_cos_l183_183799


namespace value_S3_S2_S5_S3_l183_183406

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
variable {d : ℝ}
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (d_ne_zero : d ≠ 0)
variable (h_geom_seq : (a 1 + 2 * d) ^ 2 = (a 1) * (a 1 + 3 * d))
variable (S_def : ∀ n, S n = n * a 1 + d * (n * (n - 1)) / 2)

theorem value_S3_S2_S5_S3 : (S 3 - S 2) / (S 5 - S 3) = 2 := by
  sorry

end value_S3_S2_S5_S3_l183_183406


namespace faye_age_l183_183966

variable (C D E F : ℕ)

-- Conditions
axiom h1 : D = 16
axiom h2 : D = E - 4
axiom h3 : E = C + 5
axiom h4 : F = C + 2

-- Goal: Prove that F = 17
theorem faye_age : F = 17 :=
by
  sorry

end faye_age_l183_183966


namespace radius_of_larger_circle_is_25_over_3_l183_183346

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := (5 / 2) * r 

theorem radius_of_larger_circle_is_25_over_3
  (rAB rBD : ℝ)
  (h_ratio : 2 * rBD = 5 * rBD / 2)
  (h_ab : rAB = 8)
  (h_tangent : ∀ rBD, (5 * rBD / 2 - 8) ^ 2 = 64 + rBD ^ 2) :
  radius_of_larger_circle (10 / 3) = 25 / 3 :=
  by
  sorry

end radius_of_larger_circle_is_25_over_3_l183_183346


namespace water_fall_amount_l183_183555

theorem water_fall_amount (M_before J_before M_after J_after n : ℕ) 
  (h1 : M_before = 48) 
  (h2 : M_before = J_before + 32)
  (h3 : M_after = M_before + n) 
  (h4 : J_after = J_before + n)
  (h5 : M_after = 2 * J_after) : 
  n = 16 :=
by 
  -- proof omitted
  sorry

end water_fall_amount_l183_183555


namespace derek_initial_lunch_cost_l183_183283

-- Definitions based on conditions
def derek_initial_money : ℕ := 40
def derek_dad_lunch_cost : ℕ := 11
def derek_more_lunch_cost : ℕ := 5
def dave_initial_money : ℕ := 50
def dave_mom_lunch_cost : ℕ := 7
def dave_difference : ℕ := 33

-- Variable X to represent Derek's initial lunch cost
variable (X : ℕ)

-- Definitions based on conditions
def derek_total_spending (X : ℕ) := X + derek_dad_lunch_cost + derek_more_lunch_cost
def derek_remaining_money (X : ℕ) := derek_initial_money - derek_total_spending X
def dave_remaining_money := dave_initial_money - dave_mom_lunch_cost

-- The main theorem to prove Derek spent $14 initially
theorem derek_initial_lunch_cost (h : dave_remaining_money = derek_remaining_money X + dave_difference) : X = 14 := by
  sorry

end derek_initial_lunch_cost_l183_183283


namespace potato_yield_l183_183531

/-- Mr. Green's gardening problem -/
theorem potato_yield
  (steps_length : ℝ)
  (steps_width : ℝ)
  (step_size : ℝ)
  (yield_rate : ℝ)
  (feet_length := steps_length * step_size)
  (feet_width := steps_width * step_size)
  (area := feet_length * feet_width)
  (yield := area * yield_rate) :
  steps_length = 18 →
  steps_width = 25 →
  step_size = 2.5 →
  yield_rate = 0.75 →
  yield = 2109.375 :=
by
  sorry

end potato_yield_l183_183531


namespace oldest_brother_age_ratio_l183_183319

-- Define the ages
def rick_age : ℕ := 15
def youngest_brother_age : ℕ := 3
def smallest_brother_age : ℕ := youngest_brother_age + 2
def middle_brother_age : ℕ := smallest_brother_age * 2
def oldest_brother_age : ℕ := middle_brother_age * 3

-- Define the ratio
def expected_ratio : ℕ := oldest_brother_age / rick_age

theorem oldest_brother_age_ratio : expected_ratio = 2 := by
  sorry 

end oldest_brother_age_ratio_l183_183319


namespace passengers_at_station_in_an_hour_l183_183859

-- Define the conditions
def train_interval_minutes := 5
def passengers_off_per_train := 200
def passengers_on_per_train := 320

-- Define the time period we're considering
def time_period_minutes := 60

-- Calculate the expected values based on conditions
def expected_trains_per_hour := time_period_minutes / train_interval_minutes
def expected_passengers_off_per_hour := passengers_off_per_train * expected_trains_per_hour
def expected_passengers_on_per_hour := passengers_on_per_train * expected_trains_per_hour
def expected_total_passengers := expected_passengers_off_per_hour + expected_passengers_on_per_hour

theorem passengers_at_station_in_an_hour :
  expected_total_passengers = 6240 :=
by
  -- Structure of the proof omitted. Just ensuring conditions and expected value defined.
  sorry

end passengers_at_station_in_an_hour_l183_183859


namespace triangle_cosine_rule_c_triangle_tangent_C_l183_183636

-- Define a proof statement for the cosine rule-based proof of c = 4.
theorem triangle_cosine_rule_c (a b : ℝ) (angleB : ℝ) (ha : a = 2)
                              (hb : b = 2 * Real.sqrt 3) (hB : angleB = π / 3) :
  ∃ (c : ℝ), c = 4 := by
  sorry

-- Define a proof statement for the tangent identity-based proof of tan C = 3 * sqrt 3 / 5.
theorem triangle_tangent_C (tanA : ℝ) (tanB : ℝ) (htA : tanA = 2 * Real.sqrt 3)
                           (htB : tanB = Real.sqrt 3) :
  ∃ (tanC : ℝ), tanC = 3 * Real.sqrt 3 / 5 := by
  sorry

end triangle_cosine_rule_c_triangle_tangent_C_l183_183636


namespace smallest_tree_height_correct_l183_183962

-- Defining the conditions
def TallestTreeHeight : ℕ := 108
def MiddleTreeHeight (tallest : ℕ) : ℕ := (tallest / 2) - 6
def SmallestTreeHeight (middle : ℕ) : ℕ := middle / 4

-- Proof statement
theorem smallest_tree_height_correct :
  SmallestTreeHeight (MiddleTreeHeight TallestTreeHeight) = 12 :=
by
  -- Here we would put the proof, but we are skipping it with sorry.
  sorry

end smallest_tree_height_correct_l183_183962


namespace part1_l183_183225

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, x > 0 → f x < 0) :
  a > 1 :=
sorry

end part1_l183_183225


namespace shaded_area_correct_l183_183857

noncomputable def shaded_area (side_large side_small : ℝ) (pi_value : ℝ) : ℝ :=
  let area_large_square := side_large^2
  let area_large_circle := pi_value * (side_large / 2)^2
  let area_large_heart := area_large_square + area_large_circle
  let area_small_square := side_small^2
  let area_small_circle := pi_value * (side_small / 2)^2
  let area_small_heart := area_small_square + area_small_circle
  area_large_heart - area_small_heart

theorem shaded_area_correct : shaded_area 40 20 3.14 = 2142 :=
by
  -- Proof goes here
  sorry

end shaded_area_correct_l183_183857


namespace terrier_hush_interval_l183_183496

-- Definitions based on conditions
def poodle_barks_per_terrier_bark : ℕ := 2
def total_poodle_barks : ℕ := 24
def terrier_hushes : ℕ := 6

-- Derived values based on definitions
def total_terrier_barks := total_poodle_barks / poodle_barks_per_terrier_bark
def interval_hush := total_terrier_barks / terrier_hushes

-- The theorem stating the terrier's hush interval
theorem terrier_hush_interval : interval_hush = 2 := by
  have h1 : total_terrier_barks = 12 := by sorry
  have h2 : interval_hush = 2 := by sorry
  exact h2

end terrier_hush_interval_l183_183496


namespace decimal_equiv_of_one_fourth_cubed_l183_183316

theorem decimal_equiv_of_one_fourth_cubed : (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by sorry

end decimal_equiv_of_one_fourth_cubed_l183_183316


namespace solution_set_inequality_l183_183732

theorem solution_set_inequality (x : ℝ) : 
  (x + 5) * (3 - 2 * x) ≤ 6 ↔ (x ≤ -9/2 ∨ x ≥ 1) :=
by
  sorry  -- proof skipped as instructed

end solution_set_inequality_l183_183732


namespace right_triangle_side_length_l183_183701

theorem right_triangle_side_length (a c b : ℕ) (h₁ : a = 6) (h₂ : c = 10) (h₃ : c * c = a * a + b * b) : b = 8 :=
by {
  sorry
}

end right_triangle_side_length_l183_183701


namespace lcm_18_24_l183_183825

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_24 : 24 = 2^3 * 3 := by norm_num
  sorry

end lcm_18_24_l183_183825


namespace train_speed_is_144_l183_183367

-- Definitions for the conditions
def length_of_train_passing_pole (S : ℝ) := S * 8
def length_of_train_passing_stationary_train (S : ℝ) := S * 18 - 400

-- The main theorem to prove the speed of the train
theorem train_speed_is_144 (S : ℝ) :
  (length_of_train_passing_pole S = length_of_train_passing_stationary_train S) →
  (S * 3.6 = 144) :=
by
  sorry

end train_speed_is_144_l183_183367


namespace rectangle_side_multiple_of_6_l183_183652

theorem rectangle_side_multiple_of_6 (a b : ℕ) (h : ∃ n : ℕ, a * b = n * 6) : a % 6 = 0 ∨ b % 6 = 0 :=
sorry

end rectangle_side_multiple_of_6_l183_183652


namespace b_should_pay_360_l183_183114

theorem b_should_pay_360 :
  let total_cost : ℝ := 870
  let a_horses  : ℝ := 12
  let a_months  : ℝ := 8
  let b_horses  : ℝ := 16
  let b_months  : ℝ := 9
  let c_horses  : ℝ := 18
  let c_months  : ℝ := 6
  let a_horse_months := a_horses * a_months
  let b_horse_months := b_horses * b_months
  let c_horse_months := c_horses * c_months
  let total_horse_months := a_horse_months + b_horse_months + c_horse_months
  let cost_per_horse_month := total_cost / total_horse_months
  let b_cost := b_horse_months * cost_per_horse_month
  b_cost = 360 :=
by sorry

end b_should_pay_360_l183_183114


namespace additional_emails_per_day_l183_183094

theorem additional_emails_per_day
  (emails_per_day_before : ℕ)
  (half_days : ℕ)
  (total_days : ℕ)
  (total_emails : ℕ)
  (emails_received_first_half : ℕ := emails_per_day_before * half_days)
  (emails_received_second_half : ℕ := total_emails - emails_received_first_half)
  (emails_per_day_after : ℕ := emails_received_second_half / half_days) :
  emails_per_day_before = 20 → half_days = 15 → total_days = 30 → total_emails = 675 → (emails_per_day_after - emails_per_day_before = 5) :=
by
  intros
  sorry

end additional_emails_per_day_l183_183094


namespace henry_present_age_l183_183366

theorem henry_present_age (H J : ℕ) (h1 : H + J = 41) (h2 : H - 7 = 2 * (J - 7)) : H = 25 :=
sorry

end henry_present_age_l183_183366


namespace greatest_root_of_gx_l183_183328

theorem greatest_root_of_gx :
  ∃ x : ℝ, (10 * x^4 - 16 * x^2 + 3 = 0) ∧ (∀ y : ℝ, (10 * y^4 - 16 * y^2 + 3 = 0) → x ≥ y) ∧ x = Real.sqrt (3 / 5) := 
sorry

end greatest_root_of_gx_l183_183328


namespace problem_solution_l183_183803

noncomputable def verify_solution (x y z : ℝ) : Prop :=
  x = 12 ∧ y = 10 ∧ z = 8 →
  (x > 4) ∧ (y > 4) ∧ (z > 4) →
  ( ( (x + 3)^2 / (y + z - 3) ) + 
    ( (y + 5)^2 / (z + x - 5) ) + 
    ( (z + 7)^2 / (x + y - 7) ) = 45)

theorem problem_solution :
  verify_solution 12 10 8 := by
  sorry

end problem_solution_l183_183803


namespace same_color_points_exist_l183_183373

theorem same_color_points_exist (d : ℝ) (colored_plane : ℝ × ℝ → Prop) :
  (∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ colored_plane p1 = colored_plane p2 ∧ dist p1 p2 = d) := 
sorry

end same_color_points_exist_l183_183373


namespace geometric_seq_min_value_l183_183162

theorem geometric_seq_min_value (b : ℕ → ℝ) (s : ℝ) (h1 : b 1 = 1) (h2 : ∀ n : ℕ, b (n + 1) = s * b n) : 
  ∃ s : ℝ, 3 * b 1 + 4 * b 2 = -9 / 16 :=
by
  sorry

end geometric_seq_min_value_l183_183162


namespace football_game_attendance_l183_183603

-- Define the initial conditions
def saturday : ℕ := 80
def monday : ℕ := saturday - 20
def wednesday : ℕ := monday + 50
def friday : ℕ := saturday + monday
def total_week_actual : ℕ := saturday + monday + wednesday + friday
def total_week_expected : ℕ := 350

-- Define the proof statement
theorem football_game_attendance : total_week_actual - total_week_expected = 40 :=
by 
  -- Proof steps would go here
  sorry

end football_game_attendance_l183_183603


namespace people_not_in_pool_l183_183793

-- Define families and their members
def karen_donald_family : ℕ := 2 + 6
def tom_eva_family : ℕ := 2 + 4
def luna_aidan_family : ℕ := 2 + 5
def isabel_jake_family : ℕ := 2 + 3

-- Total number of people
def total_people : ℕ := karen_donald_family + tom_eva_family + luna_aidan_family + isabel_jake_family

-- Number of legs in the pool and people in the pool
def legs_in_pool : ℕ := 34
def people_in_pool : ℕ := legs_in_pool / 2

-- People not in the pool: people who went to store and went to bed
def store_people : ℕ := 2
def bed_people : ℕ := 3
def not_available_people : ℕ := store_people + bed_people

-- Prove (given conditions) number of people not in the pool
theorem people_not_in_pool : total_people - people_in_pool - not_available_people = 4 :=
by
  -- ...proof steps or "sorry"
  sorry

end people_not_in_pool_l183_183793


namespace edward_original_amount_l183_183395

-- Given conditions
def spent : ℝ := 16
def remaining : ℝ := 6

-- Question: How much did Edward have before he spent his money?
-- Correct answer: 22
theorem edward_original_amount : (spent + remaining) = 22 :=
by sorry

end edward_original_amount_l183_183395


namespace kevin_hop_distance_l183_183091

theorem kevin_hop_distance :
  (1/4) + (3/16) + (9/64) + (27/256) + (81/1024) + (243/4096) = 3367 / 4096 := 
by
  sorry 

end kevin_hop_distance_l183_183091


namespace num_squares_figure8_perimeter_figure12_perimeter_figureC_eq_38_ratio_perimeter_figure29_figureD_l183_183044

-- Condition: Figure 1 is formed by 3 identical squares of side length 1 cm.
def squares_in_figure1 : ℕ := 3

-- Condition: Perimeter of Figure 1 is 8 cm.
def perimeter_figure1 : ℝ := 8

-- Condition: Each subsequent figure adds 2 squares.
def squares_in_figure (n : ℕ) : ℕ :=
  squares_in_figure1 + 2 * (n - 1)

-- Condition: Each subsequent figure increases perimeter by 2 cm.
def perimeter_figure (n : ℕ) : ℝ :=
  perimeter_figure1 + 2 * (n - 1)

-- Proof problem (a): Prove that the number of squares in Figure 8 is 17.
theorem num_squares_figure8 :
  squares_in_figure 8 = 17 :=
sorry

-- Proof problem (b): Prove that the perimeter of Figure 12 is 30 cm.
theorem perimeter_figure12 :
  perimeter_figure 12 = 30 :=
sorry

-- Proof problem (c): Prove that the positive integer C for which the perimeter of Figure C is 38 cm is 16.
theorem perimeter_figureC_eq_38 :
  ∃ C : ℕ, perimeter_figure C = 38 :=
sorry

-- Proof problem (d): Prove that the positive integer D for which the ratio of the perimeter of Figure 29 to the perimeter of Figure D is 4/11 is 85.
theorem ratio_perimeter_figure29_figureD :
  ∃ D : ℕ, (perimeter_figure 29 / perimeter_figure D) = (4 / 11) :=
sorry

end num_squares_figure8_perimeter_figure12_perimeter_figureC_eq_38_ratio_perimeter_figure29_figureD_l183_183044


namespace math_team_combinations_l183_183839

def numGirls : ℕ := 4
def numBoys : ℕ := 7
def girlsToChoose : ℕ := 2
def boysToChoose : ℕ := 3

def comb (n k : ℕ) : ℕ := n.choose k

theorem math_team_combinations : 
  comb numGirls girlsToChoose * comb numBoys boysToChoose = 210 := 
by
  sorry

end math_team_combinations_l183_183839


namespace sample_size_correct_l183_183712

variable (total_employees young_employees middle_aged_employees elderly_employees young_in_sample sample_size : ℕ)

-- Conditions
def total_number_of_employees := 75
def number_of_young_employees := 35
def number_of_middle_aged_employees := 25
def number_of_elderly_employees := 15
def number_of_young_in_sample := 7
def stratified_sampling := true

-- The proof problem statement
theorem sample_size_correct :
  total_employees = total_number_of_employees ∧ 
  young_employees = number_of_young_employees ∧ 
  middle_aged_employees = number_of_middle_aged_employees ∧ 
  elderly_employees = number_of_elderly_employees ∧ 
  young_in_sample = number_of_young_in_sample ∧ 
  stratified_sampling → 
  sample_size = 15 := by sorry

end sample_size_correct_l183_183712


namespace solve_equation1_solve_equation2_solve_equation3_solve_equation4_l183_183466

theorem solve_equation1 (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
by sorry

theorem solve_equation2 (x : ℝ) : (x - 2)^2 = 5 * (x - 2) ↔ (x = 2 ∨ x = 7) :=
by sorry

theorem solve_equation3 (x : ℝ) : x^2 - 5*x + 6 = 0 ↔ (x = 2 ∨ x = 3) :=
by sorry

theorem solve_equation4 (x : ℝ) : x^2 - 4*x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

end solve_equation1_solve_equation2_solve_equation3_solve_equation4_l183_183466


namespace swim_distance_l183_183341

theorem swim_distance (v d : ℝ) (c : ℝ := 2.5) :
  (8 = d / (v + c)) ∧ (8 = 24 / (v - c)) → d = 84 :=
by
  sorry

end swim_distance_l183_183341


namespace number_of_lizards_l183_183078

theorem number_of_lizards (total_geckos : ℕ) (insects_per_gecko : ℕ) (total_insects_eaten : ℕ) (insects_per_lizard : ℕ) 
  (gecko_total_insects : total_geckos * insects_per_gecko = 5 * 6) (lizard_insects: insects_per_lizard = 2 * insects_per_gecko)
  (total_insects : total_insects_eaten = 66) : 
  (total_insects_eaten - total_geckos * insects_per_gecko) / insects_per_lizard = 3 :=
by 
  sorry

end number_of_lizards_l183_183078


namespace least_number_to_add_for_divisibility_by_11_l183_183647

theorem least_number_to_add_for_divisibility_by_11 : ∃ k : ℕ, 11002 + k ≡ 0 [MOD 11] ∧ k = 9 := by
  sorry

end least_number_to_add_for_divisibility_by_11_l183_183647


namespace find_radius_l183_183376

theorem find_radius
  (r_1 r_2 r_3 : ℝ)
  (h_cone : r_2 = 2 * r_1 ∧ r_3 = 3 * r_1 ∧ r_1 + r_2 + r_3 = 18) :
  r_1 = 3 :=
by
  sorry

end find_radius_l183_183376


namespace impossible_to_color_25_cells_l183_183275

theorem impossible_to_color_25_cells :
  ¬ ∃ (n : ℕ) (n_k : ℕ → ℕ), n = 25 ∧ (∀ k, k > 0 → k < 5 → (k % 2 = 1 → ∃ c : ℕ, n_k c = k)) :=
by
  sorry

end impossible_to_color_25_cells_l183_183275


namespace DennisHas70Marbles_l183_183995

-- Definitions according to the conditions
def LaurieMarbles : Nat := 37
def KurtMarbles : Nat := LaurieMarbles - 12
def DennisMarbles : Nat := KurtMarbles + 45

-- The proof problem statement
theorem DennisHas70Marbles : DennisMarbles = 70 :=
by
  sorry

end DennisHas70Marbles_l183_183995


namespace num_possible_y_l183_183311

theorem num_possible_y : 
  (∃ (count : ℕ), count = (54 - 26 + 1) ∧ 
  ∀ (y : ℤ), 25 < y ∧ y < 55 ↔ (26 ≤ y ∧ y ≤ 54)) :=
by {
  sorry 
}

end num_possible_y_l183_183311


namespace set_C_cannot_form_triangle_l183_183477

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given conditions
def set_A := (3, 6, 8)
def set_B := (3, 8, 9)
def set_C := (3, 6, 9)
def set_D := (6, 8, 9)

theorem set_C_cannot_form_triangle : ¬ is_triangle 3 6 9 :=
by
  -- Proof is omitted
  sorry

end set_C_cannot_form_triangle_l183_183477


namespace train_car_count_l183_183939

theorem train_car_count
    (cars_first_15_sec : ℕ)
    (time_first_15_sec : ℕ)
    (total_time_minutes : ℕ)
    (total_additional_seconds : ℕ)
    (constant_speed : Prop)
    (h1 : cars_first_15_sec = 9)
    (h2 : time_first_15_sec = 15)
    (h3 : total_time_minutes = 3)
    (h4 : total_additional_seconds = 30)
    (h5 : constant_speed) :
    0.6 * (3 * 60 + 30) = 126 := by
  sorry

end train_car_count_l183_183939


namespace determine_a_range_l183_183179

variable (a : ℝ)

-- Define proposition p as a function
def p : Prop := ∀ x : ℝ, x^2 + x > a

-- Negation of Proposition q
def not_q : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 2 - a ≠ 0

-- The main theorem to be stated, proving the range of 'a'
theorem determine_a_range (h₁ : p a) (h₂ : not_q a) : -2 < a ∧ a < -1 / 4 := sorry

end determine_a_range_l183_183179


namespace xyz_inequality_l183_183182

theorem xyz_inequality (x y z : ℝ) (h_condition : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end xyz_inequality_l183_183182


namespace find_original_cost_price_l183_183704

variable (C S : ℝ)

-- Conditions
def original_profit (C S : ℝ) : Prop := S = 1.25 * C
def new_profit_condition (C S : ℝ) : Prop := 1.04 * C = S - 12.60

-- Main Theorem
theorem find_original_cost_price (h1 : original_profit C S) (h2 : new_profit_condition C S) : C = 60 := 
sorry

end find_original_cost_price_l183_183704


namespace x_intercept_is_one_l183_183785

theorem x_intercept_is_one (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, -1)) (h2 : (x2, y2) = (-2, 3)) :
    ∃ x : ℝ, (0 = ((y2 - y1) / (x2 - x1)) * (x - x1) + y1) ∧ x = 1 :=
by
  sorry

end x_intercept_is_one_l183_183785


namespace unique_shirt_and_tie_outfits_l183_183180

theorem unique_shirt_and_tie_outfits :
  let shirts := 10
  let ties := 8
  let choose n k := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose shirts 5 * choose ties 4 = 17640 :=
by
  sorry

end unique_shirt_and_tie_outfits_l183_183180


namespace psychologist_charge_difference_l183_183960

-- Define the variables and conditions
variables (F A : ℝ)
axiom cond1 : F + 4 * A = 250
axiom cond2 : F + A = 115

theorem psychologist_charge_difference : F - A = 25 :=
by
  -- conditions are already stated as axioms, we'll just provide the target theorem
  sorry

end psychologist_charge_difference_l183_183960


namespace part1_part2_l183_183611

-- Definitions for part 1
def prop_p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def prop_q (x : ℝ) : Prop := (x - 3) / (x + 2) < 0

-- Definitions for part 2
def neg_prop_q (x : ℝ) : Prop := ¬((x - 3) / (x + 2) < 0)
def neg_prop_p (a x : ℝ) : Prop := ¬(x^2 - 4*a*x + 3*a^2 < 0)

-- Proof problems
theorem part1 (a : ℝ) (x : ℝ) (h : a = 1) (hpq : prop_p a x ∧ prop_q x) : 1 < x ∧ x < 3 := 
by
  sorry

theorem part2 (a : ℝ) (h : ∀ x, neg_prop_q x → neg_prop_p a x) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end part1_part2_l183_183611


namespace intersection_A_complement_B_l183_183547

-- Definitions of sets A and B and their complement in the universal set R, which is the real numbers.
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2 * x > 0}
def complement_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- The proof statement verifying the intersection of set A with the complement of set B.
theorem intersection_A_complement_B : A ∩ complement_R_B = {0, 1, 2} := by
  sorry

end intersection_A_complement_B_l183_183547


namespace max_ab_l183_183379

theorem max_ab {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 6) : ab ≤ 9 :=
sorry

end max_ab_l183_183379


namespace slope_tangent_line_at_x1_l183_183164

def f (x c : ℝ) : ℝ := (x-2)*(x^2 + c)
def f_prime (x c : ℝ) := (x^2 + c) + (x-2) * 2 * x

theorem slope_tangent_line_at_x1 (c : ℝ) (h : f_prime 2 c = 0) : f_prime 1 c = -5 := by
  sorry

end slope_tangent_line_at_x1_l183_183164


namespace remainder_is_20_l183_183512

theorem remainder_is_20 :
  ∀ (larger smaller quotient remainder : ℕ),
    (larger = 1634) →
    (larger - smaller = 1365) →
    (larger = quotient * smaller + remainder) →
    (quotient = 6) →
    remainder = 20 :=
by
  intros larger smaller quotient remainder h_larger h_difference h_division h_quotient
  sorry

end remainder_is_20_l183_183512


namespace binomial_square_l183_183159

theorem binomial_square (a b : ℝ) : (2 * a - 3 * b)^2 = 4 * a^2 - 12 * a * b + 9 * b^2 :=
by
  sorry

end binomial_square_l183_183159


namespace inequality_holds_if_b_greater_than_2_l183_183320

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_b_greater_than_2  :
  (b > 0) → (∃ x, |x-5| + |x-7| < b) ↔ (b > 2) := sorry

end inequality_holds_if_b_greater_than_2_l183_183320


namespace domain_of_f_l183_183006

noncomputable def domain_f : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem domain_of_f : domain_f = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end domain_of_f_l183_183006


namespace polynomial_roots_to_determinant_l183_183387

noncomputable def determinant_eq (a b c m p q : ℂ) : Prop :=
  (Matrix.det ![
    ![a, 1, 1],
    ![1, b, 1],
    ![1, 1, c]
  ] = 2 - m - q)

theorem polynomial_roots_to_determinant (a b c m p q : ℂ) 
  (h1 : Polynomial.eval a (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h2 : Polynomial.eval b (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h3 : Polynomial.eval c (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  : determinant_eq a b c m p q :=
by sorry

end polynomial_roots_to_determinant_l183_183387


namespace total_full_parking_spots_correct_l183_183990

-- Define the number of parking spots on each level
def total_parking_spots (level : ℕ) : ℕ :=
  100 + (level - 1) * 50

-- Define the number of open spots on each level
def open_parking_spots (level : ℕ) : ℕ :=
  if level = 1 then 58
  else if level <= 4 then 58 - 3 * (level - 1)
  else 49 + 10 * (level - 4)

-- Define the number of full parking spots on each level
def full_parking_spots (level : ℕ) : ℕ :=
  total_parking_spots level - open_parking_spots level

-- Sum up the full parking spots on all 7 levels to get the total full spots
def total_full_parking_spots : ℕ :=
  List.sum (List.map full_parking_spots [1, 2, 3, 4, 5, 6, 7])

-- Theorem to prove the total number of full parking spots
theorem total_full_parking_spots_correct : total_full_parking_spots = 1329 :=
by
  sorry

end total_full_parking_spots_correct_l183_183990


namespace logarithmic_function_through_point_l183_183418

noncomputable def log_function_expression (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem logarithmic_function_through_point (f : ℝ → ℝ) :
  (∀ x a : ℝ, a > 0 ∧ a ≠ 1 → f x = log_function_expression a x) ∧ f 4 = 2 →
  ∃ g : ℝ → ℝ, ∀ x : ℝ, g x = log_function_expression 2 x :=
by {
  sorry
}

end logarithmic_function_through_point_l183_183418


namespace problem_sequence_k_term_l183_183702

theorem problem_sequence_k_term (a : ℕ → ℤ) (S : ℕ → ℤ) (h₀ : ∀ n, S n = n^2 - 9 * n)
    (h₁ : ∀ n, a n = S n - S (n - 1)) (h₂ : 5 < a 8 ∧ a 8 < 8) : 8 = 8 :=
sorry

end problem_sequence_k_term_l183_183702


namespace ratio_of_150_to_10_l183_183074

theorem ratio_of_150_to_10 : 150 / 10 = 15 := by 
  sorry

end ratio_of_150_to_10_l183_183074


namespace find_number_that_satisfies_congruences_l183_183106

theorem find_number_that_satisfies_congruences :
  ∃ m : ℕ, 
  (m % 13 = 12) ∧ 
  (m % 11 = 10) ∧ 
  (m % 7 = 6) ∧ 
  (m % 5 = 4) ∧ 
  (m % 3 = 2) ∧ 
  m = 15014 :=
by
  sorry

end find_number_that_satisfies_congruences_l183_183106


namespace middle_number_l183_183860

theorem middle_number (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : a + b = 18) (h4 : a + c = 23) (h5 : b + c = 27) : b = 11 := by
  sorry

end middle_number_l183_183860


namespace businessmen_neither_coffee_nor_tea_l183_183415

theorem businessmen_neither_coffee_nor_tea :
  ∀ (total_count coffee tea both neither : ℕ),
    total_count = 30 →
    coffee = 15 →
    tea = 13 →
    both = 6 →
    neither = total_count - (coffee + tea - both) →
    neither = 8 := 
by
  intros total_count coffee tea both neither ht hc ht2 hb hn
  rw [ht, hc, ht2, hb] at hn
  simp at hn
  exact hn

end businessmen_neither_coffee_nor_tea_l183_183415


namespace expression_parity_l183_183206

theorem expression_parity (a b c : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) : (3^a + (b + 1)^2 * c) % 2 = 1 :=
by sorry

end expression_parity_l183_183206


namespace complement_union_l183_183267

open Finset

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 4}
def N : Finset ℕ := {2, 4}

theorem complement_union :
  U \ (M ∪ N) = {1, 3} := by
  sorry

end complement_union_l183_183267


namespace rosie_pies_l183_183098

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l183_183098


namespace silk_per_dress_l183_183053

theorem silk_per_dress (initial_silk : ℕ) (friends : ℕ) (silk_per_friend : ℕ) (total_dresses : ℕ)
  (h1 : initial_silk = 600)
  (h2 : friends = 5)
  (h3 : silk_per_friend = 20)
  (h4 : total_dresses = 100)
  (remaining_silk := initial_silk - friends * silk_per_friend) :
  remaining_silk / total_dresses = 5 :=
by
  -- proof goes here
  sorry

end silk_per_dress_l183_183053


namespace fewer_cans_l183_183692

theorem fewer_cans (sarah_yesterday lara_more alex_yesterday sarah_today lara_today alex_today : ℝ)
  (H1 : sarah_yesterday = 50.5)
  (H2 : lara_more = 30.3)
  (H3 : alex_yesterday = 90.2)
  (H4 : sarah_today = 40.7)
  (H5 : lara_today = 70.5)
  (H6 : alex_today = 55.3) :
  (sarah_yesterday + (sarah_yesterday + lara_more) + alex_yesterday) - (sarah_today + lara_today + alex_today) = 55 :=
by {
  -- Sorry to skip the proof
  sorry
}

end fewer_cans_l183_183692


namespace min_x_squared_plus_y_squared_l183_183402

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end min_x_squared_plus_y_squared_l183_183402


namespace complete_the_square_l183_183295

theorem complete_the_square (x : ℝ) : 
  ∃ (a h k : ℝ), a = 1 ∧ h = 7 / 2 ∧ k = -49 / 4 ∧ x^2 - 7 * x = a * (x - h) ^ 2 + k :=
by
  use 1, 7 / 2, -49 / 4
  sorry

end complete_the_square_l183_183295


namespace vacuum_cleaner_cost_l183_183055

-- Variables
variables (V : ℝ)

-- Conditions
def cost_of_dishwasher := 450
def coupon := 75
def total_spent := 625

-- The main theorem to prove
theorem vacuum_cleaner_cost : V + cost_of_dishwasher - coupon = total_spent → V = 250 :=
by
  -- Proof logic goes here
  sorry

end vacuum_cleaner_cost_l183_183055


namespace inequality_chain_l183_183797

theorem inequality_chain (a : ℝ) (h : a - 1 > 0) : -a < -1 ∧ -1 < 1 ∧ 1 < a := by
  sorry

end inequality_chain_l183_183797


namespace depth_of_pond_l183_183469

theorem depth_of_pond (L W V D : ℝ) (hL : L = 20) (hW : W = 10) (hV : V = 1000) (hV_formula : V = L * W * D) : D = 5 := by
  -- at this point, you could start the proof which involves deriving D from hV and hV_formula using arithmetic rules.
  sorry

end depth_of_pond_l183_183469


namespace larger_number_l183_183456

theorem larger_number (x y : ℤ) (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
  sorry

end larger_number_l183_183456


namespace photos_last_weekend_45_l183_183714

theorem photos_last_weekend_45 (photos_animals photos_flowers photos_scenery total_photos_this_weekend photos_last_weekend : ℕ)
  (h1 : photos_animals = 10)
  (h2 : photos_flowers = 3 * photos_animals)
  (h3 : photos_scenery = photos_flowers - 10)
  (h4 : total_photos_this_weekend = photos_animals + photos_flowers + photos_scenery)
  (h5 : photos_last_weekend = total_photos_this_weekend - 15) :
  photos_last_weekend = 45 :=
sorry

end photos_last_weekend_45_l183_183714


namespace largest_y_coordinate_l183_183292

theorem largest_y_coordinate (x y : ℝ) :
  (x - 3)^2 / 49 + (y - 2)^2 / 25 = 0 → y = 2 := 
by 
  -- Proof will be provided here
  sorry

end largest_y_coordinate_l183_183292


namespace num_points_on_ellipse_with_area_l183_183777

-- Define the line equation
def line_eq (x y : ℝ) : Prop := (x / 4) + (y / 3) = 1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1

-- Define the area condition for the triangle
def area_condition (xA yA xB yB xP yP : ℝ) : Prop :=
  abs (xA * (yB - yP) + xB * (yP - yA) + xP * (yA - yB)) = 6

-- Define the main theorem statement
theorem num_points_on_ellipse_with_area (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  ∃ P1 P2 : ℝ × ℝ, 
    (ellipse_eq P1.1 P1.2) ∧ 
    (ellipse_eq P2.1 P2.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P1.1 P1.2) ∧ 
    (area_condition A.1 A.2 B.1 B.2 P2.1 P2.2) ∧ 
    P1 ≠ P2 := sorry

end num_points_on_ellipse_with_area_l183_183777


namespace foreman_can_establish_corr_foreman_cannot_with_less_l183_183260

-- Define the given conditions:
def num_rooms (n : ℕ) := 2^n
def num_checks (n : ℕ) := 2 * n

-- Part (a)
theorem foreman_can_establish_corr (n : ℕ) : 
  ∃ (c : ℕ), c = num_checks n ∧ (c ≥ 2 * n) :=
by
  sorry

-- Part (b)
theorem foreman_cannot_with_less (n : ℕ) : 
  ¬ (∃ (c : ℕ), c = 2 * n - 1 ∧ (c < 2 * n)) :=
by
  sorry

end foreman_can_establish_corr_foreman_cannot_with_less_l183_183260


namespace compute_fraction_sum_l183_183024

variable (a b c : ℝ)
open Real

theorem compute_fraction_sum (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -15)
                            (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 6) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 12 := 
sorry

end compute_fraction_sum_l183_183024


namespace max_square_plots_l183_183666

theorem max_square_plots (length width available_fencing : ℕ) 
(h : length = 30 ∧ width = 60 ∧ available_fencing = 2500) : 
  ∃ n : ℕ, n = 72 ∧ ∀ s : ℕ, ((30 * (60 / s - 1)) + (60 * (30 / s - 1)) ≤ 2500) → ((30 / s) * (60 / s) = n) := by
  sorry

end max_square_plots_l183_183666


namespace gen_formula_arithmetic_seq_sum_maximizes_at_5_l183_183312

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a n = a 1 + (n - 1) * d

variables (an : ℕ → ℤ) (Sn : ℕ → ℤ)
variable (d : ℤ)

theorem gen_formula_arithmetic_seq (h1 : an 3 = 5) (h2 : an 10 = -9) :
  ∀ n, an n = 11 - 2 * n :=
sorry

theorem sum_maximizes_at_5 (h_seq : ∀ n, an n = 11 - 2 * n) :
  ∀ n, Sn n = (n * 10 - n^2) → (∃ n, ∀ k, Sn n ≥ Sn k) :=
sorry

end gen_formula_arithmetic_seq_sum_maximizes_at_5_l183_183312


namespace chord_slope_of_ellipse_l183_183675

theorem chord_slope_of_ellipse :
  (∃ (x1 y1 x2 y2 : ℝ), (x1 + x2)/2 = 4 ∧ (y1 + y2)/2 = 2 ∧
    (x1^2)/36 + (y1^2)/9 = 1 ∧ (x2^2)/36 + (y2^2)/9 = 1) →
    (∃ k : ℝ, k = (y1 - y2)/(x1 - x2) ∧ k = -1/2) :=
sorry

end chord_slope_of_ellipse_l183_183675


namespace complement_of_M_in_U_l183_183480

def U := Set.univ (α := ℝ)
def M := {x : ℝ | x < -2 ∨ x > 8}
def compl_M := {x : ℝ | -2 ≤ x ∧ x ≤ 8}

theorem complement_of_M_in_U : compl_M = U \ M :=
by
  sorry

end complement_of_M_in_U_l183_183480


namespace perpendicular_tangent_line_exists_and_correct_l183_183421

theorem perpendicular_tangent_line_exists_and_correct :
  ∃ L : ℝ → ℝ → Prop,
    (∀ x y, L x y ↔ 3 * x + y + 6 = 0) ∧
    (∀ x y, 2 * x - 6 * y + 1 = 0 → 3 * x + y + 6 ≠ 0) ∧
    (∃ a b : ℝ, 
       b = a^3 + 3*a^2 - 5 ∧ 
       (a, b) ∈ { p : ℝ × ℝ | ∃ f' : ℝ → ℝ, f' a = 3 * a^2 + 6 * a ∧ f' a * 3 + 1 = 0 } ∧
       L a b)
:= 
sorry

end perpendicular_tangent_line_exists_and_correct_l183_183421


namespace relation_between_a_b_c_l183_183134

theorem relation_between_a_b_c :
  let a := (3/7 : ℝ) ^ (2/7)
  let b := (2/7 : ℝ) ^ (3/7)
  let c := (2/7 : ℝ) ^ (2/7)
  a > c ∧ c > b :=
by {
  let a := (3/7 : ℝ) ^ (2/7)
  let b := (2/7 : ℝ) ^ (3/7)
  let c := (2/7 : ℝ) ^ (2/7)
  sorry
}

end relation_between_a_b_c_l183_183134


namespace ned_price_per_game_l183_183583

def number_of_games : Nat := 15
def non_working_games : Nat := 6
def total_earnings : Nat := 63
def number_of_working_games : Nat := number_of_games - non_working_games
def price_per_working_game : Nat := total_earnings / number_of_working_games

theorem ned_price_per_game : price_per_working_game = 7 :=
by
  sorry

end ned_price_per_game_l183_183583


namespace part1_part2_l183_183013

def A (x y : ℝ) : ℝ := 3 * x ^ 2 + 2 * x * y - 2 * x - 1
def B (x y : ℝ) : ℝ := - x ^ 2 + x * y - 1

theorem part1 (x y : ℝ) : A x y + 3 * B x y = 5 * x * y - 2 * x - 4 := by
  sorry

theorem part2 (y : ℝ) : (∀ x : ℝ, 5 * x * y - 2 * x - 4 = -4) → y = 2 / 5 := by
  sorry

end part1_part2_l183_183013


namespace xyz_value_l183_183634

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24) 
  (h2 : x ^ 2 * (y + z) + y ^ 2 * (x + z) + z ^ 2 * (x + y) = 9) : 
  x * y * z = 5 :=
by
  sorry

end xyz_value_l183_183634


namespace probability_all_quit_same_tribe_l183_183345

-- Define the number of participants and the number of tribes
def numParticipants : ℕ := 18
def numTribes : ℕ := 2
def tribeSize : ℕ := 9 -- Each tribe has 9 members

-- Define the problem statement
theorem probability_all_quit_same_tribe : 
  (numParticipants.choose 3) = 816 ∧
  ((tribeSize.choose 3) * numTribes) = 168 ∧
  ((tribeSize.choose 3) * numTribes) / (numParticipants.choose 3) = 7 / 34 :=
by
  sorry

end probability_all_quit_same_tribe_l183_183345


namespace max_ball_height_l183_183485

/-- 
The height (in feet) of a ball traveling on a parabolic path is given by -20t^2 + 80t + 36,
where t is the time after launch. This theorem shows that the maximum height of the ball is 116 feet.
-/
theorem max_ball_height : ∃ t : ℝ, ∀ t', -20 * t^2 + 80 * t + 36 ≤ -20 * t'^2 + 80 * t' + 36 → -20 * t^2 + 80 * t + 36 = 116 :=
sorry

end max_ball_height_l183_183485


namespace real_solution_x_condition_l183_183075

theorem real_solution_x_condition (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 + 6 * x * y + 2 * x + 1 = 0) ↔ (x < 2 - Real.sqrt 6 ∨ x > 2 + Real.sqrt 6) :=
by
  sorry

end real_solution_x_condition_l183_183075


namespace red_side_probability_l183_183668

theorem red_side_probability
  (num_cards : ℕ)
  (num_black_black : ℕ)
  (num_black_red : ℕ)
  (num_red_red : ℕ)
  (num_red_sides_total : ℕ)
  (num_red_sides_with_red_other_side : ℕ) :
  num_cards = 8 →
  num_black_black = 4 →
  num_black_red = 2 →
  num_red_red = 2 →
  num_red_sides_total = (num_red_red * 2 + num_black_red) →
  num_red_sides_with_red_other_side = (num_red_red * 2) →
  (num_red_sides_with_red_other_side / num_red_sides_total : ℝ) = 2 / 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end red_side_probability_l183_183668


namespace geometric_sequence_fourth_term_l183_183844

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) 
  (h1 : 3 * x + 3 = r * x)
  (h2 : 6 * x + 6 = r * (3 * x + 3)) :
  x = -3 ∧ r = 2 → (x * r^3 = -24) :=
by
  sorry

end geometric_sequence_fourth_term_l183_183844


namespace minimize_f_l183_183816

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end minimize_f_l183_183816


namespace average_diesel_rate_l183_183594

theorem average_diesel_rate (r1 r2 r3 r4 : ℝ) (H1: (r1 + r2 + r3 + r4) / 4 = 1.52) :
    ((r1 + r2 + r3 + r4) / 4 = 1.52) :=
by
  exact H1

end average_diesel_rate_l183_183594


namespace initial_fish_count_l183_183019

variable (x : ℕ)

theorem initial_fish_count (initial_fish : ℕ) (given_fish : ℕ) (total_fish : ℕ)
  (h1 : total_fish = initial_fish + given_fish)
  (h2 : total_fish = 69)
  (h3 : given_fish = 47) :
  initial_fish = 22 :=
by
  sorry

end initial_fish_count_l183_183019


namespace ab_value_l183_183772

theorem ab_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 30) (h4 : 3 * a * b + 5 * a = 4 * b + 180) : a * b = 29 :=
sorry

end ab_value_l183_183772


namespace pizza_left_for_Wally_l183_183676

theorem pizza_left_for_Wally (a b c : ℚ) (ha : a = 1/3) (hb : b = 1/6) (hc : c = 1/4) :
  1 - (a + b + c) = 1/4 :=
by
  sorry

end pizza_left_for_Wally_l183_183676


namespace point_reflection_l183_183043

-- Definition of point and reflection over x-axis
def P : ℝ × ℝ := (-2, 3)

def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Statement to prove
theorem point_reflection : reflect_x_axis P = (-2, -3) :=
by
  -- Proof goes here
  sorry

end point_reflection_l183_183043


namespace constant_condition_for_quadrant_I_solution_l183_183392

-- Define the given conditions
def equations (c : ℚ) (x y : ℚ) : Prop :=
  (x - 2 * y = 5) ∧ (c * x + 3 * y = 2)

-- Define the condition for the solution to be in Quadrant I
def isQuadrantI (x y : ℚ) : Prop :=
  (x > 0) ∧ (y > 0)

-- The theorem to be proved
theorem constant_condition_for_quadrant_I_solution (c : ℚ) :
  (∃ x y : ℚ, equations c x y ∧ isQuadrantI x y) ↔ (-3/2 < c ∧ c < 2/5) :=
by
  sorry

end constant_condition_for_quadrant_I_solution_l183_183392


namespace vector_dot_product_parallel_l183_183971

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, -1)
noncomputable def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem vector_dot_product_parallel (m : ℝ) (h_parallel : is_parallel a (a.1 + m, a.2 + (-1))) :
  (a.1 * m + a.2 * (-1) = -5 / 2) :=
sorry

end vector_dot_product_parallel_l183_183971


namespace necessary_but_not_sufficient_l183_183187

-- Define sets M and N
def M (x : ℝ) : Prop := x < 5
def N (x : ℝ) : Prop := x > 3

-- Define the union and intersection of M and N
def M_union_N (x : ℝ) : Prop := M x ∨ N x
def M_inter_N (x : ℝ) : Prop := M x ∧ N x

-- Theorem statement: Prove the necessity but not sufficiency
theorem necessary_but_not_sufficient (x : ℝ) :
  M_inter_N x → M_union_N x ∧ ¬(M_union_N x → M_inter_N x) := 
sorry

end necessary_but_not_sufficient_l183_183187


namespace min_value_a_plus_3b_l183_183734

theorem min_value_a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a * b - 3 = a + 3 * b) :
  ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, y = a + 3 * b → y ≥ 6 :=
sorry

end min_value_a_plus_3b_l183_183734


namespace point_on_line_l183_183189

theorem point_on_line (m : ℝ) (P : ℝ × ℝ) (line_eq : ℝ × ℝ → Prop) (h : P = (2, m)) 
  (h_line : line_eq = fun P => 3 * P.1 + P.2 = 2) : 
  3 * 2 + m = 2 → m = -4 :=
by
  intro h1
  linarith

end point_on_line_l183_183189


namespace remainder_of_large_power_l183_183350

def powerMod (base exp mod_ : ℕ) : ℕ := (base ^ exp) % mod_

theorem remainder_of_large_power :
  powerMod 2 (2^(2^2)) 500 = 536 :=
sorry

end remainder_of_large_power_l183_183350


namespace lawn_care_company_expense_l183_183001

theorem lawn_care_company_expense (cost_blade : ℕ) (num_blades : ℕ) (cost_string : ℕ) :
  cost_blade = 8 → num_blades = 4 → cost_string = 7 → 
  (num_blades * cost_blade + cost_string = 39) :=
by
  intro h1 h2 h3
  sorry

end lawn_care_company_expense_l183_183001


namespace Margo_James_pairs_probability_l183_183677

def total_students : ℕ := 32
def Margo_pairs_prob : ℚ := 1 / 31
def James_pairs_prob : ℚ := 1 / 30
def total_prob : ℚ := Margo_pairs_prob * James_pairs_prob

theorem Margo_James_pairs_probability :
  total_prob = 1 / 930 := 
by
  -- sorry allows us to skip the proof steps, only statement needed
  sorry

end Margo_James_pairs_probability_l183_183677


namespace sqrt_four_eq_two_or_neg_two_l183_183378

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 :=
by 
  sorry

end sqrt_four_eq_two_or_neg_two_l183_183378


namespace square_binomial_constant_l183_183058

theorem square_binomial_constant (y : ℝ) : ∃ b : ℝ, (y^2 + 12*y + 50 = (y + 6)^2 + b) ∧ b = 14 := 
by
  sorry

end square_binomial_constant_l183_183058


namespace false_converse_implication_l183_183649

theorem false_converse_implication : ∃ x : ℝ, (0 < x) ∧ (x - 3 ≤ 0) := by
  sorry

end false_converse_implication_l183_183649


namespace gcd_fact_8_10_l183_183111

-- Definitions based on the conditions in a)
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Question and conditions translated to a proof problem in Lean
theorem gcd_fact_8_10 : Nat.gcd (fact 8) (fact 10) = 40320 := by
  sorry

end gcd_fact_8_10_l183_183111


namespace total_age_of_siblings_l183_183640

def age_total (Susan Arthur Tom Bob : ℕ) : ℕ := Susan + Arthur + Tom + Bob

theorem total_age_of_siblings :
  ∀ (Susan Arthur Tom Bob : ℕ),
    (Arthur = Susan + 2) →
    (Tom = Bob - 3) →
    (Bob = 11) →
    (Susan = 15) →
    age_total Susan Arthur Tom Bob = 51 :=
by
  intros Susan Arthur Tom Bob h1 h2 h3 h4
  rw [h4, h1, h3, h2]    -- Use the conditions
  norm_num               -- Simplify numerical expressions
  sorry                  -- Placeholder for the proof

end total_age_of_siblings_l183_183640


namespace problem_l183_183235

theorem problem (x : ℝ) (h : 15 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = Real.sqrt 17 := 
by sorry

end problem_l183_183235


namespace compute_sixth_power_sum_l183_183487

theorem compute_sixth_power_sum (ζ1 ζ2 ζ3 : ℂ) 
  (h1 : ζ1 + ζ2 + ζ3 = 2)
  (h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5)
  (h3 : ζ1^4 + ζ2^4 + ζ3^4 = 29) :
  ζ1^6 + ζ2^6 + ζ3^6 = 101.40625 := 
by
  sorry

end compute_sixth_power_sum_l183_183487


namespace calculate_total_interest_rate_l183_183648

noncomputable def total_investment : ℝ := 10000
noncomputable def amount_invested_11_percent : ℝ := 3750
noncomputable def amount_invested_9_percent : ℝ := total_investment - amount_invested_11_percent
noncomputable def interest_rate_9_percent : ℝ := 0.09
noncomputable def interest_rate_11_percent : ℝ := 0.11

noncomputable def interest_from_9_percent : ℝ := interest_rate_9_percent * amount_invested_9_percent
noncomputable def interest_from_11_percent : ℝ := interest_rate_11_percent * amount_invested_11_percent

noncomputable def total_interest : ℝ := interest_from_9_percent + interest_from_11_percent

noncomputable def total_interest_rate : ℝ := (total_interest / total_investment) * 100

theorem calculate_total_interest_rate :
  total_interest_rate = 9.75 :=
by 
  sorry

end calculate_total_interest_rate_l183_183648


namespace variance_of_X_is_correct_l183_183484

/-!
  There is a batch of products, among which there are 12 genuine items and 4 defective items.
  If 3 items are drawn with replacement, and X represents the number of defective items drawn,
  prove that the variance of X is 9 / 16 given that X follows a binomial distribution B(3, 1 / 4).
-/

noncomputable def variance_of_binomial : Prop :=
  let n := 3
  let p := 1 / 4
  let variance := n * p * (1 - p)
  variance = 9 / 16

theorem variance_of_X_is_correct : variance_of_binomial := by
  sorry

end variance_of_X_is_correct_l183_183484


namespace position_1011th_square_l183_183168

-- Define the initial position and transformations
inductive SquarePosition
| ABCD : SquarePosition
| DABC : SquarePosition
| BADC : SquarePosition
| DCBA : SquarePosition

open SquarePosition

def R1 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DABC
  | DABC => BADC
  | BADC => DCBA
  | DCBA => ABCD

def R2 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DCBA
  | DCBA => ABCD
  | DABC => BADC
  | BADC => DABC

def transform : ℕ → SquarePosition
| 0 => ABCD
| n + 1 => if n % 2 = 0 then R1 (transform n) else R2 (transform n)

theorem position_1011th_square : transform 1011 = DCBA :=
by {
  sorry
}

end position_1011th_square_l183_183168


namespace value_of_a_l183_183302

theorem value_of_a (M : Set ℝ) (N : Set ℝ) (a : ℝ) 
  (hM : M = {-1, 0, 1, 2}) (hN : N = {x | x^2 - a * x < 0}) 
  (hIntersect : M ∩ N = {1, 2}) : 
  a = 3 := 
sorry

end value_of_a_l183_183302


namespace sandra_socks_l183_183453

variables (x y z : ℕ)

theorem sandra_socks :
  x + y + z = 15 →
  2 * x + 3 * y + 5 * z = 36 →
  x ≤ 6 →
  y ≤ 6 →
  z ≤ 6 →
  x = 11 :=
by
  sorry

end sandra_socks_l183_183453


namespace price_of_each_armchair_l183_183885

theorem price_of_each_armchair
  (sofa_price : ℕ)
  (coffee_table_price : ℕ)
  (total_invoice : ℕ)
  (num_armchairs : ℕ)
  (h_sofa : sofa_price = 1250)
  (h_coffee_table : coffee_table_price = 330)
  (h_invoice : total_invoice = 2430)
  (h_num_armchairs : num_armchairs = 2) :
  (total_invoice - (sofa_price + coffee_table_price)) / num_armchairs = 425 := 
by 
  sorry

end price_of_each_armchair_l183_183885


namespace weight_mixture_is_correct_l183_183674

noncomputable def weight_mixture_in_kg (weight_a_per_liter weight_b_per_liter : ℝ)
  (ratio_a ratio_b total_volume_liters weight_conversion : ℝ) : ℝ :=
  let total_parts := ratio_a + ratio_b
  let volume_per_part := total_volume_liters / total_parts
  let volume_a := ratio_a * volume_per_part
  let volume_b := ratio_b * volume_per_part
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  total_weight_gm / weight_conversion

theorem weight_mixture_is_correct :
  weight_mixture_in_kg 900 700 3 2 4 1000 = 3.280 :=
by
  -- Calculation should follow from the def
  sorry

end weight_mixture_is_correct_l183_183674


namespace farmer_plough_remaining_area_l183_183656

theorem farmer_plough_remaining_area :
  ∀ (x R : ℕ),
  (90 * x = 3780) →
  (85 * (x + 2) + R = 3780) →
  R = 40 :=
by
  intros x R h1 h2
  sorry

end farmer_plough_remaining_area_l183_183656


namespace trip_total_time_trip_average_speed_l183_183830

structure Segment where
  distance : ℝ -- in kilometers
  speed : ℝ -- average speed in km/hr
  break_time : ℝ -- in minutes

def seg1 := Segment.mk 12 13 15
def seg2 := Segment.mk 18 16 30
def seg3 := Segment.mk 25 20 45
def seg4 := Segment.mk 35 25 60
def seg5 := Segment.mk 50 22 0

noncomputable def total_time_minutes (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + (s.distance / s.speed) * 60 + s.break_time) 0

noncomputable def total_distance (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + s.distance) 0

noncomputable def overall_average_speed (segs : List Segment) : ℝ :=
  total_distance segs / (total_time_minutes segs / 60)

def segments := [seg1, seg2, seg3, seg4, seg5]

theorem trip_total_time : total_time_minutes segments = 568.24 := by sorry
theorem trip_average_speed : overall_average_speed segments = 14.78 := by sorry

end trip_total_time_trip_average_speed_l183_183830


namespace lindy_distance_traveled_l183_183344

/-- Jack and Christina are standing 240 feet apart on a level surface. 
Jack walks in a straight line toward Christina at a constant speed of 5 feet per second. 
Christina walks in a straight line toward Jack at a constant speed of 3 feet per second. 
Lindy runs at a constant speed of 9 feet per second from Christina to Jack, back to Christina, back to Jack, and so forth. 
The total distance Lindy travels when the three meet at one place is 270 feet. -/
theorem lindy_distance_traveled
    (initial_distance : ℝ)
    (jack_speed : ℝ)
    (christina_speed : ℝ)
    (lindy_speed : ℝ)
    (time_to_meet : ℝ)
    (total_distance_lindy : ℝ) :
    initial_distance = 240 ∧
    jack_speed = 5 ∧
    christina_speed = 3 ∧
    lindy_speed = 9 ∧
    time_to_meet = (initial_distance / (jack_speed + christina_speed)) ∧
    total_distance_lindy = lindy_speed * time_to_meet →
    total_distance_lindy = 270 :=
by
  sorry

end lindy_distance_traveled_l183_183344


namespace johnny_weekly_earnings_l183_183856

-- Define the conditions mentioned in the problem.
def number_of_dogs_at_once : ℕ := 3
def thirty_minute_walk_payment : ℝ := 15
def sixty_minute_walk_payment : ℝ := 20
def work_hours_per_day : ℝ := 4
def sixty_minute_walks_needed_per_day : ℕ := 6
def work_days_per_week : ℕ := 5

-- Prove Johnny's weekly earnings given the conditions
theorem johnny_weekly_earnings :
  let sixty_minute_walks_per_day := sixty_minute_walks_needed_per_day / number_of_dogs_at_once
  let sixty_minute_earnings_per_day := sixty_minute_walks_per_day * number_of_dogs_at_once * sixty_minute_walk_payment
  let remaining_hours_per_day := work_hours_per_day - sixty_minute_walks_per_day
  let thirty_minute_walks_per_day := remaining_hours_per_day * 2 -- each 30-minute walk takes 0.5 hours
  let thirty_minute_earnings_per_day := thirty_minute_walks_per_day * number_of_dogs_at_once * thirty_minute_walk_payment
  let daily_earnings := sixty_minute_earnings_per_day + thirty_minute_earnings_per_day
  let weekly_earnings := daily_earnings * work_days_per_week
  weekly_earnings = 1500 :=
by
  sorry

end johnny_weekly_earnings_l183_183856


namespace min_value_expr_l183_183709

theorem min_value_expr (a b : ℝ) (h1 : 2 * a + b = a * b) (h2 : a > 0) (h3 : b > 0) : 
  ∃ a b, (a > 0 ∧ b > 0 ∧ 2 * a + b = a * b) ∧ (∀ x y, (x > 0 ∧ y > 0 ∧ 2 * x + y = x * y) → (1 / (x - 1) + 2 / (y - 2)) ≥ 2) ∧ ((1 / (a - 1) + 2 / (b - 2)) = 2) :=
by
  sorry

end min_value_expr_l183_183709


namespace solve_system_l183_183155

def system_of_equations (x y : ℤ) : Prop :=
  (x^2 * y + x * y^2 + 3 * x + 3 * y + 24 = 0) ∧
  (x^3 * y - x * y^3 + 3 * x^2 - 3 * y^2 - 48 = 0)

theorem solve_system : system_of_equations (-3) (-1) :=
by {
  -- Proof details are omitted
  sorry
}

end solve_system_l183_183155


namespace paint_snake_l183_183876

theorem paint_snake (num_cubes : ℕ) (paint_per_cube : ℕ) (end_paint : ℕ) (total_paint : ℕ) 
  (h_cubes : num_cubes = 2016)
  (h_paint_per_cube : paint_per_cube = 60)
  (h_end_paint : end_paint = 20)
  (h_total_paint : total_paint = 121000) :
  total_paint = (num_cubes * paint_per_cube) + 2 * end_paint :=
by
  rw [h_cubes, h_paint_per_cube, h_end_paint]
  sorry

end paint_snake_l183_183876


namespace anna_total_value_l183_183494

theorem anna_total_value (total_bills : ℕ) (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ)
  (h1 : total_bills = 12) (h2 : five_dollar_bills = 4) (h3 : ten_dollar_bills = total_bills - five_dollar_bills) :
  5 * five_dollar_bills + 10 * ten_dollar_bills = 100 := by
  sorry

end anna_total_value_l183_183494


namespace find_twentieth_special_number_l183_183448

theorem find_twentieth_special_number :
  ∃ n : ℕ, (n ≡ 2 [MOD 3]) ∧ (n ≡ 5 [MOD 8]) ∧ (∀ k < 20, ∃ m : ℕ, (m ≡ 2 [MOD 3]) ∧ (m ≡ 5 [MOD 8]) ∧ m < n) ∧ (n = 461) := 
sorry

end find_twentieth_special_number_l183_183448


namespace lines_intersect_at_point_l183_183551

def ParametricLine1 (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 4 - 3 * t)

def ParametricLine2 (u : ℝ) : ℝ × ℝ :=
  (-2 + 3 * u, 5 - u)

theorem lines_intersect_at_point :
  ∃ t u : ℝ, ParametricLine1 t = ParametricLine2 u ∧ ParametricLine1 t = (-5, 13) :=
by
  sorry

end lines_intersect_at_point_l183_183551


namespace coplanar_lines_condition_l183_183723

theorem coplanar_lines_condition (h : ℝ) : 
  (∃ c : ℝ, 
    (2 : ℝ) = 3 * c ∧ 
    (-1 : ℝ) = c ∧ 
    (h : ℝ) = -2 * c) ↔ 
  (h = 2) :=
by
  sorry

end coplanar_lines_condition_l183_183723


namespace parallelogram_rectangle_l183_183369

/-- A quadrilateral is a parallelogram if both pairs of opposite sides are equal,
and it is a rectangle if its diagonals are equal. -/
structure Quadrilateral :=
  (side1 side2 side3 side4 : ℝ)
  (diag1 diag2 : ℝ)

structure Parallelogram extends Quadrilateral :=
  (opposite_sides_equal : side1 = side3 ∧ side2 = side4)

def is_rectangle (p : Parallelogram) : Prop :=
  p.diag1 = p.diag2 → (p.side1^2 + p.side2^2 = p.side3^2 + p.side4^2)

theorem parallelogram_rectangle (p : Parallelogram) : is_rectangle p :=
  sorry

end parallelogram_rectangle_l183_183369


namespace inequality_xy_gt_xz_l183_183407

theorem inequality_xy_gt_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 1) : 
  x * y > x * z := 
by
  sorry  -- Proof is not required as per the instructions

end inequality_xy_gt_xz_l183_183407


namespace find_x_plus_y_l183_183515

-- Define the points A, B, and C with given conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := 1}
def C : Point := {x := 2, y := 4}

-- Define what it means for C to divide AB in the ratio 2:1
open Point

def divides_in_ratio (A B C : Point) (r₁ r₂ : ℝ) :=
  (C.x = (r₁ * A.x + r₂ * B.x) / (r₁ + r₂))
  ∧ (C.y = (r₁ * A.y + r₂ * B.y) / (r₁ + r₂))

-- Prove that x + y = 8 given the conditions
theorem find_x_plus_y {x y : ℝ} (B : Point) (H_B : B = {x := x, y := y}) :
  divides_in_ratio A B C 2 1 →
  x + y = 8 :=
by
  intro h
  sorry

end find_x_plus_y_l183_183515


namespace Manu_takes_12_more_seconds_l183_183751

theorem Manu_takes_12_more_seconds (P M A : ℕ) 
  (hP : P = 60) 
  (hA1 : A = 36) 
  (hA2 : A = M / 2) : 
  M - P = 12 :=
by
  sorry

end Manu_takes_12_more_seconds_l183_183751


namespace friends_count_l183_183163

noncomputable def university_students := 1995

theorem friends_count (students : ℕ)
  (knows_each_other : (ℕ → ℕ → Prop))
  (acquaintances : ℕ → ℕ)
  (h_university_students : students = university_students)
  (h_knows_iff_same_acq : ∀ a b, knows_each_other a b ↔ acquaintances a = acquaintances b)
  (h_not_knows_iff_diff_acq : ∀ a b, ¬ knows_each_other a b ↔ acquaintances a ≠ acquaintances b) :
  ∃ a, acquaintances a ≥ 62 ∧ ¬ ∃ a, acquaintances a ≥ 63 :=
sorry

end friends_count_l183_183163


namespace inequality_proof_l183_183678

theorem inequality_proof (x y : ℝ) (hx: 0 < x) (hy: 0 < y) : 
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧ 
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 := 
by 
  sorry

end inequality_proof_l183_183678


namespace absents_probability_is_correct_l183_183248

-- Conditions
def probability_absent := 1 / 10
def probability_present := 9 / 10

-- Calculation of combined probability
def combined_probability : ℚ :=
  3 * (probability_absent * probability_absent * probability_present)

-- Conversion to percentage
def percentage_probability : ℚ :=
  combined_probability * 100

-- Theorem statement
theorem absents_probability_is_correct :
  percentage_probability = 2.7 := 
sorry

end absents_probability_is_correct_l183_183248


namespace peanut_butter_last_days_l183_183757

-- Definitions for the problem conditions
def daily_consumption : ℕ := 2
def servings_per_jar : ℕ := 15
def num_jars : ℕ := 4

-- The statement to prove
theorem peanut_butter_last_days : 
  (num_jars * servings_per_jar) / daily_consumption = 30 :=
by
  sorry

end peanut_butter_last_days_l183_183757


namespace books_a_count_l183_183633

-- Variables representing the number of books (a) and (b)
variables (A B : ℕ)

-- Conditions given in the problem
def condition1 : Prop := A + B = 20
def condition2 : Prop := A = B + 4

-- The theorem to prove
theorem books_a_count (h1 : condition1 A B) (h2 : condition2 A B) : A = 12 :=
sorry

end books_a_count_l183_183633


namespace probability_of_friends_in_same_lunch_group_l183_183871

theorem probability_of_friends_in_same_lunch_group :
  let groups := 4
  let students := 720
  let group_size := students / groups
  let probability := (1 / groups) * (1 / groups) * (1 / groups)
  students % groups = 0 ->  -- Students can be evenly divided into groups
  groups > 0 ->             -- There is at least one group
  probability = (1 : ℝ) / 64 :=
by
  intros
  sorry

end probability_of_friends_in_same_lunch_group_l183_183871


namespace find_a_l183_183894

theorem find_a (k x y a : ℝ) (hkx : k ≤ x) (hx3 : x ≤ 3) (hy7 : a ≤ y) (hy7' : y ≤ 7) (hy : y = k * x + 1) :
  a = 5 ∨ a = 1 - 3 * Real.sqrt 6 :=
sorry

end find_a_l183_183894


namespace ratio_sum_of_arithmetic_sequences_l183_183632

-- Definitions for the arithmetic sequences
def a_num := 3
def d_num := 3
def l_num := 99

def a_den := 4
def d_den := 4
def l_den := 96

-- Number of terms in each sequence
def n_num := (l_num - a_num) / d_num + 1
def n_den := (l_den - a_den) / d_den + 1

-- Sum of the sequences using the sum formula for arithmetic series
def S_num := n_num * (a_num + l_num) / 2
def S_den := n_den * (a_den + l_den) / 2

-- The theorem statement
theorem ratio_sum_of_arithmetic_sequences : S_num / S_den = 1683 / 1200 := by sorry

end ratio_sum_of_arithmetic_sequences_l183_183632


namespace range_of_m_l183_183397

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h₁ : x1 < x2) (h₂ : y1 < y2)
  (A_on_line : y1 = (2 * m - 1) * x1 + 1)
  (B_on_line : y2 = (2 * m - 1) * x2 + 1) :
  m > 0.5 :=
sorry

end range_of_m_l183_183397


namespace license_plates_count_l183_183036

-- Definitions from conditions
def num_digits : ℕ := 4
def num_digits_choices : ℕ := 10
def num_letters : ℕ := 3
def num_letters_choices : ℕ := 26

-- Define the blocks and their possible arrangements
def digits_permutations : ℕ := num_digits_choices^num_digits
def letters_permutations : ℕ := num_letters_choices^num_letters
def block_positions : ℕ := 5

-- We need to show that total possible license plates is 878,800,000.
def total_plates : ℕ := digits_permutations * letters_permutations * block_positions

-- The theorem statement
theorem license_plates_count :
  total_plates = 878800000 := by
  sorry

end license_plates_count_l183_183036


namespace mitch_family_milk_l183_183120

variable (total_milk soy_milk regular_milk : ℚ)

-- Conditions
axiom cond1 : total_milk = 0.6
axiom cond2 : soy_milk = 0.1
axiom cond3 : regular_milk + soy_milk = total_milk

-- Theorem statement
theorem mitch_family_milk : regular_milk = 0.5 :=
by
  sorry

end mitch_family_milk_l183_183120


namespace adult_ticket_cost_l183_183521

-- Definitions based on the conditions
def num_adults : ℕ := 10
def num_children : ℕ := 11
def total_bill : ℝ := 124
def child_ticket_cost : ℝ := 4

-- The proof which determines the cost of one adult ticket
theorem adult_ticket_cost : ∃ (A : ℝ), A * num_adults = total_bill - (num_children * child_ticket_cost) ∧ A = 8 := 
by
  sorry

end adult_ticket_cost_l183_183521


namespace hyperbola_eccentricity_l183_183427

variables (a b c e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
          (c_eq : c = 4) (b_eq : b = 2 * Real.sqrt 3)
          (hyperbola_eq : c ^ 2 = a ^ 2 + b ^ 2)
          (projection_cond : 2 < (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ∧ (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ≤ 4)

theorem hyperbola_eccentricity : e = c / a := 
by
  sorry

end hyperbola_eccentricity_l183_183427


namespace volume_ratio_l183_183627

noncomputable def salinity_bay (salt_bay volume_bay : ℝ) : ℝ :=
  salt_bay / volume_bay

noncomputable def salinity_sea_excluding_bay (salt_sea_excluding_bay volume_sea_excluding_bay : ℝ) : ℝ :=
  salt_sea_excluding_bay / volume_sea_excluding_bay

noncomputable def salinity_whole_sea (salt_sea volume_sea : ℝ) : ℝ :=
  salt_sea / volume_sea

theorem volume_ratio (salt_bay volume_bay salt_sea_excluding_bay volume_sea_excluding_bay : ℝ) 
  (h_bay : salinity_bay salt_bay volume_bay = 240 / 1000)
  (h_sea_excluding_bay : salinity_sea_excluding_bay salt_sea_excluding_bay volume_sea_excluding_bay = 110 / 1000)
  (h_whole_sea : salinity_whole_sea (salt_bay + salt_sea_excluding_bay) (volume_bay + volume_sea_excluding_bay) = 120 / 1000) :
  (volume_bay + volume_sea_excluding_bay) / volume_bay = 13 := 
sorry

end volume_ratio_l183_183627


namespace lines_non_intersect_l183_183147

theorem lines_non_intersect (k : ℝ) : 
  (¬∃ t s : ℝ, (1 + 2 * t = -1 + 3 * s ∧ 3 - 5 * t = 4 + k * s)) → 
  k = -15 / 2 :=
by
  intro h
  -- Now left to define proving steps using sorry
  sorry

end lines_non_intersect_l183_183147


namespace incorrect_statement_D_l183_183947

theorem incorrect_statement_D (k b x : ℝ) (hk : k < 0) (hb : b > 0) (hx : x > -b / k) :
  k * x + b ≤ 0 :=
by
  sorry

end incorrect_statement_D_l183_183947


namespace probability_of_consonant_initials_is_10_over_13_l183_183753

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U' ∨ c = 'Y'

def is_consonant (c : Char) : Prop :=
  ¬(is_vowel c) ∧ c ≠ 'W' 

noncomputable def probability_of_consonant_initials : ℚ :=
  let total_letters := 26
  let number_of_vowels := 6
  let number_of_consonants := total_letters - number_of_vowels
  number_of_consonants / total_letters

theorem probability_of_consonant_initials_is_10_over_13 :
  probability_of_consonant_initials = 10 / 13 :=
by
  sorry

end probability_of_consonant_initials_is_10_over_13_l183_183753


namespace income_increase_percentage_l183_183441

theorem income_increase_percentage (I : ℝ) (P : ℝ) (h1 : 0 < I)
  (h2 : 0 ≤ P) (h3 : 0.75 * I + 0.075 * I = 0.825 * I) 
  (h4 : 1.5 * (0.25 * I) = ((I * (1 + P / 100)) - 0.825 * I)) 
  : P = 20 := by
sorry

end income_increase_percentage_l183_183441


namespace canoe_kayak_problem_l183_183181

theorem canoe_kayak_problem (C K : ℕ) 
  (h1 : 9 * C + 12 * K = 432)
  (h2 : C = (4 * K) / 3) : 
  C - K = 6 := by
sorry

end canoe_kayak_problem_l183_183181


namespace max_value_of_y_l183_183178

open Real

noncomputable def y (x : ℝ) : ℝ := 
  (sin (π / 4 + x) - sin (π / 4 - x)) * sin (π / 3 + x)

theorem max_value_of_y : 
  ∃ x : ℝ, (∀ x, y x ≤ 3 * sqrt 2 / 4) ∧ (∀ k : ℤ, x = k * π + π / 3 → y x = 3 * sqrt 2 / 4) :=
sorry

end max_value_of_y_l183_183178


namespace buildings_subset_count_l183_183252

theorem buildings_subset_count :
  let buildings := Finset.range (16 + 1) \ {0}
  ∃ S ⊆ buildings, ∀ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S → ∃ k, (b - a = 2 * k + 1) ∨ (a - b = 2 * k + 1) ∧ Finset.card S = 510 :=
sorry

end buildings_subset_count_l183_183252


namespace cyclic_quadrilaterals_count_l183_183331

noncomputable def num_cyclic_quadrilaterals (n : ℕ) : ℕ :=
  if n = 32 then 568 else 0 -- encapsulating the problem's answer

theorem cyclic_quadrilaterals_count :
  num_cyclic_quadrilaterals 32 = 568 :=
sorry

end cyclic_quadrilaterals_count_l183_183331


namespace sum_first_8_terms_l183_183782

variable {α : Type*} [LinearOrderedField α]

-- Define the arithmetic sequence
def arithmetic_sequence (a_1 d : α) (n : ℕ) : α := a_1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a_1 d : α) (n : ℕ) : α :=
  (n * (2 * a_1 + (n - 1) * d)) / 2

-- Define the given condition
variable (a_1 d : α)
variable (h : arithmetic_sequence a_1 d 3 = 20 - arithmetic_sequence a_1 d 6)

-- Statement of the problem
theorem sum_first_8_terms : sum_arithmetic_sequence a_1 d 8 = 80 :=
by
  sorry

end sum_first_8_terms_l183_183782


namespace equation_has_real_roots_l183_183596

theorem equation_has_real_roots (a b : ℝ) (h : ¬ (a = 0 ∧ b = 0)) :
  ∃ x : ℝ, x ≠ 1 ∧ (a^2 / x + b^2 / (x - 1) = 1) :=
by
  sorry

end equation_has_real_roots_l183_183596


namespace selina_sold_shirts_l183_183838

/-- Selina's selling problem -/
theorem selina_sold_shirts :
  let pants_price := 5
  let shorts_price := 3
  let shirts_price := 4
  let num_pants := 3
  let num_shorts := 5
  let remaining_money := 30 + (2 * 10)
  let money_from_pants := num_pants * pants_price
  let money_from_shorts := num_shorts * shorts_price
  let total_money_from_pants_and_shorts := money_from_pants + money_from_shorts
  let total_money_from_shirts := remaining_money - total_money_from_pants_and_shorts
  let num_shirts := total_money_from_shirts / shirts_price
  num_shirts = 5 := by
{
  sorry
}

end selina_sold_shirts_l183_183838


namespace bike_ride_time_l183_183848

theorem bike_ride_time (y : ℚ) : 
  let speed_fast := 25
  let speed_slow := 10
  let total_distance := 170
  let total_time := 10
  (speed_fast * y + speed_slow * (total_time - y) = total_distance) 
  → y = 14 / 3 := 
by 
  sorry

end bike_ride_time_l183_183848


namespace problem_1_problem_2_l183_183773

noncomputable def a : ℝ := sorry
def m : ℝ := sorry
def n : ℝ := sorry
def k : ℝ := sorry

theorem problem_1 (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) (h4 : a ≠ 0) : 
  a^(3*m + 2*n - k) = 4 := 
sorry

theorem problem_2 (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) (h4 : a ≠ 0) : 
  k - 3*m - n = 0 := 
sorry

end problem_1_problem_2_l183_183773


namespace base6_addition_l183_183216

/-- Adding two numbers in base 6 -/
theorem base6_addition : (3454 : ℕ) + (12345 : ℕ) = (142042 : ℕ) := by
  sorry

end base6_addition_l183_183216


namespace work_completion_time_l183_183183

noncomputable def work_done_by_woman_per_day : ℝ := 1 / 50
noncomputable def work_done_by_child_per_day : ℝ := 1 / 100
noncomputable def total_work_done_by_5_women_per_day : ℝ := 5 * work_done_by_woman_per_day
noncomputable def total_work_done_by_10_children_per_day : ℝ := 10 * work_done_by_child_per_day
noncomputable def combined_work_per_day : ℝ := total_work_done_by_5_women_per_day + total_work_done_by_10_children_per_day

theorem work_completion_time (h1 : 10 / 5 = 2) (h2 : 10 / 10 = 1) :
  1 / combined_work_per_day = 5 :=
by
  sorry

end work_completion_time_l183_183183


namespace probability_at_least_two_same_l183_183307

theorem probability_at_least_two_same :
  let total_outcomes := (8 ^ 4 : ℕ)
  let num_diff_outcomes := (8 * 7 * 6 * 5 : ℕ)
  let probability_diff := (num_diff_outcomes : ℝ) / total_outcomes
  let probability_at_least_two := 1 - probability_diff
  probability_at_least_two = (151 : ℝ) / 256 :=
by
  sorry

end probability_at_least_two_same_l183_183307


namespace num_real_solutions_l183_183564

theorem num_real_solutions (x : ℝ) (A B : Set ℝ) (hx : x ∈ A) (hx2 : x^2 ∈ A) :
  A = {0, 1, 2, x} → B = {1, x^2} → A ∪ B = A → 
  ∃! y : ℝ, y = -Real.sqrt 2 ∨ y = Real.sqrt 2 :=
by
  intro hA hB hA_union_B
  sorry

end num_real_solutions_l183_183564


namespace find_number_l183_183233

theorem find_number (x : ℝ) : 50 + (x * 12) / (180 / 3) = 51 ↔ x = 5 := by
  sorry

end find_number_l183_183233


namespace simplify_expression_l183_183527

noncomputable def givenExpression : ℝ := 
  abs (-0.01) ^ 2 - (-5 / 8) ^ 0 - 3 ^ (Real.log 2 / Real.log 3) + 
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 5) + Real.log 5

theorem simplify_expression : givenExpression = -1.9999 := by
  sorry

end simplify_expression_l183_183527


namespace grandpa_age_times_jungmin_age_l183_183944

-- Definitions based on the conditions
def grandpa_age_last_year : ℕ := 71
def jungmin_age_last_year : ℕ := 8
def grandpa_age_this_year : ℕ := grandpa_age_last_year + 1
def jungmin_age_this_year : ℕ := jungmin_age_last_year + 1

-- The statement to prove
theorem grandpa_age_times_jungmin_age :
  grandpa_age_this_year / jungmin_age_this_year = 8 :=
by
  sorry

end grandpa_age_times_jungmin_age_l183_183944


namespace sufficient_drivers_and_correct_time_l183_183449

-- Conditions definitions
def one_way_minutes := 2 * 60 + 40  -- 2 hours 40 minutes in minutes
def round_trip_minutes := 2 * one_way_minutes  -- round trip in minutes
def rest_minutes := 60  -- mandatory rest period in minutes

-- Time checks for drivers
def driver_a_return := 12 * 60 + 40  -- Driver A returns at 12:40 PM in minutes
def driver_a_next_trip := driver_a_return + rest_minutes  -- Driver A's next trip time
def driver_d_departure := 13 * 60 + 5  -- Driver D departs at 13:05 in minutes

-- Verify sufficiency of four drivers and time correctness
theorem sufficient_drivers_and_correct_time : 
  4 = 4 ∧ (driver_a_next_trip + round_trip_minutes = 21 * 60 + 30) :=
by
  -- Explain the reasoning path that leads to this conclusion within this block
  sorry

end sufficient_drivers_and_correct_time_l183_183449


namespace find_x_values_for_inverse_l183_183483

def f (x : ℝ) : ℝ := x^2 - 3 * x - 4

theorem find_x_values_for_inverse :
  ∃ (x : ℝ), (f x = 2 + 2 * Real.sqrt 2 ∨ f x = 2 - 2 * Real.sqrt 2) ∧ f x = x :=
sorry

end find_x_values_for_inverse_l183_183483


namespace power_of_two_divides_sub_one_l183_183393

theorem power_of_two_divides_sub_one (k : ℕ) (h_odd : k % 2 = 1) : ∀ n ≥ 1, 2^(n+2) ∣ k^(2^n) - 1 :=
by
  sorry

end power_of_two_divides_sub_one_l183_183393


namespace max_height_l183_183849

noncomputable def ball_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 50 * t + 15

theorem max_height : ∃ t : ℝ, t < 50 / 4.9 ∧ ball_height t = 142.65 :=
sorry

end max_height_l183_183849


namespace min_square_side_length_l183_183382

theorem min_square_side_length (s : ℝ) (h : s^2 ≥ 625) : s ≥ 25 :=
sorry

end min_square_side_length_l183_183382


namespace valentines_given_l183_183119

-- Let x be the number of boys and y be the number of girls
variables (x y : ℕ)

-- Condition 1: the number of valentines is 28 more than the total number of students.
axiom valentines_eq : x * y = x + y + 28

-- Theorem: Prove that the total number of valentines given is 60.
theorem valentines_given : x * y = 60 :=
by
  sorry

end valentines_given_l183_183119


namespace inequality_proof_l183_183910

theorem inequality_proof (a b x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : x / a < y / b) :
  (1 / 2) * (x / a + y / b) > (x + y) / (a + b) := by
  sorry

end inequality_proof_l183_183910


namespace quadratic_completing_the_square_l183_183902

theorem quadratic_completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 → (x - 2)^2 = 6 :=
by sorry

end quadratic_completing_the_square_l183_183902


namespace percent_in_range_70_to_79_is_correct_l183_183219

-- Define the total number of students.
def total_students : Nat := 8 + 12 + 11 + 5 + 7

-- Define the number of students within the $70\%-79\%$ range.
def students_70_to_79 : Nat := 11

-- Define the percentage of the students within the $70\%-79\%$ range.
def percent_70_to_79 : ℚ := (students_70_to_79 : ℚ) / (total_students : ℚ) * 100

theorem percent_in_range_70_to_79_is_correct : percent_70_to_79 = 25.58 := by
  sorry

end percent_in_range_70_to_79_is_correct_l183_183219


namespace arccos_pi_over_3_l183_183230

theorem arccos_pi_over_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_pi_over_3_l183_183230


namespace carlos_improved_lap_time_l183_183274

-- Define the initial condition using a function to denote time per lap initially
def initial_lap_time : ℕ := (45 * 60) / 15

-- Define the later condition using a function to denote time per lap later on
def current_lap_time : ℕ := (42 * 60) / 18

-- Define the proof that calculates the improvement in seconds
theorem carlos_improved_lap_time : initial_lap_time - current_lap_time = 40 := by
  sorry

end carlos_improved_lap_time_l183_183274


namespace smallest_nonneg_integer_l183_183411

theorem smallest_nonneg_integer (n : ℕ) (h : 0 ≤ n ∧ n < 53) :
  50 * n ≡ 47 [MOD 53] → n = 2 :=
by
  sorry

end smallest_nonneg_integer_l183_183411


namespace number_of_ways_to_arrange_matches_l183_183601

open Nat

theorem number_of_ways_to_arrange_matches :
  (factorial 7) * (2 ^ 3) = 40320 := by
  sorry

end number_of_ways_to_arrange_matches_l183_183601


namespace time_A_problems_60_l183_183866

variable (t : ℕ) -- time in minutes per type B problem

def time_per_A_problem := 2 * t
def time_per_C_problem := t / 2
def total_time_for_A_problems := 20 * time_per_A_problem

theorem time_A_problems_60 (hC : 80 * time_per_C_problem = 60) : total_time_for_A_problems = 60 := by
  sorry

end time_A_problems_60_l183_183866


namespace linear_elimination_l183_183282

theorem linear_elimination (a b : ℤ) (x y : ℤ) :
  (a = 2) ∧ (b = -5) → 
  (a * (5 * x - 2 * y) + b * (2 * x + 3 * y) = 0) → 
  (10 * x - 4 * y + -10 * x - 15 * y = 8 + -45) :=
by
  sorry

end linear_elimination_l183_183282


namespace lesser_fraction_l183_183146

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 :=
sorry

end lesser_fraction_l183_183146


namespace num_true_statements_l183_183685

theorem num_true_statements :
  (∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) ∧
  (∀ x y a, a ≠ 0 → (a^2 * x ≥ a^2 * y → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x ≥ y → x / a^2 ≥ y / a^2)) →
  ((∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) →
   (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y))) :=
sorry

end num_true_statements_l183_183685


namespace remainder_of_83_div_9_l183_183290

theorem remainder_of_83_div_9 : ∃ r : ℕ, 83 = 9 * 9 + r ∧ r = 2 :=
by {
  sorry
}

end remainder_of_83_div_9_l183_183290


namespace rectangle_dimensions_l183_183029

theorem rectangle_dimensions
  (l w : ℕ)
  (h1 : 2 * l + 2 * w = l * w)
  (h2 : w = l - 3) :
  l = 6 ∧ w = 3 :=
by
  sorry

end rectangle_dimensions_l183_183029


namespace percentage_difference_l183_183080

variable (p : ℝ) (j : ℝ) (t : ℝ)

def condition_1 := j = 0.75 * p
def condition_2 := t = 0.9375 * p

theorem percentage_difference : (j = 0.75 * p) → (t = 0.9375 * p) → ((t - j) / t * 100 = 20) :=
by
  intros h1 h2
  rw [h1, h2]
  -- This will use the derived steps from the solution, and ultimately show 20
  sorry

end percentage_difference_l183_183080


namespace find_number_l183_183025

theorem find_number (x : ℝ) (h : 0.8 * x = (2/5 : ℝ) * 25 + 22) : x = 40 :=
by
  sorry

end find_number_l183_183025


namespace degrees_for_combined_research_l183_183761

-- Define the conditions as constants.
def microphotonics_percentage : ℝ := 0.10
def home_electronics_percentage : ℝ := 0.24
def food_additives_percentage : ℝ := 0.15
def gmo_percentage : ℝ := 0.29
def industrial_lubricants_percentage : ℝ := 0.08
def nanotechnology_percentage : ℝ := 0.07

noncomputable def remaining_percentage : ℝ :=
  1 - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage +
    gmo_percentage + industrial_lubricants_percentage + nanotechnology_percentage)

noncomputable def total_percentage : ℝ :=
  remaining_percentage + nanotechnology_percentage

noncomputable def degrees_in_circle : ℝ := 360

noncomputable def degrees_representing_combined_research : ℝ :=
  total_percentage * degrees_in_circle

-- State the theorem to prove the correct answer
theorem degrees_for_combined_research : degrees_representing_combined_research = 50.4 :=
by
  -- Proof will go here
  sorry

end degrees_for_combined_research_l183_183761


namespace initial_population_l183_183174

theorem initial_population (P : ℝ) 
  (h1 : P * 0.90 * 0.95 * 0.85 * 1.08 = 6514) : P = 8300 :=
by
  -- Given conditions lead to the final population being 6514
  -- We need to show that the initial population P was 8300
  sorry

end initial_population_l183_183174


namespace find_m_l183_183333

-- Definitions for the given vectors
def a : ℝ × ℝ := (3, 4)
def b (m : ℝ) : ℝ × ℝ := (-1, 2 * m)
def c (m : ℝ) : ℝ × ℝ := (m, -4)

-- Definition of vector addition
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Definition of dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition that c is perpendicular to a + b
def perpendicular_condition (m : ℝ) : Prop :=
  dot_product (c m) (vector_add a (b m)) = 0

-- Proof statement
theorem find_m : ∃ m : ℝ, perpendicular_condition m ∧ m = -8 / 3 :=
sorry

end find_m_l183_183333


namespace max_n_for_positive_sum_l183_183737

-- Define the arithmetic sequence \(a_n\)
def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + n * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (a d : ℤ) (n : ℕ) := n * (2 * a + (n-1) * d) / 2

theorem max_n_for_positive_sum 
  (a : ℤ) 
  (d : ℤ) 
  (h_max_sum : ∃ m : ℕ, S_n a d m = S_n a d (m+1))
  (h_ratio : (arithmetic_sequence a d 15) / (arithmetic_sequence a d 14) < -1) :
  27 = 27 :=
sorry

end max_n_for_positive_sum_l183_183737


namespace region_area_l183_183377

theorem region_area : 
  (∃ (x y : ℝ), abs (4 * x - 16) + abs (3 * y + 9) ≤ 6) →
  (∀ (A : ℝ), (∀ x y : ℝ, abs (4 * x - 16) + abs (3 * y + 9) ≤ 6 → 0 ≤ A ∧ A = 6)) :=
by
  intro h exist_condtion
  sorry

end region_area_l183_183377


namespace equal_area_split_l183_183613

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circle1 : Circle := { center := (10, 90), radius := 4 }
def circle2 : Circle := { center := (15, 80), radius := 4 }
def circle3 : Circle := { center := (20, 85), radius := 4 }

theorem equal_area_split :
  ∃ m : ℝ, ∀ x y : ℝ, m * (x - 15) = y - 80 ∧ m = 0 ∧   
    ∀ circle : Circle, circle ∈ [circle1, circle2, circle3] →
      ∃ k : ℝ, k * (x - circle.center.1) + y - circle.center.2 = 0 :=
sorry

end equal_area_split_l183_183613


namespace percentage_blue_shirts_l183_183622

theorem percentage_blue_shirts (total_students := 600) 
 (percent_red := 23)
 (percent_green := 15)
 (students_other := 102)
 : (100 - (percent_red + percent_green + (students_other / total_students) * 100)) = 45 := by
  sorry

end percentage_blue_shirts_l183_183622


namespace neg_of_exists_a_l183_183208

theorem neg_of_exists_a (a : ℝ) : ¬ (∃ a : ℝ, a^2 + 1 < 2 * a) :=
by
  sorry

end neg_of_exists_a_l183_183208


namespace fourth_graders_bought_more_markers_l183_183989

-- Define the conditions
def cost_per_marker : ℕ := 20
def total_payment_fifth_graders : ℕ := 180
def total_payment_fourth_graders : ℕ := 200

-- Compute the number of markers bought by fifth and fourth graders
def markers_bought_by_fifth_graders : ℕ := total_payment_fifth_graders / cost_per_marker
def markers_bought_by_fourth_graders : ℕ := total_payment_fourth_graders / cost_per_marker

-- Statement to prove
theorem fourth_graders_bought_more_markers : 
  markers_bought_by_fourth_graders - markers_bought_by_fifth_graders = 1 := by
  sorry

end fourth_graders_bought_more_markers_l183_183989


namespace hypotenuse_length_l183_183276

variable (a b c : ℝ)

-- Given conditions
theorem hypotenuse_length (h1 : b = 3 * a) 
                          (h2 : a^2 + b^2 + c^2 = 500) 
                          (h3 : c^2 = a^2 + b^2) : 
                          c = 5 * Real.sqrt 10 := 
by 
  sorry

end hypotenuse_length_l183_183276


namespace ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l183_183957

theorem ratio_of_area_to_square_of_perimeter_of_equilateral_triangle :
  let a := 10
  let area := (10 * 10 * (Real.sqrt 3) / 4)
  let perimeter := 3 * 10
  let square_of_perimeter := perimeter * perimeter
  (area / square_of_perimeter) = (Real.sqrt 3 / 36) := by
  -- Proof to be completed
  sorry

end ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l183_183957


namespace find_number_of_moles_of_CaCO3_formed_l183_183529

-- Define the molar ratios and the given condition in structures.
structure Reaction :=
  (moles_CaOH2 : ℕ)
  (moles_CO2 : ℕ)
  (moles_CaCO3 : ℕ)

-- Define a balanced reaction for Ca(OH)2 + CO2 -> CaCO3 + H2O with 1:1 molar ratio.
def balanced_reaction (r : Reaction) : Prop :=
  r.moles_CaOH2 = r.moles_CO2 ∧ r.moles_CaCO3 = r.moles_CO2

-- Define the given condition, which is we have 3 moles of CO2 and formed 3 moles of CaCO3.
def given_condition : Reaction :=
  { moles_CaOH2 := 3, moles_CO2 := 3, moles_CaCO3 := 3 }

-- Theorem: Given 3 moles of CO2, we need to prove 3 moles of CaCO3 are formed based on the balanced reaction.
theorem find_number_of_moles_of_CaCO3_formed :
  balanced_reaction given_condition :=
by {
  -- This part will contain the proof when implemented.
  sorry
}

end find_number_of_moles_of_CaCO3_formed_l183_183529


namespace number_of_exchanges_l183_183807

theorem number_of_exchanges (n : ℕ) (hz_initial : ℕ) (hl_initial : ℕ) 
  (hz_decrease : ℕ) (hl_decrease : ℕ) (k : ℕ) :
  hz_initial = 200 →
  hl_initial = 20 →
  hz_decrease = 6 →
  hl_decrease = 1 →
  k = 11 →
  (hz_initial - n * hz_decrease) = k * (hl_initial - n * hl_decrease) →
  n = 4 := 
sorry

end number_of_exchanges_l183_183807


namespace find_n_value_l183_183474

theorem find_n_value (n : ℤ) : (5^3 - 7 = 6^2 + n) ↔ (n = 82) :=
by
  sorry

end find_n_value_l183_183474


namespace part1_part2_l183_183277

-- Definition of sets A, B, and Proposition p for Part 1
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a = 0}
def p (a : ℝ) : Prop := ∀ x ∈ B a, x ∈ A

-- Part 1: Prove the range of a
theorem part1 (a : ℝ) : (p a) → 0 < a ∧ a ≤ 1 :=
  by sorry

-- Definition of sets A and C for Part 2
def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 3 > 0}
def necessary_condition (m : ℝ) : Prop := ∀ x ∈ A, x ∈ C m

-- Part 2: Prove the range of m
theorem part2 (m : ℝ) : necessary_condition m → m ≤ 7 / 2 :=
  by sorry

end part1_part2_l183_183277


namespace total_number_of_applications_l183_183195

def in_state_apps := 200
def out_state_apps := 2 * in_state_apps
def total_apps := in_state_apps + out_state_apps

theorem total_number_of_applications : total_apps = 600 := by
  sorry

end total_number_of_applications_l183_183195


namespace ursula_hourly_wage_l183_183371

def annual_salary : ℝ := 16320
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

theorem ursula_hourly_wage : 
  (annual_salary / months_per_year) / (hours_per_day * days_per_month) = 8.50 := by 
  sorry

end ursula_hourly_wage_l183_183371


namespace average_running_time_l183_183975

variable (s : ℕ) -- Number of seventh graders

-- let sixth graders run 20 minutes per day
-- let seventh graders run 18 minutes per day
-- let eighth graders run 15 minutes per day
-- sixth graders = 3 * seventh graders
-- eighth graders = 2 * seventh graders

def sixthGradersRunningTime : ℕ := 20 * (3 * s)
def seventhGradersRunningTime : ℕ := 18 * s
def eighthGradersRunningTime : ℕ := 15 * (2 * s)

def totalRunningTime : ℕ := sixthGradersRunningTime s + seventhGradersRunningTime s + eighthGradersRunningTime s
def totalStudents : ℕ := 3 * s + s + 2 * s

theorem average_running_time : totalRunningTime s / totalStudents s = 18 :=
by sorry

end average_running_time_l183_183975


namespace crop_yield_growth_l183_183205

-- Definitions based on conditions
def initial_yield := 300
def final_yield := 363
def eqn (x : ℝ) : Prop := initial_yield * (1 + x)^2 = final_yield

-- The theorem we need to prove
theorem crop_yield_growth (x : ℝ) : eqn x :=
by
  sorry

end crop_yield_growth_l183_183205


namespace edward_final_money_l183_183129

theorem edward_final_money 
  (spring_earnings : ℕ)
  (summer_earnings : ℕ)
  (supplies_cost : ℕ)
  (h_spring : spring_earnings = 2)
  (h_summer : summer_earnings = 27)
  (h_supplies : supplies_cost = 5)
  : spring_earnings + summer_earnings - supplies_cost = 24 := 
sorry

end edward_final_money_l183_183129


namespace odd_function_decreasing_function_max_min_values_on_interval_l183_183790

variable (f : ℝ → ℝ)

axiom func_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom func_negative_for_positive : ∀ x : ℝ, (0 < x) → f x < 0
axiom func_value_at_one : f 1 = -2

theorem odd_function : ∀ x : ℝ, f (-x) = -f x := by
  have f_zero : f 0 = 0 := by sorry
  sorry

theorem decreasing_function : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → f x₁ > f x₂ := by sorry

theorem max_min_values_on_interval :
  (f (-3) = 6) ∧ (f 3 = -6) := by sorry

end odd_function_decreasing_function_max_min_values_on_interval_l183_183790


namespace income_final_amount_l183_183497

noncomputable def final_amount (income : ℕ) : ℕ :=
  let children_distribution := (income * 45) / 100
  let wife_deposit := (income * 30) / 100
  let remaining_after_distribution := income - children_distribution - wife_deposit
  let donation := (remaining_after_distribution * 5) / 100
  remaining_after_distribution - donation

theorem income_final_amount : final_amount 200000 = 47500 := by
  -- Proof omitted
  sorry

end income_final_amount_l183_183497


namespace plane_equation_correct_l183_183831

def plane_equation (x y z : ℝ) : ℝ := 10 * x - 5 * y + 4 * z - 141

noncomputable def gcd (a b c d : ℤ) : ℤ := Int.gcd (Int.gcd a b) (Int.gcd c d)

theorem plane_equation_correct :
  (∀ x y z, plane_equation x y z = 0 ↔ 10 * x - 5 * y + 4 * z - 141 = 0)
  ∧ gcd 10 (-5) 4 (-141) = 1
  ∧ 10 > 0 := by
  sorry

end plane_equation_correct_l183_183831


namespace minimum_value_inequality_l183_183096

theorem minimum_value_inequality {a b : ℝ} (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * Real.log b / Real.log a + 2 * Real.log a / Real.log b = 7) :
  a^2 + 3 / (b - 1) ≥ 2 * Real.sqrt 3 + 1 :=
sorry

end minimum_value_inequality_l183_183096


namespace wrapping_paper_area_l183_183865

variable (l w h : ℝ)
variable (l_gt_w : l > w)

def area_wrapping_paper (l w h : ℝ) : ℝ :=
  3 * (l + w) * h

theorem wrapping_paper_area :
  area_wrapping_paper l w h = 3 * (l + w) * h :=
sorry

end wrapping_paper_area_l183_183865


namespace valid_sequences_length_21_l183_183221

def valid_sequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else valid_sequences (n - 3) + valid_sequences (n - 4)

theorem valid_sequences_length_21 : valid_sequences 21 = 38 :=
by
  sorry

end valid_sequences_length_21_l183_183221


namespace beads_to_remove_l183_183858

-- Definitions for the conditions given in the problem
def initial_blue_beads : Nat := 49
def initial_red_bead : Nat := 1
def total_initial_beads : Nat := initial_blue_beads + initial_red_bead
def target_blue_percentage : Nat := 90 -- percentage

-- The goal to prove
theorem beads_to_remove (initial_blue_beads : Nat) (initial_red_bead : Nat)
    (target_blue_percentage : Nat) : Nat :=
    let target_total_beads := (initial_red_bead * 100) / target_blue_percentage
    total_initial_beads - target_total_beads
-- Expected: beads_to_remove 49 1 90 = 40

example : beads_to_remove initial_blue_beads initial_red_bead target_blue_percentage = 40 := by 
    sorry

end beads_to_remove_l183_183858


namespace k_value_and_set_exists_l183_183251

theorem k_value_and_set_exists
  (x1 x2 x3 x4 : ℚ)
  (h1 : (x1 + x2) / (x3 + x4) = -1)
  (h2 : (x1 + x3) / (x2 + x4) = -1)
  (h3 : (x1 + x4) / (x2 + x3) = -1)
  (hne : x1 ≠ x2 ∨ x1 ≠ x3 ∨ x1 ≠ x4 ∨ x2 ≠ x3 ∨ x2 ≠ x4 ∨ x3 ≠ x4) :
  ∃ (A B C : ℚ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ x1 = A ∧ x2 = B ∧ x3 = C ∧ x4 = -A - B - C := 
sorry

end k_value_and_set_exists_l183_183251


namespace science_homework_is_50_minutes_l183_183532

-- Define the times for each homework and project in minutes
def total_time : ℕ := 3 * 60  -- 3 hours converted to minutes
def math_homework : ℕ := 45
def english_homework : ℕ := 30
def history_homework : ℕ := 25
def special_project : ℕ := 30

-- Define a function to compute the time for science homework
def science_homework_time 
  (total_time : ℕ) 
  (math_time : ℕ) 
  (english_time : ℕ) 
  (history_time : ℕ) 
  (project_time : ℕ) : ℕ :=
  total_time - (math_time + english_time + history_time + project_time)

-- The theorem to prove the time Porche's science homework takes
theorem science_homework_is_50_minutes : 
  science_homework_time total_time math_homework english_homework history_homework special_project = 50 := 
sorry

end science_homework_is_50_minutes_l183_183532


namespace flag_distance_false_l183_183103

theorem flag_distance_false (track_length : ℕ) (num_flags : ℕ) (flag1_flagN : 2 ≤ num_flags)
  (h1 : track_length = 90) (h2 : num_flags = 10) :
  ¬ (track_length / (num_flags - 1) = 9) :=
by
  sorry

end flag_distance_false_l183_183103


namespace relative_speed_of_trains_l183_183953

def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600

theorem relative_speed_of_trains 
  (speed_train1_kmph : ℕ) 
  (speed_train2_kmph : ℕ) 
  (h1 : speed_train1_kmph = 216) 
  (h2 : speed_train2_kmph = 180) : 
  kmph_to_mps speed_train1_kmph - kmph_to_mps speed_train2_kmph = 10 := 
by 
  sorry

end relative_speed_of_trains_l183_183953


namespace cyclist_speed_l183_183580

theorem cyclist_speed (v : ℝ) (h : 0.7142857142857143 * (30 + v) = 50) : v = 40 :=
by
  sorry

end cyclist_speed_l183_183580


namespace most_economical_is_small_l183_183587

noncomputable def most_economical_size (c_S q_S c_M q_M c_L q_L : ℝ) :=
  c_M = 1.3 * c_S ∧
  q_M = 0.85 * q_L ∧
  q_L = 1.5 * q_S ∧
  c_L = 1.4 * c_M →
  (c_S / q_S < c_M / q_M) ∧ (c_S / q_S < c_L / q_L)

theorem most_economical_is_small (c_S q_S c_M q_M c_L q_L : ℝ) :
  most_economical_size c_S q_S c_M q_M c_L q_L := by 
  sorry

end most_economical_is_small_l183_183587


namespace sin_transformation_l183_183018

theorem sin_transformation (α : ℝ) (h : Real.sin (3 * Real.pi / 2 + α) = 3 / 5) :
  Real.sin (Real.pi / 2 + 2 * α) = -7 / 25 :=
by
  sorry

end sin_transformation_l183_183018


namespace min_copy_paste_actions_l183_183691

theorem min_copy_paste_actions :
  ∀ (n : ℕ), (n ≥ 10) ∧ (n ≤ n) → (2^n ≥ 1001) :=
by sorry

end min_copy_paste_actions_l183_183691


namespace part1_part2_l183_183713

-- Part (1) statement
theorem part1 {x : ℝ} : (|x - 1| + |x + 2| >= 5) ↔ (x <= -3 ∨ x >= 2) := 
sorry

-- Part (2) statement
theorem part2 (a : ℝ) : (∀ x : ℝ, (|a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3)) → a = -3 :=
sorry

end part1_part2_l183_183713


namespace correct_result_l183_183641

theorem correct_result (x : ℕ) (h : x + 65 = 125) : x + 95 = 155 :=
sorry

end correct_result_l183_183641


namespace sin_law_ratio_l183_183804

theorem sin_law_ratio {A B C : ℝ} {a b c : ℝ} (hA : a = 1) (hSinA : Real.sin A = 1 / 3) :
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 3 := 
  sorry

end sin_law_ratio_l183_183804


namespace find_y1_l183_183432

noncomputable def y1_proof : Prop :=
∃ (y1 y2 y3 : ℝ), 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1 ∧
(1 - y1)^2 + (y1 - y2)^2 + (y2 - y3)^2 + y3^2 = 1 / 9 ∧
y1 = 1 / 2

-- Statement to be proven:
theorem find_y1 : y1_proof :=
sorry

end find_y1_l183_183432


namespace fraction_equal_l183_183765

theorem fraction_equal {a b x : ℝ} (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + b) / (a - b) = (x + 1) / (x - 1) := 
by
  sorry

end fraction_equal_l183_183765


namespace pigeons_percentage_l183_183654

theorem pigeons_percentage (total_birds pigeons sparrows crows doves non_sparrows : ℕ)
  (h_total : total_birds = 100)
  (h_pigeons : pigeons = 40)
  (h_sparrows : sparrows = 20)
  (h_crows : crows = 15)
  (h_doves : doves = 25)
  (h_non_sparrows : non_sparrows = total_birds - sparrows) :
  (pigeons / non_sparrows : ℚ) * 100 = 50 :=
sorry

end pigeons_percentage_l183_183654


namespace find_a_plus_b_l183_183922

noncomputable def A : ℝ := 3
noncomputable def B : ℝ := -1

noncomputable def l : ℝ := -1 -- Slope of line l (since angle is 3π/4)

noncomputable def l1_slope : ℝ := 1 -- Slope of line l1 which is perpendicular to l

noncomputable def a : ℝ := 0 -- Calculated from k_{AB} = 1

noncomputable def b : ℝ := -2 -- Calculated from line parallel condition

theorem find_a_plus_b : a + b = -2 :=
by
  sorry

end find_a_plus_b_l183_183922


namespace num_distinct_remainders_of_prime_squared_mod_120_l183_183419

theorem num_distinct_remainders_of_prime_squared_mod_120:
  ∀ p : ℕ, Prime p → p > 5 → (p^2 % 120 = 1 ∨ p^2 % 120 = 49) := 
sorry

end num_distinct_remainders_of_prime_squared_mod_120_l183_183419


namespace inequality_of_cubic_powers_l183_183874

theorem inequality_of_cubic_powers 
  (a b: ℝ) (h : a ≠ 0 ∧ b ≠ 0) 
  (h_cond : a * |a| > b * |b|) : 
  a^3 > b^3 := by
  sorry

end inequality_of_cubic_powers_l183_183874


namespace triangle_angle_measure_l183_183549

theorem triangle_angle_measure
  (D E F : ℝ)
  (hD : D = 70)
  (hE : E = 2 * F + 18)
  (h_sum : D + E + F = 180) :
  F = 92 / 3 :=
by
  sorry

end triangle_angle_measure_l183_183549


namespace find_c_minus_2d_l183_183705

theorem find_c_minus_2d :
  ∃ (c d : ℕ), (c > d) ∧ (c - 2 * d = 0) ∧ (∀ x : ℕ, (x^2 - 18 * x + 72 = (x - c) * (x - d))) :=
by
  sorry

end find_c_minus_2d_l183_183705


namespace average_speed_l183_183193

theorem average_speed (v1 v2 t1 t2 total_time total_distance : ℝ)
  (h1 : v1 = 50)
  (h2 : t1 = 4)
  (h3 : v2 = 80)
  (h4 : t2 = 4)
  (h5 : total_time = t1 + t2)
  (h6 : total_distance = v1 * t1 + v2 * t2) :
  (total_distance / total_time = 65) :=
by
  sorry

end average_speed_l183_183193


namespace part1_part2_l183_183167

noncomputable def f (x : ℝ) : ℝ := |x| + |x + 1|

theorem part1 (x : ℝ) : f x > 3 ↔ x > 1 ∨ x < -2 :=
by
  sorry

theorem part2 (m : ℝ) (hx : ∀ x : ℝ, m^2 + 3 * m + 2 * f x ≥ 0) : m ≤ -2 ∨ m ≥ -1 :=
by
  sorry

end part1_part2_l183_183167


namespace Polynomial_has_root_l183_183862

noncomputable def P : ℝ → ℝ := sorry

variables (a1 a2 a3 b1 b2 b3 : ℝ)

axiom h1 : a1 * a2 * a3 ≠ 0
axiom h2 : ∀ x : ℝ, P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)

theorem Polynomial_has_root : ∃ x : ℝ, P x = 0 :=
sorry

end Polynomial_has_root_l183_183862


namespace max_det_value_l183_183972

theorem max_det_value :
  ∃ θ : ℝ, 
    (1 * ((5 + Real.sin θ) * 9 - 6 * 8) 
     - 2 * (4 * 9 - 6 * (7 + Real.cos θ)) 
     + 3 * (4 * 8 - (5 + Real.sin θ) * (7 + Real.cos θ))) 
     = 93 :=
sorry

end max_det_value_l183_183972


namespace Ada_initial_seat_l183_183950

-- We have 6 seats
def Seats := Fin 6

-- Friends' movements expressed in terms of seat positions changes
variable (Bea Ceci Dee Edie Fred Ada : Seats)

-- Conditions about the movements
variable (beMovedRight : Bea.val + 1 = Ada.val)
variable (ceMovedLeft : Ceci.val = Ada.val + 2)
variable (deeMovedRight : Dee.val + 1 = Ada.val)
variable (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
  edie_new = Fred ∧ fred_new = Edie)

-- Ada returns to an end seat (1 or 6)
axiom adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩

-- Theorem to prove Ada's initial position
theorem Ada_initial_seat (Bea Ceci Dee Edie Fred Ada : Seats)
  (beMovedRight : Bea.val + 1 = Ada.val)
  (ceMovedLeft : Ceci.val = Ada.val + 2)
  (deeMovedRight : Dee.val + 1 = Ada.val)
  (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
    edie_new = Fred ∧ fred_new = Edie)
  (adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩) :
  Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩ := sorry

end Ada_initial_seat_l183_183950


namespace employee_overtime_hours_l183_183072

theorem employee_overtime_hours (gross_pay : ℝ) (rate_regular : ℝ) (regular_hours : ℕ) (rate_overtime : ℝ) :
  gross_pay = 622 → rate_regular = 11.25 → regular_hours = 40 → rate_overtime = 16 →
  ∃ (overtime_hours : ℕ), overtime_hours = 10 :=
by
  sorry

end employee_overtime_hours_l183_183072


namespace unitD_questionnaires_l183_183602

theorem unitD_questionnaires :
  ∀ (numA numB numC numD total_drawn : ℕ),
  (2 * numB = numA + numC) →  -- arithmetic sequence condition for B
  (2 * numC = numB + numD) →  -- arithmetic sequence condition for C
  (numA + numB + numC + numD = 1000) →  -- total number condition
  (total_drawn = 150) →  -- total drawn condition
  (numB = 30) →  -- unit B condition
  (total_drawn = (30 - d) + 30 + (30 + d) + (30 + 2 * d)) →
  (d = 15) →
  30 + 2 * d = 60 :=
by
  sorry

end unitD_questionnaires_l183_183602


namespace parallel_line_slope_l183_183085

theorem parallel_line_slope (x y : ℝ) : (∃ (c : ℝ), 3 * x - 6 * y = c) → (1 / 2) = 1 / 2 :=
by sorry

end parallel_line_slope_l183_183085


namespace customer_bought_29_eggs_l183_183742

-- Defining the conditions
def baskets : List ℕ := [4, 6, 12, 13, 22, 29]
def total_eggs : ℕ := 86
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Stating the problem
theorem customer_bought_29_eggs :
  ∃ eggs_in_basket,
    eggs_in_basket ∈ baskets ∧
    is_multiple_of_three (total_eggs - eggs_in_basket) ∧
    eggs_in_basket = 29 :=
by sorry

end customer_bought_29_eggs_l183_183742


namespace field_dimension_solution_l183_183689

theorem field_dimension_solution (m : ℝ) (h₁ : (3 * m + 10) * (m - 5) = 72) : m = 7 :=
sorry

end field_dimension_solution_l183_183689


namespace line_eq_l183_183261

-- Conditions
def circle_eq (x y : ℝ) (a : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 5 - a

def midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  2*xm = x1 + x2 ∧ 2*ym = y1 + y2

-- Theorem statement
theorem line_eq (a : ℝ) (h : a < 3) :
  circle_eq 0 1 a →
  ∃ l : ℝ → ℝ, (∀ x, l x = x - 1) :=
sorry

end line_eq_l183_183261


namespace roberto_outfits_l183_183511

-- Roberto's wardrobe constraints
def num_trousers : ℕ := 5
def num_shirts : ℕ := 6
def num_jackets : ℕ := 4
def num_shoes : ℕ := 3
def restricted_jacket_shoes : ℕ := 2

-- The total number of valid outfits
def total_outfits_with_constraint : ℕ := 330

-- Proving the equivalent of the problem statement
theorem roberto_outfits :
  (num_trousers * num_shirts * (num_jackets - 1) * num_shoes) + (num_trousers * num_shirts * 1 * restricted_jacket_shoes) = total_outfits_with_constraint :=
by
  sorry

end roberto_outfits_l183_183511


namespace inequality_proof_l183_183374

theorem inequality_proof (a b c : ℝ) (hab : a * b < 0) : 
  a^2 + b^2 + c^2 > 2 * a * b + 2 * b * c + 2 * c * a := 
by 
  sorry

end inequality_proof_l183_183374


namespace inequality_abc_l183_183663

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_abc_l183_183663


namespace geometric_sequence_a3a5_l183_183388

theorem geometric_sequence_a3a5 (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 4 = 5) : a 3 * a 5 = 25 :=
by
  sorry

end geometric_sequence_a3a5_l183_183388


namespace find_remainder_l183_183118

variable (x y remainder : ℕ)
variable (h1 : x = 7 * y + 3)
variable (h2 : 2 * x = 18 * y + remainder)
variable (h3 : 11 * y - x = 1)

theorem find_remainder : remainder = 2 := 
by
  sorry

end find_remainder_l183_183118


namespace geometric_seq_min_3b2_7b3_l183_183165

theorem geometric_seq_min_3b2_7b3 (b_1 b_2 b_3 : ℝ) (r : ℝ) 
  (h_seq : b_1 = 2) (h_geom : b_2 = b_1 * r) (h_geom2 : b_3 = b_1 * r^2) :
  3 * b_2 + 7 * b_3 ≥ -16 / 7 :=
by
  -- Include the necessary definitions to support the setup
  have h_b1 : b_1 = 2 := h_seq
  have h_b2 : b_2 = 2 * r := by rw [h_geom, h_b1]
  have h_b3 : b_3 = 2 * r^2 := by rw [h_geom2, h_b1]
  sorry

end geometric_seq_min_3b2_7b3_l183_183165


namespace angle_between_hands_at_seven_l183_183150

-- Define the conditions
def clock_parts := 12 -- The clock is divided into 12 parts
def degrees_per_part := 30 -- Each part is 30 degrees

-- Define the position of the hour and minute hands at 7:00 AM
def hour_position_at_seven := 7 -- Hour hand points to 7
def minute_position_at_seven := 0 -- Minute hand points to 12

-- Calculate the number of parts between the two positions
def parts_between_hands := if minute_position_at_seven = 0 then hour_position_at_seven else 12 - hour_position_at_seven

-- Calculate the angle between the hour hand and the minute hand at 7:00 AM
def angle_at_seven := degrees_per_part * parts_between_hands

-- State the theorem
theorem angle_between_hands_at_seven : angle_at_seven = 150 :=
by
  sorry

end angle_between_hands_at_seven_l183_183150


namespace dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l183_183447

-- Define a regular dodecagon
def dodecagon_sides : ℕ := 12

-- Prove that the number of diagonals in a regular dodecagon is 54
theorem dodecagon_diagonals_eq_54 : (dodecagon_sides * (dodecagon_sides - 3)) / 2 = 54 :=
by sorry

-- Prove that the number of possible triangles formed from a regular dodecagon vertices is 220
theorem dodecagon_triangles_eq_220 : Nat.choose dodecagon_sides 3 = 220 :=
by sorry

end dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l183_183447


namespace matrix_product_is_zero_l183_183560

def vec3 := (ℝ × ℝ × ℝ)

def M1 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((0, 2 * c, -2 * b),
   (-2 * c, 0, 2 * a),
   (2 * b, -2 * a, 0))

def M2 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((2 * a^2, a^2 + b^2, a^2 + c^2),
   (a^2 + b^2, 2 * b^2, b^2 + c^2),
   (a^2 + c^2, b^2 + c^2, 2 * c^2))

def matrix_mul (m1 m2 : vec3 × vec3 × vec3) : vec3 × vec3 × vec3 := sorry

theorem matrix_product_is_zero (a b c : ℝ) :
  matrix_mul (M1 a b c) (M2 a b c) = ((0, 0, 0), (0, 0, 0), (0, 0, 0)) := by
  sorry

end matrix_product_is_zero_l183_183560


namespace general_term_formula_l183_183577

/-- Define that the point (n, S_n) lies on the function y = 2x^2 + x, hence S_n = 2 * n^2 + n --/
def S_n (n : ℕ) : ℕ := 2 * n^2 + n

/-- Define the nth term of the sequence a_n --/
def a_n (n : ℕ) : ℕ := if n = 0 then 0 else 4 * n - 1

theorem general_term_formula (n : ℕ) (hn : 0 < n) :
  a_n n = S_n n - S_n (n - 1) :=
by
  sorry

end general_term_formula_l183_183577


namespace sin_cos_sixth_power_sum_l183_183130

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 0.8125 :=
by
  sorry

end sin_cos_sixth_power_sum_l183_183130


namespace probability_of_consecutive_triplets_l183_183498

def total_ways_to_select_3_days (n : ℕ) : ℕ :=
  Nat.choose n 3

def number_of_consecutive_triplets (n : ℕ) : ℕ :=
  n - 2

theorem probability_of_consecutive_triplets :
  let total_ways := total_ways_to_select_3_days 10
  let consecutive_triplets := number_of_consecutive_triplets 10
  (consecutive_triplets : ℚ) / total_ways = 1 / 15 :=
by
  sorry

end probability_of_consecutive_triplets_l183_183498


namespace value_expression_l183_183149

theorem value_expression (x : ℝ) (h : x = 1 + Real.sqrt 2) : x^4 - 4 * x^3 + 4 * x^2 + 4 = 5 :=
by
  sorry

end value_expression_l183_183149


namespace intersection_point_of_lines_l183_183653

theorem intersection_point_of_lines : ∃ (x y : ℝ), x + y = 5 ∧ x - y = 1 ∧ x = 3 ∧ y = 2 :=
by
  sorry

end intersection_point_of_lines_l183_183653


namespace part1_part2_l183_183438

-- Given Definitions
variable (p : ℕ) [hp : Fact (p > 3)] [prime : Fact (Nat.Prime p)]
variable (A_l : ℕ → ℕ)

-- Assertions to Prove
theorem part1 (l : ℕ) (hl : 1 ≤ l ∧ l ≤ p - 2) : A_l l % p = 0 :=
sorry

theorem part2 (l : ℕ) (hl : 1 < l ∧ l < p ∧ l % 2 = 1) : A_l l % (p * p) = 0 :=
sorry

end part1_part2_l183_183438


namespace quadratic_eq_integer_roots_iff_l183_183715

theorem quadratic_eq_integer_roots_iff (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x * y = n ∧ x + y = 4) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end quadratic_eq_integer_roots_iff_l183_183715


namespace rate_in_still_water_l183_183054

theorem rate_in_still_water (with_stream_speed against_stream_speed : ℕ) 
  (h₁ : with_stream_speed = 16) 
  (h₂ : against_stream_speed = 12) : 
  (with_stream_speed + against_stream_speed) / 2 = 14 := 
by
  sorry

end rate_in_still_water_l183_183054


namespace intersection_product_is_15_l183_183128

-- Define the first circle equation as a predicate
def first_circle (x y : ℝ) : Prop :=
  x^2 - 4 * x + y^2 - 6 * y + 12 = 0

-- Define the second circle equation as a predicate
def second_circle (x y : ℝ) : Prop :=
  x^2 - 10 * x + y^2 - 6 * y + 34 = 0

-- The Lean statement for the proof problem
theorem intersection_product_is_15 :
  ∃ x y : ℝ, first_circle x y ∧ second_circle x y ∧ (x * y = 15) :=
by
  sorry

end intersection_product_is_15_l183_183128


namespace first_player_wins_if_take_one_initial_l183_183896

theorem first_player_wins_if_take_one_initial :
  ∃ strategy : ℕ → ℕ, 
    (∀ n, strategy n = if n % 3 = 0 then 1 else 2) ∧ 
    strategy 99 = 1 ∧ 
    strategy 100 = 1 :=
sorry

end first_player_wins_if_take_one_initial_l183_183896


namespace inverse_proposition_of_square_positive_l183_183841

theorem inverse_proposition_of_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by
  intro h
  intros x h₁
  sorry

end inverse_proposition_of_square_positive_l183_183841


namespace cube_root_of_sum_powers_l183_183651

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem cube_root_of_sum_powers :
  cube_root (2^7 + 2^7 + 2^7) = 4 * cube_root 2 :=
by
  sorry

end cube_root_of_sum_powers_l183_183651


namespace xy_addition_l183_183442

theorem xy_addition (x y : ℕ) (h1 : x * y = 24) (h2 : x - y = 5) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 11 := 
sorry

end xy_addition_l183_183442


namespace water_lilies_half_pond_l183_183690

theorem water_lilies_half_pond (growth_rate : ℕ → ℕ) (start_day : ℕ) (full_covered_day : ℕ) 
  (h_growth : ∀ n, growth_rate (n + 1) = 2 * growth_rate n) 
  (h_start : growth_rate start_day = 1) 
  (h_full_covered : growth_rate full_covered_day = 2^(full_covered_day - start_day)) : 
  growth_rate (full_covered_day - 1) = 2^(full_covered_day - start_day - 1) :=
by
  sorry

end water_lilies_half_pond_l183_183690


namespace sin_B_value_l183_183683

variable {A B C : Real}
variable {a b c : Real}
variable {sin_A sin_B sin_C : Real}

-- Given conditions as hypotheses
axiom h1 : c = 2 * a
axiom h2 : b * sin_B - a * sin_A = (1 / 2) * a * sin_C

-- The statement to prove
theorem sin_B_value : sin_B = Real.sqrt 7 / 4 :=
by
  -- Proof omitted
  sorry

end sin_B_value_l183_183683


namespace asthma_distribution_l183_183460

noncomputable def total_children := 490
noncomputable def boys := 280
noncomputable def general_asthma_ratio := 2 / 7
noncomputable def boys_asthma_ratio := 1 / 9

noncomputable def total_children_with_asthma := general_asthma_ratio * total_children
noncomputable def boys_with_asthma := boys_asthma_ratio * boys
noncomputable def girls_with_asthma := total_children_with_asthma - boys_with_asthma

theorem asthma_distribution
  (h_general_asthma: general_asthma_ratio = 2 / 7)
  (h_total_children: total_children = 490)
  (h_boys: boys = 280)
  (h_boys_asthma: boys_asthma_ratio = 1 / 9):
  boys_with_asthma = 31 ∧ girls_with_asthma = 109 :=
by
  sorry

end asthma_distribution_l183_183460


namespace student_adjustment_l183_183845

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem student_adjustment : 
  let front_row_size := 4
  let back_row_size := 8
  let total_students := 12
  let num_to_select := 2
  let ways_to_select := binomial back_row_size num_to_select
  let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
  ways_to_select * ways_to_permute = 840 :=
  by {
    let front_row_size := 4
    let back_row_size := 8
    let total_students := 12
    let num_to_select := 2
    let ways_to_select := binomial back_row_size num_to_select
    let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
    exact sorry
  }

end student_adjustment_l183_183845


namespace angle_sum_property_l183_183935

theorem angle_sum_property
  (angle1 angle2 angle3 : ℝ) 
  (h1 : angle1 = 58) 
  (h2 : angle2 = 35) 
  (h3 : angle3 = 42) : 
  angle1 + angle2 + angle3 + (180 - (angle1 + angle2 + angle3)) = 180 := 
by 
  sorry

end angle_sum_property_l183_183935


namespace odd_n_cubed_plus_23n_divisibility_l183_183867

theorem odd_n_cubed_plus_23n_divisibility (n : ℤ) (h1 : n % 2 = 1) : (n^3 + 23 * n) % 24 = 0 := 
by 
  sorry

end odd_n_cubed_plus_23n_divisibility_l183_183867


namespace manager_salary_l183_183635

theorem manager_salary
    (average_salary_employees : ℝ)
    (num_employees : ℕ)
    (increase_in_average_due_to_manager : ℝ)
    (total_salary_20_employees : ℝ)
    (new_average_salary : ℝ)
    (total_salary_with_manager : ℝ) :
  average_salary_employees = 1300 →
  num_employees = 20 →
  increase_in_average_due_to_manager = 100 →
  total_salary_20_employees = average_salary_employees * num_employees →
  new_average_salary = average_salary_employees + increase_in_average_due_to_manager →
  total_salary_with_manager = new_average_salary * (num_employees + 1) →
  total_salary_with_manager - total_salary_20_employees = 3400 :=
by 
  sorry

end manager_salary_l183_183635


namespace complex_number_real_implies_m_is_5_l183_183645

theorem complex_number_real_implies_m_is_5 (m : ℝ) (h : m^2 - 2 * m - 15 = 0) : m = 5 :=
  sorry

end complex_number_real_implies_m_is_5_l183_183645


namespace ellipse_condition_sufficient_not_necessary_l183_183444

theorem ellipse_condition_sufficient_not_necessary (n : ℝ) :
  (-1 < n) ∧ (n < 2) → 
  (2 - n > 0) ∧ (n + 1 > 0) ∧ (2 - n > n + 1) :=
by
  intro h
  sorry

end ellipse_condition_sufficient_not_necessary_l183_183444


namespace tiled_floor_area_correct_garden_area_correct_seating_area_correct_l183_183934

noncomputable def length_room : ℝ := 20
noncomputable def width_room : ℝ := 12
noncomputable def width_veranda : ℝ := 2
noncomputable def length_pool : ℝ := 15
noncomputable def width_pool : ℝ := 6

noncomputable def area (length width : ℝ) : ℝ := length * width

noncomputable def area_room : ℝ := area length_room width_room
noncomputable def area_pool : ℝ := area length_pool width_pool
noncomputable def area_tiled_floor : ℝ := area_room - area_pool

noncomputable def total_length : ℝ := length_room + 2 * width_veranda
noncomputable def total_width : ℝ := width_room + 2 * width_veranda
noncomputable def area_total : ℝ := area total_length total_width
noncomputable def area_veranda : ℝ := area_total - area_room
noncomputable def area_garden : ℝ := area_veranda / 2
noncomputable def area_seating : ℝ := area_veranda / 2

theorem tiled_floor_area_correct : area_tiled_floor = 150 := by
  sorry

theorem garden_area_correct : area_garden = 72 := by
  sorry

theorem seating_area_correct : area_seating = 72 := by
  sorry

end tiled_floor_area_correct_garden_area_correct_seating_area_correct_l183_183934


namespace turnip_difference_l183_183186

theorem turnip_difference :
  let melanie_turnips := 139
  let benny_turnips := 113
  let caroline_turnips := 172
  (melanie_turnips + benny_turnips) - caroline_turnips = 80 :=
by
  let melanie_turnips := 139
  let benny_turnips := 113
  let caroline_turnips := 172
  show (melanie_turnips + benny_turnips) - caroline_turnips = 80
  sorry

end turnip_difference_l183_183186


namespace problem_1_problem_2_l183_183820

def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 3)
def g (x : ℝ) : ℝ := abs (2 * x - 3) + 2

theorem problem_1 (x : ℝ) :
  abs (g x) < 5 → 0 < x ∧ x < 3 :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) →
  (a ≥ -1 ∨ a ≤ -5) :=
sorry

end problem_1_problem_2_l183_183820


namespace total_chocolates_distributed_l183_183300

theorem total_chocolates_distributed 
  (boys girls : ℕ)
  (chocolates_per_boy chocolates_per_girl : ℕ)
  (h_boys : boys = 60)
  (h_girls : girls = 60)
  (h_chocolates_per_boy : chocolates_per_boy = 2)
  (h_chocolates_per_girl : chocolates_per_girl = 3) : 
  boys * chocolates_per_boy + girls * chocolates_per_girl = 300 :=
by {
  sorry
}

end total_chocolates_distributed_l183_183300


namespace arithmetic_sequence_26th_term_eq_neg48_l183_183033

def arithmetic_sequence_term (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_26th_term_eq_neg48 : 
  arithmetic_sequence_term 2 (-2) 26 = -48 :=
by
  sorry

end arithmetic_sequence_26th_term_eq_neg48_l183_183033


namespace ral_current_age_l183_183413

-- Definitions according to the conditions
def ral_three_times_suri (ral suri : ℕ) : Prop := ral = 3 * suri
def suri_in_6_years (suri : ℕ) : Prop := suri + 6 = 25

-- The proof problem statement
theorem ral_current_age (ral suri : ℕ) (h1 : ral_three_times_suri ral suri) (h2 : suri_in_6_years suri) : ral = 57 :=
by sorry

end ral_current_age_l183_183413


namespace inequality_sinx_plus_y_cosx_plus_y_l183_183475

open Real

theorem inequality_sinx_plus_y_cosx_plus_y (
  y x : ℝ
) (hx : x ∈ Set.Icc (π / 4) (3 * π / 4)) (hy : y ∈ Set.Icc (π / 4) (3 * π / 4)) :
  sin (x + y) + cos (x + y) ≤ sin x + cos x + sin y + cos y :=
sorry

end inequality_sinx_plus_y_cosx_plus_y_l183_183475


namespace lionel_distance_walked_when_met_l183_183342

theorem lionel_distance_walked_when_met (distance_between : ℕ) (lionel_speed : ℕ) (walt_speed : ℕ) (advance_time : ℕ) 
(h1 : distance_between = 48) 
(h2 : lionel_speed = 2) 
(h3 : walt_speed = 6) 
(h4 : advance_time = 2) : 
  ∃ D : ℕ, D = 15 :=
by
  sorry

end lionel_distance_walked_when_met_l183_183342


namespace solve_for_m_l183_183011

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 2 → - (1/2) * x^2 + 2 * x > m * x) → m = 1 :=
by
  -- Skip the proof by using sorry
  sorry

end solve_for_m_l183_183011


namespace find_multiple_l183_183568

theorem find_multiple (x m : ℝ) (hx : x = 3) (h : x + 17 = m * (1 / x)) : m = 60 := 
by
  sorry

end find_multiple_l183_183568
