import Mathlib

namespace quarters_for_chips_l851_85156

def total_quarters : ℕ := 16
def quarters_for_soda : ℕ := 12

theorem quarters_for_chips : (total_quarters - quarters_for_soda) = 4 :=
  by 
    sorry

end quarters_for_chips_l851_85156


namespace circle_equation_l851_85143

theorem circle_equation (x y : ℝ)
  (h_center : ∀ x y, (x - 3)^2 + (y - 1)^2 = r ^ 2)
  (h_origin : (0 - 3)^2 + (0 - 1)^2 = r ^ 2) :
  (x - 3) ^ 2 + (y - 1) ^ 2 = 10 := by
  sorry

end circle_equation_l851_85143


namespace fraction_division_result_l851_85118

theorem fraction_division_result :
  (5/6) / (-9/10) = -25/27 := 
by
  sorry

end fraction_division_result_l851_85118


namespace find_coordinates_of_b_l851_85137

theorem find_coordinates_of_b
  (x y : ℝ)
  (a : ℂ) (b : ℂ)
  (sqrt3 sqrt5 sqrt10 sqrt6 : ℝ)
  (h1 : sqrt3 = Real.sqrt 3)
  (h2 : sqrt5 = Real.sqrt 5)
  (h3 : sqrt10 = Real.sqrt 10)
  (h4 : sqrt6 = Real.sqrt 6)
  (h5 : a = ⟨sqrt3, sqrt5⟩)
  (h6 : ∃ x y : ℝ, b = ⟨x, y⟩ ∧ (sqrt3 * x + sqrt5 * y = 0) ∧ (Real.sqrt (x^2 + y^2) = 2))
  : b = ⟨- sqrt10 / 2, sqrt6 / 2⟩ ∨ b = ⟨sqrt10 / 2, - sqrt6 / 2⟩ := 
  sorry

end find_coordinates_of_b_l851_85137


namespace aquatic_reserve_total_fishes_l851_85138

-- Define the number of bodies of water
def bodies_of_water : ℕ := 6

-- Define the number of fishes per body of water
def fishes_per_body : ℕ := 175

-- Define the total number of fishes
def total_fishes : ℕ := bodies_of_water * fishes_per_body

theorem aquatic_reserve_total_fishes : bodies_of_water * fishes_per_body = 1050 := by
  -- The proof is omitted.
  sorry

end aquatic_reserve_total_fishes_l851_85138


namespace pen_distribution_l851_85107

theorem pen_distribution:
  (∃ (fountain: ℕ) (ballpoint: ℕ), fountain = 2 ∧ ballpoint = 3) ∧
  (∃ (students: ℕ), students = 4) →
  (∀ (s: ℕ), s ≥ 1 → s ≤ 4) →
  ∃ (ways: ℕ), ways = 28 :=
by
  sorry

end pen_distribution_l851_85107


namespace solve_fractional_equation_l851_85110

-- Define the fractional equation as a function
def fractional_equation (x : ℝ) : Prop :=
  (3 / 2) - (2 * x) / (3 * x - 1) = 7 / (6 * x - 2)

-- State the theorem we need to prove
theorem solve_fractional_equation : fractional_equation 2 :=
by
  -- Placeholder for proof
  sorry

end solve_fractional_equation_l851_85110


namespace jane_buys_four_bagels_l851_85190

-- Define Jane's 7-day breakfast choices
def number_of_items (b m : ℕ) := b + m = 7

-- Define the total weekly cost condition
def total_cost_divisible_by_100 (b : ℕ) := (90 * b + 40 * (7 - b)) % 100 = 0

-- The statement to prove
theorem jane_buys_four_bagels (b : ℕ) (m : ℕ) (h1 : number_of_items b m) (h2 : total_cost_divisible_by_100 b) : b = 4 :=
by
  -- proof goes here
  sorry

end jane_buys_four_bagels_l851_85190


namespace smallest_positive_integer_divisible_by_15_16_18_l851_85195

theorem smallest_positive_integer_divisible_by_15_16_18 : 
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ n = 720 := 
by
  sorry

end smallest_positive_integer_divisible_by_15_16_18_l851_85195


namespace total_candies_correct_l851_85126

-- Define the number of candies each has
def caleb_jellybeans := 3 * 12
def caleb_chocolate_bars := 5
def caleb_gummy_bears := 8
def caleb_total := caleb_jellybeans + caleb_chocolate_bars + caleb_gummy_bears

def sophie_jellybeans := (caleb_jellybeans / 2)
def sophie_chocolate_bars := 3
def sophie_gummy_bears := 12
def sophie_total := sophie_jellybeans + sophie_chocolate_bars + sophie_gummy_bears

def max_jellybeans := (2 * 12) + sophie_jellybeans
def max_chocolate_bars := 6
def max_gummy_bears := 10
def max_total := max_jellybeans + max_chocolate_bars + max_gummy_bears

-- Define the total number of candies
def total_candies := caleb_total + sophie_total + max_total

-- Theorem statement
theorem total_candies_correct : total_candies = 140 := by
  sorry

end total_candies_correct_l851_85126


namespace product_513_12_l851_85181

theorem product_513_12 : 513 * 12 = 6156 := 
  by
    sorry

end product_513_12_l851_85181


namespace domain_of_sqrt_tan_l851_85139

theorem domain_of_sqrt_tan :
  ∀ x : ℝ, (∃ k : ℤ, k * π ≤ x ∧ x < k * π + π / 2) ↔ 0 ≤ (Real.tan x) :=
sorry

end domain_of_sqrt_tan_l851_85139


namespace TotalGenuineItems_l851_85149

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem TotalGenuineItems : GenuinePurses + GenuineHandbags = 31 :=
  by
    -- proof
    sorry

end TotalGenuineItems_l851_85149


namespace sin_double_angle_l851_85114

open Real

theorem sin_double_angle
  {α : ℝ} (h1: tan α = -1/2) (h2: 0 < α ∧ α < π) :
  sin (2 * α) = -4/5 :=
sorry

end sin_double_angle_l851_85114


namespace inequality_of_sum_l851_85116

theorem inequality_of_sum 
  (a : ℕ → ℝ)
  (h : ∀ n m, 0 ≤ n → n < m → a n < a m) :
  (0 < a 1 ->
  0 < a 2 ->
  0 < a 3 ->
  0 < a 4 ->
  0 < a 5 ->
  0 < a 6 ->
  0 < a 7 ->
  0 < a 8 ->
  0 < a 9 ->
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / (a 3 + a 6 + a 9) < 3) :=
by
  intros
  sorry

end inequality_of_sum_l851_85116


namespace total_potatoes_l851_85120

theorem total_potatoes (Nancy_potatoes : ℕ) (Sandy_potatoes : ℕ) (Andy_potatoes : ℕ) 
  (h1 : Nancy_potatoes = 6) (h2 : Sandy_potatoes = 7) (h3 : Andy_potatoes = 9) : 
  Nancy_potatoes + Sandy_potatoes + Andy_potatoes = 22 :=
by
  -- The proof can be written here
  sorry

end total_potatoes_l851_85120


namespace find_radius_l851_85157

noncomputable def radius (π : ℝ) : Prop :=
  ∃ r : ℝ, π * r^2 + 2 * r - 2 * π * r = 12 ∧ r = Real.sqrt (12 / π)

theorem find_radius (π : ℝ) (hπ : π > 0) : 
  radius π :=
sorry

end find_radius_l851_85157


namespace AB_eq_B_exp_V_l851_85197

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l851_85197


namespace alcohol_to_water_ratio_l851_85177

theorem alcohol_to_water_ratio (alcohol water : ℚ) (h_alcohol : alcohol = 2/7) (h_water : water = 3/7) : alcohol / water = 2 / 3 := by
  sorry

end alcohol_to_water_ratio_l851_85177


namespace total_wire_length_l851_85167

theorem total_wire_length (S : ℕ) (L : ℕ)
  (hS : S = 20) 
  (hL : L = 2 * S) : S + L = 60 :=
by
  sorry

end total_wire_length_l851_85167


namespace range_of_k_l851_85134

theorem range_of_k (x k : ℝ):
  (2 * x + 9 > 6 * x + 1) → (x - k < 1) → (x < 2) → k ≥ 1 :=
by 
  sorry

end range_of_k_l851_85134


namespace megan_math_problems_l851_85155

theorem megan_math_problems (num_spelling_problems num_problems_per_hour num_hours total_problems num_math_problems : ℕ) 
  (h1 : num_spelling_problems = 28)
  (h2 : num_problems_per_hour = 8)
  (h3 : num_hours = 8)
  (h4 : total_problems = num_problems_per_hour * num_hours)
  (h5 : total_problems = num_spelling_problems + num_math_problems) :
  num_math_problems = 36 := 
by
  sorry

end megan_math_problems_l851_85155


namespace base_addition_l851_85191

theorem base_addition (R1 R3 : ℕ) (F1 F2 : ℚ)
    (hF1_baseR1 : F1 = 45 / (R1^2 - 1))
    (hF2_baseR1 : F2 = 54 / (R1^2 - 1))
    (hF1_baseR3 : F1 = 36 / (R3^2 - 1))
    (hF2_baseR3 : F2 = 63 / (R3^2 - 1)) :
  R1 + R3 = 20 :=
sorry

end base_addition_l851_85191


namespace cody_initial_tickets_l851_85169

theorem cody_initial_tickets (T : ℕ) (h1 : T - 25 + 6 = 30) : T = 49 :=
sorry

end cody_initial_tickets_l851_85169


namespace solution_set_for_f_ge_0_range_of_a_l851_85194

def f (x : ℝ) : ℝ := |3 * x + 1| - |2 * x + 2|

theorem solution_set_for_f_ge_0 : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3/5} ∪ {x : ℝ | x ≥ 1} :=
sorry

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x - |x + 1| ≤ |a + 1|) ↔ (a ≤ -3 ∨ a ≥ 1) :=
sorry

end solution_set_for_f_ge_0_range_of_a_l851_85194


namespace constant_difference_of_equal_derivatives_l851_85146

theorem constant_difference_of_equal_derivatives
  {f g : ℝ → ℝ}
  (h : ∀ x, deriv f x = deriv g x) :
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end constant_difference_of_equal_derivatives_l851_85146


namespace gcd_fa_fb_l851_85128

def f (x : ℤ) : ℤ := x * x - x + 2008

def a : ℤ := 102
def b : ℤ := 103

theorem gcd_fa_fb : Int.gcd (f a) (f b) = 2 := by
  sorry

end gcd_fa_fb_l851_85128


namespace swimming_pool_width_l851_85127

theorem swimming_pool_width (length width vol depth : ℝ) 
  (H_length : length = 60) 
  (H_depth : depth = 0.5) 
  (H_vol_removal : vol = 2250 / 7.48052) 
  (H_vol_eq : vol = (length * width) * depth) : 
  width = 10.019 :=
by
  -- Assuming the correctness of floating-point arithmetic for the purpose of this example
  sorry

end swimming_pool_width_l851_85127


namespace dog_rabbit_age_ratio_l851_85145

-- Definitions based on conditions
def cat_age := 8
def rabbit_age := cat_age / 2
def dog_age := 12
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

-- Theorem statement
theorem dog_rabbit_age_ratio : is_multiple dog_age rabbit_age ∧ dog_age / rabbit_age = 3 :=
by
  sorry

end dog_rabbit_age_ratio_l851_85145


namespace Amanda_car_round_trip_time_l851_85173

theorem Amanda_car_round_trip_time (bus_time : ℕ) (car_reduction : ℕ) (bus_one_way_trip : bus_time = 40) (car_time_reduction : car_reduction = 5) : 
  (2 * (bus_time - car_reduction)) = 70 := 
by
  sorry

end Amanda_car_round_trip_time_l851_85173


namespace product_of_k_values_l851_85153

theorem product_of_k_values (a b c k : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_eq : a / (1 + b) = k ∧ b / (1 + c) = k ∧ c / (1 + a) = k) : k = -1 :=
by
  sorry

end product_of_k_values_l851_85153


namespace no_fixed_point_range_of_a_fixed_point_in_interval_l851_85104

-- Problem (1)
theorem no_fixed_point_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + a ≠ x) →
  3 - 2 * Real.sqrt 2 < a ∧ a < 3 + 2 * Real.sqrt 2 :=
by
  sorry

-- Problem (2)
theorem fixed_point_in_interval (f : ℝ → ℝ) (n : ℤ) :
  (∀ x : ℝ, f x = -Real.log x + 3) →
  (∃ x₀ : ℝ, f x₀ = x₀ ∧ n ≤ x₀ ∧ x₀ < n + 1) →
  n = 2 :=
by
  sorry

end no_fixed_point_range_of_a_fixed_point_in_interval_l851_85104


namespace dress_total_selling_price_l851_85117

theorem dress_total_selling_price (original_price discount_rate tax_rate : ℝ) 
  (h1 : original_price = 100) (h2 : discount_rate = 0.30) (h3 : tax_rate = 0.15) : 
  (original_price * (1 - discount_rate) * (1 + tax_rate)) = 80.5 := by
  sorry

end dress_total_selling_price_l851_85117


namespace rita_bought_5_dresses_l851_85183

def pants_cost := 3 * 12
def jackets_cost := 4 * 30
def total_cost_pants_jackets := pants_cost + jackets_cost
def amount_spent := 400 - 139
def total_cost_dresses := amount_spent - total_cost_pants_jackets - 5
def number_of_dresses := total_cost_dresses / 20

theorem rita_bought_5_dresses : number_of_dresses = 5 :=
by sorry

end rita_bought_5_dresses_l851_85183


namespace number_added_is_minus_168_l851_85166

theorem number_added_is_minus_168 (N : ℕ) (X : ℤ) (h1 : N = 180)
  (h2 : N + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = (1/15 : ℚ) * N) : X = -168 :=
by
  sorry

end number_added_is_minus_168_l851_85166


namespace set_intersection_complement_l851_85115

/-- Definition of the universal set U. -/
def U := ({1, 2, 3, 4, 5} : Set ℕ)

/-- Definition of the set M. -/
def M := ({3, 4, 5} : Set ℕ)

/-- Definition of the set N. -/
def N := ({2, 3} : Set ℕ)

/-- Statement of the problem to be proven. -/
theorem set_intersection_complement :
  ((U \ N) ∩ M) = ({4, 5} : Set ℕ) :=
by
  sorry

end set_intersection_complement_l851_85115


namespace balls_in_boxes_l851_85103

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), balls = 5 ∧ boxes = 3 → 
  ∃ (ways : ℕ), ways = 21 :=
by
  sorry

end balls_in_boxes_l851_85103


namespace L_shaped_figure_perimeter_is_14_l851_85185

-- Define the side length of each square as a constant
def side_length : ℕ := 2

-- Define the horizontal base length
def base_length : ℕ := 3 * side_length

-- Define the height of the vertical stack
def vertical_stack_height : ℕ := 2 * side_length

-- Define the total perimeter of the "L" shaped figure
def L_shaped_figure_perimeter : ℕ :=
  base_length + side_length + vertical_stack_height + side_length + side_length + vertical_stack_height

-- The theorem that states the perimeter of the L-shaped figure is 14 units
theorem L_shaped_figure_perimeter_is_14 : L_shaped_figure_perimeter = 14 := sorry

end L_shaped_figure_perimeter_is_14_l851_85185


namespace tangent_line_equation_l851_85198

theorem tangent_line_equation
  (x y : ℝ)
  (h₁ : x^2 + y^2 = 5)
  (hM : x = -1 ∧ y = 2) :
  x - 2 * y + 5 = 0 :=
by
  sorry

end tangent_line_equation_l851_85198


namespace Nicki_total_miles_run_l851_85111

theorem Nicki_total_miles_run:
  ∀ (miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year : ℕ),
  miles_per_week_first_half = 20 →
  miles_per_week_second_half = 30 →
  weeks_in_year = 52 →
  weeks_per_half_year = weeks_in_year / 2 →
  (miles_per_week_first_half * weeks_per_half_year) + (miles_per_week_second_half * weeks_per_half_year) = 1300 :=
by
  intros miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year
  intros h1 h2 h3 h4
  sorry

end Nicki_total_miles_run_l851_85111


namespace speed_of_water_is_10_l851_85196

/-- Define the conditions -/
def swimming_speed_in_still_water : ℝ := 12 -- km/h
def time_to_swim_against_current : ℝ := 4 -- hours
def distance_against_current : ℝ := 8 -- km

/-- Define the effective speed against the current and the proof goal -/
def speed_of_water (v : ℝ) : Prop :=
  (swimming_speed_in_still_water - v) = distance_against_current / time_to_swim_against_current

theorem speed_of_water_is_10 : speed_of_water 10 :=
by
  unfold speed_of_water
  sorry

end speed_of_water_is_10_l851_85196


namespace smallest_n_is_29_l851_85184

noncomputable def smallest_possible_n (r g b : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm (10 * r) (16 * g)) (18 * b) / 25

theorem smallest_n_is_29 (r g b : ℕ) (h : 10 * r = 16 * g ∧ 16 * g = 18 * b) :
  smallest_possible_n r g b = 29 :=
by
  sorry

end smallest_n_is_29_l851_85184


namespace ranking_of_scores_l851_85102

-- Let the scores of Ann, Bill, Carol, and Dick be A, B, C, and D respectively.

variables (A B C D : ℝ)

-- Conditions
axiom cond1 : B + D = A + C
axiom cond2 : C + B > D + A
axiom cond3 : C > A + B

-- Statement of the problem
theorem ranking_of_scores : C > D ∧ D > B ∧ B > A :=
by
  -- Placeholder for proof (proof steps aren't required)
  sorry

end ranking_of_scores_l851_85102


namespace circumcircle_radius_l851_85175

-- Here we define the necessary conditions and prove the radius.
theorem circumcircle_radius
  (A B C : Type)
  (AB : ℝ)
  (angle_B : ℝ)
  (angle_A : ℝ)
  (h_AB : AB = 2)
  (h_angle_B : angle_B = 120)
  (h_angle_A : angle_A = 30) :
  ∃ R, R = 2 :=
by
  -- We will skip the proof using sorry
  sorry

end circumcircle_radius_l851_85175


namespace drawing_time_total_l851_85180

theorem drawing_time_total
  (bianca_school : ℕ)
  (bianca_home : ℕ)
  (lucas_school : ℕ)
  (lucas_home : ℕ)
  (h_bianca_school : bianca_school = 22)
  (h_bianca_home : bianca_home = 19)
  (h_lucas_school : lucas_school = 10)
  (h_lucas_home : lucas_home = 35) :
  bianca_school + bianca_home + lucas_school + lucas_home = 86 := 
by
  -- Proof would go here
  sorry

end drawing_time_total_l851_85180


namespace xiao_ding_distance_l851_85121

variable (x y z w : ℕ)

theorem xiao_ding_distance (h1 : x = 4 * y)
                          (h2 : z = x / 2 + 20)
                          (h3 : w = 2 * z - 15)
                          (h4 : x + y + z + w = 705) : 
                          y = 60 := 
sorry

end xiao_ding_distance_l851_85121


namespace minimum_buses_required_l851_85148

-- Condition definitions
def one_way_trip_time : ℕ := 50
def stop_time : ℕ := 10
def departure_interval : ℕ := 6

-- Total round trip time
def total_round_trip_time : ℕ := 2 * one_way_trip_time + 2 * stop_time

-- The total number of buses needed to ensure the bus departs every departure_interval minutes
-- from both stations A and B.
theorem minimum_buses_required : 
  (total_round_trip_time / departure_interval) = 20 := by
  sorry

end minimum_buses_required_l851_85148


namespace one_fourths_in_seven_halves_l851_85112

theorem one_fourths_in_seven_halves : (7 / 2) / (1 / 4) = 14 := by
  sorry

end one_fourths_in_seven_halves_l851_85112


namespace three_digit_numbers_divisible_by_13_count_l851_85162

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l851_85162


namespace math_problem_l851_85108

noncomputable def a : ℝ := 3.67
noncomputable def b : ℝ := 4.83
noncomputable def c : ℝ := 2.57
noncomputable def d : ℝ := -0.12
noncomputable def x : ℝ := 7.25
noncomputable def y : ℝ := -0.55

theorem math_problem :
  (3 * a * (4 * b - 2 * y)^2) / (5 * c * d^3 * 0.5 * x) - (2 * x * y^3) / (a * b^2 * c) = -57.179729 := 
sorry

end math_problem_l851_85108


namespace sin_neg_30_eq_neg_one_half_l851_85192

theorem sin_neg_30_eq_neg_one_half : Real.sin (-30 / 180 * Real.pi) = -1 / 2 := by
  -- Proof goes here
  sorry

end sin_neg_30_eq_neg_one_half_l851_85192


namespace largest_mersenne_prime_lt_500_l851_85186

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_mersenne_prime (p : ℕ) : Prop :=
  is_prime p ∧ is_prime (2^p - 1)

theorem largest_mersenne_prime_lt_500 : 
  ∀ n, is_mersenne_prime n → 2^n - 1 < 500 → 2^n - 1 ≤ 127 :=
by
  -- Proof goes here
  sorry

end largest_mersenne_prime_lt_500_l851_85186


namespace cube_side_length_l851_85113

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = 1 / 4 * 6 * n^3) : n = 4 := 
by 
  sorry

end cube_side_length_l851_85113


namespace volume_of_wall_is_16128_l851_85140

def wall_width : ℝ := 4
def wall_height : ℝ := 6 * wall_width
def wall_length : ℝ := 7 * wall_height

def wall_volume : ℝ := wall_length * wall_width * wall_height

theorem volume_of_wall_is_16128 :
  wall_volume = 16128 := by
  sorry

end volume_of_wall_is_16128_l851_85140


namespace sum_of_B_coordinates_l851_85119

theorem sum_of_B_coordinates 
  (x y : ℝ) 
  (A : ℝ × ℝ) 
  (M : ℝ × ℝ)
  (midpoint_x : (A.1 + x) / 2 = M.1) 
  (midpoint_y : (A.2 + y) / 2 = M.2) 
  (A_conds : A = (7, -1))
  (M_conds : M = (4, 3)) :
  x + y = 8 :=
by 
  sorry

end sum_of_B_coordinates_l851_85119


namespace cost_of_8_dozen_oranges_l851_85101

noncomputable def cost_per_dozen (cost_5_dozen : ℝ) : ℝ :=
  cost_5_dozen / 5

noncomputable def cost_8_dozen (cost_5_dozen : ℝ) : ℝ :=
  8 * cost_per_dozen cost_5_dozen

theorem cost_of_8_dozen_oranges (cost_5_dozen : ℝ) (h : cost_5_dozen = 39) : cost_8_dozen cost_5_dozen = 62.4 :=
by
  sorry

end cost_of_8_dozen_oranges_l851_85101


namespace relationship_t_s_l851_85179

theorem relationship_t_s (a b : ℝ) : 
  let t := a + 2 * b
  let s := a + b^2 + 1
  t <= s :=
by
  sorry

end relationship_t_s_l851_85179


namespace beats_per_week_l851_85159

def beats_per_minute : ℕ := 200
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7

theorem beats_per_week : beats_per_minute * minutes_per_hour * hours_per_day * days_per_week = 168000 := by
  sorry

end beats_per_week_l851_85159


namespace perfect_square_polynomial_l851_85182

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2) ↔ (x = -1 ∨ x = 0 ∨ x = 3) :=
sorry

end perfect_square_polynomial_l851_85182


namespace number_of_ordered_triples_l851_85135

theorem number_of_ordered_triples (x y z : ℝ) (hx : x + y = 3) (hy : xy - z^2 = 4)
  (hnn : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) : 
  ∃! (x y z : ℝ), (x + y = 3) ∧ (xy - z^2 = 4) ∧ (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) :=
sorry

end number_of_ordered_triples_l851_85135


namespace inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l851_85158

theorem inf_div_p_n2n_plus_one (p : ℕ) (hp : Nat.Prime p) (h_odd : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ (n * 2^n + 1) :=
sorry

theorem n_div_3_n2n_plus_one :
  (∃ k : ℕ, ∀ n, n = 6 * k + 1 ∨ n = 6 * k + 2 → 3 ∣ (n * 2^n + 1)) :=
sorry

end inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l851_85158


namespace pyramid_cross_section_distance_l851_85123

theorem pyramid_cross_section_distance 
  (A1 A2 : ℝ) (d : ℝ) (h : ℝ) 
  (hA1 : A1 = 125 * Real.sqrt 3)
  (hA2 : A2 = 500 * Real.sqrt 3)
  (hd : d = 12) :
  h = 24 :=
by
  sorry

end pyramid_cross_section_distance_l851_85123


namespace fewer_seats_on_right_than_left_l851_85142

theorem fewer_seats_on_right_than_left : 
  ∀ (left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats : ℕ),
    left_seats = 15 →
    back_seat_capacity = 9 →
    people_per_seat = 3 →
    bus_capacity = 90 →
    right_seats = (bus_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat →
    fewer_seats = left_seats - right_seats →
    fewer_seats = 3 :=
by
  intros left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats
  sorry

end fewer_seats_on_right_than_left_l851_85142


namespace relationship_between_M_and_N_l851_85136

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 4
def N : ℝ := (a - 1) * (a - 3)

theorem relationship_between_M_and_N : M a > N a :=
by sorry

end relationship_between_M_and_N_l851_85136


namespace translated_parabola_eq_l851_85106

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 3 * x^2

-- Function to translate a parabola equation downward by a units
def translate_downward (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f x - a

-- Function to translate a parabola equation rightward by b units
def translate_rightward (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f (x - b)

-- The new parabola equation after translating the given parabola downward by 3 units and rightward by 2 units
def new_parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 9

-- The main theorem stating that translating the original parabola downward by 3 units and rightward by 2 units results in the new parabola equation
theorem translated_parabola_eq :
  ∀ x : ℝ, translate_rightward (translate_downward original_parabola 3) 2 x = new_parabola x :=
by
  sorry

end translated_parabola_eq_l851_85106


namespace general_term_formula_l851_85109

theorem general_term_formula :
  ∀ n : ℕ, (0 < n) → 
  (-1)^n * (2*n + 1) / (2*n) = ((-1) : ℝ)^n * ((2*n + 1) : ℝ) / (2*n) :=
by {
  sorry
}

end general_term_formula_l851_85109


namespace middle_number_of_consecutive_sum_30_l851_85176

theorem middle_number_of_consecutive_sum_30 (n : ℕ) (h : n + (n + 1) + (n + 2) = 30) : n + 1 = 10 :=
by
  sorry

end middle_number_of_consecutive_sum_30_l851_85176


namespace min_value_x_plus_y_l851_85165

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 19 / x + 98 / y = 1) : 
  x + y ≥ 117 + 14 * Real.sqrt 38 :=
  sorry

end min_value_x_plus_y_l851_85165


namespace convex_polygons_from_fifteen_points_l851_85160

theorem convex_polygons_from_fifteen_points 
    (h : ∀ (n : ℕ), n = 15) :
    ∃ (k : ℕ), k = 32192 :=
by
  sorry

end convex_polygons_from_fifteen_points_l851_85160


namespace heartsuit_symmetric_solution_l851_85147

def heartsuit (a b : ℝ) : ℝ :=
  a^3 * b - a^2 * b^2 + a * b^3

theorem heartsuit_symmetric_solution :
  ∀ x y : ℝ, (heartsuit x y = heartsuit y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end heartsuit_symmetric_solution_l851_85147


namespace percent_absent_of_students_l851_85171

theorem percent_absent_of_students
  (boys girls : ℕ)
  (total_students := boys + girls)
  (boys_absent_fraction girls_absent_fraction : ℚ)
  (boys_absent_fraction_eq : boys_absent_fraction = 1 / 8)
  (girls_absent_fraction_eq : girls_absent_fraction = 1 / 4)
  (total_students_eq : total_students = 160)
  (boys_eq : boys = 80)
  (girls_eq : girls = 80) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 18.75 :=
by
  sorry

end percent_absent_of_students_l851_85171


namespace sum_of_cubes_l851_85105

theorem sum_of_cubes {x y : ℝ} (h₁ : x + y = 0) (h₂ : x * y = -1) : x^3 + y^3 = 0 :=
by
  sorry

end sum_of_cubes_l851_85105


namespace work_days_in_week_l851_85152

theorem work_days_in_week (total_toys_per_week : ℕ) (toys_produced_each_day : ℕ) (h1 : total_toys_per_week = 6500) (h2 : toys_produced_each_day = 1300) : 
  total_toys_per_week / toys_produced_each_day = 5 :=
by
  sorry

end work_days_in_week_l851_85152


namespace common_ratio_of_increasing_geometric_sequence_l851_85144

theorem common_ratio_of_increasing_geometric_sequence 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_inc : ∀ n, a n < a (n + 1))
  (h_a2 : a 2 = 2)
  (h_a4_a3 : a 4 - a 3 = 4) : 
  q = 2 :=
by
  -- sorry - placeholder for proof
  sorry

end common_ratio_of_increasing_geometric_sequence_l851_85144


namespace suraj_average_l851_85170

theorem suraj_average : 
  ∀ (A : ℝ), 
    (16 * A + 92 = 17 * (A + 4)) → 
      (A + 4) = 28 :=
by
  sorry

end suraj_average_l851_85170


namespace find_positive_real_solution_l851_85193

theorem find_positive_real_solution (x : ℝ) (h1 : x > 0) (h2 : (x - 5) / 8 = 5 / (x - 8)) : x = 13 := 
sorry

end find_positive_real_solution_l851_85193


namespace num_boys_and_girls_l851_85163

def num_ways_to_select (x : ℕ) := (x * (x - 1) / 2) * (8 - x) * 6

theorem num_boys_and_girls (x : ℕ) (h1 : num_ways_to_select x = 180) :
    x = 5 ∨ x = 6 :=
by
  sorry

end num_boys_and_girls_l851_85163


namespace sequence_length_l851_85129

theorem sequence_length :
  ∀ (n : ℕ), 
    (2 + 4 * (n - 1) = 2010) → n = 503 :=
by
    intro n
    intro h
    sorry

end sequence_length_l851_85129


namespace sum_evaluation_l851_85168

noncomputable def T : ℝ := ∑' k : ℕ, (2*k+1) / 5^(k+1)

theorem sum_evaluation : T = 5 / 16 := sorry

end sum_evaluation_l851_85168


namespace max_xy_of_perpendicular_l851_85199

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (y, 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 

theorem max_xy_of_perpendicular (x y : ℝ) 
  (h_perp : dot_product (vector_a x) (vector_b y) = 0) : xy ≤ 1/2 :=
by
  sorry

end max_xy_of_perpendicular_l851_85199


namespace standard_deviation_is_one_l851_85161

def mean : ℝ := 10.5
def value : ℝ := 8.5

theorem standard_deviation_is_one (σ : ℝ) (h : value = mean - 2 * σ) : σ = 1 :=
by {
  sorry
}

end standard_deviation_is_one_l851_85161


namespace max_ab_value_l851_85122

theorem max_ab_value {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : 6 * a + 8 * b = 72) : ab = 27 :=
by {
  sorry
}

end max_ab_value_l851_85122


namespace evaluate_expression_l851_85133

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by sorry

end evaluate_expression_l851_85133


namespace sum_of_first_seven_terms_l851_85131

variable {a_n : ℕ → ℝ} {d : ℝ}

-- Define the arithmetic progression condition.
def arithmetic_progression (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n n = a_n 0 + n * d

-- We are given that the sequence is an arithmetic progression.
axiom sequence_is_arithmetic_progression : arithmetic_progression a_n d

-- We are also given that the sum of the 3rd, 4th, and 5th terms is 12.
axiom sum_of_terms_is_12 : a_n 2 + a_n 3 + a_n 4 = 12

-- We need to prove that the sum of the first seven terms is 28.
theorem sum_of_first_seven_terms : (a_n 0) + (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) + (a_n 6) = 28 := 
  sorry

end sum_of_first_seven_terms_l851_85131


namespace minimum_value_of_A_l851_85151

open Real

noncomputable def A (x y z : ℝ) : ℝ :=
  ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x * y + y * z + z * x)

theorem minimum_value_of_A (x y z : ℝ) (h : 3 ≤ x) (h2 : 3 ≤ y) (h3 : 3 ≤ z) :
  ∃ v : ℝ, (∀ a b c : ℝ, 3 ≤ a ∧ 3 ≤ b ∧ 3 ≤ c → A a b c ≥ v) ∧ v = 1 :=
sorry

end minimum_value_of_A_l851_85151


namespace binary_to_decimal_l851_85188

theorem binary_to_decimal : 
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0) = 54 :=
by 
  sorry

end binary_to_decimal_l851_85188


namespace cylinder_base_radius_l851_85154

theorem cylinder_base_radius (a : ℝ) (h_a_pos : 0 < a) :
  ∃ (R : ℝ), R = 7 * a * Real.sqrt 3 / 24 := 
    sorry

end cylinder_base_radius_l851_85154


namespace sqrt_expression_eq_1720_l851_85187

theorem sqrt_expression_eq_1720 : Real.sqrt ((43 * 42 * 41 * 40) + 1) = 1720 := by
  sorry

end sqrt_expression_eq_1720_l851_85187


namespace sum_of_numbers_with_lcm_and_ratio_l851_85150

theorem sum_of_numbers_with_lcm_and_ratio 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 48)
  (h_ratio : a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3) : 
  a + b = 80 := 
by sorry

end sum_of_numbers_with_lcm_and_ratio_l851_85150


namespace max_additional_pies_l851_85125

theorem max_additional_pies (initial_cherries used_cherries cherries_per_pie : ℕ) 
  (h₀ : initial_cherries = 500) 
  (h₁ : used_cherries = 350) 
  (h₂ : cherries_per_pie = 35) :
  (initial_cherries - used_cherries) / cherries_per_pie = 4 := 
by
  sorry

end max_additional_pies_l851_85125


namespace correctTechnologyUsedForVolcanicAshMonitoring_l851_85164

-- Define the choices
inductive Technology
| RemoteSensing : Technology
| GPS : Technology
| GIS : Technology
| DigitalEarth : Technology

-- Define the problem conditions
def primaryTechnologyUsedForVolcanicAshMonitoring := Technology.RemoteSensing

-- The statement to prove
theorem correctTechnologyUsedForVolcanicAshMonitoring : primaryTechnologyUsedForVolcanicAshMonitoring = Technology.RemoteSensing :=
by
  sorry

end correctTechnologyUsedForVolcanicAshMonitoring_l851_85164


namespace find_number_l851_85132

-- Define the number 40 and the percentage 90.
def num : ℝ := 40
def percent : ℝ := 0.9

-- Define the condition that 4/5 of x is smaller than 90% of 40 by 16
def condition (x : ℝ) : Prop := (4/5 : ℝ) * x = percent * num - 16

-- Proof statement in Lean 4
theorem find_number : ∃ x : ℝ, condition x ∧ x = 25 :=
by 
  use 25
  unfold condition
  norm_num
  sorry

end find_number_l851_85132


namespace triangle_equilateral_of_constraints_l851_85141

theorem triangle_equilateral_of_constraints {a b c : ℝ}
  (h1 : a^4 = b^4 + c^4 - b^2 * c^2)
  (h2 : b^4 = c^4 + a^4 - a^2 * c^2) : 
  a = b ∧ b = c :=
by 
  sorry

end triangle_equilateral_of_constraints_l851_85141


namespace magnitude_of_z_l851_85100

open Complex

noncomputable def z : ℂ := (1 - I) / (1 + I) + 2 * I

theorem magnitude_of_z : Complex.abs z = 1 := by
  sorry

end magnitude_of_z_l851_85100


namespace quadrilateral_side_length_l851_85174

-- Definitions
def inscribed_quadrilateral (a b c d r : ℝ) : Prop :=
  ∃ (O : ℝ) (A B C D : ℝ), 
    O = r ∧ 
    A = a ∧ B = b ∧ C = c ∧ 
    (r^2 + r^2 = (a^2 + b^2) / 2) ∧
    (r^2 + r^2 = (b^2 + c^2) / 2) ∧
    (r^2 + r^2 = (c^2 + d^2) / 2)

-- Theorem statement
theorem quadrilateral_side_length :
  inscribed_quadrilateral 250 250 100 200 250 :=
sorry

end quadrilateral_side_length_l851_85174


namespace quadratic_eq_is_general_form_l851_85124

def quadratic_eq_general_form (x : ℝ) : Prop :=
  x^2 - 2 * (3 * x - 2) + (x + 1) = x^2 - 5 * x + 5

theorem quadratic_eq_is_general_form :
  quadratic_eq_general_form x :=
sorry

end quadratic_eq_is_general_form_l851_85124


namespace not_on_line_l851_85189

-- Defining the point (0,20)
def pt : ℝ × ℝ := (0, 20)

-- Defining the line equation
def line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- The proof problem stating that for all real numbers m and b, if m + b < 0, 
-- then the point (0, 20) cannot be on the line y = mx + b
theorem not_on_line (m b : ℝ) (h : m + b < 0) : ¬line m b pt := by
  sorry

end not_on_line_l851_85189


namespace largest_angle_l851_85172

-- Assume the conditions
def angle_a : ℝ := 50
def angle_b : ℝ := 70
def angle_c (y : ℝ) : ℝ := 180 - (angle_a + angle_b)

-- State the proposition
theorem largest_angle (y : ℝ) (h : y = angle_c y) : angle_b = 70 := by
  sorry

end largest_angle_l851_85172


namespace balloon_permutations_l851_85178

theorem balloon_permutations : 
  (Nat.factorial 7 / 
  ((Nat.factorial 1) * 
  (Nat.factorial 1) * 
  (Nat.factorial 2) * 
  (Nat.factorial 2) * 
  (Nat.factorial 1))) = 1260 := by
  sorry

end balloon_permutations_l851_85178


namespace range_of_a_l851_85130

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 2 * x > x^2 + a) → a < -8 :=
by
  intro h
  -- Complete the proof by showing that 2x - x^2 has a minimum value of -8 on [-2, 3] and hence proving a < -8.
  sorry

end range_of_a_l851_85130
