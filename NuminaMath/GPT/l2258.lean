import Mathlib

namespace floor_S_value_l2258_225899

theorem floor_S_value
  (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (h_ab_squared : a^2 + b^2 = 1458)
  (h_cd_squared : c^2 + d^2 = 1458)
  (h_ac_product : a * c = 1156)
  (h_bd_product : b * d = 1156) :
  (⌊a + b + c + d⌋ = 77) := 
sorry

end floor_S_value_l2258_225899


namespace function_below_x_axis_l2258_225854

theorem function_below_x_axis (k : ℝ) :
  (∀ x : ℝ, (k^2 - k - 2) * x^2 - (k - 2) * x - 1 < 0) ↔ (-2 / 5 < k ∧ k ≤ 2) :=
by
  sorry

end function_below_x_axis_l2258_225854


namespace palindrome_digital_clock_l2258_225829

theorem palindrome_digital_clock (no_leading_zero : ∀ h : ℕ, h < 10 → ¬ ∃ h₂ : ℕ, h₂ = h * 1000)
                                 (max_hour : ∀ h : ℕ, h ≥ 24 → false) :
  ∃ n : ℕ, n = 61 := by
  sorry

end palindrome_digital_clock_l2258_225829


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l2258_225835

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l2258_225835


namespace hyperbola_h_k_a_b_sum_l2258_225880

noncomputable def h : ℝ := 1
noncomputable def k : ℝ := -3
noncomputable def a : ℝ := 3
noncomputable def c : ℝ := 3 * Real.sqrt 5
noncomputable def b : ℝ := 6

theorem hyperbola_h_k_a_b_sum :
  h + k + a + b = 7 :=
by
  sorry

end hyperbola_h_k_a_b_sum_l2258_225880


namespace stream_speed_l2258_225847

theorem stream_speed (v : ℝ) (h1 : ∀ d : ℝ, d > 0 → ((1:ℝ) / (5 - v) = 2 * (1 / (5 + v)))) : 
  v = 5 / 3 :=
by
  -- Variables and assumptions
  have h1 : ∀ d : ℝ, d > 0 → ((1:ℝ) / (5 - v) = 2 * (1 / (5 + v))) := sorry
  -- To prove
  sorry

end stream_speed_l2258_225847


namespace value_of_a_l2258_225855

theorem value_of_a (a : ℝ) (A : Set ℝ) (h : ∀ x, x ∈ A ↔ |x - a| < 1) : A = Set.Ioo 1 3 → a = 2 :=
by
  intro ha
  have : Set.Ioo 1 3 = {x | ∃ y, y ∈ Set.Ioi (1 : ℝ) ∧ y ∈ Set.Iio (3 : ℝ)} := by sorry
  sorry

end value_of_a_l2258_225855


namespace quadratic_transformation_l2258_225830

theorem quadratic_transformation (p q r : ℝ) :
  (∀ x : ℝ, p * x^2 + q * x + r = 3 * (x - 5)^2 + 15) →
  (∀ x : ℝ, 4 * p * x^2 + 4 * q * x + 4 * r = 12 * (x - 5)^2 + 60) :=
by
  intro h
  exact sorry

end quadratic_transformation_l2258_225830


namespace music_commercials_ratio_l2258_225817

theorem music_commercials_ratio (T C: ℕ) (hT: T = 112) (hC: C = 40) : (T - C) / C = 9 / 5 := by
  sorry

end music_commercials_ratio_l2258_225817


namespace build_bridge_l2258_225859

/-- It took 6 days for 60 workers, all working together at the same rate, to build a bridge.
    Prove that if only 30 workers had been available, it would have taken 12 total days to build the bridge. -/
theorem build_bridge (days_60_workers : ℕ) (num_60_workers : ℕ) (same_rate : Prop) : 
  (days_60_workers = 6) → (num_60_workers = 60) → (same_rate = ∀ n m, n * days_60_workers = m * days_30_workers) → (days_30_workers = 12) :=
by
  sorry

end build_bridge_l2258_225859


namespace ceiling_floor_expression_l2258_225862

theorem ceiling_floor_expression :
  (Int.ceil ((12:ℚ) / 5 * ((-19:ℚ) / 4 - 3)) - Int.floor (((12:ℚ) / 5) * Int.floor ((-19:ℚ) / 4)) = -6) :=
by 
  sorry

end ceiling_floor_expression_l2258_225862


namespace solve_y_l2258_225842

theorem solve_y (y : ℝ) (h : 5 * y^(1/4) - 3 * (y / y^(3/4)) = 9 + y^(1/4)) : y = 6561 := 
by 
  sorry

end solve_y_l2258_225842


namespace smallest_value_of_y_square_l2258_225822

-- Let's define the conditions
variable (EF GH y : ℝ)

-- The given conditions of the problem
def is_isosceles_trapezoid (EF GH y : ℝ) : Prop :=
  EF = 100 ∧ GH = 25 ∧ y > 0

def has_tangent_circle (EF GH y : ℝ) : Prop :=
  is_isosceles_trapezoid EF GH y ∧ 
  ∃ P : ℝ, P = EF / 2

-- Main proof statement
theorem smallest_value_of_y_square (EF GH y : ℝ)
  (h1 : is_isosceles_trapezoid EF GH y)
  (h2 : has_tangent_circle EF GH y) :
  y^2 = 1875 :=
  sorry

end smallest_value_of_y_square_l2258_225822


namespace cars_on_river_road_l2258_225805

variable (B C M : ℕ)

theorem cars_on_river_road
  (h1 : ∃ B C : ℕ, B / C = 1 / 3) -- ratio of buses to cars is 1:3
  (h2 : ∀ B C : ℕ, C = B + 40) -- 40 fewer buses than cars
  (h3 : ∃ B C M : ℕ, B + C + M = 720) -- total number of vehicles is 720
  : C = 60 :=
sorry

end cars_on_river_road_l2258_225805


namespace anna_ate_cupcakes_l2258_225873

-- Given conditions
def total_cupcakes : Nat := 60
def cupcakes_given_away (total : Nat) : Nat := (4 * total) / 5
def cupcakes_remaining (total : Nat) : Nat := total - cupcakes_given_away total
def anna_cupcakes_left : Nat := 9

-- Proving the number of cupcakes Anna ate
theorem anna_ate_cupcakes : cupcakes_remaining total_cupcakes - anna_cupcakes_left = 3 := by
  sorry

end anna_ate_cupcakes_l2258_225873


namespace least_possible_students_l2258_225851

def TotalNumberOfStudents : ℕ := 35
def NumberOfStudentsWithBrownEyes : ℕ := 15
def NumberOfStudentsWithLunchBoxes : ℕ := 25
def NumberOfStudentsWearingGlasses : ℕ := 10

theorem least_possible_students (TotalNumberOfStudents NumberOfStudentsWithBrownEyes NumberOfStudentsWithLunchBoxes NumberOfStudentsWearingGlasses : ℕ) :
  ∃ n, n = 5 :=
sorry

end least_possible_students_l2258_225851


namespace solve_for_x_l2258_225818

theorem solve_for_x (x : ℝ) : (3 / 2) * x - 3 = 15 → x = 12 := 
by
  sorry

end solve_for_x_l2258_225818


namespace fido_area_reach_l2258_225885

theorem fido_area_reach (r : ℝ) (s : ℝ) (a b : ℕ) (h1 : r = s * (1 / (Real.tan (Real.pi / 8))))
  (h2 : a = 2) (h3 : b = 8)
  (h_fraction : (Real.pi * r ^ 2) / (2 * (1 + Real.sqrt 2) * (r ^ 2 * (Real.tan (Real.pi / 8)) ^ 2)) = (Real.pi * Real.sqrt a) / b) :
  a * b = 16 := by
  sorry

end fido_area_reach_l2258_225885


namespace ice_cubes_per_tray_l2258_225891

theorem ice_cubes_per_tray (total_ice_cubes : ℕ) (number_of_trays : ℕ) (h1 : total_ice_cubes = 72) (h2 : number_of_trays = 8) : 
  total_ice_cubes / number_of_trays = 9 :=
by
  sorry

end ice_cubes_per_tray_l2258_225891


namespace player_matches_average_increase_l2258_225800

theorem player_matches_average_increase 
  (n T : ℕ) 
  (h1 : T = 32 * n) 
  (h2 : (T + 76) / (n + 1) = 36) : 
  n = 10 := 
by 
  sorry

end player_matches_average_increase_l2258_225800


namespace cube_root_of_5_irrational_l2258_225843

theorem cube_root_of_5_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ)^3 = 5 * (q : ℚ)^3 := by
  sorry

end cube_root_of_5_irrational_l2258_225843


namespace standing_next_to_boris_l2258_225884

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l2258_225884


namespace problem_statement_l2258_225870

theorem problem_statement (a b : ℝ) (C : ℝ) (sin_C : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sin_C = (Real.sqrt 15) / 4) :
  Real.cos C = 1 / 4 :=
sorry

end problem_statement_l2258_225870


namespace athlete_last_finish_l2258_225894

theorem athlete_last_finish (v1 v2 v3 : ℝ) (h1 : v1 > v2) (h2 : v2 > v3) :
  let T1 := 1 / v1 + 2 / v2 
  let T2 := 1 / v2 + 2 / v3
  let T3 := 1 / v3 + 2 / v1
  T2 > T1 ∧ T2 > T3 :=
by
  sorry

end athlete_last_finish_l2258_225894


namespace portfolio_value_after_two_years_l2258_225832

def initial_portfolio := 80

def first_year_growth_rate := 0.15
def add_after_6_months := 28
def withdraw_after_9_months := 10

def second_year_growth_first_6_months := 0.10
def second_year_decline_last_6_months := 0.04

def final_portfolio_value := 115.59

theorem portfolio_value_after_two_years 
  (initial_portfolio : ℝ)
  (first_year_growth_rate : ℝ)
  (add_after_6_months : ℕ)
  (withdraw_after_9_months : ℕ)
  (second_year_growth_first_6_months : ℝ)
  (second_year_decline_last_6_months : ℝ)
  (final_portfolio_value : ℝ) :
  (initial_portfolio = 80) →
  (first_year_growth_rate = 0.15) →
  (add_after_6_months = 28) →
  (withdraw_after_9_months = 10) →
  (second_year_growth_first_6_months = 0.10) →
  (second_year_decline_last_6_months = 0.04) →
  (final_portfolio_value = 115.59) :=
by
  sorry

end portfolio_value_after_two_years_l2258_225832


namespace find_m_l2258_225869

open Real

def vec := (ℝ × ℝ)

def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def a : vec := (-1, 2)
def b (m : ℝ) : vec := (3, m)
def sum (m : ℝ) : vec := (a.1 + (b m).1, a.2 + (b m).2)

theorem find_m (m : ℝ) (h : dot_product a (sum m) = 0) : m = -1 :=
by {
  sorry
}

end find_m_l2258_225869


namespace jackie_apples_l2258_225875

theorem jackie_apples (a : ℕ) (j : ℕ) (h1 : a = 9) (h2 : a = j + 3) : j = 6 :=
by
  sorry

end jackie_apples_l2258_225875


namespace externally_tangent_intersect_two_points_l2258_225808

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0
def circle2 (x y r : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = r^2 ∧ r > 0

theorem externally_tangent (r : ℝ) : 
  (∃ x y : ℝ, circle2 x y r) →
  (∃ x y : ℝ, circle1 x y) → 
  (dist (1, 1) (4, 5) = r + 1) → 
  r = 4 := 
sorry

theorem intersect_two_points (r : ℝ) : 
  (∃ x y : ℝ, circle2 x y r) → 
  (∃ x y : ℝ, circle1 x y) → 
  (|r - 1| < dist (1, 1) (4, 5) ∧ dist (1, 1) (4, 5) < r + 1) → 
  4 < r ∧ r < 6 :=
sorry

end externally_tangent_intersect_two_points_l2258_225808


namespace polynomial_unique_l2258_225807

noncomputable def p (x : ℝ) : ℝ := x^2 + 1

theorem polynomial_unique (p : ℝ → ℝ) 
  (h1 : p 2 = 5) 
  (h2 : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) : 
  ∀ x : ℝ, p x = x^2 + 1 :=
by
  sorry

end polynomial_unique_l2258_225807


namespace second_watermelon_correct_weight_l2258_225868

-- Define various weights involved as given in the conditions
def first_watermelon_weight : ℝ := 9.91
def total_watermelon_weight : ℝ := 14.02

-- Define the weight of the second watermelon
def second_watermelon_weight : ℝ :=
  total_watermelon_weight - first_watermelon_weight

-- State the theorem to prove that the weight of the second watermelon is 4.11 pounds
theorem second_watermelon_correct_weight : second_watermelon_weight = 4.11 :=
by
  -- This ensures the statement can be built successfully in Lean 4
  sorry

end second_watermelon_correct_weight_l2258_225868


namespace advance_agency_fees_eq_8280_l2258_225888

-- Conditions
variables (Commission GivenFees Incentive AdvanceAgencyFees : ℝ)
-- Given values
variables (h_comm : Commission = 25000) 
          (h_given : GivenFees = 18500) 
          (h_incent : Incentive = 1780)

-- The problem statement to prove
theorem advance_agency_fees_eq_8280 
    (h_comm : Commission = 25000) 
    (h_given : GivenFees = 18500) 
    (h_incent : Incentive = 1780)
    : AdvanceAgencyFees = 26780 - GivenFees :=
by
  sorry

end advance_agency_fees_eq_8280_l2258_225888


namespace max_value_of_a_l2258_225860

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem max_value_of_a (a b c d : ℝ) (h_deriv_bounds : ∀ x, 0 ≤ x → x ≤ 1 → abs (3 * a * x^2 + 2 * b * x + c) ≤ 1) (h_a_nonzero : a ≠ 0) :
  a ≤ 8 / 3 :=
sorry

end max_value_of_a_l2258_225860


namespace championship_titles_l2258_225820

theorem championship_titles {S T : ℕ} (h_S : S = 4) (h_T : T = 3) : S^T = 64 := by
  rw [h_S, h_T]
  norm_num

end championship_titles_l2258_225820


namespace identify_stolen_treasure_l2258_225845

-- Define the magic square arrangement
def magic_square (bags : ℕ → ℕ) :=
  bags 0 + bags 1 + bags 2 = 15 ∧
  bags 3 + bags 4 + bags 5 = 15 ∧
  bags 6 + bags 7 + bags 8 = 15 ∧
  bags 0 + bags 3 + bags 6 = 15 ∧
  bags 1 + bags 4 + bags 7 = 15 ∧
  bags 2 + bags 5 + bags 8 = 15 ∧
  bags 0 + bags 4 + bags 8 = 15 ∧
  bags 2 + bags 4 + bags 6 = 15

-- Define the stolen treasure detection function
def stolen_treasure (bags : ℕ → ℕ) : Prop :=
  ∃ altered_bag_idx : ℕ, (bags altered_bag_idx ≠ altered_bag_idx + 1)

-- The main theorem
theorem identify_stolen_treasure (bags : ℕ → ℕ) (h_magic_square : magic_square bags) : ∃ altered_bag_idx : ℕ, stolen_treasure bags :=
sorry

end identify_stolen_treasure_l2258_225845


namespace unique_and_double_solutions_l2258_225821

theorem unique_and_double_solutions (a : ℝ) :
  (∃ (x : ℝ), 5 + |x - 2| = a ∧ ∀ y, 5 + |y - 2| = a → y = x ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 7 - |2*x1 + 6| = a ∧ 7 - |2*x2 + 6| = a)) ∨
  (∃ (x : ℝ), 7 - |2*x + 6| = a ∧ ∀ y, 7 - |2*y + 6| = a → y = x ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 5 + |x1 - 2| = a ∧ 5 + |x2 - 2| = a)) ↔ a = 5 ∨ a = 7 :=
by
  sorry

end unique_and_double_solutions_l2258_225821


namespace Gracie_height_is_correct_l2258_225834

-- Given conditions
def Griffin_height : ℤ := 61
def Grayson_height : ℤ := Griffin_height + 2
def Gracie_height : ℤ := Grayson_height - 7

-- The proof problem: Prove that Gracie's height is 56 inches.
theorem Gracie_height_is_correct : Gracie_height = 56 := by
  sorry

end Gracie_height_is_correct_l2258_225834


namespace sum_of_B_and_C_in_base_6_l2258_225809

def digit_base_6 (n: Nat) : Prop :=
  n > 0 ∧ n < 6

theorem sum_of_B_and_C_in_base_6
  (A B C : Nat)
  (hA : digit_base_6 A)
  (hB : digit_base_6 B)
  (hC : digit_base_6 C)
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : 43 * (A + B + C) = 216 * A) :
  B + C = 5 := by
  sorry

end sum_of_B_and_C_in_base_6_l2258_225809


namespace original_selling_price_l2258_225828

/-- A boy sells a book for some amount and he gets a loss of 10%.
To gain 10%, the selling price should be Rs. 550.
Prove that the original selling price of the book was Rs. 450. -/
theorem original_selling_price (CP : ℝ) (h1 : 1.10 * CP = 550) :
    0.90 * CP = 450 := 
sorry

end original_selling_price_l2258_225828


namespace marilyn_ends_up_with_55_caps_l2258_225819

def marilyn_initial_caps := 165
def caps_shared_with_nancy := 78
def caps_received_from_charlie := 23

def remaining_caps (initial caps_shared caps_received: ℕ) :=
  initial - caps_shared + caps_received

def caps_given_away (total_caps: ℕ) :=
  total_caps / 2

def final_caps (initial caps_shared caps_received: ℕ) :=
  remaining_caps initial caps_shared caps_received - caps_given_away (remaining_caps initial caps_shared caps_received)

theorem marilyn_ends_up_with_55_caps :
  final_caps marilyn_initial_caps caps_shared_with_nancy caps_received_from_charlie = 55 :=
by
  sorry

end marilyn_ends_up_with_55_caps_l2258_225819


namespace overall_loss_is_450_l2258_225886

noncomputable def total_worth_stock : ℝ := 22499.999999999996

noncomputable def selling_price_20_percent_stock (W : ℝ) : ℝ :=
    0.20 * W * 1.10

noncomputable def selling_price_80_percent_stock (W : ℝ) : ℝ :=
    0.80 * W * 0.95

noncomputable def total_selling_price (W : ℝ) : ℝ :=
    selling_price_20_percent_stock W + selling_price_80_percent_stock W

noncomputable def overall_loss (W : ℝ) : ℝ :=
    W - total_selling_price W

theorem overall_loss_is_450 :
  overall_loss total_worth_stock = 450 := by
  sorry

end overall_loss_is_450_l2258_225886


namespace Andy_earnings_l2258_225801

/-- Andy's total earnings during an 8-hour shift. --/
theorem Andy_earnings (hours : ℕ) (hourly_wage : ℕ) (num_racquets : ℕ) (pay_per_racquet : ℕ)
  (num_grommets : ℕ) (pay_per_grommet : ℕ) (num_stencils : ℕ) (pay_per_stencil : ℕ)
  (h_shift : hours = 8) (h_hourly : hourly_wage = 9) (h_racquets : num_racquets = 7)
  (h_pay_racquets : pay_per_racquet = 15) (h_grommets : num_grommets = 2)
  (h_pay_grommets : pay_per_grommet = 10) (h_stencils : num_stencils = 5)
  (h_pay_stencils : pay_per_stencil = 1) :
  (hours * hourly_wage + num_racquets * pay_per_racquet + num_grommets * pay_per_grommet +
  num_stencils * pay_per_stencil) = 202 :=
by
  sorry

end Andy_earnings_l2258_225801


namespace inequality_proof_l2258_225815

theorem inequality_proof (a b c : ℝ) (h1 : b < c) (h2 : 1 < a) (h3 : a < b + c) (h4 : b + c < a + 1) : b < a :=
by
  sorry

end inequality_proof_l2258_225815


namespace nesbitt_inequality_l2258_225811

variable (a b c d : ℝ)

-- Assume a, b, c, d are positive real numbers
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom pos_d : 0 < d

theorem nesbitt_inequality :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end nesbitt_inequality_l2258_225811


namespace tangent_line_intersecting_lines_l2258_225836

variable (x y : ℝ)

-- Definition of the circle
def circle_C : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Definition of the point
def point_A : Prop := x = 1 ∧ y = 0

-- (I) Prove that if l is tangent to circle C and passes through A, l is 3x - 4y - 3 = 0
theorem tangent_line (l : ℝ → ℝ) (h : ∀ x, l x = k * (x - 1)) :
  (∀ {x y}, circle_C x y → 3 * x - 4 * y - 3 = 0) :=
by
  sorry

-- (II) Prove that the maximum area of triangle CPQ intersecting circle C is 2, and l's equations are y = 7x - 7 or y = x - 1
theorem intersecting_lines (k : ℝ) :
  (∃ x y, circle_C x y ∧ point_A x y) →
  (∃ k : ℝ, k = 7 ∨ k = 1) :=
by
  sorry

end tangent_line_intersecting_lines_l2258_225836


namespace solve_inequality_l2258_225812

theorem solve_inequality (a : ℝ) :
  (a = 0 → {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a > 0 → {x : ℝ | x ≥ 2 / a} ∪ {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (-2 < a ∧ a < 0 → {x : ℝ | 2 / a ≤ x ∧ x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a = -2 → {x : ℝ | x = -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a < -2 → {x : ℝ | -1 ≤ x ∧ x ≤ 2 / a} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) :=
by 
  sorry

end solve_inequality_l2258_225812


namespace election_total_votes_l2258_225893

theorem election_total_votes (V : ℝ)
  (h_majority : ∃ O, 0.84 * V = O + 476)
  (h_total_votes : ∀ O, V = 0.84 * V + O) :
  V = 700 :=
sorry

end election_total_votes_l2258_225893


namespace frac_multiplication_l2258_225882

theorem frac_multiplication : 
    ((2/3:ℚ)^4 * (1/5) * (3/4) = 4/135) :=
by
  sorry

end frac_multiplication_l2258_225882


namespace geometric_series_sum_l2258_225867

theorem geometric_series_sum :
  let a := (2 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := 6
  (a * (1 - r^n) / (1 - r) = 728 / 243) := 
by
  let a := (2 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := 6
  show a * (1 - r^n) / (1 - r) = 728 / 243
  sorry

end geometric_series_sum_l2258_225867


namespace mass_percentage_of_Cl_in_NH4Cl_l2258_225833

-- Definition of the molar masses (conditions)
def molar_mass_N : ℝ := 14.01
def molar_mass_H : ℝ := 1.01
def molar_mass_Cl : ℝ := 35.45

-- Definition of the molar mass of NH4Cl
def molar_mass_NH4Cl : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_Cl

-- The expected mass percentage of Cl in NH4Cl
def expected_mass_percentage_Cl : ℝ := 66.26

-- The proof statement
theorem mass_percentage_of_Cl_in_NH4Cl :
  (molar_mass_Cl / molar_mass_NH4Cl) * 100 = expected_mass_percentage_Cl :=
by 
  -- The body of the proof is omitted, as it is not necessary to provide the proof.
  sorry

end mass_percentage_of_Cl_in_NH4Cl_l2258_225833


namespace triangle_equilateral_l2258_225853

theorem triangle_equilateral
  (a b c : ℝ)
  (h : a^4 + b^4 + c^4 - a^2 * b^2 - b^2 * c^2 - a^2 * c^2 = 0) :
  a = b ∧ b = c ∧ a = c := 
by
  sorry

end triangle_equilateral_l2258_225853


namespace range_of_m_l2258_225871

theorem range_of_m (x y : ℝ) (m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hineq : ∀ x > 0, ∀ y > 0, 2 * y / x + 8 * x / y ≥ m^2 + 2 * m) : 
  -4 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l2258_225871


namespace total_people_large_seats_is_84_l2258_225816

-- Definition of the number of large seats
def large_seats : Nat := 7

-- Definition of the number of people each large seat can hold
def people_per_large_seat : Nat := 12

-- Definition of the total number of people that can ride on large seats
def total_people_large_seats : Nat := large_seats * people_per_large_seat

-- Statement that we need to prove
theorem total_people_large_seats_is_84 : total_people_large_seats = 84 := by
  sorry

end total_people_large_seats_is_84_l2258_225816


namespace not_valid_base_five_l2258_225895

theorem not_valid_base_five (k : ℕ) (h₁ : k = 5) : ¬(∀ d ∈ [3, 2, 5, 0, 1], d < k) :=
by
  sorry

end not_valid_base_five_l2258_225895


namespace find_ab_l2258_225876

theorem find_ab (a b c : ℕ) (H_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (H_b : b = 1) (H_ccb : (10 * a + b)^2 = 100 * c + 10 * c + b) (H_gt : 100 * c + 10 * c + b > 300) : (10 * a + b) = 21 :=
by
  sorry

end find_ab_l2258_225876


namespace cos_angle_plus_pi_over_two_l2258_225881

theorem cos_angle_plus_pi_over_two (α : ℝ) (h1 : Real.cos α = 1 / 5) (h2 : α ∈ Set.Icc (-2 * Real.pi) (-3 * Real.pi / 2) ∪ Set.Icc (0) (Real.pi / 2)) :
  Real.cos (α + Real.pi / 2) = 2 * Real.sqrt 6 / 5 :=
sorry

end cos_angle_plus_pi_over_two_l2258_225881


namespace monotone_increasing_solve_inequality_l2258_225879

section MathProblem

variable {f : ℝ → ℝ}

theorem monotone_increasing (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y) 
(h₂ : ∀ x : ℝ, 1 < x → 0 < f x) : 
∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := sorry

theorem solve_inequality (h₃ : f 2 = 1) (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y) 
(h₅ : ∀ x : ℝ, 1 < x → 0 < f x) :
∀ x : ℝ, 0 < x → f x + f (x - 3) ≤ 2 → 3 < x ∧ x ≤ 4 := sorry

end MathProblem

end monotone_increasing_solve_inequality_l2258_225879


namespace overall_average_tickets_sold_l2258_225831

variable {M : ℕ} -- number of male members
variable {F : ℕ} -- number of female members
variable (male_to_female_ratio : M * 2 = F) -- 1:2 ratio
variable (average_female : ℕ) (average_male : ℕ) -- average tickets sold by female/male members
variable (total_tickets_female : F * average_female = 70 * F) -- Total tickets sold by female members
variable (total_tickets_male : M * average_male = 58 * M) -- Total tickets sold by male members

-- The overall average number of raffle tickets sold per member is 66.
theorem overall_average_tickets_sold 
  (h1 : 70 * F + 58 * M = 198 * M) -- total tickets sold
  (h2 : M + F = 3 * M) -- total number of members
  : (70 * F + 58 * M) / (M + F) = 66 := by
  sorry

end overall_average_tickets_sold_l2258_225831


namespace ratio_of_b_to_a_l2258_225827

open Real

theorem ratio_of_b_to_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a * sin (π / 5) + b * cos (π / 5)) / (a * cos (π / 5) - b * sin (π / 5)) = tan (8 * π / 15) 
  → b / a = sqrt 3 :=
by
  intro h
  sorry

end ratio_of_b_to_a_l2258_225827


namespace black_white_ratio_l2258_225866

theorem black_white_ratio :
  let original_black := 18
  let original_white := 39
  let replaced_black := original_black + 13
  let inner_border_black := (9^2 - 7^2)
  let outer_border_white := (11^2 - 9^2)
  let total_black := replaced_black + inner_border_black
  let total_white := original_white + outer_border_white
  let ratio_black_white := total_black / total_white
  ratio_black_white = 63 / 79 :=
sorry

end black_white_ratio_l2258_225866


namespace jerry_total_games_l2258_225823

-- Conditions
def initial_games : ℕ := 7
def birthday_games : ℕ := 2

-- Statement
theorem jerry_total_games : initial_games + birthday_games = 9 := by sorry

end jerry_total_games_l2258_225823


namespace each_worker_paid_40_l2258_225840

variable (n_orchids : ℕ) (price_per_orchid : ℕ)
variable (n_money_plants : ℕ) (price_per_money_plant : ℕ)
variable (new_pots_cost : ℕ) (leftover_money : ℕ)
variable (n_workers : ℕ)

noncomputable def total_earnings : ℤ :=
  n_orchids * price_per_orchid + n_money_plants * price_per_money_plant

noncomputable def total_spent : ℤ :=
  new_pots_cost + leftover_money

noncomputable def amount_paid_to_workers : ℤ :=
  total_earnings n_orchids price_per_orchid n_money_plants price_per_money_plant - 
  total_spent new_pots_cost leftover_money

noncomputable def amount_paid_to_each_worker : ℤ :=
  amount_paid_to_workers n_orchids price_per_orchid n_money_plants price_per_money_plant 
    new_pots_cost leftover_money / n_workers

theorem each_worker_paid_40 :
  amount_paid_to_each_worker 20 50 15 25 150 1145 2 = 40 := by
  sorry

end each_worker_paid_40_l2258_225840


namespace range_of_values_for_a_l2258_225810

theorem range_of_values_for_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ x1^2 + a * x1 + a^2 - 1 = 0 ∧ x2^2 + a * x2 + a^2 - 1 = 0) → (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_values_for_a_l2258_225810


namespace total_cookies_collected_l2258_225861

theorem total_cookies_collected 
  (abigail_boxes : ℕ) (grayson_boxes : ℕ) (olivia_boxes : ℕ) (cookies_per_box : ℕ)
  (h1 : abigail_boxes = 2) (h2 : grayson_boxes = 3) (h3 : olivia_boxes = 3) (h4 : cookies_per_box = 48) :
  (abigail_boxes * cookies_per_box) + ((grayson_boxes * (cookies_per_box / 4))) + (olivia_boxes * cookies_per_box) = 276 := 
by 
  sorry

end total_cookies_collected_l2258_225861


namespace num_values_f100_eq_0_l2258_225883

def f0 (x : ℝ) : ℝ := x + |x - 100| - |x + 100|

def fn : ℕ → ℝ → ℝ
| 0, x   => f0 x
| (n+1), x => |fn n x| - 1

theorem num_values_f100_eq_0 : ∃ (xs : Finset ℝ), ∀ x ∈ xs, fn 100 x = 0 ∧ xs.card = 301 :=
by
  sorry

end num_values_f100_eq_0_l2258_225883


namespace popsicle_sticks_l2258_225897

theorem popsicle_sticks (total_sticks : ℕ) (gino_sticks : ℕ) (my_sticks : ℕ) 
  (h1 : total_sticks = 113) (h2 : gino_sticks = 63) (h3 : total_sticks = gino_sticks + my_sticks) : 
  my_sticks = 50 :=
  sorry

end popsicle_sticks_l2258_225897


namespace friends_pay_6_22_l2258_225889

noncomputable def cost_per_friend : ℕ :=
  let hamburgers := 5 * 3
  let fries := 4 * 120 / 100
  let soda := 5 * 50 / 100
  let spaghetti := 270 / 100
  let milkshakes := 3 * 250 / 100
  let nuggets := 2 * 350 / 100
  let total_bill := hamburgers + fries + soda + spaghetti + milkshakes + nuggets
  let discount := total_bill * 10 / 100
  let discounted_bill := total_bill - discount
  let birthday_friend := discounted_bill * 30 / 100
  let remaining_amount := discounted_bill - birthday_friend
  remaining_amount / 4

theorem friends_pay_6_22 : cost_per_friend = 622 / 100 :=
by
  sorry

end friends_pay_6_22_l2258_225889


namespace find_x_l2258_225857

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 9^n - 1) (h2 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 ≥ 2 ∧ p2 ≥ 2 ∧ p3 ≥ 2 ∧ x = p1 * p2 * p3 ∧ (p1 = 11 ∨ p2 = 11 ∨ p3 = 11)) :
    x = 59048 := 
sorry

end find_x_l2258_225857


namespace gcd_consecutive_odd_product_l2258_225825

theorem gcd_consecutive_odd_product (n : ℕ) (hn : n % 2 = 0 ∧ n > 0) : 
  Nat.gcd ((n+1)*(n+3)*(n+7)*(n+9)) 15 = 15 := 
sorry

end gcd_consecutive_odd_product_l2258_225825


namespace jaya_rank_from_bottom_l2258_225813

theorem jaya_rank_from_bottom (n t : ℕ) (h_n : n = 53) (h_t : t = 5) : n - t + 1 = 50 := by
  sorry

end jaya_rank_from_bottom_l2258_225813


namespace solution_x_percentage_of_alcohol_l2258_225874

variable (P : ℝ) -- percentage of alcohol by volume in solution x, in decimal form

theorem solution_x_percentage_of_alcohol :
  (0.30 : ℝ) * 200 + P * 200 = 0.20 * 400 → P = 0.10 :=
by
  intro h
  sorry

end solution_x_percentage_of_alcohol_l2258_225874


namespace margin_expression_l2258_225896

variable (n : ℕ) (C S M : ℝ)

theorem margin_expression (H1 : M = (1 / n) * C) (H2 : C = S - M) : 
  M = (1 / (n + 1)) * S := 
by
  sorry

end margin_expression_l2258_225896


namespace jordan_rectangle_width_l2258_225865

theorem jordan_rectangle_width
  (w : ℝ)
  (len_carol : ℝ := 5)
  (wid_carol : ℝ := 24)
  (len_jordan : ℝ := 12)
  (area_carol_eq_area_jordan : (len_carol * wid_carol) = (len_jordan * w)) :
  w = 10 := by
  sorry

end jordan_rectangle_width_l2258_225865


namespace middle_number_of_pairs_l2258_225852

theorem middle_number_of_pairs (x y z : ℕ) (h1 : x + y = 15) (h2 : x + z = 18) (h3 : y + z = 21) : y = 9 := 
by
  sorry

end middle_number_of_pairs_l2258_225852


namespace fraction_numerator_l2258_225838

theorem fraction_numerator (x : ℚ) 
  (h1 : ∃ (n : ℚ), n = 4 * x - 9) 
  (h2 : x / (4 * x - 9) = 3 / 4) 
  : x = 27 / 8 := sorry

end fraction_numerator_l2258_225838


namespace maximum_k_l2258_225826

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

-- Prove that the maximum integer value k satisfying k(x - 2) < f(x) for all x > 2 is 4.
theorem maximum_k (x : ℝ) (hx : x > 2) : ∃ k : ℤ, k = 4 ∧ (∀ x > 2, k * (x - 2) < f x) :=
sorry

end maximum_k_l2258_225826


namespace area_of_trapezoid_l2258_225887

-- Definitions of geometric properties and conditions
def is_perpendicular (a b c : ℝ) : Prop := a + b = 90 -- representing ∠ABC = 90°
def tangent_length (bc ad : ℝ) (O : ℝ) : Prop := bc * ad = O -- representing BC tangent to O with diameter AD
def is_diameter (ad r : ℝ) : Prop := ad = 2 * r -- AD being the diameter of the circle with radius r

-- Given conditions in the problem
variables (AB BC CD AD r O : ℝ) (h1 : is_perpendicular AB BC 90) (h2 : is_perpendicular BC CD 90)
          (h3 : tangent_length BC AD O) (h4 : is_diameter AD r) (h5 : BC = 2 * CD)
          (h6 : AB = 9) (h7 : CD = 3)

-- Statement to prove the area is 36
theorem area_of_trapezoid : (AB + CD) * CD = 36 := by
  sorry

end area_of_trapezoid_l2258_225887


namespace mouse_lives_correct_l2258_225814

def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7

theorem mouse_lives_correct : mouse_lives = 13 :=
by
  sorry

end mouse_lives_correct_l2258_225814


namespace inscribed_circle_radius_l2258_225804

theorem inscribed_circle_radius (a r : ℝ) (unit_square : a = 1)
  (touches_arc_AC : ∀ (x : ℝ × ℝ), x.1^2 + x.2^2 = (a - r)^2)
  (touches_arc_BD : ∀ (y : ℝ × ℝ), y.1^2 + y.2^2 = (a - r)^2)
  (touches_side_AB : ∀ (z : ℝ × ℝ), z.1 = r ∨ z.2 = r) :
  r = 3 / 8 := by sorry

end inscribed_circle_radius_l2258_225804


namespace smallest_number_of_rectangles_l2258_225803

-- Defining the given problem conditions
def rectangle_area : ℕ := 3 * 4
def smallest_square_side_length : ℕ := 12

-- Lean 4 statement to prove the problem
theorem smallest_number_of_rectangles 
    (h : ∃ n : ℕ, n * n = smallest_square_side_length * smallest_square_side_length)
    (h1 : ∃ m : ℕ, m * rectangle_area = smallest_square_side_length * smallest_square_side_length) :
    m = 9 :=
by
  sorry

end smallest_number_of_rectangles_l2258_225803


namespace fraction_zero_when_x_is_three_l2258_225863

theorem fraction_zero_when_x_is_three (x : ℝ) (h : x ≠ -3) : (x^2 - 9) / (x + 3) = 0 ↔ x = 3 :=
by 
  sorry

end fraction_zero_when_x_is_three_l2258_225863


namespace param_line_segment_l2258_225850

theorem param_line_segment:
  ∃ (a b c d : ℤ), b = 1 ∧ d = -3 ∧ a + b = -4 ∧ c + d = 9 ∧ a^2 + b^2 + c^2 + d^2 = 179 :=
by
  -- Here, you can use sorry to indicate that proof steps are not required as requested
  sorry

end param_line_segment_l2258_225850


namespace maximumNumberOfGirls_l2258_225849

theorem maximumNumberOfGirls {B : Finset ℕ} (hB : B.card = 5) :
  ∃ G : Finset ℕ, ∀ g ∈ G, ∃ b1 b2 : ℕ, b1 ≠ b2 ∧ b1 ∈ B ∧ b2 ∈ B ∧ dist g b1 = 5 ∧ dist g b2 = 5 ∧ G.card = 20 :=
sorry

end maximumNumberOfGirls_l2258_225849


namespace original_population_960_l2258_225839

variable (original_population : ℝ)

def new_population_increased := original_population + 800
def new_population_decreased := 0.85 * new_population_increased original_population

theorem original_population_960 
  (h1: new_population_decreased original_population = new_population_increased original_population + 24) :
  original_population = 960 := 
by
  -- here comes the proof, but we are omitting it as per the instructions
  sorry

end original_population_960_l2258_225839


namespace benedict_house_size_l2258_225892

variable (K B : ℕ)

theorem benedict_house_size
    (h1 : K = 4 * B + 600)
    (h2 : K = 10000) : B = 2350 := by
sorry

end benedict_house_size_l2258_225892


namespace max_books_john_can_buy_l2258_225806

-- Define the key variables and conditions
def johns_money : ℕ := 3745
def book_cost : ℕ := 285
def sales_tax_rate : ℚ := 0.05

-- Define the total cost per book including tax
def total_cost_per_book : ℝ := book_cost + book_cost * sales_tax_rate

-- Define the inequality problem
theorem max_books_john_can_buy : ∃ (x : ℕ), 300 * x ≤ johns_money ∧ 300 * (x + 1) > johns_money :=
by
  sorry

end max_books_john_can_buy_l2258_225806


namespace eval_fraction_expr_l2258_225837

theorem eval_fraction_expr :
  (2 ^ 2010 * 3 ^ 2012) / (6 ^ 2011) = 3 / 2 := 
sorry

end eval_fraction_expr_l2258_225837


namespace perimeter_of_square_l2258_225877

theorem perimeter_of_square (a : Real) (h_a : a ^ 2 = 144) : 4 * a = 48 :=
by
  sorry

end perimeter_of_square_l2258_225877


namespace fraction_unclaimed_l2258_225856

def exists_fraction_unclaimed (x : ℕ) : Prop :=
  let claimed_by_Eva := (1 / 2 : ℚ) * x
  let remaining_after_Eva := x - claimed_by_Eva
  let claimed_by_Liam := (3 / 8 : ℚ) * x
  let remaining_after_Liam := remaining_after_Eva - claimed_by_Liam
  let claimed_by_Noah := (1 / 8 : ℚ) * remaining_after_Eva
  let remaining_after_Noah := remaining_after_Liam - claimed_by_Noah
  remaining_after_Noah / x = (75 / 128 : ℚ)

theorem fraction_unclaimed {x : ℕ} : exists_fraction_unclaimed x :=
by
  sorry

end fraction_unclaimed_l2258_225856


namespace isosceles_triangle_sin_cos_rational_l2258_225864

theorem isosceles_triangle_sin_cos_rational
  (a h : ℤ) -- Given BC and AD as integers
  (c : ℚ)  -- AB = AC = c
  (ha : 4 * c^2 = 4 * h^2 + a^2) : -- From c^2 = h^2 + (a^2 / 4)
  ∃ (sinA cosA : ℚ), 
    sinA = (a * h) / (h^2 + (a^2 / 4)) ∧
    cosA = (2 * h^2) / (h^2 + (a^2 / 4)) - 1 :=
sorry

end isosceles_triangle_sin_cos_rational_l2258_225864


namespace binomial_square_evaluation_l2258_225802

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l2258_225802


namespace solve_n_m_l2258_225898

noncomputable def exponents_of_linear_equation (n m : ℕ) (x y : ℝ) : Prop :=
2 * x ^ (n - 3) - (1 / 3) * y ^ (2 * m + 1) = 0

theorem solve_n_m (n m : ℕ) (x y : ℝ) (h_linear : exponents_of_linear_equation n m x y) :
  n ^ m = 1 :=
sorry

end solve_n_m_l2258_225898


namespace parallelogram_area_l2258_225878

theorem parallelogram_area (base height : ℝ) (h_base : base = 12) (h_height : height = 10) :
  base * height = 120 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l2258_225878


namespace ayen_total_jog_time_l2258_225872

def jog_time_weekday : ℕ := 30
def jog_time_tuesday : ℕ := jog_time_weekday + 5
def jog_time_friday : ℕ := jog_time_weekday + 25

def total_weekday_jog_time : ℕ := jog_time_weekday * 3
def total_jog_time : ℕ := total_weekday_jog_time + jog_time_tuesday + jog_time_friday

theorem ayen_total_jog_time : total_jog_time / 60 = 3 := by
  sorry

end ayen_total_jog_time_l2258_225872


namespace neg_p_sufficient_not_necessary_for_neg_q_l2258_225890

noncomputable def p (x : ℝ) : Prop := abs (x + 1) > 0
noncomputable def q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬ p x → ¬ q x) ∧ ¬ (∀ x, ¬ q x → ¬ p x) :=
by
  sorry

end neg_p_sufficient_not_necessary_for_neg_q_l2258_225890


namespace jack_can_return_3900_dollars_l2258_225848

/-- Jack's Initial Gift Card Values and Counts --/
def best_buy_card_value : ℕ := 500
def walmart_card_value : ℕ := 200
def initial_best_buy_cards : ℕ := 6
def initial_walmart_cards : ℕ := 9

/-- Jack's Sent Gift Card Counts --/
def sent_best_buy_cards : ℕ := 1
def sent_walmart_cards : ℕ := 2

/-- Calculate the remaining dollar value of Jack's gift cards. --/
def remaining_gift_cards_value : ℕ := 
  (initial_best_buy_cards * best_buy_card_value - sent_best_buy_cards * best_buy_card_value) +
  (initial_walmart_cards * walmart_card_value - sent_walmart_cards * walmart_card_value)

/-- Proving the remaining value of gift cards Jack can return is $3900. --/
theorem jack_can_return_3900_dollars : remaining_gift_cards_value = 3900 := by
  sorry

end jack_can_return_3900_dollars_l2258_225848


namespace number_of_students_taking_french_l2258_225846

def total_students : ℕ := 79
def students_taking_german : ℕ := 22
def students_taking_both : ℕ := 9
def students_not_enrolled_in_either : ℕ := 25

theorem number_of_students_taking_french :
  ∃ F : ℕ, (total_students = F + students_taking_german - students_taking_both + students_not_enrolled_in_either) ∧ F = 41 :=
by
  sorry

end number_of_students_taking_french_l2258_225846


namespace ratio_of_areas_l2258_225858

theorem ratio_of_areas (r s_3 s_2 : ℝ) (h1 : s_3^2 = r^2) (h2 : s_2^2 = 2 * r^2) :
  (s_3^2 / s_2^2) = 1 / 2 := by
  sorry

end ratio_of_areas_l2258_225858


namespace find_m_l2258_225824

noncomputable def f (x m : ℝ) : ℝ := (x^2 + m*x) * Real.exp x

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem find_m (m : ℝ) :
  is_monotonically_decreasing (f (m := m)) (-3/2) 1 ∧
  (-3/2)^2 + (m + 2)*(-3/2) + m = 0 ∧
  1^2 + (m + 2)*1 + m = 0 →
  m = -3/2 :=
by
  sorry

end find_m_l2258_225824


namespace beautiful_ratio_l2258_225844

theorem beautiful_ratio (A B C : Type) (l1 l2 b : ℕ) 
  (h : l1 + l2 + b = 20) (h1 : l1 = 8 ∨ l2 = 8 ∨ b = 8) :
  (b / l1 = 1/2) ∨ (b / l2 = 1/2) ∨ (l1 / l2 = 4/3) ∨ (l2 / l1 = 4/3) :=
by
  sorry

end beautiful_ratio_l2258_225844


namespace annual_interest_rate_l2258_225841

theorem annual_interest_rate
  (principal : ℝ) (monthly_payment : ℝ) (months : ℕ)
  (H1 : principal = 150) (H2 : monthly_payment = 13) (H3 : months = 12) :
  (monthly_payment * months - principal) / principal * 100 = 4 :=
by
  sorry

end annual_interest_rate_l2258_225841
