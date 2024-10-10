import Mathlib

namespace quadratic_factorization_l3928_392858

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 6 * x - 8 = 2 * (x - 4) * (x + 1) := by
  sorry

end quadratic_factorization_l3928_392858


namespace sin_sum_l3928_392853

theorem sin_sum (α β : ℝ) : Real.sin (α + β) = Real.sin α * Real.cos β + Real.cos α * Real.sin β := by
  sorry

end sin_sum_l3928_392853


namespace count_numbers_with_nine_is_1848_l3928_392809

/-- Counts the number of integers between 1000 and 9000 with four distinct digits, including at least one '9' -/
def count_numbers_with_nine : ℕ := 
  let first_digit_nine := 9 * 8 * 7
  let nine_in_other_positions := 3 * 8 * 8 * 7
  first_digit_nine + nine_in_other_positions

/-- Theorem stating that the count of integers between 1000 and 9000 
    with four distinct digits, including at least one '9', is 1848 -/
theorem count_numbers_with_nine_is_1848 : 
  count_numbers_with_nine = 1848 := by sorry

end count_numbers_with_nine_is_1848_l3928_392809


namespace peters_height_is_96_inches_l3928_392826

/-- Given a tree height, tree shadow length, and Peter's shadow length,
    calculate Peter's height in inches. -/
def peters_height_inches (tree_height foot_to_inch : ℕ) 
                         (tree_shadow peter_shadow : ℚ) : ℚ :=
  (tree_height : ℚ) / tree_shadow * peter_shadow * foot_to_inch

/-- Theorem stating that Peter's height is 96 inches given the problem conditions. -/
theorem peters_height_is_96_inches :
  peters_height_inches 60 12 15 2 = 96 := by
  sorry

#eval peters_height_inches 60 12 15 2

end peters_height_is_96_inches_l3928_392826


namespace bakery_flour_usage_l3928_392844

theorem bakery_flour_usage :
  0.2 + 0.1 + 0.15 + 0.05 + 0.1 = 0.6 := by
  sorry

end bakery_flour_usage_l3928_392844


namespace right_isosceles_triangle_median_area_l3928_392836

theorem right_isosceles_triangle_median_area (h : ℝ) :
  h > 0 →
  let leg := h / Real.sqrt 2
  let area := (1 / 2) * leg * leg
  let median_area := area / 2
  (h = 16) → median_area = 32 := by sorry

end right_isosceles_triangle_median_area_l3928_392836


namespace senior_mean_score_l3928_392899

theorem senior_mean_score 
  (total_students : ℕ) 
  (overall_mean : ℝ) 
  (senior_count : ℕ) 
  (non_senior_count : ℕ) 
  (h1 : total_students = 200)
  (h2 : overall_mean = 120)
  (h3 : non_senior_count = 2 * senior_count)
  (h4 : total_students = senior_count + non_senior_count)
  (h5 : senior_count > 0)
  (h6 : non_senior_count > 0) :
  ∃ (senior_mean non_senior_mean : ℝ),
    non_senior_mean = 0.8 * senior_mean ∧
    (senior_count : ℝ) * senior_mean + (non_senior_count : ℝ) * non_senior_mean = (total_students : ℝ) * overall_mean ∧
    senior_mean = 138 := by
  sorry


end senior_mean_score_l3928_392899


namespace not_divisible_by_n_plus_4_l3928_392847

theorem not_divisible_by_n_plus_4 (n : ℕ+) : ¬ ∃ k : ℤ, (n.val^2 + 8*n.val + 15 : ℤ) = k * (n.val + 4) := by
  sorry

end not_divisible_by_n_plus_4_l3928_392847


namespace katherines_fruit_ratio_l3928_392831

/-- Katherine's fruit problem -/
theorem katherines_fruit_ratio : ∀ (pears apples bananas : ℕ),
  apples = 4 →
  bananas = 5 →
  pears + apples + bananas = 21 →
  pears / apples = 3 :=
by
  sorry

#check katherines_fruit_ratio

end katherines_fruit_ratio_l3928_392831


namespace ab_value_l3928_392887

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 39) : a * b = 18 := by
  sorry

end ab_value_l3928_392887


namespace pencil_length_l3928_392817

/-- Prove that given the conditions of the pen, rubber, and pencil lengths, the pencil is 12 cm long -/
theorem pencil_length (rubber pen pencil : ℝ) 
  (pen_rubber : pen = rubber + 3)
  (pencil_pen : pencil = pen + 2)
  (total_length : rubber + pen + pencil = 29) :
  pencil = 12 := by
  sorry

end pencil_length_l3928_392817


namespace last_defective_on_fifth_draw_l3928_392864

def number_of_arrangements (n_total : ℕ) (n_genuine : ℕ) (n_defective : ℕ) : ℕ :=
  (n_total.choose (n_defective - 1)) * (n_defective.factorial) * n_genuine

theorem last_defective_on_fifth_draw :
  let n_total := 9
  let n_genuine := 5
  let n_defective := 4
  let n_draws := 5
  number_of_arrangements n_draws n_genuine n_defective = 480 :=
by sorry

end last_defective_on_fifth_draw_l3928_392864


namespace next_number_is_two_l3928_392819

-- Define the sequence pattern
def sequence_pattern (n : ℕ) : ℕ → ℕ
| 0 => 1
| m + 1 => 
  let peak := n + 1
  let cycle_length := 2 * peak - 1
  let position := (m + 1) % cycle_length
  if position < peak then position + 1
  else 2 * peak - position - 1

-- Define the specific sequence from the problem
def given_sequence : List ℕ := [1, 1, 2, 1, 2, 3, 2, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 4, 2, 1, 2, 3, 4, 5, 6, 5, 3, 1, 2, 3, 4, 5, 6, 7, 6, 4, 2, 1, 2, 3, 4, 5, 6, 7, 8, 7, 5, 3, 1]

-- Theorem to prove
theorem next_number_is_two : 
  ∃ (n : ℕ), sequence_pattern n (given_sequence.length) = 2 :=
by sorry

end next_number_is_two_l3928_392819


namespace P_sufficient_not_necessary_l3928_392872

def condition_P (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 0

def condition_Q (x y : ℝ) : Prop := (x - 1) * (y - 1) = 0

theorem P_sufficient_not_necessary :
  (∀ x y : ℝ, condition_P x y → condition_Q x y) ∧
  ¬(∀ x y : ℝ, condition_Q x y → condition_P x y) := by
  sorry

end P_sufficient_not_necessary_l3928_392872


namespace largest_base7_3digit_in_decimal_l3928_392880

/-- The largest three-digit number in base 7 -/
def largest_base7_3digit : ℕ := 6 * 7^2 + 6 * 7^1 + 6 * 7^0

/-- Converts a base 7 number to decimal -/
def base7_to_decimal (n : ℕ) : ℕ := n

theorem largest_base7_3digit_in_decimal :
  base7_to_decimal largest_base7_3digit = 342 := by sorry

end largest_base7_3digit_in_decimal_l3928_392880


namespace notebook_distribution_l3928_392825

theorem notebook_distribution (C : ℕ) (N : ℕ) : 
  (N / C = C / 8) → 
  (N = 16 * (C / 2)) → 
  (N = 512) :=
by
  sorry

end notebook_distribution_l3928_392825


namespace age_difference_is_nine_l3928_392822

/-- The age difference between Bella's brother and Bella -/
def ageDifference (bellasAge : ℕ) (totalAge : ℕ) : ℕ :=
  totalAge - bellasAge - bellasAge

/-- Proof that the age difference is 9 years -/
theorem age_difference_is_nine :
  ageDifference 5 19 = 9 := by
  sorry

end age_difference_is_nine_l3928_392822


namespace expression_equality_l3928_392857

/-- Proof that the given expression K is equal to 80xyz(x^2 + y^2 + z^2) -/
theorem expression_equality (x y z : ℝ) :
  (x + y + z)^5 - (-x + y + z)^5 - (x - y + z)^5 - (x + y - z)^5 = 80 * x * y * z * (x^2 + y^2 + z^2) := by
  sorry

end expression_equality_l3928_392857


namespace f_maps_neg_two_three_to_one_neg_six_l3928_392896

/-- The mapping f that transforms a point (x, y) to (x+y, xy) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

/-- Theorem stating that f maps (-2, 3) to (1, -6) -/
theorem f_maps_neg_two_three_to_one_neg_six :
  f (-2, 3) = (1, -6) := by sorry

end f_maps_neg_two_three_to_one_neg_six_l3928_392896


namespace complex_magnitude_special_angle_l3928_392854

theorem complex_magnitude_special_angle : 
  let z : ℂ := Complex.mk (Real.sin (π / 3)) (Real.cos (π / 6))
  ‖z‖ = 1 := by
  sorry

end complex_magnitude_special_angle_l3928_392854


namespace farmer_land_calculation_l3928_392875

theorem farmer_land_calculation (total_land : ℝ) : 
  (0.05 * 0.9 * total_land + 0.05 * 0.9 * total_land = 90) → total_land = 1000 :=
by
  sorry

end farmer_land_calculation_l3928_392875


namespace sum_of_complex_roots_l3928_392889

theorem sum_of_complex_roots (a₁ a₂ a₃ : ℂ)
  (h1 : a₁^2 + a₂^2 + a₃^2 = 0)
  (h2 : a₁^3 + a₂^3 + a₃^3 = 0)
  (h3 : a₁^4 + a₂^4 + a₃^4 = 0) :
  a₁ + a₂ + a₃ = 0 := by
sorry

end sum_of_complex_roots_l3928_392889


namespace cos_squared_half_diff_l3928_392829

theorem cos_squared_half_diff (α β : Real) 
  (h1 : Real.sin α + Real.sin β = Real.sqrt 6 / 3)
  (h2 : Real.cos α + Real.cos β = Real.sqrt 3 / 3) : 
  (Real.cos ((α - β) / 2))^2 = 1/4 := by
sorry

end cos_squared_half_diff_l3928_392829


namespace octal_subtraction_correct_l3928_392837

/-- Represents a number in base 8 -/
def OctalNumber := List Nat

/-- Converts a list of digits in base 8 to a natural number -/
def octal_to_nat (x : OctalNumber) : Nat :=
  x.foldr (fun digit acc => acc * 8 + digit) 0

/-- Subtracts two octal numbers -/
def octal_subtract (x y : OctalNumber) : OctalNumber :=
  sorry -- Implementation of octal subtraction

theorem octal_subtraction_correct :
  octal_subtract [7, 3, 2, 4] [3, 6, 5, 7] = [4, 4, 4, 5] :=
by sorry

end octal_subtraction_correct_l3928_392837


namespace degree_to_radian_conversion_l3928_392820

theorem degree_to_radian_conversion (π : Real) :
  (180 : Real) * (π / 3) = 60 * π :=
by sorry

end degree_to_radian_conversion_l3928_392820


namespace perpendicular_vectors_x_value_l3928_392886

-- Define vectors a and b
def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (2, -1)

-- Define the perpendicularity condition
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Theorem statement
theorem perpendicular_vectors_x_value :
  ∃ x : ℝ, is_perpendicular (a.1 - x * b.1, a.2 - x * b.2) (a.1 - b.1, a.2 - b.2) ∧ x = -7/3 :=
sorry

end perpendicular_vectors_x_value_l3928_392886


namespace install_time_proof_l3928_392856

/-- The time required to install the remaining windows -/
def time_to_install_remaining (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ) : ℕ :=
  (total_windows - installed_windows) * time_per_window

/-- Proof that the time to install remaining windows is 36 hours -/
theorem install_time_proof (total_windows : ℕ) (installed_windows : ℕ) (time_per_window : ℕ)
  (h1 : total_windows = 14)
  (h2 : installed_windows = 5)
  (h3 : time_per_window = 4) :
  time_to_install_remaining total_windows installed_windows time_per_window = 36 :=
by sorry

end install_time_proof_l3928_392856


namespace trigonometric_identity_l3928_392871

theorem trigonometric_identity :
  3 * Real.tan (10 * π / 180) + 4 * Real.sqrt 3 * Real.sin (10 * π / 180) = Real.sqrt 3 := by
  sorry

end trigonometric_identity_l3928_392871


namespace max_bishops_on_mountain_board_l3928_392870

/-- A chessboard with two mountains --/
structure MountainChessboard :=
  (black_regions : ℕ)
  (white_regions : ℕ)

/-- The maximum number of non-attacking bishops on a mountain chessboard --/
def max_bishops (board : MountainChessboard) : ℕ :=
  board.black_regions + board.white_regions

/-- Theorem: The maximum number of non-attacking bishops on the given mountain chessboard is 19 --/
theorem max_bishops_on_mountain_board :
  ∃ (board : MountainChessboard), 
    board.black_regions = 11 ∧ 
    board.white_regions = 8 ∧ 
    max_bishops board = 19 := by
  sorry

#eval max_bishops ⟨11, 8⟩

end max_bishops_on_mountain_board_l3928_392870


namespace red_marbles_fraction_l3928_392838

theorem red_marbles_fraction (total : ℚ) (h : total > 0) : 
  let initial_blue := (2 / 3) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3 / 5 := by sorry

end red_marbles_fraction_l3928_392838


namespace negative_expression_l3928_392885

/-- Given real numbers U, V, W, X, and Y with the following properties:
    U and W are negative,
    V and Y are positive,
    X is near zero (small in absolute value),
    prove that U - V is negative. -/
theorem negative_expression (U V W X Y : ℝ) 
  (hU : U < 0) (hW : W < 0) 
  (hV : V > 0) (hY : Y > 0) 
  (hX : ∃ ε > 0, abs X < ε ∧ ε < 1) : 
  U - V < 0 := by
  sorry

end negative_expression_l3928_392885


namespace school_play_tickets_l3928_392810

theorem school_play_tickets (student_price adult_price adult_count total : ℕ) 
  (h1 : student_price = 6)
  (h2 : adult_price = 8)
  (h3 : adult_count = 12)
  (h4 : total = 216) :
  ∃ student_count : ℕ, student_count * student_price + adult_count * adult_price = total ∧ student_count = 20 := by
  sorry

end school_play_tickets_l3928_392810


namespace secretary_project_hours_l3928_392883

theorem secretary_project_hours (total_hours : ℕ) (ratio1 ratio2 ratio3 : ℕ) 
  (h1 : ratio1 = 3) (h2 : ratio2 = 7) (h3 : ratio3 = 13) 
  (h_total : total_hours = 253) 
  (h_ratio : ratio1 + ratio2 + ratio3 > 0) :
  (ratio3 * total_hours) / (ratio1 + ratio2 + ratio3) = 143 := by
  sorry

end secretary_project_hours_l3928_392883


namespace chromium_54_neutrons_l3928_392816

/-- The number of neutrons in an atom of chromium-54 -/
def neutrons_per_atom : ℕ := 54 - 24

/-- Avogadro's constant (atoms per mole) -/
def avogadro : ℝ := 6.022e23

/-- Amount of substance in moles -/
def amount : ℝ := 0.025

/-- Approximate number of neutrons in the given amount of chromium-54 -/
def total_neutrons : ℝ := amount * avogadro * neutrons_per_atom

theorem chromium_54_neutrons : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1e23 ∧ |total_neutrons - 4.5e23| < ε :=
sorry

end chromium_54_neutrons_l3928_392816


namespace f_continuous_iff_a_eq_one_l3928_392833

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 0 then Real.exp (3 * x) else a + 5 * x

theorem f_continuous_iff_a_eq_one (a : ℝ) :
  Continuous (f a) ↔ a = 1 := by sorry

end f_continuous_iff_a_eq_one_l3928_392833


namespace additional_distance_for_target_average_speed_l3928_392840

/-- Proves that given an initial trip of 20 miles at 40 mph, an additional 90 miles
    driven at 60 mph will result in an average speed of 55 mph for the entire trip. -/
theorem additional_distance_for_target_average_speed
  (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_avg_speed : ℝ)
  (additional_distance : ℝ) :
  initial_distance = 20 →
  initial_speed = 40 →
  second_speed = 60 →
  target_avg_speed = 55 →
  additional_distance = 90 →
  (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = target_avg_speed :=
by sorry

end additional_distance_for_target_average_speed_l3928_392840


namespace minimum_heat_for_piston_ejection_l3928_392827

/-- The minimum amount of heat required to shoot a piston out of a cylinder -/
theorem minimum_heat_for_piston_ejection
  (l₁ : Real) (l₂ : Real) (M : Real) (S : Real) (v : Real) (p₀ : Real) (g : Real)
  (h₁ : l₁ = 0.1) -- 10 cm in meters
  (h₂ : l₂ = 0.15) -- 15 cm in meters
  (h₃ : M = 10) -- 10 kg
  (h₄ : S = 0.001) -- 10 cm² in m²
  (h₅ : v = 1) -- 1 mole
  (h₆ : p₀ = 10^5) -- 10⁵ Pa
  (h₇ : g = 10) -- 10 m/s²
  : ∃ Q : Real, Q = 127.5 ∧ Q ≥ 0 := by
  sorry

end minimum_heat_for_piston_ejection_l3928_392827


namespace no_real_roots_for_sqrt_equation_l3928_392804

theorem no_real_roots_for_sqrt_equation :
  ¬ ∃ x : ℝ, Real.sqrt (x + 4) - Real.sqrt (x - 3) + 1 = 0 := by
  sorry

end no_real_roots_for_sqrt_equation_l3928_392804


namespace arctan_sum_right_triangle_l3928_392898

theorem arctan_sum_right_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a = 2 * b) :
  Real.arctan (b / a) + Real.arctan (a / b) = π / 2 := by
sorry

end arctan_sum_right_triangle_l3928_392898


namespace S_intersect_T_eq_T_l3928_392839

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end S_intersect_T_eq_T_l3928_392839


namespace kathleen_savings_problem_l3928_392869

/-- Kathleen's savings and expenses problem -/
theorem kathleen_savings_problem (june july august september : ℚ)
  (school_supplies clothes gift book donation : ℚ) :
  june = 21 →
  july = 46 →
  august = 45 →
  september = 32 →
  school_supplies = 12 →
  clothes = 54 →
  gift = 37 →
  book = 25 →
  donation = 10 →
  let october : ℚ := august / 2
  let november : ℚ := 2 * september - 20
  let total_savings : ℚ := june + july + august + september + october + november
  let total_expenses : ℚ := school_supplies + clothes + gift + book + donation
  let aunt_bonus : ℚ := if total_savings > 200 ∧ donation = 10 then 25 else 0
  total_savings - total_expenses + aunt_bonus = 97.5 := by
  sorry

end kathleen_savings_problem_l3928_392869


namespace new_average_after_changes_l3928_392874

def initial_count : ℕ := 60
def initial_average : ℚ := 40
def removed_number1 : ℕ := 50
def removed_number2 : ℕ := 60
def added_number : ℕ := 35

theorem new_average_after_changes :
  let initial_sum := initial_count * initial_average
  let sum_after_removal := initial_sum - (removed_number1 + removed_number2)
  let final_sum := sum_after_removal + added_number
  let final_count := initial_count - 1
  final_sum / final_count = 39.41 := by sorry

end new_average_after_changes_l3928_392874


namespace remaining_gift_card_value_l3928_392868

def bestBuyCardValue : ℕ := 500
def walmartCardValue : ℕ := 200

def initialBestBuyCards : ℕ := 6
def initialWalmartCards : ℕ := 9

def sentBestBuyCards : ℕ := 1
def sentWalmartCards : ℕ := 2

theorem remaining_gift_card_value :
  (initialBestBuyCards - sentBestBuyCards) * bestBuyCardValue +
  (initialWalmartCards - sentWalmartCards) * walmartCardValue = 3900 := by
  sorry

end remaining_gift_card_value_l3928_392868


namespace video_game_enemies_l3928_392818

/-- The number of points earned per enemy defeated -/
def points_per_enemy : ℕ := 5

/-- The number of enemies left undefeated -/
def enemies_left : ℕ := 6

/-- The total points earned when all but 6 enemies are defeated -/
def total_points : ℕ := 10

/-- The total number of enemies in the level -/
def total_enemies : ℕ := 8

theorem video_game_enemies :
  total_enemies = (total_points / points_per_enemy) + enemies_left := by
  sorry

end video_game_enemies_l3928_392818


namespace geometric_sequence_problem_l3928_392865

theorem geometric_sequence_problem (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  q > 0 ∧ 
  (∀ n, a (n + 1) = a n * q) ∧ 
  (∀ n, a n > 0) ∧
  (a 1 = 1 / q^2) ∧
  (S 5 = S 2 + 2) →
  q = (Real.sqrt 5 - 1) / 2 := by
sorry

end geometric_sequence_problem_l3928_392865


namespace abc_inequality_l3928_392824

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  9 * a * b * c ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c < 1/4 + 3 * a * b * c := by
  sorry

end abc_inequality_l3928_392824


namespace max_profit_at_initial_price_l3928_392850

/-- Represents the daily profit function for a clothing store -/
def daily_profit (x : ℝ) : ℝ :=
  (30 - x) * (30 + x)

/-- Theorem stating that the maximum daily profit occurs at the initial selling price -/
theorem max_profit_at_initial_price :
  ∀ x : ℝ, daily_profit 0 ≥ daily_profit x :=
sorry

end max_profit_at_initial_price_l3928_392850


namespace midpoint_path_and_intersection_l3928_392813

/-- The path C traced by the midpoint of PQ, where P is (0, 4) and Q moves on x^2 + y^2 = 8 -/
def path_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 2

/-- The line l that intersects path C -/
def line_l (k x y : ℝ) : Prop := y = k * x

/-- The condition for point E on line segment AB -/
def point_E_condition (m n : ℝ) (OA OB : ℝ) : Prop :=
  3 / (m^2 + n^2) = 1 / OA^2 + 1 / OB^2

/-- The main theorem -/
theorem midpoint_path_and_intersection :
  ∀ (x y k m n OA OB : ℝ),
  path_C x y →
  line_l k x y →
  point_E_condition m n OA OB →
  -Real.sqrt 6 / 2 < m →
  m < Real.sqrt 6 / 2 →
  m ≠ 0 →
  n = Real.sqrt (3 * m^2 + 9) / 3 :=
by sorry

end midpoint_path_and_intersection_l3928_392813


namespace rooks_attack_after_knight_moves_l3928_392812

/-- Represents a position on the chess board -/
structure Position :=
  (row : Fin 15)
  (col : Fin 15)

/-- Represents a knight's move -/
inductive KnightMove
  | move1 : KnightMove  -- represents +2,+1 or -2,-1
  | move2 : KnightMove  -- represents +2,-1 or -2,+1
  | move3 : KnightMove  -- represents +1,+2 or -1,-2
  | move4 : KnightMove  -- represents +1,-2 or -1,+2

/-- Applies a knight's move to a position -/
def applyKnightMove (p : Position) (m : KnightMove) : Position :=
  sorry

/-- Checks if two positions are in the same row or column -/
def sameRowOrColumn (p1 p2 : Position) : Prop :=
  p1.row = p2.row ∨ p1.col = p2.col

theorem rooks_attack_after_knight_moves 
  (initial_positions : Fin 15 → Position)
  (h_no_initial_attack : ∀ i j, i ≠ j → ¬(sameRowOrColumn (initial_positions i) (initial_positions j)))
  (moves : Fin 15 → KnightMove) :
  ∃ i j, i ≠ j ∧ sameRowOrColumn (applyKnightMove (initial_positions i) (moves i)) (applyKnightMove (initial_positions j) (moves j)) :=
sorry

end rooks_attack_after_knight_moves_l3928_392812


namespace initial_cookies_count_l3928_392807

/-- The number of cookies Paul took out in 4 days -/
def cookies_taken_4_days : ℕ := 24

/-- The number of days Paul took cookies out -/
def days_taken : ℕ := 4

/-- The number of cookies remaining after a week -/
def cookies_remaining : ℕ := 28

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Proves that the initial number of cookies in the jar is 52 -/
theorem initial_cookies_count : ℕ := by
  sorry

end initial_cookies_count_l3928_392807


namespace polynomial_ratio_condition_l3928_392834

/-- A polynomial f(x) = x^2 - α x + 1 can be expressed as a ratio of two polynomials
    with non-negative coefficients if and only if α < 2. -/
theorem polynomial_ratio_condition (α : ℝ) :
  (∃ (P Q : ℝ → ℝ), (∀ x, P x ≥ 0 ∧ Q x ≥ 0) ∧
    (∀ x, x^2 - α * x + 1 = P x / Q x)) ↔ α < 2 :=
sorry

end polynomial_ratio_condition_l3928_392834


namespace remaining_books_l3928_392876

/-- Given an initial number of books and a number of books sold,
    proves that the remaining number of books is equal to
    the difference between the initial number and the number sold. -/
theorem remaining_books (initial : ℕ) (sold : ℕ) (h : sold ≤ initial) :
  initial - sold = initial - sold :=
by sorry

end remaining_books_l3928_392876


namespace expressions_not_always_equal_l3928_392830

theorem expressions_not_always_equal :
  ∃ (a b c : ℝ), a + b + c = 0 ∧ a + b * c ≠ (a + b) * (a + c) := by
  sorry

end expressions_not_always_equal_l3928_392830


namespace subset_implies_a_values_l3928_392895

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem subset_implies_a_values (a : ℝ) : 
  B a ⊆ A → a ∈ ({-1/2, 1/3, 0} : Set ℝ) := by
  sorry

end subset_implies_a_values_l3928_392895


namespace quadratic_real_root_condition_l3928_392879

-- Define the quadratic equation
def quadratic_equation (s t x : ℝ) : Prop :=
  s * x^2 + t * x + s - 1 = 0

-- Define the existence of a real root
def has_real_root (s t : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation s t x

-- Main theorem
theorem quadratic_real_root_condition (s : ℝ) :
  (s ≠ 0 ∧ ∀ t : ℝ, has_real_root s t) ↔ (0 < s ∧ s ≤ 1) :=
sorry

end quadratic_real_root_condition_l3928_392879


namespace rectangle_length_equals_6_3_l3928_392835

-- Define the parameters
def triangle_base : ℝ := 7.2
def triangle_height : ℝ := 7
def rectangle_width : ℝ := 4

-- Define the theorem
theorem rectangle_length_equals_6_3 :
  let triangle_area := (triangle_base * triangle_height) / 2
  let rectangle_length := triangle_area / rectangle_width
  rectangle_length = 6.3 := by
  sorry

end rectangle_length_equals_6_3_l3928_392835


namespace simplify_complex_fraction_l3928_392862

theorem simplify_complex_fraction :
  (1 / ((1 / (Real.sqrt 5 + 2)) - (2 / (Real.sqrt 7 - 3)))) =
  ((Real.sqrt 5 + Real.sqrt 7 - 1) / (11 + 2 * Real.sqrt 35)) := by
  sorry

end simplify_complex_fraction_l3928_392862


namespace tank_capacity_l3928_392894

/-- Represents the capacity of a tank and its inlet/outlet pipes. -/
structure TankSystem where
  capacity : ℝ
  outlet_time : ℝ
  inlet_rate : ℝ
  combined_time : ℝ

/-- Theorem stating the capacity of the tank given the conditions. -/
theorem tank_capacity (t : TankSystem)
  (h1 : t.outlet_time = 5)
  (h2 : t.inlet_rate = 4 * 60)  -- 4 litres/min converted to litres/hour
  (h3 : t.combined_time = 8)
  : t.capacity = 3200 := by
  sorry

end tank_capacity_l3928_392894


namespace chocolate_candy_cost_difference_l3928_392815

/-- Calculates the cost difference between chocolates and candy bars --/
theorem chocolate_candy_cost_difference :
  let initial_money : ℚ := 50
  let candy_price : ℚ := 4
  let candy_discount_rate : ℚ := 0.2
  let candy_discount_threshold : ℕ := 3
  let candy_quantity : ℕ := 5
  let chocolate_price : ℚ := 6
  let chocolate_tax_rate : ℚ := 0.05
  let chocolate_quantity : ℕ := 4

  let candy_cost : ℚ := if candy_quantity ≥ candy_discount_threshold
    then candy_quantity * candy_price * (1 - candy_discount_rate)
    else candy_quantity * candy_price

  let chocolate_cost : ℚ := chocolate_quantity * chocolate_price * (1 + chocolate_tax_rate)

  chocolate_cost - candy_cost = 9.2 :=
by
  sorry


end chocolate_candy_cost_difference_l3928_392815


namespace circle_radii_theorem_l3928_392828

/-- The configuration of circles as described in the problem -/
structure CircleConfiguration where
  r : ℝ  -- radius of white circles
  red_radius : ℝ  -- radius of Adam's red circle
  green_radius : ℝ  -- radius of Eva's green circle

/-- The theorem stating the radii of the red and green circles -/
theorem circle_radii_theorem (config : CircleConfiguration) :
  config.red_radius = (Real.sqrt 2 - 1) * config.r ∧
  config.green_radius = (2 * Real.sqrt 3 - 3) / 3 * config.r :=
by sorry

end circle_radii_theorem_l3928_392828


namespace watch_dealer_profit_l3928_392800

theorem watch_dealer_profit (n d : ℕ) (h1 : d > 0) : 
  (∃ m : ℕ, d = 3 * m) →
  (10 * n - 30 = 100) →
  (∀ k : ℕ, k < n → ¬(10 * k - 30 = 100)) →
  n = 13 := by
sorry

end watch_dealer_profit_l3928_392800


namespace arithmetic_geometric_inequality_l3928_392808

/-- Arithmetic sequence -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Geometric sequence -/
def geometric_sequence (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

theorem arithmetic_geometric_inequality (b₁ q : ℝ) (m : ℕ) 
  (h₁ : b₁ > 0) 
  (h₂ : m > 0) 
  (h₃ : 1 < q) 
  (h₄ : q < (2 : ℝ)^(1 / m)) :
  ∃ d : ℝ, ∀ n : ℕ, 2 ≤ n ∧ n ≤ m + 1 → 
    |arithmetic_sequence b₁ d n - geometric_sequence b₁ q n| ≤ b₁ ∧
    b₁ * (q^m - 2) / m ≤ d ∧ d ≤ b₁ * q^m / m :=
sorry

end arithmetic_geometric_inequality_l3928_392808


namespace total_cost_star_wars_toys_l3928_392849

/-- The total cost of Star Wars toys, including a lightsaber, given the cost of other toys -/
theorem total_cost_star_wars_toys (other_toys_cost : ℕ) : 
  other_toys_cost = 1000 → 
  (2 * other_toys_cost + other_toys_cost) = 3 * other_toys_cost := by
  sorry

#check total_cost_star_wars_toys

end total_cost_star_wars_toys_l3928_392849


namespace marvelous_class_size_l3928_392841

theorem marvelous_class_size :
  ∀ (girls : ℕ) (boys : ℕ) (jelly_beans : ℕ),
    -- Each girl received twice as many jelly beans as there were girls
    (2 * girls * girls +
    -- Each boy received three times as many jelly beans as there were boys
    3 * boys * boys = 
    -- Total jelly beans given out
    jelly_beans) →
    -- She brought 645 jelly beans and had 3 left
    (jelly_beans = 645 - 3) →
    -- The number of boys was three more than twice the number of girls
    (boys = 2 * girls + 3) →
    -- The total number of students
    (girls + boys = 18) := by
  sorry

end marvelous_class_size_l3928_392841


namespace right_triangle_area_l3928_392823

theorem right_triangle_area (a b : ℝ) (h1 : a = 25) (h2 : b = 20) :
  (1 / 2 : ℝ) * a * b = 250 := by
  sorry

end right_triangle_area_l3928_392823


namespace min_absolute_T_l3928_392845

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

def T (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (a n) + (a (n+1)) + (a (n+2)) + (a (n+3)) + (a (n+4)) + (a (n+5))

theorem min_absolute_T (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 5 = 15 →
  a 10 = -10 →
  (∃ n : ℕ, ∀ m : ℕ, |T a n| ≤ |T a m|) →
  (∃ n : ℕ, n = 5 ∨ n = 6 ∧ ∀ m : ℕ, |T a n| ≤ |T a m|) :=
by sorry

end min_absolute_T_l3928_392845


namespace orangeade_water_ratio_l3928_392806

/-- Represents the orangeade mixing and selling scenario over two days -/
structure OrangeadeScenario where
  orange_juice : ℝ  -- Amount of orange juice used (same for both days)
  water_day1 : ℝ    -- Amount of water used on day 1
  water_day2 : ℝ    -- Amount of water used on day 2
  price_day1 : ℝ    -- Price per glass on day 1
  price_day2 : ℝ    -- Price per glass on day 2
  glasses_day1 : ℝ  -- Number of glasses sold on day 1
  glasses_day2 : ℝ  -- Number of glasses sold on day 2

/-- The conditions of the orangeade scenario -/
def scenario_conditions (s : OrangeadeScenario) : Prop :=
  s.orange_juice > 0 ∧
  s.water_day1 = s.orange_juice ∧
  s.price_day1 = 0.48 ∧
  s.glasses_day1 * (s.orange_juice + s.water_day1) = s.glasses_day2 * (s.orange_juice + s.water_day2) ∧
  s.price_day1 * s.glasses_day1 = s.price_day2 * s.glasses_day2

/-- The main theorem: under the given conditions, the ratio of water used on day 2 to orange juice is 1:1 -/
theorem orangeade_water_ratio (s : OrangeadeScenario) 
  (h : scenario_conditions s) : s.water_day2 = s.orange_juice :=
sorry


end orangeade_water_ratio_l3928_392806


namespace current_velocity_velocity_of_current_l3928_392811

/-- Calculates the velocity of the current given rowing conditions -/
theorem current_velocity (still_water_speed : ℝ) (total_time : ℝ) (distance : ℝ) : ℝ :=
  let v : ℝ := 2  -- The velocity of the current we want to prove
  have h1 : still_water_speed = 10 := by sorry
  have h2 : total_time = 30 := by sorry
  have h3 : distance = 144 := by sorry
  have h4 : (distance / (still_water_speed - v) + distance / (still_water_speed + v)) = total_time := by sorry
  v

/-- The main theorem stating the velocity of the current -/
theorem velocity_of_current : current_velocity 10 30 144 = 2 := by sorry

end current_velocity_velocity_of_current_l3928_392811


namespace continuity_at_nine_l3928_392851

def f (x : ℝ) : ℝ := 4 * x^2 + 4

theorem continuity_at_nine :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 9| < δ → |f x - f 9| < ε :=
by sorry

end continuity_at_nine_l3928_392851


namespace acidic_mixture_concentration_l3928_392890

/-- Proves that mixing liquids from two containers with given concentrations
    results in a mixture with the desired concentration. -/
theorem acidic_mixture_concentration
  (volume1 : ℝ) (volume2 : ℝ) (conc1 : ℝ) (conc2 : ℝ) (target_conc : ℝ)
  (x : ℝ) (y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0)
  (h_vol1 : volume1 = 54) (h_vol2 : volume2 = 48)
  (h_conc1 : conc1 = 0.35) (h_conc2 : conc2 = 0.25)
  (h_target : target_conc = 0.75)
  (h_mixture : conc1 * x + conc2 * y = target_conc * (x + y)) :
  (conc1 * x + conc2 * y) / (x + y) = target_conc :=
sorry

end acidic_mixture_concentration_l3928_392890


namespace smallest_n_for_polynomial_roots_l3928_392877

theorem smallest_n_for_polynomial_roots : ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, 0 < k → k < n →
    ¬∃ (a b : ℤ), ∃ (x y : ℝ),
      0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
      k * x^2 + a * x + b = 0 ∧
      k * y^2 + a * y + b = 0) ∧
  (∃ (a b : ℤ), ∃ (x y : ℝ),
    0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
    n * x^2 + a * x + b = 0 ∧
    n * y^2 + a * y + b = 0) ∧
  n = 5 :=
by sorry

end smallest_n_for_polynomial_roots_l3928_392877


namespace correct_average_l3928_392843

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 16 →
  incorrect_num = 26 →
  correct_num = 46 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 18 := by
  sorry

end correct_average_l3928_392843


namespace triangle_angle_arithmetic_sequence_property_l3928_392882

-- Define a structure for a triangle
structure Triangle :=
  (a b c : ℝ)  -- sides
  (A B C : ℝ)  -- angles in radians

-- Define the property of angles forming an arithmetic sequence
def arithmeticSequence (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B - t.A = d ∧ t.C - t.B = d

-- State the theorem
theorem triangle_angle_arithmetic_sequence_property (t : Triangle) 
  (h1 : t.a > 0) (h2 : t.b > 0) (h3 : t.c > 0)  -- positive sides
  (h4 : arithmeticSequence t)  -- angles form arithmetic sequence
  : 1 / (t.a + t.b) + 1 / (t.b + t.c) = 3 / (t.a + t.b + t.c) := by
  sorry

end triangle_angle_arithmetic_sequence_property_l3928_392882


namespace total_increase_in_two_centuries_l3928_392805

/-- Represents the increase in height per decade in meters -/
def increase_per_decade : ℝ := 90

/-- Represents the number of decades in 2 centuries -/
def decades_in_two_centuries : ℕ := 20

/-- Represents the total increase in height over 2 centuries in meters -/
def total_increase : ℝ := increase_per_decade * decades_in_two_centuries

/-- Theorem stating that the total increase in height over 2 centuries is 1800 meters -/
theorem total_increase_in_two_centuries : total_increase = 1800 := by
  sorry

end total_increase_in_two_centuries_l3928_392805


namespace probability_sum_11_three_dice_l3928_392888

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The target sum we're looking for -/
def targetSum : ℕ := 11

/-- The number of dice being rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of ways to roll a sum of 11 with three dice -/
def favorableOutcomes : ℕ := 24

/-- The probability of rolling a sum of 11 with three standard six-sided dice is 1/9 -/
theorem probability_sum_11_three_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 9 := by sorry

end probability_sum_11_three_dice_l3928_392888


namespace square_roots_of_specific_integers_l3928_392814

theorem square_roots_of_specific_integers : ∃ (m₁ m₂ : ℕ),
  m₁^2 = 170569 ∧
  m₂^2 = 175561 ∧
  m₁ = 413 ∧
  m₂ = 419 := by
sorry

end square_roots_of_specific_integers_l3928_392814


namespace dog_treat_expenditure_l3928_392852

/-- Represents the cost and nutritional value of dog treats -/
structure DogTreat where
  cost : ℚ
  np : ℕ

/-- Calculates the discounted price based on quantity and discount rate -/
def discountedPrice (regularPrice : ℚ) (quantity : ℕ) (discountRate : ℚ) : ℚ :=
  regularPrice * (1 - discountRate)

/-- Theorem: The total expenditure on dog treats for the month is $11.70 -/
theorem dog_treat_expenditure :
  let treatA : DogTreat := { cost := 0.1, np := 1 }
  let treatB : DogTreat := { cost := 0.15, np := 2 }
  let quantityA : ℕ := 50
  let quantityB : ℕ := 60
  let discountRateA : ℚ := 0.1
  let discountRateB : ℚ := 0.2
  let totalNP : ℕ := quantityA * treatA.np + quantityB * treatB.np
  let regularPriceA : ℚ := treatA.cost * quantityA
  let regularPriceB : ℚ := treatB.cost * quantityB
  let discountedPriceA : ℚ := discountedPrice regularPriceA quantityA discountRateA
  let discountedPriceB : ℚ := discountedPrice regularPriceB quantityB discountRateB
  let totalExpenditure : ℚ := discountedPriceA + discountedPriceB
  totalNP ≥ 40 ∧ totalExpenditure = 11.7 := by
  sorry


end dog_treat_expenditure_l3928_392852


namespace player_one_wins_l3928_392821

/-- Represents the number of coins a player can take -/
def ValidMove (player : ℕ) (coins : ℕ) : Prop :=
  match player with
  | 1 => coins % 2 = 1 ∧ 1 ≤ coins ∧ coins ≤ 99
  | 2 => coins % 2 = 0 ∧ 2 ≤ coins ∧ coins ≤ 100
  | _ => False

/-- The game state -/
structure GameState where
  coins : ℕ
  currentPlayer : ℕ

/-- A winning strategy for a player -/
def WinningStrategy (player : ℕ) : Prop :=
  ∀ (state : GameState), state.currentPlayer = player →
    ∃ (move : ℕ), ValidMove player move ∧
      (state.coins < move ∨
       ¬∃ (opponentMove : ℕ), ValidMove (3 - player) opponentMove ∧
         state.coins - move - opponentMove ≥ 0)

/-- The main theorem: Player 1 has a winning strategy -/
theorem player_one_wins : WinningStrategy 1 := by
  sorry

#check player_one_wins

end player_one_wins_l3928_392821


namespace smallest_layer_sugar_l3928_392866

/-- Represents a three-layer cake with sugar requirements -/
structure ThreeLayerCake where
  smallest_layer : ℝ
  second_layer : ℝ
  third_layer : ℝ
  second_is_twice_first : second_layer = 2 * smallest_layer
  third_is_thrice_second : third_layer = 3 * second_layer
  third_layer_sugar : third_layer = 12

/-- Proves that the smallest layer of the cake requires 2 cups of sugar -/
theorem smallest_layer_sugar (cake : ThreeLayerCake) : cake.smallest_layer = 2 := by
  sorry

#check smallest_layer_sugar

end smallest_layer_sugar_l3928_392866


namespace min_value_of_exponential_sum_l3928_392801

theorem min_value_of_exponential_sum (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 6) :
  (9 : ℝ)^x + 3^y ≥ 54 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 6 ∧ (9 : ℝ)^x + 3^y = 54 := by
  sorry

end min_value_of_exponential_sum_l3928_392801


namespace percentage_of_wax_used_l3928_392848

def original_wax_20oz : ℕ := 5
def original_wax_5oz : ℕ := 5
def original_wax_1oz : ℕ := 25
def new_candles : ℕ := 3
def new_candle_size : ℕ := 5

def total_original_wax : ℕ := original_wax_20oz * 20 + original_wax_5oz * 5 + original_wax_1oz * 1
def wax_used_for_new_candles : ℕ := new_candles * new_candle_size

theorem percentage_of_wax_used (total_original_wax wax_used_for_new_candles : ℕ) :
  (wax_used_for_new_candles : ℚ) / (total_original_wax : ℚ) * 100 = 10 :=
sorry

end percentage_of_wax_used_l3928_392848


namespace complex_subtraction_simplification_l3928_392802

theorem complex_subtraction_simplification :
  (4 - 3 * Complex.I) - (7 - 5 * Complex.I) = -3 + 2 * Complex.I := by
  sorry

end complex_subtraction_simplification_l3928_392802


namespace closest_multiple_of_17_to_3513_l3928_392878

theorem closest_multiple_of_17_to_3513 :
  ∀ k : ℤ, |3519 - 3513| ≤ |17 * k - 3513| :=
by
  sorry

end closest_multiple_of_17_to_3513_l3928_392878


namespace product_of_roots_l3928_392859

theorem product_of_roots (x : ℂ) :
  2 * x^3 - 3 * x^2 - 10 * x + 14 = 0 →
  ∃ r₁ r₂ r₃ : ℂ, (x - r₁) * (x - r₂) * (x - r₃) = 2 * x^3 - 3 * x^2 - 10 * x + 14 ∧ r₁ * r₂ * r₃ = -7 := by
  sorry

end product_of_roots_l3928_392859


namespace sum_of_two_angles_in_triangle_l3928_392893

/-- Theorem: In a triangle where one angle is 72°, the sum of the other two angles is 108° -/
theorem sum_of_two_angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : B = 72) : 
  A + C = 108 := by
  sorry

end sum_of_two_angles_in_triangle_l3928_392893


namespace pentagonal_prism_diagonals_l3928_392867

/-- A regular pentagonal prism -/
structure RegularPentagonalPrism where
  /-- The number of vertices on each base -/
  base_vertices : ℕ
  /-- The total number of vertices -/
  total_vertices : ℕ
  /-- The number of base vertices is 5 -/
  base_is_pentagon : base_vertices = 5
  /-- The total number of vertices is twice the number of base vertices -/
  total_is_double_base : total_vertices = 2 * base_vertices

/-- A diagonal in a regular pentagonal prism -/
def is_diagonal (prism : RegularPentagonalPrism) (v1 v2 : ℕ) : Prop :=
  v1 ≠ v2 ∧ 
  v1 < prism.total_vertices ∧ 
  v2 < prism.total_vertices ∧
  (v1 < prism.base_vertices ↔ v2 ≥ prism.base_vertices)

/-- The total number of diagonals in a regular pentagonal prism -/
def total_diagonals (prism : RegularPentagonalPrism) : ℕ :=
  (prism.base_vertices * prism.base_vertices)

theorem pentagonal_prism_diagonals (prism : RegularPentagonalPrism) : 
  total_diagonals prism = 10 := by
  sorry

end pentagonal_prism_diagonals_l3928_392867


namespace solve_equation_l3928_392842

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
  sorry

end solve_equation_l3928_392842


namespace triangle_area_l3928_392846

/-- The area of a triangle with base 4 and height 8 is 16 -/
theorem triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 4 ∧ height = 8 →
    area = (base * height) / 2 →
    area = 16

/-- Proof of the theorem -/
lemma prove_triangle_area : triangle_area 4 8 16 := by
  sorry

end triangle_area_l3928_392846


namespace p_necessary_not_sufficient_for_q_l3928_392860

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, |x - 1| < 2 → (x + 2) * (x - 3) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x - 3) < 0 ∧ |x - 1| ≥ 2) := by
  sorry

end p_necessary_not_sufficient_for_q_l3928_392860


namespace bill_muffin_batches_l3928_392873

/-- The cost of blueberries in dollars per 6 ounce carton -/
def blueberry_cost : ℚ := 5

/-- The cost of raspberries in dollars per 12 ounce carton -/
def raspberry_cost : ℚ := 3

/-- The amount of fruit in ounces required for each batch of muffins -/
def fruit_per_batch : ℚ := 12

/-- The total savings in dollars by using raspberries instead of blueberries -/
def total_savings : ℚ := 22

/-- The number of batches Bill plans to make -/
def num_batches : ℕ := 3

/-- Theorem stating that given the costs, fruit requirement, and total savings,
    Bill plans to make 3 batches of muffins -/
theorem bill_muffin_batches :
  (blueberry_cost * 2 - raspberry_cost) * (num_batches : ℚ) ≤ total_savings ∧
  (blueberry_cost * 2 - raspberry_cost) * ((num_batches + 1) : ℚ) > total_savings :=
by sorry

end bill_muffin_batches_l3928_392873


namespace solution_set_quadratic_inequality_l3928_392897

theorem solution_set_quadratic_inequality :
  {x : ℝ | x * (x - 1) > 0} = Set.Iio 0 ∪ Set.Ioi 1 :=
by sorry

end solution_set_quadratic_inequality_l3928_392897


namespace ashleys_age_l3928_392881

/-- Given that Ashley and Mary's ages are in the ratio 4:7 and their sum is 22, 
    prove that Ashley's age is 8 years. -/
theorem ashleys_age (ashley mary : ℕ) 
  (h_ratio : ashley * 7 = mary * 4)
  (h_sum : ashley + mary = 22) : 
  ashley = 8 := by
  sorry

end ashleys_age_l3928_392881


namespace inequality_implies_range_l3928_392803

theorem inequality_implies_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |2*x + 2| ≥ a^2 + (1/2)*a + 2) → 
  -1/2 ≤ a ∧ a ≤ 0 := by
sorry

end inequality_implies_range_l3928_392803


namespace complex_power_difference_zero_l3928_392855

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference_zero : (1 + i)^20 - (1 - i)^20 = 0 := by sorry

end complex_power_difference_zero_l3928_392855


namespace only_324_and_648_have_property_l3928_392863

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property we're looking for
def hasProperty (x : ℕ) : Prop :=
  x = 36 * sumOfDigits x

-- State the theorem
theorem only_324_and_648_have_property :
  ∀ x : ℕ, hasProperty x ↔ x = 324 ∨ x = 648 :=
sorry

end only_324_and_648_have_property_l3928_392863


namespace francine_daily_drive_distance_l3928_392861

/-- The number of days Francine doesn't go to work each week -/
def days_off_per_week : ℕ := 3

/-- The total distance Francine drives to work in 4 weeks (in km) -/
def total_distance_4_weeks : ℕ := 2240

/-- The number of weeks in the given period -/
def num_weeks : ℕ := 4

/-- The number of working days in a week -/
def work_days_per_week : ℕ := 7 - days_off_per_week

/-- The total number of working days in 4 weeks -/
def total_work_days : ℕ := work_days_per_week * num_weeks

/-- The distance Francine drives to work each day (in km) -/
def daily_distance : ℕ := total_distance_4_weeks / total_work_days

theorem francine_daily_drive_distance :
  daily_distance = 280 := by sorry

end francine_daily_drive_distance_l3928_392861


namespace restaurant_bill_theorem_l3928_392892

theorem restaurant_bill_theorem :
  let num_people : ℕ := 7
  let regular_spend : ℕ := 11
  let num_regular : ℕ := 6
  let extra_spend : ℕ := 6
  let total_spend : ℕ := regular_spend * num_regular + 
    (regular_spend * num_regular + (total_spend / num_people + extra_spend))
  total_spend = 84 := by sorry

end restaurant_bill_theorem_l3928_392892


namespace jam_weight_l3928_392884

/-- Calculates the weight of jam given the initial and final suitcase weights and other item weights --/
theorem jam_weight 
  (initial_weight : ℝ) 
  (final_weight : ℝ) 
  (perfume_weight : ℝ) 
  (perfume_count : ℕ) 
  (chocolate_weight : ℝ) 
  (soap_weight : ℝ) 
  (soap_count : ℕ) 
  (h1 : initial_weight = 5) 
  (h2 : final_weight = 11) 
  (h3 : perfume_weight = 1.2 / 16) 
  (h4 : perfume_count = 5) 
  (h5 : chocolate_weight = 4) 
  (h6 : soap_weight = 5 / 16) 
  (h7 : soap_count = 2) : 
  final_weight - (initial_weight + perfume_weight * perfume_count + chocolate_weight + soap_weight * soap_count) = 1 := by
  sorry

#check jam_weight

end jam_weight_l3928_392884


namespace distance_at_two_point_five_l3928_392891

/-- The distance traveled by a ball rolling down an inclined plane -/
def distance (t : ℝ) : ℝ := 10 * t^2

/-- Theorem: The distance traveled at t = 2.5 seconds is 62.5 feet -/
theorem distance_at_two_point_five :
  distance 2.5 = 62.5 := by sorry

end distance_at_two_point_five_l3928_392891


namespace diagonal_not_parallel_to_sides_l3928_392832

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 0) :
  n * (2 * n - 3) > 2 * n * (n - 2) := by
  sorry

#check diagonal_not_parallel_to_sides

end diagonal_not_parallel_to_sides_l3928_392832
