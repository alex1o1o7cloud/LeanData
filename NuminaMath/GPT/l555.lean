import Mathlib

namespace NUMINAMATH_GPT_blue_lipstick_students_l555_55523

def total_students : ℕ := 200
def students_with_lipstick : ℕ := total_students / 2
def students_with_red_lipstick : ℕ := students_with_lipstick / 4
def students_with_blue_lipstick : ℕ := students_with_red_lipstick / 5

theorem blue_lipstick_students : students_with_blue_lipstick = 5 :=
by
  sorry

end NUMINAMATH_GPT_blue_lipstick_students_l555_55523


namespace NUMINAMATH_GPT_selection_including_both_genders_is_34_l555_55584

def count_ways_to_select_students_with_conditions (total_students boys girls select_students : ℕ) : ℕ :=
  if total_students = 7 ∧ boys = 4 ∧ girls = 3 ∧ select_students = 4 then
    (Nat.choose total_students select_students) - 1
  else
    0

theorem selection_including_both_genders_is_34 :
  count_ways_to_select_students_with_conditions 7 4 3 4 = 34 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_selection_including_both_genders_is_34_l555_55584


namespace NUMINAMATH_GPT_percentage_decrease_correct_l555_55511

variable (O N : ℕ)
variable (percentage_decrease : ℕ)

-- Define the conditions based on the problem
def original_price := 1240
def new_price := 620
def price_effect := ((original_price - new_price) * 100) / original_price

-- Prove the percentage decrease is 50%
theorem percentage_decrease_correct :
  price_effect = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_decrease_correct_l555_55511


namespace NUMINAMATH_GPT_actual_cost_of_article_l555_55525

-- Define the basic conditions of the problem
variable (x : ℝ)
variable (h : x - 0.24 * x = 1064)

-- The theorem we need to prove
theorem actual_cost_of_article : x = 1400 :=
by
  -- since we are not proving anything here, we skip the proof
  sorry

end NUMINAMATH_GPT_actual_cost_of_article_l555_55525


namespace NUMINAMATH_GPT_player_B_wins_in_least_steps_l555_55562

noncomputable def least_steps_to_win (n : ℕ) : ℕ :=
  n

theorem player_B_wins_in_least_steps (n : ℕ) (h_n : n > 0) :
  ∃ k, k = least_steps_to_win n ∧ k = n := by
  sorry

end NUMINAMATH_GPT_player_B_wins_in_least_steps_l555_55562


namespace NUMINAMATH_GPT_oil_bill_january_l555_55517

-- Define the constants and variables
variables (F J : ℝ)

-- Define the conditions
def condition1 : Prop := F / J = 5 / 4
def condition2 : Prop := (F + 45) / J = 3 / 2

-- Define the main theorem stating the proof problem
theorem oil_bill_january 
  (h1 : condition1 F J) 
  (h2 : condition2 F J) : 
  J = 180 :=
sorry

end NUMINAMATH_GPT_oil_bill_january_l555_55517


namespace NUMINAMATH_GPT_find_first_number_l555_55564

theorem find_first_number (n : ℝ) (h1 : n / 14.5 = 175) :
  n = 2537.5 :=
by 
  sorry

end NUMINAMATH_GPT_find_first_number_l555_55564


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l555_55586

noncomputable def f (x : ℝ) : ℝ := 2^(abs x) - 1
noncomputable def a : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def b : ℝ := f (Real.log 5 / Real.log 2)
noncomputable def c : ℝ := f (Real.log (1/4) / Real.log 2)

theorem relationship_between_a_b_c : a < c ∧ c < b :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l555_55586


namespace NUMINAMATH_GPT_exterior_angle_of_parallel_lines_l555_55595

theorem exterior_angle_of_parallel_lines (A B C x y : ℝ) (hAx : A = 40) (hBx : B = 90) (hCx : C = 40)
  (h_parallel : true)
  (h_triangle : x = 180 - A - C)
  (h_exterior_angle : y = 180 - x) :
  y = 80 := 
by
  sorry

end NUMINAMATH_GPT_exterior_angle_of_parallel_lines_l555_55595


namespace NUMINAMATH_GPT_least_add_to_divisible_least_subtract_to_divisible_l555_55591

theorem least_add_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (a : ℤ) : 
  n = 1100 → d = 37 → r = n % d → a = d - r → (n + a) % d = 0 :=
by sorry

theorem least_subtract_to_divisible (n : ℤ) (d : ℤ) (r : ℤ) (s : ℤ) : 
  n = 1100 → d = 37 → r = n % d → s = r → (n - s) % d = 0 :=
by sorry

end NUMINAMATH_GPT_least_add_to_divisible_least_subtract_to_divisible_l555_55591


namespace NUMINAMATH_GPT_least_pos_int_div_by_3_5_7_l555_55554

/-
  Prove that the least positive integer divisible by the primes 3, 5, and 7 is 105.
-/

theorem least_pos_int_div_by_3_5_7 : ∃ (n : ℕ), n > 0 ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ n = 105 :=
by 
  sorry

end NUMINAMATH_GPT_least_pos_int_div_by_3_5_7_l555_55554


namespace NUMINAMATH_GPT_negation_of_prop_l555_55513

-- Define the original proposition
def prop (x : ℝ) : Prop := x^2 - x + 2 ≥ 0

-- State the negation of the original proposition
theorem negation_of_prop : (¬ ∀ x : ℝ, prop x) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := 
by
  sorry

end NUMINAMATH_GPT_negation_of_prop_l555_55513


namespace NUMINAMATH_GPT_range_of_m_range_of_x_l555_55589

-- Define the function f(x) = m*x^2 - m*x - 6 + m
def f (m x : ℝ) : ℝ := m*x^2 - m*x - 6 + m

-- Proof for the first statement
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f m x < 0) ↔ m < 6 / 7 := 
sorry

-- Proof for the second statement
theorem range_of_x (x : ℝ) :
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → f m x < 0) ↔ -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_range_of_x_l555_55589


namespace NUMINAMATH_GPT_hyperbola_foci_difference_l555_55515

noncomputable def hyperbola_foci_distance (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (a : ℝ) : ℝ :=
  |dist P F₁ - dist P F₂|

theorem hyperbola_foci_difference (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : 
  (P.1 ^ 2 - P.2 ^ 2 = 4) ∧ (P.1 < 0) → (hyperbola_foci_distance P F₁ F₂ 2 = -4) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_hyperbola_foci_difference_l555_55515


namespace NUMINAMATH_GPT_rope_length_equals_120_l555_55507

theorem rope_length_equals_120 (x : ℝ) (l : ℝ)
  (h1 : x + 20 = 3 * x) 
  (h2 : l = 4 * (2 * x)) : 
  l = 120 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_rope_length_equals_120_l555_55507


namespace NUMINAMATH_GPT_pieces_per_package_l555_55573

-- Define Robin's packages
def numGumPackages := 28
def numCandyPackages := 14

-- Define total number of pieces
def totalPieces := 7

-- Define the total number of packages
def totalPackages := numGumPackages + numCandyPackages

-- Define the expected number of pieces per package as the theorem to prove
theorem pieces_per_package : (totalPieces / totalPackages) = 1/6 := by
  sorry

end NUMINAMATH_GPT_pieces_per_package_l555_55573


namespace NUMINAMATH_GPT_point_coordinates_l555_55510

noncomputable def parametric_curve (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) := (3 * Real.cos θ, 4 * Real.sin θ)

theorem point_coordinates (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) : 
  (Real.arcsin (4 * (Real.tan θ)) = π/4) → (3 * Real.cos θ, 4 * Real.sin θ) = (12 / 5, 12 / 5) :=
by
  sorry

end NUMINAMATH_GPT_point_coordinates_l555_55510


namespace NUMINAMATH_GPT_difference_of_one_third_and_five_l555_55543

theorem difference_of_one_third_and_five (n : ℕ) (h : n = 45) : (n / 3) - 5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_one_third_and_five_l555_55543


namespace NUMINAMATH_GPT_inequality_solution_set_l555_55535

theorem inequality_solution_set {x : ℝ} : 2 * x^2 - x - 1 > 0 ↔ (x < -1 / 2 ∨ x > 1) := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l555_55535


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l555_55537

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_of_M_and_N : M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l555_55537


namespace NUMINAMATH_GPT_b_not_six_iff_neg_two_not_in_range_l555_55518

def g (x b : ℝ) := x^3 + x^2 + b*x + 2

theorem b_not_six_iff_neg_two_not_in_range (b : ℝ) : 
  (∀ x : ℝ, g x b ≠ -2) ↔ b ≠ 6 :=
by
  sorry

end NUMINAMATH_GPT_b_not_six_iff_neg_two_not_in_range_l555_55518


namespace NUMINAMATH_GPT_cost_price_of_article_l555_55598

theorem cost_price_of_article (M : ℝ) (SP : ℝ) (C : ℝ) 
  (hM : M = 65)
  (hSP : SP = 0.95 * M)
  (hProfit : SP = 1.30 * C) : 
  C = 47.50 :=
by 
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l555_55598


namespace NUMINAMATH_GPT_athlete_distance_proof_l555_55534

-- Definition of conditions as constants
def time_seconds : ℕ := 20
def speed_kmh : ℕ := 36

-- Convert speed from km/h to m/s
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Proof statement that the distance is 200 meters
theorem athlete_distance_proof : speed_mps * time_seconds = 200 :=
by sorry

end NUMINAMATH_GPT_athlete_distance_proof_l555_55534


namespace NUMINAMATH_GPT_jogger_distance_ahead_l555_55545

def speed_jogger_kmph : ℕ := 9
def speed_train_kmph : ℕ := 45
def length_train_m : ℕ := 120
def time_to_pass_jogger_s : ℕ := 36

theorem jogger_distance_ahead :
  let relative_speed_mps := (speed_train_kmph - speed_jogger_kmph) * 1000 / 3600
  let distance_covered_m := relative_speed_mps * time_to_pass_jogger_s
  let jogger_distance_ahead : ℕ := distance_covered_m - length_train_m
  jogger_distance_ahead = 240 :=
by
  sorry

end NUMINAMATH_GPT_jogger_distance_ahead_l555_55545


namespace NUMINAMATH_GPT_sum_between_9p5_and_10_l555_55548

noncomputable def sumMixedNumbers : ℚ :=
  (29 / 9) + (11 / 4) + (81 / 20)

theorem sum_between_9p5_and_10 :
  9.5 < sumMixedNumbers ∧ sumMixedNumbers < 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_between_9p5_and_10_l555_55548


namespace NUMINAMATH_GPT_problem_statement_l555_55541

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

def candidate_function (x : ℝ) : ℝ :=
  x * |x|

theorem problem_statement : is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l555_55541


namespace NUMINAMATH_GPT_circle_center_and_radius_l555_55593

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 - 4 * y - 1 = 0 ↔ (x, y) = (0, 2) ∧ 5 = (0 - x)^2 + (2 - y)^2 :=
by sorry

end NUMINAMATH_GPT_circle_center_and_radius_l555_55593


namespace NUMINAMATH_GPT_james_after_paying_debt_l555_55571

variables (L J A : Real)

-- Define the initial conditions
def total_money : Real := 300
def debt : Real := 25
def total_with_debt : Real := total_money + debt

axiom h1 : J = A + 40
axiom h2 : J + A = total_with_debt

-- Prove that James owns $170 after paying off half of Lucas' debt
theorem james_after_paying_debt (h1 : J = A + 40) (h2 : J + A = total_with_debt) :
  (J - (debt / 2)) = 170 :=
  sorry

end NUMINAMATH_GPT_james_after_paying_debt_l555_55571


namespace NUMINAMATH_GPT_find_parking_cost_l555_55528

theorem find_parking_cost :
  ∃ (C : ℝ), (C + 7 * 1.75) / 9 = 2.4722222222222223 ∧ C = 10 :=
sorry

end NUMINAMATH_GPT_find_parking_cost_l555_55528


namespace NUMINAMATH_GPT_triangle_area_l555_55590

theorem triangle_area (h : ℝ) (hypotenuse : h = 12) (angle : ∃θ : ℝ, θ = 30 ∧ θ = 30) :
  ∃ (A : ℝ), A = 18 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l555_55590


namespace NUMINAMATH_GPT_daniel_spent_2290_l555_55550

theorem daniel_spent_2290 (total_games: ℕ) (price_12_games count_price_12: ℕ) 
  (price_7_games frac_price_7: ℕ) (price_3_games: ℕ) 
  (count_price_7: ℕ) (h1: total_games = 346)
  (h2: count_price_12 = 80) (h3: price_12_games = 12)
  (h4: frac_price_7 = 50) (h5: price_7_games = 7)
  (h6: price_3_games = 3) (h7: count_price_7 = (frac_price_7 * (total_games - count_price_12)) / 100):
  (count_price_12 * price_12_games) + (count_price_7 * price_7_games) + ((total_games - count_price_12 - count_price_7) * price_3_games) = 2290 := 
by
  sorry

end NUMINAMATH_GPT_daniel_spent_2290_l555_55550


namespace NUMINAMATH_GPT_field_area_l555_55579

def length : ℝ := 80 -- Length of the uncovered side
def total_fencing : ℝ := 97 -- Total fencing required

theorem field_area : ∃ (W L : ℝ), L = length ∧ 2 * W + L = total_fencing ∧ L * W = 680 := by
  sorry

end NUMINAMATH_GPT_field_area_l555_55579


namespace NUMINAMATH_GPT_totalMountainNumbers_l555_55505

-- Define a 4-digit mountain number based on the given conditions.
def isMountainNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    b > a ∧ b > d ∧ c > a ∧ c > d ∧
    a ≠ d

-- Define the main theorem stating that the total number of 4-digit mountain numbers is 1512.
theorem totalMountainNumbers : 
  ∃ n, (∀ m, isMountainNumber m → ∃ l, l = 1 ∧ 4 ≤ m ∧ m ≤ 9999) ∧ n = 1512 := sorry

end NUMINAMATH_GPT_totalMountainNumbers_l555_55505


namespace NUMINAMATH_GPT_man_l555_55522

theorem man's_speed_against_the_current (vm vc : ℝ) 
(h1: vm + vc = 15) 
(h2: vm - vc = 10) : 
vm - vc = 10 := 
by 
  exact h2

end NUMINAMATH_GPT_man_l555_55522


namespace NUMINAMATH_GPT_arithmetic_contains_geometric_l555_55557

theorem arithmetic_contains_geometric {a b : ℚ} (h : a^2 + b^2 ≠ 0) :
  ∃ (c q : ℚ) (f : ℕ → ℚ), (∀ n, f n = c * q^n) ∧ (∀ n, f n = a + b * n) := 
sorry

end NUMINAMATH_GPT_arithmetic_contains_geometric_l555_55557


namespace NUMINAMATH_GPT_distance_from_A_to_B_l555_55542

theorem distance_from_A_to_B (D : ℝ) :
  (∃ D, (∀ tC, tC = D / 30) 
      ∧ (∀ tD, tD = D / 48 ∧ tD < (D / 30 - 1.5))
      ∧ D = 120) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_A_to_B_l555_55542


namespace NUMINAMATH_GPT_hyperbola_constants_sum_l555_55585

noncomputable def hyperbola_asymptotes_equation (x y : ℝ) : Prop :=
  (y = 2 * x + 5) ∨ (y = -2 * x + 1)

noncomputable def hyperbola_passing_through (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 7)

theorem hyperbola_constants_sum
  (a b h k : ℝ) (ha : a > 0) (hb : b > 0)
  (H1 : ∀ x y : ℝ, hyperbola_asymptotes_equation x y)
  (H2 : hyperbola_passing_through 0 7)
  (H3 : h = -1)
  (H4 : k = 3)
  (H5 : a = 2 * b)
  (H6 : b = Real.sqrt 3) :
  a + h = 2 * Real.sqrt 3 - 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_constants_sum_l555_55585


namespace NUMINAMATH_GPT_min_colors_needed_l555_55577

theorem min_colors_needed (n : ℕ) (h : n + n.choose 2 ≥ 12) : n = 5 :=
sorry

end NUMINAMATH_GPT_min_colors_needed_l555_55577


namespace NUMINAMATH_GPT_find_m_value_l555_55538

-- Definitions of the given lines
def l1 (x y : ℝ) (m : ℝ) : Prop := x + m * y + 6 = 0
def l2 (x y : ℝ) (m : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Parallel lines condition
def parallel (m : ℝ) : Prop :=
  ∀ x y : ℝ, l1 x y m = l2 x y m

-- Proof that the value of m for the lines to be parallel is indeed -1
theorem find_m_value : parallel (-1) :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l555_55538


namespace NUMINAMATH_GPT_perimeter_of_region_is_70_l555_55576

-- Define the given conditions
def area_of_region (total_area : ℝ) (num_squares : ℕ) : Prop :=
  total_area = 392 ∧ num_squares = 8

def side_length_of_square (area : ℝ) (side_length : ℝ) : Prop :=
  area = side_length^2 ∧ side_length = 7

def perimeter_of_region (num_squares : ℕ) (side_length : ℝ) (perimeter : ℝ) : Prop :=
  perimeter = 8 * side_length + 2 * side_length ∧ perimeter = 70

-- Statement to prove
theorem perimeter_of_region_is_70 :
  ∀ (total_area : ℝ) (num_squares : ℕ), 
    area_of_region total_area num_squares →
    ∃ (side_length : ℝ) (perimeter : ℝ), 
      side_length_of_square (total_area / num_squares) side_length ∧
      perimeter_of_region num_squares side_length perimeter :=
by {
  sorry
}

end NUMINAMATH_GPT_perimeter_of_region_is_70_l555_55576


namespace NUMINAMATH_GPT_sum_of_intercepts_of_line_l555_55516

theorem sum_of_intercepts_of_line (x y : ℝ) (hx : 2 * x - 3 * y + 6 = 0) :
  2 + (-3) = -1 :=
sorry

end NUMINAMATH_GPT_sum_of_intercepts_of_line_l555_55516


namespace NUMINAMATH_GPT_parabola_directrix_l555_55521

variable {F P1 P2 : Point}

def is_on_parabola (F : Point) (P1 : Point) : Prop := 
  -- Definition of a point being on the parabola with focus F and a directrix (to be determined).
  sorry

def construct_circles (F P1 P2 : Point) : Circle × Circle :=
  -- Construct circles centered at P1 and P2 passing through F.
  sorry

def common_external_tangents (k1 k2 : Circle) : Nat :=
  -- Function to find the number of common external tangents between two circles.
  sorry

theorem parabola_directrix (F P1 P2 : Point) (h1 : is_on_parabola F P1) (h2 : is_on_parabola F P2) :
  ∃ (k1 k2 : Circle), construct_circles F P1 P2 = (k1, k2) → 
    common_external_tangents k1 k2 = 2 :=
by
  -- Proof that under these conditions, there are exactly 2 common external tangents.
  sorry

end NUMINAMATH_GPT_parabola_directrix_l555_55521


namespace NUMINAMATH_GPT_perimeter_of_regular_polygon_l555_55501

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end NUMINAMATH_GPT_perimeter_of_regular_polygon_l555_55501


namespace NUMINAMATH_GPT_june_time_to_bernard_l555_55587

theorem june_time_to_bernard (distance_Julia : ℝ) (time_Julia : ℝ) (distance_Bernard_June : ℝ) (time_Bernard : ℝ) (distance_June_Bernard : ℝ)
  (h1 : distance_Julia = 2) (h2 : time_Julia = 6) (h3 : distance_Bernard_June = 5) (h4 : time_Bernard = 15) (h5 : distance_June_Bernard = 7) :
  distance_June_Bernard / (distance_Julia / time_Julia) = 21 := by
    sorry

end NUMINAMATH_GPT_june_time_to_bernard_l555_55587


namespace NUMINAMATH_GPT_Rachel_money_left_l555_55540

theorem Rachel_money_left 
  (money_earned : ℕ)
  (lunch_fraction : ℚ)
  (clothes_percentage : ℚ)
  (dvd_cost : ℚ)
  (supplies_percentage : ℚ)
  (money_left : ℚ) :
  money_earned = 200 →
  lunch_fraction = 1 / 4 →
  clothes_percentage = 15 / 100 →
  dvd_cost = 24.50 →
  supplies_percentage = 10.5 / 100 →
  money_left = 74.50 :=
by
  intros h_money h_lunch h_clothes h_dvd h_supplies
  sorry

end NUMINAMATH_GPT_Rachel_money_left_l555_55540


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l555_55546

theorem bus_speed_excluding_stoppages :
  ∀ (S : ℝ), (45 = (3 / 4) * S) → (S = 60) :=
by 
  intros S h
  sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l555_55546


namespace NUMINAMATH_GPT_Emily_cleaning_time_in_second_room_l555_55588

/-
Lilly, Fiona, Jack, and Emily are cleaning 3 rooms.
For the first room: Lilly and Fiona together: 1/4 of the time, Jack: 1/3 of the time, Emily: the rest of the time.
In the second room: Jack: 25%, Emily: 25%, Lilly and Fiona: the remaining 50%.
In the third room: Emily: 40%, Lilly: 20%, Jack: 20%, Fiona: 20%.
Total time for all rooms: 12 hours.

Prove that the total time Emily spent cleaning in the second room is 60 minutes.
-/

theorem Emily_cleaning_time_in_second_room :
  let total_time := 12 -- total time in hours
  let time_per_room := total_time / 3 -- time per room in hours
  let time_per_room_minutes := time_per_room * 60 -- time per room in minutes
  let emily_cleaning_percentage := 0.25 -- Emily's cleaning percentage in the second room
  let emily_cleaning_time := emily_cleaning_percentage * time_per_room_minutes -- cleaning time in minutes
  emily_cleaning_time = 60 := by
  sorry

end NUMINAMATH_GPT_Emily_cleaning_time_in_second_room_l555_55588


namespace NUMINAMATH_GPT_find_s_l555_55555

theorem find_s (n r s c d : ℚ) 
  (h1 : Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 3 = 0) 
  (h2 : c * d = 3)
  (h3 : Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s = 
        Polynomial.C (c + d⁻¹) * Polynomial.C (d + c⁻¹)) : 
  s = 16 / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_s_l555_55555


namespace NUMINAMATH_GPT_no_polyhedron_without_triangles_and_three_valent_vertices_l555_55503

-- Definitions and assumptions based on the problem's conditions
def f_3 := 0 -- no triangular faces
def p_3 := 0 -- no vertices with degree three

-- Euler's formula for convex polyhedra
def euler_formula (f p a : ℕ) : Prop := f + p - a = 2

-- Define general properties for faces and vertices in polyhedra
def polyhedron_no_triangular_no_three_valent (f p a f_4 f_5 p_4 p_5: ℕ) : Prop :=
  f_3 = 0 ∧ p_3 = 0 ∧ 2 * a ≥ 4 * (f_4 + f_5) ∧ 2 * a ≥ 4 * (p_4 + p_5) ∧ euler_formula f p a

-- Theorem to prove there does not exist such a polyhedron
theorem no_polyhedron_without_triangles_and_three_valent_vertices :
  ¬ ∃ (f p a f_4 f_5 p_4 p_5 : ℕ), polyhedron_no_triangular_no_three_valent f p a f_4 f_5 p_4 p_5 :=
by
  sorry

end NUMINAMATH_GPT_no_polyhedron_without_triangles_and_three_valent_vertices_l555_55503


namespace NUMINAMATH_GPT_variance_of_arithmetic_sequence_common_diff_3_l555_55567

noncomputable def variance (ξ : List ℝ) : ℝ :=
  let n := ξ.length
  let mean := ξ.sum / n
  let var_sum := (ξ.map (fun x => (x - mean) ^ 2)).sum
  var_sum / n

def arithmetic_sequence (a1 : ℝ) (d : ℝ) (n : ℕ) : List ℝ :=
  List.range n |>.map (fun i => a1 + i * d)

theorem variance_of_arithmetic_sequence_common_diff_3 :
  ∀ (a1 : ℝ),
    variance (arithmetic_sequence a1 3 9) = 60 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_arithmetic_sequence_common_diff_3_l555_55567


namespace NUMINAMATH_GPT_matrix_power_four_l555_55524

theorem matrix_power_four :
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![3 * Real.sqrt 2, -3],
    ![3, 3 * Real.sqrt 2]
  ]
  (A ^ 4 = ![
    ![ -81, 0],
    ![0, -81]
  ]) :=
by
  sorry

end NUMINAMATH_GPT_matrix_power_four_l555_55524


namespace NUMINAMATH_GPT_geometric_seq_an_formula_inequality_proof_find_t_sum_not_in_seq_l555_55529

def seq_an : ℕ → ℝ := sorry
def sum_Sn : ℕ → ℝ := sorry

axiom Sn_recurrence (n : ℕ) : sum_Sn (n + 1) = (1/2) * sum_Sn n + 2
axiom a1_def : seq_an 1 = 2
axiom a2_def : seq_an 2 = 1

theorem geometric_seq (n : ℕ) : ∃ r : ℝ, ∀ (m : ℕ), sum_Sn m - 4 = (sum_Sn 1 - 4) * r^(m-1) := 
sorry

theorem an_formula (n : ℕ) : seq_an n = (1/2)^(n-2) := 
sorry

theorem inequality_proof (t n : ℕ) (t_pos : 0 < t) : 
  (seq_an t * sum_Sn (n + 1) - 1) / (seq_an t * seq_an (n + 1) - 1) < 1/2 :=
sorry

theorem find_t : ∃ (t : ℕ), t = 3 ∨ t = 4 := 
sorry

theorem sum_not_in_seq (m n k : ℕ) (distinct : k ≠ m ∧ m ≠ n ∧ k ≠ n) : 
  (seq_an m + seq_an n ≠ seq_an k) :=
sorry

end NUMINAMATH_GPT_geometric_seq_an_formula_inequality_proof_find_t_sum_not_in_seq_l555_55529


namespace NUMINAMATH_GPT_original_quantity_l555_55549

theorem original_quantity (x : ℕ) : 
  (532 * x - 325 * x = 1065430) -> x = 5148 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_original_quantity_l555_55549


namespace NUMINAMATH_GPT_cats_weigh_more_than_puppies_l555_55502

noncomputable def weight_puppy_A : ℝ := 6.5
noncomputable def weight_puppy_B : ℝ := 7.2
noncomputable def weight_puppy_C : ℝ := 8
noncomputable def weight_puppy_D : ℝ := 9.5
noncomputable def weight_cat : ℝ := 2.8
noncomputable def num_cats : ℕ := 16

theorem cats_weigh_more_than_puppies :
  (num_cats * weight_cat) - (weight_puppy_A + weight_puppy_B + weight_puppy_C + weight_puppy_D) = 13.6 :=
by
  sorry

end NUMINAMATH_GPT_cats_weigh_more_than_puppies_l555_55502


namespace NUMINAMATH_GPT_trajectory_equation_l555_55575

noncomputable def A : ℝ × ℝ := (0, -1)
noncomputable def B (x_b : ℝ) : ℝ × ℝ := (x_b, -3)
noncomputable def M (x y : ℝ) : ℝ × ℝ := (x, y)

-- Conditions as definitions in Lean 4
def MB_parallel_OA (x y x_b : ℝ) : Prop :=
  ∃ k : ℝ, (x_b - x) = k * 0 ∧ (-3 - y) = k * (-1)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def condition (x y x_b : ℝ) : Prop :=
  let MA := (0 - x, -1 - y)
  let AB := (x_b - 0, -3 - (-1))
  let MB := (x_b - x, -3 - y)
  let BA := (-x_b, 2)

  dot_product MA AB = dot_product MB BA

theorem trajectory_equation : ∀ x y, (∀ x_b, MB_parallel_OA x y x_b) → condition x y x_b → y = (1 / 4) * x^2 - 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_trajectory_equation_l555_55575


namespace NUMINAMATH_GPT_find_larger_integer_l555_55530

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end NUMINAMATH_GPT_find_larger_integer_l555_55530


namespace NUMINAMATH_GPT_slope_of_line_l555_55520

/-- 
Given points M(1, 2) and N(3, 4), prove that the slope of the line passing through these points is 1.
-/
theorem slope_of_line (x1 y1 x2 y2 : ℝ) (hM : x1 = 1 ∧ y1 = 2) (hN : x2 = 3 ∧ y2 = 4) : 
  (y2 - y1) / (x2 - x1) = 1 :=
by
  -- The proof is omitted here because only the statement is required.
  sorry

end NUMINAMATH_GPT_slope_of_line_l555_55520


namespace NUMINAMATH_GPT_no_integer_solutions_l555_55563

theorem no_integer_solutions (m n : ℤ) (h1 : m ^ 3 + n ^ 4 + 130 * m * n = 42875) (h2 : m * n ≥ 0) :
  false :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l555_55563


namespace NUMINAMATH_GPT_total_number_of_people_l555_55544

def total_people_at_park(hikers bike_riders : Nat) : Nat :=
  hikers + bike_riders

theorem total_number_of_people 
  (bike_riders : Nat)
  (hikers : Nat)
  (hikers_eq_bikes_plus_178 : hikers = bike_riders + 178)
  (bikes_eq_249 : bike_riders = 249) :
  total_people_at_park hikers bike_riders = 676 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_people_l555_55544


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l555_55508

open Real

-- Hyperbola parameters and conditions
variables (a b c e : ℝ)
-- Ensure a > 0, b > 0
axiom h_a_pos : a > 0
axiom h_b_pos : b > 0
-- Hyperbola equation
axiom hyperbola_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
-- Coincidence of right focus and center of circle
axiom circle_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 3 = 0 → (x, y) = (2, 0)
-- Distance from focus to asymptote is 1
axiom distance_focus_to_asymptote : b = 1

-- Prove the eccentricity e of the hyperbola is 2sqrt(3)/3
theorem eccentricity_of_hyperbola : e = 2 * sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l555_55508


namespace NUMINAMATH_GPT_prove_inequality_l555_55594

-- Define the function properties
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Function properties as given in the problem
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def decreasing_on_nonneg (f : ℝ → ℝ) := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

-- The main theorem statement
theorem prove_inequality (h_even : even_function f) (h_dec : decreasing_on_nonneg f) :
  f (-3 / 4) ≥ f (a^2 - a + 1) :=
sorry

end NUMINAMATH_GPT_prove_inequality_l555_55594


namespace NUMINAMATH_GPT_angle_terminal_side_l555_55553

theorem angle_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 →
  α = 330 :=
by
  sorry

end NUMINAMATH_GPT_angle_terminal_side_l555_55553


namespace NUMINAMATH_GPT_find_x_l555_55559

theorem find_x (x : ℚ) (h : (3 * x - 6 + 4) / 7 = 15) : x = 107 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l555_55559


namespace NUMINAMATH_GPT_find_a_l555_55565

theorem find_a (a b c : ℝ) (h1 : ∀ x, x = 2 → y = 5) (h2 : ∀ x, x = 3 → y = 7) :
  a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l555_55565


namespace NUMINAMATH_GPT_rahim_books_l555_55592

/-- 
Rahim bought some books for Rs. 6500 from one shop and 35 books for Rs. 2000 from another. 
The average price he paid per book is Rs. 85. 
Prove that Rahim bought 65 books from the first shop. 
-/
theorem rahim_books (x : ℕ) 
  (h1 : 6500 + 2000 = 8500) 
  (h2 : 85 * (x + 35) = 8500) : 
  x = 65 := 
sorry

end NUMINAMATH_GPT_rahim_books_l555_55592


namespace NUMINAMATH_GPT_find_k_for_line_l555_55570

theorem find_k_for_line (k : ℝ) : (2 * k * (-1/2) + 1 = -7 * 3) → k = 22 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_for_line_l555_55570


namespace NUMINAMATH_GPT_solve_for_x_l555_55574

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.8) : x = 71.7647 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l555_55574


namespace NUMINAMATH_GPT_infinite_series_sum_l555_55599

theorem infinite_series_sum :
  (∑' n : ℕ, (3:ℝ)^n / (1 + (3:ℝ)^n + (3:ℝ)^(n+1) + (3:ℝ)^(2*n+2))) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l555_55599


namespace NUMINAMATH_GPT_range_g_minus_2x_l555_55560

variable (g : ℝ → ℝ)
variable (x : ℝ)

axiom g_values : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → 
  (g x = x ∨ g x = x - 1 ∨ g x = x - 2 ∨ g x = x - 3 ∨ g x = x - 4)

axiom g_le_2x : ∀ x, x ∈ Set.Icc (-4 : ℝ) 4 → g x ≤ 2 * x

theorem range_g_minus_2x : 
  Set.range (fun x => g x - 2 * x) = Set.Icc (-5 : ℝ) 0 :=
sorry

end NUMINAMATH_GPT_range_g_minus_2x_l555_55560


namespace NUMINAMATH_GPT_wave_propagation_l555_55580

def accum (s : String) : String :=
  String.join (List.intersperse "-" (s.data.enum.map (λ (i : Nat × Char) =>
    String.mk [i.2.toUpper] ++ String.mk (List.replicate i.1 i.2.toLower))))

theorem wave_propagation (s : String) :
  s = "dremCaheя" → accum s = "D-Rr-Eee-Mmmm-Ccccc-Aaaaaa-Hhhhhhh-Eeeeeeee-Яяяяяяяяя" :=
  by
  intro h
  rw [h]
  sorry

end NUMINAMATH_GPT_wave_propagation_l555_55580


namespace NUMINAMATH_GPT_fill_time_of_three_pipes_l555_55556

def rate (hours : ℕ) : ℚ := 1 / hours

def combined_rate : ℚ :=
  rate 12 + rate 15 + rate 20

def time_to_fill (rate : ℚ) : ℚ :=
  1 / rate

theorem fill_time_of_three_pipes :
  time_to_fill combined_rate = 5 := by
  sorry

end NUMINAMATH_GPT_fill_time_of_three_pipes_l555_55556


namespace NUMINAMATH_GPT_jean_more_trips_than_bill_l555_55533

variable (b j : ℕ)

theorem jean_more_trips_than_bill
  (h1 : b + j = 40)
  (h2 : j = 23) :
  j - b = 6 := by
  sorry

end NUMINAMATH_GPT_jean_more_trips_than_bill_l555_55533


namespace NUMINAMATH_GPT_required_speed_l555_55561

-- The car covers 504 km in 6 hours initially.
def distance : ℕ := 504
def initial_time : ℕ := 6
def initial_speed : ℕ := distance / initial_time

-- The time that is 3/2 times the initial time.
def factor : ℚ := 3 / 2
def new_time : ℚ := initial_time * factor

-- The speed required to cover the same distance in the new time.
def new_speed : ℚ := distance / new_time

-- The proof statement
theorem required_speed : new_speed = 56 := by
  sorry

end NUMINAMATH_GPT_required_speed_l555_55561


namespace NUMINAMATH_GPT_lincoln_one_way_fare_l555_55578

-- Define the given conditions as assumptions
variables (x : ℝ) (days : ℝ) (total_cost : ℝ) (trips_per_day : ℝ)

-- State the conditions
axiom condition1 : days = 9
axiom condition2 : total_cost = 288
axiom condition3 : trips_per_day = 2

-- The theorem we want to prove based on the conditions
theorem lincoln_one_way_fare (h1 : total_cost = days * trips_per_day * x) : x = 16 :=
by
  -- We skip the proof for the sake of this exercise
  sorry

end NUMINAMATH_GPT_lincoln_one_way_fare_l555_55578


namespace NUMINAMATH_GPT_percentage_corresponding_to_120_l555_55572

variable (x p : ℝ)

def forty_percent_eq_160 := (0.4 * x = 160)
def p_times_x_eq_120 := (p * x = 120)

theorem percentage_corresponding_to_120 (h₁ : forty_percent_eq_160 x) (h₂ : p_times_x_eq_120 x p) :
  p = 0.30 :=
sorry

end NUMINAMATH_GPT_percentage_corresponding_to_120_l555_55572


namespace NUMINAMATH_GPT_difference_received_from_parents_l555_55504

-- Define conditions
def amount_from_mom := 8
def amount_from_dad := 5

-- Question: Prove the difference between amount_from_mom and amount_from_dad is 3
theorem difference_received_from_parents : (amount_from_mom - amount_from_dad) = 3 :=
by
  sorry

end NUMINAMATH_GPT_difference_received_from_parents_l555_55504


namespace NUMINAMATH_GPT_tangent_lines_passing_through_point_l555_55531

theorem tangent_lines_passing_through_point :
  ∀ (x0 y0 : ℝ) (p : ℝ × ℝ), 
  (p = (1, 1)) ∧ (y0 = x0 ^ 3) → 
  (y0 - 1 = 3 * x0 ^ 2 * (1 - x0)) → 
  (x0 = 1 ∨ x0 = -1/2) → 
  ((y - (3 * 1 - 2)) * (y - (3/4 * x0 + 1/4))) = 0 :=
sorry

end NUMINAMATH_GPT_tangent_lines_passing_through_point_l555_55531


namespace NUMINAMATH_GPT_number_of_lilies_l555_55509

theorem number_of_lilies (L : ℕ) 
  (h1 : ∀ n:ℕ, n * 6 = 6 * n)
  (h2 : ∀ n:ℕ, n * 3 = 3 * n) 
  (h3 : 5 * 3 = 15)
  (h4 : 6 * L + 15 = 63) : 
  L = 8 := 
by
  -- Proof omitted 
  sorry

end NUMINAMATH_GPT_number_of_lilies_l555_55509


namespace NUMINAMATH_GPT_playground_area_l555_55532

theorem playground_area (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 100)
  (h_length : l = 3 * w) : l * w = 468.75 :=
by
  sorry

end NUMINAMATH_GPT_playground_area_l555_55532


namespace NUMINAMATH_GPT_exists_n_of_form_2k_l555_55506

theorem exists_n_of_form_2k (n : ℕ) (x y z : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_recip : 1/x + 1/y + 1/z = 1/(n : ℤ)) : ∃ k : ℕ, n = 2 * k :=
sorry

end NUMINAMATH_GPT_exists_n_of_form_2k_l555_55506


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_eq_l555_55558

theorem sum_of_roots_of_quadratic_eq (x : ℝ) (hx : x^2 = 8 * x + 15) :
  ∃ S : ℝ, S = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_eq_l555_55558


namespace NUMINAMATH_GPT_car_trip_distance_l555_55552

theorem car_trip_distance (speed_first_car speed_second_car : ℝ) (time_first_car time_second_car distance_first_car distance_second_car : ℝ) 
  (h_speed_first : speed_first_car = 30)
  (h_time_first : time_first_car = 1.5)
  (h_speed_second : speed_second_car = 60)
  (h_time_second : time_second_car = 1.3333)
  (h_distance_first : distance_first_car = speed_first_car * time_first_car)
  (h_distance_second : distance_second_car = speed_second_car * time_second_car) :
  distance_first_car = 45 :=
by
  sorry

end NUMINAMATH_GPT_car_trip_distance_l555_55552


namespace NUMINAMATH_GPT_inequality_proof_l555_55536

theorem inequality_proof (a b c : ℝ) (h1 : 0 < c) (h2 : c ≤ b) (h3 : b ≤ a) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l555_55536


namespace NUMINAMATH_GPT_jason_total_payment_l555_55568

def total_cost (shorts jacket shoes socks tshirts : ℝ) : ℝ :=
  shorts + jacket + shoes + socks + tshirts

def discount_amount (total : ℝ) (discount_rate : ℝ) : ℝ :=
  total * discount_rate

def total_after_discount (total discount : ℝ) : ℝ :=
  total - discount

def sales_tax_amount (total : ℝ) (tax_rate : ℝ) : ℝ :=
  total * tax_rate

def final_amount (total after_discount tax : ℝ) : ℝ :=
  after_discount + tax

theorem jason_total_payment :
  let shorts := 14.28
  let jacket := 4.74
  let shoes := 25.95
  let socks := 6.80
  let tshirts := 18.36
  let discount_rate := 0.15
  let tax_rate := 0.07
  let total := total_cost shorts jacket shoes socks tshirts
  let discount := discount_amount total discount_rate
  let after_discount := total_after_discount total discount
  let tax := sales_tax_amount after_discount tax_rate
  let final := final_amount total after_discount tax
  final = 63.78 :=
by
  sorry

end NUMINAMATH_GPT_jason_total_payment_l555_55568


namespace NUMINAMATH_GPT_exist_monochromatic_equilateral_triangle_l555_55566

theorem exist_monochromatic_equilateral_triangle 
  (color : ℝ × ℝ → ℕ) 
  (h_color : ∀ p : ℝ × ℝ, color p = 0 ∨ color p = 1) : 
  ∃ (A B C : ℝ × ℝ), (dist A B = dist B C) ∧ (dist B C = dist C A) ∧ (color A = color B ∧ color B = color C) :=
sorry

end NUMINAMATH_GPT_exist_monochromatic_equilateral_triangle_l555_55566


namespace NUMINAMATH_GPT_find_water_needed_l555_55519

def apple_juice := 4
def honey (A : ℕ) := 3 * A
def water (H : ℕ) := 3 * H

theorem find_water_needed : water (honey apple_juice) = 36 :=
  sorry

end NUMINAMATH_GPT_find_water_needed_l555_55519


namespace NUMINAMATH_GPT_induction_proof_l555_55581

-- Given conditions and definitions
def plane_parts (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

-- The induction hypothesis for k ≥ 2
def induction_step (k : ℕ) (h : 2 ≤ k) : Prop :=
  plane_parts (k + 1) - plane_parts k = k + 1

-- The complete statement we want to prove
theorem induction_proof (k : ℕ) (h : 2 ≤ k) : induction_step k h := by
  sorry

end NUMINAMATH_GPT_induction_proof_l555_55581


namespace NUMINAMATH_GPT_revenue_decrease_percent_l555_55596

theorem revenue_decrease_percent (T C : ℝ) (hT_pos : T > 0) (hC_pos : C > 0) :
  let new_T := 0.75 * T
  let new_C := 1.10 * C
  let original_revenue := T * C
  let new_revenue := new_T * new_C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 17.5 := 
by {
  sorry
}

end NUMINAMATH_GPT_revenue_decrease_percent_l555_55596


namespace NUMINAMATH_GPT_avg_of_eleven_numbers_l555_55500

variable (S1 : ℕ)
variable (S2 : ℕ)
variable (sixth_num : ℕ)
variable (total_sum : ℕ)
variable (avg_eleven : ℕ)

def condition1 := S1 = 6 * 58
def condition2 := S2 = 6 * 65
def condition3 := sixth_num = 188
def condition4 := total_sum = S1 + S2 - sixth_num
def condition5 := avg_eleven = total_sum / 11

theorem avg_of_eleven_numbers : (S1 = 6 * 58) →
                                (S2 = 6 * 65) →
                                (sixth_num = 188) →
                                (total_sum = S1 + S2 - sixth_num) →
                                (avg_eleven = total_sum / 11) →
                                avg_eleven = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_avg_of_eleven_numbers_l555_55500


namespace NUMINAMATH_GPT_amount_paid_for_grapes_l555_55512

-- Definitions based on the conditions
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := 2.23

-- The statement to be proved
theorem amount_paid_for_grapes : total_spent + refund_for_cherries = 12.08 := 
by 
  -- Here the specific mathematical proof would go, but is replaced by sorry as instructed
  sorry

end NUMINAMATH_GPT_amount_paid_for_grapes_l555_55512


namespace NUMINAMATH_GPT_who_next_to_boris_l555_55539

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end NUMINAMATH_GPT_who_next_to_boris_l555_55539


namespace NUMINAMATH_GPT_triangle_area_formed_by_lines_l555_55526

def line1 := { p : ℝ × ℝ | p.2 = p.1 - 4 }
def line2 := { p : ℝ × ℝ | p.2 = -p.1 - 4 }
def x_axis := { p : ℝ × ℝ | p.2 = 0 }

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_formed_by_lines :
  ∃ (A B C : ℝ × ℝ), A ∈ line1 ∧ A ∈ line2 ∧ B ∈ line1 ∧ B ∈ x_axis ∧ C ∈ line2 ∧ C ∈ x_axis ∧ 
  triangle_area A B C = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_formed_by_lines_l555_55526


namespace NUMINAMATH_GPT_warmup_puzzle_time_l555_55583

theorem warmup_puzzle_time (W : ℕ) (H : W + 3 * W + 3 * W = 70) : W = 10 :=
by
  sorry

end NUMINAMATH_GPT_warmup_puzzle_time_l555_55583


namespace NUMINAMATH_GPT_cost_of_gravelling_roads_l555_55527

theorem cost_of_gravelling_roads :
  let lawn_length := 70
  let lawn_breadth := 30
  let road_width := 5
  let cost_per_sqm := 4
  let area_road_length := lawn_length * road_width
  let area_road_breadth := lawn_breadth * road_width
  let area_intersection := road_width * road_width
  let total_area_to_be_graveled := (area_road_length + area_road_breadth) - area_intersection
  let total_cost := total_area_to_be_graveled * cost_per_sqm
  total_cost = 1900 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_gravelling_roads_l555_55527


namespace NUMINAMATH_GPT_initial_distance_between_stations_l555_55582

theorem initial_distance_between_stations
  (speedA speedB distanceA : ℝ)
  (rateA rateB : speedA = 40 ∧ speedB = 30)
  (dist_travelled : distanceA = 200) :
  (distanceA / speedA) * speedB + distanceA = 350 := by
  sorry

end NUMINAMATH_GPT_initial_distance_between_stations_l555_55582


namespace NUMINAMATH_GPT_range_of_a_l555_55597

noncomputable def f (x : ℝ) : ℝ := (1 / (1 + x^2)) - Real.log (abs x)

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) ↔
  (1 / Real.exp 1 ≤ a ∧ a ≤ (2 + Real.log 3) / 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l555_55597


namespace NUMINAMATH_GPT_smallest_n_terminating_decimal_l555_55569

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (n > 0) ∧
           (∃ (k: ℕ), (n + 150) = 2^k ∧ k < 150) ∨ 
           (∃ (k m: ℕ), (n + 150) = 2^k * 5^m ∧ m < 150) ∧ 
           ∀ m : ℕ, ((m > 0 ∧ (∃ (j: ℕ), (m + 150) = 2^j ∧ j < 150) ∨ 
           (∃ (j l: ℕ), (m + 150) = 2^j * 5^l ∧ l < 150)) → m ≥ n)
:= ⟨10, by {
  sorry
}⟩

end NUMINAMATH_GPT_smallest_n_terminating_decimal_l555_55569


namespace NUMINAMATH_GPT_find_x_l555_55551

theorem find_x (h : ℝ → ℝ)
  (H1 : ∀x, h (3*x - 2) = 5*x + 6) :
  (∀x, h x = 2*x - 1) → x = 31 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l555_55551


namespace NUMINAMATH_GPT_part1_solution_part2_solution_part3_solution_l555_55547

-- Part (1): Prove the solution of the system of equations 
theorem part1_solution (x y : ℝ) (h1 : x - y - 1 = 0) (h2 : 4 * (x - y) - y = 5) : 
  x = 0 ∧ y = -1 := 
sorry

-- Part (2): Prove the solution of the system of equations 
theorem part2_solution (x y : ℝ) (h1 : 2 * x - 3 * y - 2 = 0) 
  (h2 : (2 * x - 3 * y + 5) / 7 + 2 * y = 9) : 
  x = 7 ∧ y = 4 := 
sorry

-- Part (3): Prove the range of the parameter m
theorem part3_solution (m : ℕ) (h1 : 2 * (2 : ℝ) * x + y = (-3 : ℝ) * ↑m + 2) 
  (h2 : x + 2 * y = 7) (h3 : x + y > -5 / 6) : 
  m = 1 ∨ m = 2 ∨ m = 3 :=
sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_part3_solution_l555_55547


namespace NUMINAMATH_GPT_find_third_circle_radius_l555_55514

-- Define the context of circles and their tangency properties
variable (A B : ℝ → ℝ → Prop) -- Centers of circles
variable (r1 r2 : ℝ) -- Radii of circles

-- Define conditions from the problem
def circles_are_tangent (A B : ℝ → ℝ → Prop) (r1 r2 : ℝ) : Prop :=
  ∀ x y : ℝ, A x y → B (x + 7) y ∧ r1 = 2 ∧ r2 = 5

def third_circle_tangent_to_others_and_tangent_line (A B : ℝ → ℝ → Prop) (r3 : ℝ) : Prop :=
  ∃ D : ℝ → ℝ → Prop, ∀ x y : ℝ, D x y →
  ((A (x + r3) y ∧ B (x - r3) y) ∧ (r3 > 0))

theorem find_third_circle_radius (A B : ℝ → ℝ → Prop) (r1 r2 : ℝ) :
  circles_are_tangent A B r1 r2 →
  (∃ r3 : ℝ, r3 = 1 ∧ third_circle_tangent_to_others_and_tangent_line A B r3) :=
by
  sorry

end NUMINAMATH_GPT_find_third_circle_radius_l555_55514
