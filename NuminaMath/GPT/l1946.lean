import Mathlib

namespace NUMINAMATH_GPT_alcohol_water_ratio_l1946_194626

theorem alcohol_water_ratio 
  (P_alcohol_pct : ℝ) (Q_alcohol_pct : ℝ) 
  (P_volume : ℝ) (Q_volume : ℝ) 
  (mixture_alcohol : ℝ) (mixture_water : ℝ)
  (h1 : P_alcohol_pct = 62.5)
  (h2 : Q_alcohol_pct = 87.5)
  (h3 : P_volume = 4)
  (h4 : Q_volume = 4)
  (ha : mixture_alcohol = (P_volume * (P_alcohol_pct / 100)) + (Q_volume * (Q_alcohol_pct / 100)))
  (hm : mixture_water = (P_volume + Q_volume) - mixture_alcohol) :
  mixture_alcohol / mixture_water = 3 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_water_ratio_l1946_194626


namespace NUMINAMATH_GPT_lines_intersect_at_l1946_194695

def Line1 (t : ℝ) : ℝ × ℝ :=
  let x := 1 + 3 * t
  let y := 2 - t
  (x, y)

def Line2 (u : ℝ) : ℝ × ℝ :=
  let x := -1 + 4 * u
  let y := 4 + 3 * u
  (x, y)

theorem lines_intersect_at :
  ∃ t u : ℝ, Line1 t = Line2 u ∧
             Line1 t = (-53 / 17, 56 / 17) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_l1946_194695


namespace NUMINAMATH_GPT_tan_pink_violet_probability_l1946_194650

noncomputable def probability_tan_pink_violet_consecutive_order : ℚ :=
  let num_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5)
  let total_ways := Nat.factorial 12
  num_ways / total_ways

theorem tan_pink_violet_probability :
  probability_tan_pink_violet_consecutive_order = 1 / 27720 := by
  sorry

end NUMINAMATH_GPT_tan_pink_violet_probability_l1946_194650


namespace NUMINAMATH_GPT_abs_minus_five_plus_three_l1946_194628

theorem abs_minus_five_plus_three : |(-5 + 3)| = 2 := 
by
  sorry

end NUMINAMATH_GPT_abs_minus_five_plus_three_l1946_194628


namespace NUMINAMATH_GPT_train_passes_jogger_in_37_seconds_l1946_194634

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_lead_m : ℝ := 250
noncomputable def train_length_m : ℝ := 120

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def total_distance_m : ℝ := jogger_lead_m + train_length_m

theorem train_passes_jogger_in_37_seconds :
  total_distance_m / relative_speed_mps = 37 := by
  sorry

end NUMINAMATH_GPT_train_passes_jogger_in_37_seconds_l1946_194634


namespace NUMINAMATH_GPT_find_unit_prices_minimal_cost_l1946_194668

-- Definitions for part 1
def unitPrices (x y : ℕ) : Prop :=
  20 * x + 30 * y = 2920 ∧ x - y = 11 

-- Definitions for part 2
def costFunction (m : ℕ) : ℕ :=
  52 * m + 48 * (40 - m)

def additionalPurchase (m : ℕ) : Prop :=
  m ≥ 40 / 3

-- Statement for unit prices proof
theorem find_unit_prices (x y : ℕ) (h1 : 20 * x + 30 * y = 2920) (h2 : x - y = 11) : x = 65 ∧ y = 54 := 
  sorry

-- Statement for minimal cost proof
theorem minimal_cost (m : ℕ) (x y : ℕ) 
  (hx : 20 * x + 30 * y = 2920) 
  (hy : x - y = 11)
  (hx_65 : x = 65)
  (hy_54 : y = 54)
  (hm : m ≥ 40 / 3) : 
  costFunction m = 1976 ∧ m = 14 :=
  sorry

end NUMINAMATH_GPT_find_unit_prices_minimal_cost_l1946_194668


namespace NUMINAMATH_GPT_fraction_black_part_l1946_194681

theorem fraction_black_part (L : ℝ) (blue_part : ℝ) (white_part_fraction : ℝ) 
  (h1 : L = 8) (h2 : blue_part = 3.5) (h3 : white_part_fraction = 0.5) : 
  (8 - (3.5 + 0.5 * (8 - 3.5))) / 8 = 9 / 32 :=
by
  sorry

end NUMINAMATH_GPT_fraction_black_part_l1946_194681


namespace NUMINAMATH_GPT_petes_average_speed_l1946_194663

theorem petes_average_speed
    (map_distance : ℝ := 5) 
    (time_taken : ℝ := 1.5) 
    (map_scale : ℝ := 0.05555555555555555) :
    (map_distance / map_scale) / time_taken = 60 := 
by
    sorry

end NUMINAMATH_GPT_petes_average_speed_l1946_194663


namespace NUMINAMATH_GPT_expand_and_simplify_l1946_194620

theorem expand_and_simplify (x : ℝ) : (x + 6) * (x - 11) = x^2 - 5 * x - 66 :=
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l1946_194620


namespace NUMINAMATH_GPT_friend_jogging_time_l1946_194652

theorem friend_jogging_time (D : ℝ) (my_time : ℝ) (friend_speed : ℝ) :
  my_time = 3 * 60 →
  friend_speed = 2 * (D / my_time) →
  (D / friend_speed) = 90 :=
by
  sorry

end NUMINAMATH_GPT_friend_jogging_time_l1946_194652


namespace NUMINAMATH_GPT_joan_dimes_spent_l1946_194610

theorem joan_dimes_spent (initial_dimes remaining_dimes spent_dimes : ℕ) 
    (h_initial: initial_dimes = 5) 
    (h_remaining: remaining_dimes = 3) : 
    spent_dimes = initial_dimes - remaining_dimes := 
by 
    sorry

end NUMINAMATH_GPT_joan_dimes_spent_l1946_194610


namespace NUMINAMATH_GPT_vanessa_points_l1946_194694

theorem vanessa_points (total_points : ℕ) (num_other_players : ℕ) (avg_points_other : ℕ) 
  (h1 : total_points = 65) (h2 : num_other_players = 7) (h3 : avg_points_other = 5) :
  ∃ vp : ℕ, vp = 30 :=
by
  sorry

end NUMINAMATH_GPT_vanessa_points_l1946_194694


namespace NUMINAMATH_GPT_distinct_solutions_of_transformed_eq_l1946_194614

open Function

variable {R : Type} [Field R]

def cubic_func (a b c d : R) (x : R) : R := a*x^3 + b*x^2 + c*x + d

noncomputable def three_distinct_roots {a b c d : R} (f : R → R)
  (h : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) : Prop :=
∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ f α = 0 ∧ f β = 0 ∧ f γ = 0

theorem distinct_solutions_of_transformed_eq
  {a b c d : R} (h : ∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ (cubic_func a b c d α) = 0 ∧ (cubic_func a b c d β) = 0 ∧ (cubic_func a b c d γ) = 0) :
  ∃ p q, p ≠ q ∧ (4 * (cubic_func a b c d p) * (3 * a * p + b) = (3 * a * p^2 + 2 * b * p + c)^2) ∧ 
              (4 * (cubic_func a b c d q) * (3 * a * q + b) = (3 * a * q^2 + 2 * b * q + c)^2) := sorry

end NUMINAMATH_GPT_distinct_solutions_of_transformed_eq_l1946_194614


namespace NUMINAMATH_GPT_max_min_of_f_in_M_l1946_194671

noncomputable def domain (x : ℝ) : Prop := 3 - 4*x + x^2 > 0

def M : Set ℝ := { x | domain x }

noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 3 * 4^x

theorem max_min_of_f_in_M :
  ∃ (xₘ xₘₐₓ : ℝ), xₘ ∈ M ∧ xₘₐₓ ∈ M ∧ 
  (∀ x ∈ M, f xₘₐₓ ≥ f x) ∧ 
  (∀ x ∈ M, f xₘ ≠ f xₓₐₓ) :=
by
  sorry

end NUMINAMATH_GPT_max_min_of_f_in_M_l1946_194671


namespace NUMINAMATH_GPT_sum_x_y_is_4_l1946_194613

theorem sum_x_y_is_4 {x y : ℝ} (h : x / (1 - (I : ℂ)) + y / (1 - 2 * I) = 5 / (1 - 3 * I)) : x + y = 4 :=
sorry

end NUMINAMATH_GPT_sum_x_y_is_4_l1946_194613


namespace NUMINAMATH_GPT_number_of_hockey_players_l1946_194654

theorem number_of_hockey_players 
  (cricket_players : ℕ) 
  (football_players : ℕ) 
  (softball_players : ℕ) 
  (total_players : ℕ) 
  (hockey_players : ℕ) 
  (h1 : cricket_players = 10) 
  (h2 : football_players = 16) 
  (h3 : softball_players = 13) 
  (h4 : total_players = 51) 
  (calculation : hockey_players = total_players - (cricket_players + football_players + softball_players)) : 
  hockey_players = 12 :=
by 
  rw [h1, h2, h3, h4] at calculation
  exact calculation

end NUMINAMATH_GPT_number_of_hockey_players_l1946_194654


namespace NUMINAMATH_GPT_min_value_x_2y_l1946_194646

theorem min_value_x_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y + 2*x*y = 8) : x + 2*y ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_x_2y_l1946_194646


namespace NUMINAMATH_GPT_translation_coordinates_l1946_194655

theorem translation_coordinates (A : ℝ × ℝ) (T : ℝ × ℝ) (A' : ℝ × ℝ) 
  (hA : A = (-4, 3)) (hT : T = (2, 0)) (hA' : A' = (A.1 + T.1, A.2 + T.2)) : 
  A' = (-2, 3) := sorry

end NUMINAMATH_GPT_translation_coordinates_l1946_194655


namespace NUMINAMATH_GPT_ellipse_equation_l1946_194625

theorem ellipse_equation {a b : ℝ} 
  (center_origin : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → x + y = 0)
  (foci_on_x : ∀ c : ℝ, c = a / 2)
  (perimeter_triangle : ∀ A B : ℝ, A + B + 2 * c = 16) :
  a = 4 ∧ b^2 = 12 → (∀ x y : ℝ, x^2/16 + y^2/12 = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l1946_194625


namespace NUMINAMATH_GPT_minimum_value_F_l1946_194657

noncomputable def minimum_value_condition (x y : ℝ) : Prop :=
  x^2 + y^2 + 25 = 10 * (x + y)

noncomputable def F (x y : ℝ) : ℝ :=
  6 * y + 8 * x - 9

theorem minimum_value_F :
  (∃ x y : ℝ, minimum_value_condition x y) → ∃ x y : ℝ, minimum_value_condition x y ∧ F x y = 11 :=
sorry

end NUMINAMATH_GPT_minimum_value_F_l1946_194657


namespace NUMINAMATH_GPT_fraction_addition_l1946_194637

theorem fraction_addition :
  (1 / 6) + (1 / 3) + (5 / 9) = 19 / 18 :=
by
  sorry

end NUMINAMATH_GPT_fraction_addition_l1946_194637


namespace NUMINAMATH_GPT_quadratic_has_solutions_l1946_194615

theorem quadratic_has_solutions :
  (1 + Real.sqrt 2)^2 - 2 * (1 + Real.sqrt 2) - 1 = 0 ∧ 
  (1 - Real.sqrt 2)^2 - 2 * (1 - Real.sqrt 2) - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_solutions_l1946_194615


namespace NUMINAMATH_GPT_g_of_f_three_l1946_194602

def f (x : ℝ) : ℝ := x^3 - 2
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_of_f_three : g (f 3) = 1902 :=
by
  sorry

end NUMINAMATH_GPT_g_of_f_three_l1946_194602


namespace NUMINAMATH_GPT_pump_no_leak_fill_time_l1946_194691

noncomputable def pump_fill_time (P t l : ℝ) :=
  1 / P - 1 / l = 1 / t

theorem pump_no_leak_fill_time :
  ∃ P : ℝ, pump_fill_time P (13 / 6) 26 ∧ P = 2 :=
by
  sorry

end NUMINAMATH_GPT_pump_no_leak_fill_time_l1946_194691


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l1946_194619

theorem bus_speed_excluding_stoppages (v : Real) 
  (h1 : ∀ x, x = 41) 
  (h2 : ∀ y, y = 14.444444444444443 / 60) : 
  v = 54 := 
by
  -- Proving the statement. Proof steps are skipped.
  sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l1946_194619


namespace NUMINAMATH_GPT_red_pairs_count_l1946_194600

def num_green_students : Nat := 63
def num_red_students : Nat := 69
def total_pairs : Nat := 66
def num_green_pairs : Nat := 27

theorem red_pairs_count : 
  (num_red_students - (num_green_students - num_green_pairs * 2)) / 2 = 30 := 
by sorry

end NUMINAMATH_GPT_red_pairs_count_l1946_194600


namespace NUMINAMATH_GPT_assorted_candies_count_l1946_194678

theorem assorted_candies_count
  (total_candies : ℕ)
  (chewing_gums : ℕ)
  (chocolate_bars : ℕ)
  (assorted_candies : ℕ) :
  total_candies = 50 →
  chewing_gums = 15 →
  chocolate_bars = 20 →
  assorted_candies = total_candies - (chewing_gums + chocolate_bars) →
  assorted_candies = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_assorted_candies_count_l1946_194678


namespace NUMINAMATH_GPT_sandcastle_ratio_l1946_194618

-- Definitions based on conditions in a)
def sandcastles_on_marks_beach : ℕ := 20
def towers_per_sandcastle_marks_beach : ℕ := 10
def towers_per_sandcastle_jeffs_beach : ℕ := 5
def total_combined_sandcastles_and_towers : ℕ := 580

-- The main statement to prove
theorem sandcastle_ratio : 
  ∃ (J : ℕ), 
  (sandcastles_on_marks_beach + (towers_per_sandcastle_marks_beach * sandcastles_on_marks_beach) + J + (towers_per_sandcastle_jeffs_beach * J) = total_combined_sandcastles_and_towers) ∧ 
  (J / sandcastles_on_marks_beach = 3) :=
by 
  sorry

end NUMINAMATH_GPT_sandcastle_ratio_l1946_194618


namespace NUMINAMATH_GPT_nate_cooking_for_people_l1946_194670

/-- Given that 8 jumbo scallops weigh one pound, scallops cost $24.00 per pound, Nate is pairing 2 scallops with a corn bisque per person, and he spends $48 on scallops. We want to prove that Nate is cooking for 8 people. -/
theorem nate_cooking_for_people :
  (8 : ℕ) = 8 →
  (24 : ℕ) = 24 →
  (2 : ℕ) = 2 →
  (48 : ℕ) = 48 →
  let scallops_per_pound := 8
  let cost_per_pound := 24
  let scallops_per_person := 2
  let money_spent := 48
  let pounds_of_scallops := money_spent / cost_per_pound
  let total_scallops := scallops_per_pound * pounds_of_scallops
  let people := total_scallops / scallops_per_person
  people = 8 :=
by
  sorry

end NUMINAMATH_GPT_nate_cooking_for_people_l1946_194670


namespace NUMINAMATH_GPT_minimum_distance_from_circle_to_line_l1946_194622

noncomputable def point_on_circle (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

def line_eq (p : ℝ × ℝ) : ℝ :=
  p.1 - p.2 + 4

noncomputable def distance_from_point_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + 4| / Real.sqrt (1^2 + 1^2)

theorem minimum_distance_from_circle_to_line :
  ∀ θ : ℝ, (∃ θ, distance_from_point_to_line (point_on_circle θ) = 2 * Real.sqrt 2 - 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_distance_from_circle_to_line_l1946_194622


namespace NUMINAMATH_GPT_case_one_case_two_l1946_194667

theorem case_one (n : ℝ) (h : n > -1) : n^3 + 1 > n^2 + n :=
sorry

theorem case_two (n : ℝ) (h : n < -1) : n^3 + 1 < n^2 + n :=
sorry

end NUMINAMATH_GPT_case_one_case_two_l1946_194667


namespace NUMINAMATH_GPT_smallest_integer_remainder_l1946_194677

theorem smallest_integer_remainder (b : ℕ) :
  (b ≡ 2 [MOD 3]) ∧ (b ≡ 3 [MOD 5]) → b = 8 := 
by
  sorry

end NUMINAMATH_GPT_smallest_integer_remainder_l1946_194677


namespace NUMINAMATH_GPT_sum_of_possible_b_values_l1946_194697

noncomputable def g (x b : ℝ) : ℝ := x^2 - b * x + 3 * b

theorem sum_of_possible_b_values :
  (∀ (x₀ x₁ : ℝ), g x₀ x₁ = 0 → g x₀ x₁ = (x₀ - x₁) * (x₀ - 3)) → ∃ b : ℝ, b = 12 ∨ b = 16 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_b_values_l1946_194697


namespace NUMINAMATH_GPT_sum_of_integers_l1946_194608

theorem sum_of_integers (x y : ℤ) (h1 : x ^ 2 + y ^ 2 = 130) (h2 : x * y = 36) (h3 : x - y = 4) : x + y = 4 := 
by sorry

end NUMINAMATH_GPT_sum_of_integers_l1946_194608


namespace NUMINAMATH_GPT_faith_change_l1946_194666

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def two_twenty_bills : ℕ := 2 * 20
def loose_coins : ℕ := 3
def total_cost : ℕ := flour_cost + cake_stand_cost
def total_given : ℕ := two_twenty_bills + loose_coins
def change : ℕ := total_given - total_cost

theorem faith_change : change = 10 := by
  sorry

end NUMINAMATH_GPT_faith_change_l1946_194666


namespace NUMINAMATH_GPT_third_median_length_l1946_194648

theorem third_median_length (a b: ℝ) (h_a: a = 5) (h_b: b = 8)
  (area: ℝ) (h_area: area = 6 * Real.sqrt 15) (m: ℝ):
  m = 3 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_third_median_length_l1946_194648


namespace NUMINAMATH_GPT_cubical_tank_water_volume_l1946_194661

theorem cubical_tank_water_volume 
    (s : ℝ) -- side length of the cube in feet
    (h_fill : 1 / 4 * s = 1) -- tank is filled to 0.25 of its capacity, water level is 1 foot
    (h_volume_water : 0.25 * (s ^ 3) = 16) -- 0.25 of the tank's total volume is the volume of water
    : s ^ 3 = 64 := 
by
  sorry

end NUMINAMATH_GPT_cubical_tank_water_volume_l1946_194661


namespace NUMINAMATH_GPT_simplify_cbrt_expr_l1946_194616

-- Define the cube root function.
def cbrt (x : ℝ) : ℝ := x^(1/3)

-- Define the original expression under the cube root.
def original_expr : ℝ := 40^3 + 70^3 + 100^3

-- Define the simplified expression.
def simplified_expr : ℝ := 10 * cbrt 1407

theorem simplify_cbrt_expr : cbrt original_expr = simplified_expr := by
  -- Declaration that proof is not provided to ensure Lean statement is complete.
  sorry

end NUMINAMATH_GPT_simplify_cbrt_expr_l1946_194616


namespace NUMINAMATH_GPT_minimum_value_1_minimum_value_2_l1946_194653

noncomputable section

open Real -- Use the real numbers

theorem minimum_value_1 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + y^2 + z^2 >= 8 / 3 :=
by
  sorry  -- Proof omitted
 
theorem minimum_value_2 (x y z : ℝ) (h : x - 2 * y + z = 4) : x^2 + (y - 1)^2 + z^2 >= 6 :=
by
  sorry  -- Proof omitted

end NUMINAMATH_GPT_minimum_value_1_minimum_value_2_l1946_194653


namespace NUMINAMATH_GPT_factorize_polynomial_l1946_194696

theorem factorize_polynomial (x y : ℝ) : 2 * x^3 - 8 * x^2 * y + 8 * x * y^2 = 2 * x * (x - 2 * y) ^ 2 := 
by sorry

end NUMINAMATH_GPT_factorize_polynomial_l1946_194696


namespace NUMINAMATH_GPT_unique_hexagon_angles_sides_identity_1_identity_2_l1946_194640

noncomputable def lengths_angles_determined 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : Prop :=
  -- Assuming this is the expression we need to handle:
  ∀ (δ ε ζ : ℝ),
    δ = 180 - α ∧
    ε = 180 - β ∧
    ζ = 180 - γ →
  ∃ (angles_determined : Prop),
    angles_determined

theorem unique_hexagon_angles_sides 
  (a b c d e f : ℝ) (α β γ : ℝ) 
  (h₀ : α + β + γ < 180) : 
  lengths_angles_determined a b c d e f α β γ h₀ :=
sorry

theorem identity_1 
  (a b c d : ℝ) 
  (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) : 
  (1 / a + 1 / c = 1 / b + 1 / d) :=
sorry

theorem identity_2 
  (a b c d e f : ℝ) 
  (h₀ : true) : 
  ((a + f) * (b + d) * (c + e) = (a + e) * (b + f) * (c + d)) :=
sorry

end NUMINAMATH_GPT_unique_hexagon_angles_sides_identity_1_identity_2_l1946_194640


namespace NUMINAMATH_GPT_pipe_length_difference_l1946_194631

theorem pipe_length_difference 
  (total_length shorter_piece : ℝ)
  (h1: total_length = 68) 
  (h2: shorter_piece = 28) : 
  ∃ longer_piece diff : ℝ, longer_piece = total_length - shorter_piece ∧ diff = longer_piece - shorter_piece ∧ diff = 12 :=
by
  sorry

end NUMINAMATH_GPT_pipe_length_difference_l1946_194631


namespace NUMINAMATH_GPT_find_chord_points_l1946_194603

/-
Define a parabola and check if the points given form a chord that intersects 
the point (8,4) in the ratio 1:4.
-/

def parabola (P : ℝ × ℝ) : Prop :=
  P.snd^2 = 4 * P.fst

def divides_in_ratio (C A B : ℝ × ℝ) (m n : ℝ) : Prop :=
  (A.fst * n + B.fst * m = C.fst * (m + n)) ∧ 
  (A.snd * n + B.snd * m = C.snd * (m + n))

theorem find_chord_points :
  ∃ (P1 P2 : ℝ × ℝ),
  parabola P1 ∧
  parabola P2 ∧
  divides_in_ratio (8, 4) P1 P2 1 4 ∧ 
  ((P1 = (1, 2) ∧ P2 = (36, 12)) ∨ (P1 = (9, 6) ∧ P2 = (4, -4))) :=
sorry

end NUMINAMATH_GPT_find_chord_points_l1946_194603


namespace NUMINAMATH_GPT_total_eggs_needed_l1946_194699

-- Define the conditions
def eggsFromAndrew : ℕ := 155
def eggsToBuy : ℕ := 67

-- Define the total number of eggs
def totalEggs : ℕ := eggsFromAndrew + eggsToBuy

-- The theorem to be proven
theorem total_eggs_needed : totalEggs = 222 := by
  sorry

end NUMINAMATH_GPT_total_eggs_needed_l1946_194699


namespace NUMINAMATH_GPT_factorial_sum_mod_30_l1946_194688

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map factorial |>.sum

def remainder_when_divided_by (m k : Nat) : Nat :=
  m % k

theorem factorial_sum_mod_30 : remainder_when_divided_by (sum_of_factorials 100) 30 = 3 :=
by
  sorry

end NUMINAMATH_GPT_factorial_sum_mod_30_l1946_194688


namespace NUMINAMATH_GPT_ratio_of_cards_lost_l1946_194644

-- Definitions based on the conditions
def purchases_per_week : ℕ := 20
def weeks_per_year : ℕ := 52
def cards_left : ℕ := 520

-- Main statement to be proved
theorem ratio_of_cards_lost (total_cards : ℕ := purchases_per_week * weeks_per_year)
                            (cards_lost : ℕ := total_cards - cards_left) :
                            (cards_lost : ℚ) / total_cards = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_cards_lost_l1946_194644


namespace NUMINAMATH_GPT_range_of_m_l1946_194617

-- Definitions based on the given conditions
def setA : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def setB (m : ℝ) : Set ℝ := {x | 2 * m - 1 < x ∧ x < m + 1}

-- Lean statement of the problem
theorem range_of_m (m : ℝ) (h : setB m ⊆ setA) : m ≥ -1 :=
sorry  -- proof is not required

end NUMINAMATH_GPT_range_of_m_l1946_194617


namespace NUMINAMATH_GPT_sum_of_remaining_digit_is_correct_l1946_194605

-- Define the local value calculation function for a particular digit with its place value
def local_value (digit place_value : ℕ) : ℕ := digit * place_value

-- Define the number in question
def number : ℕ := 2345

-- Define the local values for each digit in their respective place values
def local_value_2 : ℕ := local_value 2 1000
def local_value_3 : ℕ := local_value 3 100
def local_value_4 : ℕ := local_value 4 10
def local_value_5 : ℕ := local_value 5 1

-- Define the given sum of the local values
def given_sum : ℕ := 2345

-- Define the sum of the local values of the digits 2, 3, and 5
def sum_of_other_digits : ℕ := local_value_2 + local_value_3 + local_value_5

-- Define the target sum which is the sum of the local value of the remaining digit
def target_sum : ℕ := given_sum - sum_of_other_digits

-- Prove that the sum of the local value of the remaining digit is equal to 40
theorem sum_of_remaining_digit_is_correct : target_sum = 40 := 
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_sum_of_remaining_digit_is_correct_l1946_194605


namespace NUMINAMATH_GPT_equal_share_of_marbles_l1946_194606

-- Define the number of marbles bought by each friend based on the conditions
def wolfgang_marbles : ℕ := 16
def ludo_marbles : ℕ := wolfgang_marbles + (wolfgang_marbles / 4)
def michael_marbles : ℕ := 2 * (wolfgang_marbles + ludo_marbles) / 3
def shania_marbles : ℕ := 2 * ludo_marbles
def gabriel_marbles : ℕ := (wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles) - 1
def total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles + gabriel_marbles
def marbles_per_friend : ℕ := total_marbles / 5

-- Mathematical equivalent proof problem
theorem equal_share_of_marbles : marbles_per_friend = 39 := by
  sorry

end NUMINAMATH_GPT_equal_share_of_marbles_l1946_194606


namespace NUMINAMATH_GPT_abs_diff_61st_terms_l1946_194689

noncomputable def seq_C (n : ℕ) : ℤ := 20 + 15 * (n - 1)
noncomputable def seq_D (n : ℕ) : ℤ := 20 - 15 * (n - 1)

theorem abs_diff_61st_terms :
  |seq_C 61 - seq_D 61| = 1800 := sorry

end NUMINAMATH_GPT_abs_diff_61st_terms_l1946_194689


namespace NUMINAMATH_GPT_ThreePowerTowerIsLarger_l1946_194669

-- original power tower definitions
def A : ℕ := 3^(3^(3^3))
def B : ℕ := 2^(2^(2^(2^2)))

-- reduced forms given from the conditions
def reducedA : ℕ := 3^(3^27)
def reducedB : ℕ := 2^(2^16)

theorem ThreePowerTowerIsLarger : reducedA > reducedB := by
  sorry

end NUMINAMATH_GPT_ThreePowerTowerIsLarger_l1946_194669


namespace NUMINAMATH_GPT_find_k_intersection_on_line_l1946_194672

theorem find_k_intersection_on_line (k : ℝ) :
  (∃ (x y : ℝ), x - 2 * y - 2 * k = 0 ∧ 2 * x - 3 * y - k = 0 ∧ 3 * x - y = 0) → k = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_k_intersection_on_line_l1946_194672


namespace NUMINAMATH_GPT_set_intersection_l1946_194641

def setM : Set ℝ := {x | x^2 - 1 < 0}
def setN : Set ℝ := {y | ∃ x ∈ setM, y = Real.log (x + 2)}

theorem set_intersection : setM ∩ setN = {y | 0 < y ∧ y < Real.log 3} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_l1946_194641


namespace NUMINAMATH_GPT_eval_exp_l1946_194683

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end NUMINAMATH_GPT_eval_exp_l1946_194683


namespace NUMINAMATH_GPT_solution_set_l1946_194682

noncomputable def f (x : ℝ) : ℝ :=
  x * Real.sin x + Real.cos x + x^2

theorem solution_set (x : ℝ) :
  f (Real.log x) + f (Real.log (1 / x)) < 2 * f 1 ↔ (1 / Real.exp 1 < x ∧ x < Real.exp 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_l1946_194682


namespace NUMINAMATH_GPT_almond_butter_servings_l1946_194612

def convert_mixed_to_fraction (a b : ℤ) (n : ℕ) : ℚ :=
  (a * n + b) / n

def servings (total servings_fraction : ℚ) : ℚ :=
  total / servings_fraction

theorem almond_butter_servings :
  servings (convert_mixed_to_fraction 35 2 3) (convert_mixed_to_fraction 2 1 2) = 14 + 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_almond_butter_servings_l1946_194612


namespace NUMINAMATH_GPT_right_triangle_eqn_roots_indeterminate_l1946_194604

theorem right_triangle_eqn_roots_indeterminate 
  (a b c : ℝ) (h : a^2 + c^2 = b^2) : 
  ¬(∃ Δ, Δ = 4 - 4 * c^2 ∧ (Δ > 0 ∨ Δ = 0 ∨ Δ < 0)) →
  (¬∃ x, a * (x^2 - 1) - 2 * x + b * (x^2 + 1) = 0 ∨
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ a * (x₁^2 - 1) - 2 * x₁ + b * (x₁^2 + 1) = 0 ∧ a * (x₂^2 - 1) - 2 * x₂ + b * (x₂^2 + 1) = 0) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_eqn_roots_indeterminate_l1946_194604


namespace NUMINAMATH_GPT_carrie_pays_199_27_l1946_194684

noncomputable def carrie_payment : ℝ :=
  let shirts := 8 * 12
  let pants := 4 * 25
  let jackets := 4 * 75
  let skirts := 3 * 30
  let shoes := 2 * 50
  let shirts_discount := 0.20 * shirts
  let jackets_discount := 0.20 * jackets
  let skirts_discount := 0.10 * skirts
  let total_cost := shirts + pants + jackets + skirts + shoes
  let discounted_cost := (shirts - shirts_discount) + (pants) + (jackets - jackets_discount) + (skirts - skirts_discount) + shoes
  let mom_payment := 2 / 3 * discounted_cost
  let carrie_payment := discounted_cost - mom_payment
  carrie_payment

theorem carrie_pays_199_27 : carrie_payment = 199.27 :=
by
  sorry

end NUMINAMATH_GPT_carrie_pays_199_27_l1946_194684


namespace NUMINAMATH_GPT_scientific_notation_142000_l1946_194651

theorem scientific_notation_142000 : (142000 : ℝ) = 1.42 * 10^5 := sorry

end NUMINAMATH_GPT_scientific_notation_142000_l1946_194651


namespace NUMINAMATH_GPT_ratio_fifth_terms_l1946_194609

variable (a_n b_n S_n T_n : ℕ → ℚ)

-- Conditions
variable (h : ∀ n, S_n n / T_n n = (9 * n + 2) / (n + 7))

-- Define the 5th term
def a_5 (S_n : ℕ → ℚ) : ℚ := S_n 9 / 9
def b_5 (T_n : ℕ → ℚ) : ℚ := T_n 9 / 9

-- Prove that the ratio of the 5th terms is 83 / 16
theorem ratio_fifth_terms :
  (a_5 S_n) / (b_5 T_n) = 83 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_fifth_terms_l1946_194609


namespace NUMINAMATH_GPT_substitutions_made_in_first_half_l1946_194664

-- Definitions based on given problem conditions
def total_players : ℕ := 24
def starters : ℕ := 11
def non_players : ℕ := 7
def first_half_substitutions (S : ℕ) : ℕ := S
def second_half_substitutions (S : ℕ) : ℕ := 2 * S
def total_players_played (S : ℕ) := starters + first_half_substitutions S + second_half_substitutions S
def remaining_players : ℕ := total_players - non_players

-- Proof problem statement
theorem substitutions_made_in_first_half (S : ℕ) (h : total_players_played S = remaining_players) : S = 2 :=
by
  sorry

end NUMINAMATH_GPT_substitutions_made_in_first_half_l1946_194664


namespace NUMINAMATH_GPT_remainder_1394_mod_2535_l1946_194607

-- Definition of the least number satisfying the given conditions
def L : ℕ := 1394

-- Proof statement: proving the remainder of division
theorem remainder_1394_mod_2535 : (1394 % 2535) = 1394 :=
by sorry

end NUMINAMATH_GPT_remainder_1394_mod_2535_l1946_194607


namespace NUMINAMATH_GPT_ram_initial_deposit_l1946_194630

theorem ram_initial_deposit :
  ∃ P: ℝ, P + 100 = 1100 ∧ 1.20 * 1100 = 1320 ∧ P * 1.32 = 1320 ∧ P = 1000 :=
by
  existsi (1000 : ℝ)
  sorry

end NUMINAMATH_GPT_ram_initial_deposit_l1946_194630


namespace NUMINAMATH_GPT_find_tangent_line_equation_l1946_194675

noncomputable def tangent_line_equation (f : ℝ → ℝ) (perp_line : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  let y₀ := f x₀
  let slope_perp_to_tangent := -2
  let slope_tangent := -1 / 2
  slope_perp_to_tangent = -1 / (deriv f x₀) ∧
  x₀ = 1 ∧ y₀ = 1 ∧
  ∃ (a b c : ℝ), a * x₀ + b * y₀ + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3

theorem find_tangent_line_equation :
  tangent_line_equation (fun (x : ℝ) => Real.sqrt x) (fun (x : ℝ) => -2 * x - 4) 1 := by
  sorry

end NUMINAMATH_GPT_find_tangent_line_equation_l1946_194675


namespace NUMINAMATH_GPT_intersection_with_x_axis_intersection_with_y_axis_l1946_194649

theorem intersection_with_x_axis (x y : ℝ) : y = -2 * x + 4 ∧ y = 0 ↔ x = 2 ∧ y = 0 := by
  sorry

theorem intersection_with_y_axis (x y : ℝ) : y = -2 * x + 4 ∧ x = 0 ↔ x = 0 ∧ y = 4 := by
  sorry

end NUMINAMATH_GPT_intersection_with_x_axis_intersection_with_y_axis_l1946_194649


namespace NUMINAMATH_GPT_coin_landing_heads_prob_l1946_194629

theorem coin_landing_heads_prob (p : ℝ) (h : p^2 * (1 - p)^3 = 0.03125) : p = 0.5 :=
by
sorry

end NUMINAMATH_GPT_coin_landing_heads_prob_l1946_194629


namespace NUMINAMATH_GPT_route_y_saves_time_l1946_194647

theorem route_y_saves_time (distance_X speed_X : ℕ)
                           (distance_Y_WOCZ distance_Y_CZ speed_Y speed_Y_CZ : ℕ)
                           (time_saved_in_minutes : ℚ) :
  distance_X = 8 → 
  speed_X = 40 → 
  distance_Y_WOCZ = 6 → 
  distance_Y_CZ = 1 → 
  speed_Y = 50 → 
  speed_Y_CZ = 25 → 
  time_saved_in_minutes = 2.4 →
  (distance_X / speed_X : ℚ) * 60 - 
  ((distance_Y_WOCZ / speed_Y + distance_Y_CZ / speed_Y_CZ) * 60) = time_saved_in_minutes :=
by
  intros
  sorry

end NUMINAMATH_GPT_route_y_saves_time_l1946_194647


namespace NUMINAMATH_GPT_least_multiple_25_gt_500_l1946_194633

theorem least_multiple_25_gt_500 : ∃ (k : ℕ), 25 * k > 500 ∧ (∀ m : ℕ, (25 * m > 500 → 25 * k ≤ 25 * m)) :=
by
  use 21
  sorry

end NUMINAMATH_GPT_least_multiple_25_gt_500_l1946_194633


namespace NUMINAMATH_GPT_ratio_friday_to_monday_l1946_194639

-- Definitions from conditions
def rabbits : ℕ := 16
def monday_toys : ℕ := 6
def wednesday_toys : ℕ := 2 * monday_toys
def saturday_toys : ℕ := wednesday_toys / 2
def total_toys : ℕ := 3 * rabbits

-- Definition to represent the number of toys bought on Friday
def friday_toys : ℕ := total_toys - (monday_toys + wednesday_toys + saturday_toys)

-- Theorem to prove the ratio is 4:1
theorem ratio_friday_to_monday : friday_toys / monday_toys = 4 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_friday_to_monday_l1946_194639


namespace NUMINAMATH_GPT_solve_system_of_equations_simplify_expression_l1946_194679

-- Statement for system of equations
theorem solve_system_of_equations (s t : ℚ) 
  (h1 : 2 * s + 3 * t = 2) 
  (h2 : 2 * s - 6 * t = -1) :
  s = 1 / 2 ∧ t = 1 / 3 :=
sorry

-- Statement for simplifying the expression
theorem simplify_expression (x y : ℚ) :
  ((x - y)^2 + (x + y) * (x - y)) / (2 * x) = x - y :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_simplify_expression_l1946_194679


namespace NUMINAMATH_GPT_find_f_5pi_div_3_l1946_194621

variable (f : ℝ → ℝ)

-- Define the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem find_f_5pi_div_3
  (h_odd : is_odd_function f)
  (h_periodic : is_periodic_function f π)
  (h_def : ∀ x, 0 ≤ x → x ≤ π/2 → f x = Real.sin x) :
  f (5 * π / 3) = - (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_GPT_find_f_5pi_div_3_l1946_194621


namespace NUMINAMATH_GPT_exists_infinitely_many_m_l1946_194692

theorem exists_infinitely_many_m (k : ℕ) (hk : 0 < k) : 
  ∃ᶠ m in at_top, 3 ^ k ∣ m ^ 3 + 10 :=
sorry

end NUMINAMATH_GPT_exists_infinitely_many_m_l1946_194692


namespace NUMINAMATH_GPT_sum_of_b_values_l1946_194659

theorem sum_of_b_values :
  let discriminant (b : ℝ) := (b + 6) ^ 2 - 4 * 3 * 12
  ∃ b1 b2 : ℝ, discriminant b1 = 0 ∧ discriminant b2 = 0 ∧ b1 + b2 = -12 :=
by sorry

end NUMINAMATH_GPT_sum_of_b_values_l1946_194659


namespace NUMINAMATH_GPT_expected_value_of_fair_6_sided_die_l1946_194686

noncomputable def fair_die_expected_value : ℝ :=
  (1/6) * 1 + (1/6) * 2 + (1/6) * 3 + (1/6) * 4 + (1/6) * 5 + (1/6) * 6

theorem expected_value_of_fair_6_sided_die : fair_die_expected_value = 3.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_fair_6_sided_die_l1946_194686


namespace NUMINAMATH_GPT_emily_lives_total_l1946_194638

variable (x : ℤ)

def total_lives_after_stages (x : ℤ) : ℤ :=
  let lives_after_stage1 := x + 25
  let lives_after_stage2 := lives_after_stage1 + 24
  let lives_after_stage3 := lives_after_stage2 + 15
  lives_after_stage3

theorem emily_lives_total : total_lives_after_stages x = x + 64 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_emily_lives_total_l1946_194638


namespace NUMINAMATH_GPT_simplify_expression_l1946_194627

variable {x y z : ℝ} 
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)

theorem simplify_expression :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (x * y * z)⁻¹ * (x + y + z)⁻¹ :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1946_194627


namespace NUMINAMATH_GPT_distinct_even_numbers_between_100_and_999_l1946_194645

def count_distinct_even_numbers_between_100_and_999 : ℕ :=
  let possible_units_digits := 5 -- {0, 2, 4, 6, 8}
  let possible_hundreds_digits := 8 -- {1, 2, ..., 9} excluding the chosen units digit
  let possible_tens_digits := 8 -- {0, 1, 2, ..., 9} excluding the chosen units and hundreds digits
  possible_units_digits * possible_hundreds_digits * possible_tens_digits

theorem distinct_even_numbers_between_100_and_999 : count_distinct_even_numbers_between_100_and_999 = 320 :=
  by sorry

end NUMINAMATH_GPT_distinct_even_numbers_between_100_and_999_l1946_194645


namespace NUMINAMATH_GPT_line_does_not_pass_through_second_quadrant_l1946_194656

theorem line_does_not_pass_through_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ x - y - a^2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_line_does_not_pass_through_second_quadrant_l1946_194656


namespace NUMINAMATH_GPT_find_amount_l1946_194676

-- Given conditions
variables (x A : ℝ)

theorem find_amount :
  (0.65 * x = 0.20 * A) → (x = 190) → (A = 617.5) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_amount_l1946_194676


namespace NUMINAMATH_GPT_gnollish_valid_sentence_count_is_48_l1946_194660

-- Define the problem parameters
def gnollish_words : List String := ["word1", "word2", "splargh", "glumph", "kreeg"]

def valid_sentence_count : Nat :=
  let total_sentences := 4 * 4 * 4
  let invalid_sentences :=
    4 +         -- (word) splargh glumph
    4 +         -- splargh glumph (word)
    4 +         -- (word) splargh kreeg
    4           -- splargh kreeg (word)
  total_sentences - invalid_sentences

-- Prove that the number of valid 3-word sentences is 48
theorem gnollish_valid_sentence_count_is_48 : valid_sentence_count = 48 := by
  sorry

end NUMINAMATH_GPT_gnollish_valid_sentence_count_is_48_l1946_194660


namespace NUMINAMATH_GPT_find_y_l1946_194685

theorem find_y (y : ℝ) (h : (8 + 15 + 22 + 5 + y) / 5 = 12) : y = 10 :=
by
  -- the proof is skipped
  sorry

end NUMINAMATH_GPT_find_y_l1946_194685


namespace NUMINAMATH_GPT_manny_remaining_money_l1946_194665

def cost_chair (cost_total_chairs : ℕ) (number_of_chairs : ℕ) : ℕ :=
  cost_total_chairs / number_of_chairs

def cost_table (cost_chair : ℕ) (chairs_for_table : ℕ) : ℕ :=
  cost_chair * chairs_for_table

def total_cost (cost_table : ℕ) (cost_chairs : ℕ) : ℕ :=
  cost_table + cost_chairs

def remaining_money (initial_amount : ℕ) (spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

theorem manny_remaining_money : remaining_money 100 (total_cost (cost_table (cost_chair 55 5) 3) ((cost_chair 55 5) * 2)) = 45 :=
by
  sorry

end NUMINAMATH_GPT_manny_remaining_money_l1946_194665


namespace NUMINAMATH_GPT_find_amount_with_r_l1946_194632

variable (p q r : ℝ)

-- Condition 1: p, q, and r have Rs. 6000 among themselves.
def total_amount : Prop := p + q + r = 6000

-- Condition 2: r has two-thirds of the total amount with p and q.
def r_amount : Prop := r = (2 / 3) * (p + q)

theorem find_amount_with_r (h1 : total_amount p q r) (h2 : r_amount p q r) : r = 2400 := by
  sorry

end NUMINAMATH_GPT_find_amount_with_r_l1946_194632


namespace NUMINAMATH_GPT_product_of_numbers_l1946_194624

variable (x y : ℕ)

theorem product_of_numbers : x + y = 120 ∧ x - y = 6 → x * y = 3591 := by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1946_194624


namespace NUMINAMATH_GPT_smallest_n_exists_l1946_194674

theorem smallest_n_exists (G : Type) [Fintype G] [DecidableEq G] (connected : G → G → Prop)
  (distinct_naturals : G → ℕ) :
  (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 = 1) ∧
  (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 > 1) →
  (∀ n : ℕ, 
    (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) n = 1) ∧
    (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) n > 1) →
    15 ≤ n) :=
sorry

end NUMINAMATH_GPT_smallest_n_exists_l1946_194674


namespace NUMINAMATH_GPT_probability_getting_wet_l1946_194635

theorem probability_getting_wet 
  (P_R : ℝ := 1/2)
  (P_notT : ℝ := 1/2)
  (h1 : 0 ≤ P_R ∧ P_R ≤ 1)
  (h2 : 0 ≤ P_notT ∧ P_notT ≤ 1) 
  : P_R * P_notT = 1/4 := 
by
  -- Proof that the probability of getting wet equals 1/4
  sorry

end NUMINAMATH_GPT_probability_getting_wet_l1946_194635


namespace NUMINAMATH_GPT_elastic_collision_ball_speed_l1946_194698

open Real

noncomputable def final_ball_speed (v_car v_ball : ℝ) : ℝ :=
  let relative_speed := v_ball + v_car
  relative_speed + v_car

theorem elastic_collision_ball_speed :
  let v_car := 5
  let v_ball := 6
  final_ball_speed v_car v_ball = 16 := 
by
  sorry

end NUMINAMATH_GPT_elastic_collision_ball_speed_l1946_194698


namespace NUMINAMATH_GPT_find_n_l1946_194680

theorem find_n (n : ℕ) : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ n → n = 29 := by
  intros h
  sorry

end NUMINAMATH_GPT_find_n_l1946_194680


namespace NUMINAMATH_GPT_harry_less_than_half_selena_l1946_194611

-- Definitions of the conditions
def selena_book_pages := 400
def harry_book_pages := 180
def half (n : ℕ) := n / 2

-- The theorem to prove that Harry's book is 20 pages less than half of Selena's book.
theorem harry_less_than_half_selena :
  harry_book_pages = half selena_book_pages - 20 := 
by
  sorry

end NUMINAMATH_GPT_harry_less_than_half_selena_l1946_194611


namespace NUMINAMATH_GPT_jim_caught_fish_l1946_194601

variable (ben judy billy susie jim caught_back total_filets : ℕ)

def caught_fish : ℕ :=
  ben + judy + billy + susie + jim - caught_back

theorem jim_caught_fish (h_ben : ben = 4)
                        (h_judy : judy = 1)
                        (h_billy : billy = 3)
                        (h_susie : susie = 5)
                        (h_caught_back : caught_back = 3)
                        (h_total_filets : total_filets = 24)
                        (h_filets_per_fish : ∀ f : ℕ, total_filets = f * 2 → caught_fish ben judy billy susie jim caught_back = f) :
  jim = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jim_caught_fish_l1946_194601


namespace NUMINAMATH_GPT_problem_statement_l1946_194673

theorem problem_statement :
  (-2010)^2011 = - (2010 ^ 2011) :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_problem_statement_l1946_194673


namespace NUMINAMATH_GPT_cube_splitting_height_l1946_194662

/-- If we split a cube with an edge of 1 meter into small cubes with an edge of 1 millimeter,
what will be the height of a column formed by stacking all the small cubes one on top of another? -/
theorem cube_splitting_height :
  let edge_meter := 1
  let edge_mm := 1000
  let num_cubes := (edge_meter * edge_mm) ^ 3
  let height_mm := num_cubes * edge_mm
  let height_km := height_mm / (1000 * 1000 * 1000)
  height_km = 1000 :=
by
  sorry

end NUMINAMATH_GPT_cube_splitting_height_l1946_194662


namespace NUMINAMATH_GPT_prod_eq_of_eqs_l1946_194623

variable (a : ℝ) (m n p q : ℕ)
variable (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1)
variable (h4 : a^m + a^n = a^p + a^q) (h5 : a^{3*m} + a^{3*n} = a^{3*p} + a^{3*q})

theorem prod_eq_of_eqs : m * n = p * q := by
  sorry

end NUMINAMATH_GPT_prod_eq_of_eqs_l1946_194623


namespace NUMINAMATH_GPT_M_inter_N_eq_l1946_194658

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_inter_N_eq : (M ∩ N) = Set.Ico 0 (1/2) := 
by
  sorry

end NUMINAMATH_GPT_M_inter_N_eq_l1946_194658


namespace NUMINAMATH_GPT_solve_for_x_l1946_194642

theorem solve_for_x (x : ℝ) (h : (x / 4) / 2 = 4 / (x / 2)) : x = 8 ∨ x = -8 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1946_194642


namespace NUMINAMATH_GPT_harkamal_grapes_purchase_l1946_194687

-- Define the conditions as parameters and constants
def cost_per_kg_grapes := 70
def kg_mangoes := 9
def cost_per_kg_mangoes := 45
def total_payment := 965

-- The theorem stating Harkamal purchased 8 kg of grapes
theorem harkamal_grapes_purchase : 
  ∃ G : ℕ, (cost_per_kg_grapes * G + cost_per_kg_mangoes * kg_mangoes = total_payment) ∧ G = 8 :=
by
  use 8
  unfold cost_per_kg_grapes cost_per_kg_mangoes kg_mangoes total_payment
  show 70 * 8 + 45 * 9 = 965 ∧ 8 = 8
  sorry

end NUMINAMATH_GPT_harkamal_grapes_purchase_l1946_194687


namespace NUMINAMATH_GPT_find_uncommon_cards_l1946_194636

def numRare : ℕ := 19
def numCommon : ℕ := 30
def costRare : ℝ := 1
def costUncommon : ℝ := 0.50
def costCommon : ℝ := 0.25
def totalCostDeck : ℝ := 32

theorem find_uncommon_cards (U : ℕ) (h : U * costUncommon + numRare * costRare + numCommon * costCommon = totalCostDeck) : U = 11 := by
  sorry

end NUMINAMATH_GPT_find_uncommon_cards_l1946_194636


namespace NUMINAMATH_GPT_man_climbs_out_of_well_in_65_days_l1946_194643

theorem man_climbs_out_of_well_in_65_days (depth climb slip net_days last_climb : ℕ) 
  (h_depth : depth = 70)
  (h_climb : climb = 6)
  (h_slip : slip = 5)
  (h_net_days : net_days = 64)
  (h_last_climb : last_climb = 1) :
  ∃ days : ℕ, days = net_days + last_climb ∧ days = 65 := by
  sorry

end NUMINAMATH_GPT_man_climbs_out_of_well_in_65_days_l1946_194643


namespace NUMINAMATH_GPT_asher_speed_l1946_194690

theorem asher_speed :
  (5 * 60 ≠ 0) → (6600 / (5 * 60) = 22) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_asher_speed_l1946_194690


namespace NUMINAMATH_GPT_trigonometric_identity_l1946_194693

open Real

theorem trigonometric_identity :
  (sin (20 * π / 180) * sin (80 * π / 180) - cos (160 * π / 180) * sin (10 * π / 180) = 1 / 2) :=
by
  -- Trigonometric calculations
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1946_194693
