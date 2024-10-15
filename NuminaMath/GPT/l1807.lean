import Mathlib

namespace NUMINAMATH_GPT_find_m_value_l1807_180719

def power_function_increasing (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2*m - 1 > 0)

theorem find_m_value (m : ℝ) (h : power_function_increasing m) : m = -1 :=
  sorry

end NUMINAMATH_GPT_find_m_value_l1807_180719


namespace NUMINAMATH_GPT_solution_of_inequality_l1807_180728

theorem solution_of_inequality (x : ℝ) : -2 * x - 1 < -1 → x > 0 :=
by
  sorry

end NUMINAMATH_GPT_solution_of_inequality_l1807_180728


namespace NUMINAMATH_GPT_auditorium_seats_l1807_180741

variable (S : ℕ)

theorem auditorium_seats (h1 : 2 * S / 5 + S / 10 + 250 = S) : S = 500 :=
by
  sorry

end NUMINAMATH_GPT_auditorium_seats_l1807_180741


namespace NUMINAMATH_GPT_remaining_battery_life_l1807_180757

theorem remaining_battery_life :
  let capacity1 := 60
  let capacity2 := 80
  let capacity3 := 120
  let used1 := capacity1 * (3 / 4 : ℚ)
  let used2 := capacity2 * (1 / 2 : ℚ)
  let used3 := capacity3 * (2 / 3 : ℚ)
  let remaining1 := capacity1 - used1 - 2
  let remaining2 := capacity2 - used2 - 2
  let remaining3 := capacity3 - used3 - 2
  remaining1 + remaining2 + remaining3 = 89 := 
by
  sorry

end NUMINAMATH_GPT_remaining_battery_life_l1807_180757


namespace NUMINAMATH_GPT_arithmetic_sequence_term_l1807_180731

theorem arithmetic_sequence_term (a d n : ℕ) (h₀ : a = 1) (h₁ : d = 3) (h₂ : a + (n - 1) * d = 6019) :
  n = 2007 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_l1807_180731


namespace NUMINAMATH_GPT_triangle_angles_arithmetic_progression_l1807_180744

theorem triangle_angles_arithmetic_progression (α β γ : ℝ) (a c : ℝ) :
  (α < β) ∧ (β < γ) ∧ (α + β + γ = 180) ∧
  (∃ x : ℝ, β = α + x ∧ γ = β + x) ∧
  (a = c / 2) → 
  (α = 30) ∧ (β = 60) ∧ (γ = 90) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_triangle_angles_arithmetic_progression_l1807_180744


namespace NUMINAMATH_GPT_arithmetic_sequence_probability_l1807_180706

theorem arithmetic_sequence_probability (n p : ℕ) (h_cond : n + p = 2008) (h_neg : n = 161) (h_pos : p = 2008 - 161) :
  ∃ a b : ℕ, (a = 1715261 ∧ b = 2016024 ∧ a + b = 3731285) ∧ (a / b = 1715261 / 2016024) := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_probability_l1807_180706


namespace NUMINAMATH_GPT_total_distance_traveled_l1807_180707

theorem total_distance_traveled
  (r1 r2 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25):
  let arc_outer := 1/4 * 2 * Real.pi * r2
  let radial := r2 - r1
  let circ_inner := 2 * Real.pi * r1
  let return_radial := radial
  let total_distance := arc_outer + radial + circ_inner + return_radial
  total_distance = 42.5 * Real.pi + 20 := 
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l1807_180707


namespace NUMINAMATH_GPT_surface_area_to_lateral_surface_ratio_cone_l1807_180771

noncomputable def cone_surface_lateral_area_ratio : Prop :=
  let radius : ℝ := 1
  let theta : ℝ := (2 * Real.pi) / 3
  let lateral_surface_area := Real.pi * radius^2 * (theta / (2 * Real.pi))
  let base_radius := (2 * Real.pi * radius * (theta / (2 * Real.pi))) / (2 * Real.pi)
  let base_area := Real.pi * base_radius^2
  let surface_area := lateral_surface_area + base_area
  (surface_area / lateral_surface_area) = (4 / 3)

theorem surface_area_to_lateral_surface_ratio_cone :
  cone_surface_lateral_area_ratio :=
  by
  sorry

end NUMINAMATH_GPT_surface_area_to_lateral_surface_ratio_cone_l1807_180771


namespace NUMINAMATH_GPT_horner_evaluation_at_2_l1807_180732

def f (x : ℤ) : ℤ := 3 * x^5 - 2 * x^4 + 2 * x^3 - 4 * x^2 - 7

theorem horner_evaluation_at_2 : f 2 = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_horner_evaluation_at_2_l1807_180732


namespace NUMINAMATH_GPT_principal_amount_l1807_180703

theorem principal_amount (P : ℝ) (r t : ℝ) (d : ℝ) 
  (h1 : r = 7)
  (h2 : t = 2)
  (h3 : d = 49)
  (h4 : P * ((1 + r / 100) ^ t - 1) - P * (r * t / 100) = d) :
  P = 10000 :=
by sorry

end NUMINAMATH_GPT_principal_amount_l1807_180703


namespace NUMINAMATH_GPT_arithmetic_mean_34_58_l1807_180745

theorem arithmetic_mean_34_58 :
  (3 / 4 : ℚ) + (5 / 8 : ℚ) / 2 = 11 / 16 := sorry

end NUMINAMATH_GPT_arithmetic_mean_34_58_l1807_180745


namespace NUMINAMATH_GPT_tens_digit_of_13_pow_2023_l1807_180717

theorem tens_digit_of_13_pow_2023 :
  ∀ (n : ℕ), (13 ^ (2023 % 20) ≡ 13 ^ n [MOD 100]) ∧ (13 ^ n ≡ 97 [MOD 100]) → (13 ^ 2023) % 100 / 10 % 10 = 9 :=
by
sorry

end NUMINAMATH_GPT_tens_digit_of_13_pow_2023_l1807_180717


namespace NUMINAMATH_GPT_integer_pairs_l1807_180712

def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem integer_pairs (a b : ℤ) :
  (is_perfect_square (a^2 + 4 * b) ∧ is_perfect_square (b^2 + 4 * a)) ↔ 
  (a = 0 ∧ b = 0) ∨ (a = -4 ∧ b = -4) ∨ (a = 4 ∧ b = -4) ∨
  (∃ (k : ℕ), a = k^2 ∧ b = 0) ∨ (∃ (k : ℕ), a = 0 ∧ b = k^2) ∨
  (a = -6 ∧ b = -5) ∨ (a = -5 ∧ b = -6) ∨
  (∃ (t : ℕ), a = t ∧ b = 1 - t) ∨ (∃ (t : ℕ), a = 1 - t ∧ b = t) :=
sorry

end NUMINAMATH_GPT_integer_pairs_l1807_180712


namespace NUMINAMATH_GPT_length_AP_eq_sqrt2_l1807_180735

/-- In square ABCD with side length 2, a circle ω with center at (1, 0)
    and radius 1 is inscribed. The circle intersects CD at point M,
    and line AM intersects ω at a point P different from M.
    Prove that the length of AP is √2. -/
theorem length_AP_eq_sqrt2 :
  let A := (0, 2)
  let M := (2, 0)
  let P : ℝ × ℝ := (1, 1)
  dist A P = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_length_AP_eq_sqrt2_l1807_180735


namespace NUMINAMATH_GPT_circle_equation_l1807_180778

theorem circle_equation :
  ∃ (a : ℝ), (y - a)^2 + x^2 = 1 ∧ (1 - 0)^2 + (2 - a)^2 = 1 ∧
  ∀ a, (1 - 0)^2 + (2 - a)^2 = 1 → a = 2 →
  x^2 + (y - 2)^2 = 1 := by sorry

end NUMINAMATH_GPT_circle_equation_l1807_180778


namespace NUMINAMATH_GPT_wire_cut_square_octagon_area_l1807_180769

theorem wire_cut_square_octagon_area (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (equal_area : (a / 4)^2 = (2 * (b / 8)^2 * (1 + Real.sqrt 2))) : 
  a / b = Real.sqrt ((1 + Real.sqrt 2) / 2) := 
  sorry

end NUMINAMATH_GPT_wire_cut_square_octagon_area_l1807_180769


namespace NUMINAMATH_GPT_totalPieces_l1807_180782

   -- Definitions given by the conditions
   def packagesGum := 21
   def packagesCandy := 45
   def packagesMints := 30
   def piecesPerGumPackage := 9
   def piecesPerCandyPackage := 12
   def piecesPerMintPackage := 8

   -- Define the total pieces of gum, candy, and mints
   def totalPiecesGum := packagesGum * piecesPerGumPackage
   def totalPiecesCandy := packagesCandy * piecesPerCandyPackage
   def totalPiecesMints := packagesMints * piecesPerMintPackage

   -- The mathematical statement to prove
   theorem totalPieces :
     totalPiecesGum + totalPiecesCandy + totalPiecesMints = 969 :=
   by
     -- Proof is skipped
     sorry
   
end NUMINAMATH_GPT_totalPieces_l1807_180782


namespace NUMINAMATH_GPT_find_k_l1807_180753

def distances (S x y k : ℝ) := (S - x * 0.75) * x / (x + y) + 0.75 * x = S * x / (x + y) - 18 ∧
                              S * x / (x + y) - (S - y / 3) * x / (x + y) = k

theorem find_k (S x y k : ℝ) (h₁ : x * y / (x + y) = 24) (h₂ : k = 24 / 3)
  : k = 8 :=
by 
  -- We need to fill in the proof steps here
  sorry

end NUMINAMATH_GPT_find_k_l1807_180753


namespace NUMINAMATH_GPT_unique_rectangle_dimensions_l1807_180777

theorem unique_rectangle_dimensions (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < a ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = a * b / 4 :=
sorry

end NUMINAMATH_GPT_unique_rectangle_dimensions_l1807_180777


namespace NUMINAMATH_GPT_jessica_quarters_l1807_180750

theorem jessica_quarters (initial_quarters borrowed_quarters remaining_quarters : ℕ)
  (h1 : initial_quarters = 8)
  (h2 : borrowed_quarters = 3) :
  remaining_quarters = initial_quarters - borrowed_quarters → remaining_quarters = 5 :=
by
  intro h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_jessica_quarters_l1807_180750


namespace NUMINAMATH_GPT_min_value_fraction_l1807_180773

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) : 
  ∃x : ℝ, (x = (1/a + 2/b)) ∧ (∀y : ℝ, (y = (1/a + 2/b)) → y ≥ 8) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l1807_180773


namespace NUMINAMATH_GPT_exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l1807_180710

theorem exists_half_perimeter_area_rectangle_6x1 :
  ∃ x₁ x₂ : ℝ, (6 * 1 / 2 = (6 + 1) / 2) ∧
                x₁ * x₂ = 3 ∧
                (x₁ + x₂ = 3.5) ∧
                (x₁ = 2 ∨ x₁ = 1.5) ∧
                (x₂ = 2 ∨ x₂ = 1.5)
:= by
  sorry

theorem not_exists_half_perimeter_area_rectangle_2x1 :
  ¬(∃ x : ℝ, x * (1.5 - x) = 1)
:= by
  sorry

end NUMINAMATH_GPT_exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l1807_180710


namespace NUMINAMATH_GPT_joan_gave_apples_l1807_180701

theorem joan_gave_apples (initial_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : initial_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  -- Show that given_apples is obtained by subtracting remaining_apples from initial_apples
  sorry

end NUMINAMATH_GPT_joan_gave_apples_l1807_180701


namespace NUMINAMATH_GPT_gain_percent_l1807_180776

def cycle_gain_percent (cp sp : ℕ) : ℚ :=
  (sp - cp) / cp * 100

theorem gain_percent {cp sp : ℕ} (h1 : cp = 1500) (h2 : sp = 1620) : cycle_gain_percent cp sp = 8 := by
  sorry

end NUMINAMATH_GPT_gain_percent_l1807_180776


namespace NUMINAMATH_GPT_distance_traveled_is_6000_l1807_180766

-- Define the conditions and the question in Lean 4
def footprints_per_meter_Pogo := 4
def footprints_per_meter_Grimzi := 3 / 6
def combined_total_footprints := 27000

theorem distance_traveled_is_6000 (D : ℕ) :
  footprints_per_meter_Pogo * D + footprints_per_meter_Grimzi * D = combined_total_footprints →
  D = 6000 :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_is_6000_l1807_180766


namespace NUMINAMATH_GPT_possible_to_select_three_numbers_l1807_180792

theorem possible_to_select_three_numbers (n : ℕ) (a : ℕ → ℕ) (h : ∀ i j, i < j → a i < a j) (h_bound : ∀ i, a i < 2 * n) :
  ∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ a i + a j = a k := sorry

end NUMINAMATH_GPT_possible_to_select_three_numbers_l1807_180792


namespace NUMINAMATH_GPT_angle_relationship_l1807_180786

variables {VU VW : ℝ} {x y z : ℝ} (h1 : VU = VW) 
          (angle_UXZ : ℝ) (angle_VYZ : ℝ) (angle_VZX : ℝ)
          (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z)

theorem angle_relationship (h1 : VU = VW) (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z) : 
    x = (y - z) / 2 := 
by 
    sorry

end NUMINAMATH_GPT_angle_relationship_l1807_180786


namespace NUMINAMATH_GPT_train_a_speed_54_l1807_180700

noncomputable def speed_of_train_A (length_A length_B : ℕ) (speed_B : ℕ) (time_to_cross : ℕ) : ℕ :=
  let total_distance := length_A + length_B
  let relative_speed := total_distance / time_to_cross
  let relative_speed_km_per_hr := relative_speed * 36 / 10
  let speed_A := relative_speed_km_per_hr - speed_B
  speed_A

theorem train_a_speed_54 
  (length_A length_B : ℕ)
  (speed_B : ℕ)
  (time_to_cross : ℕ)
  (h_length_A : length_A = 150)
  (h_length_B : length_B = 150)
  (h_speed_B : speed_B = 36)
  (h_time_to_cross : time_to_cross = 12) :
  speed_of_train_A length_A length_B speed_B time_to_cross = 54 := by
  sorry

end NUMINAMATH_GPT_train_a_speed_54_l1807_180700


namespace NUMINAMATH_GPT_expected_value_twelve_sided_die_l1807_180798

/-- A twelve-sided die has its faces numbered from 1 to 12.
    Prove that the expected value of a roll of this die is 6.5. -/
theorem expected_value_twelve_sided_die : 
  (1 / 12 : ℚ) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) = 6.5 := 
by
  sorry

end NUMINAMATH_GPT_expected_value_twelve_sided_die_l1807_180798


namespace NUMINAMATH_GPT_PolygonNumberSides_l1807_180725

theorem PolygonNumberSides (n : ℕ) (h : n - (1 / 2 : ℝ) * (n * (n - 3)) / 2 = 0) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_PolygonNumberSides_l1807_180725


namespace NUMINAMATH_GPT_probability_two_boys_and_three_girls_l1807_180708

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_boys_and_three_girls :
  binomial_probability 5 2 0.5 = 0.3125 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_boys_and_three_girls_l1807_180708


namespace NUMINAMATH_GPT_origin_inside_ellipse_iff_abs_k_range_l1807_180797

theorem origin_inside_ellipse_iff_abs_k_range (k : ℝ) :
  (k^2 * 0^2 + 0^2 - 4 * k * 0 + 2 * k * 0 + k^2 - 1 < 0) ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end NUMINAMATH_GPT_origin_inside_ellipse_iff_abs_k_range_l1807_180797


namespace NUMINAMATH_GPT_value_of_x_minus_2y_l1807_180772

theorem value_of_x_minus_2y (x y : ℝ) (h : 0.5 * x = y + 20) : x - 2 * y = 40 :=
sorry

end NUMINAMATH_GPT_value_of_x_minus_2y_l1807_180772


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1807_180723

open Set

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x^2 - x ≤ 0}
  let B := ({0, 1, 2} : Set ℝ)
  A ∩ B = ({0, 1} : Set ℝ) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1807_180723


namespace NUMINAMATH_GPT_max_value_f_max_value_f_at_13_l1807_180749

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_f : ∀ x : ℝ, f x ≤ 1 / 3 := by
  sorry

theorem max_value_f_at_13 : ∃ x : ℝ, f x = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_max_value_f_max_value_f_at_13_l1807_180749


namespace NUMINAMATH_GPT_pancakes_needed_l1807_180789

def short_stack_pancakes : ℕ := 3
def big_stack_pancakes : ℕ := 5
def short_stack_customers : ℕ := 9
def big_stack_customers : ℕ := 6

theorem pancakes_needed : (short_stack_customers * short_stack_pancakes + big_stack_customers * big_stack_pancakes) = 57 :=
by
  sorry

end NUMINAMATH_GPT_pancakes_needed_l1807_180789


namespace NUMINAMATH_GPT_harry_ron_difference_l1807_180747

-- Define the amounts each individual paid
def harry_paid : ℕ := 150
def ron_paid : ℕ := 180
def hermione_paid : ℕ := 210

-- Define the total amount
def total_paid : ℕ := harry_paid + ron_paid + hermione_paid

-- Define the amount each should have paid
def equal_share : ℕ := total_paid / 3

-- Define the amount Harry owes to Hermione
def harry_owes : ℕ := equal_share - harry_paid

-- Define the amount Ron owes to Hermione
def ron_owes : ℕ := equal_share - ron_paid

-- Define the difference between what Harry and Ron owe Hermione
def difference : ℕ := harry_owes - ron_owes

-- Prove that the difference is 30
theorem harry_ron_difference : difference = 30 := by
  sorry

end NUMINAMATH_GPT_harry_ron_difference_l1807_180747


namespace NUMINAMATH_GPT_difference_of_squares_is_39_l1807_180785

theorem difference_of_squares_is_39 (L S : ℕ) (h1 : L = 8) (h2 : L - S = 3) : L^2 - S^2 = 39 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_is_39_l1807_180785


namespace NUMINAMATH_GPT_trains_meet_distance_from_delhi_l1807_180795

-- Define the speeds of the trains as constants
def speed_bombay_express : ℕ := 60  -- kmph
def speed_rajdhani_express : ℕ := 80  -- kmph

-- Define the time difference in hours between the departures of the two trains
def time_difference : ℕ := 2  -- hours

-- Define the distance the Bombay Express travels before the Rajdhani Express starts
def distance_head_start : ℕ := speed_bombay_express * time_difference

-- Define the relative speed between the two trains
def relative_speed : ℕ := speed_rajdhani_express - speed_bombay_express

-- Define the time taken for the Rajdhani Express to catch up with the Bombay Express
def time_to_meet : ℕ := distance_head_start / relative_speed

-- The final meeting distance from Delhi for the Rajdhani Express
def meeting_distance : ℕ := speed_rajdhani_express * time_to_meet

-- Theorem stating the solution to the problem
theorem trains_meet_distance_from_delhi : meeting_distance = 480 :=
by sorry  -- proof is omitted

end NUMINAMATH_GPT_trains_meet_distance_from_delhi_l1807_180795


namespace NUMINAMATH_GPT_length_of_each_side_is_25_nails_l1807_180796

-- Definitions based on the conditions
def nails_per_side := 25
def total_nails := 96

-- The theorem stating the equivalent mathematical problem
theorem length_of_each_side_is_25_nails
  (n : ℕ) (h1 : n = nails_per_side * 4 - 4)
  (h2 : total_nails = 96):
  n = nails_per_side :=
by
  sorry

end NUMINAMATH_GPT_length_of_each_side_is_25_nails_l1807_180796


namespace NUMINAMATH_GPT_geometric_sequence_a3_is_15_l1807_180705

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
a1 * q^(n - 1)

theorem geometric_sequence_a3_is_15 (q : ℝ) (a1 : ℝ) (a5 : ℝ) 
  (h1 : a1 = 3) (h2 : a5 = 75) (h_seq : ∀ n, a5 = geometric_sequence a1 q n) :
  geometric_sequence a1 q 3 = 15 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_is_15_l1807_180705


namespace NUMINAMATH_GPT_compute_expression_l1807_180784

/-- Definitions of parts of the expression --/
def expr1 := 6 ^ 2
def expr2 := 4 * 5
def expr3 := 2 ^ 3
def expr4 := 4 ^ 2 / 2

/-- Main statement to prove --/
theorem compute_expression : expr1 + expr2 - expr3 + expr4 = 56 := 
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1807_180784


namespace NUMINAMATH_GPT_stream_current_l1807_180722

noncomputable def solve_stream_current : Prop :=
  ∃ (r w : ℝ), (24 / (r + w) + 6 = 24 / (r - w)) ∧ (24 / (3 * r + w) + 2 = 24 / (3 * r - w)) ∧ (w = 2)

theorem stream_current : solve_stream_current :=
  sorry

end NUMINAMATH_GPT_stream_current_l1807_180722


namespace NUMINAMATH_GPT_value_of_2a_plus_b_l1807_180752

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def is_tangent_perpendicular (a b : ℝ) : Prop :=
  let f' := (fun x => (1 : ℝ) / x - a)
  let slope_perpendicular_line := - (1/3 : ℝ)
  f' 1 * slope_perpendicular_line = -1 

def point_on_function (a b : ℝ) : Prop :=
  f a 1 = b

theorem value_of_2a_plus_b (a b : ℝ) 
  (h_tangent_perpendicular : is_tangent_perpendicular a b)
  (h_point_on_function : point_on_function a b) : 
  2 * a + b = -2 := sorry

end NUMINAMATH_GPT_value_of_2a_plus_b_l1807_180752


namespace NUMINAMATH_GPT_triangle_angle_sum_l1807_180775

theorem triangle_angle_sum (a : ℝ) (x : ℝ) :
  0 < 2 * a + 20 ∧ 0 < 3 * a - 15 ∧ 0 < 175 - 5 * a ∧
  2 * a + 20 + 3 * a - 15 + x = 180 → 
  x = 175 - 5 * a ∧ max (2 * a + 20) (max (3 * a - 15) (175 - 5 * a)) = 88 :=
sorry

end NUMINAMATH_GPT_triangle_angle_sum_l1807_180775


namespace NUMINAMATH_GPT_bread_loaves_l1807_180743

theorem bread_loaves (loaf_cost : ℝ) (pb_cost : ℝ) (total_money : ℝ) (leftover_money : ℝ) : ℝ :=
  let spent_money := total_money - leftover_money
  let remaining_money := spent_money - pb_cost
  remaining_money / loaf_cost

example : bread_loaves 2.25 2 14 5.25 = 3 := by
  sorry

end NUMINAMATH_GPT_bread_loaves_l1807_180743


namespace NUMINAMATH_GPT_quadratic_inequality_range_a_l1807_180739

theorem quadratic_inequality_range_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_range_a_l1807_180739


namespace NUMINAMATH_GPT_number_exceeds_20_percent_by_40_eq_50_l1807_180736

theorem number_exceeds_20_percent_by_40_eq_50 (x : ℝ) (h : x = 0.20 * x + 40) : x = 50 := by
  sorry

end NUMINAMATH_GPT_number_exceeds_20_percent_by_40_eq_50_l1807_180736


namespace NUMINAMATH_GPT_c_share_l1807_180759

theorem c_share (A B C : ℝ) 
  (h1 : A = (1 / 2) * B)
  (h2 : B = (1 / 2) * C)
  (h3 : A + B + C = 392) : 
  C = 224 :=
by
  sorry

end NUMINAMATH_GPT_c_share_l1807_180759


namespace NUMINAMATH_GPT_find_roots_l1807_180755

theorem find_roots : ∀ z : ℂ, (z^2 + 2 * z = 3 - 4 * I) → (z = 1 - I ∨ z = -3 + I) :=
by
  intro z
  intro h
  sorry

end NUMINAMATH_GPT_find_roots_l1807_180755


namespace NUMINAMATH_GPT_positive_integer_solutions_value_of_m_when_sum_is_zero_fixed_solution_integer_values_of_m_l1807_180767

-- Definitions for the conditions
def eq1 (x y : ℝ) := x + 2 * y = 6
def eq2 (x y m : ℝ) := x - 2 * y + m * x + 5 = 0

-- Theorem for part (1)
theorem positive_integer_solutions :
  {x y : ℕ} → eq1 x y → (x = 4 ∧ y = 1) ∨ (x = 2 ∧ y = 2) :=
sorry

-- Theorem for part (2)
theorem value_of_m_when_sum_is_zero (x y : ℝ) (h : x + y = 0) :
  eq1 x y → ∃ m : ℝ, eq2 x y m → m = -13/6 :=
sorry

-- Theorem for part (3)
theorem fixed_solution (m : ℝ) : eq2 0 2.5 m :=
sorry

-- Theorem for part (4)
theorem integer_values_of_m (x : ℤ) :
  (∃ y : ℤ, eq1 x y ∧ ∃ m : ℤ, eq2 x y m) → m = -1 ∨ m = -3 :=
sorry

end NUMINAMATH_GPT_positive_integer_solutions_value_of_m_when_sum_is_zero_fixed_solution_integer_values_of_m_l1807_180767


namespace NUMINAMATH_GPT_line_length_l1807_180791

theorem line_length (L : ℝ) (h : 0.75 * L - 0.4 * L = 28) : L = 80 := 
by
  sorry

end NUMINAMATH_GPT_line_length_l1807_180791


namespace NUMINAMATH_GPT_scientific_notation_of_32000000_l1807_180780

theorem scientific_notation_of_32000000 : 32000000 = 3.2 * 10^7 := 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_32000000_l1807_180780


namespace NUMINAMATH_GPT_percentage_saved_l1807_180762

theorem percentage_saved (amount_saved : ℝ) (amount_spent : ℝ) (h1 : amount_saved = 5) (h2 : amount_spent = 45) : 
  (amount_saved / (amount_spent + amount_saved)) * 100 = 10 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_saved_l1807_180762


namespace NUMINAMATH_GPT_box_dimensions_l1807_180748

theorem box_dimensions (x y z : ℝ) (h1 : x * y * z = 160) 
  (h2 : y * z = 80) (h3 : x * z = 40) (h4 : x * y = 32) : 
  x = 4 ∧ y = 8 ∧ z = 10 :=
by
  -- Placeholder for the actual proof steps
  sorry

end NUMINAMATH_GPT_box_dimensions_l1807_180748


namespace NUMINAMATH_GPT_smallest_solution_l1807_180761

theorem smallest_solution (x : ℝ) (h : x * |x| = 3 * x - 2) : 
  x = 1 ∨ x = 2 ∨ x = (-(3 + Real.sqrt 17)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_l1807_180761


namespace NUMINAMATH_GPT_product_of_last_two_digits_l1807_180754

theorem product_of_last_two_digits (A B : ℕ) (h₁ : A + B = 17) (h₂ : 4 ∣ (10 * A + B)) :
  A * B = 72 := sorry

end NUMINAMATH_GPT_product_of_last_two_digits_l1807_180754


namespace NUMINAMATH_GPT_value_of_expression_l1807_180764

theorem value_of_expression (a b : ℝ) (h : a + b = 4) : a^2 + 2 * a * b + b^2 = 16 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1807_180764


namespace NUMINAMATH_GPT_smallest_b_in_arithmetic_series_l1807_180704

theorem smallest_b_in_arithmetic_series (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_arith_series : a = b - d ∧ c = b + d) (h_product : a * b * c = 125) : b ≥ 5 :=
sorry

end NUMINAMATH_GPT_smallest_b_in_arithmetic_series_l1807_180704


namespace NUMINAMATH_GPT_loan_period_l1807_180724

theorem loan_period (principal : ℝ) (rate_A rate_C : ℝ) (gain : ℝ) (years : ℝ) :
  principal = 3500 ∧ rate_A = 0.1 ∧ rate_C = 0.12 ∧ gain = 210 →
  (rate_C * principal * years - rate_A * principal * years) = gain →
  years = 3 :=
by
  sorry

end NUMINAMATH_GPT_loan_period_l1807_180724


namespace NUMINAMATH_GPT_restaurant_chili_paste_needs_l1807_180726

theorem restaurant_chili_paste_needs:
  let large_can_volume := 25
  let small_can_volume := 15
  let large_cans_required := 45
  let total_volume := large_cans_required * large_can_volume
  let small_cans_needed := total_volume / small_can_volume
  small_cans_needed - large_cans_required = 30 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_chili_paste_needs_l1807_180726


namespace NUMINAMATH_GPT_find_T_b_minus_T_neg_b_l1807_180751

noncomputable def T (r : ℝ) : ℝ := 20 / (1 - r)

theorem find_T_b_minus_T_neg_b (b : ℝ) (h1 : -1 < b ∧ b < 1) (h2 : T b * T (-b) = 3240) (h3 : 1 - b^2 = 100 / 810) :
  T b - T (-b) = 324 * b :=
by
  sorry

end NUMINAMATH_GPT_find_T_b_minus_T_neg_b_l1807_180751


namespace NUMINAMATH_GPT_minimum_sum_of_x_and_y_l1807_180730

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 4 * y = x * y

theorem minimum_sum_of_x_and_y (x y : ℝ) (h : conditions x y) : x + y ≥ 9 := by
  sorry

end NUMINAMATH_GPT_minimum_sum_of_x_and_y_l1807_180730


namespace NUMINAMATH_GPT_smallest_among_5_8_4_l1807_180702

theorem smallest_among_5_8_4 : ∀ (x y z : ℕ), x = 5 → y = 8 → z = 4 → z ≤ x ∧ z ≤ y :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  exact ⟨by norm_num, by norm_num⟩

end NUMINAMATH_GPT_smallest_among_5_8_4_l1807_180702


namespace NUMINAMATH_GPT_popsicle_stick_count_l1807_180713

variable (Sam Sid Steve : ℕ)

def number_of_sticks (Sam Sid Steve : ℕ) : ℕ :=
  Sam + Sid + Steve

theorem popsicle_stick_count 
  (h1 : Sam = 3 * Sid)
  (h2 : Sid = 2 * Steve)
  (h3 : Steve = 12) :
  number_of_sticks Sam Sid Steve = 108 :=
by
  sorry

end NUMINAMATH_GPT_popsicle_stick_count_l1807_180713


namespace NUMINAMATH_GPT_harry_terry_difference_l1807_180711

-- Define Harry's answer
def H : ℤ := 8 - (2 + 5)

-- Define Terry's answer
def T : ℤ := 8 - 2 + 5

-- State the theorem to prove H - T = -10
theorem harry_terry_difference : H - T = -10 := by
  sorry

end NUMINAMATH_GPT_harry_terry_difference_l1807_180711


namespace NUMINAMATH_GPT_marble_count_l1807_180765

theorem marble_count (r g b : ℕ) (h1 : g + b = 6) (h2 : r + b = 8) (h3 : r + g = 4) : r + g + b = 9 :=
sorry

end NUMINAMATH_GPT_marble_count_l1807_180765


namespace NUMINAMATH_GPT_max_participants_l1807_180790

structure MeetingRoom where
  rows : ℕ
  cols : ℕ
  seating : ℕ → ℕ → Bool -- A function indicating if a seat (i, j) is occupied (true) or not (false)
  row_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating i (j+1) → seating i (j+2) → False
  col_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating (i+1) j → seating (i+2) j → False

theorem max_participants {room : MeetingRoom} (h : room.rows = 4 ∧ room.cols = 4) : 
  (∃ n : ℕ, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → n < 12) ∧
            (∀ m, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → m < 12) → m ≤ 11)) :=
  sorry

end NUMINAMATH_GPT_max_participants_l1807_180790


namespace NUMINAMATH_GPT_closest_point_in_plane_l1807_180756

noncomputable def closest_point (x y z : ℚ) : Prop :=
  ∃ (t : ℚ), 
    x = 2 + 2 * t ∧ 
    y = 3 - 3 * t ∧ 
    z = 1 + 4 * t ∧ 
    2 * (2 + 2 * t) - 3 * (3 - 3 * t) + 4 * (1 + 4 * t) = 40

theorem closest_point_in_plane :
  closest_point (92 / 29) (16 / 29) (145 / 29) :=
by
  sorry

end NUMINAMATH_GPT_closest_point_in_plane_l1807_180756


namespace NUMINAMATH_GPT_nuts_in_mason_car_l1807_180779

-- Define the constants for the rates of stockpiling
def busy_squirrel_rate := 30 -- nuts per day
def sleepy_squirrel_rate := 20 -- nuts per day
def days := 40 -- number of days
def num_busy_squirrels := 2 -- number of busy squirrels
def num_sleepy_squirrels := 1 -- number of sleepy squirrels

-- Define the total number of nuts
def total_nuts_in_mason_car : ℕ :=
  (num_busy_squirrels * busy_squirrel_rate * days) +
  (num_sleepy_squirrels * sleepy_squirrel_rate * days)

theorem nuts_in_mason_car :
  total_nuts_in_mason_car = 3200 :=
sorry

end NUMINAMATH_GPT_nuts_in_mason_car_l1807_180779


namespace NUMINAMATH_GPT_possible_values_for_a_l1807_180787

def A : Set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + 4 = 0}

theorem possible_values_for_a (a : ℝ) : (B a).Nonempty ∧ B a ⊆ A ↔ a = 4 :=
sorry

end NUMINAMATH_GPT_possible_values_for_a_l1807_180787


namespace NUMINAMATH_GPT_ascorbic_acid_oxygen_mass_percentage_l1807_180770

noncomputable def mass_percentage_oxygen_in_ascorbic_acid : Float := 54.49

theorem ascorbic_acid_oxygen_mass_percentage :
  let C_mass := 12.01
  let H_mass := 1.01
  let O_mass := 16.00
  let ascorbic_acid_formula := (6, 8, 6) -- (number of C, number of H, number of O)
  let total_mass := 6 * C_mass + 8 * H_mass + 6 * O_mass
  let O_mass_total := 6 * O_mass
  mass_percentage_oxygen_in_ascorbic_acid = (O_mass_total / total_mass) * 100 := by
  sorry

end NUMINAMATH_GPT_ascorbic_acid_oxygen_mass_percentage_l1807_180770


namespace NUMINAMATH_GPT_sum_of_six_smallest_multiples_of_12_l1807_180742

-- Define the six smallest distinct positive integer multiples of 12
def multiples_of_12 : List ℕ := [12, 24, 36, 48, 60, 72]

-- Define their sum
def sum_of_multiples : ℕ := multiples_of_12.sum

-- The proof statement
theorem sum_of_six_smallest_multiples_of_12 : sum_of_multiples = 252 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_six_smallest_multiples_of_12_l1807_180742


namespace NUMINAMATH_GPT_initial_number_of_men_l1807_180709

theorem initial_number_of_men (M A : ℕ) : 
  (∀ (M A : ℕ), ((M * A) - 40 + 61) / M = (A + 3)) ∧ (30.5 = 30.5) → 
  M = 7 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l1807_180709


namespace NUMINAMATH_GPT_margarets_mean_score_l1807_180763

noncomputable def mean (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

open List

theorem margarets_mean_score :
  let scores := [86, 88, 91, 93, 95, 97, 99, 100]
  let cyprians_mean := 92
  let num_scores := 8
  let cyprians_scores := 4
  let margarets_scores := num_scores - cyprians_scores
  (scores.sum - cyprians_scores * cyprians_mean) / margarets_scores = 95.25 :=
by
  sorry

end NUMINAMATH_GPT_margarets_mean_score_l1807_180763


namespace NUMINAMATH_GPT_cary_net_calorie_deficit_is_250_l1807_180716

-- Define the conditions
def miles_walked : ℕ := 3
def candy_bar_calories : ℕ := 200
def calories_per_mile : ℕ := 150

-- Define the function to calculate total calories burned
def total_calories_burned (miles : ℕ) (calories_per_mile : ℕ) : ℕ :=
  miles * calories_per_mile

-- Define the function to calculate net calorie deficit
def net_calorie_deficit (total_calories : ℕ) (candy_calories : ℕ) : ℕ :=
  total_calories - candy_calories

-- The statement to be proven
theorem cary_net_calorie_deficit_is_250 :
  net_calorie_deficit (total_calories_burned miles_walked calories_per_mile) candy_bar_calories = 250 :=
  by sorry

end NUMINAMATH_GPT_cary_net_calorie_deficit_is_250_l1807_180716


namespace NUMINAMATH_GPT_rectangular_field_area_l1807_180738

theorem rectangular_field_area
  (x : ℝ) 
  (length := 3 * x) 
  (breadth := 4 * x) 
  (perimeter := 2 * (length + breadth))
  (cost_per_meter : ℝ := 0.25) 
  (total_cost : ℝ := 87.5) 
  (paise_per_rupee : ℝ := 100)
  (perimeter_eq_cost : 14 * x * cost_per_meter * paise_per_rupee = total_cost * paise_per_rupee) :
  (length * breadth = 7500) := 
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l1807_180738


namespace NUMINAMATH_GPT_factorization_correct_l1807_180793

noncomputable def factor_polynomial : Polynomial ℝ :=
  Polynomial.X^6 - 64

theorem factorization_correct : 
  factor_polynomial = 
  (Polynomial.X - 2) * 
  (Polynomial.X + 2) * 
  (Polynomial.X^4 + 4 * Polynomial.X^2 + 16) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1807_180793


namespace NUMINAMATH_GPT_multiplicative_magic_square_h_sum_l1807_180729

theorem multiplicative_magic_square_h_sum :
  ∃ (h_vals : List ℕ), 
  (∀ h ∈ h_vals, ∃ (e : ℕ), e > 0 ∧ 25 * e = h ∧ 
    ∃ (b c d f g : ℕ), 
    75 * b * c = d * e * f ∧ 
    d * e * f = g * h * 3 ∧ 
    g * h * 3 = c * f * 3 ∧ 
    c * f * 3 = 75 * e * g
  ) ∧ h_vals.sum = 150 :=
by { sorry }

end NUMINAMATH_GPT_multiplicative_magic_square_h_sum_l1807_180729


namespace NUMINAMATH_GPT_gcd_891_810_l1807_180768

theorem gcd_891_810 : Nat.gcd 891 810 = 81 := 
by
  sorry

end NUMINAMATH_GPT_gcd_891_810_l1807_180768


namespace NUMINAMATH_GPT_shaded_area_proof_l1807_180781

noncomputable def total_shaded_area (side_length: ℝ) (large_square_ratio: ℝ) (small_square_ratio: ℝ): ℝ := 
  let S := side_length / large_square_ratio
  let T := S / small_square_ratio
  let large_square_area := S ^ 2
  let small_square_area := T ^ 2
  large_square_area + 12 * small_square_area

theorem shaded_area_proof
  (h1: ∀ side_length, side_length = 15)
  (h2: ∀ large_square_ratio, large_square_ratio = 5)
  (h3: ∀ small_square_ratio, small_square_ratio = 4)
  : total_shaded_area 15 5 4 = 15.75 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_proof_l1807_180781


namespace NUMINAMATH_GPT_my_problem_l1807_180774

theorem my_problem (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  a^2 * b + b^2 * c + c^2 * a < a^2 * c + b^2 * a + c^2 * b := 
sorry

end NUMINAMATH_GPT_my_problem_l1807_180774


namespace NUMINAMATH_GPT_discounted_price_is_correct_l1807_180758

def original_price_of_cork (C : ℝ) : Prop :=
  C + (C + 2.00) = 2.10

def discounted_price_of_cork (C : ℝ) : ℝ :=
  C - (C * 0.12)

theorem discounted_price_is_correct :
  ∃ C : ℝ, original_price_of_cork C ∧ discounted_price_of_cork C = 0.044 :=
by
  sorry

end NUMINAMATH_GPT_discounted_price_is_correct_l1807_180758


namespace NUMINAMATH_GPT_cube_faces_l1807_180783

theorem cube_faces : ∀ (c : {s : Type | ∃ (x y z : ℝ), s = ({ (x0, y0, z0) : ℝ × ℝ × ℝ | x0 ≤ x ∧ y0 ≤ y ∧ z0 ≤ z}) }), 
  ∃ (f : ℕ), f = 6 :=
by 
  -- proof would be written here
  sorry

end NUMINAMATH_GPT_cube_faces_l1807_180783


namespace NUMINAMATH_GPT_sally_needs_8_napkins_l1807_180734

theorem sally_needs_8_napkins :
  let tablecloth_length := 102
  let tablecloth_width := 54
  let napkin_length := 6
  let napkin_width := 7
  let total_material_needed := 5844
  let tablecloth_area := tablecloth_length * tablecloth_width
  let napkin_area := napkin_length * napkin_width
  let material_needed_for_napkins := total_material_needed - tablecloth_area
  let number_of_napkins := material_needed_for_napkins / napkin_area
  number_of_napkins = 8 :=
by
  sorry

end NUMINAMATH_GPT_sally_needs_8_napkins_l1807_180734


namespace NUMINAMATH_GPT_natasha_destination_distance_l1807_180794

theorem natasha_destination_distance
  (over_speed : ℕ)
  (time : ℕ)
  (speed_limit : ℕ)
  (actual_speed : ℕ)
  (distance : ℕ) :
  (over_speed = 10) →
  (time = 1) →
  (speed_limit = 50) →
  (actual_speed = speed_limit + over_speed) →
  (distance = actual_speed * time) →
  (distance = 60) :=
by
  sorry

end NUMINAMATH_GPT_natasha_destination_distance_l1807_180794


namespace NUMINAMATH_GPT_certain_number_is_correct_l1807_180720

def m : ℕ := 72483

theorem certain_number_is_correct : 9999 * m = 724827405 := by
  sorry

end NUMINAMATH_GPT_certain_number_is_correct_l1807_180720


namespace NUMINAMATH_GPT_Bret_catches_12_frogs_l1807_180746

-- Conditions from the problem
def frogs_caught_by_Alster : Nat := 2
def frogs_caught_by_Quinn : Nat := 2 * frogs_caught_by_Alster
def frogs_caught_by_Bret : Nat := 3 * frogs_caught_by_Quinn

-- Statement of the theorem to be proved
theorem Bret_catches_12_frogs : frogs_caught_by_Bret = 12 :=
by
  sorry

end NUMINAMATH_GPT_Bret_catches_12_frogs_l1807_180746


namespace NUMINAMATH_GPT_max_x_y_given_condition_l1807_180799

theorem max_x_y_given_condition (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 1/x + 1/y = 5) : x + y ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_x_y_given_condition_l1807_180799


namespace NUMINAMATH_GPT_input_statement_is_INPUT_l1807_180727

-- Define the type for statements
inductive Statement
| PRINT
| INPUT
| IF
| END

-- Define roles for the types of statements
def isOutput (s : Statement) : Prop := s = Statement.PRINT
def isInput (s : Statement) : Prop := s = Statement.INPUT
def isConditional (s : Statement) : Prop := s = Statement.IF
def isTermination (s : Statement) : Prop := s = Statement.END

-- Theorem to prove INPUT is the input statement
theorem input_statement_is_INPUT :
  isInput Statement.INPUT := by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_input_statement_is_INPUT_l1807_180727


namespace NUMINAMATH_GPT_geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l1807_180737

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ} (hq : 0 < q) (hq2 : q ≠ 1)

-- ① If $a_{1}=1$ and the common ratio is $\frac{1}{2}$, then $S_{n} < 2$;
theorem geom_seq_sum_lt_two (h₁ : a 1 = 1) (hq_half : q = 1 / 2) (n : ℕ) : S n < 2 := sorry

-- ② The sequence $\{a_{n}^{2}\}$ must be a geometric sequence
theorem geom_seq_squared (h_geom : ∀ n, a (n + 1) = q * a n) : ∃ r : ℝ, ∀ n, a n ^ 2 = r ^ n := sorry

-- ④ For any positive integer $n$, $a{}_{n}^{2}+a{}_{n+2}^{2}\geqslant 2a{}_{n+1}^{2}$
theorem geom_seq_square_inequality (h_geom : ∀ n, a (n + 1) = q * a n) (n : ℕ) (hn : 0 < n) : 
  a n ^ 2 + a (n + 2) ^ 2 ≥ 2 * a (n + 1) ^ 2 := sorry

end NUMINAMATH_GPT_geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l1807_180737


namespace NUMINAMATH_GPT_inequality_proof_l1807_180788

variable {a b c : ℝ}

theorem inequality_proof (ha : a > 0) (hb : b > 0) (hc : c > 0) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1807_180788


namespace NUMINAMATH_GPT_negation_forall_pos_l1807_180740

theorem negation_forall_pos (h : ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) :
  ∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_forall_pos_l1807_180740


namespace NUMINAMATH_GPT_find_polynomial_l1807_180718

noncomputable def polynomial_p (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ t x y a b c : ℝ,
    (P (t * x) (t * y) = t ^ n * P x y) ∧
    (P (a + b) c + P (b + c) a + P (c + a) b = 0) ∧
    (P 1 0 = 1)

theorem find_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) (h : polynomial_p n P) :
  ∀ x y : ℝ, P x y = x^n - y^n :=
sorry

end NUMINAMATH_GPT_find_polynomial_l1807_180718


namespace NUMINAMATH_GPT_range_of_m_for_real_roots_value_of_m_for_specific_roots_l1807_180733

open Real

variable {m x : ℝ}

def quadratic (m : ℝ) (x : ℝ) := x^2 + 2*(m-1)*x + m^2 + 2 = 0
  
theorem range_of_m_for_real_roots (h : ∃ x : ℝ, quadratic m x) : m ≤ -1/2 :=
sorry

theorem value_of_m_for_specific_roots
  (h : quadratic m x)
  (Hroots : ∃ x1 x2 : ℝ, quadratic m x1 ∧ quadratic m x2 ∧ (x1 - x2)^2 = 18 - x1 * x2) :
  m = -2 :=
sorry

end NUMINAMATH_GPT_range_of_m_for_real_roots_value_of_m_for_specific_roots_l1807_180733


namespace NUMINAMATH_GPT_quadratic_k_value_l1807_180760

theorem quadratic_k_value (a b c k : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : 4 * b * b - k * a * c = 0): 
  k = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_k_value_l1807_180760


namespace NUMINAMATH_GPT_contrapositive_statement_l1807_180715

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬is_even n

theorem contrapositive_statement (a b : ℕ) :
  (¬(is_odd a ∧ is_odd b) ∧ ¬(is_even a ∧ is_even b)) → ¬is_even (a + b) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_statement_l1807_180715


namespace NUMINAMATH_GPT_min_repetitions_2002_div_by_15_l1807_180714

-- Define the function that generates the number based on repetitions of "2002" and appending "15"
def generate_number (n : ℕ) : ℕ :=
  let repeated := (List.replicate n 2002).foldl (λ acc x => acc * 10000 + x) 0
  repeated * 100 + 15

-- Define the minimum n for which the generated number is divisible by 15
def min_n_divisible_by_15 : ℕ := 3

-- The theorem stating the problem with its conditions (divisibility by 15)
theorem min_repetitions_2002_div_by_15 :
  ∀ n : ℕ, (generate_number n % 15 = 0) ↔ (n ≥ min_n_divisible_by_15) :=
sorry

end NUMINAMATH_GPT_min_repetitions_2002_div_by_15_l1807_180714


namespace NUMINAMATH_GPT_sequence_a_n_sum_T_n_l1807_180721

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (T : ℕ → ℕ)

theorem sequence_a_n (n : ℕ) (hS : ∀ n, S n = 2 * a n - n) :
  a n = 2 ^ n - 1 :=
sorry

theorem sum_T_n (n : ℕ) (hb : ∀ n, b n = (2 * n + 1) * (a n + 1)) 
  (ha : ∀ n, a n = 2 ^ n - 1) :
  T n = 2 + (2 * n - 1) * 2 ^ (n + 1) :=
sorry

end NUMINAMATH_GPT_sequence_a_n_sum_T_n_l1807_180721
