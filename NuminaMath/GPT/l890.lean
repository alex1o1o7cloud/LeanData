import Mathlib

namespace brick_wall_completion_time_l890_89025

def rate (hours : ℚ) : ℚ := 1 / hours

/-- Avery can build a brick wall in 3 hours. -/
def avery_rate : ℚ := rate 3
/-- Tom can build a brick wall in 2.5 hours. -/
def tom_rate : ℚ := rate 2.5
/-- Catherine can build a brick wall in 4 hours. -/
def catherine_rate : ℚ := rate 4
/-- Derek can build a brick wall in 5 hours. -/
def derek_rate : ℚ := rate 5

/-- Combined rate for Avery, Tom, and Catherine working together. -/
def combined_rate_1 : ℚ := avery_rate + tom_rate + catherine_rate
/-- Combined rate for Tom and Catherine working together. -/
def combined_rate_2 : ℚ := tom_rate + catherine_rate
/-- Combined rate for Tom, Catherine, and Derek working together. -/
def combined_rate_3 : ℚ := tom_rate + catherine_rate + derek_rate

/-- Total time taken to complete the wall. -/
def total_time (t : ℚ) : Prop :=
  t = 2

theorem brick_wall_completion_time (t : ℚ) : total_time t :=
by
  sorry

end brick_wall_completion_time_l890_89025


namespace pencils_ratio_l890_89057

theorem pencils_ratio (C J : ℕ) (hJ : J = 18) 
    (hJ_to_A : J_to_A = J / 3) (hJ_left : J_left = J - J_to_A)
    (hJ_left_eq : J_left = C + 3) :
    (C : ℚ) / (J : ℚ) = 1 / 2 :=
by
  sorry

end pencils_ratio_l890_89057


namespace quadruples_characterization_l890_89072

/-- Proving the characterization of quadruples (a, b, c, d) of non-negative integers 
such that ab = 2(1 + cd) and there exists a non-degenerate triangle with sides (a - c), 
(b - d), and (c + d). -/
theorem quadruples_characterization :
  ∀ (a b c d : ℕ), 
    ab = 2 * (1 + cd) ∧ 
    (a - c) + (b - d) > c + d ∧ 
    (a - c) + (c + d) > b - d ∧ 
    (b - d) + (c + d) > a - c ∧
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    (a = 1 ∧ b = 2 ∧ c = 0 ∧ d = 1) ∨ 
    (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 0) :=
by sorry

end quadruples_characterization_l890_89072


namespace find_number_of_each_coin_l890_89092

-- Define the number of coins
variables (n d q : ℕ)

-- Given conditions
axiom twice_as_many_nickels_as_quarters : n = 2 * q
axiom same_number_of_dimes_as_quarters : d = q
axiom total_value_of_coins : 5 * n + 10 * d + 25 * q = 1520

-- Statement to prove
theorem find_number_of_each_coin :
  q = 304 / 9 ∧
  n = 2 * (304 / 9) ∧
  d = 304 / 9 :=
sorry

end find_number_of_each_coin_l890_89092


namespace probability_blue_ball_l890_89038

-- Define the probabilities of drawing a red and yellow ball
def P_red : ℝ := 0.48
def P_yellow : ℝ := 0.35

-- Define the total probability formula in this sample space
def total_probability (P_red P_yellow P_blue : ℝ) : Prop :=
  P_red + P_yellow + P_blue = 1

-- The theorem we need to prove
theorem probability_blue_ball :
  ∃ P_blue : ℝ, total_probability P_red P_yellow P_blue ∧ P_blue = 0.17 :=
sorry

end probability_blue_ball_l890_89038


namespace part1_part2_l890_89053

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^3 + k * Real.log x
noncomputable def f' (x : ℝ) (k : ℝ) : ℝ := 3 * x^2 + k / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x k - f' x k + 9 / x

-- Part (1): Prove the monotonic intervals and extreme values for k = 6:
theorem part1 :
  (∀ x : ℝ, 0 < x ∧ x < 1 → g x 6 < g 1 6) ∧
  (∀ x : ℝ, 1 < x → g x 6 > g 1 6) ∧
  (g 1 6 = 1) := sorry

-- Part (2): Prove the given inequality for k ≥ -3:
theorem part2 (k : ℝ) (hk : k ≥ -3) (x1 x2 : ℝ) (hx1 : x1 ≥ 1) (hx2 : x2 ≥ 1) (h : x1 > x2) :
  (f' x1 k + f' x2 k) / 2 > (f x1 k - f x2 k) / (x1 - x2) := sorry

end part1_part2_l890_89053


namespace factorization_of_x10_minus_1024_l890_89046

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end factorization_of_x10_minus_1024_l890_89046


namespace fraction_of_yard_occupied_l890_89080

noncomputable def area_triangle_flower_bed : ℝ := 
  2 * (0.5 * (10:ℝ) * (10:ℝ))

noncomputable def area_circular_flower_bed : ℝ := 
  Real.pi * (2:ℝ)^2

noncomputable def total_area_flower_beds : ℝ := 
  area_triangle_flower_bed + area_circular_flower_bed

noncomputable def area_yard : ℝ := 
  (40:ℝ) * (10:ℝ)

noncomputable def fraction_occupied := 
  total_area_flower_beds / area_yard

theorem fraction_of_yard_occupied : 
  fraction_occupied = 0.2814 := 
sorry

end fraction_of_yard_occupied_l890_89080


namespace rectangle_area_error_percentage_l890_89066

theorem rectangle_area_error_percentage (L W : ℝ) :
  let L' := 1.10 * L
  let W' := 0.95 * W
  let A := L * W 
  let A' := L' * W'
  let error := A' - A
  let error_percentage := (error / A) * 100
  error_percentage = 4.5 := by
  sorry

end rectangle_area_error_percentage_l890_89066


namespace domain_of_h_l890_89079

open Real

theorem domain_of_h : ∀ x : ℝ, |x - 5| + |x + 2| ≠ 0 := by
  intro x
  sorry

end domain_of_h_l890_89079


namespace inequality_solution_range_of_a_l890_89021

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

def range_y := Set.Icc (-2 : ℝ) 2

def subset_property (a : ℝ) : Prop := 
  Set.Icc a (2 * a - 1) ⊆ range_y

theorem inequality_solution (x : ℝ) :
  f x ≤ x^2 - 3 * x + 1 ↔ x ≤ 1 ∨ x ≥ 3 := sorry

theorem range_of_a (a : ℝ) :
  subset_property a ↔ 1 ≤ a ∧ a ≤ 3 / 2 := sorry

end inequality_solution_range_of_a_l890_89021


namespace exists_n_good_but_not_succ_good_l890_89000

def S (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

def n_good (n : ℕ) (a : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), 
    a_seq n = a ∧ (∀ i : Fin n, a_seq (Fin.succ i) = a_seq i - S (a_seq i))

theorem exists_n_good_but_not_succ_good (n : ℕ) : 
  ∃ a, n_good n a ∧ ¬ n_good (n + 1) a := 
sorry

end exists_n_good_but_not_succ_good_l890_89000


namespace calculate_expression_l890_89035

theorem calculate_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) * (3 + 2) = X :=
by
  sorry

end calculate_expression_l890_89035


namespace corner_cells_different_colors_l890_89006

theorem corner_cells_different_colors 
  (colors : Fin 4 → Prop)
  (painted : (Fin 100 × Fin 100) → Fin 4)
  (h : ∀ (i j : Fin 99), 
    ∃ f g h k, 
      f ≠ g ∧ f ≠ h ∧ f ≠ k ∧
      g ≠ h ∧ g ≠ k ∧ 
      h ≠ k ∧ 
      painted (i, j) = f ∧ 
      painted (i.succ, j) = g ∧ 
      painted (i, j.succ) = h ∧ 
      painted (i.succ, j.succ) = k) :
  painted (0, 0) ≠ painted (99, 0) ∧
  painted (0, 0) ≠ painted (0, 99) ∧
  painted (0, 0) ≠ painted (99, 99) ∧
  painted (99, 0) ≠ painted (0, 99) ∧
  painted (99, 0) ≠ painted (99, 99) ∧
  painted (0, 99) ≠ painted (99, 99) :=
  sorry

end corner_cells_different_colors_l890_89006


namespace find_radius_l890_89099

-- Definitions based on conditions
def circle_radius (r : ℝ) : Prop := r = 2

-- Specification based on the question and conditions
theorem find_radius (r : ℝ) : circle_radius r :=
by
  -- Skip the proof
  sorry

end find_radius_l890_89099


namespace negation_of_square_positivity_l890_89082

theorem negation_of_square_positivity :
  (¬ ∀ n : ℕ, n * n > 0) ↔ (∃ n : ℕ, n * n ≤ 0) :=
  sorry

end negation_of_square_positivity_l890_89082


namespace days_to_complete_work_l890_89001

variable {P W D : ℕ}

axiom condition_1 : 2 * P * 3 = W / 2
axiom condition_2 : P * D = W

theorem days_to_complete_work : D = 12 :=
by
  -- As an axiom or sorry is used, the proof is omitted.
  sorry

end days_to_complete_work_l890_89001


namespace area_difference_zero_l890_89011

theorem area_difference_zero
  (AG CE : ℝ)
  (s : ℝ)
  (area_square area_rectangle : ℝ)
  (h1 : AG = 2)
  (h2 : CE = 2)
  (h3 : s = 2)
  (h4 : area_square = s^2)
  (h5 : area_rectangle = 2 * 2) :
  (area_square - area_rectangle = 0) :=
by sorry

end area_difference_zero_l890_89011


namespace final_amount_correct_l890_89088

def wallet_cost : ℝ := 22
def purse_cost : ℝ := 4 * wallet_cost - 3
def shoes_cost : ℝ := wallet_cost + purse_cost + 7
def total_cost_before_discount : ℝ := wallet_cost + purse_cost + shoes_cost
def discount_rate : ℝ := 0.10
def discounted_amount : ℝ := total_cost_before_discount * discount_rate
def final_amount : ℝ := total_cost_before_discount - discounted_amount

theorem final_amount_correct :
  final_amount = 198.90 := by
  -- Here we would provide the proof of the theorem
  sorry

end final_amount_correct_l890_89088


namespace range_of_x_plus_y_l890_89029

theorem range_of_x_plus_y (x y : ℝ) (hx1 : y = 3 * ⌊x⌋ + 4) (hx2 : y = 4 * ⌊x - 3⌋ + 7) (hxnint : ¬ ∃ z : ℤ, x = z): 
  40 < x + y ∧ x + y < 41 :=
by
  sorry

end range_of_x_plus_y_l890_89029


namespace honda_day_shift_production_l890_89058

theorem honda_day_shift_production (S : ℕ) (day_shift_production : ℕ)
  (h1 : day_shift_production = 4 * S)
  (h2 : day_shift_production + S = 5500) :
  day_shift_production = 4400 :=
sorry

end honda_day_shift_production_l890_89058


namespace no_convex_27gon_with_distinct_integer_angles_l890_89005

noncomputable def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

def is_convex (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i, angles i < 180

def all_distinct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i j, i ≠ j → angles i ≠ angles j

def sum_is_correct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  Finset.sum (Finset.univ : Finset (Fin n)) angles = sum_of_interior_angles n

theorem no_convex_27gon_with_distinct_integer_angles :
  ¬ ∃ (angles : Fin 27 → ℕ), is_convex 27 angles ∧ all_distinct 27 angles ∧ sum_is_correct 27 angles :=
by
  sorry

end no_convex_27gon_with_distinct_integer_angles_l890_89005


namespace number_of_members_in_league_l890_89096

-- Define the conditions
def pair_of_socks_cost := 4
def t_shirt_cost := pair_of_socks_cost + 6
def cap_cost := t_shirt_cost - 3
def total_cost_per_member := 2 * (pair_of_socks_cost + t_shirt_cost + cap_cost)
def league_total_expenditure := 3144

-- Prove that the number of members in the league is 75
theorem number_of_members_in_league : 
  (∃ (n : ℕ), total_cost_per_member * n = league_total_expenditure) → 
  (∃ (n : ℕ), n = 75) :=
by
  sorry

end number_of_members_in_league_l890_89096


namespace solution_set_of_inequality_l890_89039

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_inequality_l890_89039


namespace license_plates_count_l890_89015

def number_of_license_plates : ℕ :=
  let digit_choices := 10^5
  let letter_block_choices := 3 * 26^2
  let block_positions := 6
  digit_choices * letter_block_choices * block_positions

theorem license_plates_count : number_of_license_plates = 1216800000 := by
  -- proof steps here
  sorry

end license_plates_count_l890_89015


namespace jars_water_fraction_l890_89024

theorem jars_water_fraction (S L W : ℝ) (h1 : W = 1/6 * S) (h2 : W = 1/5 * L) : 
  (2 * W / L) = 2 / 5 :=
by
  -- We are only stating the theorem here, not proving it.
  sorry

end jars_water_fraction_l890_89024


namespace seventh_triangular_number_is_28_l890_89074

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_triangular_number_is_28 : triangular_number 7 = 28 :=
by
  /- proof goes here -/
  sorry

end seventh_triangular_number_is_28_l890_89074


namespace classrooms_student_rabbit_difference_l890_89095

-- Definitions from conditions
def students_per_classroom : Nat := 20
def rabbits_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Theorem statement
theorem classrooms_student_rabbit_difference :
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 102 := by
  sorry

end classrooms_student_rabbit_difference_l890_89095


namespace missing_number_is_twelve_l890_89093

theorem missing_number_is_twelve
  (x : ℤ)
  (h : 10010 - x * 3 * 2 = 9938) :
  x = 12 :=
sorry

end missing_number_is_twelve_l890_89093


namespace smallest_x_abs_eq_18_l890_89045

theorem smallest_x_abs_eq_18 : 
  ∃ x : ℝ, (|2 * x + 5| = 18) ∧ (∀ y : ℝ, (|2 * y + 5| = 18) → x ≤ y) :=
sorry

end smallest_x_abs_eq_18_l890_89045


namespace eliza_tom_difference_l890_89060

theorem eliza_tom_difference (q : ℕ) : 
  let eliza_quarters := 7 * q + 3
  let tom_quarters := 2 * q + 8
  let quarter_difference := (7 * q + 3) - (2 * q + 8)
  let nickel_value := 5
  let groups_of_5 := quarter_difference / 5
  let difference_in_cents := nickel_value * groups_of_5
  difference_in_cents = 5 * (q - 1) := by
  sorry

end eliza_tom_difference_l890_89060


namespace sum_a1_to_a5_l890_89014

-- Define the conditions
def equation_holds (x a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  x^5 + 2 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5

-- State the theorem
theorem sum_a1_to_a5 (a0 a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, equation_holds x a0 a1 a2 a3 a4 a5) :
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  sorry

end sum_a1_to_a5_l890_89014


namespace percentage_less_than_l890_89034

namespace PercentProblem

noncomputable def A (C : ℝ) : ℝ := 0.65 * C
noncomputable def B (C : ℝ) : ℝ := 0.8923076923076923 * A C

theorem percentage_less_than (C : ℝ) (hC : C ≠ 0) : (C - B C) / C = 0.42 :=
by
  sorry

end PercentProblem

end percentage_less_than_l890_89034


namespace front_view_correct_l890_89044

-- Define the number of blocks in each column
def Blocks_Column_A : Nat := 3
def Blocks_Column_B : Nat := 5
def Blocks_Column_C : Nat := 2
def Blocks_Column_D : Nat := 4

-- Define the front view representation
def front_view : List Nat := [3, 5, 2, 4]

-- Statement to be proved
theorem front_view_correct :
  [Blocks_Column_A, Blocks_Column_B, Blocks_Column_C, Blocks_Column_D] = front_view :=
by
  sorry

end front_view_correct_l890_89044


namespace ratio_of_speeds_l890_89048

variable (a b : ℝ)

theorem ratio_of_speeds (h1 : b = 1 / 60) (h2 : a + b = 1 / 12) : a / b = 4 := 
sorry

end ratio_of_speeds_l890_89048


namespace solve_puzzle_l890_89028

theorem solve_puzzle (x1 x2 x3 x4 x5 x6 x7 x8 : ℕ) : 
  (8 + x1 + x2 = 20) →
  (x1 + x2 + x3 = 20) →
  (x2 + x3 + x4 = 20) →
  (x3 + x4 + x5 = 20) →
  (x4 + x5 + 5 = 20) →
  (x5 + 5 + x6 = 20) →
  (5 + x6 + x7 = 20) →
  (x6 + x7 + x8 = 20) →
  (x1 = 7 ∧ x2 = 5 ∧ x3 = 8 ∧ x4 = 7 ∧ x5 = 5 ∧ x6 = 8 ∧ x7 = 7 ∧ x8 = 5) :=
by {
  sorry
}

end solve_puzzle_l890_89028


namespace quadrilateral_angle_cosine_proof_l890_89098

variable (AB BC CD AD : ℝ)
variable (ϕ B C : ℝ)

theorem quadrilateral_angle_cosine_proof :
  AD^2 = AB^2 + BC^2 + CD^2 - 2 * (AB * BC * Real.cos B + BC * CD * Real.cos C + CD * AB * Real.cos ϕ) :=
by
  sorry

end quadrilateral_angle_cosine_proof_l890_89098


namespace birth_date_16_Jan_1993_l890_89037

noncomputable def year_of_birth (current_date : Nat) (age_years : Nat) :=
  current_date - age_years * 365

noncomputable def month_of_birth (current_date : Nat) (age_years : Nat) (age_months : Nat) :=
  current_date - (age_years * 12 + age_months) * 30

theorem birth_date_16_Jan_1993 :
  let boy_age_years := 10
  let boy_age_months := 1
  let current_date := 16 + 31 * 12 * 2003 -- 16th February 2003 represented in days
  let full_months_lived := boy_age_years * 12 + boy_age_months
  full_months_lived - boy_age_years = 111 → 
  year_of_birth current_date boy_age_years = 1993 ∧ month_of_birth current_date boy_age_years boy_age_months = 31 * 1 * 1993 := 
sorry

end birth_date_16_Jan_1993_l890_89037


namespace pages_difference_l890_89009

def second_chapter_pages : ℕ := 18
def third_chapter_pages : ℕ := 3

theorem pages_difference : second_chapter_pages - third_chapter_pages = 15 := by 
  sorry

end pages_difference_l890_89009


namespace value_of_x_l890_89062

theorem value_of_x :
  ∀ (x : ℕ), 
    x = 225 + 2 * 15 * 9 + 81 → 
    x = 576 := 
by
  intro x h
  sorry

end value_of_x_l890_89062


namespace Marcus_walking_speed_l890_89031

def bath_time : ℕ := 20  -- in minutes
def blow_dry_time : ℕ := bath_time / 2  -- in minutes
def trail_distance : ℝ := 3  -- in miles
def total_dog_time : ℕ := 60  -- in minutes

theorem Marcus_walking_speed :
  let walking_time := total_dog_time - (bath_time + blow_dry_time)
  let walking_time_hours := (walking_time:ℝ) / 60
  (trail_distance / walking_time_hours) = 6 := by
  sorry

end Marcus_walking_speed_l890_89031


namespace skipping_rates_l890_89012

theorem skipping_rates (x y : ℕ) (h₀ : 300 / (x + 19) = 270 / x) (h₁ : y = x + 19) :
  x = 171 ∧ y = 190 := by
  sorry

end skipping_rates_l890_89012


namespace axis_of_symmetry_parabola_eq_l890_89003

theorem axis_of_symmetry_parabola_eq : ∀ (x y p : ℝ), 
  y = -2 * x^2 → 
  (x^2 = -2 * p * y) → 
  (p = 1/4) →  
  (y = p / 2) → 
  y = 1 / 8 := by 
  intros x y p h1 h2 h3 h4
  sorry

end axis_of_symmetry_parabola_eq_l890_89003


namespace solution_set_of_quadratic_l890_89089

theorem solution_set_of_quadratic (a b x : ℝ) (h1 : a = 5) (h2 : b = -6) :
  (2 ≤ x ∧ x ≤ 3) → (bx^2 - ax - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end solution_set_of_quadratic_l890_89089


namespace Jane_age_l890_89030

theorem Jane_age (x : ℕ) 
  (h1 : ∃ n1 : ℕ, x - 1 = n1 ^ 2) 
  (h2 : ∃ n2 : ℕ, x + 1 = n2 ^ 3) : 
  x = 26 :=
sorry

end Jane_age_l890_89030


namespace leak_drain_time_l890_89050

theorem leak_drain_time :
  ∀ (P L : ℝ),
  P = 1/6 →
  P - L = 1/12 →
  (1/L) = 12 :=
by
  intros P L hP hPL
  sorry

end leak_drain_time_l890_89050


namespace fraction_to_decimal_l890_89026

theorem fraction_to_decimal (n d : ℕ) (hn : n = 53) (hd : d = 160) (gcd_nd : Nat.gcd n d = 1)
  (prime_factorization_d : ∃ k l : ℕ, d = 2^k * 5^l) : ∃ dec : ℚ, (n:ℚ) / (d:ℚ) = dec ∧ dec = 0.33125 :=
by sorry

end fraction_to_decimal_l890_89026


namespace all_have_perp_property_l890_89004

def M₁ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, x^3 - 2 * x^2 + 3)}
def M₂ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, Real.log (2 - x) / Real.log 2)}
def M₃ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 2 - 2^x)}
def M₄ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 1 - Real.sin x)}

def perp_property (M : Set (ℝ × ℝ)) : Prop :=
∀ p ∈ M, ∃ q ∈ M, p.1 * q.1 + p.2 * q.2 = 0

-- Theorem statement
theorem all_have_perp_property :
  perp_property M₁ ∧ perp_property M₂ ∧ perp_property M₃ ∧ perp_property M₄ :=
sorry

end all_have_perp_property_l890_89004


namespace divisible_by_two_of_square_l890_89022

theorem divisible_by_two_of_square {a : ℤ} (h : 2 ∣ a^2) : 2 ∣ a :=
sorry

end divisible_by_two_of_square_l890_89022


namespace find_number_of_values_l890_89018

theorem find_number_of_values (n S : ℕ) (h1 : S / n = 250) (h2 : S + 30 = 251 * n) : n = 30 :=
sorry

end find_number_of_values_l890_89018


namespace set_inter_complement_l890_89008

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem set_inter_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  sorry

end set_inter_complement_l890_89008


namespace average_age_condition_l890_89090

theorem average_age_condition (n : ℕ) 
  (h1 : (↑n * 14) / n = 14) 
  (h2 : ((↑n * 14) + 34) / (n + 1) = 16) : 
  n = 9 := 
by 
-- Proof goes here
sorry

end average_age_condition_l890_89090


namespace carol_sold_cupcakes_l890_89085

variable (initial_cupcakes := 30) (additional_cupcakes := 28) (final_cupcakes := 49)

theorem carol_sold_cupcakes : (initial_cupcakes + additional_cupcakes - final_cupcakes = 9) :=
by sorry

end carol_sold_cupcakes_l890_89085


namespace quadratic_inequality_solution_l890_89049

theorem quadratic_inequality_solution (x : ℝ) : 2 * x^2 - 5 * x - 3 ≥ 0 ↔ x ≤ -1/2 ∨ x ≥ 3 := 
by
  sorry

end quadratic_inequality_solution_l890_89049


namespace scale_length_l890_89077

theorem scale_length (length_of_part : ℕ) (number_of_parts : ℕ) (h1 : number_of_parts = 2) (h2 : length_of_part = 40) :
  number_of_parts * length_of_part = 80 := 
by
  sorry

end scale_length_l890_89077


namespace minimum_value_of_quadratic_polynomial_l890_89081

-- Define the quadratic polynomial
def quadratic_polynomial (x : ℝ) : ℝ := x^2 + 14 * x + 3

-- Statement to prove
theorem minimum_value_of_quadratic_polynomial : ∃ x : ℝ, quadratic_polynomial x = quadratic_polynomial (-7) :=
sorry

end minimum_value_of_quadratic_polynomial_l890_89081


namespace calvin_buys_chips_days_per_week_l890_89041

-- Define the constants based on the problem conditions
def cost_per_pack : ℝ := 0.50
def total_amount_spent : ℝ := 10
def number_of_weeks : ℕ := 4

-- Define the proof statement
theorem calvin_buys_chips_days_per_week : 
  (total_amount_spent / cost_per_pack) / number_of_weeks = 5 := 
by
  -- Placeholder proof
  sorry

end calvin_buys_chips_days_per_week_l890_89041


namespace k_not_possible_l890_89087

theorem k_not_possible (S : ℕ → ℚ) (a b : ℕ → ℚ) (n k : ℕ) (k_gt_2 : k > 2) :
  (S n = (n^2 + n) / 2) →
  (a n = S n - S (n - 1)) →
  (b n = 1 / a n) →
  (2 * b (n + 2) = b n + b (n + k)) →
  k ≠ 4 ∧ k ≠ 10 :=
by
  -- Proof goes here (skipped)
  sorry

end k_not_possible_l890_89087


namespace original_volume_l890_89007

variable {π : Real} (r h : Real)

theorem original_volume (hπ : π ≠ 0) (hr : r ≠ 0) (hh : h ≠ 0) (condition : 3 * π * r^2 * h = 180) : π * r^2 * h = 60 := by
  sorry

end original_volume_l890_89007


namespace geometric_sequence_expression_l890_89061

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (h_q : q = 4)
  (h_geom : ∀ n, a (n + 1) = q * a n) (h_sum : a 0 + a 1 + a 2 = 21) :
  ∀ n, a n = 4 ^ n :=
by sorry

end geometric_sequence_expression_l890_89061


namespace expand_and_simplify_l890_89042

theorem expand_and_simplify :
  (x : ℝ) → (x^2 - 3 * x + 3) * (x^2 + 3 * x + 3) = x^4 - 3 * x^2 + 9 :=
by 
  sorry

end expand_and_simplify_l890_89042


namespace inequality_solution_l890_89094

-- Define the condition for the denominator being positive
def denom_positive (x : ℝ) : Prop :=
  x^2 + 2*x + 7 > 0

-- Statement of the problem
theorem inequality_solution (x : ℝ) (h : denom_positive x) :
  (x + 6) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ -6 :=
sorry

end inequality_solution_l890_89094


namespace gifts_needed_l890_89073

def num_teams : ℕ := 7
def num_gifts_per_team : ℕ := 2

theorem gifts_needed (h1 : num_teams = 7) (h2 : num_gifts_per_team = 2) : num_teams * num_gifts_per_team = 14 := 
by
  -- proof skipped
  sorry

end gifts_needed_l890_89073


namespace contrapositive_proof_l890_89032

theorem contrapositive_proof (a b : ℕ) : (a = 1 ∧ b = 2) → (a + b = 3) :=
by {
  sorry
}

end contrapositive_proof_l890_89032


namespace rock_paper_scissors_score_divisible_by_3_l890_89070

theorem rock_paper_scissors_score_divisible_by_3 
  (R : ℕ) 
  (rock_shown : ℕ) 
  (scissors_shown : ℕ) 
  (paper_shown : ℕ)
  (points : ℕ)
  (h_equal_shows : 3 * ((rock_shown + scissors_shown + paper_shown) / 3) = rock_shown + scissors_shown + paper_shown)
  (h_points_awarded : ∀ (r s p : ℕ), r + s + p = 3 → (r = 2 ∧ s = 1 ∧ p = 0) ∨ (r = 0 ∧ s = 2 ∧ p = 1) ∨ (r = 1 ∧ s = 0 ∧ p = 2) → points % 3 = 0) :
  points % 3 = 0 := 
sorry

end rock_paper_scissors_score_divisible_by_3_l890_89070


namespace largest_multiple_negation_greater_than_neg150_l890_89075

theorem largest_multiple_negation_greater_than_neg150 (n : ℤ) (h₁ : n % 6 = 0) (h₂ : -n > -150) : n = 144 := 
sorry

end largest_multiple_negation_greater_than_neg150_l890_89075


namespace infinitely_many_sum_form_l890_89054

theorem infinitely_many_sum_form {a : ℕ → ℕ} (h : ∀ n, a n < a (n + 1)) :
  ∀ i, ∃ᶠ n in at_top, ∃ r s j, r > 0 ∧ s > 0 ∧ i < j ∧ a n = r * a i + s * a j := 
by
  sorry

end infinitely_many_sum_form_l890_89054


namespace system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l890_89047

theorem system_of_inequalities_solution_set : 
  (∀ x : ℝ, (2 * x - 1 < 7) → (x + 1 > 2) ↔ (1 < x ∧ x < 4)) := 
by 
  sorry

theorem quadratic_equation_when_m_is_2 : 
  (∀ x : ℝ, x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3)) := 
by 
  sorry

end system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l890_89047


namespace min_value_of_derivative_l890_89086

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1 / a) * x

noncomputable def f' (a : ℝ) : ℝ := 3 * 2^2 + 4 * a * 2 + (1 / a)

theorem min_value_of_derivative (a : ℝ) (h : a > 0) : 
  f' a ≥ 12 + 8 * Real.sqrt 2 :=
sorry

end min_value_of_derivative_l890_89086


namespace combined_perimeter_two_right_triangles_l890_89063

theorem combined_perimeter_two_right_triangles :
  ∀ (h1 h2 : ℝ),
    (h1^2 = 15^2 + 20^2) ∧
    (h2^2 = 9^2 + 12^2) ∧
    (h1 = h2) →
    (15 + 20 + h1) + (9 + 12 + h2) = 106 := by
  sorry

end combined_perimeter_two_right_triangles_l890_89063


namespace bromine_is_liquid_at_25C_1atm_l890_89052

-- Definitions for the melting and boiling points
def melting_point (element : String) : Float :=
  match element with
  | "Br" => -7.2
  | "Kr" => -157.4 -- Not directly used, but included for completeness
  | "P" => 44.1 -- Not directly used, but included for completeness
  | "Xe" => -111.8 -- Not directly used, but included for completeness
  | _ => 0.0 -- default case; not used

def boiling_point (element : String) : Float :=
  match element with
  | "Br" => 58.8
  | "Kr" => -153.4
  | "P" => 280.5 -- Not directly used, but included for completeness
  | "Xe" => -108.1
  | _ => 0.0 -- default case; not used

-- Define the condition of the problem
def is_liquid_at (element : String) (temperature : Float) (pressure : Float) : Bool :=
  melting_point element < temperature ∧ temperature < boiling_point element

-- Goal statement
theorem bromine_is_liquid_at_25C_1atm : is_liquid_at "Br" 25 1 = true :=
by
  sorry

end bromine_is_liquid_at_25C_1atm_l890_89052


namespace ellipse_slope_condition_l890_89036

theorem ellipse_slope_condition (a b x y x₀ y₀ : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h_ellipse1 : x^2 / a^2 + y^2 / b^2 = 1) 
  (h_ellipse2 : x₀^2 / a^2 + y₀^2 / b^2 = 1) 
  (hA : x ≠ x₀ ∨ y ≠ y₀) 
  (hB : x ≠ -x₀ ∨ y ≠ -y₀) :
  ((y - y₀) / (x - x₀)) * ((y + y₀) / (x + x₀)) = -b^2 / a^2 := 
sorry

end ellipse_slope_condition_l890_89036


namespace sum_of_pairs_l890_89076

theorem sum_of_pairs (a : ℕ → ℝ) (h1 : ∀ n, a n ≠ 0)
  (h2 : ∀ n, a n * a (n + 3) = a (n + 2) * a (n + 5))
  (h3 : a 1 * a 2 + a 3 * a 4 + a 5 * a 6 = 6) :
  a 1 * a 2 + a 3 * a 4 + a 5 * a 6 + a 7 * a 8 + a 9 * a 10 + a 11 * a 12 + 
  a 13 * a 14 + a 15 * a 16 + a 17 * a 18 + a 19 * a 20 + a 21 * a 22 + 
  a 23 * a 24 + a 25 * a 26 + a 27 * a 28 + a 29 * a 30 + a 31 * a 32 + 
  a 33 * a 34 + a 35 * a 36 + a 37 * a 38 + a 39 * a 40 + a 41 * a 42 = 42 := 
sorry

end sum_of_pairs_l890_89076


namespace compare_pow_value_l890_89056

theorem compare_pow_value : 
  ∀ (x : ℝ) (n : ℕ), x = 0.01 → n = 1000 → (1 + x)^n > 1000 := 
by 
  intros x n hx hn
  rw [hx, hn]
  sorry

end compare_pow_value_l890_89056


namespace expand_polynomial_l890_89017

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l890_89017


namespace range_of_a_l890_89019

theorem range_of_a (x a : ℝ) (h₁ : x > 1) (h₂ : a ≤ x + 1 / (x - 1)) : 
  a < 3 :=
sorry

end range_of_a_l890_89019


namespace quadratic_zeros_l890_89010

theorem quadratic_zeros (a b : ℝ) (h1 : (4 - 2 * a + b = 0)) (h2 : (9 + 3 * a + b = 0)) : a + b = -7 := 
by
  sorry

end quadratic_zeros_l890_89010


namespace num_pairs_eq_seven_l890_89068

theorem num_pairs_eq_seven :
  ∃ S : Finset (Nat × Nat), 
    (∀ (a b : Nat), (a, b) ∈ S ↔ (0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧ (a + 1 / b) / (1 / a + b) = 13)) ∧
    S.card = 7 :=
sorry

end num_pairs_eq_seven_l890_89068


namespace determine_c_l890_89083

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem determine_c (a b : ℝ) (m c : ℝ) 
  (h1 : ∀ x, 0 ≤ x → f x a b = x^2 + a * x + b)
  (h2 : ∃ m : ℝ, ∀ x : ℝ, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
sorry

end determine_c_l890_89083


namespace given_problem_l890_89084

noncomputable def improper_fraction_5_2_7 : ℚ := 37 / 7
noncomputable def improper_fraction_6_1_3 : ℚ := 19 / 3
noncomputable def improper_fraction_3_1_2 : ℚ := 7 / 2
noncomputable def improper_fraction_2_1_5 : ℚ := 11 / 5

theorem given_problem :
  71 * (improper_fraction_5_2_7 - improper_fraction_6_1_3) / (improper_fraction_3_1_2 + improper_fraction_2_1_5) = -13 - 37 / 1197 := 
  sorry

end given_problem_l890_89084


namespace inequality_correct_l890_89091

variable {a b c : ℝ}

theorem inequality_correct (h : a * b < 0) : |a - c| ≤ |a - b| + |b - c| :=
sorry

end inequality_correct_l890_89091


namespace find_three_fifths_of_neg_twelve_sevenths_l890_89064

def a : ℚ := -12 / 7
def b : ℚ := 3 / 5
def c : ℚ := -36 / 35

theorem find_three_fifths_of_neg_twelve_sevenths : b * a = c := by 
  -- sorry is a placeholder for the actual proof
  sorry

end find_three_fifths_of_neg_twelve_sevenths_l890_89064


namespace valid_numbers_eq_l890_89040

-- Definition of the number representation
def is_valid_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999 ∧
  ∃ (a b c : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    x = 100 * a + 10 * b + c ∧
    x = a^3 + b^3 + c^3

-- The theorem to prove
theorem valid_numbers_eq : 
  {x : ℕ | is_valid_number x} = {153, 407} :=
by
  sorry

end valid_numbers_eq_l890_89040


namespace tutors_next_together_l890_89055

-- Define the conditions given in the problem
def Elisa_work_days := 5
def Frank_work_days := 6
def Giselle_work_days := 8
def Hector_work_days := 9

-- Theorem statement to prove the number of days until they all work together again
theorem tutors_next_together (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = Elisa_work_days) 
  (h2 : d2 = Frank_work_days) 
  (h3 : d3 = Giselle_work_days) 
  (h4 : d4 = Hector_work_days) : 
  Nat.lcm (Nat.lcm (Nat.lcm d1 d2) d3) d4 = 360 := 
by
  -- Translate the problem statement into Lean terms and structure
  sorry

end tutors_next_together_l890_89055


namespace smallest_nat_satisfying_conditions_l890_89071

theorem smallest_nat_satisfying_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 2) ∧ 
  (x % 5 = 2) ∧ 
  (x % 6 = 2) ∧ 
  (x % 12 = 2) ∧ 
  (∀ y : ℕ, (y % 4 = 2) ∧ (y % 5 = 2) ∧ (y % 6 = 2) ∧ (y % 12 = 2) → x ≤ y) :=
  sorry

end smallest_nat_satisfying_conditions_l890_89071


namespace blown_out_sand_dunes_l890_89078

theorem blown_out_sand_dunes (p_remain p_lucky p_both : ℝ) (h_rem: p_remain = 1 / 3) (h_luck: p_lucky = 2 / 3)
(h_both: p_both = 0.08888888888888889) : 
  ∃ N : ℕ, N = 8 :=
by
  sorry

end blown_out_sand_dunes_l890_89078


namespace interval_between_births_l890_89067

variables {A1 A2 A3 A4 A5 : ℝ}
variable {x : ℝ}

def ages (A1 A2 A3 A4 A5 : ℝ) := A1 + A2 + A3 + A4 + A5 = 50
def youngest (A1 : ℝ) := A1 = 4
def interval (x : ℝ) := x = 3.4

theorem interval_between_births
  (h_age_sum: ages A1 A2 A3 A4 A5)
  (h_youngest: youngest A1)
  (h_ages: A2 = A1 + x ∧ A3 = A1 + 2 * x ∧ A4 = A1 + 3 * x ∧ A5 = A1 + 4 * x) :
  interval x :=
by {
  sorry
}

end interval_between_births_l890_89067


namespace log_x_y_eq_sqrt_3_l890_89097

variable (x y z : ℝ)
variable (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
variable (h1 : x ^ (Real.log z / Real.log y) = 2)
variable (h2 : y ^ (Real.log x / Real.log y) = 4)
variable (h3 : z ^ (Real.log y / Real.log x) = 8)

theorem log_x_y_eq_sqrt_3 : Real.log y / Real.log x = Real.sqrt 3 :=
by
  sorry

end log_x_y_eq_sqrt_3_l890_89097


namespace product_equation_l890_89033

theorem product_equation (a b : ℝ) (h1 : ∀ (a b : ℝ), 0.2 * b = 0.9 * a - b) : 
  0.9 * a - b = 0.2 * b :=
by
  sorry

end product_equation_l890_89033


namespace cyclic_points_exist_l890_89065

noncomputable def f (x : ℝ) : ℝ := 
if x < (1 / 3) then 
  2 * x + (1 / 3) 
else 
  (3 / 2) * (1 - x)

theorem cyclic_points_exist :
  ∃ (x0 x1 x2 x3 x4 : ℝ), 
  0 ≤ x0 ∧ x0 ≤ 1 ∧
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  0 ≤ x4 ∧ x4 ≤ 1 ∧
  x0 ≠ x1 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ x4 ≠ x0 ∧
  f x0 = x1 ∧ f x1 = x2 ∧ f x2 = x3 ∧ f x3 = x4 ∧ f x4 = x0 :=
sorry

end cyclic_points_exist_l890_89065


namespace boxes_of_bolts_purchased_l890_89013

theorem boxes_of_bolts_purchased 
  (bolts_per_box : ℕ) 
  (nuts_per_box : ℕ) 
  (num_nut_boxes : ℕ) 
  (leftover_bolts : ℕ) 
  (leftover_nuts : ℕ) 
  (total_bolts_nuts_used : ℕ)
  (B : ℕ) :
  bolts_per_box = 11 →
  nuts_per_box = 15 →
  num_nut_boxes = 3 →
  leftover_bolts = 3 →
  leftover_nuts = 6 →
  total_bolts_nuts_used = 113 →
  B = 7 :=
by
  intros
  sorry

end boxes_of_bolts_purchased_l890_89013


namespace a_n_formula_b_n_geometric_sequence_l890_89069

noncomputable def a_n (n : ℕ) : ℝ := 3 * n - 1

def S_n (n : ℕ) : ℝ := sorry -- Sum of the first n terms of b_n

def b_n (n : ℕ) : ℝ := 2 - 2 * S_n n

theorem a_n_formula (n : ℕ) : a_n n = 3 * n - 1 :=
by { sorry }

theorem b_n_geometric_sequence : ∀ n ≥ 2, b_n n / b_n (n - 1) = 1 / 3 :=
by { sorry }

end a_n_formula_b_n_geometric_sequence_l890_89069


namespace find_k_l890_89023

noncomputable def curve_C (x y : ℝ) : Prop :=
  x^2 + (y^2 / 4) = 1

noncomputable def line_eq (k x y : ℝ) : Prop :=
  y = k * x + 1

theorem find_k (k : ℝ) :
  (∃ A B : ℝ × ℝ, (curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧ line_eq k A.1 A.2 ∧ line_eq k B.1 B.2 ∧ 
   (A.1 * B.1 + A.2 * B.2 = 0))) ↔ (k = 1/2 ∨ k = -1/2) :=
sorry

end find_k_l890_89023


namespace work_completion_l890_89020

theorem work_completion (A B C : ℝ) (h₁ : A + B = 1 / 18) (h₂ : B + C = 1 / 24) (h₃ : A + C = 1 / 36) : 
  1 / (A + B + C) = 16 := 
by
  sorry

end work_completion_l890_89020


namespace perfect_square_quotient_l890_89059

theorem perfect_square_quotient {a b : ℕ} (hpos: 0 < a ∧ 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end perfect_square_quotient_l890_89059


namespace card_draw_count_l890_89016

theorem card_draw_count : 
  let total_cards := 12
  let red_cards := 3
  let yellow_cards := 3
  let blue_cards := 3
  let green_cards := 3
  let total_ways := Nat.choose total_cards 3
  let invalid_same_color := 4 * Nat.choose 3 3
  let invalid_two_red := Nat.choose red_cards 2 * Nat.choose (total_cards - red_cards) 1
  total_ways - invalid_same_color - invalid_two_red = 189 :=
by
  sorry

end card_draw_count_l890_89016


namespace sufficient_not_necessary_l890_89043

theorem sufficient_not_necessary (b c: ℝ) : (c < 0) → ∃ x y : ℝ, x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0 :=
by
  sorry

end sufficient_not_necessary_l890_89043


namespace intersection_point_exists_l890_89027

theorem intersection_point_exists
  (m n a b : ℝ)
  (h1 : m * a + 2 * m * b = 5)
  (h2 : n * a - 2 * n * b = 7)
  : (∃ x y : ℝ, 
    (y = (5 / (2 * m)) - (1 / 2) * x) ∧ 
    (y = (1 / 2) * x - (7 / (2 * n))) ∧
    (x = a) ∧ (y = b)) :=
sorry

end intersection_point_exists_l890_89027


namespace pizza_area_percentage_increase_l890_89002

theorem pizza_area_percentage_increase :
  let r1 := 6
  let r2 := 4
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  let deltaA := A1 - A2
  let N := (deltaA / A2) * 100
  N = 125 := by
  sorry

end pizza_area_percentage_increase_l890_89002


namespace maximum_candies_after_20_hours_l890_89051

-- Define a function to compute the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Define the recursive function to model the candy process
def candies_after_hours (n : ℕ) (hours : ℕ) : ℕ :=
  if hours = 0 then n 
  else candies_after_hours (n + sum_of_digits n) (hours - 1)

theorem maximum_candies_after_20_hours :
  candies_after_hours 1 20 = 148 :=
sorry

end maximum_candies_after_20_hours_l890_89051
