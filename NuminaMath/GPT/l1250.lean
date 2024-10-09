import Mathlib

namespace domain_of_f_x_plus_2_l1250_125061

theorem domain_of_f_x_plus_2 (f : ℝ → ℝ) (dom_f_x_minus_1 : ∀ x, 1 ≤ x ∧ x ≤ 2 → 0 ≤ x-1 ∧ x-1 ≤ 1) :
  ∀ y, 0 ≤ y ∧ y ≤ 1 ↔ -2 ≤ y-2 ∧ y-2 ≤ -1 :=
by
  sorry

end domain_of_f_x_plus_2_l1250_125061


namespace evaluate_expression_l1250_125037

theorem evaluate_expression : 4^1 + 3^2 - 2^3 + 1^4 = 6 := by
  -- We will skip the proof steps here using sorry
  sorry

end evaluate_expression_l1250_125037


namespace annual_decrease_rate_l1250_125081

def initial_population : ℝ := 8000
def population_after_two_years : ℝ := 3920

theorem annual_decrease_rate :
  ∃ r : ℝ, (0 < r ∧ r < 1) ∧ (initial_population * (1 - r)^2 = population_after_two_years) ∧ r = 0.3 :=
by
  sorry

end annual_decrease_rate_l1250_125081


namespace organizingCommitteeWays_l1250_125024

-- Define the problem context
def numberOfTeams : Nat := 5
def membersPerTeam : Nat := 8
def hostTeamSelection : Nat := 4
def otherTeamsSelection : Nat := 2

-- Define binomial coefficient
def binom (n k : Nat) : Nat := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of ways to select committee members
def totalCommitteeWays : Nat := numberOfTeams * 
                                 (binom membersPerTeam hostTeamSelection) * 
                                 ((binom membersPerTeam otherTeamsSelection) ^ (numberOfTeams - 1))

-- The theorem to prove
theorem organizingCommitteeWays : 
  totalCommitteeWays = 215134600 := 
    sorry

end organizingCommitteeWays_l1250_125024


namespace faster_speed_l1250_125080

theorem faster_speed (D : ℝ) (v : ℝ) (h₁ : D = 33.333333333333336) 
                      (h₂ : 10 * (D + 20) = v * D) : v = 16 :=
by
  sorry

end faster_speed_l1250_125080


namespace prob_divisible_by_5_l1250_125020

theorem prob_divisible_by_5 (M: ℕ) (h1: 100 ≤ M ∧ M < 1000) (h2: M % 10 = 5): 
  (∃ (k: ℕ), M = 5 * k) :=
by
  sorry

end prob_divisible_by_5_l1250_125020


namespace number_of_members_l1250_125056

def cost_knee_pads : ℤ := 6
def cost_jersey : ℤ := cost_knee_pads + 7
def total_cost_per_member : ℤ := 2 * (cost_knee_pads + cost_jersey)
def total_expenditure : ℤ := 3120

theorem number_of_members (n : ℤ) (h : n * total_cost_per_member = total_expenditure) : n = 82 :=
sorry

end number_of_members_l1250_125056


namespace crayons_allocation_correct_l1250_125099

noncomputable def crayons_allocation : Prop :=
  ∃ (F B J S : ℕ), 
    F + B + J + S = 96 ∧ 
    F = 2 * B ∧ 
    J = 3 * S ∧ 
    B = 12 ∧ 
    F = 24 ∧ 
    J = 45 ∧ 
    S = 15

theorem crayons_allocation_correct : crayons_allocation :=
  sorry

end crayons_allocation_correct_l1250_125099


namespace orthocenter_circumradii_equal_l1250_125044

-- Define a triangle with its orthocenter and circumradius
variables {A B C H : Point} (R r : ℝ)

-- Assume H is the orthocenter of triangle ABC
def is_orthocenter (H : Point) (A B C : Point) : Prop := 
  sorry -- This should state the definition or properties of an orthocenter

-- Assume the circumradius of triangle ABC is R 
def is_circumradius_ABC (A B C : Point) (R : ℝ) : Prop :=
  sorry -- This should capture the circumradius property

-- Assume circumradius of triangle BHC is r
def is_circumradius_BHC (B H C : Point) (r : ℝ) : Prop :=
  sorry -- This should capture the circumradius property
  
-- Prove that if H is the orthocenter of triangle ABC, the circumradius of ABC is R 
-- and the circumradius of BHC is r, then R = r
theorem orthocenter_circumradii_equal (h_orthocenter : is_orthocenter H A B C) 
  (h_circumradius_ABC : is_circumradius_ABC A B C R)
  (h_circumradius_BHC : is_circumradius_BHC B H C r) : R = r :=
  sorry

end orthocenter_circumradii_equal_l1250_125044


namespace value_of_a5_l1250_125082

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n m : ℕ, a n * r ^ (m - n) = a m

theorem value_of_a5 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h : a 3 * a 7 = 64) :
  a 5 = 8 ∨ a 5 = -8 :=
by
  sorry

end value_of_a5_l1250_125082


namespace local_min_4_l1250_125022

def seq (n : ℕ) : ℝ := n^3 - 48 * n + 5

theorem local_min_4 (m : ℕ) (h1 : seq (m-1) > seq m) (h2 : seq (m+1) > seq m) : m = 4 :=
sorry

end local_min_4_l1250_125022


namespace min_value_xy_l1250_125027

theorem min_value_xy (x y : ℝ) (h : x * y = 1) : x^2 + 4 * y^2 ≥ 4 := by
  sorry

end min_value_xy_l1250_125027


namespace gcd_78_182_l1250_125002

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := 
by
  sorry

end gcd_78_182_l1250_125002


namespace find_x_l1250_125001

  -- Definition of the vectors
  def a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
  def b : ℝ × ℝ := (2, 1)

  -- Condition that vectors are parallel
  def are_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

  -- Theorem statement
  theorem find_x (x : ℝ) (h : are_parallel (a x) b) : x = 5 :=
  sorry
  
end find_x_l1250_125001


namespace gcd_pow_diff_l1250_125045

theorem gcd_pow_diff :
  gcd (2 ^ 2100 - 1) (2 ^ 2091 - 1) = 511 := 
sorry

end gcd_pow_diff_l1250_125045


namespace calc_expr_solve_fractional_eq_l1250_125043

-- Problem 1: Calculate the expression
theorem calc_expr : (-2)^2 - (64:ℝ)^(1/3) + (-3)^0 - (1/3)^0 = 0 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

-- Problem 2: Solve the fractional equation
theorem solve_fractional_eq (x : ℝ) (h : x ≠ -1) : 
  (x / (x + 1) = 5 / (2 * x + 2) - 1) ↔ x = 3 / 4 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

end calc_expr_solve_fractional_eq_l1250_125043


namespace birth_year_l1250_125016

theorem birth_year (x : ℤ) (h : 1850 < x^2 - 10 - x ∧ 1849 ≤ x^2 - 10 - x ∧ x^2 - 10 - x ≤ 1880) : 
x^2 - 10 - x ≠ 1849 ∧ x^2 - 10 - x ≠ 1855 ∧ x^2 - 10 - x ≠ 1862 ∧ x^2 - 10 - x ≠ 1871 ∧ x^2 - 10 - x ≠ 1880 := 
sorry

end birth_year_l1250_125016


namespace range_of_a_l1250_125096

open Set Real

theorem range_of_a (a : ℝ) (α : ℝ → Prop) (β : ℝ → Prop) (hα : ∀ x, α x ↔ x ≥ a) (hβ : ∀ x, β x ↔ |x - 1| < 1)
  (h : ∀ x, (β x → α x) ∧ (∃ x, α x ∧ ¬β x)) : a ≤ 0 :=
by
  sorry

end range_of_a_l1250_125096


namespace number_of_green_pens_l1250_125084

theorem number_of_green_pens
  (black_pens : ℕ := 6)
  (red_pens : ℕ := 7)
  (green_pens : ℕ)
  (probability_black : (black_pens : ℚ) / (black_pens + red_pens + green_pens : ℚ) = 1 / 3) :
  green_pens = 5 := 
sorry

end number_of_green_pens_l1250_125084


namespace solve_for_x_l1250_125089

theorem solve_for_x (y : ℝ) (x : ℝ) 
  (h : x / (x - 1) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 3)) : 
  x = (y^2 + 3 * y - 2) / 2 := 
by 
  sorry

end solve_for_x_l1250_125089


namespace x_intercept_of_line_l1250_125017

def point1 := (10, 3)
def point2 := (-12, -8)

theorem x_intercept_of_line :
  let m := (point2.snd - point1.snd) / (point2.fst - point1.fst)
  let line_eq (x : ℝ) := m * (x - point1.fst) + point1.snd
  ∃ x : ℝ, line_eq x = 0 ∧ x = 4 :=
by
  sorry

end x_intercept_of_line_l1250_125017


namespace option_D_correct_l1250_125097

variable (x : ℝ)

theorem option_D_correct : (2 * x^7) / x = 2 * x^6 := sorry

end option_D_correct_l1250_125097


namespace custom_mul_of_two_and_neg_three_l1250_125004

-- Define the custom operation "*"
def custom.mul (a b : Int) : Int := a * b

-- The theorem to prove that 2 * (-3) using custom.mul equals -6
theorem custom_mul_of_two_and_neg_three : custom.mul 2 (-3) = -6 :=
by
  -- This is where the proof would go
  sorry

end custom_mul_of_two_and_neg_three_l1250_125004


namespace closest_perfect_square_to_350_l1250_125093

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l1250_125093


namespace weight_of_bowling_ball_l1250_125006

-- Define weights of bowling ball and canoe
variable (b c : ℚ)

-- Problem conditions
def cond1 : Prop := (9 * b = 5 * c)
def cond2 : Prop := (4 * c = 120)

-- The statement to prove
theorem weight_of_bowling_ball (h1 : cond1 b c) (h2 : cond2 c) : b = 50 / 3 := sorry

end weight_of_bowling_ball_l1250_125006


namespace smallest_value_among_options_l1250_125086

theorem smallest_value_among_options (x : ℕ) (h : x = 9) :
    min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min ((x+3)/8) ((x-3)/8)))) = (3/4) :=
by
  sorry

end smallest_value_among_options_l1250_125086


namespace cookies_per_person_l1250_125025

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℕ) (h1 : total_cookies = 35) (h2 : num_people = 5) :
  total_cookies / num_people = 7 := 
by {
  sorry
}

end cookies_per_person_l1250_125025


namespace total_cost_38_pencils_56_pens_l1250_125065

def numberOfPencils : ℕ := 38
def costPerPencil : ℝ := 2.50
def numberOfPens : ℕ := 56
def costPerPen : ℝ := 3.50
def totalCost := numberOfPencils * costPerPencil + numberOfPens * costPerPen

theorem total_cost_38_pencils_56_pens : totalCost = 291 := 
by
  -- leaving the proof as a placeholder
  sorry

end total_cost_38_pencils_56_pens_l1250_125065


namespace area_of_grey_region_l1250_125085

open Nat

theorem area_of_grey_region
  (a1 a2 b : ℕ)
  (h1 : a1 = 8 * 10)
  (h2 : a2 = 9 * 12)
  (hb : b = 37)
  : (a2 - (a1 - b) = 65) := by
  sorry

end area_of_grey_region_l1250_125085


namespace tens_digit_of_9_to_2023_l1250_125062

theorem tens_digit_of_9_to_2023 :
  (9^2023 % 100) / 10 % 10 = 8 :=
sorry

end tens_digit_of_9_to_2023_l1250_125062


namespace sum_of_terms_l1250_125030

-- Defining the arithmetic progression
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

-- Given conditions
theorem sum_of_terms (a d : ℕ) (h : (a + 3 * d) + (a + 11 * d) = 20) :
  12 * (a + 11 * d) / 2 = 60 :=
by
  sorry

end sum_of_terms_l1250_125030


namespace weight_of_3_moles_HClO2_correct_l1250_125058

def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.453
def atomic_weight_O : ℝ := 15.999

def molecular_weight_HClO2 : ℝ := (1 * atomic_weight_H) + (1 * atomic_weight_Cl) + (2 * atomic_weight_O)
def weight_of_3_moles_HClO2 : ℝ := 3 * molecular_weight_HClO2

theorem weight_of_3_moles_HClO2_correct : weight_of_3_moles_HClO2 = 205.377 := by
  sorry

end weight_of_3_moles_HClO2_correct_l1250_125058


namespace find_repair_charge_l1250_125069

theorem find_repair_charge
    (cost_oil_change : ℕ)
    (cost_car_wash : ℕ)
    (num_oil_changes : ℕ)
    (num_repairs : ℕ)
    (num_car_washes : ℕ)
    (total_earnings : ℕ)
    (R : ℕ) :
    (cost_oil_change = 20) →
    (cost_car_wash = 5) →
    (num_oil_changes = 5) →
    (num_repairs = 10) →
    (num_car_washes = 15) →
    (total_earnings = 475) →
    5 * cost_oil_change + 10 * R + 15 * cost_car_wash = total_earnings →
    R = 30 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end find_repair_charge_l1250_125069


namespace ellipse_focus_distance_l1250_125031

theorem ellipse_focus_distance (m : ℝ) (a b c : ℝ)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m + y^2 / 16 = 1)
  (focus_distance : ∀ P : ℝ × ℝ, ∃ F1 F2 : ℝ × ℝ, dist P F1 = 3 ∧ dist P F2 = 7) :
  m = 25 := 
  sorry

end ellipse_focus_distance_l1250_125031


namespace count_even_numbers_is_320_l1250_125068

noncomputable def count_even_numbers_with_distinct_digits : Nat := 
  let unit_choices := 5  -- Choices for the unit digit (0, 2, 4, 6, 8)
  let hundreds_choices := 8  -- Choices for the hundreds digit (1 to 9, excluding the unit digit)
  let tens_choices := 8  -- Choices for the tens digit (0 to 9, excluding the hundreds and unit digit)
  unit_choices * hundreds_choices * tens_choices

theorem count_even_numbers_is_320 : count_even_numbers_with_distinct_digits = 320 := by
  sorry

end count_even_numbers_is_320_l1250_125068


namespace arithmetic_sum_problem_l1250_125049

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sum_problem
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms S a)
  (h_S10 : S 10 = 4) :
  a 3 + a 8 = 4 / 5 := 
sorry

end arithmetic_sum_problem_l1250_125049


namespace trig_identity_l1250_125015

open Real

theorem trig_identity (theta : ℝ) (h : tan theta = 2) : 
  (sin (π / 2 + theta) - cos (π - theta)) / (sin (π / 2 - theta) - sin (π - theta)) = -2 :=
by
  sorry

end trig_identity_l1250_125015


namespace find_missing_number_l1250_125052

theorem find_missing_number (n x : ℕ) (h : n * (n + 1) / 2 - x = 2012) : x = 4 := by
  sorry

end find_missing_number_l1250_125052


namespace pq_iff_cond_l1250_125033

def p (a : ℝ) := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem pq_iff_cond (a : ℝ) : (p a ∧ q a) ↔ (a ≤ -2 ∨ a = 1) := 
by
  sorry

end pq_iff_cond_l1250_125033


namespace moles_of_water_formed_l1250_125077

-- Defining the relevant constants
def NH4Cl_moles : ℕ := sorry  -- Some moles of Ammonium chloride (NH4Cl)
def NaOH_moles : ℕ := 3       -- 3 moles of Sodium hydroxide (NaOH)
def H2O_moles : ℕ := 3        -- The total moles of Water (H2O) formed

-- Statement of the problem
theorem moles_of_water_formed :
  NH4Cl_moles ≥ NaOH_moles → H2O_moles = 3 :=
sorry

end moles_of_water_formed_l1250_125077


namespace max_pairs_correct_l1250_125046

def max_pairs (n : ℕ) : ℕ :=
  if h : n > 1 then (n * n) / 4 else 0

theorem max_pairs_correct (n : ℕ) (h : n ≥ 2) :
  (max_pairs n = (n * n) / 4) :=
by sorry

end max_pairs_correct_l1250_125046


namespace neg_p_l1250_125026

theorem neg_p :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end neg_p_l1250_125026


namespace james_nickels_count_l1250_125019

-- Definitions
def total_cents : ℕ := 685
def more_nickels_than_quarters := 11

-- Variables representing the number of nickels and quarters
variables (n q : ℕ)

-- Conditions
axiom h1 : 5 * n + 25 * q = total_cents
axiom h2 : n = q + more_nickels_than_quarters

-- Theorem stating the number of nickels
theorem james_nickels_count : n = 32 := 
by
  -- Proof will go here, marked as "sorry" to complete the statement
  sorry

end james_nickels_count_l1250_125019


namespace incorrect_conclusion_intersection_l1250_125010

theorem incorrect_conclusion_intersection :
  ∀ (x : ℝ), (0 = -2 * x + 4) → (x = 2) :=
by
  intro x h
  sorry

end incorrect_conclusion_intersection_l1250_125010


namespace construct_trihedral_angle_l1250_125087

-- Define the magnitudes of dihedral angles
variables (α β γ : ℝ)

-- Problem statement
theorem construct_trihedral_angle (h₀ : 0 < α) (h₁ : 0 < β) (h₂ : 0 < γ) :
  ∃ (trihedral_angle : Type), true := 
sorry

end construct_trihedral_angle_l1250_125087


namespace books_read_in_8_hours_l1250_125036

def reading_speed := 100 -- pages per hour
def book_pages := 400 -- pages per book
def hours_available := 8 -- hours

theorem books_read_in_8_hours :
  (hours_available * reading_speed) / book_pages = 2 :=
by
  sorry

end books_read_in_8_hours_l1250_125036


namespace dan_total_marbles_l1250_125021

theorem dan_total_marbles (violet_marbles : ℕ) (red_marbles : ℕ) (h₁ : violet_marbles = 64) (h₂ : red_marbles = 14) : violet_marbles + red_marbles = 78 :=
sorry

end dan_total_marbles_l1250_125021


namespace orangeade_price_second_day_l1250_125076

theorem orangeade_price_second_day :
  ∀ (X O : ℝ), (2 * X * 0.60 = 3 * X * E) → (E = 2 * 0.60 / 3) →
  E = 0.40 := by
  intros X O h₁ h₂
  sorry

end orangeade_price_second_day_l1250_125076


namespace total_cost_correct_l1250_125094

-- Defining the conditions
def charges_per_week : ℕ := 3
def weeks_per_year : ℕ := 52
def cost_per_charge : ℝ := 0.78

-- Defining the total cost proof statement
theorem total_cost_correct : (charges_per_week * weeks_per_year : ℝ) * cost_per_charge = 121.68 :=
by
  sorry

end total_cost_correct_l1250_125094


namespace problem1_problem2_l1250_125042

-- Statement for Question (1)
theorem problem1 (x : ℝ) (h : |x - 1| + x ≥ x + 2) : x ≤ -1 ∨ x ≥ 3 :=
  sorry

-- Statement for Question (2)
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + x ≤ 3 * x → x ≥ 2) : a = 6 :=
  sorry

end problem1_problem2_l1250_125042


namespace probability_of_collinear_dots_in_5x5_grid_l1250_125011

def collinear_dots_probability (total_dots chosen_dots collinear_sets : ℕ) : ℚ :=
  (collinear_sets : ℚ) / (Nat.choose total_dots chosen_dots)

theorem probability_of_collinear_dots_in_5x5_grid :
  collinear_dots_probability 25 4 12 = 12 / 12650 := by
  sorry

end probability_of_collinear_dots_in_5x5_grid_l1250_125011


namespace problem_statement_l1250_125048

noncomputable def g : ℝ → ℝ := sorry

theorem problem_statement 
  (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - x * y^2 - x + 2) :
  ∃ (m t : ℕ), (m = 1) ∧ (t = 3) ∧ (m * t = 3) :=
sorry

end problem_statement_l1250_125048


namespace range_of_a_l1250_125074

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x)^2 + a * Real.cos x + a

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 1) → a ≤ 0 :=
by
  sorry

end range_of_a_l1250_125074


namespace carpet_needed_correct_l1250_125038

def length_room : ℕ := 15
def width_room : ℕ := 9
def length_closet : ℕ := 3
def width_closet : ℕ := 2

def area_room : ℕ := length_room * width_room
def area_closet : ℕ := length_closet * width_closet
def area_to_carpet : ℕ := area_room - area_closet
def sq_ft_to_sq_yd (sqft: ℕ) : ℕ := (sqft + 8) / 9  -- Adding 8 to ensure proper rounding up

def carpet_needed : ℕ := sq_ft_to_sq_yd area_to_carpet

theorem carpet_needed_correct :
  carpet_needed = 15 := by
  sorry

end carpet_needed_correct_l1250_125038


namespace greatest_x_for_lcm_l1250_125063

theorem greatest_x_for_lcm (x : ℕ) (h_lcm : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 :=
by
  sorry

end greatest_x_for_lcm_l1250_125063


namespace bananas_in_each_bunch_l1250_125012

theorem bananas_in_each_bunch (x: ℕ) : (6 * x + 5 * 7 = 83) → x = 8 :=
by
  intro h
  sorry

end bananas_in_each_bunch_l1250_125012


namespace probability_at_least_one_boy_and_one_girl_l1250_125000

noncomputable def mathematics_club_prob : ℚ :=
  let boys := 14
  let girls := 10
  let total_members := 24
  let total_committees := Nat.choose total_members 5
  let boys_committees := Nat.choose boys 5
  let girls_committees := Nat.choose girls 5
  let committees_with_at_least_one_boy_and_one_girl := total_committees - (boys_committees + girls_committees)
  let probability := (committees_with_at_least_one_boy_and_one_girl : ℚ) / (total_committees : ℚ)
  probability

theorem probability_at_least_one_boy_and_one_girl :
  mathematics_club_prob = (4025 : ℚ) / 4251 :=
by
  sorry

end probability_at_least_one_boy_and_one_girl_l1250_125000


namespace sum_first_eight_terms_geometric_sequence_l1250_125039

noncomputable def sum_of_geometric_sequence (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_eight_terms_geometric_sequence :
  sum_of_geometric_sequence (1/2) (1/3) 8 = 9840 / 6561 :=
by
  sorry

end sum_first_eight_terms_geometric_sequence_l1250_125039


namespace fraction_white_tulips_l1250_125018

theorem fraction_white_tulips : 
  ∀ (total_tulips yellow_fraction red_fraction pink_fraction white_fraction : ℝ),
  total_tulips = 60 →
  yellow_fraction = 1 / 2 →
  red_fraction = 1 / 3 →
  pink_fraction = 1 / 4 →
  white_fraction = 
    ((total_tulips * (1 - yellow_fraction)) * (1 - red_fraction) * (1 - pink_fraction)) / total_tulips →
  white_fraction = 1 / 4 :=
by
  intros total_tulips yellow_fraction red_fraction pink_fraction white_fraction 
    h_total h_yellow h_red h_pink h_white
  sorry

end fraction_white_tulips_l1250_125018


namespace jail_time_ratio_l1250_125050

def arrests (days : ℕ) (cities : ℕ) (arrests_per_day : ℕ) : ℕ := days * cities * arrests_per_day
def jail_days_before_trial (total_arrests : ℕ) (days_before_trial : ℕ) : ℕ := total_arrests * days_before_trial
def weeks_from_days (days : ℕ) : ℕ := days / 7
def time_after_trial (total_jail_time_weeks : ℕ) (weeks_before_trial : ℕ) : ℕ := total_jail_time_weeks - weeks_before_trial
def total_possible_jail_time (total_arrests : ℕ) (sentence_weeks : ℕ) : ℕ := total_arrests * sentence_weeks
def ratio (after_trial_weeks : ℕ) (total_possible_weeks : ℕ) : ℚ := after_trial_weeks / total_possible_weeks

theorem jail_time_ratio 
    (days : ℕ := 30) 
    (cities : ℕ := 21)
    (arrests_per_day : ℕ := 10)
    (days_before_trial : ℕ := 4)
    (total_jail_time_weeks : ℕ := 9900)
    (sentence_weeks : ℕ := 2) :
    ratio 
      (time_after_trial 
        total_jail_time_weeks 
        (weeks_from_days 
          (jail_days_before_trial 
            (arrests days cities arrests_per_day) 
            days_before_trial))) 
      (total_possible_jail_time 
        (arrests days cities arrests_per_day) 
        sentence_weeks) = 1/2 := 
by
  -- We leave the proof as an exercise
  sorry

end jail_time_ratio_l1250_125050


namespace range_of_m_l1250_125007

theorem range_of_m (x y m : ℝ) 
  (h1 : x - 2 * y = 1) 
  (h2 : 2 * x + y = 4 * m) 
  (h3 : x + 3 * y < 6) : 
  m < 7 / 4 := 
sorry

end range_of_m_l1250_125007


namespace smallest_integer_solution_l1250_125071

theorem smallest_integer_solution (x : ℤ) (h : 10 - 5 * x < -18) : x = 6 :=
sorry

end smallest_integer_solution_l1250_125071


namespace comprehensive_survey_is_C_l1250_125088

def option (label : String) (description : String) := (label, description)

def A := option "A" "Investigating the current mental health status of middle school students nationwide"
def B := option "B" "Investigating the compliance of food in our city"
def C := option "C" "Investigating the physical and mental conditions of classmates in the class"
def D := option "D" "Investigating the viewership ratings of Nanjing TV's 'Today's Life'"

theorem comprehensive_survey_is_C (suitable: (String × String → Prop)) :
  suitable C :=
sorry

end comprehensive_survey_is_C_l1250_125088


namespace whole_numbers_between_sqrt_18_and_sqrt_98_l1250_125035

theorem whole_numbers_between_sqrt_18_and_sqrt_98 :
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  (largest_whole_num - smallest_whole_num + 1) = 5 :=
by
  -- Introduce variables
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  -- Sorry indicates the proof steps are skipped
  sorry

end whole_numbers_between_sqrt_18_and_sqrt_98_l1250_125035


namespace segment_ratio_ae_ad_l1250_125041

/-- Given points B, C, and E lie on line segment AD, and the following conditions:
  1. The length of segment AB is twice the length of segment BD.
  2. The length of segment AC is 5 times the length of segment CD.
  3. The length of segment BE is one-third the length of segment EC.
Prove that the fraction of the length of segment AD that segment AE represents is 17/24. -/
theorem segment_ratio_ae_ad (AB BD AC CD BE EC AD AE : ℝ)
    (h1 : AB = 2 * BD)
    (h2 : AC = 5 * CD)
    (h3 : BE = (1/3) * EC)
    (h4 : AD = 6 * CD)
    (h5 : AE = 4.25 * CD) :
    AE / AD = 17 / 24 := 
  by 
  sorry

end segment_ratio_ae_ad_l1250_125041


namespace arithmetic_progression_y_value_l1250_125051

theorem arithmetic_progression_y_value (x y : ℚ) 
  (h1 : x = 2)
  (h2 : 2 * y - x = (y + x + 3) - (2 * y - x))
  (h3 : (3 * y + x) - (y + x + 3) = (y + x + 3) - (2 * y - x)) : 
  y = 10 / 3 :=
by
  sorry

end arithmetic_progression_y_value_l1250_125051


namespace interest_rate_same_l1250_125083

theorem interest_rate_same (initial_amount: ℝ) (interest_earned: ℝ) 
  (time_period1: ℝ) (time_period2: ℝ) (principal: ℝ) (initial_rate: ℝ) : 
  initial_amount * initial_rate * time_period2 = interest_earned * 100 ↔ initial_rate = 12 
  :=
by
  sorry

end interest_rate_same_l1250_125083


namespace students_not_receiving_A_l1250_125028

theorem students_not_receiving_A (total_students : ℕ) (students_A_physics : ℕ) (students_A_chemistry : ℕ) (students_A_both : ℕ) (h_total : total_students = 40) (h_A_physics : students_A_physics = 10) (h_A_chemistry : students_A_chemistry = 18) (h_A_both : students_A_both = 6) : (total_students - ((students_A_physics + students_A_chemistry) - students_A_both)) = 18 := 
by
  sorry

end students_not_receiving_A_l1250_125028


namespace range_of_x_l1250_125070

def p (x : ℝ) : Prop := x^2 - 5 * x + 6 ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem range_of_x (x : ℝ) (hpq : p x ∨ q x) (hnq : ¬ q x) : x ≤ 0 ∨ x ≥ 4 :=
by sorry

end range_of_x_l1250_125070


namespace age_of_B_l1250_125064

theorem age_of_B (a b c d : ℕ) 
  (h1: a + b + c + d = 112)
  (h2: a + c = 58)
  (h3: 2 * b + 3 * d = 135)
  (h4: b + d = 54) :
  b = 27 :=
by
  sorry

end age_of_B_l1250_125064


namespace min_value_expression_l1250_125055

theorem min_value_expression (y : ℝ) (hy : y > 0) : 9 * y + 1 / y^6 ≥ 10 :=
by
  sorry

end min_value_expression_l1250_125055


namespace percentage_increase_l1250_125059

variable (P N N' : ℝ)
variable (h : P * 0.90 * N' = P * N * 1.035)

theorem percentage_increase :
  ((N' - N) / N) * 100 = 15 :=
by
  -- By given condition, we have the equation:
  -- P * 0.90 * N' = P * N * 1.035
  sorry

end percentage_increase_l1250_125059


namespace new_rectangle_perimeters_l1250_125066

theorem new_rectangle_perimeters {l w : ℕ} (h_l : l = 4) (h_w : w = 2) :
  (∃ P, P = 2 * (8 + 2) ∨ P = 2 * (4 + 4)) ∧ (P = 20 ∨ P = 16) :=
by
  sorry

end new_rectangle_perimeters_l1250_125066


namespace ellipse_equation_l1250_125009

theorem ellipse_equation (b : Real) (c : Real)
  (h₁ : 0 < b ∧ b < 5) 
  (h₂ : 25 - b^2 = c^2)
  (h₃ : 5 + c = 2 * b) :
  ∃ (b : Real), (b^2 = 16) ∧ (∀ x y : Real, (x^2 / 25 + y^2 / b^2 = 1 ↔ x^2 / 25 + y^2 / 16 = 1)) := 
sorry

end ellipse_equation_l1250_125009


namespace melanie_total_plums_l1250_125003

-- Define the initial conditions
def melaniePlums : Float := 7.0
def samGavePlums : Float := 3.0

-- State the theorem to prove
theorem melanie_total_plums : melaniePlums + samGavePlums = 10.0 := 
by
  sorry

end melanie_total_plums_l1250_125003


namespace balls_into_boxes_all_ways_balls_into_boxes_one_empty_l1250_125078

/-- There are 4 different balls and 4 different boxes. -/
def balls : ℕ := 4
def boxes : ℕ := 4

/-- The number of ways to put 4 different balls into 4 different boxes is 256. -/
theorem balls_into_boxes_all_ways : (balls ^ boxes) = 256 := by
  sorry

/-- The number of ways to put 4 different balls into 4 different boxes such that exactly one box remains empty is 144. -/
theorem balls_into_boxes_one_empty : (boxes.choose 1 * (balls ^ (boxes - 1))) = 144 := by
  sorry

end balls_into_boxes_all_ways_balls_into_boxes_one_empty_l1250_125078


namespace percentage_exceed_l1250_125098

theorem percentage_exceed (x y : ℝ) (h : y = x + 0.2 * x) :
  (y - x) / x * 100 = 20 :=
by
  -- Proof goes here
  sorry

end percentage_exceed_l1250_125098


namespace solve_inequality_l1250_125091

def inequality_solution (x : ℝ) : Prop := |2 * x - 1| - x ≥ 2 

theorem solve_inequality (x : ℝ) : 
  inequality_solution x ↔ (x ≥ 3 ∨ x ≤ -1/3) :=
by sorry

end solve_inequality_l1250_125091


namespace find_a_l1250_125092

theorem find_a (r s a : ℚ) (h1 : s^2 = 16) (h2 : 2 * r * s = 15) (h3 : a = r^2) : a = 225/64 := by
  sorry

end find_a_l1250_125092


namespace sum_of_angles_FC_correct_l1250_125079

noncomputable def circleGeometry (A B C D E F : Point)
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E)
  (arcAB : ℝ) (arcDE : ℝ) : Prop :=
  let arcFull := 360;
  let angleF := 6;  -- Derived from the intersecting chords theorem
  let angleC := 36; -- Derived from the inscribed angle theorem
  arcAB = 60 ∧ arcDE = 72 ∧
  0 ≤ angleF ∧ 0 ≤ angleC ∧
  angleF + angleC = 42

theorem sum_of_angles_FC_correct (A B C D E F : Point) 
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) :
  circleGeometry A B C D E F onCircle 60 72 :=
by
  sorry  -- Proof to be filled

end sum_of_angles_FC_correct_l1250_125079


namespace remainder_5_pow_2048_mod_17_l1250_125029

theorem remainder_5_pow_2048_mod_17 : (5 ^ 2048) % 17 = 0 :=
by
  sorry

end remainder_5_pow_2048_mod_17_l1250_125029


namespace max_d_6_digit_multiple_33_l1250_125067

theorem max_d_6_digit_multiple_33 (x d e : ℕ) 
  (hx : 1 ≤ x ∧ x ≤ 9) 
  (hd : 0 ≤ d ∧ d ≤ 9) 
  (he : 0 ≤ e ∧ e ≤ 9)
  (h1 : (x * 100000 + 50000 + d * 1000 + 300 + 30 + e) ≥ 100000) 
  (h2 : (x + d + e + 11) % 3 = 0)
  (h3 : ((x + d - e - 5 + 11) % 11 = 0)) :
  d = 9 := 
sorry

end max_d_6_digit_multiple_33_l1250_125067


namespace determine_x_squared_plus_y_squared_l1250_125057

theorem determine_x_squared_plus_y_squared (x y : ℝ) 
(h : (x^2 + y^2 + 2) * (x^2 + y^2 - 3) = 6) : x^2 + y^2 = 4 :=
sorry

end determine_x_squared_plus_y_squared_l1250_125057


namespace max_trees_l1250_125095

theorem max_trees (interval distance road_length number_of_intervals add_one : ℕ) 
  (h_interval: interval = 4) 
  (h_distance: distance = 28) 
  (h_intervals: number_of_intervals = distance / interval)
  (h_add: add_one = number_of_intervals + 1) :
  add_one = 8 :=
sorry

end max_trees_l1250_125095


namespace michael_has_16_blocks_l1250_125032

-- Define the conditions
def number_of_boxes : ℕ := 8
def blocks_per_box : ℕ := 2

-- Define the expected total number of blocks
def total_blocks : ℕ := 16

-- State the theorem
theorem michael_has_16_blocks (n_boxes blocks_per_b : ℕ) :
  n_boxes = number_of_boxes → 
  blocks_per_b = blocks_per_box → 
  n_boxes * blocks_per_b = total_blocks :=
by intros h1 h2; rw [h1, h2]; sorry

end michael_has_16_blocks_l1250_125032


namespace min_distance_sum_coordinates_l1250_125073

theorem min_distance_sum_coordinates (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, -1)) :
  ∃ P : ℝ × ℝ, P = (0, 3) ∧ ∀ Q : ℝ × ℝ, Q.1 = 0 → |A.1 - Q.1| + |A.2 - Q.2| + |B.1 - Q.1| + |B.2 - Q.2| ≥ |A.1 - (0 : ℝ)| + |A.2 - (3 : ℝ)| + |B.1 - (0 : ℝ)| + |B.2 - (3 : ℝ)| := 
sorry

end min_distance_sum_coordinates_l1250_125073


namespace container_capacity_l1250_125072

-- Definitions based on the conditions
def tablespoons_per_cup := 3
def ounces_per_cup := 8
def tablespoons_added := 15

-- Problem statement
theorem container_capacity : 
  (tablespoons_added / tablespoons_per_cup) * ounces_per_cup = 40 :=
  sorry

end container_capacity_l1250_125072


namespace proof_F_4_f_5_l1250_125008

def f (a : ℤ) : ℤ := a - 2

def F (a b : ℤ) : ℤ := a * b + b^2

theorem proof_F_4_f_5 :
  F 4 (f 5) = 21 := by
  sorry

end proof_F_4_f_5_l1250_125008


namespace determine_m_for_divisibility_by_11_l1250_125014

def is_divisible_by_11 (n : ℤ) : Prop :=
  n % 11 = 0

def sum_digits_odd_pos : ℤ :=
  8 + 6 + 2 + 8

def sum_digits_even_pos (m : ℤ) : ℤ :=
  5 + m + 4

theorem determine_m_for_divisibility_by_11 :
  ∃ m : ℤ, is_divisible_by_11 (sum_digits_odd_pos - sum_digits_even_pos m) ∧ m = 4 := 
by
  sorry

end determine_m_for_divisibility_by_11_l1250_125014


namespace largest_possible_s_l1250_125040

theorem largest_possible_s :
  ∃ s r : ℕ, (r ≥ s) ∧ (s ≥ 5) ∧ (122 * r - 120 * s = r * s) ∧ (s = 121) :=
by sorry

end largest_possible_s_l1250_125040


namespace geometric_series_sum_l1250_125005

theorem geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 4
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 16383 / 49152 :=
by
  sorry

end geometric_series_sum_l1250_125005


namespace division_of_decimals_l1250_125054

theorem division_of_decimals : 0.36 / 0.004 = 90 := by
  sorry

end division_of_decimals_l1250_125054


namespace simplify_expression_l1250_125053

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (3 * x^2 * x^3) = 29 * x^5 := 
  sorry

end simplify_expression_l1250_125053


namespace pair_d_same_function_l1250_125060

theorem pair_d_same_function : ∀ x : ℝ, x = (x ^ 5) ^ (1 / 5) := 
by
  intro x
  sorry

end pair_d_same_function_l1250_125060


namespace kona_distance_proof_l1250_125023

-- Defining the distances as constants
def distance_to_bakery : ℕ := 9
def distance_from_grandmother_to_home : ℕ := 27
def additional_trip_distance : ℕ := 6

-- Defining the variable for the distance from bakery to grandmother's house
def x : ℕ := 30

-- Main theorem to prove the distance
theorem kona_distance_proof :
  distance_to_bakery + x + distance_from_grandmother_to_home = 2 * x + additional_trip_distance :=
by
  sorry

end kona_distance_proof_l1250_125023


namespace election_debate_conditions_l1250_125034

theorem election_debate_conditions (n : ℕ) (h_n : n ≥ 3) :
  ¬ ∃ (p : ℕ), n = 2 * (2 ^ p - 2) + 1 :=
sorry

end election_debate_conditions_l1250_125034


namespace cos_value_l1250_125090

theorem cos_value (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (2 * π / 3 - α) = 1 / 3 :=
by
  sorry

end cos_value_l1250_125090


namespace find_initial_maple_trees_l1250_125013

def initial_maple_trees (final_maple_trees planted_maple_trees : ℕ) : ℕ :=
  final_maple_trees - planted_maple_trees

theorem find_initial_maple_trees : initial_maple_trees 11 9 = 2 := by
  sorry

end find_initial_maple_trees_l1250_125013


namespace find_positive_number_l1250_125047

theorem find_positive_number 
  (x : ℝ) (h_pos : x > 0) 
  (h_eq : (2 / 3) * x = (16 / 216) * (1 / x)) : 
  x = 1 / 3 :=
by
  -- This is indicating that we're skipping the actual proof steps
  sorry

end find_positive_number_l1250_125047


namespace StockPriceAdjustment_l1250_125075

theorem StockPriceAdjustment (P₀ P₁ P₂ P₃ P₄ : ℝ) (january_increase february_decrease march_increase : ℝ) :
  P₀ = 150 →
  january_increase = 0.10 →
  february_decrease = 0.15 →
  march_increase = 0.30 →
  P₁ = P₀ * (1 + january_increase) →
  P₂ = P₁ * (1 - february_decrease) →
  P₃ = P₂ * (1 + march_increase) →
  142.5 <= P₃ * (1 - 0.17) ∧ P₃ * (1 - 0.17) <= 157.5 :=
by
  intros hP₀ hJanuaryIncrease hFebruaryDecrease hMarchIncrease hP₁ hP₂ hP₃
  sorry

end StockPriceAdjustment_l1250_125075
