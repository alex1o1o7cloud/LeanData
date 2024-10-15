import Mathlib

namespace NUMINAMATH_GPT_points_on_same_side_after_25_seconds_l835_83511

def movement_time (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) : ℕ :=
  25

theorem points_on_same_side_after_25_seconds (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) :
  side_length = 100 ∧ perimeter = 400 ∧ speed_A = 5 ∧ speed_B = 10 ∧ start_mid_B = 50 →
  movement_time side_length perimeter speed_A speed_B start_mid_B = 25 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_points_on_same_side_after_25_seconds_l835_83511


namespace NUMINAMATH_GPT_range_of_a_l835_83564

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a ≥ 0) → (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l835_83564


namespace NUMINAMATH_GPT_cost_of_soccer_basketball_balls_max_basketballs_l835_83512

def cost_of_balls (x y : ℕ) : Prop :=
  (7 * x = 5 * y) ∧ (40 * x + 20 * y = 3400)

def cost_constraint (x y m : ℕ) : Prop :=
  (x = 50) ∧ (y = 70) ∧ (70 * m + 50 * (100 - m) ≤ 6300)

theorem cost_of_soccer_basketball_balls (x y : ℕ) (h : cost_of_balls x y) : x = 50 ∧ y = 70 :=
  by sorry

theorem max_basketballs (x y m : ℕ) (h : cost_constraint x y m) : m ≤ 65 :=
  by sorry

end NUMINAMATH_GPT_cost_of_soccer_basketball_balls_max_basketballs_l835_83512


namespace NUMINAMATH_GPT_jack_cleaning_time_is_one_hour_l835_83506

def jackGrove : ℕ × ℕ := (4, 5)
def timeToCleanEachTree : ℕ := 6
def timeReductionFactor : ℕ := 2
def totalCleaningTimeWithHelpMin : ℕ :=
  (jackGrove.fst * jackGrove.snd) * (timeToCleanEachTree / timeReductionFactor)
def totalCleaningTimeWithHelpHours : ℕ :=
  totalCleaningTimeWithHelpMin / 60

theorem jack_cleaning_time_is_one_hour :
  totalCleaningTimeWithHelpHours = 1 := by
  sorry

end NUMINAMATH_GPT_jack_cleaning_time_is_one_hour_l835_83506


namespace NUMINAMATH_GPT_smallest_d_for_range_of_g_l835_83539

theorem smallest_d_for_range_of_g :
  ∃ d, (∀ x : ℝ, x^2 + 4 * x + d = 3) → d = 7 := by
  sorry

end NUMINAMATH_GPT_smallest_d_for_range_of_g_l835_83539


namespace NUMINAMATH_GPT_cost_price_perc_of_selling_price_l835_83534

theorem cost_price_perc_of_selling_price
  (SP : ℝ) (CP : ℝ) (P : ℝ)
  (h1 : P = SP - CP)
  (h2 : P = (4.166666666666666 / 100) * SP) :
  CP = SP * 0.9583333333333334 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_perc_of_selling_price_l835_83534


namespace NUMINAMATH_GPT_money_raised_is_correct_l835_83544

noncomputable def cost_per_dozen : ℚ := 2.40
noncomputable def selling_price_per_donut : ℚ := 1
noncomputable def dozens : ℕ := 10

theorem money_raised_is_correct :
  (dozens * 12 * selling_price_per_donut - dozens * cost_per_dozen) = 96 := by
sorry

end NUMINAMATH_GPT_money_raised_is_correct_l835_83544


namespace NUMINAMATH_GPT_combine_octahedrons_tetrahedrons_to_larger_octahedron_l835_83504

theorem combine_octahedrons_tetrahedrons_to_larger_octahedron (edge : ℝ) :
  ∃ (octahedrons : ℕ) (tetrahedrons : ℕ),
    octahedrons = 6 ∧ tetrahedrons = 8 ∧
    (∃ (new_octahedron_edge : ℝ), new_octahedron_edge = 2 * edge) :=
by {
  -- The proof will construct the larger octahedron
  sorry
}

end NUMINAMATH_GPT_combine_octahedrons_tetrahedrons_to_larger_octahedron_l835_83504


namespace NUMINAMATH_GPT_determine_parallel_planes_l835_83502

def Plane : Type := sorry
def Line : Type := sorry
def Parallel (x y : Line) : Prop := sorry
def Skew (x y : Line) : Prop := sorry
def PlaneParallel (α β : Plane) : Prop := sorry

variables (α β : Plane) (a b : Line)
variable (hSkew : Skew a b)
variable (hαa : Parallel a α) 
variable (hαb : Parallel b α)
variable (hβa : Parallel a β)
variable (hβb : Parallel b β)

theorem determine_parallel_planes : PlaneParallel α β := sorry

end NUMINAMATH_GPT_determine_parallel_planes_l835_83502


namespace NUMINAMATH_GPT_total_skips_l835_83538

-- Definitions of the given conditions
def BobsSkipsPerRock := 12
def JimsSkipsPerRock := 15
def NumberOfRocks := 10

-- Statement of the theorem to be proved
theorem total_skips :
  (BobsSkipsPerRock * NumberOfRocks) + (JimsSkipsPerRock * NumberOfRocks) = 270 :=
by
  sorry

end NUMINAMATH_GPT_total_skips_l835_83538


namespace NUMINAMATH_GPT_Series_value_l835_83533

theorem Series_value :
  (∑' n : ℕ, (2^n) / (7^(2^n) + 1)) = 1 / 6 :=
sorry

end NUMINAMATH_GPT_Series_value_l835_83533


namespace NUMINAMATH_GPT_book_arrangements_l835_83552

theorem book_arrangements (n : ℕ) (b1 b2 b3 b4 b5 : ℕ) (h_b123 : b1 < b2 ∧ b2 < b3):
  n = 20 := sorry

end NUMINAMATH_GPT_book_arrangements_l835_83552


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_roots_l835_83535

theorem quadratic_has_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 2*x₁ + k = 0) ∧ (x₂^2 - 2*x₂ + k = 0))
  ↔ k < 1 :=
by sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_roots_l835_83535


namespace NUMINAMATH_GPT_part1_part2_l835_83562

noncomputable def f (a x : ℝ) : ℝ := (a / x) - Real.log x

theorem part1 (a : ℝ) (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2) 
(hf : f a x1 = -3) (hf2 : f a x2 = -3) : a ∈ Set.Ioo (-Real.exp 2) 0 :=
sorry

theorem part2 (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2)
(hfa : f (-2) x1 = -3) (hfb : f (-2) x2 = -3) : x1 + x2 > 4 :=
sorry

end NUMINAMATH_GPT_part1_part2_l835_83562


namespace NUMINAMATH_GPT_different_answers_due_to_different_cuts_l835_83549

noncomputable def problem_89914 (bub : Type) (cut : bub → (bub × bub)) (is_log_cut : bub → Prop) (is_halved_log : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_log_cut b) → is_halved_log (cut b)

noncomputable def problem_89915 (bub : Type) (cut : bub → (bub × bub)) (is_sector_cut : bub → Prop) (is_sectors : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_sector_cut b) → is_sectors (cut b)

theorem different_answers_due_to_different_cuts
  (bub : Type)
  (cut : bub → (bub × bub))
  (is_log_cut : bub → Prop)
  (is_halved_log : bub × bub → Prop)
  (is_sector_cut : bub → Prop)
  (is_sectors : bub × bub → Prop) :
  problem_89914 bub cut is_log_cut is_halved_log ∧ problem_89915 bub cut is_sector_cut is_sectors →
  ∃ b : bub, (is_log_cut b ∧ ¬ is_sector_cut b) ∨ (¬ is_log_cut b ∧ is_sector_cut b) := sorry

end NUMINAMATH_GPT_different_answers_due_to_different_cuts_l835_83549


namespace NUMINAMATH_GPT_john_made_money_l835_83510

theorem john_made_money 
  (repair_cost : ℕ := 20000) 
  (discount_percentage : ℕ := 20) 
  (prize_money : ℕ := 70000) 
  (keep_percentage : ℕ := 90) : 
  (prize_money * keep_percentage / 100) - (repair_cost - (repair_cost * discount_percentage / 100)) = 47000 := 
by 
  sorry

end NUMINAMATH_GPT_john_made_money_l835_83510


namespace NUMINAMATH_GPT_find_number_l835_83543

theorem find_number (n : ℝ) (h : (1/2) * n + 5 = 11) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l835_83543


namespace NUMINAMATH_GPT_coin_probability_l835_83550

theorem coin_probability (p : ℝ) (h1 : p < 1/2) (h2 : (Nat.choose 6 3) * p^3 * (1-p)^3 = 1/20) : p = 1/400 := sorry

end NUMINAMATH_GPT_coin_probability_l835_83550


namespace NUMINAMATH_GPT_matrix_mult_7_l835_83573

theorem matrix_mult_7 (M : Matrix (Fin 3) (Fin 3) ℝ) (v : Fin 3 → ℝ) : 
  (∀ v, M.mulVec v = (7 : ℝ) • v) ↔ M = 7 • 1 :=
by
  sorry

end NUMINAMATH_GPT_matrix_mult_7_l835_83573


namespace NUMINAMATH_GPT_simplify_sqrt_mul_l835_83565

theorem simplify_sqrt_mul : (Real.sqrt 5 * Real.sqrt (4 / 5) = 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_mul_l835_83565


namespace NUMINAMATH_GPT_inequality_ABC_l835_83587

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end NUMINAMATH_GPT_inequality_ABC_l835_83587


namespace NUMINAMATH_GPT_maximum_height_l835_83570

-- Define the quadratic function h(t)
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 60

-- Define our proof problem
theorem maximum_height : ∃ t : ℝ, h t = 140 :=
by
  let t := -80 / (2 * -20)
  use t
  sorry

end NUMINAMATH_GPT_maximum_height_l835_83570


namespace NUMINAMATH_GPT_simplify_and_rationalize_l835_83582

noncomputable def expression := 
  (Real.sqrt 8 / Real.sqrt 3) * 
  (Real.sqrt 25 / Real.sqrt 30) * 
  (Real.sqrt 16 / Real.sqrt 21)

theorem simplify_and_rationalize :
  expression = 4 * Real.sqrt 14 / 63 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_rationalize_l835_83582


namespace NUMINAMATH_GPT_range_of_k_for_distinct_roots_l835_83586
-- Import necessary libraries

-- Define the quadratic equation and conditions
noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the property of having distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c > 0

-- Define the specific problem instance and range condition
theorem range_of_k_for_distinct_roots (k : ℝ) :
  has_two_distinct_real_roots 1 2 k ↔ k < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_for_distinct_roots_l835_83586


namespace NUMINAMATH_GPT_log_roots_equivalence_l835_83592

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 5 / Real.log 3
noncomputable def c : ℝ := Real.log 2 / Real.log 5

theorem log_roots_equivalence :
  (x : ℝ) → (x = a ∨ x = b ∨ x = c) ↔ (x^3 - (a + b + c)*x^2 + (a*b + b*c + c*a)*x - a*b*c = 0) := by
  sorry

end NUMINAMATH_GPT_log_roots_equivalence_l835_83592


namespace NUMINAMATH_GPT_increasing_on_real_iff_a_range_l835_83579

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a / x

theorem increasing_on_real_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -3 ≤ a ∧ a ≤ -2 := 
by
  sorry

end NUMINAMATH_GPT_increasing_on_real_iff_a_range_l835_83579


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l835_83537

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (x > 1) ∨ (x ≤ -1) := 
by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l835_83537


namespace NUMINAMATH_GPT_find_x_and_y_l835_83547

variable {x y : ℝ}

-- Given condition
def angleDCE : ℝ := 58

-- Proof statements
theorem find_x_and_y : x = 180 - angleDCE ∧ y = 180 - angleDCE := by
  sorry

end NUMINAMATH_GPT_find_x_and_y_l835_83547


namespace NUMINAMATH_GPT_fraction_color_films_l835_83557

variables {x y : ℕ} (h₁ : y ≠ 0) (h₂ : x ≠ 0)

theorem fraction_color_films (h₃ : 30 * x > 0) (h₄ : 6 * y > 0) :
  (6 * y : ℚ) / ((3 * y / 10) + 6 * y) = 20 / 21 := by
  sorry

end NUMINAMATH_GPT_fraction_color_films_l835_83557


namespace NUMINAMATH_GPT_dan_marbles_l835_83554

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  original_marbles = 128 →
  given_marbles = 32 →
  remaining_marbles = original_marbles - given_marbles →
  remaining_marbles = 96 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_dan_marbles_l835_83554


namespace NUMINAMATH_GPT_points_satisfying_clubsuit_l835_83522

def clubsuit (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem points_satisfying_clubsuit (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x + y = 0) :=
by
  sorry

end NUMINAMATH_GPT_points_satisfying_clubsuit_l835_83522


namespace NUMINAMATH_GPT_max_non_managers_depA_l835_83556

theorem max_non_managers_depA (mA : ℕ) (nA : ℕ) (sA : ℕ) (gA : ℕ) (totalA : ℕ) :
  mA = 9 ∧ (8 * nA > 37 * mA) ∧ (sA = 2 * gA) ∧ (nA = sA + gA) ∧ (mA + nA ≤ 250) →
  nA = 39 :=
by
  sorry

end NUMINAMATH_GPT_max_non_managers_depA_l835_83556


namespace NUMINAMATH_GPT_simplify_expr_l835_83500

theorem simplify_expr : 
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = (5 : ℚ) / 4 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l835_83500


namespace NUMINAMATH_GPT_estimated_total_score_l835_83580

noncomputable def regression_score (x : ℝ) : ℝ := 7.3 * x - 96.9

theorem estimated_total_score (x : ℝ) (h : x = 95) : regression_score x = 596 :=
by
  rw [h]
  -- skipping the actual calculation steps
  sorry

end NUMINAMATH_GPT_estimated_total_score_l835_83580


namespace NUMINAMATH_GPT_value_of_a_l835_83518

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - 3 * x - Real.log x + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem value_of_a (a x0 : ℝ) (h : f x0 a = 3) : a = 1 - Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l835_83518


namespace NUMINAMATH_GPT_sector_area_given_angle_radius_sector_max_area_perimeter_l835_83542

open Real

theorem sector_area_given_angle_radius :
  ∀ (α : ℝ) (R : ℝ), α = 60 * (π / 180) ∧ R = 10 →
  (α / 360 * 2 * π * R) = 10 * π / 3 ∧ 
  (α * π * R^2 / 360) = 50 * π / 3 :=
by
  intros α R h
  rcases h with ⟨hα, hR⟩
  sorry

theorem sector_max_area_perimeter :
  ∀ (r α: ℝ), (2 * r + r * α) = 8 →
  α = 2 →
  r = 2 :=
by
  intros r α h ha
  sorry

end NUMINAMATH_GPT_sector_area_given_angle_radius_sector_max_area_perimeter_l835_83542


namespace NUMINAMATH_GPT_problem_l835_83505

theorem problem (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y + x * y = 3) :
  (0 < x * y ∧ x * y ≤ 1) ∧ (∀ z : ℝ, z = x + 2 * y → z = 4 * Real.sqrt 2 - 3) :=
by
  sorry

end NUMINAMATH_GPT_problem_l835_83505


namespace NUMINAMATH_GPT_eq_radicals_same_type_l835_83563

theorem eq_radicals_same_type (a b : ℕ) (h1 : a - 1 = 2) (h2 : 3 * b - 1 = 7 - b) : a + b = 5 :=
by
  sorry

end NUMINAMATH_GPT_eq_radicals_same_type_l835_83563


namespace NUMINAMATH_GPT_prob_of_selecting_blue_ball_l835_83578

noncomputable def prob_select_ball :=
  let prob_X := 1 / 3
  let prob_Y := 1 / 3
  let prob_Z := 1 / 3
  let prob_blue_X := 7 / 10
  let prob_blue_Y := 1 / 2
  let prob_blue_Z := 2 / 5
  prob_X * prob_blue_X + prob_Y * prob_blue_Y + prob_Z * prob_blue_Z

theorem prob_of_selecting_blue_ball :
  prob_select_ball = 8 / 15 :=
by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_prob_of_selecting_blue_ball_l835_83578


namespace NUMINAMATH_GPT_plane_equivalent_l835_83516

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2*s - 3*t, 1 + s, 4 - 3*s + t)

def plane_equation (x y z : ℝ) : Prop :=
  x - 7*y + 3*z - 8 = 0

theorem plane_equivalent :
  ∃ (s t : ℝ), parametric_plane s t = (x, y, z) ↔ plane_equation x y z :=
by
  sorry

end NUMINAMATH_GPT_plane_equivalent_l835_83516


namespace NUMINAMATH_GPT_average_speed_l835_83588

theorem average_speed (d d1 d2 s1 s2 : ℝ)
    (h1 : d = 100)
    (h2 : d1 = 50)
    (h3 : d2 = 50)
    (h4 : s1 = 20)
    (h5 : s2 = 50) :
    d / ((d1 / s1) + (d2 / s2)) = 28.57 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l835_83588


namespace NUMINAMATH_GPT_cubic_expression_solution_l835_83529

theorem cubic_expression_solution (r s : ℝ) (h₁ : 3 * r^2 - 4 * r - 7 = 0) (h₂ : 3 * s^2 - 4 * s - 7 = 0) :
  (3 * r^3 - 3 * s^3) / (r - s) = 37 / 3 :=
sorry

end NUMINAMATH_GPT_cubic_expression_solution_l835_83529


namespace NUMINAMATH_GPT_minimum_value_of_z_l835_83566

theorem minimum_value_of_z
  (x y : ℝ)
  (h1 : 3 * x + y - 6 ≥ 0)
  (h2 : x - y - 2 ≤ 0)
  (h3 : y - 3 ≤ 0) :
  ∃ z, z = 4 * x + y ∧ z = 7 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_z_l835_83566


namespace NUMINAMATH_GPT_appropriate_sampling_method_l835_83591

-- Defining the sizes of the boxes
def size_large : ℕ := 120
def size_medium : ℕ := 60
def size_small : ℕ := 20

-- Define a sample size
def sample_size : ℕ := 25

-- Define the concept of appropriate sampling method as being equivalent to stratified sampling in this context
theorem appropriate_sampling_method : 3 > 0 → sample_size > 0 → size_large = 120 ∧ size_medium = 60 ∧ size_small = 20 → 
("stratified sampling" = "stratified sampling") :=
by 
  sorry

end NUMINAMATH_GPT_appropriate_sampling_method_l835_83591


namespace NUMINAMATH_GPT_election_votes_l835_83540

theorem election_votes (P : ℕ) (M : ℕ) (V : ℕ) (hP : P = 60) (hM : M = 1300) :
  V = 6500 :=
by
  sorry

end NUMINAMATH_GPT_election_votes_l835_83540


namespace NUMINAMATH_GPT_problem_statement_l835_83514

def reading_method (n : ℕ) : String := sorry
-- Assume reading_method correctly implements the reading method for integers

def is_read_with_only_one_zero (n : ℕ) : Prop :=
  (reading_method n).count '0' = 1

theorem problem_statement : is_read_with_only_one_zero 83721000 = false := sorry

end NUMINAMATH_GPT_problem_statement_l835_83514


namespace NUMINAMATH_GPT_x_plus_y_l835_83568

variables {e1 e2 : ℝ → ℝ → Prop} -- Represents the vectors as properties of reals
variables {x y : ℝ} -- Real numbers x and y

-- Assuming non-collinearity of e1 and e2 (This means e1 and e2 are independent)
axiom non_collinear : e1 ≠ e2 

-- Given condition translated into Lean
axiom main_equation : (3 * x - 4 * y = 6) ∧ (2 * x - 3 * y = 3)

-- Prove that x + y = 9
theorem x_plus_y : x + y = 9 := 
by
  sorry -- Proof will be provided here

end NUMINAMATH_GPT_x_plus_y_l835_83568


namespace NUMINAMATH_GPT_minimum_value_is_correct_l835_83551

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  (x + 1/y) * (x + 1/y - 2024) + (y + 1/x) * (y + 1/x - 2024) + 2024

theorem minimum_value_is_correct (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (minimum_value x y) ≥ -2050208 := 
sorry

end NUMINAMATH_GPT_minimum_value_is_correct_l835_83551


namespace NUMINAMATH_GPT_running_speed_l835_83520

variables (w t_w t_r : ℝ)

-- Given conditions
def walking_speed : w = 8 := sorry
def walking_time_hours : t_w = 4.75 := sorry
def running_time_hours : t_r = 2 := sorry

-- Prove the man's running speed
theorem running_speed (w t_w t_r : ℝ) 
  (H1 : w = 8) 
  (H2 : t_w = 4.75) 
  (H3 : t_r = 2) : 
  (w * t_w) / t_r = 19 := 
sorry

end NUMINAMATH_GPT_running_speed_l835_83520


namespace NUMINAMATH_GPT_value_of_f_three_l835_83536

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * Real.cos x - x

theorem value_of_f_three (a b : ℝ) (h : f a b (-3) = 7) : f a b 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_three_l835_83536


namespace NUMINAMATH_GPT_num_isosceles_triangles_with_perimeter_30_l835_83571

theorem num_isosceles_triangles_with_perimeter_30 : 
  (∃ (s : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ s → 2 * a + b = 30 ∧ (a ≥ b) ∧ b ≠ 0 ∧ a + a > b ∧ a + b > a ∧ b + a > a) 
    ∧ s.card = 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_num_isosceles_triangles_with_perimeter_30_l835_83571


namespace NUMINAMATH_GPT_usual_time_to_school_l835_83567

theorem usual_time_to_school (S T t : ℝ) (h : 1.2 * S * (T - t) = S * T) : T = 6 * t :=
by
  sorry

end NUMINAMATH_GPT_usual_time_to_school_l835_83567


namespace NUMINAMATH_GPT_factory_processing_time_eq_l835_83559

variable (x : ℝ) (initial_rate : ℝ := x)
variable (parts : ℝ := 500)
variable (first_stage_parts : ℝ := 100)
variable (remaining_parts : ℝ := parts - first_stage_parts)
variable (total_days : ℝ := 6)
variable (new_rate : ℝ := 2 * initial_rate)

theorem factory_processing_time_eq (h : x > 0) : (first_stage_parts / initial_rate) + (remaining_parts / new_rate) = total_days := 
by
  sorry

end NUMINAMATH_GPT_factory_processing_time_eq_l835_83559


namespace NUMINAMATH_GPT_angle_B_is_60_l835_83596

noncomputable def triangle_with_centroid (a b c : ℝ) (GA GB GC : ℝ) : Prop :=
  56 * a * GA + 40 * b * GB + 35 * c * GC = 0

theorem angle_B_is_60 {a b c GA GB GC : ℝ} (h : 56 * a * GA + 40 * b * GB + 35 * c * GC = 0) :
  ∃ B : ℝ, B = 60 :=
sorry

end NUMINAMATH_GPT_angle_B_is_60_l835_83596


namespace NUMINAMATH_GPT_find_k_l835_83558

theorem find_k (k : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + 2 * k = 0 ∧ x = 1) : k = 1 / 2 :=
by {
  sorry 
}

end NUMINAMATH_GPT_find_k_l835_83558


namespace NUMINAMATH_GPT_range_of_m_l835_83524

variable {α : Type*} [LinearOrder α]

def increasing (f : α → α) : Prop :=
  ∀ ⦃x y : α⦄, x < y → f x < f y

theorem range_of_m 
  (f : ℝ → ℝ) 
  (h_inc : increasing f) 
  (h_cond : ∀ m : ℝ, f (m + 3) ≤ f 5) : 
  {m : ℝ | f (m + 3) ≤ f 5} = {m : ℝ | m ≤ 2} := 
sorry

end NUMINAMATH_GPT_range_of_m_l835_83524


namespace NUMINAMATH_GPT_part1_part2_l835_83507

open Set

-- Definitions from conditions in a)
def R : Set ℝ := univ
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Question part (1)
theorem part1 (a : ℝ) (h : a = 1) :
  (compl A) ∪ B a = {x | x ≤ -2 ∨ x > 1} :=
by 
  simp [h]
  sorry

-- Question part (2)
theorem part2 (a : ℝ) :
  A ⊆ B a → a ≤ -2 :=
by 
  sorry

end NUMINAMATH_GPT_part1_part2_l835_83507


namespace NUMINAMATH_GPT_ten_pow_n_plus_eight_div_nine_is_integer_l835_83508

theorem ten_pow_n_plus_eight_div_nine_is_integer (n : ℕ) : ∃ k : ℤ, 10^n + 8 = 9 * k := 
sorry

end NUMINAMATH_GPT_ten_pow_n_plus_eight_div_nine_is_integer_l835_83508


namespace NUMINAMATH_GPT_boys_love_marbles_l835_83531

def total_marbles : ℕ := 26
def marbles_per_boy : ℕ := 2
def num_boys_love_marbles : ℕ := total_marbles / marbles_per_boy

theorem boys_love_marbles : num_boys_love_marbles = 13 := by
  rfl

end NUMINAMATH_GPT_boys_love_marbles_l835_83531


namespace NUMINAMATH_GPT_total_spent_is_64_l835_83525

def deck_price : ℕ := 8
def victors_decks : ℕ := 6
def friends_decks : ℕ := 2

def victors_spending : ℕ := victors_decks * deck_price
def friends_spending : ℕ := friends_decks * deck_price
def total_spending : ℕ := victors_spending + friends_spending

theorem total_spent_is_64 : total_spending = 64 := by
  sorry

end NUMINAMATH_GPT_total_spent_is_64_l835_83525


namespace NUMINAMATH_GPT_largest_value_is_E_l835_83584

-- Define the given values
def A := 1 - 0.1
def B := 1 - 0.01
def C := 1 - 0.001
def D := 1 - 0.0001
def E := 1 - 0.00001

-- Main theorem statement
theorem largest_value_is_E : E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end NUMINAMATH_GPT_largest_value_is_E_l835_83584


namespace NUMINAMATH_GPT_gum_cost_700_eq_660_cents_l835_83528

-- defining the cost function
def gum_cost (n : ℕ) : ℝ :=
  if n ≤ 500 then n * 0.01
  else 5 + (n - 500) * 0.008

-- proving the specific case for 700 pieces of gum
theorem gum_cost_700_eq_660_cents : gum_cost 700 = 6.60 := by
  sorry

end NUMINAMATH_GPT_gum_cost_700_eq_660_cents_l835_83528


namespace NUMINAMATH_GPT_cyclic_quadrilateral_JMIT_l835_83501

theorem cyclic_quadrilateral_JMIT
  (a b c : ℂ)
  (I J M N T : ℂ)
  (hI : I = -(a*b + b*c + c*a))
  (hJ : J = a*b - b*c + c*a)
  (hM : M = (b^2 + c^2) / 2)
  (hN : N = b*c)
  (hT : T = 2*a^2 - b*c) :
  ∃ (k : ℝ), k = ((M - I) * (T - J)) / ((J - I) * (T - M)) :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_JMIT_l835_83501


namespace NUMINAMATH_GPT_probability_target_hit_l835_83594

theorem probability_target_hit {P_A P_B : ℚ}
  (hA : P_A = 1 / 2) 
  (hB : P_B = 1 / 3) 
  : (1 - (1 - P_A) * (1 - P_B)) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_probability_target_hit_l835_83594


namespace NUMINAMATH_GPT_xy_uv_zero_l835_83526

theorem xy_uv_zero (x y u v : ℝ) (h1 : x^2 + y^2 = 1) (h2 : u^2 + v^2 = 1) (h3 : x * u + y * v = 0) : x * y + u * v = 0 :=
by
  sorry

end NUMINAMATH_GPT_xy_uv_zero_l835_83526


namespace NUMINAMATH_GPT_cupcakes_frosted_in_10_minutes_l835_83546

theorem cupcakes_frosted_in_10_minutes (r1 r2 time : ℝ) (cagney_rate lacey_rate : r1 = 1 / 15 ∧ r2 = 1 / 25)
  (time_in_seconds : time = 600) :
  (1 / ((1 / r1) + (1 / r2)) * time) = 64 := by
  sorry

end NUMINAMATH_GPT_cupcakes_frosted_in_10_minutes_l835_83546


namespace NUMINAMATH_GPT_equation_represents_hyperbola_l835_83595

theorem equation_represents_hyperbola (x y : ℝ) :
  x^2 - 4*y^2 - 2*x + 8*y - 8 = 0 → ∃ a b h k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (a * (x - h)^2 - b * (y - k)^2 = 1) := 
sorry

end NUMINAMATH_GPT_equation_represents_hyperbola_l835_83595


namespace NUMINAMATH_GPT_star_24_75_l835_83553

noncomputable def star (a b : ℝ) : ℝ := sorry 

-- Conditions
axiom star_one_one : star 1 1 = 2
axiom star_ab_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : star (a * b) b = a * (star b b)
axiom star_a_one (a : ℝ) (h : 0 < a) : star a 1 = 2 * a

-- Theorem to prove
theorem star_24_75 : star 24 75 = 1800 := 
by 
  sorry

end NUMINAMATH_GPT_star_24_75_l835_83553


namespace NUMINAMATH_GPT_identify_clothing_l835_83530

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end NUMINAMATH_GPT_identify_clothing_l835_83530


namespace NUMINAMATH_GPT_unique_solution_of_diophantine_l835_83589

theorem unique_solution_of_diophantine (m n : ℕ) (hm_pos : m > 0) (hn_pos: n > 0) :
  m^2 = Int.sqrt n + Int.sqrt (2 * n + 1) → (m = 13 ∧ n = 4900) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_of_diophantine_l835_83589


namespace NUMINAMATH_GPT_find_m_l835_83581

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -13.6 :=
by sorry

end NUMINAMATH_GPT_find_m_l835_83581


namespace NUMINAMATH_GPT_greatest_possible_perimeter_l835_83575

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end NUMINAMATH_GPT_greatest_possible_perimeter_l835_83575


namespace NUMINAMATH_GPT_find_a_l835_83515

variable {a : ℝ}

def p (a : ℝ) := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧ x₁ * x₁ + 2 * a * x₁ + 1 = 0 ∧ x₂ * x₂ + 2 * a * x₂ + 1 = 0

def q (a : ℝ) := ∀ x : ℝ, a * x * x - a * x + 1 > 0 

theorem find_a (a : ℝ) : (p a ∨ q a) ∧ ¬ q a → a ≤ -1 :=
sorry

end NUMINAMATH_GPT_find_a_l835_83515


namespace NUMINAMATH_GPT_smallest_number_ending_in_6_moved_front_gives_4_times_l835_83561

theorem smallest_number_ending_in_6_moved_front_gives_4_times (x m n : ℕ) 
  (h1 : n = 10 * x + 6)
  (h2 : 6 * 10^m + x = 4 * n) :
  n = 1538466 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_ending_in_6_moved_front_gives_4_times_l835_83561


namespace NUMINAMATH_GPT_watch_all_episodes_in_67_weeks_l835_83577

def total_episodes : ℕ := 201
def episodes_per_week : ℕ := 1 + 2

theorem watch_all_episodes_in_67_weeks :
  total_episodes / episodes_per_week = 67 := by 
  sorry

end NUMINAMATH_GPT_watch_all_episodes_in_67_weeks_l835_83577


namespace NUMINAMATH_GPT_two_digit_number_representation_l835_83548

def tens_digit := ℕ
def units_digit := ℕ

theorem two_digit_number_representation (b a : ℕ) : 
  (∀ (b a : ℕ), 10 * b + a = 10 * b + a) := sorry

end NUMINAMATH_GPT_two_digit_number_representation_l835_83548


namespace NUMINAMATH_GPT_red_car_initial_distance_ahead_l835_83509

theorem red_car_initial_distance_ahead 
    (Speed_red Speed_black : ℕ) (Time : ℝ)
    (H1 : Speed_red = 10)
    (H2 : Speed_black = 50)
    (H3 : Time = 0.5) :
    let Distance_black := Speed_black * Time
    let Distance_red := Speed_red * Time
    Distance_black - Distance_red = 20 := 
by
  let Distance_black := Speed_black * Time
  let Distance_red := Speed_red * Time
  sorry

end NUMINAMATH_GPT_red_car_initial_distance_ahead_l835_83509


namespace NUMINAMATH_GPT_tom_total_seashells_l835_83513

-- Define the number of seashells Tom gave to Jessica.
def seashells_given_to_jessica : ℕ := 2

-- Define the number of seashells Tom still has.
def seashells_tom_has_now : ℕ := 3

-- Theorem stating that the total number of seashells Tom found is the sum of seashells_given_to_jessica and seashells_tom_has_now.
theorem tom_total_seashells : seashells_given_to_jessica + seashells_tom_has_now = 5 := 
by
  sorry

end NUMINAMATH_GPT_tom_total_seashells_l835_83513


namespace NUMINAMATH_GPT_tetrahedrons_from_triangular_prism_l835_83555

theorem tetrahedrons_from_triangular_prism : 
  let n := 6
  let choose4 := Nat.choose n 4
  let coplanar_cases := 3
  choose4 - coplanar_cases = 12 := by
  sorry

end NUMINAMATH_GPT_tetrahedrons_from_triangular_prism_l835_83555


namespace NUMINAMATH_GPT_find_a_100_l835_83519

noncomputable def a : Nat → Nat
| 0 => 0
| 1 => 2
| (n+1) => a n + 2 * n

theorem find_a_100 : a 100 = 9902 := 
  sorry

end NUMINAMATH_GPT_find_a_100_l835_83519


namespace NUMINAMATH_GPT_sin_pi_minus_a_l835_83585

theorem sin_pi_minus_a (a : ℝ) (h_cos_a : Real.cos a = Real.sqrt 5 / 3) (h_range_a : a ∈ Set.Ioo (-Real.pi / 2) 0) : 
  Real.sin (Real.pi - a) = -2 / 3 :=
by sorry

end NUMINAMATH_GPT_sin_pi_minus_a_l835_83585


namespace NUMINAMATH_GPT_initial_violet_balloons_l835_83560

-- Let's define the given conditions
def red_balloons : ℕ := 4
def violet_balloons_lost : ℕ := 3
def violet_balloons_now : ℕ := 4

-- Define the statement to prove
theorem initial_violet_balloons :
  (violet_balloons_now + violet_balloons_lost) = 7 :=
by
  sorry

end NUMINAMATH_GPT_initial_violet_balloons_l835_83560


namespace NUMINAMATH_GPT_CD_is_b_minus_a_minus_c_l835_83574

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (A B C D : V) (a b c : V)

def AB : V := a
def AD : V := b
def BC : V := c

theorem CD_is_b_minus_a_minus_c (h1 : A + a = B) (h2 : A + b = D) (h3 : B + c = C) :
  D - C = b - a - c :=
by sorry

end NUMINAMATH_GPT_CD_is_b_minus_a_minus_c_l835_83574


namespace NUMINAMATH_GPT_ratio_of_wire_lengths_l835_83599

theorem ratio_of_wire_lengths 
  (bonnie_wire_length : ℕ := 80)
  (roark_wire_length : ℕ := 12000) :
  bonnie_wire_length / roark_wire_length = 1 / 150 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_wire_lengths_l835_83599


namespace NUMINAMATH_GPT_function_periodicity_l835_83527

theorem function_periodicity (f : ℝ → ℝ) (h1 : ∀ x, f (-x) + f x = 0)
  (h2 : ∀ x, f (x + 1) = f (1 - x)) (h3 : f 1 = 5) : f 2015 = -5 :=
sorry

end NUMINAMATH_GPT_function_periodicity_l835_83527


namespace NUMINAMATH_GPT_suma_work_rate_l835_83572

theorem suma_work_rate (W : ℕ) : 
  (∀ W, (W / 6) + (W / S) = W / 4) → S = 24 :=
by
  intro h
  -- detailed proof would actually go here
  sorry

end NUMINAMATH_GPT_suma_work_rate_l835_83572


namespace NUMINAMATH_GPT_fraction_of_products_inspected_jane_l835_83583

theorem fraction_of_products_inspected_jane 
  (P : ℝ) 
  (J : ℝ) 
  (John_rejection_rate : ℝ) 
  (Jane_rejection_rate : ℝ)
  (Total_rejection_rate : ℝ) 
  (hJohn : John_rejection_rate = 0.005) 
  (hJane : Jane_rejection_rate = 0.008) 
  (hTotal : Total_rejection_rate = 0.0075) 
  : J = 5 / 6 := by
{
  sorry
}

end NUMINAMATH_GPT_fraction_of_products_inspected_jane_l835_83583


namespace NUMINAMATH_GPT_satisfactory_fraction_is_28_over_31_l835_83597

-- Define the number of students for each grade
def students_with_grade_A := 8
def students_with_grade_B := 7
def students_with_grade_C := 6
def students_with_grade_D := 4
def students_with_grade_E := 3
def students_with_grade_F := 3

-- Calculate the total number of students with satisfactory grades
def satisfactory_grades := students_with_grade_A + students_with_grade_B + students_with_grade_C + students_with_grade_D + students_with_grade_E

-- Calculate the total number of students
def total_students := satisfactory_grades + students_with_grade_F

-- Define the fraction of satisfactory grades
def satisfactory_fraction : ℚ := satisfactory_grades / total_students

-- The main proposition that the satisfactory fraction is 28/31
theorem satisfactory_fraction_is_28_over_31 : satisfactory_fraction = 28 / 31 := by {
  sorry
}

end NUMINAMATH_GPT_satisfactory_fraction_is_28_over_31_l835_83597


namespace NUMINAMATH_GPT_find_ordered_pair_l835_83523

theorem find_ordered_pair (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) := by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l835_83523


namespace NUMINAMATH_GPT_remainder_div_polynomial_l835_83517

theorem remainder_div_polynomial :
  ∀ (x : ℝ), 
  ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ), 
    R x = (3^101 - 2^101) * x + (2^101 - 2 * 3^101) ∧
    x^101 = (x^2 - 5 * x + 6) * Q x + R x :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_polynomial_l835_83517


namespace NUMINAMATH_GPT_hectares_per_day_initial_l835_83545

variable (x : ℝ) -- x is the number of hectares one tractor ploughs initially per day

-- Condition 1: A field can be ploughed by 6 tractors in 4 days.
def total_area_initial := 6 * x * 4

-- Condition 2: 6 tractors plough together a certain number of hectares per day, denoted as x hectares/day.
-- This is incorporated in the variable declaration of x.

-- Condition 3: If 2 tractors are moved to another field, the remaining 4 tractors can plough the same field in 5 days.
-- Condition 4: One of the 4 tractors ploughs 144 hectares a day when 4 tractors plough the field in 5 days.
def total_area_with_4_tractors := 4 * 144 * 5

-- The statement that equates the two total area expressions.
theorem hectares_per_day_initial : total_area_initial x = total_area_with_4_tractors := by
  sorry

end NUMINAMATH_GPT_hectares_per_day_initial_l835_83545


namespace NUMINAMATH_GPT_parabola_directrix_standard_eq_l835_83569

theorem parabola_directrix_standard_eq (y : ℝ) (x : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ ∀ (P : {P // P ≠ x ∨ P ≠ y}), 
  (y + 1) = p) → x^2 = 4 * y :=
sorry

end NUMINAMATH_GPT_parabola_directrix_standard_eq_l835_83569


namespace NUMINAMATH_GPT_conjugate_axis_length_l835_83521

variable (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
variable (e : ℝ) (h3 : e = Real.sqrt 7 / 2)
variable (c : ℝ) (h4 : c = a * e)
variable (P : ℝ × ℝ) (h5 : P = (c, b^2 / a))
variable (F1 F2 : ℝ × ℝ) (h6 : F1 = (-c, 0)) (h7 : F2 = (c, 0))
variable (h8 : dist P F2 = 9 / 2)
variable (h9 : P.1 = c) (h10 : P.2 = b^2 / a)
variable (h11 : PF_2 ⊥ F_1F_2)

theorem conjugate_axis_length : 2 * b = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_conjugate_axis_length_l835_83521


namespace NUMINAMATH_GPT_no_psafe_numbers_l835_83503

def is_psafe (n p : ℕ) : Prop := 
  ¬ (n % p = 0 ∨ n % p = 1 ∨ n % p = 2 ∨ n % p = 3 ∨ n % p = p - 3 ∨ n % p = p - 2 ∨ n % p = p - 1)

theorem no_psafe_numbers (N : ℕ) (hN : N = 10000) :
  ∀ n, (n ≤ N ∧ is_psafe n 5 ∧ is_psafe n 7 ∧ is_psafe n 11) → false :=
by
  sorry

end NUMINAMATH_GPT_no_psafe_numbers_l835_83503


namespace NUMINAMATH_GPT_factorization_problem_l835_83593

theorem factorization_problem :
  ∃ a b : ℤ, (∀ y : ℤ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b))
    ∧ (a - b = 11) := by
  sorry

end NUMINAMATH_GPT_factorization_problem_l835_83593


namespace NUMINAMATH_GPT_cans_to_collect_l835_83598

theorem cans_to_collect
  (martha_cans : ℕ)
  (diego_half_plus_ten : ℕ)
  (total_cans_required : ℕ)
  (martha_cans_collected : martha_cans = 90)
  (diego_collected : diego_half_plus_ten = (martha_cans / 2) + 10)
  (goal_cans : total_cans_required = 150) :
  total_cans_required - (martha_cans + diego_half_plus_ten) = 5 :=
by
  sorry

end NUMINAMATH_GPT_cans_to_collect_l835_83598


namespace NUMINAMATH_GPT_distinctKeyArrangements_l835_83590

-- Given conditions as definitions in Lean.
def houseNextToCar : Prop := sorry
def officeNextToBike : Prop := sorry
def noDifferenceByRotationOrReflection (arr1 arr2 : List ℕ) : Prop := sorry

-- Main statement to be proven
theorem distinctKeyArrangements : 
  houseNextToCar ∧ officeNextToBike ∧ (∀ (arr1 arr2 : List ℕ), noDifferenceByRotationOrReflection arr1 arr2 ↔ arr1 = arr2) 
  → ∃ n : ℕ, n = 16 :=
by sorry

end NUMINAMATH_GPT_distinctKeyArrangements_l835_83590


namespace NUMINAMATH_GPT_fruit_ratio_l835_83576

variable (A P B : ℕ)
variable (n : ℕ)

theorem fruit_ratio (h1 : A = 4) (h2 : P = n * A) (h3 : A + P + B = 21) (h4 : B = 5) : P / A = 3 := by
  sorry

end NUMINAMATH_GPT_fruit_ratio_l835_83576


namespace NUMINAMATH_GPT_count_harmonic_vals_l835_83532

def floor (x : ℝ) : ℤ := sorry -- or use Mathlib function
def frac (x : ℝ) : ℝ := x - (floor x)

def is_harmonic_progression (a b c : ℝ) : Prop := 
  (1 / a) = (2 / b) - (1 / c)

theorem count_harmonic_vals :
  (∃ x, is_harmonic_progression x (floor x) (frac x)) ∧
  (∃! x1 x2, is_harmonic_progression x1 (floor x1) (frac x1) ∧
               is_harmonic_progression x2 (floor x2) (frac x2)) ∧
  x1 ≠ x2 :=
  sorry

end NUMINAMATH_GPT_count_harmonic_vals_l835_83532


namespace NUMINAMATH_GPT_remainder_of_899830_divided_by_16_is_6_l835_83541

theorem remainder_of_899830_divided_by_16_is_6 :
  ∃ k : ℕ, 899830 = 16 * k + 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_899830_divided_by_16_is_6_l835_83541
