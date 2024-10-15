import Mathlib

namespace NUMINAMATH_GPT_quadratic_positive_intervals_l754_75406

-- Problem setup
def quadratic (x : ℝ) : ℝ := x^2 - x - 6

-- Define the roots of the quadratic function
def is_root (a b : ℝ) (f : ℝ → ℝ) := f a = 0 ∧ f b = 0

-- Proving the intervals where the quadratic function is greater than 0
theorem quadratic_positive_intervals :
  is_root (-2) 3 quadratic →
  { x : ℝ | quadratic x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 3 } :=
by
  sorry

end NUMINAMATH_GPT_quadratic_positive_intervals_l754_75406


namespace NUMINAMATH_GPT_part1_part2_l754_75417

open Complex

theorem part1 {m : ℝ} : m + (m^2 + 2) * I = 0 -> m = 0 :=
by sorry

theorem part2 {m : ℝ} (h : (m + I)^2 - 2 * (m + I) + 2 = 0) :
    (let z1 := m + I
     let z2 := 2 + m * I
     im ((z2 / z1) : ℂ) = -1 / 2) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l754_75417


namespace NUMINAMATH_GPT_alvin_age_l754_75452

theorem alvin_age (A S : ℕ) (h_s : S = 10) (h_cond : S = 1/2 * A - 5) : A = 30 := by
  sorry

end NUMINAMATH_GPT_alvin_age_l754_75452


namespace NUMINAMATH_GPT_partA_l754_75435

theorem partA (a b c : ℤ) (h : ∀ x : ℤ, ∃ k : ℤ, a * x ^ 2 + b * x + c = k ^ 4) : a = 0 ∧ b = 0 := 
sorry

end NUMINAMATH_GPT_partA_l754_75435


namespace NUMINAMATH_GPT_evaluate_expression_l754_75422

theorem evaluate_expression : 3 + 2 * (8 - 3) = 13 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l754_75422


namespace NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_squared_l754_75456

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_reciprocal_squared_l754_75456


namespace NUMINAMATH_GPT_fractions_equiv_l754_75461

theorem fractions_equiv:
  (8 : ℝ) / (7 * 67) = (0.8 : ℝ) / (0.7 * 67) :=
by
  sorry

end NUMINAMATH_GPT_fractions_equiv_l754_75461


namespace NUMINAMATH_GPT_find_a_l754_75412

-- Given conditions
variables (x y z a : ℤ)

def conditions : Prop :=
  (x - 10) * (y - a) * (z - 2) = 1000 ∧
  ∃ (x y z : ℤ), x + y + z = 7

theorem find_a (x y z : ℤ) (h : conditions x y z 1) : a = 1 := 
  by
    sorry

end NUMINAMATH_GPT_find_a_l754_75412


namespace NUMINAMATH_GPT_cider_apples_production_l754_75411

def apples_total : Real := 8.0
def baking_fraction : Real := 0.30
def cider_fraction : Real := 0.60

def apples_remaining : Real := apples_total * (1 - baking_fraction)
def apples_for_cider : Real := apples_remaining * cider_fraction

theorem cider_apples_production : 
    apples_for_cider = 3.4 := 
by
  sorry

end NUMINAMATH_GPT_cider_apples_production_l754_75411


namespace NUMINAMATH_GPT_find_m_of_quadratic_function_l754_75488

theorem find_m_of_quadratic_function :
  ∀ (m : ℝ), (m + 1 ≠ 0) → ((m + 1) * x ^ (m^2 + 1) + 5 = a * x^2 + b * x + c) → m = 1 :=
by
  intro m h h_quad
  -- Proof Here
  sorry

end NUMINAMATH_GPT_find_m_of_quadratic_function_l754_75488


namespace NUMINAMATH_GPT_grandfather_older_than_grandmother_l754_75479

noncomputable def Milena_age : ℕ := 7

noncomputable def Grandmother_age : ℕ := Milena_age * 9

noncomputable def Grandfather_age : ℕ := Milena_age + 58

theorem grandfather_older_than_grandmother :
  Grandfather_age - Grandmother_age = 2 := by
  sorry

end NUMINAMATH_GPT_grandfather_older_than_grandmother_l754_75479


namespace NUMINAMATH_GPT_value_of_f_sum_l754_75418

variable (a b c m : ℝ)

def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f_sum :
  f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_sum_l754_75418


namespace NUMINAMATH_GPT_more_pens_than_pencils_l754_75401

-- Define the number of pencils (P) and pens (Pe)
def num_pencils : ℕ := 15 * 80

-- Define the number of pens (Pe) is more than twice the number of pencils (P)
def num_pens (Pe : ℕ) : Prop := Pe > 2 * num_pencils

-- State the total cost equation in terms of pens and pencils
def total_cost_eq (Pe : ℕ) : Prop := (5 * Pe + 4 * num_pencils = 18300)

-- Prove that the number of more pens than pencils is 1500
theorem more_pens_than_pencils (Pe : ℕ) (h1 : num_pens Pe) (h2 : total_cost_eq Pe) : (Pe - num_pencils = 1500) :=
by
  sorry

end NUMINAMATH_GPT_more_pens_than_pencils_l754_75401


namespace NUMINAMATH_GPT_pie_count_correct_l754_75484

structure Berries :=
  (strawberries : ℕ)
  (blueberries : ℕ)
  (raspberries : ℕ)

def christine_picking : Berries := {strawberries := 10, blueberries := 8, raspberries := 20}

def rachel_picking : Berries :=
  let c := christine_picking
  {strawberries := 2 * c.strawberries,
   blueberries := 2 * c.blueberries,
   raspberries := c.raspberries / 2}

def total_berries (b1 b2 : Berries) : Berries :=
  {strawberries := b1.strawberries + b2.strawberries,
   blueberries := b1.blueberries + b2.blueberries,
   raspberries := b1.raspberries + b2.raspberries}

def pie_requirements : Berries := {strawberries := 3, blueberries := 2, raspberries := 4}

def max_pies (total : Berries) (requirements : Berries) : Berries :=
  {strawberries := total.strawberries / requirements.strawberries,
   blueberries := total.blueberries / requirements.blueberries,
   raspberries := total.raspberries / requirements.raspberries}

def correct_pies : Berries := {strawberries := 10, blueberries := 12, raspberries := 7}

theorem pie_count_correct :
  let total := total_berries christine_picking rachel_picking;
  max_pies total pie_requirements = correct_pies :=
by {
  sorry
}

end NUMINAMATH_GPT_pie_count_correct_l754_75484


namespace NUMINAMATH_GPT_curvature_formula_l754_75448

noncomputable def curvature_squared (x y : ℝ → ℝ) (t : ℝ) :=
  let x' := (deriv x t)
  let y' := (deriv y t)
  let x'' := (deriv (deriv x) t)
  let y'' := (deriv (deriv y) t)
  (x'' * y' - y'' * x')^2 / (x'^2 + y'^2)^3

theorem curvature_formula (x y : ℝ → ℝ) (t : ℝ) :
  let k_sq := curvature_squared x y t
  k_sq = ((deriv (deriv x) t * deriv y t - deriv (deriv y) t * deriv x t)^2 /
         ((deriv x t)^2 + (deriv y t)^2)^3) := 
by 
  sorry

end NUMINAMATH_GPT_curvature_formula_l754_75448


namespace NUMINAMATH_GPT_fraction_of_marbles_taken_away_l754_75497

theorem fraction_of_marbles_taken_away (Chris_marbles Ryan_marbles remaining_marbles total_marbles taken_away_marbles : ℕ) 
    (hChris : Chris_marbles = 12) 
    (hRyan : Ryan_marbles = 28) 
    (hremaining : remaining_marbles = 20) 
    (htotal : total_marbles = Chris_marbles + Ryan_marbles) 
    (htaken_away : taken_away_marbles = total_marbles - remaining_marbles) : 
    (taken_away_marbles : ℚ) / total_marbles = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_of_marbles_taken_away_l754_75497


namespace NUMINAMATH_GPT_imaginary_part_of_z_l754_75496

-- Define the imaginary unit i where i^2 = -1
def imaginary_unit : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := (2 + imaginary_unit) * (1 - imaginary_unit)

-- State the theorem to prove the imaginary part of z
theorem imaginary_part_of_z : Complex.im z = -1 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l754_75496


namespace NUMINAMATH_GPT_find_digit_D_l754_75480

theorem find_digit_D (A B C D : ℕ) (h1 : A + B = A + 10 * (B / 10)) (h2 : D + 10 * (A / 10) = A + C)
  (h3 : A + 10 * (B / 10) - C = A) (h4 : 0 ≤ A) (h5 : A ≤ 9) (h6 : 0 ≤ B) (h7 : B ≤ 9)
  (h8 : 0 ≤ C) (h9 : C ≤ 9) (h10 : 0 ≤ D) (h11 : D ≤ 9) : D = 9 := 
sorry

end NUMINAMATH_GPT_find_digit_D_l754_75480


namespace NUMINAMATH_GPT_computer_multiplications_l754_75419

def rate : ℕ := 15000
def time : ℕ := 2 * 3600
def expected_multiplications : ℕ := 108000000

theorem computer_multiplications : rate * time = expected_multiplications := by
  sorry

end NUMINAMATH_GPT_computer_multiplications_l754_75419


namespace NUMINAMATH_GPT_milk_production_group_B_l754_75430

theorem milk_production_group_B (a b c d e : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_pos_d : d > 0) (h_pos_e : e > 0) :
  ((1.2 * b * d * e) / (a * c)) = 1.2 * (b / (a * c)) * d * e := 
by
  sorry

end NUMINAMATH_GPT_milk_production_group_B_l754_75430


namespace NUMINAMATH_GPT_liliane_has_44_44_more_cookies_l754_75476

variables (J : ℕ) (L O : ℕ) (totalCookies : ℕ)

def liliane_has_more_30_percent (J L : ℕ) : Prop :=
  L = J + (3 * J / 10)

def oliver_has_less_10_percent (J O : ℕ) : Prop :=
  O = J - (J / 10)

def total_cookies (J L O totalCookies : ℕ) : Prop :=
  J + L + O = totalCookies

theorem liliane_has_44_44_more_cookies
  (h1 : liliane_has_more_30_percent J L)
  (h2 : oliver_has_less_10_percent J O)
  (h3 : total_cookies J L O totalCookies)
  (h4 : totalCookies = 120) :
  (L - O) * 100 / O = 4444 / 100 := sorry

end NUMINAMATH_GPT_liliane_has_44_44_more_cookies_l754_75476


namespace NUMINAMATH_GPT_problem_statement_l754_75400

-- Define the operation * based on the given mathematical definition
def op (a b : ℕ) : ℤ := a * (a - b)

-- The core theorem to prove the expression in the problem
theorem problem_statement : op 2 3 + op (6 - 2) 4 = -2 :=
by
  -- This is where the proof would go, but it's omitted with sorry.
  sorry

end NUMINAMATH_GPT_problem_statement_l754_75400


namespace NUMINAMATH_GPT_number_of_true_propositions_l754_75433

noncomputable def proposition1 : Prop := ∀ (x : ℝ), x^2 - 3 * x + 2 > 0
noncomputable def proposition2 : Prop := ∃ (x : ℚ), x^2 = 2
noncomputable def proposition3 : Prop := ∃ (x : ℝ), x^2 - 1 = 0
noncomputable def proposition4 : Prop := ∀ (x : ℝ), 4 * x^2 > 2 * x - 1 + 3 * x^2

theorem number_of_true_propositions : (¬ proposition1 ∧ ¬ proposition2 ∧ proposition3 ∧ ¬ proposition4) → 1 = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_l754_75433


namespace NUMINAMATH_GPT_gobblean_total_words_l754_75420

-- Define the Gobblean alphabet and its properties.
def gobblean_letters := 6
def max_word_length := 4

-- Function to calculate number of permutations without repetition for a given length.
def num_words (length : ℕ) : ℕ :=
  if length = 1 then 6
  else if length = 2 then 6 * 5
  else if length = 3 then 6 * 5 * 4
  else if length = 4 then 6 * 5 * 4 * 3
  else 0

-- Main theorem stating the total number of possible words.
theorem gobblean_total_words : 
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4) = 516 :=
by
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_gobblean_total_words_l754_75420


namespace NUMINAMATH_GPT_total_time_of_four_sets_of_stairs_l754_75494

def time_first : ℕ := 15
def time_increment : ℕ := 10
def num_sets : ℕ := 4

theorem total_time_of_four_sets_of_stairs :
  let a := time_first
  let d := time_increment
  let n := num_sets
  let l := a + (n - 1) * d
  let S := n / 2 * (a + l)
  S = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_time_of_four_sets_of_stairs_l754_75494


namespace NUMINAMATH_GPT_inequality_solution_l754_75463

open Set Real

theorem inequality_solution (x : ℝ) :
  (1 / (x + 1) + 3 / (x + 7) ≥ 2 / 3) ↔ (x ∈ Ioo (-7 : ℝ) (-4) ∪ Ioo (-1) (2) ∪ {(-4 : ℝ), 2}) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l754_75463


namespace NUMINAMATH_GPT_geometric_sequence_s4_l754_75485

noncomputable def geometric_sequence_sum : ℕ → ℝ → ℝ → ℝ
| 0, a1, q => 0
| (n+1), a1, q => a1 * (1 - q^(n+1)) / (1 - q)

variable (a1 q : ℝ) (n : ℕ)

theorem geometric_sequence_s4  (h1 : a1 * (q^1) * (q^3) = 16) (h2 : geometric_sequence_sum 2 a1 q + a1 * (q^2) = 7) :
  geometric_sequence_sum 3 a1 q = 15 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_s4_l754_75485


namespace NUMINAMATH_GPT_twelfth_term_l754_75450

-- Definitions based on the given conditions
def a_3_condition (a d : ℚ) : Prop := a + 2 * d = 10
def a_6_condition (a d : ℚ) : Prop := a + 5 * d = 20

-- The main theorem stating that the twelfth term is 40
theorem twelfth_term (a d : ℚ) (h1 : a_3_condition a d) (h2 : a_6_condition a d) :
  a + 11 * d = 40 :=
sorry

end NUMINAMATH_GPT_twelfth_term_l754_75450


namespace NUMINAMATH_GPT_find_two_digit_number_l754_75427

theorem find_two_digit_number (n : ℕ) (h1 : n % 9 = 7) (h2 : n % 7 = 5) (h3 : n % 3 = 1) (h4 : 10 ≤ n) (h5 : n < 100) : n = 61 := 
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l754_75427


namespace NUMINAMATH_GPT_teams_working_together_l754_75453

theorem teams_working_together
    (m n : ℕ) 
    (hA : ∀ t : ℕ, t = m → (t ≥ 0)) 
    (hB : ∀ t : ℕ, t = n → (t ≥ 0)) : 
  ∃ t : ℕ, t = (m * n) / (m + n) :=
by
  sorry

end NUMINAMATH_GPT_teams_working_together_l754_75453


namespace NUMINAMATH_GPT_trisha_take_home_pay_l754_75451

theorem trisha_take_home_pay :
  let hourly_wage := 15
  let hours_per_week := 40
  let weeks_per_year := 52
  let withholding_percentage := 0.2

  let annual_gross_pay := hourly_wage * hours_per_week * weeks_per_year
  let withholding_amount := annual_gross_pay * withholding_percentage
  let take_home_pay := annual_gross_pay - withholding_amount

  take_home_pay = 24960 :=
by
  sorry

end NUMINAMATH_GPT_trisha_take_home_pay_l754_75451


namespace NUMINAMATH_GPT_find_parabola_constant_l754_75464

theorem find_parabola_constant (a b c : ℝ) (h_vertex : ∀ y, (4:ℝ) = -5 / 4 * y * y + 5 / 2 * y + c)
  (h_point : (-1:ℝ) = -5 / 4 * (3:ℝ) ^ 2 + 5 / 2 * (3:ℝ) + c ) :
  c = 11 / 4 :=
sorry

end NUMINAMATH_GPT_find_parabola_constant_l754_75464


namespace NUMINAMATH_GPT_solution_set_of_f_inequality_l754_75490

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_deriv : ∀ x, f' x < f x)
variable (h_even : ∀ x, f (x + 2) = f (-x + 2))
variable (h_initial : f 0 = Real.exp 4)

theorem solution_set_of_f_inequality :
  {x : ℝ | f x < Real.exp x} = {x : ℝ | x > 4} := 
sorry

end NUMINAMATH_GPT_solution_set_of_f_inequality_l754_75490


namespace NUMINAMATH_GPT_five_digit_numbers_l754_75434

def divisible_by_4_and_9 (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 9 = 0)

def is_candidate (n : ℕ) : Prop :=
  ∃ a b, n = 10000 * a + 1000 + 200 + 30 + b ∧ a < 10 ∧ b < 10

theorem five_digit_numbers :
  ∀ (n : ℕ), is_candidate n → divisible_by_4_and_9 n → n = 11232 ∨ n = 61236 :=
by
  sorry

end NUMINAMATH_GPT_five_digit_numbers_l754_75434


namespace NUMINAMATH_GPT_frank_won_skee_ball_tickets_l754_75403

noncomputable def tickets_whack_a_mole : ℕ := 33
noncomputable def candies_bought : ℕ := 7
noncomputable def tickets_per_candy : ℕ := 6
noncomputable def total_tickets_spent : ℕ := candies_bought * tickets_per_candy
noncomputable def tickets_skee_ball : ℕ := total_tickets_spent - tickets_whack_a_mole

theorem frank_won_skee_ball_tickets : tickets_skee_ball = 9 :=
  by
  sorry

end NUMINAMATH_GPT_frank_won_skee_ball_tickets_l754_75403


namespace NUMINAMATH_GPT_abs_inequality_m_eq_neg4_l754_75426

theorem abs_inequality_m_eq_neg4 (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ (m = -4) :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_m_eq_neg4_l754_75426


namespace NUMINAMATH_GPT_fraction_subtraction_l754_75444

theorem fraction_subtraction : 
  (4 + 6 + 8 + 10) / (3 + 5 + 7) - (3 + 5 + 7 + 9) / (4 + 6 + 8) = 8 / 15 :=
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l754_75444


namespace NUMINAMATH_GPT_positive_difference_median_mode_l754_75467

-- Definition of the data set
def data : List ℕ := [12, 13, 14, 15, 15, 22, 22, 22, 31, 31, 38, 39, 40, 41, 42, 43, 52, 58, 59]

-- Definition of the mode
def mode (l : List ℕ) : ℕ := 22  -- Specific to the data set provided

-- Definition of the median
def median (l : List ℕ) : ℕ := 31  -- Specific to the data set provided

-- Proof statement
theorem positive_difference_median_mode : 
  (median data - mode data) = 9 := by 
  sorry

end NUMINAMATH_GPT_positive_difference_median_mode_l754_75467


namespace NUMINAMATH_GPT_neither_long_furred_nor_brown_dogs_is_8_l754_75437

def total_dogs : ℕ := 45
def long_furred_dogs : ℕ := 29
def brown_dogs : ℕ := 17
def long_furred_and_brown_dogs : ℕ := 9

def neither_long_furred_nor_brown_dogs : ℕ :=
  total_dogs - (long_furred_dogs + brown_dogs - long_furred_and_brown_dogs)

theorem neither_long_furred_nor_brown_dogs_is_8 :
  neither_long_furred_nor_brown_dogs = 8 := 
by 
  -- Here we can use substitution and calculation steps used in the solution
  sorry

end NUMINAMATH_GPT_neither_long_furred_nor_brown_dogs_is_8_l754_75437


namespace NUMINAMATH_GPT_minimum_kinds_of_candies_l754_75438

/-- In a candy store, a salesperson placed 91 candies of several kinds in a row on the counter.
    It turned out that between any two candies of the same kind, there is an even number of candies. 
    What is the minimum number of kinds of candies that could be? -/
theorem minimum_kinds_of_candies (n : ℕ) (h : 91 < 2 * n) : n ≥ 46 :=
sorry

end NUMINAMATH_GPT_minimum_kinds_of_candies_l754_75438


namespace NUMINAMATH_GPT_distance_between_cities_l754_75443

variable (D : ℝ) -- D is the distance between City A and City B
variable (time_AB : ℝ) -- Time from City A to City B
variable (time_BA : ℝ) -- Time from City B to City A
variable (saved_time : ℝ) -- Time saved per trip
variable (avg_speed : ℝ) -- Average speed for the round trip with saved time

theorem distance_between_cities :
  time_AB = 6 → time_BA = 4.5 → saved_time = 0.5 → avg_speed = 90 →
  D = 427.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_cities_l754_75443


namespace NUMINAMATH_GPT_smallest_value_of_a_squared_plus_b_l754_75428

theorem smallest_value_of_a_squared_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → a * x^3 + b * y^2 ≥ x * y - 1) :
    a^2 + b = 2 / (3 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_a_squared_plus_b_l754_75428


namespace NUMINAMATH_GPT_range_of_p_l754_75407

def h (x : ℝ) : ℝ := 4 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → -1 ≤ p x ∧ p x ≤ 1023 :=
by
  sorry

end NUMINAMATH_GPT_range_of_p_l754_75407


namespace NUMINAMATH_GPT_measure_of_B_l754_75481

-- Define the conditions (angles and their relationships)
variable (angle_P angle_R angle_O angle_B angle_L angle_S : ℝ)
variable (sum_of_angles : angle_P + angle_R + angle_O + angle_B + angle_L + angle_S = 720)
variable (supplementary_O_S : angle_O + angle_S = 180)
variable (right_angle_L : angle_L = 90)
variable (congruent_angles : angle_P = angle_R ∧ angle_R = angle_B)

-- Prove the measure of angle B
theorem measure_of_B : angle_B = 150 := by
  sorry

end NUMINAMATH_GPT_measure_of_B_l754_75481


namespace NUMINAMATH_GPT_algebraic_expression_value_l754_75471

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2023 * a - 1 = 0) : 
  a * (a + 1) * (a - 1) + 2023 * a^2 + 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l754_75471


namespace NUMINAMATH_GPT_percent_difference_l754_75415

theorem percent_difference (a b : ℝ) : 
  a = 67.5 * 250 / 100 → 
  b = 52.3 * 180 / 100 → 
  (a - b) = 74.61 :=
by
  intros ha hb
  rw [ha, hb]
  -- omitted proof
  sorry

end NUMINAMATH_GPT_percent_difference_l754_75415


namespace NUMINAMATH_GPT_smallest_x_value_l754_75457

theorem smallest_x_value (x : ℝ) (h : |4 * x + 9| = 37) : x = -11.5 :=
sorry

end NUMINAMATH_GPT_smallest_x_value_l754_75457


namespace NUMINAMATH_GPT_cubic_roots_and_k_value_l754_75414

theorem cubic_roots_and_k_value (k r₃ : ℝ) :
  (∃ r₃, 3 - 2 + r₃ = -5 ∧ 3 * (-2) * r₃ = -12 ∧ k = 3 * (-2) + (-2) * r₃ + r₃ * 3) →
  (k = -12 ∧ r₃ = -6) :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_and_k_value_l754_75414


namespace NUMINAMATH_GPT_t50_mod_7_l754_75492

def T (n : ℕ) : ℕ :=
  match n with
  | 0     => 3
  | n + 1 => 3 ^ T n

theorem t50_mod_7 : T 50 % 7 = 6 := sorry

end NUMINAMATH_GPT_t50_mod_7_l754_75492


namespace NUMINAMATH_GPT_solve_for_n_l754_75465

-- Define the problem statement
theorem solve_for_n : ∃ n : ℕ, (3 * n^2 + n = 219) ∧ (n = 9) := 
sorry

end NUMINAMATH_GPT_solve_for_n_l754_75465


namespace NUMINAMATH_GPT_remainder_of_sum_mod_11_l754_75493

theorem remainder_of_sum_mod_11 :
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_mod_11_l754_75493


namespace NUMINAMATH_GPT_gcd_of_18_and_30_l754_75446

theorem gcd_of_18_and_30 : Nat.gcd 18 30 = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_18_and_30_l754_75446


namespace NUMINAMATH_GPT_four_point_questions_l754_75486

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 := 
sorry

end NUMINAMATH_GPT_four_point_questions_l754_75486


namespace NUMINAMATH_GPT_find_other_root_l754_75440

variables {a b c : ℝ}

theorem find_other_root
  (h_eq : ∀ x : ℝ, a * (b - c) * x^2 + b * (c - a) * x + c * (a - b) = 0)
  (root1 : a * (b - c) * 1^2 + b * (c - a) * 1 + c * (a - b) = 0) :
  ∃ k : ℝ, k = c * (a - b) / (a * (b - c)) ∧
           a * (b - c) * k^2 + b * (c - a) * k + c * (a - b) = 0 := 
sorry

end NUMINAMATH_GPT_find_other_root_l754_75440


namespace NUMINAMATH_GPT_base3_to_base10_conversion_l754_75449

theorem base3_to_base10_conversion : 
  1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0 = 142 :=
by {
  -- calculations
  sorry
}

end NUMINAMATH_GPT_base3_to_base10_conversion_l754_75449


namespace NUMINAMATH_GPT_branches_sum_one_main_stem_l754_75447

theorem branches_sum_one_main_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_branches_sum_one_main_stem_l754_75447


namespace NUMINAMATH_GPT_no_rational_satisfies_l754_75432

theorem no_rational_satisfies (a b c d : ℚ) : ¬ ((a + b * Real.sqrt 3)^4 + (c + d * Real.sqrt 3)^4 = 1 + Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_no_rational_satisfies_l754_75432


namespace NUMINAMATH_GPT_heartsuit_fraction_l754_75441

-- Define the operation heartsuit
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Define the proof statement
theorem heartsuit_fraction :
  (heartsuit 2 4) / (heartsuit 4 2) = 2 :=
by
  -- We use 'sorry' to skip the actual proof steps
  sorry

end NUMINAMATH_GPT_heartsuit_fraction_l754_75441


namespace NUMINAMATH_GPT_book_price_percentage_change_l754_75404

theorem book_price_percentage_change (P : ℝ) (x : ℝ) (h : P * (1 - (x / 100) ^ 2) = 0.90 * P) : x = 32 := by
sorry

end NUMINAMATH_GPT_book_price_percentage_change_l754_75404


namespace NUMINAMATH_GPT_sum_of_mnp_l754_75477

theorem sum_of_mnp (m n p : ℕ) (h_gcd : gcd m (gcd n p) = 1)
  (h : ∀ x : ℝ, 5 * x^2 - 11 * x + 6 = 0 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_mnp_l754_75477


namespace NUMINAMATH_GPT_three_digit_numbers_l754_75470

theorem three_digit_numbers (n : ℕ) (a b c : ℕ) (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : n = 100 * a + 10 * b + c)
  (h3 : b^2 = a * c)
  (h4 : (10 * b + c) % 4 = 0) :
  n = 124 ∨ n = 248 ∨ n = 444 ∨ n = 964 ∨ n = 888 :=
sorry

end NUMINAMATH_GPT_three_digit_numbers_l754_75470


namespace NUMINAMATH_GPT_work_ratio_l754_75466

theorem work_ratio (m b : ℝ) (h1 : 12 * m + 16 * b = 1 / 5) (h2 : 13 * m + 24 * b = 1 / 4) : m = 2 * b :=
by sorry

end NUMINAMATH_GPT_work_ratio_l754_75466


namespace NUMINAMATH_GPT_probability_same_color_ball_draw_l754_75458

theorem probability_same_color_ball_draw (red white : ℕ) 
    (h_red : red = 2) (h_white : white = 2) : 
    let total_outcomes := (red + white) * (red + white)
    let same_color_outcomes := 2 * (red * red + white * white)
    same_color_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_same_color_ball_draw_l754_75458


namespace NUMINAMATH_GPT_find_current_l754_75423

noncomputable def V : ℂ := 2 + 3 * Complex.I
noncomputable def Z : ℂ := 2 - 2 * Complex.I

theorem find_current : (V / Z) = (-1 / 4 : ℂ) + (5 / 4 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_GPT_find_current_l754_75423


namespace NUMINAMATH_GPT_train_speed_in_kmh_l754_75489

theorem train_speed_in_kmh (length_of_train : ℕ) (time_to_cross : ℕ) (speed_in_m_per_s : ℕ) (speed_in_km_per_h : ℕ) :
  length_of_train = 300 →
  time_to_cross = 12 →
  speed_in_m_per_s = length_of_train / time_to_cross →
  speed_in_km_per_h = speed_in_m_per_s * 3600 / 1000 →
  speed_in_km_per_h = 90 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_in_kmh_l754_75489


namespace NUMINAMATH_GPT_right_angles_in_2_days_l754_75478

-- Definitions
def hands_right_angle_twice_a_day (n : ℕ) : Prop :=
  n = 22

def right_angle_12_hour_frequency : Nat := 22
def hours_per_day : Nat := 24
def days : Nat := 2

-- Theorem to prove
theorem right_angles_in_2_days :
  hands_right_angle_twice_a_day right_angle_12_hour_frequency →
  right_angle_12_hour_frequency * (hours_per_day / 12) * days = 88 :=
by
  unfold hands_right_angle_twice_a_day
  intros 
  sorry

end NUMINAMATH_GPT_right_angles_in_2_days_l754_75478


namespace NUMINAMATH_GPT_elmo_to_laura_books_ratio_l754_75429

-- Definitions of the conditions given in the problem
def ElmoBooks : ℕ := 24
def StuBooks : ℕ := 4
def LauraBooks : ℕ := 2 * StuBooks

-- Ratio calculation and proof of the ratio being 3:1
theorem elmo_to_laura_books_ratio : (ElmoBooks : ℚ) / (LauraBooks : ℚ) = 3 / 1 := by
  sorry

end NUMINAMATH_GPT_elmo_to_laura_books_ratio_l754_75429


namespace NUMINAMATH_GPT_cheryl_more_points_l754_75425

-- Define the number of each type of eggs each child found
def kevin_small_eggs : Nat := 5
def kevin_large_eggs : Nat := 3

def bonnie_small_eggs : Nat := 13
def bonnie_medium_eggs : Nat := 7
def bonnie_large_eggs : Nat := 2

def george_small_eggs : Nat := 9
def george_medium_eggs : Nat := 6
def george_large_eggs : Nat := 1

def cheryl_small_eggs : Nat := 56
def cheryl_medium_eggs : Nat := 30
def cheryl_large_eggs : Nat := 15

-- Define the points for each type of egg
def small_egg_points : Nat := 1
def medium_egg_points : Nat := 3
def large_egg_points : Nat := 5

-- Calculate the total points for each child
def kevin_points : Nat := kevin_small_eggs * small_egg_points + kevin_large_eggs * large_egg_points
def bonnie_points : Nat := bonnie_small_eggs * small_egg_points + bonnie_medium_eggs * medium_egg_points + bonnie_large_eggs * large_egg_points
def george_points : Nat := george_small_eggs * small_egg_points + george_medium_eggs * medium_egg_points + george_large_eggs * large_egg_points
def cheryl_points : Nat := cheryl_small_eggs * small_egg_points + cheryl_medium_eggs * medium_egg_points + cheryl_large_eggs * large_egg_points

-- Statement of the proof problem
theorem cheryl_more_points : cheryl_points - (kevin_points + bonnie_points + george_points) = 125 :=
by
  -- Here would go the proof steps
  sorry

end NUMINAMATH_GPT_cheryl_more_points_l754_75425


namespace NUMINAMATH_GPT_inequality_abc_l754_75442

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (b * c / a) + (a * c / b) + (a * b / c) ≥ a + b + c := 
  sorry

end NUMINAMATH_GPT_inequality_abc_l754_75442


namespace NUMINAMATH_GPT_tens_digit_of_11_pow_12_pow_13_l754_75439

theorem tens_digit_of_11_pow_12_pow_13 :
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  tens_digit = 2 :=
by 
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  show tens_digit = 2
  sorry

end NUMINAMATH_GPT_tens_digit_of_11_pow_12_pow_13_l754_75439


namespace NUMINAMATH_GPT_matrix_multiplication_l754_75473

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_multiplication :
  (A - B = A * B) →
  (A * B = ![![7, -2], ![4, -3]]) →
  (B * A = ![![6, -2], ![4, -4]]) :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_matrix_multiplication_l754_75473


namespace NUMINAMATH_GPT_square_triangle_ratios_l754_75416

theorem square_triangle_ratios (s t : ℝ) 
  (P_s := 4 * s) 
  (R_s := s * Real.sqrt 2 / 2)
  (P_t := 3 * t) 
  (R_t := t * Real.sqrt 3 / 3) 
  (h : s = t) : 
  (P_s / P_t = 4 / 3) ∧ (R_s / R_t = Real.sqrt 6 / 2) := 
by
  sorry

end NUMINAMATH_GPT_square_triangle_ratios_l754_75416


namespace NUMINAMATH_GPT_angle_of_isosceles_trapezoid_in_monument_l754_75483

-- Define the larger interior angle x of an isosceles trapezoid in the monument
def larger_interior_angle_of_trapezoid (x : ℝ) : Prop :=
  ∃ n : ℕ, 
    n = 12 ∧
    ∃ α : ℝ, 
      α = 360 / (2 * n) ∧
      ∃ θ : ℝ, 
        θ = (180 - α) / 2 ∧
        x = 180 - θ

-- The theorem stating the larger interior angle x is 97.5 degrees
theorem angle_of_isosceles_trapezoid_in_monument : larger_interior_angle_of_trapezoid 97.5 :=
by 
  sorry

end NUMINAMATH_GPT_angle_of_isosceles_trapezoid_in_monument_l754_75483


namespace NUMINAMATH_GPT_find_a_l754_75436

theorem find_a (a : ℝ) (h : -1 ^ 2 + 2 * -1 + a = 0) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l754_75436


namespace NUMINAMATH_GPT_find_n_l754_75459

theorem find_n (m n : ℕ) (h1: m = 34)
               (h2: (1^(m+1) / 5^(m+1)) * (1^n / 4^n) = 1 / (2 * 10^35)) : 
               n = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l754_75459


namespace NUMINAMATH_GPT_sqrt_sum_of_fractions_as_fraction_l754_75402

theorem sqrt_sum_of_fractions_as_fraction :
  (Real.sqrt ((36 / 49) + (16 / 9) + (1 / 16))) = (45 / 28) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_of_fractions_as_fraction_l754_75402


namespace NUMINAMATH_GPT_count_divisibles_in_range_l754_75469

theorem count_divisibles_in_range :
  let lower_bound := (2:ℤ)^10
  let upper_bound := (2:ℤ)^18
  let divisor := (2:ℤ)^9 
  (upper_bound - lower_bound) / divisor + 1 = 511 :=
by 
  sorry

end NUMINAMATH_GPT_count_divisibles_in_range_l754_75469


namespace NUMINAMATH_GPT_speed_of_stream_l754_75499

variable (D : ℝ) -- The distance rowed in both directions
variable (vs : ℝ) -- The speed of the stream
variable (Vb : ℝ := 78) -- The speed of the boat in still water

theorem speed_of_stream (h : (D / (Vb - vs) = 2 * (D / (Vb + vs)))) : vs = 26 := by
    sorry

end NUMINAMATH_GPT_speed_of_stream_l754_75499


namespace NUMINAMATH_GPT_standard_ellipse_eq_l754_75462

def ellipse_standard_eq (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem standard_ellipse_eq (P: ℝ × ℝ) (Q: ℝ × ℝ) (a b : ℝ) (h1 : P = (-3, 0)) (h2 : Q = (0, -2)) :
  ellipse_standard_eq 3 2 x y :=
by
  sorry

end NUMINAMATH_GPT_standard_ellipse_eq_l754_75462


namespace NUMINAMATH_GPT_complex_problem_l754_75474

theorem complex_problem (a b : ℝ) (i : ℂ) (hi : i^2 = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_problem_l754_75474


namespace NUMINAMATH_GPT_circle_center_and_radius_l754_75421

theorem circle_center_and_radius (x y : ℝ) : 
  (x^2 + y^2 - 6 * x = 0) → ((x - 3)^2 + (y - 0)^2 = 9) :=
by
  intro h
  -- The proof is left as an exercise.
  sorry

end NUMINAMATH_GPT_circle_center_and_radius_l754_75421


namespace NUMINAMATH_GPT_factor_expression_l754_75495

theorem factor_expression (a b c : ℝ) :
    (a + b) * (b + c) * (c + a) = 
    ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l754_75495


namespace NUMINAMATH_GPT_smallest_fraction_l754_75413

theorem smallest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 3) (h2 : f2 = 3 / 4) (h3 : f3 = 5 / 6) 
  (h4 : f4 = 5 / 8) (h5 : f5 = 11 / 12) : f4 = 5 / 8 ∧ f4 < f1 ∧ f4 < f2 ∧ f4 < f3 ∧ f4 < f5 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_fraction_l754_75413


namespace NUMINAMATH_GPT_packages_of_noodles_tom_needs_l754_75454

def beef_weight : ℕ := 10
def noodles_needed_factor : ℕ := 2
def noodles_available : ℕ := 4
def noodle_package_weight : ℕ := 2

theorem packages_of_noodles_tom_needs :
  (beef_weight * noodles_needed_factor - noodles_available) / noodle_package_weight = 8 :=
by
  sorry

end NUMINAMATH_GPT_packages_of_noodles_tom_needs_l754_75454


namespace NUMINAMATH_GPT_no_valid_n_l754_75460

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def greatest_prime_factor (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n.minFac

theorem no_valid_n (n : ℕ) (h1 : n > 1)
  (h2 : is_prime (greatest_prime_factor n))
  (h3 : greatest_prime_factor n = Nat.sqrt n)
  (h4 : is_prime (greatest_prime_factor (n + 36)))
  (h5 : greatest_prime_factor (n + 36) = Nat.sqrt (n + 36)) :
  false :=
sorry

end NUMINAMATH_GPT_no_valid_n_l754_75460


namespace NUMINAMATH_GPT_gcd_840_1764_l754_75491

theorem gcd_840_1764 : Int.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_GPT_gcd_840_1764_l754_75491


namespace NUMINAMATH_GPT_car_race_probability_l754_75431

theorem car_race_probability :
  let pX := 1/8
  let pY := 1/12
  let pZ := 1/6
  pX + pY + pZ = 3/8 :=
by
  sorry

end NUMINAMATH_GPT_car_race_probability_l754_75431


namespace NUMINAMATH_GPT_three_character_license_plates_l754_75482

theorem three_character_license_plates :
  let consonants := 20
  let vowels := 6
  (consonants * consonants * vowels = 2400) :=
by
  sorry

end NUMINAMATH_GPT_three_character_license_plates_l754_75482


namespace NUMINAMATH_GPT_range_of_m_l754_75475

theorem range_of_m (p_false : ¬ (∀ x : ℝ, ∃ m : ℝ, 2 * x + 1 + m = 0)) : ∀ m : ℝ, m ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l754_75475


namespace NUMINAMATH_GPT_non_shaded_area_l754_75408

theorem non_shaded_area (r : ℝ) (A : ℝ) (shaded : ℝ) (non_shaded : ℝ) :
  (r = 5) ∧ (A = 4 * (π * r^2)) ∧ (shaded = 8 * (1 / 4 * π * r^2 - (1 / 2 * r * r))) ∧
  (non_shaded = A - shaded) → 
  non_shaded = 50 * π + 100 :=
by
  intro h
  obtain ⟨r_eq_5, A_eq, shaded_eq, non_shaded_eq⟩ := h
  rw [r_eq_5] at *
  sorry

end NUMINAMATH_GPT_non_shaded_area_l754_75408


namespace NUMINAMATH_GPT_positive_solution_is_perfect_square_l754_75468

theorem positive_solution_is_perfect_square
  (t : ℤ)
  (n : ℕ)
  (h : n > 0)
  (root_cond : (n : ℤ)^2 + (4 * t - 1) * n + 4 * t^2 = 0) :
  ∃ k : ℕ, n = k^2 :=
sorry

end NUMINAMATH_GPT_positive_solution_is_perfect_square_l754_75468


namespace NUMINAMATH_GPT_chemist_mixing_solution_l754_75445

theorem chemist_mixing_solution (x : ℝ) : 0.30 * x = 0.20 * (x + 1) → x = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_chemist_mixing_solution_l754_75445


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l754_75455

-- Problem 1
theorem problem1 (x : ℝ) : 0.75 * x = (1 / 2) * 12 → x = 8 := 
by 
  intro h
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (0.7 / x) = (14 / 5) → x = 0.25 := 
by 
  intro h
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (1 / 6) * x = (2 / 15) * (2 / 3) → x = (8 / 15) := 
by 
  intro h
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : 4.5 * x = 4 * 27 → x = 24 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l754_75455


namespace NUMINAMATH_GPT_simplify_fraction_l754_75410

theorem simplify_fraction (m : ℤ) : 
  let c := 2 
  let d := 4 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = (1 / 2 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l754_75410


namespace NUMINAMATH_GPT_find_direction_vector_l754_75405

def line_parametrization (v d : ℝ × ℝ) (t x y : ℝ) : ℝ × ℝ :=
  (v.fst + t * d.fst, v.snd + t * d.snd)

theorem find_direction_vector : 
  ∀ d: ℝ × ℝ, ∀ t: ℝ,
    ∀ (v : ℝ × ℝ) (x y : ℝ), 
    v = (-3, -1) → 
    y = (2 * x + 3) / 5 →
    x + 3 ≤ 0 →
    dist (line_parametrization v d t x y) (-3, -1) = t →
    d = (5/2, 1) :=
by
  intros d t v x y hv hy hcond hdist
  sorry

end NUMINAMATH_GPT_find_direction_vector_l754_75405


namespace NUMINAMATH_GPT_value_of_x_l754_75487

theorem value_of_x (x : ℝ) : (1 / 8) * (2 : ℝ) ^ 32 = (4 : ℝ) ^ x → x = 29 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l754_75487


namespace NUMINAMATH_GPT_sphere_volume_l754_75409

theorem sphere_volume (C : ℝ) (h : C = 30) : 
  ∃ (V : ℝ), V = 4500 / (π^2) :=
by sorry

end NUMINAMATH_GPT_sphere_volume_l754_75409


namespace NUMINAMATH_GPT_probability_of_same_color_l754_75424

-- Defining the given conditions
def green_balls := 6
def red_balls := 4
def total_balls := green_balls + red_balls

def probability_same_color : ℚ :=
  let prob_green := (green_balls / total_balls) * (green_balls / total_balls)
  let prob_red := (red_balls / total_balls) * (red_balls / total_balls)
  prob_green + prob_red

-- Statement of the problem rewritten in Lean 4
theorem probability_of_same_color :
  probability_same_color = 13 / 25 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_same_color_l754_75424


namespace NUMINAMATH_GPT_axis_of_symmetry_l754_75472

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) : 
  ∀ x : ℝ, f x = f (4 - x) := 
  by sorry

end NUMINAMATH_GPT_axis_of_symmetry_l754_75472


namespace NUMINAMATH_GPT_profit_percentage_l754_75498

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 78) :
  ((selling_price - cost_price) / cost_price) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l754_75498
