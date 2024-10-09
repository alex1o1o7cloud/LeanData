import Mathlib

namespace slope_of_asymptotes_l743_74334

noncomputable def hyperbola_asymptote_slope (x y : ℝ) : Prop :=
  (x^2 / 144 - y^2 / 81 = 1)

theorem slope_of_asymptotes (x y : ℝ) (h : hyperbola_asymptote_slope x y) :
  ∃ m : ℝ, m = 3 / 4 ∨ m = -3 / 4 :=
sorry

end slope_of_asymptotes_l743_74334


namespace movies_left_to_watch_l743_74302

theorem movies_left_to_watch (total_movies : ℕ) (movies_watched : ℕ) : total_movies = 17 ∧ movies_watched = 7 → (total_movies - movies_watched) = 10 :=
by
  sorry

end movies_left_to_watch_l743_74302


namespace arithmetic_sequence_sum_l743_74377

-- Definitions for the conditions
def a := 70
def d := 3
def n := 10
def l := 97

-- Sum of the arithmetic series
def S := (n / 2) * (a + l)

-- Final calculation
theorem arithmetic_sequence_sum :
  3 * (70 + 73 + 76 + 79 + 82 + 85 + 88 + 91 + 94 + 97) = 2505 :=
by
  -- Lean will calculate these interactively when proving.
  sorry

end arithmetic_sequence_sum_l743_74377


namespace final_population_correct_l743_74380

noncomputable def initialPopulation : ℕ := 300000
noncomputable def immigration : ℕ := 50000
noncomputable def emigration : ℕ := 30000

noncomputable def populationAfterImmigration : ℕ := initialPopulation + immigration
noncomputable def populationAfterEmigration : ℕ := populationAfterImmigration - emigration

noncomputable def pregnancies : ℕ := populationAfterEmigration / 8
noncomputable def twinPregnancies : ℕ := pregnancies / 4
noncomputable def singlePregnancies : ℕ := pregnancies - twinPregnancies

noncomputable def totalBirths : ℕ := twinPregnancies * 2 + singlePregnancies
noncomputable def finalPopulation : ℕ := populationAfterEmigration + totalBirths

theorem final_population_correct : finalPopulation = 370000 :=
by
  sorry

end final_population_correct_l743_74380


namespace circle_radius_l743_74346

-- Parameters of the problem
variables (k : ℝ) (r : ℝ)
-- Conditions
axiom cond_k_positive : k > 8
axiom tangency_y_8 : r = k - 8
axiom tangency_y_x : r = k / (Real.sqrt 2)

-- Statement to prove
theorem circle_radius (k : ℝ) (hk : k > 8) (r : ℝ) (hr1 : r = k - 8) (hr2 : r = k / (Real.sqrt 2)) : r = 8 * Real.sqrt 2 + 8 :=
sorry

end circle_radius_l743_74346


namespace sum_of_possible_values_of_k_l743_74300

open Complex

theorem sum_of_possible_values_of_k (x y z k : ℂ) (hxyz : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h : x / (1 - y + z) = k ∧ y / (1 - z + x) = k ∧ z / (1 - x + y) = k) : k = 1 :=
by
  sorry

end sum_of_possible_values_of_k_l743_74300


namespace david_chemistry_marks_l743_74382

theorem david_chemistry_marks :
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects
  chemistry = 97 :=
by
  -- Definition of variables
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects

  -- Assert the final value
  show chemistry = 97
  sorry

end david_chemistry_marks_l743_74382


namespace min_value_of_function_l743_74333

theorem min_value_of_function (x : ℝ) (h : x > 0) : (∃ y : ℝ, y = x^2 + 3 * x + 1 ∧ ∀ z, z = x^2 + 3 * x + 1 → y ≤ z) → y = 5 :=
by
  sorry

end min_value_of_function_l743_74333


namespace evaluate_expression_l743_74358

theorem evaluate_expression :
  -5^2 + 2 * (-3)^2 - (-8) / (-1 + 1/3) = -13 :=
by 
  sorry

end evaluate_expression_l743_74358


namespace units_digit_product_of_four_consecutive_integers_l743_74336

theorem units_digit_product_of_four_consecutive_integers (n : ℕ) (h : n % 2 = 1) : (n * (n + 1) * (n + 2) * (n + 3)) % 10 = 0 := 
by 
  sorry

end units_digit_product_of_four_consecutive_integers_l743_74336


namespace rectangular_prism_volume_dependency_l743_74337

theorem rectangular_prism_volume_dependency (a : ℝ) (V : ℝ) (h : a > 2) :
  V = a * 2 * 1 → (∀ a₀ > 2, a ≠ a₀ → V ≠ a₀ * 2 * 1) :=
by
  sorry

end rectangular_prism_volume_dependency_l743_74337


namespace remainder_9_5_4_6_5_7_mod_7_l743_74320

theorem remainder_9_5_4_6_5_7_mod_7 :
  ((9^5 + 4^6 + 5^7) % 7) = 2 :=
by sorry

end remainder_9_5_4_6_5_7_mod_7_l743_74320


namespace sum_of_digits_divisible_by_9_l743_74317

theorem sum_of_digits_divisible_by_9 (D E : ℕ) (hD : D < 10) (hE : E < 10) : 
  (D + E + 37) % 9 = 0 → ((D + E = 8) ∨ (D + E = 17)) →
  (8 + 17 = 25) := 
by
  intro h1 h2
  sorry

end sum_of_digits_divisible_by_9_l743_74317


namespace maria_profit_disks_l743_74385

theorem maria_profit_disks (cost_price_per_5 : ℝ) (sell_price_per_4 : ℝ) (desired_profit : ℝ) : 
  (cost_price_per_5 = 6) → (sell_price_per_4 = 8) → (desired_profit = 120) →
  (150 : ℝ) = desired_profit / ((sell_price_per_4 / 4) - (cost_price_per_5 / 5)) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end maria_profit_disks_l743_74385


namespace option_B_not_well_defined_l743_74331

-- Definitions based on given conditions 
def is_well_defined_set (description : String) : Prop :=
  match description with
  | "All positive numbers" => True
  | "All elderly people" => False
  | "All real numbers that are not equal to 0" => True
  | "The four great inventions of ancient China" => True
  | _ => False

-- Theorem stating option B "All elderly people" is not a well-defined set
theorem option_B_not_well_defined : ¬ is_well_defined_set "All elderly people" :=
  by sorry

end option_B_not_well_defined_l743_74331


namespace fraction_denominator_l743_74323

theorem fraction_denominator (x y Z : ℚ) (h : x / y = 7 / 3) (h2 : (x + y) / Z = 2.5) :
    Z = (4 * y) / 3 :=
by sorry

end fraction_denominator_l743_74323


namespace solve_for_z_l743_74347

theorem solve_for_z (i z : ℂ) (h0 : i^2 = -1) (h1 : i / z = 1 + i) : z = (1 + i) / 2 :=
by
  sorry

end solve_for_z_l743_74347


namespace a5_value_l743_74330

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Assume the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom sum_S6 : S 6 = 12
axiom term_a2 : a 2 = 5
axiom sum_formula (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Prove a5 is -1
theorem a5_value (h_arith : arithmetic_sequence a)
  (h_S6 : S 6 = 12) (h_a2 : a 2 = 5) (h_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 5 = -1 :=
sorry

end a5_value_l743_74330


namespace interest_rate_per_annum_l743_74366

theorem interest_rate_per_annum :
  ∃ (r : ℝ), 338 = 312.50 * (1 + r) ^ 2 :=
by
  sorry

end interest_rate_per_annum_l743_74366


namespace container_emptying_l743_74305

theorem container_emptying (a b c : ℕ) : ∃ m n k : ℕ,
  (m = 0 ∨ n = 0 ∨ k = 0) ∧
  (∀ a' b' c', 
    (a' = a ∧ b' = b ∧ c' = c) ∨ 
    (a' + 2 * b' = a' ∧ b' = b ∧ c' + 2 * b' = c') ∨ 
    (a' + 2 * c' = a' ∧ b' + 2 * c' = b' ∧ c' = c') ∨ 
    (a + 2 * b' + c' = a' + 2 * m * (a + b') ∧ b' = n * (a + b') ∧ c' = k * (a + b')) 
  -> (a' = 0 ∨ b' = 0 ∨ c' = 0)) :=
sorry

end container_emptying_l743_74305


namespace time_to_upload_file_l743_74318

-- Define the conditions
def file_size : ℕ := 160
def upload_speed : ℕ := 8

-- Define the question as a proof goal
theorem time_to_upload_file :
  file_size / upload_speed = 20 := 
sorry

end time_to_upload_file_l743_74318


namespace perpendicular_k_value_parallel_k_value_l743_74391

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 2)
def u (k : ℝ) : ℝ × ℝ := (k - 1, 2 * k + 2)
def v : ℝ × ℝ := (4, -4)

noncomputable def is_perpendicular (x y : ℝ × ℝ) : Prop :=
  x.1 * y.1 + x.2 * y.2 = 0

noncomputable def is_parallel (x y : ℝ × ℝ) : Prop :=
  x.1 * y.2 = x.2 * y.1

theorem perpendicular_k_value :
  is_perpendicular (u (-3)) v :=
by sorry

theorem parallel_k_value :
  is_parallel (u (-1/3)) v :=
by sorry

end perpendicular_k_value_parallel_k_value_l743_74391


namespace number_of_solutions_l743_74321

theorem number_of_solutions : ∃ n : ℕ, 1 < n ∧ 
  (∃ a b : ℕ, gcd a b = 1 ∧
  (∃ x y : ℕ, x^(a*n) + y^(b*n) = 2^2010)) ∧
  (∃ count : ℕ, count = 54) :=
sorry

end number_of_solutions_l743_74321


namespace negation_of_inequality_l743_74319

theorem negation_of_inequality :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 := 
sorry

end negation_of_inequality_l743_74319


namespace quadrilateral_area_lemma_l743_74386

-- Define the coordinates of the vertices
structure Point where
  x : ℤ
  y : ℤ

def A : Point := ⟨1, 3⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨2, 1⟩
def D : Point := ⟨2006, 2007⟩

-- Function to calculate the area of a quadrilateral given its vertices
def quadrilateral_area (A B C D : Point) : ℤ := 
  let triangle_area (P Q R : Point) : ℤ :=
    (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x) / 2
  triangle_area A B C + triangle_area A C D

-- The statement to be proved
theorem quadrilateral_area_lemma : quadrilateral_area A B C D = 3008 := 
  sorry

end quadrilateral_area_lemma_l743_74386


namespace cos_14_pi_over_3_l743_74327

theorem cos_14_pi_over_3 : Real.cos (14 * Real.pi / 3) = -1 / 2 :=
by 
  -- Proof is omitted according to the instructions
  sorry

end cos_14_pi_over_3_l743_74327


namespace valid_expression_l743_74389

theorem valid_expression (x : ℝ) : 
  (x - 1 ≥ 0 ∧ x - 2 ≠ 0) ↔ (x ≥ 1 ∧ x ≠ 2) := 
by
  sorry

end valid_expression_l743_74389


namespace jane_number_of_muffins_l743_74359

theorem jane_number_of_muffins 
    (m b c : ℕ) 
    (h1 : m + b + c = 6) 
    (h2 : b = 2) 
    (h3 : (50 * m + 75 * b + 65 * c) % 100 = 0) : 
    m = 4 := 
sorry

end jane_number_of_muffins_l743_74359


namespace simplify_and_evaluate_expression_l743_74361

theorem simplify_and_evaluate_expression :
  ∀ (x y : ℝ), 
  x = -1 / 3 → y = -2 → 
  (3 * x + 2 * y) * (3 * x - 2 * y) - 5 * x * (x - y) - (2 * x - y)^2 = -14 :=
by
  intros x y hx hy
  sorry

end simplify_and_evaluate_expression_l743_74361


namespace tony_comics_average_l743_74368

theorem tony_comics_average :
  let a1 := 10
  let d := 6
  let n := 8
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  (S_n n) / n = 31 := by
  sorry

end tony_comics_average_l743_74368


namespace find_number_l743_74310

theorem find_number (x : ℤ) (h : x = 1) : x + 1 = 2 :=
  by
  sorry

end find_number_l743_74310


namespace school_girls_more_than_boys_l743_74364

def num_initial_girls := 632
def num_initial_boys := 410
def num_new_girls := 465
def num_total_girls := num_initial_girls + num_new_girls
def num_difference_girls_boys := num_total_girls - num_initial_boys

theorem school_girls_more_than_boys :
  num_difference_girls_boys = 687 :=
by
  sorry

end school_girls_more_than_boys_l743_74364


namespace swimming_distance_l743_74369

theorem swimming_distance
  (t : ℝ) (d_up : ℝ) (d_down : ℝ) (v_man : ℝ) (v_stream : ℝ)
  (h1 : v_man = 5) (h2 : t = 5) (h3 : d_up = 20) 
  (h4 : d_up = (v_man - v_stream) * t) :
  d_down = (v_man + v_stream) * t :=
by
  sorry

end swimming_distance_l743_74369


namespace kids_prefer_peas_l743_74326

variable (total_kids children_prefer_carrots children_prefer_corn : ℕ)

theorem kids_prefer_peas (H1 : children_prefer_carrots = 9)
(H2 : children_prefer_corn = 5)
(H3 : children_prefer_corn * 4 = total_kids) :
total_kids - (children_prefer_carrots + children_prefer_corn) = 6 := by
sorry

end kids_prefer_peas_l743_74326


namespace average_income_A_B_l743_74322

def monthly_incomes (A B C : ℝ) : Prop :=
  (A = 4000) ∧
  ((B + C) / 2 = 6250) ∧
  ((A + C) / 2 = 5200)

theorem average_income_A_B (A B C X : ℝ) (h : monthly_incomes A B C) : X = 5050 :=
by
  have hA : A = 4000 := h.1
  have hBC : (B + C) / 2 = 6250 := h.2.1
  have hAC : (A + C) / 2 = 5200 := h.2.2
  sorry

end average_income_A_B_l743_74322


namespace david_marks_in_biology_l743_74312

theorem david_marks_in_biology (marks_english marks_math marks_physics marks_chemistry : ℕ)
  (average_marks num_subjects total_marks_known : ℕ)
  (h1 : marks_english = 76)
  (h2 : marks_math = 65)
  (h3 : marks_physics = 82)
  (h4 : marks_chemistry = 67)
  (h5 : average_marks = 75)
  (h6 : num_subjects = 5)
  (h7 : total_marks_known = marks_english + marks_math + marks_physics + marks_chemistry)
  (h8 : total_marks_known = 290)
  : ∃ biology_marks : ℕ, biology_marks = 85 ∧ biology_marks = (average_marks * num_subjects) - total_marks_known :=
by
  -- placeholder for proof
  sorry

end david_marks_in_biology_l743_74312


namespace sequence_formula_l743_74353

theorem sequence_formula (a : ℕ → ℚ) (h₁ : a 1 = 0)
  (h : ∀ n, a (n + 1) = 1 / (2 - a n)) :
  ∀ n, a n = (n - 1) / n :=
sorry

end sequence_formula_l743_74353


namespace time_to_cross_platform_l743_74303

-- Definitions based on the given conditions
def train_length : ℝ := 300
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 350

-- The question reformulated as a theorem in Lean 4
theorem time_to_cross_platform 
  (l_train : ℝ := train_length)
  (t_pole_cross : ℝ := time_to_cross_pole)
  (l_platform : ℝ := platform_length) :
  (l_train / t_pole_cross * (l_train + l_platform) = 39) :=
sorry

end time_to_cross_platform_l743_74303


namespace part1_part2_l743_74342

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x - 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := exp (a * x) * f x a + x

theorem part1 (a : ℝ) : 
  (a ≤ 0 → ∀ x, ∀ y, f x a ≤ y) ∧ (a > 0 → ∃ x, ∀ y, f x a ≤ y ∧ y = log (1 / a) - 2) :=
sorry

theorem part2 (a m : ℝ) (h_a : a > 0) (x1 x2 : ℝ) (h_x1 : 0 < x1) (h_x2 : x1 < x2) 
  (h_g1 : g x1 a = 0) (h_g2 : g x2 a = 0) : x1 * (x2 ^ 2) > exp m → m ≤ 3 :=
sorry

end part1_part2_l743_74342


namespace least_positive_integer_a_l743_74351

theorem least_positive_integer_a (a : ℕ) (n : ℕ) 
  (h1 : 2001 = 3 * 23 * 29)
  (h2 : 55 % 3 = 1)
  (h3 : 32 % 3 = -1)
  (h4 : 55 % 23 = 32 % 23)
  (h5 : 55 % 29 = -32 % 29)
  (h6 : n % 2 = 1)
  : a = 436 := 
sorry

end least_positive_integer_a_l743_74351


namespace sector_area_l743_74378

-- Define radius and central angle as conditions
def radius : ℝ := 1
def central_angle : ℝ := 2

-- Define the theorem to prove that the area of the sector is 1 cm² given the conditions
theorem sector_area : (1 / 2) * radius * central_angle = 1 := 
by 
  -- sorry is used to skip the actual proof
  sorry

end sector_area_l743_74378


namespace sum_partition_36_l743_74363

theorem sum_partition_36 : 
  ∃ (S : Finset ℕ), S.card = 36 ∧ S.sum id = ((Finset.range 72).sum id) / 2 :=
by
  sorry

end sum_partition_36_l743_74363


namespace expression_value_l743_74393

theorem expression_value :
  ∀ (x y : ℚ), (x = -5/4) → (y = -3/2) → -2 * x - y^2 = 1/4 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end expression_value_l743_74393


namespace find_x_l743_74370

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, 1)
def u : ℝ × ℝ := (1 + 2 * x, 4)
def v : ℝ × ℝ := (2 - 2 * x, 2)

theorem find_x (h : 2 * (1 + 2 * x) = 4 * (2 - 2 * x)) : x = 1 / 2 := by
  sorry

end find_x_l743_74370


namespace trajectory_of_point_l743_74371

theorem trajectory_of_point 
  (P : ℝ × ℝ) 
  (h1 : abs (P.1 - 4) + P.2^2 - 1 = abs (P.1 + 5)) : 
  P.2^2 = 16 * P.1 := 
sorry

end trajectory_of_point_l743_74371


namespace jori_remaining_water_l743_74367

-- Having the necessary libraries for arithmetic and fractions.

-- Definitions directly from the conditions in a).
def initial_water_quantity : ℚ := 4
def used_water_quantity : ℚ := 9 / 4 -- Converted 2 1/4 to an improper fraction

-- The statement proving the remaining quantity of water is 1 3/4 gallons.
theorem jori_remaining_water : initial_water_quantity - used_water_quantity = 7 / 4 := by
  sorry

end jori_remaining_water_l743_74367


namespace smallest_prime_factor_of_setC_l743_74307

def setC : Set ℕ := {51, 53, 54, 56, 57}

def prime_factors (n : ℕ) : Set ℕ :=
  { p | p.Prime ∧ p ∣ n }

theorem smallest_prime_factor_of_setC :
  (∃ n ∈ setC, ∀ m ∈ setC, ∀ p ∈ prime_factors n, ∀ q ∈ prime_factors m, p ≤ q) ∧
  (∃ m ∈ setC, ∀ p ∈ prime_factors 54, ∀ q ∈ prime_factors m, p = q) := 
sorry

end smallest_prime_factor_of_setC_l743_74307


namespace victor_earnings_l743_74375

def hourly_wage := 6 -- dollars per hour
def hours_monday := 5 -- hours
def hours_tuesday := 5 -- hours

theorem victor_earnings : (hourly_wage * (hours_monday + hours_tuesday)) = 60 :=
by
  sorry

end victor_earnings_l743_74375


namespace distance_traveled_eq_2400_l743_74325

-- Definitions of the conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 32
def revolutions_difference : ℕ := 5

-- Define the number of revolutions made by the back wheel
def revs_back (R : ℕ) := R

-- Define the number of revolutions made by the front wheel
def revs_front (R : ℕ) := R + revolutions_difference

-- Define the distance traveled by the back and front wheels
def distance_back (R : ℕ) : ℕ := revs_back R * circumference_back
def distance_front (R : ℕ) : ℕ := revs_front R * circumference_front

-- State the theorem without a proof (using sorry)
theorem distance_traveled_eq_2400 :
  ∃ R : ℕ, distance_back R = 2400 ∧ distance_back R = distance_front R :=
by {
  sorry
}

end distance_traveled_eq_2400_l743_74325


namespace min_value_inequality_l743_74379

theorem min_value_inequality (θ φ : ℝ) : 
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 549 - 140 * Real.sqrt 5 := 
by
  sorry

end min_value_inequality_l743_74379


namespace original_price_color_tv_l743_74329

theorem original_price_color_tv (x : ℝ) : 
  1.4 * x * 0.8 - x = 270 → x = 2250 :=
by
  intro h
  simp at h
  sorry

end original_price_color_tv_l743_74329


namespace dog_catches_fox_at_120m_l743_74328

theorem dog_catches_fox_at_120m :
  let initial_distance := 30
  let dog_leap := 2
  let fox_leap := 1
  let dog_leap_frequency := 2
  let fox_leap_frequency := 3
  let dog_distance_per_time_unit := dog_leap * dog_leap_frequency
  let fox_distance_per_time_unit := fox_leap * fox_leap_frequency
  let relative_closure_rate := dog_distance_per_time_unit - fox_distance_per_time_unit
  let time_units_to_catch := initial_distance / relative_closure_rate
  let total_dog_distance := time_units_to_catch * dog_distance_per_time_unit
  total_dog_distance = 120 := sorry

end dog_catches_fox_at_120m_l743_74328


namespace second_fisherman_more_fish_l743_74343

-- Define the given conditions
def days_in_season : ℕ := 213
def rate_first_fisherman : ℕ := 3
def rate_second_fisherman_phase_1 : ℕ := 1
def rate_second_fisherman_phase_2 : ℕ := 2
def rate_second_fisherman_phase_3 : ℕ := 4
def days_phase_1 : ℕ := 30
def days_phase_2 : ℕ := 60
def days_phase_3 : ℕ := days_in_season - (days_phase_1 + days_phase_2)

-- Define the total number of fish caught by each fisherman
def total_fish_first_fisherman : ℕ := rate_first_fisherman * days_in_season
def total_fish_second_fisherman : ℕ := 
  (rate_second_fisherman_phase_1 * days_phase_1) + 
  (rate_second_fisherman_phase_2 * days_phase_2) + 
  (rate_second_fisherman_phase_3 * days_phase_3)

-- Define the theorem statement
theorem second_fisherman_more_fish : 
  total_fish_second_fisherman = total_fish_first_fisherman + 3 := by sorry

end second_fisherman_more_fish_l743_74343


namespace Sally_quarters_l743_74306

theorem Sally_quarters : 760 + 418 - 152 = 1026 := 
by norm_num

end Sally_quarters_l743_74306


namespace sam_initial_dimes_l743_74341

theorem sam_initial_dimes (given_away : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : given_away = 7) (h2 : left = 2) (h3 : initial = given_away + left) : 
  initial = 9 := by
  rw [h1, h2] at h3
  exact h3

end sam_initial_dimes_l743_74341


namespace card_statements_are_false_l743_74388

theorem card_statements_are_false :
  ¬( ( (statements: ℕ) →
        (statements = 1 ↔ ¬statements = 1 ∧ ¬statements = 2 ∧ ¬statements = 3 ∧ ¬statements = 4 ∧ ¬statements = 5) ∧
        ( statements = 2 ↔ (statements = 1 ∨ statements = 3 ∨ statements = 4 ∨ statements = 5)) ∧
        (statements = 3 ↔ (statements = 1 ∧ statements = 2 ∧ (statements = 4 ∨ statements = 5) ) ) ∧
        (statements = 4 ↔ (statements = 1 ∧ statements = 2 ∧ statements = 3 ∧ statements != 5 ) ) ∧
        (statements = 5 ↔ (statements = 4 ) )
)) :=
sorry

end card_statements_are_false_l743_74388


namespace hyperbola_a_value_l743_74344

theorem hyperbola_a_value (a : ℝ) :
  (∀ x y : ℝ, (x^2 / (a + 3) - y^2 / 3 = 1)) ∧ 
  (∀ e : ℝ, e = 2) → 
  a = -2 :=
by sorry

end hyperbola_a_value_l743_74344


namespace no_solution_for_x_l743_74390

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (mx - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) :=
by
  sorry

end no_solution_for_x_l743_74390


namespace expression_evaluation_l743_74397

theorem expression_evaluation : 
  (50 - (2210 - 251)) + (2210 - (251 - 50)) = 100 := 
  by sorry

end expression_evaluation_l743_74397


namespace alice_commission_percentage_l743_74396

-- Definitions from the given problem
def basic_salary : ℝ := 240
def total_sales : ℝ := 2500
def savings : ℝ := 29
def savings_percentage : ℝ := 0.10

-- The target percentage we want to prove
def commission_percentage : ℝ := 0.02

-- The statement we aim to prove
theorem alice_commission_percentage :
  commission_percentage =
  (savings / savings_percentage - basic_salary) / total_sales := 
sorry

end alice_commission_percentage_l743_74396


namespace solution_set_of_inequality_l743_74308

theorem solution_set_of_inequality (a t : ℝ) (h1 : ∀ x : ℝ, x^2 - 2 * a * x + a > 0) : 
  a > 0 ∧ a < 1 → (a^(2*t + 1) < a^(t^2 + 2*t - 3) ↔ -2 < t ∧ t < 2) :=
by
  intro ha
  have h : (0 < a ∧ a < 1) := sorry
  exact sorry

end solution_set_of_inequality_l743_74308


namespace sufficient_but_not_necessary_l743_74398

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬((x < -1 ∨ x > 1) → (x < -1)) :=
by
  sorry

end sufficient_but_not_necessary_l743_74398


namespace find_integer_pairs_l743_74355

theorem find_integer_pairs (x y: ℤ) :
  x^2 - y^4 = 2009 → (x = 45 ∧ (y = 2 ∨ y = -2)) ∨ (x = -45 ∧ (y = 2 ∨ y = -2)) :=
by
  sorry

end find_integer_pairs_l743_74355


namespace find_value_of_expression_l743_74338

theorem find_value_of_expression (x : ℝ) (h : 5 * x^2 + 4 = 3 * x + 9) : (10 * x - 3)^2 = 109 := 
sorry

end find_value_of_expression_l743_74338


namespace find_a_l743_74349

noncomputable def star (a b : ℝ) := a * (a + b) + b

theorem find_a (a : ℝ) (h : star a 2.5 = 28.5) : a = 4 ∨ a = -13/2 := 
sorry

end find_a_l743_74349


namespace expand_expression_l743_74350

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l743_74350


namespace find_a20_l743_74356

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom a_arithmetic : ∀ n, a (n + 1) = a 1 + n * d
axiom a1_a3_a5_eq_105 : a 1 + a 3 + a 5 = 105
axiom a2_a4_a6_eq_99 : a 2 + a 4 + a 6 = 99

theorem find_a20 : a 20 = 1 :=
by sorry

end find_a20_l743_74356


namespace not_monotonic_in_interval_l743_74384

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a * x - 5

theorem not_monotonic_in_interval (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f a x ≠ (1/3) * x^3 - x^2 + a * x - 5) → a ≥ 1 ∨ a ≤ -3 :=
sorry

end not_monotonic_in_interval_l743_74384


namespace maximum_black_squares_l743_74316

theorem maximum_black_squares (n : ℕ) (h : n ≥ 2) : 
  (n % 2 = 0 → ∃ b : ℕ, b = (n^2 - 4) / 2) ∧ 
  (n % 2 = 1 → ∃ b : ℕ, b = (n^2 - 1) / 2) := 
by sorry

end maximum_black_squares_l743_74316


namespace number_of_honey_bees_l743_74315

theorem number_of_honey_bees (total_honey : ℕ) (honey_one_bee : ℕ) (days : ℕ) (h1 : total_honey = 30) (h2 : honey_one_bee = 1) (h3 : days = 30) : 
  (total_honey / honey_one_bee) = 30 :=
by
  -- Given total_honey = 30 grams in 30 days
  -- Given honey_one_bee = 1 gram in 30 days
  -- We need to prove (total_honey / honey_one_bee) = 30
  sorry

end number_of_honey_bees_l743_74315


namespace least_positive_integer_solution_l743_74372

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 2 [MOD 3] ∧ b ≡ 3 [MOD 4] ∧ b ≡ 4 [MOD 5] ∧ b ≡ 8 [MOD 9] ∧ b = 179 :=
by
  sorry

end least_positive_integer_solution_l743_74372


namespace power_sums_l743_74332

-- Definitions as per the given conditions
variables (m n a b : ℕ)
variables (hm : 0 < m) (hn : 0 < n)
variables (ha : 2^m = a) (hb : 2^n = b)

-- The theorem statement
theorem power_sums (hmn : 0 < m + n) : 2^(m + n) = a * b :=
by
  sorry

end power_sums_l743_74332


namespace pseudoprime_pow_minus_one_l743_74381

theorem pseudoprime_pow_minus_one (n : ℕ) (hpseudo : 2^n ≡ 2 [MOD n]) : 
  ∃ m : ℕ, 2^(2^n - 1) ≡ 1 [MOD (2^n - 1)] :=
by
  sorry

end pseudoprime_pow_minus_one_l743_74381


namespace smallest_N_triangle_ineq_l743_74354

theorem smallest_N_triangle_ineq (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c < a + b) : (a^2 + b^2 + a * b) / c^2 < 1 := 
sorry

end smallest_N_triangle_ineq_l743_74354


namespace find_solution_set_l743_74304

noncomputable def is_solution (x : ℝ) : Prop :=
(1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < 1 / 4

theorem find_solution_set :
  { x : ℝ | is_solution x } = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end find_solution_set_l743_74304


namespace find_m_l743_74376

def U : Set Nat := {1, 2, 3}
def A (m : Nat) : Set Nat := {1, m}
def complement (s t : Set Nat) : Set Nat := {x | x ∈ s ∧ x ∉ t}

theorem find_m (m : Nat) (h1 : complement U (A m) = {2}) : m = 3 :=
by
  sorry

end find_m_l743_74376


namespace sum_arithmetic_sequence_l743_74301

theorem sum_arithmetic_sequence :
  let a : ℤ := -25
  let d : ℤ := 4
  let a_n : ℤ := 19
  let n : ℤ := (a_n - a) / d + 1
  let S : ℤ := n * (a + a_n) / 2
  S = -36 :=
by 
  let a := -25
  let d := 4
  let a_n := 19
  let n := (a_n - a) / d + 1
  let S := n * (a + a_n) / 2
  show S = -36
  sorry

end sum_arithmetic_sequence_l743_74301


namespace total_games_played_l743_74387

-- Define the number of teams and games per matchup condition
def num_teams : ℕ := 10
def games_per_matchup : ℕ := 5

-- Calculate total games played during the season
theorem total_games_played : 
  5 * ((num_teams * (num_teams - 1)) / 2) = 225 := by 
  sorry

end total_games_played_l743_74387


namespace question1_question2_question3_l743_74383

def f : Nat → Nat → Nat := sorry

axiom condition1 : f 1 1 = 1
axiom condition2 : ∀ m n, f m (n + 1) = f m n + 2
axiom condition3 : ∀ m, f (m + 1) 1 = 2 * f m 1

theorem question1 (n : Nat) : f 1 n = 2 * n - 1 :=
sorry

theorem question2 (m : Nat) : f m 1 = 2 ^ (m - 1) :=
sorry

theorem question3 : f 2002 9 = 2 ^ 2001 + 16 :=
sorry

end question1_question2_question3_l743_74383


namespace packets_of_chips_l743_74309

theorem packets_of_chips (x : ℕ) 
  (h1 : ∀ x, 2 * (x : ℝ) + 1.5 * (10 : ℝ) = 45) : 
  x = 15 := 
by 
  sorry

end packets_of_chips_l743_74309


namespace sampled_individual_l743_74365

theorem sampled_individual {population_size sample_size : ℕ} (population_size_cond : population_size = 1000)
  (sample_size_cond : sample_size = 20) (sampled_number : ℕ) (sampled_number_cond : sampled_number = 15) :
  (∃ n : ℕ, sampled_number + n * (population_size / sample_size) = 65) :=
by 
  sorry

end sampled_individual_l743_74365


namespace alice_minimum_speed_exceed_l743_74392

-- Define the conditions

def distance_ab : ℕ := 30  -- Distance from city A to city B is 30 miles
def speed_bob : ℕ := 40    -- Bob's constant speed is 40 miles per hour
def bob_travel_time := distance_ab / speed_bob  -- Bob's travel time in hours
def alice_travel_time := bob_travel_time - (1 / 2)  -- Alice leaves 0.5 hours after Bob

-- Theorem stating the minimum speed Alice must exceed
theorem alice_minimum_speed_exceed : ∃ v : Real, v > 60 ∧ distance_ab / alice_travel_time ≤ v := sorry

end alice_minimum_speed_exceed_l743_74392


namespace Pythagorean_triple_l743_74373

theorem Pythagorean_triple (n : ℕ) (hn : n % 2 = 1) (hn_geq : n ≥ 3) :
  n^2 + ((n^2 - 1) / 2)^2 = ((n^2 + 1) / 2)^2 := by
  sorry

end Pythagorean_triple_l743_74373


namespace total_number_of_plugs_l743_74311

variables (pairs_mittens pairs_plugs : ℕ)

-- Conditions
def initial_pairs_mittens : ℕ := 150
def initial_pairs_plugs : ℕ := initial_pairs_mittens + 20
def added_pairs_plugs : ℕ := 30
def total_pairs_plugs : ℕ := initial_pairs_plugs + added_pairs_plugs

-- The proposition we're going to prove:
theorem total_number_of_plugs : initial_pairs_mittens = 150 ∧ initial_pairs_plugs = initial_pairs_mittens + 20 ∧ added_pairs_plugs = 30 → 
  total_pairs_plugs * 2 = 400 := sorry

end total_number_of_plugs_l743_74311


namespace cookie_difference_l743_74313

def AlyssaCookies : ℕ := 129
def AiyannaCookies : ℕ := 140
def Difference : ℕ := 11

theorem cookie_difference : AiyannaCookies - AlyssaCookies = Difference := by
  sorry

end cookie_difference_l743_74313


namespace not_converge_to_a_l743_74362

theorem not_converge_to_a (x : ℕ → ℝ) (a : ℝ) :
  (∀ ε > 0, ∀ k : ℕ, ∃ n : ℕ, n > k ∧ |x n - a| ≥ ε) →
  ¬ (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - a| < ε) :=
by sorry

end not_converge_to_a_l743_74362


namespace sum_geom_seq_nine_l743_74340

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem sum_geom_seq_nine {a : ℕ → ℝ} {q : ℝ} (h_geom : geom_seq a q)
  (h1 : a 1 * (1 + q + q^2) = 30) 
  (h2 : a 4 * (1 + q + q^2) = 120) :
  a 7 + a 8 + a 9 = 480 :=
  sorry

end sum_geom_seq_nine_l743_74340


namespace bobs_total_profit_l743_74399

-- Definitions of the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Definition of the problem statement
theorem bobs_total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end bobs_total_profit_l743_74399


namespace inequality_pos_reals_l743_74374

theorem inequality_pos_reals (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) : 
  (x^2 + 2) * (y^2 + 2) * (z^2 + 2) ≥ 9 * (x * y + y * z + z * x) :=
by
  sorry

end inequality_pos_reals_l743_74374


namespace sum_of_roots_of_P_is_8029_l743_74339

-- Define the polynomial
noncomputable def P : Polynomial ℚ :=
  (Polynomial.X - 1)^2008 + 
  3 * (Polynomial.X - 2)^2007 + 
  5 * (Polynomial.X - 3)^2006 + 
  -- Continue defining all terms up to:
  2009 * (Polynomial.X - 2008)^2 + 
  2011 * (Polynomial.X - 2009)

-- The proof problem statement
theorem sum_of_roots_of_P_is_8029 :
  (P.roots.sum = 8029) :=
sorry

end sum_of_roots_of_P_is_8029_l743_74339


namespace determine_coefficients_l743_74395

theorem determine_coefficients (a b c : ℝ) (x y : ℝ) :
  (x = 3/4 ∧ y = 5/8) →
  (a * (x - 1) + 2 * y = 1) →
  (b * |x - 1| + c * y = 3) →
  (a = 1 ∧ b = 2 ∧ c = 4) := 
by 
  intros 
  sorry

end determine_coefficients_l743_74395


namespace find_triangle_base_l743_74314

theorem find_triangle_base (left_side : ℝ) (right_side : ℝ) (base : ℝ) 
  (h_left : left_side = 12) 
  (h_right : right_side = left_side + 2)
  (h_sum : left_side + right_side + base = 50) :
  base = 24 := 
sorry

end find_triangle_base_l743_74314


namespace book_chapters_not_determinable_l743_74357

variable (pages_initially pages_later pages_total total_pages book_chapters : ℕ)

def problem_statement : Prop :=
  pages_initially = 37 ∧ pages_later = 25 ∧ pages_total = 62 ∧ total_pages = 95 ∧ book_chapters = 0

theorem book_chapters_not_determinable (h: problem_statement pages_initially pages_later pages_total total_pages book_chapters) :
  book_chapters = 0 :=
by
  sorry

end book_chapters_not_determinable_l743_74357


namespace find_number_l743_74335

theorem find_number (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 :=
by
  sorry

end find_number_l743_74335


namespace max_xy_l743_74324

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy ≤ 81 :=
by sorry

end max_xy_l743_74324


namespace train_length_l743_74360

theorem train_length (L : ℝ) (v1 v2 : ℝ) 
  (h1 : v1 = (L + 140) / 15)
  (h2 : v2 = (L + 250) / 20) 
  (h3 : v1 = v2) :
  L = 190 :=
by sorry

end train_length_l743_74360


namespace fraction_sum_of_lcm_and_gcd_l743_74352

theorem fraction_sum_of_lcm_and_gcd 
  (m n : ℕ) 
  (h_gcd : Nat.gcd m n = 6) 
  (h_lcm : Nat.lcm m n = 210) 
  (h_sum : m + n = 72) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 12 / 210 := 
by
sorry

end fraction_sum_of_lcm_and_gcd_l743_74352


namespace find_initial_amount_l743_74345

-- Definitions for conditions
def final_amount : ℝ := 5565
def rate_year1 : ℝ := 0.05
def rate_year2 : ℝ := 0.06

-- Theorem statement to prove the initial amount
theorem find_initial_amount (P : ℝ) 
  (H : final_amount = (P * (1 + rate_year1)) * (1 + rate_year2)) :
  P = 5000 := 
sorry

end find_initial_amount_l743_74345


namespace initial_integers_is_three_l743_74348

def num_initial_integers (n m : Int) : Prop :=
  3 * n + m = 17 ∧ 2 * m + n = 23

theorem initial_integers_is_three {n m : Int} (h : num_initial_integers n m) : n = 3 :=
by
  sorry

end initial_integers_is_three_l743_74348


namespace stick_segments_l743_74394

theorem stick_segments (L : ℕ) (L_nonzero : L > 0) :
  let red_segments := 8
  let blue_segments := 12
  let black_segments := 18
  let total_segments := (red_segments + blue_segments + black_segments) 
                       - (lcm red_segments blue_segments / blue_segments) 
                       - (lcm blue_segments black_segments / black_segments)
                       - (lcm red_segments black_segments / black_segments)
                       + (lcm red_segments (lcm blue_segments black_segments) / (lcm blue_segments black_segments))
  let shortest_segment_length := L / lcm red_segments (lcm blue_segments black_segments)
  (total_segments = 28) ∧ (shortest_segment_length = L / 72) := by
  sorry

end stick_segments_l743_74394
