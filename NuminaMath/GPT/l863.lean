import Mathlib

namespace NUMINAMATH_GPT_y_less_than_z_by_40_percent_l863_86312

variable {x y z : ℝ}

theorem y_less_than_z_by_40_percent (h1 : x = 1.3 * y) (h2 : x = 0.78 * z) : y = 0.6 * z :=
by
  -- The proof will be provided here
  -- We are demonstrating that y = 0.6 * z is a consequence of h1 and h2
  sorry

end NUMINAMATH_GPT_y_less_than_z_by_40_percent_l863_86312


namespace NUMINAMATH_GPT_tournament_committees_count_l863_86395

-- Definitions corresponding to the conditions
def num_teams : ℕ := 4
def team_size : ℕ := 8
def members_selected_by_winning_team : ℕ := 3
def members_selected_by_other_teams : ℕ := 2

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Counting the number of possible committees
def total_committees : ℕ :=
  let num_ways_winning_team := binom team_size members_selected_by_winning_team
  let num_ways_other_teams := binom team_size members_selected_by_other_teams
  num_teams * num_ways_winning_team * (num_ways_other_teams ^ (num_teams - 1))

-- The statement to be proved
theorem tournament_committees_count : total_committees = 4917248 := by
  sorry

end NUMINAMATH_GPT_tournament_committees_count_l863_86395


namespace NUMINAMATH_GPT_find_a10_l863_86361

theorem find_a10 (a_n : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h2 : 5 * a_n 3 = a_n 3 ^ 2)
  (h3 : (a_n 3 + 2 * d) ^ 2 = (a_n 3 - d) * (a_n 3 + 11 * d))
  (h_nonzero : d ≠ 0) :
  a_n 10 = 23 :=
sorry

end NUMINAMATH_GPT_find_a10_l863_86361


namespace NUMINAMATH_GPT_angle_ABD_30_degrees_l863_86372

theorem angle_ABD_30_degrees (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB BD : ℝ) (angle_DBC : ℝ)
  (h1 : BD = AB * (Real.sqrt 3 / 2))
  (h2 : angle_DBC = 90) : 
  ∃ angle_ABD, angle_ABD = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_ABD_30_degrees_l863_86372


namespace NUMINAMATH_GPT_Nancy_hourly_wage_l863_86385

def tuition_cost := 22000
def parents_coverage := tuition_cost / 2
def scholarship := 3000
def loan := 2 * scholarship
def working_hours := 200
def remaining_tuition := tuition_cost - parents_coverage - scholarship - loan
def hourly_wage_required := remaining_tuition / working_hours

theorem Nancy_hourly_wage : hourly_wage_required = 10 := by
  sorry

end NUMINAMATH_GPT_Nancy_hourly_wage_l863_86385


namespace NUMINAMATH_GPT_find_number_l863_86347

theorem find_number (x : ℝ) (h : (5 / 6) * x = (5 / 16) * x + 300) : x = 576 :=
sorry

end NUMINAMATH_GPT_find_number_l863_86347


namespace NUMINAMATH_GPT_general_formula_for_sequence_l863_86331

noncomputable def S := ℕ → ℚ
noncomputable def a := ℕ → ℚ

theorem general_formula_for_sequence (a : a) (S : S) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, S (n + 1) = (2 / 3) * a (n + 1) + 1 / 3) :
  ∀ n : ℕ, a n = 
  if n = 1 then 2 
  else -5 * (-2)^(n-2) := 
by 
  sorry

end NUMINAMATH_GPT_general_formula_for_sequence_l863_86331


namespace NUMINAMATH_GPT_abc_zero_l863_86330

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) 
  : a * b * c = 0 := by
  sorry

end NUMINAMATH_GPT_abc_zero_l863_86330


namespace NUMINAMATH_GPT_solve_system_l863_86357

theorem solve_system (x y z u : ℝ) :
  x^3 * y^2 * z = 2 ∧
  z^3 * u^2 * x = 32 ∧
  y^3 * z^2 * u = 8 ∧
  u^3 * x^2 * y = 8 →
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ u = 2) ∨
  (x = 1 ∧ y = -1 ∧ z = 2 ∧ u = -2) ∨
  (x = -1 ∧ y = 1 ∧ z = -2 ∧ u = 2) ∨
  (x = -1 ∧ y = -1 ∧ z = -2 ∧ u = -2) :=
sorry

end NUMINAMATH_GPT_solve_system_l863_86357


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l863_86387

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l863_86387


namespace NUMINAMATH_GPT_reams_paper_l863_86319

theorem reams_paper (total_reams reams_haley reams_sister : Nat) 
    (h1 : total_reams = 5)
    (h2 : reams_haley = 2)
    (h3 : total_reams = reams_haley + reams_sister) : 
    reams_sister = 3 := by
  sorry

end NUMINAMATH_GPT_reams_paper_l863_86319


namespace NUMINAMATH_GPT_total_legs_arms_proof_l863_86394

/-
There are 4 birds, each with 2 legs.
There are 6 dogs, each with 4 legs.
There are 5 snakes, each with no legs.
There are 2 spiders, each with 8 legs.
There are 3 horses, each with 4 legs.
There are 7 rabbits, each with 4 legs.
There are 2 octopuses, each with 8 arms.
There are 8 ants, each with 6 legs.
There is 1 unique creature with 12 legs.
We need to prove that the total number of legs and arms is 164.
-/

def total_legs_arms : Nat := 
  (4 * 2) + (6 * 4) + (5 * 0) + (2 * 8) + (3 * 4) + (7 * 4) + (2 * 8) + (8 * 6) + (1 * 12)

theorem total_legs_arms_proof : total_legs_arms = 164 := by
  sorry

end NUMINAMATH_GPT_total_legs_arms_proof_l863_86394


namespace NUMINAMATH_GPT_part1_part2_part3_l863_86335

-- Part 1
theorem part1 :
  ∀ x : ℝ, (4 * x - 3 = 1) → (x = 1) ↔ 
    (¬(x - 3 > 3 * x - 1) ∧ (4 * (x - 1) ≤ 2) ∧ (x + 2 > 0 ∧ 3 * x - 3 ≤ 1)) :=
by sorry

-- Part 2
theorem part2 :
  ∀ (m n q : ℝ), (m + 2 * n = 6) → (2 * m + n = 3 * q) → (m + n > 1) → q > -1 :=
by sorry

-- Part 3
theorem part3 :
  ∀ (k m n : ℝ), (k < 3) → (∃ x : ℝ, (3 * (x - 1) = k) ∧ (4 * x + n < x + 2 * m)) → 
    (m + n ≥ 0) → (∃! n : ℝ, ∀ x : ℝ, (2 ≤ m ∧ m < 5 / 2)) :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l863_86335


namespace NUMINAMATH_GPT_largest_multiple_of_9_less_than_100_l863_86371

theorem largest_multiple_of_9_less_than_100 : ∃ n : ℕ, n * 9 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ n * 9 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_9_less_than_100_l863_86371


namespace NUMINAMATH_GPT_max_rock_value_l863_86397

/-- Carl discovers a cave with three types of rocks:
    - 6-pound rocks worth $16 each,
    - 3-pound rocks worth $9 each,
    - 2-pound rocks worth $3 each.
    There are at least 15 of each type.
    He can carry a maximum of 20 pounds and no more than 5 rocks in total.
    Prove that the maximum value, in dollars, of the rocks he can carry is $52. -/
theorem max_rock_value :
  ∃ (max_value: ℕ),
  (∀ (c6 c3 c2: ℕ),
    (c6 + c3 + c2 ≤ 5) ∧
    (6 * c6 + 3 * c3 + 2 * c2 ≤ 20) →
    max_value ≥ 16 * c6 + 9 * c3 + 3 * c2) ∧
  max_value = 52 :=
by
  sorry

end NUMINAMATH_GPT_max_rock_value_l863_86397


namespace NUMINAMATH_GPT_total_cars_l863_86388

theorem total_cars (Tommy_cars Jessie_cars : ℕ) (older_brother_cars : ℕ) 
  (h1 : Tommy_cars = 3) 
  (h2 : Jessie_cars = 3)
  (h3 : older_brother_cars = Tommy_cars + Jessie_cars + 5) : 
  Tommy_cars + Jessie_cars + older_brother_cars = 17 := by
  sorry

end NUMINAMATH_GPT_total_cars_l863_86388


namespace NUMINAMATH_GPT_amusement_park_people_l863_86328

theorem amusement_park_people (students adults free : ℕ) (total_people paid : ℕ) :
  students = 194 →
  adults = 235 →
  free = 68 →
  total_people = students + adults →
  paid = total_people - free →
  paid - free = 293 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_amusement_park_people_l863_86328


namespace NUMINAMATH_GPT_garden_strawberry_area_l863_86373

variable (total_garden_area : Real) (fruit_fraction : Real) (strawberry_fraction : Real)
variable (h1 : total_garden_area = 64)
variable (h2 : fruit_fraction = 1 / 2)
variable (h3 : strawberry_fraction = 1 / 4)

theorem garden_strawberry_area : 
  let fruit_area := total_garden_area * fruit_fraction
  let strawberry_area := fruit_area * strawberry_fraction
  strawberry_area = 8 :=
by
  sorry

end NUMINAMATH_GPT_garden_strawberry_area_l863_86373


namespace NUMINAMATH_GPT_no_15_students_with_unique_colors_l863_86382

-- Conditions as definitions
def num_students : Nat := 30
def num_colors : Nat := 15

-- The main statement
theorem no_15_students_with_unique_colors
  (students : Fin num_students → (Fin num_colors × Fin num_colors)) :
  ¬ ∃ (subset : Fin 15 → Fin num_students),
    ∀ i j (hi : i ≠ j), (students (subset i)).1 ≠ (students (subset j)).1 ∧
                         (students (subset i)).2 ≠ (students (subset j)).2 :=
by sorry

end NUMINAMATH_GPT_no_15_students_with_unique_colors_l863_86382


namespace NUMINAMATH_GPT_sphere_pyramid_problem_l863_86349

theorem sphere_pyramid_problem (n m : ℕ) :
  (n * (n + 1) * (2 * n + 1)) / 6 + (m * (m + 1) * (m + 2)) / 6 = 605 → n = 10 ∧ m = 10 :=
by
  sorry

end NUMINAMATH_GPT_sphere_pyramid_problem_l863_86349


namespace NUMINAMATH_GPT_quadratic_reciprocal_squares_l863_86310

theorem quadratic_reciprocal_squares :
  (∃ p q : ℝ, (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = p ∨ x = q)) ∧ (1 / p^2 + 1 / q^2 = 13 / 4)) :=
by
  have quadratic_eq : (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = 1 ∨ x = 2 / 3)) := sorry
  have identity_eq : 1 / (1:ℝ)^2 + 1 / (2 / 3)^2 = 13 / 4 := sorry
  exact ⟨1, 2 / 3, quadratic_eq, identity_eq⟩

end NUMINAMATH_GPT_quadratic_reciprocal_squares_l863_86310


namespace NUMINAMATH_GPT_minimum_rectangles_needed_l863_86348

/-- The theorem that defines the minimum number of rectangles needed to cover the specified figure -/
theorem minimum_rectangles_needed 
    (rectangles : ℕ) 
    (figure : Type)
    (covers : figure → Prop) :
  rectangles = 12 :=
sorry

end NUMINAMATH_GPT_minimum_rectangles_needed_l863_86348


namespace NUMINAMATH_GPT_problem1_problem2_l863_86303

noncomputable def h (x a : ℝ) : ℝ := (x - a) * Real.exp x + a
noncomputable def f (x b : ℝ) : ℝ := x^2 - 2 * b * x - 3 * Real.exp 1 + Real.exp 1 + 15 / 2

theorem problem1 (a : ℝ) :
  ∃ c, ∀ x ∈ Set.Icc (-1:ℝ) (1:ℝ), h x a ≥ c :=
by
  sorry

theorem problem2 (b : ℝ) :
  (∀ x1 ∈ Set.Icc (-1:ℝ) (1:ℝ), ∃ x2 ∈ Set.Icc (1:ℝ) (2:ℝ), h x1 3 ≥ f x2 b) →
  b ≥ 17 / 8 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l863_86303


namespace NUMINAMATH_GPT_ratio_p_q_is_minus_one_l863_86351

theorem ratio_p_q_is_minus_one (p q : ℤ) (h : (25 / 7 : ℝ) + ((2 * q - p) / (2 * q + p) : ℝ) = 4) : (p / q : ℝ) = -1 := 
sorry

end NUMINAMATH_GPT_ratio_p_q_is_minus_one_l863_86351


namespace NUMINAMATH_GPT_ratio_y_to_x_l863_86386

theorem ratio_y_to_x (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : y / x = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_y_to_x_l863_86386


namespace NUMINAMATH_GPT_closest_to_9_l863_86383

noncomputable def optionA : ℝ := 10.01
noncomputable def optionB : ℝ := 9.998
noncomputable def optionC : ℝ := 9.9
noncomputable def optionD : ℝ := 9.01
noncomputable def target : ℝ := 9

theorem closest_to_9 : 
  abs (optionD - target) < abs (optionA - target) ∧ 
  abs (optionD - target) < abs (optionB - target) ∧ 
  abs (optionD - target) < abs (optionC - target) := 
by
  sorry

end NUMINAMATH_GPT_closest_to_9_l863_86383


namespace NUMINAMATH_GPT_directly_above_156_is_133_l863_86376

def row_numbers (k : ℕ) : ℕ := 2 * k - 1

def total_numbers_up_to_row (k : ℕ) : ℕ := k * k

def find_row (n : ℕ) : ℕ :=
  Nat.sqrt (n + 1)

def position_in_row (n k : ℕ) : ℕ :=
  n - (total_numbers_up_to_row (k - 1)) + 1

def number_directly_above (n : ℕ) : ℕ :=
  let k := find_row n
  let pos := position_in_row n k
  (total_numbers_up_to_row (k - 1) - row_numbers (k - 1)) + pos + 1

theorem directly_above_156_is_133 : number_directly_above 156 = 133 := 
  by
  sorry

end NUMINAMATH_GPT_directly_above_156_is_133_l863_86376


namespace NUMINAMATH_GPT_bunches_with_new_distribution_l863_86366

-- Given conditions
def bunches_initial := 8
def flowers_per_bunch_initial := 9
def total_flowers := bunches_initial * flowers_per_bunch_initial

-- New condition and proof requirement
def flowers_per_bunch_new := 12
def bunches_new := total_flowers / flowers_per_bunch_new

theorem bunches_with_new_distribution : bunches_new = 6 := by
  sorry

end NUMINAMATH_GPT_bunches_with_new_distribution_l863_86366


namespace NUMINAMATH_GPT_product_of_nine_integers_16_to_30_equals_15_factorial_l863_86345

noncomputable def factorial (n : Nat) : Nat :=
match n with
| 0     => 1
| (n+1) => (n+1) * factorial n

theorem product_of_nine_integers_16_to_30_equals_15_factorial :
  (16 * 18 * 20 * 21 * 22 * 25 * 26 * 27 * 28) = factorial 15 := 
by sorry

end NUMINAMATH_GPT_product_of_nine_integers_16_to_30_equals_15_factorial_l863_86345


namespace NUMINAMATH_GPT_evelyn_found_caps_l863_86305

theorem evelyn_found_caps (start_caps end_caps found_caps : ℕ) 
    (h1 : start_caps = 18) 
    (h2 : end_caps = 81) 
    (h3 : found_caps = end_caps - start_caps) :
  found_caps = 63 := by
  sorry

end NUMINAMATH_GPT_evelyn_found_caps_l863_86305


namespace NUMINAMATH_GPT_definite_integral_solution_l863_86368

noncomputable def integral_problem : ℝ := 
  by 
    sorry

theorem definite_integral_solution :
  integral_problem = (1/6 : ℝ) + Real.log 2 - Real.log 3 := 
by
  sorry

end NUMINAMATH_GPT_definite_integral_solution_l863_86368


namespace NUMINAMATH_GPT_unique_solution_mod_37_system_l863_86333

theorem unique_solution_mod_37_system :
  ∃! (a b c d : ℤ), 
  (a^2 + b * c ≡ a [ZMOD 37]) ∧
  (b * (a + d) ≡ b [ZMOD 37]) ∧
  (c * (a + d) ≡ c [ZMOD 37]) ∧
  (b * c + d^2 ≡ d [ZMOD 37]) ∧
  (a * d - b * c ≡ 1 [ZMOD 37]) :=
sorry

end NUMINAMATH_GPT_unique_solution_mod_37_system_l863_86333


namespace NUMINAMATH_GPT_length_of_platform_l863_86399

-- Given conditions
def train_length : ℝ := 100
def time_pole : ℝ := 15
def time_platform : ℝ := 40

-- Theorem to prove the length of the platform
theorem length_of_platform (L : ℝ) 
    (h_train_length : train_length = 100)
    (h_time_pole : time_pole = 15)
    (h_time_platform : time_platform = 40)
    (h_speed : (train_length / time_pole) = (100 + L) / time_platform) : 
    L = 500 / 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l863_86399


namespace NUMINAMATH_GPT_problem_solution_l863_86378

theorem problem_solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l863_86378


namespace NUMINAMATH_GPT_probability_composite_is_correct_l863_86392

noncomputable def probability_composite : ℚ :=
  1 - (25 / (8^6))

theorem probability_composite_is_correct :
  probability_composite = 262119 / 262144 :=
by
  sorry

end NUMINAMATH_GPT_probability_composite_is_correct_l863_86392


namespace NUMINAMATH_GPT_find_original_numbers_l863_86317

theorem find_original_numbers (x y : ℕ) (hx : x + y = 2022) 
  (hy : (x - 5) / 10 + 10 * y + 1 = 2252) : x = 1815 ∧ y = 207 :=
by sorry

end NUMINAMATH_GPT_find_original_numbers_l863_86317


namespace NUMINAMATH_GPT_man_l863_86384

noncomputable def man_saves (S : ℝ) : ℝ :=
0.20 * S

noncomputable def initial_expenses (S : ℝ) : ℝ :=
0.80 * S

noncomputable def new_expenses (S : ℝ) : ℝ :=
1.10 * (0.80 * S)

noncomputable def said_savings (S : ℝ) : ℝ :=
S - new_expenses S

theorem man's_monthly_salary (S : ℝ) (h : said_savings S = 500) : S = 4166.67 :=
by
  sorry

end NUMINAMATH_GPT_man_l863_86384


namespace NUMINAMATH_GPT_min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l863_86353

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Prove the minimum value for a = 1 and x in [-1, 0]
theorem min_f_a_eq_1 : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f 1 x ≥ 5 :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when a ≤ -1
theorem min_f_a_le_neg1 (h : ∀ a : ℝ, a ≤ -1) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a (-1) ≤ f a x :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when -1 < a < 0
theorem min_f_neg1_lt_a_lt_0 (h : ∀ a : ℝ, -1 < a ∧ a < 0) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a a ≤ f a x :=
by
  sorry

end NUMINAMATH_GPT_min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l863_86353


namespace NUMINAMATH_GPT_boys_variance_greater_than_girls_l863_86380

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

noncomputable def variance (scores : List ℝ) : ℝ :=
  let n := scores.length
  let mean := (scores.sum / n)
  let squared_diff := scores.map (λ x => (x - mean) ^ 2)
  (squared_diff.sum) / n

theorem boys_variance_greater_than_girls :
  variance boys_scores > variance girls_scores :=
by
  sorry

end NUMINAMATH_GPT_boys_variance_greater_than_girls_l863_86380


namespace NUMINAMATH_GPT_apples_prepared_l863_86369

variables (n_x n_l : ℕ)

theorem apples_prepared (hx : 3 * n_x = 5 * n_l - 12) (hs : 6 * n_l = 72) : n_x = 12 := 
by sorry

end NUMINAMATH_GPT_apples_prepared_l863_86369


namespace NUMINAMATH_GPT_mixed_number_sum_l863_86338

theorem mixed_number_sum : (2 + (1 / 10 : ℝ)) + (3 + (11 / 100 : ℝ)) = 5.21 := by
  sorry

end NUMINAMATH_GPT_mixed_number_sum_l863_86338


namespace NUMINAMATH_GPT_area_of_triangle_F1PF2_l863_86356

noncomputable def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / 25) + (y^2 / 16) = 1

def is_focus (f : ℝ × ℝ) : Prop := 
  f = (3, 0) ∨ f = (-3, 0)

def right_angle_at_P (F1 P F2 : ℝ × ℝ) : Prop := 
  let a1 := (F1.1 - P.1, F1.2 - P.2)
  let a2 := (F2.1 - P.1, F2.2 - P.2)
  a1.1 * a2.1 + a1.2 * a2.2 = 0

theorem area_of_triangle_F1PF2
  (P F1 F2 : ℝ × ℝ)
  (hP : point_on_ellipse P)
  (hF1 : is_focus F1)
  (hF2 : is_focus F2)
  (h_angle : right_angle_at_P F1 P F2) :
  1/2 * (P.1 - F1.1) * (P.2 - F2.2) = 16 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_F1PF2_l863_86356


namespace NUMINAMATH_GPT_find_m_l863_86337

theorem find_m (A B : Set ℝ) (m : ℝ) (hA: A = {2, m}) (hB: B = {1, m^2}) (hU: A ∪ B = {1, 2, 3, 9}) : m = 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_l863_86337


namespace NUMINAMATH_GPT_spelling_bee_participants_l863_86311

theorem spelling_bee_participants (n : ℕ)
  (h1 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (k - 1 < 74 ∨ k - 1 > 74))
  (h2 : ∀ k, k > 0 → k ≤ n → k ≠ 75 → (75 - k > 0 ∨ k - 1 > 74)) :
  n = 149 := by
  sorry

end NUMINAMATH_GPT_spelling_bee_participants_l863_86311


namespace NUMINAMATH_GPT_cinnamon_balls_required_l863_86316

theorem cinnamon_balls_required 
  (num_family_members : ℕ) 
  (cinnamon_balls_per_day : ℕ) 
  (num_days : ℕ) 
  (h_family : num_family_members = 5) 
  (h_balls_per_day : cinnamon_balls_per_day = 5) 
  (h_days : num_days = 10) : 
  num_family_members * cinnamon_balls_per_day * num_days = 50 := by
  sorry

end NUMINAMATH_GPT_cinnamon_balls_required_l863_86316


namespace NUMINAMATH_GPT_evaluate_expression_l863_86300

theorem evaluate_expression (x z : ℤ) (h1 : x = 2) (h2 : z = 1) : z * (z - 4 * x) = -7 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l863_86300


namespace NUMINAMATH_GPT_first_day_reduction_percentage_l863_86360

variables (P x : ℝ)

theorem first_day_reduction_percentage (h : P * (1 - x / 100) * 0.90 = 0.81 * P) : x = 10 :=
sorry

end NUMINAMATH_GPT_first_day_reduction_percentage_l863_86360


namespace NUMINAMATH_GPT_x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l863_86379

variable (x : ℝ)

theorem x_positive_implies_abs_positive (hx : x > 0) : |x| > 0 := sorry

theorem abs_positive_not_necessiarily_x_positive : (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := sorry

theorem x_positive_is_sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 0 → |x| > 0) ∧ 
  (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := 
  ⟨x_positive_implies_abs_positive, abs_positive_not_necessiarily_x_positive⟩

end NUMINAMATH_GPT_x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l863_86379


namespace NUMINAMATH_GPT_find_two_digit_numbers_l863_86344

def first_two_digit_number (x y : ℕ) : ℕ := 10 * x + y
def second_two_digit_number (x y : ℕ) : ℕ := 10 * (x + 5) + y

theorem find_two_digit_numbers :
  ∃ (x_2 y : ℕ), 
  (first_two_digit_number x_2 y = x_2^2 + x_2 * y + y^2) ∧ 
  (second_two_digit_number x_2 y = (x_2 + 5)^2 + (x_2 + 5) * y + y^2) ∧ 
  (second_two_digit_number x_2 y - first_two_digit_number x_2 y = 50) ∧ 
  (y = 1 ∨ y = 3) := 
sorry

end NUMINAMATH_GPT_find_two_digit_numbers_l863_86344


namespace NUMINAMATH_GPT_four_digit_perfect_square_exists_l863_86308

theorem four_digit_perfect_square_exists (x y : ℕ) (h1 : 10 ≤ x ∧ x < 100) (h2 : 10 ≤ y ∧ y < 100) (h3 : 101 * x + 100 = y^2) : 
  ∃ n, n = 8281 ∧ n = y^2 ∧ (((n / 100) : ℕ) = ((n % 100) : ℕ) + 1) :=
by 
  sorry

end NUMINAMATH_GPT_four_digit_perfect_square_exists_l863_86308


namespace NUMINAMATH_GPT_tom_age_ratio_l863_86363

theorem tom_age_ratio (T N : ℕ) (h1 : T = 2 * (T / 2)) (h2 : T - N = 3 * ((T / 2) - 3 * N)) : T / N = 16 :=
  sorry

end NUMINAMATH_GPT_tom_age_ratio_l863_86363


namespace NUMINAMATH_GPT_perimeter_of_square_l863_86364

theorem perimeter_of_square (area : ℝ) (h : area = 392) : 
  ∃ (s : ℝ), 4 * s = 56 * Real.sqrt 2 :=
by 
  use (Real.sqrt 392)
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l863_86364


namespace NUMINAMATH_GPT_total_distance_driven_l863_86342

def renaldo_distance : ℕ := 15
def ernesto_distance : ℕ := 7 + (renaldo_distance / 3)

theorem total_distance_driven :
  renaldo_distance + ernesto_distance = 27 :=
sorry

end NUMINAMATH_GPT_total_distance_driven_l863_86342


namespace NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l863_86334

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x^2 - x

theorem problem_part1 :
  (∀ x, 0 < x -> x < 1 / Real.exp 1 -> f (Real.log x + 1) < 0) ∧ 
  (∀ x, x > 1 / Real.exp 1 -> f (Real.log x + 1) > 0) ∧ 
  (f (1 / Real.exp 1) = 1 / Real.exp 1 * Real.log (1 / Real.exp 1)) :=
sorry

theorem problem_part2 (a : ℝ) :
  (∀ x, x > 0 -> f x ≤ g a x) ↔ a ≥ 1 :=
sorry

theorem problem_part3 (a : ℝ) (m : ℝ) (ha : a = 1/8) :
  (∃ m, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ (3 * f x / (4 * x) + m + g a x = 0))) ↔ 
  (7/8 < m ∧ m < (15/8 - 3/4 * Real.log 3)) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l863_86334


namespace NUMINAMATH_GPT_sqrt_expression_equal_cos_half_theta_l863_86362

noncomputable def sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta (θ : Real) : Real :=
  Real.sqrt (1 / 2 + 1 / 2 * Real.sqrt (1 / 2 + 1 / 2 * Real.cos (2 * θ))) - Real.sqrt (1 - Real.sin θ)

theorem sqrt_expression_equal_cos_half_theta (θ : Real) (h : π < θ) (h2 : θ < 3 * π / 2)
  (h3 : Real.cos θ < 0) (h4 : 0 < Real.sin (θ / 2)) (h5 : Real.cos (θ / 2) < 0) :
  sqrt_half_plus_sqrt_half_cos2theta_minus_sqrt_one_minus_sintheta θ = Real.cos (θ / 2) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_equal_cos_half_theta_l863_86362


namespace NUMINAMATH_GPT_ordering_9_8_4_12_3_16_l863_86346

theorem ordering_9_8_4_12_3_16 : (4 ^ 12 < 9 ^ 8) ∧ (9 ^ 8 = 3 ^ 16) :=
by {
  sorry
}

end NUMINAMATH_GPT_ordering_9_8_4_12_3_16_l863_86346


namespace NUMINAMATH_GPT_number_of_tangents_l863_86302

-- Define the points and conditions
variable (A B : ℝ × ℝ)
variable (dist_AB : dist A B = 8)
variable (radius_A : ℝ := 3)
variable (radius_B : ℝ := 2)

-- The goal
theorem number_of_tangents (dist_condition : dist A B = 8) : 
  ∃ n, n = 2 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_number_of_tangents_l863_86302


namespace NUMINAMATH_GPT_split_cost_evenly_l863_86391

noncomputable def cupcake_cost : ℝ := 1.50
noncomputable def number_of_cupcakes : ℝ := 12
noncomputable def total_cost : ℝ := number_of_cupcakes * cupcake_cost
noncomputable def total_people : ℝ := 2

theorem split_cost_evenly : (total_cost / total_people) = 9 :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_split_cost_evenly_l863_86391


namespace NUMINAMATH_GPT_min_length_M_intersect_N_l863_86323

-- Define the sets M and N with the given conditions
def M (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2/3}
def N (n : ℝ) : Set ℝ := {x | n - 3/4 ≤ x ∧ x ≤ n}
def M_intersect_N (m n : ℝ) : Set ℝ := M m ∩ N n

-- Define the condition that M and N are subsets of [0, 1]
def in_interval (m n : ℝ) := (M m ⊆ {x | 0 ≤ x ∧ x ≤ 1}) ∧ (N n ⊆ {x | 0 ≤ x ∧ x ≤ 1})

-- Define the length of a set given by an interval [a, b]
def length_interval (a b : ℝ) := b - a

-- Define the length of the intersection of M and N
noncomputable def length_M_intersect_N (m n : ℝ) : ℝ :=
  let a := max m (n - 3/4)
  let b := min (m + 2/3) n
  length_interval a b

-- Prove that the minimum length of M ∩ N is 5/12
theorem min_length_M_intersect_N (m n : ℝ) (h : in_interval m n) : length_M_intersect_N m n = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_min_length_M_intersect_N_l863_86323


namespace NUMINAMATH_GPT_paytons_score_l863_86322

theorem paytons_score (total_score_14_students : ℕ)
    (average_14_students : total_score_14_students / 14 = 80)
    (total_score_15_students : ℕ)
    (average_15_students : total_score_15_students / 15 = 81) :
  total_score_15_students - total_score_14_students = 95 :=
by
  sorry

end NUMINAMATH_GPT_paytons_score_l863_86322


namespace NUMINAMATH_GPT_coffee_shop_cups_l863_86320

variables (A B X Y : ℕ) (Z : ℕ)

theorem coffee_shop_cups (h1 : Z = (A * B * X) + (A * (7 - B) * Y)) : 
  Z = (A * B * X) + (A * (7 - B) * Y) := 
by
  sorry

end NUMINAMATH_GPT_coffee_shop_cups_l863_86320


namespace NUMINAMATH_GPT_plane_speed_in_still_air_l863_86329

theorem plane_speed_in_still_air (P W : ℝ) 
  (h1 : (P + W) * 3 = 900) 
  (h2 : (P - W) * 4 = 900) 
  : P = 262.5 :=
by
  sorry

end NUMINAMATH_GPT_plane_speed_in_still_air_l863_86329


namespace NUMINAMATH_GPT_converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l863_86325

theorem converse_of_P (a b : ℤ) : (a - 2 > b - 2) → (a > b) :=
by
  intro h
  exact sorry

theorem inverse_of_P (a b : ℤ) : (a ≤ b) → (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

theorem contrapositive_of_P (a b : ℤ) : (a - 2 ≤ b - 2) → (a ≤ b) :=
by
  intro h
  exact sorry

theorem negation_of_P (a b : ℤ) : (a > b) → ¬ (a - 2 ≤ b - 2) :=
by
  intro h
  exact sorry

end NUMINAMATH_GPT_converse_of_P_inverse_of_P_contrapositive_of_P_negation_of_P_l863_86325


namespace NUMINAMATH_GPT_enclosed_polygons_l863_86377

theorem enclosed_polygons (n : ℕ) :
  (∃ α β : ℝ, (15 * β) = 360 ∧ β = 180 - α ∧ (15 * α) = 180 * (n - 2) / n) ↔ n = 15 :=
by sorry

end NUMINAMATH_GPT_enclosed_polygons_l863_86377


namespace NUMINAMATH_GPT_extreme_point_f_l863_86390

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 1)

theorem extreme_point_f :
  ∃ x : ℝ, (∀ y : ℝ, y ≠ 0 → (Real.exp y * y < 0 ↔ y < x)) ∧ x = 0 :=
by
  sorry

end NUMINAMATH_GPT_extreme_point_f_l863_86390


namespace NUMINAMATH_GPT_probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l863_86350

noncomputable def probability_sum_is_multiple_of_3 : ℝ :=
  let total_events := 36
  let favorable_events := 12
  favorable_events / total_events

noncomputable def probability_sum_is_prime : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

noncomputable def probability_second_greater_than_first : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

theorem probability_sum_multiple_of_3_eq_one_third :
  probability_sum_is_multiple_of_3 = 1 / 3 :=
by sorry

theorem probability_sum_prime_eq_five_twelfths :
  probability_sum_is_prime = 5 / 12 :=
by sorry

theorem probability_second_greater_than_first_eq_five_twelfths :
  probability_second_greater_than_first = 5 / 12 :=
by sorry

end NUMINAMATH_GPT_probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l863_86350


namespace NUMINAMATH_GPT_model2_best_fit_l863_86359
-- Import necessary tools from Mathlib

-- Define the coefficients of determination for the four models
def R2_model1 : ℝ := 0.75
def R2_model2 : ℝ := 0.90
def R2_model3 : ℝ := 0.28
def R2_model4 : ℝ := 0.55

-- Define the best fitting model
def best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ) : Prop :=
  R2_2 > R2_1 ∧ R2_2 > R2_3 ∧ R2_2 > R2_4

-- Statement to prove
theorem model2_best_fit : best_fitting_model R2_model1 R2_model2 R2_model3 R2_model4 :=
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_model2_best_fit_l863_86359


namespace NUMINAMATH_GPT_number_identification_l863_86321

theorem number_identification (x : ℝ) (h : x ^ 655 / x ^ 650 = 100000) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_identification_l863_86321


namespace NUMINAMATH_GPT_slower_train_speed_l863_86398

noncomputable def speed_of_slower_train (v_f : ℕ) (l1 l2 : ℚ) (t : ℚ) : ℚ :=
  let total_distance := l1 + l2
  let time_in_hours := t / 3600
  let relative_speed := total_distance / time_in_hours
  relative_speed - v_f

theorem slower_train_speed :
  speed_of_slower_train 210 (11 / 10) (9 / 10) 24 = 90 := by
  sorry

end NUMINAMATH_GPT_slower_train_speed_l863_86398


namespace NUMINAMATH_GPT_ending_time_proof_l863_86332

def starting_time_seconds : ℕ := (1 * 3600) + (57 * 60) + 58
def glow_interval : ℕ := 13
def total_glow_count : ℕ := 382
def total_glow_duration : ℕ := total_glow_count * glow_interval
def ending_time_seconds : ℕ := starting_time_seconds + total_glow_duration

theorem ending_time_proof : 
ending_time_seconds = (3 * 3600) + (14 * 60) + 4 := by
  -- Proof starts here
  sorry

end NUMINAMATH_GPT_ending_time_proof_l863_86332


namespace NUMINAMATH_GPT_remaining_games_win_percent_l863_86343

variable (totalGames : ℕ) (firstGames : ℕ) (firstWinPercent : ℕ) (seasonWinPercent : ℕ)

-- Given conditions expressed as assumptions:
-- The total number of games played in a season is 40
axiom total_games_condition : totalGames = 40
-- The number of first games played is 30
axiom first_games_condition : firstGames = 30
-- The team won 40% of the first 30 games
axiom first_win_percent_condition : firstWinPercent = 40
-- The team won 50% of all its games in the season
axiom season_win_percent_condition : seasonWinPercent = 50

-- We need to prove that the percentage of the remaining games that the team won is 80%
theorem remaining_games_win_percent {remainingWinPercent : ℕ} :
  totalGames = 40 →
  firstGames = 30 →
  firstWinPercent = 40 →
  seasonWinPercent = 50 →
  remainingWinPercent = 80 :=
by
  intros
  sorry

end NUMINAMATH_GPT_remaining_games_win_percent_l863_86343


namespace NUMINAMATH_GPT_nonempty_solution_iff_a_gt_one_l863_86341

theorem nonempty_solution_iff_a_gt_one (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
sorry

end NUMINAMATH_GPT_nonempty_solution_iff_a_gt_one_l863_86341


namespace NUMINAMATH_GPT_correct_statements_are_C_and_D_l863_86306

theorem correct_statements_are_C_and_D
  (a b c m : ℝ)
  (ha1 : -1 < a) (ha2 : a < 5)
  (hb1 : -2 < b) (hb2 : b < 3)
  (hab : a > b)
  (h_ac2bc2 : a * c^2 > b * c^2) (hc2_pos : c^2 > 0)
  (h_ab_pos : a > b) (h_b_pos : b > 0) (hm_pos : m > 0) :
  (¬(1 < a - b ∧ a - b < 2)) ∧ (¬(a^2 > b^2)) ∧ (a > b) ∧ ((b + m) / (a + m) > b / a) :=
by sorry

end NUMINAMATH_GPT_correct_statements_are_C_and_D_l863_86306


namespace NUMINAMATH_GPT_set_subtraction_M_N_l863_86358

-- Definitions
def A : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def B : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }
def M : Set ℝ := { x | ∃ y, y = Real.sqrt (1 - x) }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

-- Statement
theorem set_subtraction_M_N : (M \ N) = { x | x < 0 } := by
  sorry

end NUMINAMATH_GPT_set_subtraction_M_N_l863_86358


namespace NUMINAMATH_GPT_solution_set_l863_86304

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom monotone_decreasing_f : ∀ {a b : ℝ}, 0 ≤ a → a ≤ b → f b ≤ f a
axiom f_half_eq_zero : f (1 / 2) = 0

theorem solution_set :
  { x : ℝ | f (Real.log x / Real.log (1 / 4)) < 0 } = 
  { x : ℝ | 0 < x ∧ x < 1 / 2 } ∪ { x : ℝ | 2 < x } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l863_86304


namespace NUMINAMATH_GPT_sum_reciprocals_l863_86374

theorem sum_reciprocals (a b α β : ℝ) (h1: 7 * a^2 + 2 * a + 6 = 0) (h2: 7 * b^2 + 2 * b + 6 = 0) 
  (h3: α = 1 / a) (h4: β = 1 / b) (h5: a + b = -2/7) (h6: a * b = 6/7) : 
  α + β = -1/3 :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocals_l863_86374


namespace NUMINAMATH_GPT_range_of_m_l863_86389

theorem range_of_m (m : ℝ) (x : ℝ) :
  (¬ (|1 - (x - 1) / 3| ≤ 2) → ¬ (x^2 - 2 * x + (1 - m^2) ≤ 0)) → 
  (|m| ≥ 9) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l863_86389


namespace NUMINAMATH_GPT_spherical_to_rectangular_coords_l863_86393

noncomputable def sphericalToRectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * (Real.sin phi) * (Real.cos theta), 
   rho * (Real.sin phi) * (Real.sin theta), 
   rho * (Real.cos phi))

theorem spherical_to_rectangular_coords :
  sphericalToRectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coords_l863_86393


namespace NUMINAMATH_GPT_fraction_value_l863_86327

theorem fraction_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l863_86327


namespace NUMINAMATH_GPT_pencils_in_drawer_l863_86309

theorem pencils_in_drawer (P : ℕ) (h1 : P + 19 + 16 = 78) : P = 43 :=
by
  sorry

end NUMINAMATH_GPT_pencils_in_drawer_l863_86309


namespace NUMINAMATH_GPT_obtuse_scalene_triangle_l863_86339

theorem obtuse_scalene_triangle {k : ℕ} (h1 : 13 < k + 17) (h2 : 17 < 13 + k)
  (h3 : 13 < k + 17) (h4 : k ≠ 13) (h5 : k ≠ 17) 
  (h6 : 17^2 > 13^2 + k^2 ∨ k^2 > 13^2 + 17^2) 
  (h7 : (k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10 ∨ k = 22 ∨ 
        k = 23 ∨ k = 24 ∨ k = 25 ∨ k = 26 ∨ k = 27 ∨ k = 28 ∨ k = 29)) :
  ∃ n, n = 14 := 
by
  sorry

end NUMINAMATH_GPT_obtuse_scalene_triangle_l863_86339


namespace NUMINAMATH_GPT_exists_x0_l863_86314

noncomputable def f (x a : ℝ) : ℝ := x^2 + (Real.log (3 * x))^2 - 2 * a * (x + 3 * Real.log (3 * x)) + 10 * a^2

theorem exists_x0 (a : ℝ) (h : a = 1 / 30) : ∃ x0 : ℝ, f x0 a ≤ 1 / 10 := 
by
  sorry

end NUMINAMATH_GPT_exists_x0_l863_86314


namespace NUMINAMATH_GPT_solve_inequality_l863_86315

theorem solve_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l863_86315


namespace NUMINAMATH_GPT_three_digit_number_constraint_l863_86326

theorem three_digit_number_constraint (B : ℕ) (h1 : 30 ≤ B ∧ B < 40) (h2 : (330 + B) % 3 = 0) (h3 : (330 + B) % 7 = 0) : B = 6 :=
sorry

end NUMINAMATH_GPT_three_digit_number_constraint_l863_86326


namespace NUMINAMATH_GPT_inequality_for_positive_real_l863_86381

theorem inequality_for_positive_real (x : ℝ) (h : 0 < x) : x + 1/x ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_positive_real_l863_86381


namespace NUMINAMATH_GPT_correct_quadratic_equation_l863_86307

-- Definitions based on conditions
def root_sum (α β : ℝ) := α + β = 8
def root_product (α β : ℝ) := α * β = 24

-- Main statement to be proven
theorem correct_quadratic_equation (α β : ℝ) (h1 : root_sum 5 3) (h2 : root_product (-6) (-4)) :
    (α - 5) * (α - 3) = 0 ∧ (α + 6) * (α + 4) = 0 → α * α - 8 * α + 24 = 0 :=
sorry

end NUMINAMATH_GPT_correct_quadratic_equation_l863_86307


namespace NUMINAMATH_GPT_jerrys_breakfast_calories_l863_86355

theorem jerrys_breakfast_calories 
    (num_pancakes : ℕ) (calories_per_pancake : ℕ) 
    (num_bacon : ℕ) (calories_per_bacon : ℕ) 
    (num_cereal : ℕ) (calories_per_cereal : ℕ) 
    (calories_total : ℕ) :
    num_pancakes = 6 →
    calories_per_pancake = 120 →
    num_bacon = 2 →
    calories_per_bacon = 100 →
    num_cereal = 1 →
    calories_per_cereal = 200 →
    calories_total = num_pancakes * calories_per_pancake
                   + num_bacon * calories_per_bacon
                   + num_cereal * calories_per_cereal →
    calories_total = 1120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  assumption

end NUMINAMATH_GPT_jerrys_breakfast_calories_l863_86355


namespace NUMINAMATH_GPT_kim_distance_traveled_l863_86313

-- Definitions based on the problem conditions:
def infantry_column_length : ℝ := 1  -- The length of the infantry column in km.
def distance_inf_covered : ℝ := 2.4  -- Distance the infantrymen covered in km.

-- Theorem statement:
theorem kim_distance_traveled (column_length : ℝ) (inf_covered : ℝ) :
  column_length = 1 →
  inf_covered = 2.4 →
  ∃ d : ℝ, d = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_kim_distance_traveled_l863_86313


namespace NUMINAMATH_GPT_can_construct_segment_l863_86375

noncomputable def constructSegment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P

theorem can_construct_segment (Ω₁ Ω₂ : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ Ω₁ ∧ B ∈ Ω₂ ∧ (A + B) / 2 = P) :=
sorry

end NUMINAMATH_GPT_can_construct_segment_l863_86375


namespace NUMINAMATH_GPT_polynomial_prime_is_11_l863_86301

def P (a : ℕ) : ℕ := a^4 - 4 * a^3 + 15 * a^2 - 30 * a + 27

theorem polynomial_prime_is_11 (a : ℕ) (hp : Nat.Prime (P a)) : P a = 11 := 
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_prime_is_11_l863_86301


namespace NUMINAMATH_GPT_tank_full_capacity_l863_86324

theorem tank_full_capacity (C : ℝ) (H1 : 0.4 * C + 36 = 0.7 * C) : C = 120 :=
by
  sorry

end NUMINAMATH_GPT_tank_full_capacity_l863_86324


namespace NUMINAMATH_GPT_circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l863_86365

-- Problem (1)
theorem circle_a_lt_8 (x y a : ℝ) (h : x^2 + y^2 - 4*x - 4*y + a = 0) : 
  a < 8 :=
by
  sorry

-- Problem (2)
theorem tangent_lines (a : ℝ) (h : a = -17) : 
  ∃ (k : ℝ), k * 7 - 6 - 7 * k = 0 ∧
  ((39 * k + 80 * (-7) - 207 = 0) ∨ (k = 7)) :=
by
  sorry

-- Problem (3)
theorem perpendicular_circle_intersection (x1 x2 y1 y2 a : ℝ) 
  (h1: 2 * x1 - y1 - 3 = 0) 
  (h2: 2 * x2 - y2 - 3 = 0) 
  (h3: x1 * x2 + y1 * y2 = 0) 
  (hpoly : 5 * x1 * x2 - 6 * (x1 + x2) + 9 = 0): 
  a = -6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_circle_a_lt_8_tangent_lines_perpendicular_circle_intersection_l863_86365


namespace NUMINAMATH_GPT_total_sheep_l863_86336

-- Define the conditions as hypotheses
variables (Aaron_sheep Beth_sheep : ℕ)
def condition1 := Aaron_sheep = 7 * Beth_sheep
def condition2 := Aaron_sheep = 532
def condition3 := Beth_sheep = 76

-- Assert that under these conditions, the total number of sheep is 608.
theorem total_sheep
  (h1 : condition1 Aaron_sheep Beth_sheep)
  (h2 : condition2 Aaron_sheep)
  (h3 : condition3 Beth_sheep) :
  Aaron_sheep + Beth_sheep = 608 :=
by sorry

end NUMINAMATH_GPT_total_sheep_l863_86336


namespace NUMINAMATH_GPT_unique_cd_exists_l863_86352

open Real

theorem unique_cd_exists (h₀ : 0 < π / 2):
  ∃! (c d : ℝ), (0 < c) ∧ (c < π / 2) ∧ (0 < d) ∧ (d < π / 2) ∧ (c < d) ∧ 
  (sin (cos c) = c) ∧ (cos (sin d) = d) := sorry

end NUMINAMATH_GPT_unique_cd_exists_l863_86352


namespace NUMINAMATH_GPT_moon_speed_kmh_l863_86396

theorem moon_speed_kmh (speed_kms : ℝ) (h : speed_kms = 0.9) : speed_kms * 3600 = 3240 :=
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_moon_speed_kmh_l863_86396


namespace NUMINAMATH_GPT_range_of_omega_l863_86354

theorem range_of_omega (ω : ℝ) (hω : ω > 2/3) :
  (∀ x : ℝ, x = (k : ℤ) * π / ω + 3 * π / (4 * ω) → (x ≤ π ∨ x ≥ 2 * π) ) →
  ω ∈ Set.Icc (3/4 : ℝ) (7/8 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_omega_l863_86354


namespace NUMINAMATH_GPT_correct_quotient_l863_86340

variable (D : ℕ) (q1 q2 : ℕ)
variable (h1 : q1 = 4900) (h2 : D - 1000 = 1200 * q1)

theorem correct_quotient : q2 = D / 2100 → q2 = 2800 :=
by
  sorry

end NUMINAMATH_GPT_correct_quotient_l863_86340


namespace NUMINAMATH_GPT_tens_digit_of_9_pow_1024_l863_86318

theorem tens_digit_of_9_pow_1024 : 
  (9^1024 % 100) / 10 % 10 = 6 := 
sorry

end NUMINAMATH_GPT_tens_digit_of_9_pow_1024_l863_86318


namespace NUMINAMATH_GPT_find_a_n_plus_b_n_l863_86370

noncomputable def a (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else if n = 2 then 3 
  else sorry -- Placeholder for proper recursive implementation

noncomputable def b (n : ℕ) : ℕ := 
  if n = 1 then 5
  else sorry -- Placeholder for proper recursive implementation

theorem find_a_n_plus_b_n (n : ℕ) (i j k l : ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3) (h3 : b 1 = 5) 
  (h4 : i + j = k + l) (h5 : a i + b j = a k + b l) : a n + b n = 4 * n + 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_n_plus_b_n_l863_86370


namespace NUMINAMATH_GPT_smallest_number_increased_by_3_divisible_by_divisors_l863_86367

theorem smallest_number_increased_by_3_divisible_by_divisors
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 27)
  (h2 : d2 = 35)
  (h3 : d3 = 25)
  (h4 : d4 = 21) :
  (n + 3) % d1 = 0 →
  (n + 3) % d2 = 0 →
  (n + 3) % d3 = 0 →
  (n + 3) % d4 = 0 →
  n = 4722 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_increased_by_3_divisible_by_divisors_l863_86367
