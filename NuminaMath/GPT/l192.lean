import Mathlib

namespace NUMINAMATH_GPT_ceil_evaluation_l192_19211

theorem ceil_evaluation : 
  (Int.ceil (4 * (8 - 1 / 3 : ℚ))) = 31 :=
by
  sorry

end NUMINAMATH_GPT_ceil_evaluation_l192_19211


namespace NUMINAMATH_GPT_divisible_by_9_l192_19269

theorem divisible_by_9 (n : ℕ) : 9 ∣ (4^n + 15 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_9_l192_19269


namespace NUMINAMATH_GPT_circle_tangent_radius_l192_19267

-- Define the radii of the three given circles
def radius1 : ℝ := 1.0
def radius2 : ℝ := 2.0
def radius3 : ℝ := 3.0

-- Define the problem statement: finding the radius of the fourth circle externally tangent to the given three circles
theorem circle_tangent_radius (r1 r2 r3 : ℝ) (cond1 : r1 = 1) (cond2 : r2 = 2) (cond3 : r3 = 3) : 
  ∃ R : ℝ, R = 6 := by
  sorry

end NUMINAMATH_GPT_circle_tangent_radius_l192_19267


namespace NUMINAMATH_GPT_find_new_length_l192_19217

def initial_length_cm : ℕ := 100
def erased_length_cm : ℕ := 24
def final_length_cm : ℕ := 76

theorem find_new_length : initial_length_cm - erased_length_cm = final_length_cm := by
  sorry

end NUMINAMATH_GPT_find_new_length_l192_19217


namespace NUMINAMATH_GPT_correct_average_l192_19255

-- Define the conditions given in the problem
def avg_incorrect : ℕ := 46 -- incorrect average
def n : ℕ := 10 -- number of values
def incorrect_num : ℕ := 25
def correct_num : ℕ := 75
def diff : ℕ := correct_num - incorrect_num

-- Define the total sums
def total_incorrect : ℕ := avg_incorrect * n
def total_correct : ℕ := total_incorrect + diff

-- Define the correct average
def avg_correct : ℕ := total_correct / n

-- Statement in Lean 4
theorem correct_average :
  avg_correct = 51 :=
by
  -- We expect users to fill the proof here
  sorry

end NUMINAMATH_GPT_correct_average_l192_19255


namespace NUMINAMATH_GPT_percentage_error_in_area_l192_19210

theorem percentage_error_in_area (S : ℝ) (h : S > 0) :
  let S' := S * 1.06
  let A := S^2
  let A' := (S')^2
  (A' - A) / A * 100 = 12.36 := by
  sorry

end NUMINAMATH_GPT_percentage_error_in_area_l192_19210


namespace NUMINAMATH_GPT_find_B_and_C_l192_19264

def values_of_B_and_C (B C : ℤ) : Prop :=
  5 * B - 3 = 32 ∧ 2 * B + 2 * C = 18

theorem find_B_and_C : ∃ B C : ℤ, values_of_B_and_C B C ∧ B = 7 ∧ C = 2 := by
  sorry

end NUMINAMATH_GPT_find_B_and_C_l192_19264


namespace NUMINAMATH_GPT_exists_zero_in_interval_l192_19206

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 1 / x

theorem exists_zero_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- This is just the Lean statement, no proof is provided
  sorry

end NUMINAMATH_GPT_exists_zero_in_interval_l192_19206


namespace NUMINAMATH_GPT_july_savings_l192_19277

theorem july_savings (january: ℕ := 100) (total_savings: ℕ := 12700) :
  let february := 2 * january
  let march := 2 * february
  let april := 2 * march
  let may := 2 * april
  let june := 2 * may
  let july := 2 * june
  let total := january + february + march + april + may + june + july
  total = total_savings → july = 6400 := 
by
  sorry

end NUMINAMATH_GPT_july_savings_l192_19277


namespace NUMINAMATH_GPT_frank_cookies_l192_19257

theorem frank_cookies (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (h1 : Millie_cookies = 4)
  (h2 : Mike_cookies = 3 * Millie_cookies)
  (h3 : Frank_cookies = Mike_cookies / 2 - 3)
  : Frank_cookies = 3 := by
  sorry

end NUMINAMATH_GPT_frank_cookies_l192_19257


namespace NUMINAMATH_GPT_inequality_solution_l192_19258

theorem inequality_solution (x : ℝ) :
  x + 1 ≥ -3 ∧ -2 * (x + 3) > 0 ↔ -4 ≤ x ∧ x < -3 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l192_19258


namespace NUMINAMATH_GPT_linda_spent_amount_l192_19281

theorem linda_spent_amount :
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  total_cost = 6.80 :=
by
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  show total_cost = 6.80
  sorry

end NUMINAMATH_GPT_linda_spent_amount_l192_19281


namespace NUMINAMATH_GPT_egyptian_method_percentage_error_l192_19266

theorem egyptian_method_percentage_error :
  let a := 6
  let b := 4
  let c := 20
  let h := Real.sqrt (c^2 - ((a - b) / 2)^2)
  let S := ((a + b) / 2) * h
  let S1 := ((a + b) * c) / 2
  let percentage_error := abs ((20 / Real.sqrt 399) - 1) * 100
  percentage_error = abs ((20 / Real.sqrt 399) - 1) * 100 := by
  sorry

end NUMINAMATH_GPT_egyptian_method_percentage_error_l192_19266


namespace NUMINAMATH_GPT_total_pieces_l192_19249

-- Define the given conditions
def pieces_eaten_per_person : ℕ := 4
def num_people : ℕ := 3

-- Theorem stating the result
theorem total_pieces (h : num_people > 0) : (num_people * pieces_eaten_per_person) = 12 := 
by
  sorry

end NUMINAMATH_GPT_total_pieces_l192_19249


namespace NUMINAMATH_GPT_expression_evaluate_l192_19251

theorem expression_evaluate (a b c : ℤ) (h1 : b = a + 2) (h2 : c = b - 10) (ha : a = 4)
(h3 : a ≠ -1) (h4 : b ≠ 2) (h5 : b ≠ -4) (h6 : c ≠ -6) : (a + 2) / (a + 1) * (b - 1) / (b - 2) * (c + 8) / (c + 6) = 3 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluate_l192_19251


namespace NUMINAMATH_GPT_hannah_jerry_difference_l192_19247

-- Define the calculations of Hannah (H) and Jerry (J)
def H : Int := 10 - (3 * 4)
def J : Int := 10 - 3 + 4

-- Prove that H - J = -13
theorem hannah_jerry_difference : H - J = -13 := by
  sorry

end NUMINAMATH_GPT_hannah_jerry_difference_l192_19247


namespace NUMINAMATH_GPT_probability_of_earning_2400_l192_19286

noncomputable def spinner_labels := ["Bankrupt", "$700", "$900", "$200", "$3000", "$800"]
noncomputable def total_possibilities := (spinner_labels.length : ℕ) ^ 3
noncomputable def favorable_outcomes := 6

theorem probability_of_earning_2400 :
  (favorable_outcomes : ℚ) / total_possibilities = 1 / 36 := by
  sorry

end NUMINAMATH_GPT_probability_of_earning_2400_l192_19286


namespace NUMINAMATH_GPT_ratio_of_areas_ratio_of_perimeters_l192_19296

-- Define side lengths
def side_length_A : ℕ := 48
def side_length_B : ℕ := 60

-- Define the area of squares
def area_square (side_length : ℕ) : ℕ := side_length * side_length

-- Define the perimeter of squares
def perimeter_square (side_length : ℕ) : ℕ := 4 * side_length

-- Theorem for the ratio of areas
theorem ratio_of_areas : (area_square side_length_A) / (area_square side_length_B) = 16 / 25 :=
by
  sorry

-- Theorem for the ratio of perimeters
theorem ratio_of_perimeters : (perimeter_square side_length_A) / (perimeter_square side_length_B) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_ratio_of_perimeters_l192_19296


namespace NUMINAMATH_GPT_pages_in_first_chapter_l192_19245

theorem pages_in_first_chapter
  (total_pages : ℕ)
  (second_chapter_pages : ℕ)
  (first_chapter_pages : ℕ)
  (h1 : total_pages = 81)
  (h2 : second_chapter_pages = 68) :
  first_chapter_pages = 81 - 68 :=
sorry

end NUMINAMATH_GPT_pages_in_first_chapter_l192_19245


namespace NUMINAMATH_GPT_islanders_liars_count_l192_19224

def number_of_liars (N : ℕ) : ℕ :=
  if N = 30 then 28 else 0

theorem islanders_liars_count : number_of_liars 30 = 28 :=
  sorry

end NUMINAMATH_GPT_islanders_liars_count_l192_19224


namespace NUMINAMATH_GPT_stones_in_pile_l192_19244

theorem stones_in_pile (initial_stones : ℕ) (final_stones_A : ℕ) (final_stones_B_min final_stones_B_max final_stones_B : ℕ) (operations : ℕ) :
  initial_stones = 2006 ∧ final_stones_A = 1990 ∧ final_stones_B_min = 2080 ∧ final_stones_B_max = 2100 ∧ operations < 20 ∧ (final_stones_B_min ≤ final_stones_B ∧ final_stones_B ≤ final_stones_B_max) 
  → final_stones_B = 2090 :=
by
  sorry

end NUMINAMATH_GPT_stones_in_pile_l192_19244


namespace NUMINAMATH_GPT_min_rectilinear_distance_l192_19209

noncomputable def rectilinear_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

theorem min_rectilinear_distance : ∀ (M : ℝ × ℝ), (M.1 - M.2 + 4 = 0) → rectilinear_distance (1, 1) M ≥ 4 :=
by
  intro M hM
  -- We only need the statement, not the proof
  sorry

end NUMINAMATH_GPT_min_rectilinear_distance_l192_19209


namespace NUMINAMATH_GPT_combinedAverageAge_l192_19292

-- Definitions
def numFifthGraders : ℕ := 50
def avgAgeFifthGraders : ℕ := 10
def numParents : ℕ := 75
def avgAgeParents : ℕ := 40

-- Calculation of total ages
def totalAgeFifthGraders := numFifthGraders * avgAgeFifthGraders
def totalAgeParents := numParents * avgAgeParents
def combinedTotalAge := totalAgeFifthGraders + totalAgeParents

-- Calculation of total number of individuals
def totalIndividuals := numFifthGraders + numParents

-- The claim to prove
theorem combinedAverageAge : 
  combinedTotalAge / totalIndividuals = 28 := by
  -- Skipping the proof details.
  sorry

end NUMINAMATH_GPT_combinedAverageAge_l192_19292


namespace NUMINAMATH_GPT_a_finishes_job_in_60_days_l192_19201

theorem a_finishes_job_in_60_days (A B : ℝ)
  (h1 : A + B = 1 / 30)
  (h2 : 20 * (A + B) = 2 / 3)
  (h3 : 20 * A = 1 / 3) :
  1 / A = 60 :=
by sorry

end NUMINAMATH_GPT_a_finishes_job_in_60_days_l192_19201


namespace NUMINAMATH_GPT_solve_for_k_l192_19278

theorem solve_for_k (k : ℝ) : (∀ x : ℝ, 3 * (5 + k * x) = 15 * x + 15) ↔ k = 5 :=
  sorry

end NUMINAMATH_GPT_solve_for_k_l192_19278


namespace NUMINAMATH_GPT_minimally_intersecting_triples_modulo_1000_eq_344_l192_19214

def minimally_intersecting_triples_count_modulo : ℕ :=
  let total_count := 57344
  total_count % 1000

theorem minimally_intersecting_triples_modulo_1000_eq_344 :
  minimally_intersecting_triples_count_modulo = 344 := by
  sorry

end NUMINAMATH_GPT_minimally_intersecting_triples_modulo_1000_eq_344_l192_19214


namespace NUMINAMATH_GPT_total_distance_traveled_l192_19223

theorem total_distance_traveled (d : ℝ) (h1 : d/3 + d/4 + d/5 = 47/60) : 3 * d = 3 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l192_19223


namespace NUMINAMATH_GPT_initial_rulers_calculation_l192_19260

variable {initial_rulers taken_rulers left_rulers : ℕ}

theorem initial_rulers_calculation 
  (h1 : taken_rulers = 25) 
  (h2 : left_rulers = 21) 
  (h3 : initial_rulers = taken_rulers + left_rulers) : 
  initial_rulers = 46 := 
by 
  sorry

end NUMINAMATH_GPT_initial_rulers_calculation_l192_19260


namespace NUMINAMATH_GPT_milk_rate_proof_l192_19274

theorem milk_rate_proof
  (initial_milk : ℕ := 30000)
  (time_pumped_out : ℕ := 4)
  (rate_pumped_out : ℕ := 2880)
  (time_adding_milk : ℕ := 7)
  (final_milk : ℕ := 28980) :
  ((final_milk - (initial_milk - time_pumped_out * rate_pumped_out)) / time_adding_milk = 1500) :=
by {
  sorry
}

end NUMINAMATH_GPT_milk_rate_proof_l192_19274


namespace NUMINAMATH_GPT_area_of_smaller_circle_l192_19261

/-
  Variables and assumptions:
  r: Radius of the smaller circle
  R: Radius of the larger circle which is three times the smaller circle. Hence, R = 3 * r.
  PA = AB = 6: Lengths of the tangent segments
  Area: Calculated area of the smaller circle
-/

theorem area_of_smaller_circle (r : ℝ) (h1 : 6 = r) (h2 : 3 * 6 = R) (h3 : 6 = r) : 
  ∃ (area : ℝ), area = (36 * Real.pi) / 7 :=
by
  sorry 

end NUMINAMATH_GPT_area_of_smaller_circle_l192_19261


namespace NUMINAMATH_GPT_find_number_l192_19227

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l192_19227


namespace NUMINAMATH_GPT_greatest_a_l192_19272

theorem greatest_a (a : ℤ) (h_pos : a > 0) : 
  (∀ x : ℤ, (x^2 + a * x = -30) → (a = 31)) :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_a_l192_19272


namespace NUMINAMATH_GPT_ratio_of_blue_to_purple_beads_l192_19259

theorem ratio_of_blue_to_purple_beads :
  ∃ (B G : ℕ), 
    7 + B + G = 46 ∧ 
    G = B + 11 ∧ 
    B / 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_blue_to_purple_beads_l192_19259


namespace NUMINAMATH_GPT_find_first_factor_of_LCM_l192_19297

-- Conditions
def HCF : ℕ := 23
def Y : ℕ := 14
def largest_number : ℕ := 322

-- Statement
theorem find_first_factor_of_LCM
  (A B : ℕ)
  (H : Nat.gcd A B = HCF)
  (max_num : max A B = largest_number)
  (lcm_eq : Nat.lcm A B = HCF * X * Y) :
  X = 23 :=
sorry

end NUMINAMATH_GPT_find_first_factor_of_LCM_l192_19297


namespace NUMINAMATH_GPT_jessica_not_work_days_l192_19241

theorem jessica_not_work_days:
  ∃ (x y z : ℕ), 
    (x + y + z = 30) ∧
    (80 * x - 40 * y + 40 * z = 1600) ∧
    (z = 5) ∧
    (y = 5) :=
by
  sorry

end NUMINAMATH_GPT_jessica_not_work_days_l192_19241


namespace NUMINAMATH_GPT_f_800_l192_19288

noncomputable def f : ℕ → ℕ := sorry

axiom axiom1 : ∀ x y : ℕ, 0 < x → 0 < y → f (x * y) = f x + f y
axiom axiom2 : f 10 = 10
axiom axiom3 : f 40 = 14

theorem f_800 : f 800 = 26 :=
by
  -- Apply the conditions here
  sorry

end NUMINAMATH_GPT_f_800_l192_19288


namespace NUMINAMATH_GPT_minimum_disks_needed_l192_19235

-- Definition of the conditions
def disk_capacity : ℝ := 2.88
def file_sizes : List (ℝ × ℕ) := [(1.2, 5), (0.9, 10), (0.6, 8), (0.3, 7)]

/-- 
Theorem: Given the capacity of each disk and the sizes and counts of different files,
we can prove that the minimum number of disks needed to store all the files without 
splitting any file is 14.
-/
theorem minimum_disks_needed (capacity : ℝ) (files : List (ℝ × ℕ)) : 
  capacity = disk_capacity ∧ files = file_sizes → ∃ m : ℕ, m = 14 :=
by
  sorry

end NUMINAMATH_GPT_minimum_disks_needed_l192_19235


namespace NUMINAMATH_GPT_meaningful_expression_l192_19284

theorem meaningful_expression (m : ℝ) :
  (2 - m ≥ 0) ∧ (m + 2 ≠ 0) ↔ (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_l192_19284


namespace NUMINAMATH_GPT_rate_of_simple_interest_l192_19293

theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (P_nonzero : P ≠ 0) : 
  (P * R * T = P / 6) → R = 1 / 42 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rate_of_simple_interest_l192_19293


namespace NUMINAMATH_GPT_mistaken_fraction_l192_19219

theorem mistaken_fraction (n correct_result student_result : ℕ) (h1 : n = 384)
  (h2 : correct_result = (5 * n) / 16) (h3 : student_result = correct_result + 200) : 
  (student_result / n : ℚ) = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_mistaken_fraction_l192_19219


namespace NUMINAMATH_GPT_part_I_part_II_l192_19204

noncomputable def f (x : ℝ) (m : ℝ) := m - |x - 2|

theorem part_I (m : ℝ) : (∀ x, f (x + 1) m >= 0 → 0 <= x ∧ x <= 2) ↔ m = 1 := by
  sorry

theorem part_II (a b c : ℝ) (m : ℝ) : (1 / a + 1 / (2 * b) + 1 / (3 * c) = m) → (m = 1) → (a + 2 * b + 3 * c >= 9) := by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l192_19204


namespace NUMINAMATH_GPT_ryan_chinese_learning_hours_l192_19242

theorem ryan_chinese_learning_hours : 
    ∀ (h_english : ℕ) (diff : ℕ), 
    h_english = 7 → 
    h_english = 2 + (h_english - diff) → 
    diff = 5 := by
  intros h_english diff h_english_eq h_english_diff_eq
  sorry

end NUMINAMATH_GPT_ryan_chinese_learning_hours_l192_19242


namespace NUMINAMATH_GPT_Q_at_one_is_zero_l192_19240

noncomputable def Q (x : ℚ) : ℚ := x^4 - 2 * x^2 + 1

theorem Q_at_one_is_zero :
  Q 1 = 0 :=
by
  -- Here we would put the formal proof in Lean language
  sorry

end NUMINAMATH_GPT_Q_at_one_is_zero_l192_19240


namespace NUMINAMATH_GPT_total_height_increase_l192_19200

def height_increase_per_decade : ℕ := 90
def decades_in_two_centuries : ℕ := (2 * 100) / 10

theorem total_height_increase :
  height_increase_per_decade * decades_in_two_centuries = 1800 := by
  sorry

end NUMINAMATH_GPT_total_height_increase_l192_19200


namespace NUMINAMATH_GPT_total_distance_12_hours_l192_19228

-- Define the initial conditions for the speed and distance calculation
def speed_increase : ℕ → ℕ
  | 0 => 50
  | n + 1 => speed_increase n + 2

def distance_in_hour (n : ℕ) : ℕ := speed_increase n

-- Define the total distance traveled in 12 hours
def total_distance (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => total_distance n + distance_in_hour n

theorem total_distance_12_hours :
  total_distance 12 = 732 := by
  sorry

end NUMINAMATH_GPT_total_distance_12_hours_l192_19228


namespace NUMINAMATH_GPT_area_of_rectangle_l192_19250

theorem area_of_rectangle (s : ℝ) (h1 : 4 * s = 100) : 2 * s * 2 * s = 2500 := by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l192_19250


namespace NUMINAMATH_GPT_simplify_and_evaluate_l192_19230

variable (x y : ℤ)

noncomputable def given_expr := (x + y) ^ 2 - 3 * x * (x + y) + (x + 2 * y) * (x - 2 * y)

theorem simplify_and_evaluate : given_expr 1 (-1) = -3 :=
by
  -- The proof is to be completed here
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l192_19230


namespace NUMINAMATH_GPT_proof_a_eq_x_and_b_eq_x_pow_x_l192_19280

theorem proof_a_eq_x_and_b_eq_x_pow_x
  {a b x : ℕ}
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_x : 0 < x)
  (h : x^(a + b) = a^b * b) :
  a = x ∧ b = x^x := 
by
  sorry

end NUMINAMATH_GPT_proof_a_eq_x_and_b_eq_x_pow_x_l192_19280


namespace NUMINAMATH_GPT_pencil_length_l192_19294

theorem pencil_length (L : ℝ) 
  (h1 : 1 / 8 * L = b) 
  (h2 : 1 / 2 * (L - 1 / 8 * L) = w) 
  (h3 : (L - 1 / 8 * L - 1 / 2 * (L - 1 / 8 * L)) = 7 / 2) :
  L = 8 :=
sorry

end NUMINAMATH_GPT_pencil_length_l192_19294


namespace NUMINAMATH_GPT_range_of_3x_minus_2y_l192_19298

variable (x y : ℝ)

theorem range_of_3x_minus_2y (h1 : -1 ≤ x + y ∧ x + y ≤ 1) (h2 : 1 ≤ x - y ∧ x - y ≤ 5) :
  ∃ (a b : ℝ), 2 ≤ a ∧ a ≤ b ∧ b ≤ 13 ∧ (3 * x - 2 * y = a ∨ 3 * x - 2 * y = b) :=
by
  sorry

end NUMINAMATH_GPT_range_of_3x_minus_2y_l192_19298


namespace NUMINAMATH_GPT_prove_fraction_l192_19276

noncomputable def michael_brothers_problem (M O Y : ℕ) :=
  Y = 5 ∧
  M + O + Y = 28 ∧
  O = 2 * (M - 1) + 1 →
  Y / O = 1 / 3

theorem prove_fraction (M O Y : ℕ) : michael_brothers_problem M O Y :=
  sorry

end NUMINAMATH_GPT_prove_fraction_l192_19276


namespace NUMINAMATH_GPT_previous_monthly_income_l192_19273

variable (I : ℝ)

-- Conditions from the problem
def condition1 (I : ℝ) : Prop := 0.40 * I = 0.25 * (I + 600)

theorem previous_monthly_income (h : condition1 I) : I = 1000 := by
  sorry

end NUMINAMATH_GPT_previous_monthly_income_l192_19273


namespace NUMINAMATH_GPT_a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l192_19234

theorem a_sq_greater_than_b_sq_neither_sufficient_nor_necessary 
  (a b : ℝ) : ¬ ((a^2 > b^2) → (a > b)) ∧  ¬ ((a > b) → (a^2 > b^2)) := sorry

end NUMINAMATH_GPT_a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l192_19234


namespace NUMINAMATH_GPT_expression_takes_many_values_l192_19285

theorem expression_takes_many_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) :
  (∃ y : ℝ, y ≠ 0 ∧ y ≠ (y + 1) ∧ 
    (3 * x ^ 2 + 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 7) / ((x - 3) * (x + 2)) = y) :=
by
  sorry

end NUMINAMATH_GPT_expression_takes_many_values_l192_19285


namespace NUMINAMATH_GPT_evaluate_binom_mul_factorial_l192_19243

theorem evaluate_binom_mul_factorial (n : ℕ) (h : n > 0) :
  (Nat.choose (n + 2) n) * n! = ((n + 2) * (n + 1) * n!) / 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_binom_mul_factorial_l192_19243


namespace NUMINAMATH_GPT_anna_should_plant_8_lettuce_plants_l192_19231

/-- Anna wants to grow some lettuce in the garden and would like to grow enough to have at least
    12 large salads.
- Conditions:
  1. Half of the lettuce will be lost to insects and rabbits.
  2. Each lettuce plant is estimated to provide 3 large salads.
  
  Proof that Anna should plant 8 lettuce plants in the garden. --/
theorem anna_should_plant_8_lettuce_plants 
    (desired_salads: ℕ)
    (salads_per_plant: ℕ)
    (loss_fraction: ℚ) :
    desired_salads = 12 →
    salads_per_plant = 3 →
    loss_fraction = 1 / 2 →
    ∃ plants: ℕ, plants = 8 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_anna_should_plant_8_lettuce_plants_l192_19231


namespace NUMINAMATH_GPT_total_flour_used_l192_19291

theorem total_flour_used :
  let wheat_flour := 0.2
  let white_flour := 0.1
  let rye_flour := 0.15
  let almond_flour := 0.05
  let oat_flour := 0.1
  wheat_flour + white_flour + rye_flour + almond_flour + oat_flour = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_total_flour_used_l192_19291


namespace NUMINAMATH_GPT_alternating_sum_l192_19287

theorem alternating_sum : 
  (1 - 3 + 5 - 7 + 9 - 11 + 13 - 15 + 17 - 19 + 21 - 23 + 25 - 27 + 29 - 31 + 33 - 35 + 37 - 39 + 41 = 21) :=
by
  sorry

end NUMINAMATH_GPT_alternating_sum_l192_19287


namespace NUMINAMATH_GPT_fraction_evaluation_l192_19222

theorem fraction_evaluation : (1 / 2) + (1 / 2 * 1 / 2) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l192_19222


namespace NUMINAMATH_GPT_partnership_profit_l192_19212

noncomputable def totalProfit (P Q R : ℕ) (unit_value_per_share : ℕ) : ℕ :=
  let profit_p := 36 * 2 + 18 * 10
  let profit_q := 24 * 12
  let profit_r := 36 * 12
  (profit_p + profit_q + profit_r) * unit_value_per_share

theorem partnership_profit (P Q R : ℕ) (unit_value_per_share : ℕ) :
  (P / Q = 3 / 2) → (Q / R = 4 / 3) → 
  (unit_value_per_share = 144 / 288) → 
  totalProfit P Q R (unit_value_per_share * 1) = 486 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_partnership_profit_l192_19212


namespace NUMINAMATH_GPT_second_number_is_correct_l192_19270

theorem second_number_is_correct (x : Real) (h : 108^2 + x^2 = 19928) : x = Real.sqrt 8264 :=
by
  sorry

end NUMINAMATH_GPT_second_number_is_correct_l192_19270


namespace NUMINAMATH_GPT_factor_expression_l192_19253

theorem factor_expression (b : ℝ) : 180 * b ^ 2 + 36 * b = 36 * b * (5 * b + 1) :=
by
  -- actual proof is omitted
  sorry

end NUMINAMATH_GPT_factor_expression_l192_19253


namespace NUMINAMATH_GPT_problem_proof_l192_19256

-- Define I, J, and K respectively to be 9^20, 3^41, 3
def I : ℕ := 9^20
def J : ℕ := 3^41
def K : ℕ := 3

theorem problem_proof : I + I + I = J := by
  -- Lean structure placeholder
  sorry

end NUMINAMATH_GPT_problem_proof_l192_19256


namespace NUMINAMATH_GPT_fruit_count_correct_l192_19238

def george_oranges := 45
def amelia_oranges := george_oranges - 18
def amelia_apples := 15
def george_apples := amelia_apples + 5

def olivia_orange_rate := 3
def olivia_apple_rate := 2
def olivia_minutes := 30
def olivia_cycle_minutes := 5
def olivia_cycles := olivia_minutes / olivia_cycle_minutes
def olivia_oranges := olivia_orange_rate * olivia_cycles
def olivia_apples := olivia_apple_rate * olivia_cycles

def total_oranges := george_oranges + amelia_oranges + olivia_oranges
def total_apples := george_apples + amelia_apples + olivia_apples
def total_fruits := total_oranges + total_apples

theorem fruit_count_correct : total_fruits = 137 := by
  sorry

end NUMINAMATH_GPT_fruit_count_correct_l192_19238


namespace NUMINAMATH_GPT_students_not_enrolled_in_any_classes_l192_19275

/--
  At a particular college, 27.5% of the 1050 students are enrolled in biology,
  32.9% of the students are enrolled in mathematics, and 15% of the students are enrolled in literature classes.
  Assuming that no student is taking more than one of these specific subjects,
  the number of students at the college who are not enrolled in biology, mathematics, or literature classes is 260.

  We want to prove the statement:
    number_students_not_enrolled_in_any_classes = 260
-/
theorem students_not_enrolled_in_any_classes 
  (total_students : ℕ) 
  (biology_percent : ℝ) 
  (mathematics_percent : ℝ) 
  (literature_percent : ℝ) 
  (no_student_in_multiple : Prop) : 
  total_students = 1050 →
  biology_percent = 27.5 →
  mathematics_percent = 32.9 →
  literature_percent = 15 →
  (total_students - (⌊biology_percent / 100 * total_students⌋ + ⌊mathematics_percent / 100 * total_students⌋ + ⌊literature_percent / 100 * total_students⌋)) = 260 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_not_enrolled_in_any_classes_l192_19275


namespace NUMINAMATH_GPT_Jake_initial_balloons_l192_19290

theorem Jake_initial_balloons (J : ℕ) 
  (h1 : 6 = (J + 3) + 1) : 
  J = 2 :=
by
  sorry

end NUMINAMATH_GPT_Jake_initial_balloons_l192_19290


namespace NUMINAMATH_GPT_max_value_of_f_f_lt_x3_minus_2x2_l192_19226

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + Real.log x + b

theorem max_value_of_f (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) :
  f a b (Real.sqrt 2 / 2) = - (3 + 2 * Real.log 2) / 4 := by
  sorry

theorem f_lt_x3_minus_2x2 (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) (x : ℝ) (hx : 0 < x) :
  f a b x < x^3 - 2 * x^2 := by
  sorry

end NUMINAMATH_GPT_max_value_of_f_f_lt_x3_minus_2x2_l192_19226


namespace NUMINAMATH_GPT_identify_functions_l192_19205

-- Define the first expression
def expr1 (x : ℝ) : ℝ := x - (x - 3)

-- Define the second expression
noncomputable def expr2 (x : ℝ) : ℝ := Real.sqrt (x - 2) + Real.sqrt (1 - x)

-- Define the third expression
noncomputable def expr3 (x : ℝ) : ℝ :=
if x < 0 then x - 1 else x + 1

-- Define the fourth expression
noncomputable def expr4 (x : ℝ) : ℝ :=
if x ∈ Set.Ioo (-1) 1 then 0 else 1

-- Proof statement
theorem identify_functions :
  (∀ x, ∃! y, expr1 x = y) ∧ (∀ x, ∃! y, expr3 x = y) ∧
  (¬ ∃ x, ∃! y, expr2 x = y) ∧ (¬ ∀ x, ∃! y, expr4 x = y) := by
    sorry

end NUMINAMATH_GPT_identify_functions_l192_19205


namespace NUMINAMATH_GPT_problem1_problem2_l192_19268

variable {a b : ℝ}

theorem problem1 (ha : a ≠ 0) (hb : b ≠ 0) :
  (4 * b^3 / a) / (2 * b / a^2) = 2 * a * b^2 :=
by 
  sorry

theorem problem2 (ha : a ≠ b) :
  (a^2 / (a - b)) + (b^2 / (a - b)) - (2 * a * b / (a - b)) = a - b :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l192_19268


namespace NUMINAMATH_GPT_triangle_equilateral_of_angle_and_side_sequences_l192_19282

theorem triangle_equilateral_of_angle_and_side_sequences 
  (A B C : ℝ) (a b c : ℝ) 
  (h_angles_arith_seq: B = (A + C) / 2)
  (h_sides_geom_seq : b^2 = a * c) 
  (h_sum_angles : A + B + C = 180) 
  (h_pos_angles : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_triangle_equilateral_of_angle_and_side_sequences_l192_19282


namespace NUMINAMATH_GPT_cars_meet_in_two_hours_l192_19246

theorem cars_meet_in_two_hours (t : ℝ) (d : ℝ) (v1 v2 : ℝ) (h1 : d = 60) (h2 : v1 = 13) (h3 : v2 = 17) (h4 : v1 * t + v2 * t = d) : t = 2 := 
by
  sorry

end NUMINAMATH_GPT_cars_meet_in_two_hours_l192_19246


namespace NUMINAMATH_GPT_cost_price_of_book_l192_19233

-- Define the variables and conditions
variable (C : ℝ)
variable (P : ℝ)
variable (S : ℝ)

-- State the conditions given in the problem
def conditions := S = 260 ∧ P = 0.20 * C ∧ S = C + P

-- State the theorem
theorem cost_price_of_book (h : conditions C P S) : C = 216.67 :=
sorry

end NUMINAMATH_GPT_cost_price_of_book_l192_19233


namespace NUMINAMATH_GPT_arrangement_count_l192_19237

theorem arrangement_count (basil_plants tomato_plants : ℕ) (b : basil_plants = 5) (t : tomato_plants = 4) : 
  (Nat.factorial (basil_plants + 1) * Nat.factorial tomato_plants) = 17280 :=
by
  rw [b, t] 
  exact Eq.refl 17280

end NUMINAMATH_GPT_arrangement_count_l192_19237


namespace NUMINAMATH_GPT_vectors_parallel_l192_19254

-- Let s and n be the direction vector and normal vector respectively
def s : ℝ × ℝ × ℝ := (2, 1, 1)
def n : ℝ × ℝ × ℝ := (-4, -2, -2)

-- Statement that vectors s and n are parallel
theorem vectors_parallel : ∃ (k : ℝ), n = (k • s) := by
  use -2
  simp [s, n]
  sorry

end NUMINAMATH_GPT_vectors_parallel_l192_19254


namespace NUMINAMATH_GPT_value_of_y_at_48_l192_19202

open Real

noncomputable def collinear_points (x : ℝ) : ℝ :=
  if x = 2 then 5
  else if x = 6 then 17
  else if x = 10 then 29
  else if x = 48 then 143
  else 0 -- placeholder value for other x (not used in proof)

theorem value_of_y_at_48 :
  (∀ (x1 x2 x3 : ℝ), x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → 
    ∃ (m : ℝ), m = (collinear_points x2 - collinear_points x1) / (x2 - x1) ∧ 
               m = (collinear_points x3 - collinear_points x2) / (x3 - x2)) →
  collinear_points 48 = 143 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_at_48_l192_19202


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l192_19221

noncomputable def a : ℝ := 81 ^ 31
noncomputable def b : ℝ := 27 ^ 41
noncomputable def c : ℝ := 9 ^ 61

theorem relationship_between_a_b_c : c < b ∧ b < a := by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l192_19221


namespace NUMINAMATH_GPT_preferred_order_for_boy_l192_19203

variable (p q : ℝ)
variable (h : p < q)

theorem preferred_order_for_boy (p q : ℝ) (h : p < q) : 
  (2 * p * q - p^2 * q) > (2 * p * q - p * q^2) := 
sorry

end NUMINAMATH_GPT_preferred_order_for_boy_l192_19203


namespace NUMINAMATH_GPT_probability_at_least_one_defective_is_correct_l192_19279

/-- Define a box containing 21 bulbs, 4 of which are defective -/
def total_bulbs : ℕ := 21
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs

/-- Define probabilities of choosing non-defective bulbs -/
def prob_first_non_defective : ℚ := non_defective_bulbs / total_bulbs
def prob_second_non_defective : ℚ := (non_defective_bulbs - 1) / (total_bulbs - 1)

/-- Calculate the probability of both bulbs being non-defective -/
def prob_both_non_defective : ℚ := prob_first_non_defective * prob_second_non_defective

/-- Calculate the probability of at least one defective bulb -/
def prob_at_least_one_defective : ℚ := 1 - prob_both_non_defective

theorem probability_at_least_one_defective_is_correct :
  prob_at_least_one_defective = 37 / 105 :=
by
  -- Sorry allows us to skip the proof
  sorry

end NUMINAMATH_GPT_probability_at_least_one_defective_is_correct_l192_19279


namespace NUMINAMATH_GPT_value_of_f_g_6_squared_l192_19225

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem value_of_f_g_6_squared : (f (g 6))^2 = 26569 :=
by
  -- Place your proof here
  sorry

end NUMINAMATH_GPT_value_of_f_g_6_squared_l192_19225


namespace NUMINAMATH_GPT_right_triangle_area_inscribed_3_4_l192_19295

theorem right_triangle_area_inscribed_3_4 (r1 r2: ℝ) (h1 : r1 = 3) (h2 : r2 = 4) : 
  ∃ (S: ℝ), S = 150 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_inscribed_3_4_l192_19295


namespace NUMINAMATH_GPT_solve_system_l192_19229

theorem solve_system :
  ∃! (x y : ℝ), (2 * x + y + 8 ≤ 0) ∧ (x^4 + 2 * x^2 * y^2 + y^4 + 9 - 10 * x^2 - 10 * y^2 = 8 * x * y) ∧ (x = -3 ∧ y = -2) := 
  by
  sorry

end NUMINAMATH_GPT_solve_system_l192_19229


namespace NUMINAMATH_GPT_geom_progr_sum_eq_l192_19248

variable (a b q : ℝ) (n p : ℕ)

theorem geom_progr_sum_eq (h : a * (1 - q ^ (n * p)) / (1 - q) = b * (1 - q ^ (n * p)) / (1 - q ^ p)) :
  b = a * (1 - q ^ p) / (1 - q) :=
by
  sorry

end NUMINAMATH_GPT_geom_progr_sum_eq_l192_19248


namespace NUMINAMATH_GPT_second_solution_salt_percent_l192_19207

theorem second_solution_salt_percent (S : ℝ) (x : ℝ) 
  (h1 : 0.14 * S - 0.14 * (S / 4) + (x / 100) * (S / 4) = 0.16 * S) : 
  x = 22 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_second_solution_salt_percent_l192_19207


namespace NUMINAMATH_GPT_interest_rate_of_additional_investment_l192_19262

section
variable (r : ℝ)

theorem interest_rate_of_additional_investment
  (h : 2800 * 0.05 + 1400 * r = 0.06 * (2800 + 1400)) :
  r = 0.08 := by
  sorry
end

end NUMINAMATH_GPT_interest_rate_of_additional_investment_l192_19262


namespace NUMINAMATH_GPT_max_earth_to_sun_distance_l192_19283

-- Define the semi-major axis a and semi-focal distance c
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_focal_distance : ℝ := 3 * 10^6

-- Define the maximum distance from the Earth to the Sun
def max_distance (a c : ℝ) : ℝ := a + c

-- Define the Lean statement to be proved
theorem max_earth_to_sun_distance :
  max_distance semi_major_axis semi_focal_distance = 1.53 * 10^8 :=
by
  -- skipping the proof for now
  sorry

end NUMINAMATH_GPT_max_earth_to_sun_distance_l192_19283


namespace NUMINAMATH_GPT_strictly_increasing_range_l192_19208

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 3) * x + 1 else a ^ x

theorem strictly_increasing_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_strictly_increasing_range_l192_19208


namespace NUMINAMATH_GPT_RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l192_19216

-- Definitions representing the conditions
def TotalCrates : ℕ := 50
def PriceGrapes : ℕ := 15
def PriceMangoes : ℕ := 20
def PricePassionFruits : ℕ := 25
def CratesGrapes : ℕ := 13
def CratesMangoes : ℕ := 20
def CratesPassionFruits : ℕ := TotalCrates - CratesGrapes - CratesMangoes

def RevenueWithoutDiscounts : ℕ :=
  (CratesGrapes * PriceGrapes) +
  (CratesMangoes * PriceMangoes) +
  (CratesPassionFruits * PricePassionFruits)

def DiscountGrapes : Float := if CratesGrapes > 10 then 0.10 else 0.0
def DiscountMangoes : Float := if CratesMangoes > 15 then 0.15 else 0.0
def DiscountPassionFruits : Float := if CratesPassionFruits > 5 then 0.20 else 0.0

def DiscountedPrice (price : ℕ) (discount : Float) : Float := 
  price.toFloat * (1.0 - discount)

def RevenueWithDiscounts : Float :=
  (CratesGrapes.toFloat * DiscountedPrice PriceGrapes DiscountGrapes) +
  (CratesMangoes.toFloat * DiscountedPrice PriceMangoes DiscountMangoes) +
  (CratesPassionFruits.toFloat * DiscountedPrice PricePassionFruits DiscountPassionFruits)

-- Proof problems
theorem RevenueWithoutDiscounts_is_1020 : RevenueWithoutDiscounts = 1020 := sorry
theorem RevenueWithDiscounts_is_855_5 : RevenueWithDiscounts = 855.5 := sorry
theorem Difference_is_164_5 : (RevenueWithoutDiscounts.toFloat - RevenueWithDiscounts) = 164.5 := sorry

end NUMINAMATH_GPT_RevenueWithoutDiscounts_is_1020_RevenueWithDiscounts_is_855_5_Difference_is_164_5_l192_19216


namespace NUMINAMATH_GPT_solve_dividend_and_divisor_l192_19220

-- Definitions for base, digits, and mathematical relationships
def base := 5
def P := 1
def Q := 2
def R := 3
def S := 4
def T := 0
def Dividend := 1 * base^6 + 2 * base^5 + 3 * base^4 + 4 * base^3 + 3 * base^2 + 2 * base^1 + 1 * base^0
def Divisor := 2 * base^2 + 3 * base^1 + 2 * base^0

-- The conditions given in the math problem
axiom condition_1 : Q + R = base
axiom condition_2 : P + 1 = Q
axiom condition_3 : Q + P = R
axiom condition_4 : S = 2 * Q
axiom condition_5 : Q^2 = S
axiom condition_6 : Dividend = 24336
axiom condition_7 : Divisor = 67

-- The goal
theorem solve_dividend_and_divisor : Dividend = 24336 ∧ Divisor = 67 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_dividend_and_divisor_l192_19220


namespace NUMINAMATH_GPT_solve_y_l192_19232

theorem solve_y 
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (remainder_condition : x = (96.12 * y))
  (division_condition : x = (96.0624 * y + 5.76)) : 
  y = 100 := 
 sorry

end NUMINAMATH_GPT_solve_y_l192_19232


namespace NUMINAMATH_GPT_unique_zero_function_l192_19252

variable (f : ℝ → ℝ)

theorem unique_zero_function (h : ∀ x y : ℝ, f (x + y) = f x - f y) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_unique_zero_function_l192_19252


namespace NUMINAMATH_GPT_largest_value_l192_19213

theorem largest_value :
  max (max (max (max (4^2) (4 * 2)) (4 - 2)) (4 / 2)) (4 + 2) = 4^2 :=
by sorry

end NUMINAMATH_GPT_largest_value_l192_19213


namespace NUMINAMATH_GPT_playerA_winning_moves_l192_19289

-- Definitions of the game
-- Circles are labeled from 1 to 9
inductive Circle
| A | B | C1 | C2 | C3 | C4 | C5 | C6 | C7

inductive Player
| A | B

def StraightLine (c1 c2 c3 : Circle) : Prop := sorry
-- The straight line property between circles is specified by the game rules

-- Initial conditions
def initial_conditions (playerA_move playerB_move : Circle) : Prop :=
  playerA_move = Circle.A ∧ playerB_move = Circle.B

-- Winning condition
def winning_move (move : Circle) : Prop := sorry
-- This will check if a move leads to a win for Player A

-- Equivalent proof problem
theorem playerA_winning_moves : ∀ (move : Circle), initial_conditions Circle.A Circle.B → 
  (move = Circle.C2 ∨ move = Circle.C3 ∨ move = Circle.C4) → winning_move move :=
by
  sorry

end NUMINAMATH_GPT_playerA_winning_moves_l192_19289


namespace NUMINAMATH_GPT_path_problem_l192_19236

noncomputable def path_bounds (N : ℕ) (h : 0 < N) : Prop :=
  ∃ p : ℕ, 4 * N ≤ p ∧ p ≤ 2 * N^2 + 2 * N

theorem path_problem (N : ℕ) (h : 0 < N) : path_bounds N h :=
  sorry

end NUMINAMATH_GPT_path_problem_l192_19236


namespace NUMINAMATH_GPT_image_of_center_after_transformations_l192_19265

-- Define the initial center of circle C
def initial_center : ℝ × ℝ := (3, -4)

-- Define a function to reflect a point across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define a function to translate a point by some units left
def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Define the final coordinates after transformations
def final_center : ℝ × ℝ :=
  translate_left (reflect_x_axis initial_center) 5

-- The theorem to prove
theorem image_of_center_after_transformations :
  final_center = (-2, 4) :=
by
  sorry

end NUMINAMATH_GPT_image_of_center_after_transformations_l192_19265


namespace NUMINAMATH_GPT_no_four_digit_numbers_divisible_by_11_l192_19299

theorem no_four_digit_numbers_divisible_by_11 (a b c d : ℕ) :
  (a + b + c + d = 9) ∧ ((a + c) - (b + d)) % 11 = 0 → false :=
by
  sorry

end NUMINAMATH_GPT_no_four_digit_numbers_divisible_by_11_l192_19299


namespace NUMINAMATH_GPT_find_radius_l192_19218

theorem find_radius (QP QO r : ℝ) (hQP : QP = 420) (hQO : QO = 427) : r = 77 :=
by
  -- Given QP^2 + r^2 = QO^2
  have h : (QP ^ 2) + (r ^ 2) = (QO ^ 2) := sorry
  -- Calculate the squares
  have h1 : (420 ^ 2) = 176400 := sorry
  have h2 : (427 ^ 2) = 182329 := sorry
  -- r^2 = 182329 - 176400
  have h3 : r ^ 2 = 5929 := sorry
  -- Therefore, r = 77
  exact sorry

end NUMINAMATH_GPT_find_radius_l192_19218


namespace NUMINAMATH_GPT_daniel_video_games_l192_19239

/--
Daniel has a collection of some video games. 80 of them, Daniel bought for $12 each.
Of the rest, 50% were bought for $7. All others had a price of $3 each.
Daniel spent $2290 on all the games in his collection.
Prove that the total number of video games in Daniel's collection is 346.
-/
theorem daniel_video_games (n : ℕ) (r : ℕ)
    (h₀ : 80 * 12 = 960)
    (h₁ : 2290 - 960 = 1330)
    (h₂ : r / 2 * 7 + r / 2 * 3 = 1330):
    n = 80 + r → n = 346 :=
by
  intro h_total
  have r_eq : r = 266 := by sorry
  rw [r_eq] at h_total
  exact h_total

end NUMINAMATH_GPT_daniel_video_games_l192_19239


namespace NUMINAMATH_GPT_average_age_of_inhabitants_l192_19263

theorem average_age_of_inhabitants (H M : ℕ) (avg_age_men avg_age_women : ℕ)
  (ratio_condition : 2 * M = 3 * H)
  (men_avg_age_condition : avg_age_men = 37)
  (women_avg_age_condition : avg_age_women = 42) :
  ((H * 37) + (M * 42)) / (H + M) = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_inhabitants_l192_19263


namespace NUMINAMATH_GPT_simplify_expression_l192_19215

theorem simplify_expression (x : ℝ) : 3 * x + 4 - x + 8 = 2 * x + 12 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l192_19215


namespace NUMINAMATH_GPT_fraction_value_l192_19271

variable (u v w x : ℝ)

-- Conditions
def cond1 : Prop := u / v = 5
def cond2 : Prop := w / v = 3
def cond3 : Prop := w / x = 2 / 3

theorem fraction_value (h1 : cond1 u v) (h2 : cond2 w v) (h3 : cond3 w x) : x / u = 9 / 10 := 
by
  sorry

end NUMINAMATH_GPT_fraction_value_l192_19271
