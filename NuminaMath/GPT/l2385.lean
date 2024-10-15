import Mathlib

namespace NUMINAMATH_GPT_candy_bars_to_buy_l2385_238504

variable (x : ℕ)

theorem candy_bars_to_buy (h1 : 25 * x + 2 * 75 + 50 = 11 * 25) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_candy_bars_to_buy_l2385_238504


namespace NUMINAMATH_GPT_range_of_m_l2385_238538

theorem range_of_m (m : ℝ) : 0 < m ∧ m < 2 ↔ (2 - m > 0 ∧ - (1 / 2) * m < 0) := by
  sorry

end NUMINAMATH_GPT_range_of_m_l2385_238538


namespace NUMINAMATH_GPT_find_value_of_x2001_plus_y2001_l2385_238568

theorem find_value_of_x2001_plus_y2001 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
x ^ 2001 + y ^ 2001 = 2 ^ 2001 ∨ x ^ 2001 + y ^ 2001 = -2 ^ 2001 := by
  sorry

end NUMINAMATH_GPT_find_value_of_x2001_plus_y2001_l2385_238568


namespace NUMINAMATH_GPT_max_min_value_function_l2385_238548

noncomputable def given_function (x : ℝ) : ℝ :=
  (Real.sin x) ^ 2 + Real.cos x + 1

theorem max_min_value_function :
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≤ 9 / 4) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 9 / 4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 2) := by
  sorry

end NUMINAMATH_GPT_max_min_value_function_l2385_238548


namespace NUMINAMATH_GPT_scientific_notation_1742000_l2385_238512

theorem scientific_notation_1742000 : 1742000 = 1.742 * 10^6 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_1742000_l2385_238512


namespace NUMINAMATH_GPT_abby_bridget_chris_probability_l2385_238590

noncomputable def seatingProbability : ℚ :=
  let totalArrangements := 720
  let favorableArrangements := 114
  favorableArrangements / totalArrangements

theorem abby_bridget_chris_probability :
  seatingProbability = 19 / 120 :=
by
  simp [seatingProbability]
  sorry

end NUMINAMATH_GPT_abby_bridget_chris_probability_l2385_238590


namespace NUMINAMATH_GPT_bisection_min_calculations_l2385_238521

theorem bisection_min_calculations 
  (a b : ℝ)
  (h_interval : a = 1.4 ∧ b = 1.5)
  (delta : ℝ)
  (h_delta : delta = 0.001) :
  ∃ n : ℕ, 0.1 / (2 ^ n) ≤ delta ∧ n = 7 :=
sorry

end NUMINAMATH_GPT_bisection_min_calculations_l2385_238521


namespace NUMINAMATH_GPT_a_gt_b_neither_sufficient_nor_necessary_l2385_238580

theorem a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) := 
sorry

end NUMINAMATH_GPT_a_gt_b_neither_sufficient_nor_necessary_l2385_238580


namespace NUMINAMATH_GPT_aarti_work_multiple_l2385_238526

-- Aarti can do a piece of work in 5 days
def days_per_unit_work := 5

-- It takes her 15 days to complete the certain multiple of work
def days_for_multiple_work := 15

-- Prove the ratio of the days for multiple work to the days per unit work equals 3
theorem aarti_work_multiple :
  days_for_multiple_work / days_per_unit_work = 3 :=
sorry

end NUMINAMATH_GPT_aarti_work_multiple_l2385_238526


namespace NUMINAMATH_GPT_sin_cos_identity_l2385_238515

variable (α : Real)

theorem sin_cos_identity (h : Real.sin α - Real.cos α = -5/4) : Real.sin α * Real.cos α = -9/32 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l2385_238515


namespace NUMINAMATH_GPT_daisy_germination_rate_theorem_l2385_238586

-- Define the conditions of the problem
variables (daisySeeds sunflowerSeeds : ℕ) (sunflowerGermination flowerProduction finalFlowerPlants : ℝ)
def conditions : Prop :=
  daisySeeds = 25 ∧ sunflowerSeeds = 25 ∧ sunflowerGermination = 0.80 ∧ flowerProduction = 0.80 ∧ finalFlowerPlants = 28

-- Define the statement that the germination rate of the daisy seeds is 60%
def germination_rate_of_daisy_seeds : Prop :=
  ∃ (daisyGerminationRate : ℝ), (conditions daisySeeds sunflowerSeeds sunflowerGermination flowerProduction finalFlowerPlants) →
  daisyGerminationRate = 0.60

-- The proof is omitted - note this is just the statement
theorem daisy_germination_rate_theorem : germination_rate_of_daisy_seeds 25 25 0.80 0.80 28 :=
sorry

end NUMINAMATH_GPT_daisy_germination_rate_theorem_l2385_238586


namespace NUMINAMATH_GPT_factor_81_minus_36x4_l2385_238530

theorem factor_81_minus_36x4 (x : ℝ) : 
    81 - 36 * x^4 = 9 * (Real.sqrt 3 - Real.sqrt 2 * x) * (Real.sqrt 3 + Real.sqrt 2 * x) * (3 + 2 * x^2) :=
sorry

end NUMINAMATH_GPT_factor_81_minus_36x4_l2385_238530


namespace NUMINAMATH_GPT_largest_int_less_than_100_with_remainder_5_l2385_238566

theorem largest_int_less_than_100_with_remainder_5 (x : ℤ) (n : ℤ) (h₁ : x = 7 * n + 5) (h₂ : x < 100) : 
  x = 96 := by
  sorry

end NUMINAMATH_GPT_largest_int_less_than_100_with_remainder_5_l2385_238566


namespace NUMINAMATH_GPT_union_sets_l2385_238597

-- Define the sets A and B as conditions
def A : Set ℝ := {0, 1}  -- Since lg 1 = 0
def B : Set ℝ := {-1, 0}

-- Define that A union B equals {-1, 0, 1}
theorem union_sets : A ∪ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_union_sets_l2385_238597


namespace NUMINAMATH_GPT_find_grade_2_l2385_238576

-- Definitions for the problem
def grade_1 := 78
def weight_1 := 20
def weight_2 := 30
def grade_3 := 90
def weight_3 := 10
def grade_4 := 85
def weight_4 := 40
def overall_average := 83

noncomputable def calc_weighted_average (G : ℕ) : ℝ :=
  (grade_1 * weight_1 + G * weight_2 + grade_3 * weight_3 + grade_4 * weight_4) / (weight_1 + weight_2 + weight_3 + weight_4)

theorem find_grade_2 (G : ℕ) : calc_weighted_average G = overall_average → G = 81 := sorry

end NUMINAMATH_GPT_find_grade_2_l2385_238576


namespace NUMINAMATH_GPT_tangent_line_at_slope_two_l2385_238547

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1

theorem tangent_line_at_slope_two :
  ∃ (x₀ y₀ : ℝ), (deriv curve x₀ = 2) ∧ (curve x₀ = y₀) ∧ (∀ x, (2 * (x - x₀) + y₀) = (2 * x)) :=
by 
  sorry

end NUMINAMATH_GPT_tangent_line_at_slope_two_l2385_238547


namespace NUMINAMATH_GPT_initial_goldfish_eq_15_l2385_238575

-- Let's define our setup as per the conditions provided
def fourGoldfishLeft := 4
def elevenGoldfishDisappeared := 11

-- Our main statement that we need to prove
theorem initial_goldfish_eq_15 : fourGoldfishLeft + elevenGoldfishDisappeared = 15 := by
  sorry

end NUMINAMATH_GPT_initial_goldfish_eq_15_l2385_238575


namespace NUMINAMATH_GPT_simplified_form_l2385_238571

def simplify_expression (x : ℝ) : ℝ :=
  (3 * x - 2) * (6 * x ^ 8 + 3 * x ^ 7 - 2 * x ^ 3 + x)

theorem simplified_form (x : ℝ) : 
  simplify_expression x = 18 * x ^ 9 - 3 * x ^ 8 - 6 * x ^ 7 - 6 * x ^ 4 - 4 * x ^ 3 + x :=
by
  sorry

end NUMINAMATH_GPT_simplified_form_l2385_238571


namespace NUMINAMATH_GPT_regular_triangular_prism_cosine_l2385_238549

-- Define the regular triangular prism and its properties
structure RegularTriangularPrism :=
  (side : ℝ) -- the side length of the base and the lateral edge

-- Define the vertices of the prism
structure Vertices :=
  (A : ℝ × ℝ × ℝ) 
  (B : ℝ × ℝ × ℝ) 
  (C : ℝ × ℝ × ℝ)
  (A1 : ℝ × ℝ × ℝ)
  (B1 : ℝ × ℝ × ℝ)
  (C1 : ℝ × ℝ × ℝ)

-- Define the cosine calculation
def cos_angle (prism : RegularTriangularPrism) (v : Vertices) : ℝ := sorry

-- Prove that the cosine of the angle between diagonals AB1 and BC1 is 1/4
theorem regular_triangular_prism_cosine (prism : RegularTriangularPrism) (v : Vertices)
  : cos_angle prism v = 1 / 4 :=
sorry

end NUMINAMATH_GPT_regular_triangular_prism_cosine_l2385_238549


namespace NUMINAMATH_GPT_baseball_attendance_difference_l2385_238517

theorem baseball_attendance_difference:
  ∃ C D: ℝ, 
    (59500 ≤ C ∧ C ≤ 80500 ∧ 69565 ≤ D ∧ D ≤ 94118) ∧ 
    (max (D - C) (C - D) = 35000 ∧ min (D - C) (C - D) = 11000) := by
  sorry

end NUMINAMATH_GPT_baseball_attendance_difference_l2385_238517


namespace NUMINAMATH_GPT_cube_cross_section_area_l2385_238594

def cube_edge_length (a : ℝ) := a > 0

def plane_perpendicular_body_diagonal := 
  ∃ (p : ℝ × ℝ × ℝ), ∀ (x y z : ℝ), 
  p = (x / 2, y / 2, z / 2) ∧ 
  (x + y + z) = (1 : ℝ)

theorem cube_cross_section_area
  (a : ℝ) 
  (h : cube_edge_length a) 
  (plane : plane_perpendicular_body_diagonal) : 
  ∃ (A : ℝ), 
  A = (3 * a^2 * Real.sqrt 3 / 4) := sorry

end NUMINAMATH_GPT_cube_cross_section_area_l2385_238594


namespace NUMINAMATH_GPT_evaluate_g_at_8_l2385_238542

def g (n : ℕ) : ℕ := n^2 - 3 * n + 29

theorem evaluate_g_at_8 : g 8 = 69 := by
  unfold g
  calc
    8^2 - 3 * 8 + 29 = 64 - 24 + 29 := by simp
                      _ = 69 := by norm_num

end NUMINAMATH_GPT_evaluate_g_at_8_l2385_238542


namespace NUMINAMATH_GPT_problem_gcf_lcm_sum_l2385_238587

-- Let A be the GCF of {15, 20, 30}
def A : ℕ := Nat.gcd (Nat.gcd 15 20) 30

-- Let B be the LCM of {15, 20, 30}
def B : ℕ := Nat.lcm (Nat.lcm 15 20) 30

-- We need to prove that A + B = 65
theorem problem_gcf_lcm_sum :
  A + B = 65 :=
by
  sorry

end NUMINAMATH_GPT_problem_gcf_lcm_sum_l2385_238587


namespace NUMINAMATH_GPT_ratio_of_areas_l2385_238573

-- Definitions based on the conditions given
variables (A B M N P Q O : Type) 
variables (AB BM BP : ℝ)

-- Assumptions
axiom hAB : AB = 6
axiom hBM : BM = 9
axiom hBP : BP = 5

-- Theorem statement
theorem ratio_of_areas (hMN : M ≠ N) (hPQ : P ≠ Q) :
  (1 / 121 : ℝ) = sorry :=
by sorry

end NUMINAMATH_GPT_ratio_of_areas_l2385_238573


namespace NUMINAMATH_GPT_part1_zero_of_f_part2_a_range_l2385_238535

-- Define the given function f
def f (x a b : ℝ) : ℝ := (x - a) * |x| + b

-- Define the problem statement for Part 1
theorem part1_zero_of_f :
  ∀ (x : ℝ),
    f x 2 3 = 0 ↔ x = -1 := 
by
  sorry

-- Define the problem statement for Part 2
theorem part2_a_range :
  ∀ (a : ℝ),
    (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → f x a (-2) < 0) ↔ a > -1 :=
by
  sorry

end NUMINAMATH_GPT_part1_zero_of_f_part2_a_range_l2385_238535


namespace NUMINAMATH_GPT_ratio_five_to_one_l2385_238595

theorem ratio_five_to_one (x : ℕ) (h : 5 * 12 = x) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_ratio_five_to_one_l2385_238595


namespace NUMINAMATH_GPT_cafeteria_apples_count_l2385_238519

def initial_apples : ℕ := 17
def used_monday : ℕ := 2
def bought_monday : ℕ := 23
def used_tuesday : ℕ := 4
def bought_tuesday : ℕ := 15
def used_wednesday : ℕ := 3

def final_apples (initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday : ℕ) : ℕ :=
  initial_apples - used_monday + bought_monday - used_tuesday + bought_tuesday - used_wednesday

theorem cafeteria_apples_count :
  final_apples initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday = 46 :=
by
  sorry

end NUMINAMATH_GPT_cafeteria_apples_count_l2385_238519


namespace NUMINAMATH_GPT_correct_sequence_of_linear_regression_analysis_l2385_238558

def linear_regression_steps : List ℕ := [2, 4, 3, 1]

theorem correct_sequence_of_linear_regression_analysis :
  linear_regression_steps = [2, 4, 3, 1] :=
by
  sorry

end NUMINAMATH_GPT_correct_sequence_of_linear_regression_analysis_l2385_238558


namespace NUMINAMATH_GPT_ad_space_length_l2385_238559

theorem ad_space_length 
  (num_companies : ℕ)
  (ads_per_company : ℕ)
  (width : ℝ)
  (cost_per_sq_ft : ℝ)
  (total_cost : ℝ) 
  (H1 : num_companies = 3)
  (H2 : ads_per_company = 10)
  (H3 : width = 5)
  (H4 : cost_per_sq_ft = 60)
  (H5 : total_cost = 108000) :
  ∃ L : ℝ, (num_companies * ads_per_company * width * L * cost_per_sq_ft = total_cost) ∧ (L = 12) :=
by
  sorry

end NUMINAMATH_GPT_ad_space_length_l2385_238559


namespace NUMINAMATH_GPT_sequence_properties_l2385_238588

noncomputable def a (n : ℕ) : ℕ := 3 * n - 1

def S (n : ℕ) : ℕ := n * (2 + 3 * n - 1) / 2

theorem sequence_properties :
  a 5 + a 7 = 34 ∧ ∀ n, S n = (3 * n ^ 2 + n) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l2385_238588


namespace NUMINAMATH_GPT_sum_of_consecutive_page_numbers_l2385_238543

def consecutive_page_numbers_product_and_sum (n m : ℤ) :=
  n * m = 20412

theorem sum_of_consecutive_page_numbers (n : ℤ) (h1 : consecutive_page_numbers_product_and_sum n (n + 1)) : n + (n + 1) = 285 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_page_numbers_l2385_238543


namespace NUMINAMATH_GPT_find_m_l2385_238506

noncomputable def g (n : ℤ) : ℤ :=
if n % 2 ≠ 0 then 2 * n + 3
else if n % 3 = 0 then n / 3
else n - 1

theorem find_m :
  ∃ m : ℤ, m % 2 ≠ 0 ∧ g (g (g m)) = 36 ∧ m = 54 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2385_238506


namespace NUMINAMATH_GPT_plates_count_l2385_238592

variable (x : ℕ)
variable (first_taken : ℕ)
variable (second_taken : ℕ)
variable (remaining_plates : ℕ := 9)

noncomputable def plates_initial : ℕ :=
  let first_batch := (x - 2) / 3
  let remaining_after_first := x - 2 - first_batch
  let second_batch := remaining_after_first / 2
  let remaining_after_second := remaining_after_first - second_batch
  remaining_after_second

theorem plates_count (x : ℕ) (h : plates_initial x = remaining_plates) : x = 29 := sorry

end NUMINAMATH_GPT_plates_count_l2385_238592


namespace NUMINAMATH_GPT_cost_of_birthday_gift_l2385_238552

theorem cost_of_birthday_gift 
  (boss_contrib : ℕ)
  (todd_contrib : ℕ)
  (employee_contrib : ℕ)
  (num_employees : ℕ)
  (h1 : boss_contrib = 15)
  (h2 : todd_contrib = 2 * boss_contrib)
  (h3 : employee_contrib = 11)
  (h4 : num_employees = 5) :
  boss_contrib + todd_contrib + num_employees * employee_contrib = 100 := by
  sorry

end NUMINAMATH_GPT_cost_of_birthday_gift_l2385_238552


namespace NUMINAMATH_GPT_find_value_of_a_l2385_238598

noncomputable def f (x a : ℝ) : ℝ := (3 * Real.log x - x^2 - a - 2)^2 + (x - a)^2

theorem find_value_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ 8) ↔ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l2385_238598


namespace NUMINAMATH_GPT_tan_alpha_solution_l2385_238501

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end NUMINAMATH_GPT_tan_alpha_solution_l2385_238501


namespace NUMINAMATH_GPT_sum_of_extremes_of_g_l2385_238584

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - abs (2 * x - 8)

theorem sum_of_extremes_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≤ g 4) ∧ (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≥ g 1) → g 4 + g 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_extremes_of_g_l2385_238584


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l2385_238537

-- Given conditions and questions
variable (x y : ℝ)
variable (h : (x - y)^2 - 2 * (x + y) + 1 = 0)

-- Part (a): Prove neither x nor y can be negative
theorem part_a (h : (x - y)^2 - 2 * (x + y) + 1 = 0) : x ≥ 0 ∧ y ≥ 0 := 
sorry

-- Part (b): Prove if x > 1 and y < x, then sqrt{x} - sqrt{y} = 1
theorem part_b (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x > 1) (hy : y < x) : 
  Real.sqrt x - Real.sqrt y = 1 := 
sorry

-- Part (c): Prove if x < 1 and y < 1, then sqrt{x} + sqrt{y} = 1
theorem part_c (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x < 1) (hy : y < 1) : 
  Real.sqrt x + Real.sqrt y = 1 := 
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l2385_238537


namespace NUMINAMATH_GPT_imaginary_unit_real_part_eq_l2385_238556

theorem imaginary_unit_real_part_eq (a : ℝ) (i : ℂ) (h : i * i = -1) :
  (∃ r : ℝ, ((3 + i) * (a + 2 * i) / (1 + i) = r)) → a = 4 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_unit_real_part_eq_l2385_238556


namespace NUMINAMATH_GPT_correct_option_is_A_l2385_238536

-- Define the conditions
def chromosome_counts (phase : String) (is_meiosis : Bool) : Nat :=
  if phase = "prophase" then 2
  else if phase = "metaphase" then 2
  else if phase = "anaphase" then if is_meiosis then 2 else 4
  else if phase = "telophase" then if is_meiosis then 1 else 2
  else 0

def dna_counts (phase : String) (is_meiosis : Bool) : Nat :=
  if phase = "prophase" then 4
  else if phase = "metaphase" then 4
  else if phase = "anaphase" then 4
  else if phase = "telophase" then 2
  else 0

def chromosome_behavior (phase : String) (is_meiosis : Bool) : String :=
  if is_meiosis && phase = "prophase" then "synapsis"
  else if is_meiosis && phase = "metaphase" then "tetrad formation"
  else if is_meiosis && phase = "anaphase" then "separation"
  else if is_meiosis && phase = "telophase" then "recombination"
  else "no special behavior"

-- Problem statement in terms of a Lean theorem
theorem correct_option_is_A :
  ∀ (phase : String),
  (chromosome_counts phase false = chromosome_counts phase true ∧
   chromosome_behavior phase false ≠ chromosome_behavior phase true ∧
   dna_counts phase false ≠ dna_counts phase true) →
  "A" = "A" :=
by 
  intro phase 
  simp only [imp_self]
  sorry

end NUMINAMATH_GPT_correct_option_is_A_l2385_238536


namespace NUMINAMATH_GPT_ratio_of_sector_CPD_l2385_238524

-- Define the given angles
def angle_AOC : ℝ := 40
def angle_DOB : ℝ := 60
def angle_COP : ℝ := 110

-- Calculate the angle CPD
def angle_CPD : ℝ := angle_COP - angle_AOC - angle_DOB

-- State the theorem to prove the ratio
theorem ratio_of_sector_CPD (hAOC : angle_AOC = 40) (hDOB : angle_DOB = 60)
(hCOP : angle_COP = 110) : 
  angle_CPD / 360 = 1 / 36 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_ratio_of_sector_CPD_l2385_238524


namespace NUMINAMATH_GPT_cos_A_side_c_l2385_238550

-- helper theorem for cosine rule usage
theorem cos_A (a b c : ℝ) (cosA cosB cosC : ℝ) (h : 3 * a * cosA = c * cosB + b * cosC) : cosA = 1 / 3 :=
by
  sorry

-- main statement combining conditions 1 and 2 with side value results
theorem side_c (a b c : ℝ) (cosA cosB cosC : ℝ) (h1 : 3 * a * cosA = c * cosB + b * cosC) (h2 : cosB + cosC = 0) (h3 : a = 1) : c = 2 :=
by
  have h_cosA : cosA = 1 / 3 := cos_A a b c cosA cosB cosC h1
  sorry

end NUMINAMATH_GPT_cos_A_side_c_l2385_238550


namespace NUMINAMATH_GPT_xiao_gao_actual_score_l2385_238554

-- Definitions from the conditions:
def standard_score : ℕ := 80
def xiao_gao_recorded_score : ℤ := 12

-- Proof problem statement:
theorem xiao_gao_actual_score : (standard_score : ℤ) + xiao_gao_recorded_score = 92 :=
by
  sorry

end NUMINAMATH_GPT_xiao_gao_actual_score_l2385_238554


namespace NUMINAMATH_GPT_solve_inequality_l2385_238577

theorem solve_inequality : 
  {x : ℝ | -x^2 - 2*x + 3 ≤ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2385_238577


namespace NUMINAMATH_GPT_remainder_of_2_pow_2017_mod_11_l2385_238503

theorem remainder_of_2_pow_2017_mod_11 : (2 ^ 2017) % 11 = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_of_2_pow_2017_mod_11_l2385_238503


namespace NUMINAMATH_GPT_complex_number_quadrant_l2385_238525

def i := Complex.I
def z := i * (1 + i)

theorem complex_number_quadrant 
  : z.re < 0 ∧ z.im > 0 := 
by
  sorry

end NUMINAMATH_GPT_complex_number_quadrant_l2385_238525


namespace NUMINAMATH_GPT_multiples_of_4_between_50_and_300_l2385_238516

theorem multiples_of_4_between_50_and_300 : 
  (∃ n : ℕ, 50 < n ∧ n < 300 ∧ n % 4 = 0) ∧ 
  (∃ k : ℕ, k = 62) :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_4_between_50_and_300_l2385_238516


namespace NUMINAMATH_GPT_complement_P_eq_Ioo_l2385_238507

def U : Set ℝ := Set.univ
def P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_of_P_in_U : Set ℝ := Set.Ioo (-1) 6

theorem complement_P_eq_Ioo :
  (U \ P) = complement_of_P_in_U :=
by sorry

end NUMINAMATH_GPT_complement_P_eq_Ioo_l2385_238507


namespace NUMINAMATH_GPT_max_profit_l2385_238546

/-- Define the cost and price of device A and device B -/
def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8

/-- Define the total number of devices -/
def total_devices : ℝ := 50

/-- Define the profits per device -/
def profit_per_A : ℝ := price_A - cost_A -- 0.3
def profit_per_B : ℝ := price_B - cost_B -- 0.4

/-- Define the function for total profit -/
def total_profit (x : ℝ) : ℝ :=
  profit_per_A * x + profit_per_B * (total_devices - x)

/-- Define the constraint -/
def constraint (x : ℝ) : Prop := 4 * x ≥ total_devices - x -- x ≥ 10

/-- The statement of the problem that needs to be proven -/
theorem max_profit :
  (total_profit x = -0.1 * x + 20) ∧ 
  ( ∀ x, constraint x → x ≥ 10 → x = 10 ∧ total_profit x = 19) :=
by
  sorry

end NUMINAMATH_GPT_max_profit_l2385_238546


namespace NUMINAMATH_GPT_freken_bok_weight_l2385_238500

variables (K F M : ℕ)

theorem freken_bok_weight 
  (h1 : K + F = M + 75) 
  (h2 : F + M = K + 45) : 
  F = 60 :=
sorry

end NUMINAMATH_GPT_freken_bok_weight_l2385_238500


namespace NUMINAMATH_GPT_binary_to_decimal_110_eq_6_l2385_238529

theorem binary_to_decimal_110_eq_6 : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_110_eq_6_l2385_238529


namespace NUMINAMATH_GPT_product_of_fractions_is_eight_l2385_238514

theorem product_of_fractions_is_eight :
  (8 / 4) * (14 / 7) * (20 / 10) * (25 / 50) * (9 / 18) * (12 / 6) * (21 / 42) * (16 / 8) = 8 :=
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_is_eight_l2385_238514


namespace NUMINAMATH_GPT_tan_sum_identity_sin_2alpha_l2385_238520

theorem tan_sum_identity_sin_2alpha (α : ℝ) (h : Real.tan (π/4 + α) = 2) : Real.sin (2*α) = 3/5 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_identity_sin_2alpha_l2385_238520


namespace NUMINAMATH_GPT_salary_of_A_l2385_238509

theorem salary_of_A (A B : ℝ) (h1 : A + B = 7000) (h2 : 0.05 * A = 0.15 * B) : A = 5250 := 
by 
  sorry

end NUMINAMATH_GPT_salary_of_A_l2385_238509


namespace NUMINAMATH_GPT_books_read_in_eight_hours_l2385_238528

noncomputable def pages_per_hour : ℕ := 120
noncomputable def pages_per_book : ℕ := 360
noncomputable def total_reading_time : ℕ := 8

theorem books_read_in_eight_hours (h1 : pages_per_hour = 120) 
                                  (h2 : pages_per_book = 360) 
                                  (h3 : total_reading_time = 8) : 
                                  total_reading_time * pages_per_hour / pages_per_book = 2 := 
by sorry

end NUMINAMATH_GPT_books_read_in_eight_hours_l2385_238528


namespace NUMINAMATH_GPT_number_of_dogs_l2385_238593

theorem number_of_dogs (cost_price selling_price total_amount : ℝ) (profit_percentage : ℝ)
    (h1 : cost_price = 1000)
    (h2 : profit_percentage = 0.30)
    (h3 : total_amount = 2600)
    (h4 : selling_price = cost_price + (profit_percentage * cost_price)) :
    total_amount / selling_price = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dogs_l2385_238593


namespace NUMINAMATH_GPT_kelvin_classes_l2385_238579

theorem kelvin_classes (c : ℕ) (h1 : Grant = 4 * c) (h2 : c + Grant = 450) : c = 90 :=
by sorry

end NUMINAMATH_GPT_kelvin_classes_l2385_238579


namespace NUMINAMATH_GPT_square_simplify_l2385_238527

   variable (y : ℝ)

   theorem square_simplify :
     (7 - Real.sqrt (y^2 - 49)) ^ 2 = y^2 - 14 * Real.sqrt (y^2 - 49) :=
   sorry
   
end NUMINAMATH_GPT_square_simplify_l2385_238527


namespace NUMINAMATH_GPT_union_M_N_eq_N_l2385_238581

def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}

theorem union_M_N_eq_N : M ∪ N = N := by
  sorry

end NUMINAMATH_GPT_union_M_N_eq_N_l2385_238581


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_l2385_238591

theorem arithmetic_geometric_mean (a b m : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : (a + b) / 2 = m * Real.sqrt (a * b)) :
  a / b = (m + Real.sqrt (m^2 + 1)) / (m - Real.sqrt (m^2 - 1)) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_l2385_238591


namespace NUMINAMATH_GPT_triangle_angle_sum_l2385_238582

theorem triangle_angle_sum (x : ℝ) (h1 : 70 + 50 + x = 180) : x = 60 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l2385_238582


namespace NUMINAMATH_GPT_fraction_subtraction_l2385_238551

theorem fraction_subtraction : (5 / 6) - (1 / 12) = (3 / 4) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l2385_238551


namespace NUMINAMATH_GPT_intersection_is_solution_l2385_238518

theorem intersection_is_solution (a b : ℝ) :
  (b = 3 * a + 6 ∧ b = 2 * a - 4) ↔ (3 * a - b = -6 ∧ 2 * a - b = 4) := 
by sorry

end NUMINAMATH_GPT_intersection_is_solution_l2385_238518


namespace NUMINAMATH_GPT_power_mod_l2385_238599

theorem power_mod : (5 ^ 2023) % 11 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_power_mod_l2385_238599


namespace NUMINAMATH_GPT_find_g7_l2385_238541

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x ^ 7 + b * x ^ 3 + d * x ^ 2 + c * x - 8

theorem find_g7 (a b c d : ℝ) (h : g (-7) a b c d = 3) (h_d : d = 0) : g 7 a b c d = -19 :=
by
  simp [g, h, h_d]
  sorry

end NUMINAMATH_GPT_find_g7_l2385_238541


namespace NUMINAMATH_GPT_sum_of_factors_is_17_l2385_238585

theorem sum_of_factors_is_17 :
  ∃ (a b c d e f g : ℤ), 
  (16 * x^4 - 81 * y^4) =
    (a * x + b * y) * 
    (c * x^2 + d * x * y + e * y^2) * 
    (f * x + g * y) ∧ 
    a + b + c + d + e + f + g = 17 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_factors_is_17_l2385_238585


namespace NUMINAMATH_GPT_tan_triple_angle_l2385_238562

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end NUMINAMATH_GPT_tan_triple_angle_l2385_238562


namespace NUMINAMATH_GPT_count_pos_integers_three_digits_l2385_238574

/-- The number of positive integers less than 50,000 having at most three distinct digits equals 7862. -/
theorem count_pos_integers_three_digits : 
  ∃ n : ℕ, n < 50000 ∧ (∀ d1 d2 d3 d4 d5 : ℕ, d1 ≠ d2 ∨ d1 ≠ d3 ∨ d1 ≠ d4 ∨ d1 ≠ d5 ∨ d2 ≠ d3 ∨ d2 ≠ d4 ∨ d2 ≠ d5 ∨ d3 ≠ d4 ∨ d3 ≠ d5 ∨ d4 ≠ d5) ∧ n = 7862 :=
sorry

end NUMINAMATH_GPT_count_pos_integers_three_digits_l2385_238574


namespace NUMINAMATH_GPT_total_number_of_seats_l2385_238502

def number_of_trains : ℕ := 3
def cars_per_train : ℕ := 12
def seats_per_car : ℕ := 24

theorem total_number_of_seats :
  number_of_trains * cars_per_train * seats_per_car = 864 := by
  sorry

end NUMINAMATH_GPT_total_number_of_seats_l2385_238502


namespace NUMINAMATH_GPT_will_jogged_for_30_minutes_l2385_238557

theorem will_jogged_for_30_minutes 
  (calories_before : ℕ)
  (calories_per_minute : ℕ)
  (net_calories_after : ℕ)
  (h1 : calories_before = 900)
  (h2 : calories_per_minute = 10)
  (h3 : net_calories_after = 600) :
  let calories_burned := calories_before - net_calories_after
  let jogging_time := calories_burned / calories_per_minute
  jogging_time = 30 := by
  sorry

end NUMINAMATH_GPT_will_jogged_for_30_minutes_l2385_238557


namespace NUMINAMATH_GPT_part1_part2_l2385_238539

def first_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x < (f y) / y

def second_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x^2 < (f y) / y^2

noncomputable def f (h : ℝ) (x : ℝ) : ℝ :=
  x^3 - 2 * h * x^2 - h * x

theorem part1 (h : ℝ) (h1 : first_order_ratio_increasing (f h)) (h2 : ¬ second_order_ratio_increasing (f h)) :
  h < 0 :=
sorry

theorem part2 (f : ℝ → ℝ) (h : second_order_ratio_increasing f) (h2 : ∃ k > 0, ∀ x > 0, f x < k) :
  ∃ k, k = 0 ∧ ∀ x > 0, f x < k :=
sorry

end NUMINAMATH_GPT_part1_part2_l2385_238539


namespace NUMINAMATH_GPT_remainder_2519_div_7_l2385_238564

theorem remainder_2519_div_7 : 2519 % 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_2519_div_7_l2385_238564


namespace NUMINAMATH_GPT_total_birds_is_1300_l2385_238540

def initial_birds : ℕ := 300
def birds_doubled (b : ℕ) : ℕ := 2 * b
def birds_reduced (b : ℕ) : ℕ := b - 200
def total_birds_three_days : ℕ := initial_birds + birds_doubled initial_birds + birds_reduced (birds_doubled initial_birds)

theorem total_birds_is_1300 : total_birds_three_days = 1300 :=
by
  unfold total_birds_three_days initial_birds birds_doubled birds_reduced
  simp
  done

end NUMINAMATH_GPT_total_birds_is_1300_l2385_238540


namespace NUMINAMATH_GPT_blue_apples_l2385_238567

theorem blue_apples (B : ℕ) (h : (12 / 5) * B = 12) : B = 5 :=
by
  sorry

end NUMINAMATH_GPT_blue_apples_l2385_238567


namespace NUMINAMATH_GPT_smallest_n_for_violet_candy_l2385_238563

theorem smallest_n_for_violet_candy (p y o n : Nat) (h : 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * n) :
  n = 8 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_n_for_violet_candy_l2385_238563


namespace NUMINAMATH_GPT_original_price_l2385_238578

theorem original_price (x : ℝ) (h1 : 0.75 * x + 12 = x - 12) (h2 : 0.90 * x - 42 = x - 12) : x = 360 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l2385_238578


namespace NUMINAMATH_GPT_solve_for_a_l2385_238508

noncomputable def a_value (a x : ℝ) : Prop :=
  (3 / 10) * a + (2 * x + 4) / 2 = 4 * (x - 1)

theorem solve_for_a (a : ℝ) : a_value a 3 → a = 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2385_238508


namespace NUMINAMATH_GPT_Charles_has_13_whistles_l2385_238523

-- Conditions
def Sean_whistles : ℕ := 45
def more_whistles_than_Charles : ℕ := 32

-- Let C be the number of whistles Charles has
def C : ℕ := Sean_whistles - more_whistles_than_Charles

-- Theorem to be proven
theorem Charles_has_13_whistles : C = 13 := by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_Charles_has_13_whistles_l2385_238523


namespace NUMINAMATH_GPT_geometric_sequence_sixth_term_l2385_238583

variable (q : ℕ) (a_2 a_6 : ℕ)

-- Given conditions:
axiom h1 : q = 2
axiom h2 : a_2 = 8

-- Prove that a_6 = 128 where a_n = a_2 * q^(n-2)
theorem geometric_sequence_sixth_term : a_6 = a_2 * q^4 → a_6 = 128 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_sixth_term_l2385_238583


namespace NUMINAMATH_GPT_ceil_square_of_neg_fraction_l2385_238555

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end NUMINAMATH_GPT_ceil_square_of_neg_fraction_l2385_238555


namespace NUMINAMATH_GPT_no_solution_l2385_238533

theorem no_solution (x : ℝ) : ¬ (x / -4 ≥ 3 + x ∧ |2*x - 1| < 4 + 2*x) := 
by sorry

end NUMINAMATH_GPT_no_solution_l2385_238533


namespace NUMINAMATH_GPT_ensemble_average_age_l2385_238589

theorem ensemble_average_age (female_avg_age : ℝ) (num_females : ℕ) (male_avg_age : ℝ) (num_males : ℕ)
  (h1 : female_avg_age = 32) (h2 : num_females = 12) (h3 : male_avg_age = 40) (h4 : num_males = 18) :
  (num_females * female_avg_age + num_males * male_avg_age) / (num_females + num_males) =  36.8 :=
by sorry

end NUMINAMATH_GPT_ensemble_average_age_l2385_238589


namespace NUMINAMATH_GPT_find_price_of_each_part_l2385_238510

def original_price (total_cost : ℝ) (num_parts : ℕ) (price_per_part : ℝ) :=
  num_parts * price_per_part = total_cost

theorem find_price_of_each_part :
  original_price 439 7 62.71 :=
by
  sorry

end NUMINAMATH_GPT_find_price_of_each_part_l2385_238510


namespace NUMINAMATH_GPT_train_length_is_135_l2385_238534

noncomputable def speed_km_per_hr : ℝ := 54
noncomputable def time_seconds : ℝ := 9
noncomputable def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
noncomputable def length_of_train : ℝ := speed_m_per_s * time_seconds

theorem train_length_is_135 : length_of_train = 135 := by
  sorry

end NUMINAMATH_GPT_train_length_is_135_l2385_238534


namespace NUMINAMATH_GPT_retail_price_of_machine_l2385_238596

theorem retail_price_of_machine 
  (wholesale_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (selling_price : ℝ) 
  (P : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.10)
  (h3 : profit_rate = 0.20)
  (h4 : selling_price = wholesale_price * (1 + profit_rate))
  (h5 : (P * (1 - discount_rate)) = selling_price) : 
  P = 120 := by
  sorry

end NUMINAMATH_GPT_retail_price_of_machine_l2385_238596


namespace NUMINAMATH_GPT_product_of_integers_abs_val_not_less_than_1_and_less_than_3_l2385_238532

theorem product_of_integers_abs_val_not_less_than_1_and_less_than_3 :
  (-2) * (-1) * 1 * 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_product_of_integers_abs_val_not_less_than_1_and_less_than_3_l2385_238532


namespace NUMINAMATH_GPT_february_five_sundays_in_twenty_first_century_l2385_238572

/-- 
  Define a function to check if a year is a leap year
-/
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

/-- 
  Define the specific condition for the problem: 
  Given a year, whether February 1st for that year is a Sunday
-/
def february_first_is_sunday (year : ℕ) : Prop :=
  -- This is a placeholder logic. In real applications, you would
  -- calculate the exact weekday of February 1st for the provided year.
  sorry

/-- 
  The list of years in the 21st century where February has 5 Sundays is 
  exactly {2004, 2032, 2060, and 2088}.
-/
theorem february_five_sundays_in_twenty_first_century :
  {year : ℕ | is_leap_year year ∧ february_first_is_sunday year ∧ (2001 ≤ year ∧ year ≤ 2100)} =
  {2004, 2032, 2060, 2088} := sorry

end NUMINAMATH_GPT_february_five_sundays_in_twenty_first_century_l2385_238572


namespace NUMINAMATH_GPT_total_worth_correct_l2385_238569

def row1_gold_bars : ℕ := 5
def row1_weight_per_bar : ℕ := 2
def row1_cost_per_kg : ℕ := 20000

def row2_gold_bars : ℕ := 8
def row2_weight_per_bar : ℕ := 3
def row2_cost_per_kg : ℕ := 18000

def row3_gold_bars : ℕ := 3
def row3_weight_per_bar : ℕ := 5
def row3_cost_per_kg : ℕ := 22000

def row4_gold_bars : ℕ := 4
def row4_weight_per_bar : ℕ := 4
def row4_cost_per_kg : ℕ := 25000

def total_worth : ℕ :=
  (row1_gold_bars * row1_weight_per_bar * row1_cost_per_kg)
  + (row2_gold_bars * row2_weight_per_bar * row2_cost_per_kg)
  + (row3_gold_bars * row3_weight_per_bar * row3_cost_per_kg)
  + (row4_gold_bars * row4_weight_per_bar * row4_cost_per_kg)

theorem total_worth_correct : total_worth = 1362000 := by
  sorry

end NUMINAMATH_GPT_total_worth_correct_l2385_238569


namespace NUMINAMATH_GPT_five_fold_function_application_l2385_238545

def f (x : ℤ) : ℤ :=
if x ≥ 0 then -x^2 + 1 else x + 9

theorem five_fold_function_application : f (f (f (f (f 2)))) = -17 :=
by
  sorry

end NUMINAMATH_GPT_five_fold_function_application_l2385_238545


namespace NUMINAMATH_GPT_polynomial_roots_distinct_and_expression_is_integer_l2385_238505

-- Defining the conditions and the main theorem
theorem polynomial_roots_distinct_and_expression_is_integer (a b c : ℂ) :
  (a^3 - a^2 - a - 1 = 0) → (b^3 - b^2 - b - 1 = 0) → (c^3 - c^2 - c - 1 = 0) → 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  ∃ k : ℤ, ((a^(1982) - b^(1982)) / (a - b) + (b^(1982) - c^(1982)) / (b - c) + (c^(1982) - a^(1982)) / (c - a) = k) :=
by
  intros h1 h2 h3
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_polynomial_roots_distinct_and_expression_is_integer_l2385_238505


namespace NUMINAMATH_GPT_area_of_parallelogram_l2385_238522

-- Define the vectors
def v : ℝ × ℝ := (7, -5)
def w : ℝ × ℝ := (14, -4)

-- Prove the area of the parallelogram
theorem area_of_parallelogram : 
  abs (v.1 * w.2 - v.2 * w.1) = 42 :=
by
  sorry

end NUMINAMATH_GPT_area_of_parallelogram_l2385_238522


namespace NUMINAMATH_GPT_combination_x_l2385_238561
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

theorem combination_x (x : ℕ) (H : C 25 (2 * x) = C 25 (x + 4)) : x = 4 ∨ x = 7 :=
by sorry

end NUMINAMATH_GPT_combination_x_l2385_238561


namespace NUMINAMATH_GPT_not_necessarily_divisible_by_66_l2385_238570

open Nat

-- Definition of what it means to be the product of four consecutive integers
def product_of_four_consecutive_integers (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (k * (k + 1) * (k + 2) * (k + 3))

-- Lean theorem statement for the proof problem
theorem not_necessarily_divisible_by_66 (n : ℕ) 
  (h1 : product_of_four_consecutive_integers n) 
  (h2 : 11 ∣ n) : ¬ (66 ∣ n) :=
sorry

end NUMINAMATH_GPT_not_necessarily_divisible_by_66_l2385_238570


namespace NUMINAMATH_GPT_equal_share_of_candles_l2385_238531

-- Define conditions
def ambika_candles : ℕ := 4
def aniyah_candles : ℕ := 6 * ambika_candles
def bree_candles : ℕ := 2 * aniyah_candles
def caleb_candles : ℕ := bree_candles + (bree_candles / 2)

-- Define the total candles and the equal share
def total_candles : ℕ := ambika_candles + aniyah_candles + bree_candles + caleb_candles
def each_share : ℕ := total_candles / 4

-- State the problem
theorem equal_share_of_candles : each_share = 37 := by
  sorry

end NUMINAMATH_GPT_equal_share_of_candles_l2385_238531


namespace NUMINAMATH_GPT_length_segment_l2385_238511

/--
Given a cylinder with a radius of 5 units capped with hemispheres at each end and having a total volume of 900π,
prove that the length of the line segment AB is 88/3 units.
-/
theorem length_segment (r : ℝ) (V : ℝ) (h : ℝ) : r = 5 ∧ V = 900 * Real.pi → h = 88 / 3 := by
  sorry

end NUMINAMATH_GPT_length_segment_l2385_238511


namespace NUMINAMATH_GPT_evaluate_expression_l2385_238560

theorem evaluate_expression :
  200 * (200 - 3) + (200 ^ 2 - 8 ^ 2) = 79336 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2385_238560


namespace NUMINAMATH_GPT_b_should_pay_348_48_l2385_238565

/-- Definitions for the given conditions --/

def horses_a : ℕ := 12
def months_a : ℕ := 8

def horses_b : ℕ := 16
def months_b : ℕ := 9

def horses_c : ℕ := 18
def months_c : ℕ := 6

def total_rent : ℕ := 841

/-- Calculate the individual and total contributions in horse-months --/

def contribution_a : ℕ := horses_a * months_a
def contribution_b : ℕ := horses_b * months_b
def contribution_c : ℕ := horses_c * months_c

def total_contributions : ℕ := contribution_a + contribution_b + contribution_c

/-- Calculate cost per horse-month and b's share of the rent --/

def cost_per_horse_month : ℚ := total_rent / total_contributions
def b_share : ℚ := contribution_b * cost_per_horse_month

/-- Lean statement to check b's share --/

theorem b_should_pay_348_48 : b_share = 348.48 := by
  sorry

end NUMINAMATH_GPT_b_should_pay_348_48_l2385_238565


namespace NUMINAMATH_GPT_part1_part2_part3_l2385_238544

-- Definitions for conditions used in the proof problems
def eq1 (a b : ℝ) : Prop := 2 * a + b = 0
def eq2 (a x : ℝ) : Prop := x = a ^ 2

-- Part 1: Prove b = 4 and x = 4 given a = -2
theorem part1 (a b x : ℝ) (h1 : a = -2) (h2 : eq1 a b) (h3 : eq2 a x) : b = 4 ∧ x = 4 :=
by sorry

-- Part 2: Prove a = -3 and x = 9 given b = 6
theorem part2 (a b x : ℝ) (h1 : b = 6) (h2 : eq1 a b) (h3 : eq2 a x) : a = -3 ∧ x = 9 :=
by sorry

-- Part 3: Prove x = 2 given a^2*x + (a + b)^2*x = 8
theorem part3 (a b x : ℝ) (h : a^2 * x + (a + b)^2 * x = 8) : x = 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_l2385_238544


namespace NUMINAMATH_GPT_race_runners_l2385_238513

theorem race_runners (k : ℕ) (h1 : 2*(k - 1) = k - 1) (h2 : 2*(2*(k + 9) - 12) = k + 9) : 3*k - 2 = 31 :=
by
  sorry

end NUMINAMATH_GPT_race_runners_l2385_238513


namespace NUMINAMATH_GPT_ball_bounce_height_l2385_238553

noncomputable def height_after_bounces (h₀ : ℝ) (r : ℝ) (b : ℕ) : ℝ :=
  h₀ * (r ^ b)

theorem ball_bounce_height
  (h₀ : ℝ) (r : ℝ) (hb : ℕ) (h₀_pos : h₀ > 0) (r_pos : 0 < r ∧ r < 1) (h₀_val : h₀ = 320) (r_val : r = 3 / 4) (height_limit : ℝ) (height_limit_val : height_limit = 40):
  (hb ≥ 6) ∧ height_after_bounces h₀ r hb < height_limit :=
by
  sorry

end NUMINAMATH_GPT_ball_bounce_height_l2385_238553
