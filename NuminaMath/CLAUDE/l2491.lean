import Mathlib

namespace alphabet_letter_count_l2491_249143

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (line_only : ℕ) :
  total = 50 →
  both = 16 →
  line_only = 30 →
  ∃ (dot_only : ℕ),
    dot_only = total - (both + line_only) ∧
    dot_only = 4 :=
by
  sorry

end alphabet_letter_count_l2491_249143


namespace race_heartbeats_l2491_249151

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  (race_distance * pace * heart_rate)

/-- Proves that the total number of heartbeats during a 30-mile race is 28800,
    given the specified heart rate and pace. -/
theorem race_heartbeats :
  total_heartbeats 160 6 30 = 28800 := by
  sorry

end race_heartbeats_l2491_249151


namespace second_section_students_correct_l2491_249124

/-- The number of students in the second section of chemistry class X -/
def students_section2 : ℕ := 35

/-- The total number of students in all four sections -/
def total_students : ℕ := 65 + students_section2 + 45 + 42

/-- The overall average of marks per student -/
def overall_average : ℚ := 5195 / 100

/-- Theorem stating that the number of students in the second section is correct -/
theorem second_section_students_correct :
  (65 * 50 + students_section2 * 60 + 45 * 55 + 42 * 45 : ℚ) / total_students = overall_average :=
sorry

end second_section_students_correct_l2491_249124


namespace kenneths_earnings_l2491_249146

theorem kenneths_earnings (spent_percentage : ℝ) (remaining_amount : ℝ) (total_earnings : ℝ) : 
  spent_percentage = 10 →
  remaining_amount = 405 →
  (100 - spent_percentage) / 100 * total_earnings = remaining_amount →
  total_earnings = 450 := by
sorry

end kenneths_earnings_l2491_249146


namespace square_inequality_l2491_249193

theorem square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end square_inequality_l2491_249193


namespace min_value_quadratic_sum_l2491_249103

theorem min_value_quadratic_sum (a b c : ℝ) (h : a + 2 * b + 3 * c = 6) :
  a^2 + 4 * b^2 + 9 * c^2 ≥ 12 := by
  sorry

end min_value_quadratic_sum_l2491_249103


namespace intersection_and_union_when_m_neg_one_subset_iff_m_range_l2491_249133

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3}

-- Theorem for part (1)
theorem intersection_and_union_when_m_neg_one :
  (A ∩ B (-1) = {x | 1 < x ∧ x ≤ 2}) ∧
  (A ∪ B (-1) = {x | x ≥ -1}) := by sorry

-- Theorem for part (2)
theorem subset_iff_m_range :
  ∀ m : ℝ, B m ⊆ A ↔ m > 1 := by sorry

end intersection_and_union_when_m_neg_one_subset_iff_m_range_l2491_249133


namespace range_of_a_l2491_249176

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + abs (x - a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a (1/2) ≥ f a x → x = 1/2) →
  (∀ x : ℝ, f a (-1/2) ≥ f a x → x = -1/2) →
  a > -1/2 ∧ a < 1/2 :=
sorry

end range_of_a_l2491_249176


namespace ellipse_m_range_l2491_249171

theorem ellipse_m_range (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (2 + m)) - (y^2 / (m + 1)) = 1 ∧ 
   ((2 + m > 0 ∧ -(m + 1) > 0) ∨ (-(m + 1) > 0 ∧ 2 + m > 0))) ↔ 
  (m ∈ Set.Ioo (-2 : ℝ) (-3/2) ∪ Set.Ioo (-3/2 : ℝ) (-1)) := by
sorry

end ellipse_m_range_l2491_249171


namespace divisibility_implication_l2491_249105

theorem divisibility_implication (a b : ℕ) : 
  a < 1000 → (∃ k : ℕ, a^21 = k * b^10) → (∃ m : ℕ, a^2 = m * b) := by
  sorry

end divisibility_implication_l2491_249105


namespace weavers_in_first_group_l2491_249121

/-- The number of weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of weavers in the second group -/
def second_group_weavers : ℕ := 12

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 36

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 12

/-- Theorem stating that the number of weavers in the first group is 4 -/
theorem weavers_in_first_group :
  first_group_weavers = 4 :=
by sorry

end weavers_in_first_group_l2491_249121


namespace quadrilaterals_from_circle_points_l2491_249147

/-- The number of distinct points on the circumference of a circle -/
def num_points : ℕ := 10

/-- The number of vertices required to form a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

/-- The number of different quadrilaterals that can be formed -/
def num_quadrilaterals : ℕ := Nat.choose num_points vertices_per_quadrilateral

theorem quadrilaterals_from_circle_points : num_quadrilaterals = 300 := by
  sorry

end quadrilaterals_from_circle_points_l2491_249147


namespace andrew_donut_problem_l2491_249102

/-- The number of donuts Andrew ate on Monday -/
def monday_donuts : ℕ := 14

/-- The number of donuts Andrew ate on Tuesday -/
def tuesday_donuts : ℕ := monday_donuts / 2

/-- The total number of donuts Andrew ate in three days -/
def total_donuts : ℕ := 49

/-- The multiplier for the number of donuts Andrew ate on Wednesday compared to Monday -/
def wednesday_multiplier : ℚ := 2

theorem andrew_donut_problem :
  monday_donuts + tuesday_donuts + (wednesday_multiplier * monday_donuts) = total_donuts :=
sorry

end andrew_donut_problem_l2491_249102


namespace park_oaks_l2491_249166

/-- The number of huge ancient oaks in a park -/
def huge_ancient_oaks (total_trees medium_firs saplings : ℕ) : ℕ :=
  total_trees - medium_firs - saplings

/-- Theorem: There are 15 huge ancient oaks in the park -/
theorem park_oaks : huge_ancient_oaks 96 23 58 = 15 := by
  sorry

end park_oaks_l2491_249166


namespace workers_calculation_l2491_249189

/-- The initial number of workers on a job -/
def initial_workers : ℕ := 20

/-- The number of days to complete the job with the initial number of workers -/
def initial_days : ℕ := 30

/-- The number of days worked before some workers leave -/
def days_before_leaving : ℕ := 15

/-- The number of workers that leave the job -/
def workers_leaving : ℕ := 5

/-- The total number of days to complete the job after some workers leave -/
def total_days : ℕ := 35

theorem workers_calculation :
  (initial_workers * days_before_leaving = (initial_workers - workers_leaving) * (total_days - days_before_leaving)) ∧
  (initial_workers * initial_days = initial_workers * days_before_leaving + (initial_workers - workers_leaving) * (total_days - days_before_leaving)) :=
sorry

end workers_calculation_l2491_249189


namespace problem_solution_l2491_249130

theorem problem_solution :
  ∃ (a b : ℤ) (c d : ℚ),
    (∀ n : ℤ, n > 0 → a ≤ n) ∧
    (∀ n : ℤ, n < 0 → n ≤ b) ∧
    (∀ q : ℚ, q ≠ 0 → |c| ≤ |q|) ∧
    (d⁻¹ = d) ∧
    ((a^2 : ℚ) - (b^2 : ℚ) + 2*d - c = 2 ∨ (a^2 : ℚ) - (b^2 : ℚ) + 2*d - c = -2) :=
by sorry

end problem_solution_l2491_249130


namespace alternating_arithmetic_series_sum_l2491_249134

def alternatingArithmeticSeries (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) : ℤ :=
  let n := (aₙ - a₁) / d + 1
  let pairs := (n - 1) / 2
  let pairSum := -d
  let leftover := if n % 2 = 0 then 0 else aₙ
  pairs * pairSum + leftover

theorem alternating_arithmetic_series_sum :
  alternatingArithmeticSeries 2 3 56 = 29 := by sorry

end alternating_arithmetic_series_sum_l2491_249134


namespace remainder_5_pow_2023_mod_6_l2491_249167

theorem remainder_5_pow_2023_mod_6 : 5^2023 % 6 = 5 := by
  sorry

end remainder_5_pow_2023_mod_6_l2491_249167


namespace cos_a_sin_b_value_l2491_249194

theorem cos_a_sin_b_value (A B : Real) (hA : 0 < A ∧ A < Real.pi / 2) (hB : 0 < B ∧ B < Real.pi / 2)
  (h : (4 + Real.tan A ^ 2) * (5 + Real.tan B ^ 2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = Real.sqrt 6 / 6 := by
  sorry

end cos_a_sin_b_value_l2491_249194


namespace right_triangle_inradius_l2491_249108

/-- The inradius of a right triangle with side lengths 12, 35, and 37 is 5 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 12 ∧ b = 35 ∧ c = 37 →  -- Side lengths
  a^2 + b^2 = c^2 →           -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 5 := by
  sorry

end right_triangle_inradius_l2491_249108


namespace solve_equation_solve_system_l2491_249131

-- Problem 1
theorem solve_equation (x : ℝ) : (x + 2) / 3 - 1 = (1 - x) / 2 ↔ x = 1 := by sorry

-- Problem 2
theorem solve_system (x y : ℝ) : x + 2*y = 8 ∧ 3*x - 4*y = 4 ↔ x = 4 ∧ y = 2 := by sorry

end solve_equation_solve_system_l2491_249131


namespace sum_even_factors_630_eq_1248_l2491_249148

/-- The sum of all positive even factors of 630 -/
def sum_even_factors_630 : ℕ := sorry

/-- 630 is the number we're examining -/
def n : ℕ := 630

/-- Theorem stating that the sum of all positive even factors of 630 is 1248 -/
theorem sum_even_factors_630_eq_1248 : sum_even_factors_630 = 1248 := by sorry

end sum_even_factors_630_eq_1248_l2491_249148


namespace minimum_sum_geometric_mean_l2491_249128

theorem minimum_sum_geometric_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) (hgm : Real.sqrt (a * b) = 1) :
  2 * (a + b) ≥ 4 ∧ (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.sqrt (x * y) = 1 ∧ 2 * (x + y) = 4) := by
  sorry

end minimum_sum_geometric_mean_l2491_249128


namespace number_conditions_l2491_249136

theorem number_conditions (x y : ℝ) : 
  (0.65 * x > 26) → 
  (0.4 * y < -3) → 
  ((x - y)^2 ≥ 100) → 
  (x > 40 ∧ y < -7.5) := by
sorry

end number_conditions_l2491_249136


namespace trishul_investment_percentage_l2491_249139

/-- Represents the investment amounts in Rupees -/
structure Investment where
  vishal : ℝ
  trishul : ℝ
  raghu : ℝ

/-- The given conditions of the investment problem -/
def investment_conditions (i : Investment) : Prop :=
  i.vishal = 1.1 * i.trishul ∧
  i.vishal + i.trishul + i.raghu = 6936 ∧
  i.raghu = 2400

/-- The theorem stating that Trishul invested 10% less than Raghu -/
theorem trishul_investment_percentage (i : Investment) 
  (h : investment_conditions i) : 
  (i.raghu - i.trishul) / i.raghu = 0.1 := by
  sorry

end trishul_investment_percentage_l2491_249139


namespace problem_1_problem_2_problem_3_l2491_249122

-- Problem 1
theorem problem_1 (a b c : ℝ) : 
  (-8 * a^4 * b^5 * c / (4 * a * b^5)) * (3 * a^3 * b^2) = -6 * a^6 * b^2 * c :=
sorry

-- Problem 2
theorem problem_2 (a : ℝ) :
  (2*a + 1)^2 - (2*a + 1)*(2*a - 1) = 4*a + 2 :=
sorry

-- Problem 3
theorem problem_3 (x y : ℝ) :
  (x - y - 2) * (x - y + 2) - (x + 2*y) * (x - 3*y) = 7*y^2 - x*y - 4 :=
sorry

end problem_1_problem_2_problem_3_l2491_249122


namespace max_abs_z_on_circle_l2491_249156

open Complex

theorem max_abs_z_on_circle (z : ℂ) : 
  (abs (z - I) = abs (3 - 4*I)) → (abs z ≤ 6) ∧ (∃ w : ℂ, abs (w - I) = abs (3 - 4*I) ∧ abs w = 6) := by
  sorry

end max_abs_z_on_circle_l2491_249156


namespace average_speed_calculation_l2491_249175

/-- Given a trip with specified distances and speeds, calculate the average speed -/
theorem average_speed_calculation (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 350)
  (h2 : distance1 = 200)
  (h3 : speed1 = 20)
  (h4 : distance2 = total_distance - distance1)
  (h5 : speed2 = 15) :
  (total_distance) / ((distance1 / speed1) + (distance2 / speed2)) = 17.5 := by
  sorry

end average_speed_calculation_l2491_249175


namespace minimum_score_for_english_l2491_249125

/-- Given the average score of two subjects and a desired average for three subjects,
    calculate the minimum score needed for the third subject. -/
def minimum_third_score (avg_two : ℝ) (desired_avg : ℝ) : ℝ :=
  3 * desired_avg - 2 * avg_two

theorem minimum_score_for_english (avg_two : ℝ) (desired_avg : ℝ)
  (h1 : avg_two = 90)
  (h2 : desired_avg ≥ 92) :
  minimum_third_score avg_two desired_avg ≥ 96 :=
sorry

end minimum_score_for_english_l2491_249125


namespace distinct_sequences_count_l2491_249188

/-- The number of sides on the die -/
def die_sides : ℕ := 6

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 10

/-- The number of distinct sequences when rolling a die -/
def num_sequences : ℕ := die_sides ^ num_rolls

theorem distinct_sequences_count : num_sequences = 60466176 := by
  sorry

end distinct_sequences_count_l2491_249188


namespace nested_fourth_root_l2491_249123

theorem nested_fourth_root (M : ℝ) (h : M > 1) :
  (M * (M * (M * M^(1/4))^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end nested_fourth_root_l2491_249123


namespace divisibility_properties_l2491_249172

theorem divisibility_properties (n : ℤ) : 
  (3 ∣ (n^3 - n)) ∧ 
  (5 ∣ (n^5 - n)) ∧ 
  (7 ∣ (n^7 - n)) ∧ 
  (11 ∣ (n^11 - n)) ∧ 
  (13 ∣ (n^13 - n)) := by
  sorry

end divisibility_properties_l2491_249172


namespace imaginary_part_of_product_l2491_249185

theorem imaginary_part_of_product : Complex.im ((1 - 3*Complex.I) * (1 - Complex.I)) = -4 := by
  sorry

end imaginary_part_of_product_l2491_249185


namespace infinitely_many_not_n_attainable_all_except_seven_3_attainable_l2491_249153

/-- Definition of an n-admissible sequence -/
def IsNAdmissibleSequence (n : ℕ) (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  (∀ k, k > 0 →
    ((a (2*k) = a (2*k-1) + 2 ∨ a (2*k) = a (2*k-1) + n) ∧
     (a (2*k+1) = 2 * a (2*k) ∨ a (2*k+1) = n * a (2*k))) ∨
    ((a (2*k) = 2 * a (2*k-1) ∨ a (2*k) = n * a (2*k-1)) ∧
     (a (2*k+1) = a (2*k) + 2 ∨ a (2*k+1) = a (2*k) + n)))

/-- Definition of n-attainable number -/
def IsNAttainable (n : ℕ) (m : ℕ) : Prop :=
  m > 1 ∧ ∃ a, IsNAdmissibleSequence n a ∧ ∃ k, a k = m

/-- There are infinitely many positive integers not n-attainable for n > 8 -/
theorem infinitely_many_not_n_attainable (n : ℕ) (hn : n > 8) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ m ∈ S, ¬IsNAttainable n m :=
sorry

/-- All positive integers except 7 are 3-attainable -/
theorem all_except_seven_3_attainable :
  ∀ m : ℕ, m > 0 ∧ m ≠ 7 → IsNAttainable 3 m :=
sorry

end infinitely_many_not_n_attainable_all_except_seven_3_attainable_l2491_249153


namespace food_drive_problem_l2491_249157

theorem food_drive_problem (total_students : ℕ) (cans_per_first_group : ℕ) (non_collecting_students : ℕ) (last_group_students : ℕ) (total_cans : ℕ) :
  total_students = 30 →
  cans_per_first_group = 12 →
  non_collecting_students = 2 →
  last_group_students = 13 →
  total_cans = 232 →
  (total_students / 2) * cans_per_first_group + 0 * non_collecting_students + last_group_students * ((total_cans - (total_students / 2) * cans_per_first_group) / last_group_students) = total_cans →
  (total_cans - (total_students / 2) * cans_per_first_group) / last_group_students = 4 :=
by sorry

end food_drive_problem_l2491_249157


namespace fencing_calculation_l2491_249192

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) : 
  area = 50 → uncovered_side = 20 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    2 * width + uncovered_side = 25 := by
  sorry

end fencing_calculation_l2491_249192


namespace multiply_64_56_l2491_249178

theorem multiply_64_56 : 64 * 56 = 3584 := by
  sorry

end multiply_64_56_l2491_249178


namespace division_problem_l2491_249181

theorem division_problem :
  ∀ (dividend divisor quotient remainder : ℕ),
    dividend = 171 →
    divisor = 21 →
    remainder = 3 →
    dividend = divisor * quotient + remainder →
    quotient = 8 := by
  sorry

end division_problem_l2491_249181


namespace original_price_calculation_shirt_price_proof_l2491_249165

/-- 
Given two successive discounts and a final sale price, 
calculate the original price of an item.
-/
theorem original_price_calculation 
  (discount1 : ℝ) 
  (discount2 : ℝ) 
  (final_price : ℝ) : ℝ :=
  let remaining_factor1 := 1 - discount1
  let remaining_factor2 := 1 - discount2
  let original_price := final_price / (remaining_factor1 * remaining_factor2)
  original_price

/-- 
Prove that given discounts of 15% and 2%, 
if the final sale price is 830, 
then the original price is approximately 996.40.
-/
theorem shirt_price_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (original_price_calculation 0.15 0.02 830 - 996.40) < ε :=
sorry

end original_price_calculation_shirt_price_proof_l2491_249165


namespace dig_time_proof_l2491_249190

/-- Represents the time (in days) it takes for a person to dig a well alone -/
structure DigTime :=
  (days : ℝ)
  (pos : days > 0)

/-- Given the dig times for three people and their combined dig time,
    proves that if two people's dig times are 24 and 48 days,
    the third person's dig time is 16 days -/
theorem dig_time_proof
  (combined_time : ℝ)
  (combined_time_pos : combined_time > 0)
  (combined_time_eq : combined_time = 8)
  (time1 time2 time3 : DigTime)
  (time2_eq : time2.days = 24)
  (time3_eq : time3.days = 48)
  (combined_rate_eq : 1 / combined_time = 1 / time1.days + 1 / time2.days + 1 / time3.days) :
  time1.days = 16 := by
sorry


end dig_time_proof_l2491_249190


namespace quadratic_roots_l2491_249116

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) 
  (h1 : a + b + c = 0) (h2 : a - b + c = 0) :
  ∃ (x y : ℝ), x = 1 ∧ y = -1 ∧ 
  (∀ z : ℝ, a * z^2 + b * z + c = 0 ↔ z = x ∨ z = y) :=
sorry

end quadratic_roots_l2491_249116


namespace surface_area_union_cones_l2491_249104

/-- The surface area of the union of two right cones with specific dimensions -/
theorem surface_area_union_cones (r h : ℝ) (hr : r = 4) (hh : h = 3) :
  let L := Real.sqrt (r^2 + h^2)
  let surface_area_one_cone := π * r^2 + π * r * L
  let lateral_area_half_cone := π * (r/2) * (Real.sqrt ((r/2)^2 + (h/2)^2))
  2 * (surface_area_one_cone - lateral_area_half_cone) = 62 * π :=
by sorry

end surface_area_union_cones_l2491_249104


namespace intersection_M_N_l2491_249142

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end intersection_M_N_l2491_249142


namespace equation_has_real_root_l2491_249120

-- Define the polynomial function
def f (K x : ℝ) : ℝ := K^2 * (x - 1) * (x - 2) * (x - 3) - x

-- Theorem statement
theorem equation_has_real_root :
  ∀ K : ℝ, ∃ x : ℝ, f K x = 0 :=
sorry

end equation_has_real_root_l2491_249120


namespace tesla_ownership_l2491_249197

/-- The number of Teslas owned by different individuals and their relationships. -/
theorem tesla_ownership (chris sam elon : ℕ) : 
  chris = 6 → 
  sam = chris / 2 → 
  elon = 13 → 
  elon - sam = 10 := by
sorry

end tesla_ownership_l2491_249197


namespace coefficient_x_squared_in_expansion_l2491_249107

theorem coefficient_x_squared_in_expansion : 
  let n : ℕ := 6
  let k : ℕ := 2
  let a : ℤ := 1
  let b : ℤ := -2
  (n.choose k) * b^k * a^(n-k) = 60 := by
  sorry

end coefficient_x_squared_in_expansion_l2491_249107


namespace percentage_reduction_proof_price_increase_proof_l2491_249111

-- Define the initial price
def initial_price : ℝ := 50

-- Define the final price after two reductions
def final_price : ℝ := 32

-- Define the profit per kilogram before price increase
def initial_profit : ℝ := 10

-- Define the initial daily sales volume
def initial_sales : ℝ := 500

-- Define the target daily profit
def target_profit : ℝ := 6000

-- Define the maximum allowed price increase
def max_price_increase : ℝ := 8

-- Define the sales volume decrease per yuan of price increase
def sales_decrease_rate : ℝ := 20

-- Theorem for the percentage reduction
theorem percentage_reduction_proof :
  ∃ x : ℝ, x > 0 ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price ∧ x = 1/5 :=
sorry

-- Theorem for the required price increase
theorem price_increase_proof :
  ∃ y : ℝ, 0 < y ∧ y ≤ max_price_increase ∧
  (initial_profit + y) * (initial_sales - sales_decrease_rate * y) = target_profit ∧
  y = 5 :=
sorry

end percentage_reduction_proof_price_increase_proof_l2491_249111


namespace range_of_m_l2491_249113

/-- Given that p: m - 1 < x < m + 1, q: (x - 2)(x - 6) < 0, and q is a necessary but not sufficient
condition for p, prove that the range of values for m is [3, 5]. -/
theorem range_of_m (m x : ℝ) 
  (hp : m - 1 < x ∧ x < m + 1)
  (hq : (x - 2) * (x - 6) < 0)
  (h_nec_not_suff : ∀ y, (m - 1 < y ∧ y < m + 1) → (y - 2) * (y - 6) < 0)
  (h_not_suff : ∃ z, (z - 2) * (z - 6) < 0 ∧ ¬(m - 1 < z ∧ z < m + 1)) :
  3 ≤ m ∧ m ≤ 5 := by
sorry

end range_of_m_l2491_249113


namespace painted_cube_probability_l2491_249161

/-- Represents a 5x5x5 cube with three adjacent faces painted -/
structure PaintedCube :=
  (size : ℕ)
  (total_cubes : ℕ)
  (painted_faces : ℕ)

/-- Calculates the number of unit cubes with exactly three painted faces -/
def three_painted_faces (cube : PaintedCube) : ℕ := sorry

/-- Calculates the number of unit cubes with exactly one painted face -/
def one_painted_face (cube : PaintedCube) : ℕ := sorry

/-- Calculates the total number of ways to choose two unit cubes -/
def total_choices (cube : PaintedCube) : ℕ := sorry

/-- Calculates the number of ways to choose one cube with three painted faces and one with one painted face -/
def favorable_choices (cube : PaintedCube) : ℕ := sorry

/-- The main theorem stating the probability -/
theorem painted_cube_probability (cube : PaintedCube) 
  (h1 : cube.size = 5)
  (h2 : cube.total_cubes = 125)
  (h3 : cube.painted_faces = 3) :
  (favorable_choices cube : ℚ) / (total_choices cube : ℚ) = 8 / 235 := by sorry

end painted_cube_probability_l2491_249161


namespace imaginary_part_of_complex_product_l2491_249132

theorem imaginary_part_of_complex_product (i : ℂ) : i * i = -1 → Complex.im (i * (1 + i) * i) = -1 := by
  sorry

end imaginary_part_of_complex_product_l2491_249132


namespace line_parallel_to_plane_l2491_249144

/-- A structure representing a line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A structure representing a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Containment relation of a line in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Theorem: If line a is parallel to line b, line a is not contained in plane α,
    and line b is contained in plane α, then line a is parallel to plane α -/
theorem line_parallel_to_plane (a b : Line3D) (α : Plane3D) :
  parallel_lines a b → ¬line_in_plane a α → line_in_plane b α → parallel_line_plane a α :=
by sorry

end line_parallel_to_plane_l2491_249144


namespace arithmetic_sequence_a10_l2491_249168

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  a3_eq_2 : a 3 = 2
  a5_plus_a8_eq_15 : a 5 + a 8 = 15

/-- The 10th term of the arithmetic sequence is 13 -/
theorem arithmetic_sequence_a10 (seq : ArithmeticSequence) : seq.a 10 = 13 := by
  sorry

end arithmetic_sequence_a10_l2491_249168


namespace first_group_size_is_correct_l2491_249163

/-- The number of men in the first group -/
def first_group_size : ℕ := 20

/-- The length of the fountain built by the first group -/
def first_fountain_length : ℕ := 56

/-- The number of days taken by the first group to build their fountain -/
def first_group_days : ℕ := 14

/-- The number of men in the second group -/
def second_group_size : ℕ := 35

/-- The length of the fountain built by the second group -/
def second_fountain_length : ℕ := 21

/-- The number of days taken by the second group to build their fountain -/
def second_group_days : ℕ := 3

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_is_correct :
  (first_group_size : ℚ) * second_fountain_length * second_group_days =
  second_group_size * first_fountain_length * first_group_days :=
by sorry

end first_group_size_is_correct_l2491_249163


namespace f_integer_values_l2491_249117

def f (a b : ℕ+) : ℚ :=
  (a.val ^ 2 + b.val ^ 2 + a.val * b.val) / (a.val * b.val - 1)

theorem f_integer_values (a b : ℕ+) (h : a.val * b.val ≠ 1) :
  (∃ n : ℤ, f a b = n) → (f a b = 4 ∨ f a b = 7) := by
  sorry

end f_integer_values_l2491_249117


namespace probability_y_leq_x_pow_five_l2491_249118

/-- The probability that y ≤ x^5 when x and y are uniformly distributed over [0,1] -/
theorem probability_y_leq_x_pow_five : Real := by
  -- Define x and y as random variables uniformly distributed over [0,1]
  -- Calculate the probability that y ≤ x^5
  -- Prove that this probability is equal to 1/6
  sorry

#check probability_y_leq_x_pow_five

end probability_y_leq_x_pow_five_l2491_249118


namespace smallest_number_of_cubes_for_given_box_l2491_249154

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of cubes to fill the given box is 90 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes ⟨27, 15, 6⟩ = 90 := by
  sorry

end smallest_number_of_cubes_for_given_box_l2491_249154


namespace cos_seven_pi_sixths_l2491_249101

theorem cos_seven_pi_sixths : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_seven_pi_sixths_l2491_249101


namespace solve_for_a_l2491_249152

theorem solve_for_a (x a : ℝ) (h1 : 2 * x - a + 5 = 0) (h2 : x = -2) : a = 1 := by
  sorry

end solve_for_a_l2491_249152


namespace nancy_carrots_l2491_249183

/-- The total number of carrots Nancy has after two days of picking and throwing out some -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

/-- Theorem stating that Nancy's total carrots is 31 given the specific numbers in the problem -/
theorem nancy_carrots : total_carrots 12 2 21 = 31 := by
  sorry

end nancy_carrots_l2491_249183


namespace largest_base5_five_digit_in_base10_l2491_249141

def largest_base5_five_digit : ℕ := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_five_digit_in_base10 : 
  largest_base5_five_digit = 3124 := by sorry

end largest_base5_five_digit_in_base10_l2491_249141


namespace orange_ribbons_l2491_249137

theorem orange_ribbons (total : ℕ) (yellow purple orange black : ℕ) : 
  yellow + purple + orange + black = total →
  4 * yellow = total →
  3 * purple = total →
  6 * orange = total →
  black = 40 →
  orange = 27 :=
by sorry

end orange_ribbons_l2491_249137


namespace simplify_fraction_l2491_249199

theorem simplify_fraction : (120 : ℚ) / 2160 = 1 / 18 := by sorry

end simplify_fraction_l2491_249199


namespace farm_distance_is_six_l2491_249158

/-- Represents the distance to the farm given the conditions of Bobby's trips -/
def distance_to_farm (initial_gas : ℝ) (supermarket_distance : ℝ) (partial_farm_trip : ℝ) 
  (final_gas : ℝ) (miles_per_gallon : ℝ) : ℝ :=
  let total_miles_driven := (initial_gas - final_gas) * miles_per_gallon
  let known_miles := 2 * supermarket_distance + 2 * partial_farm_trip
  total_miles_driven - known_miles

/-- Theorem stating that the distance to the farm is 6 miles -/
theorem farm_distance_is_six :
  distance_to_farm 12 5 2 2 2 = 6 := by
  sorry

end farm_distance_is_six_l2491_249158


namespace angle_measure_proof_l2491_249112

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x + 10) = 90) → x = 20 := by
sorry

end angle_measure_proof_l2491_249112


namespace final_mixture_concentration_l2491_249109

/-- Represents a vessel with a given capacity and alcohol concentration -/
structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

/-- Calculates the amount of alcohol in a vessel -/
def alcoholAmount (v : Vessel) : ℝ := v.capacity * v.alcoholConcentration

/-- Theorem: The concentration of the final mixture is (5.75 / 18) * 100% -/
theorem final_mixture_concentration 
  (vessel1 : Vessel)
  (vessel2 : Vessel)
  (vessel3 : Vessel)
  (vessel4 : Vessel)
  (finalVessel : ℝ) :
  vessel1.capacity = 2 →
  vessel1.alcoholConcentration = 0.25 →
  vessel2.capacity = 6 →
  vessel2.alcoholConcentration = 0.40 →
  vessel3.capacity = 3 →
  vessel3.alcoholConcentration = 0.55 →
  vessel4.capacity = 4 →
  vessel4.alcoholConcentration = 0.30 →
  finalVessel = 18 →
  (alcoholAmount vessel1 + alcoholAmount vessel2 + alcoholAmount vessel3 + alcoholAmount vessel4) / finalVessel = 5.75 / 18 := by
  sorry

#eval (5.75 / 18) * 100 -- Approximately 31.94%

end final_mixture_concentration_l2491_249109


namespace table_tennis_ball_surface_area_l2491_249184

/-- The surface area of a sphere with diameter 40 millimeters is approximately 5026.55 square millimeters. -/
theorem table_tennis_ball_surface_area :
  let diameter : ℝ := 40
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius^2
  ∃ ε > 0, abs (surface_area - 5026.55) < ε :=
by sorry

end table_tennis_ball_surface_area_l2491_249184


namespace cubic_function_extrema_l2491_249126

theorem cubic_function_extrema (a b : ℝ) (h_a : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - 3 * a * x^2 + b
  (∀ x ∈ Set.Icc (-1) 2, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 3) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ -21) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = -21) →
  a = 6 ∧ b = 3 := by
sorry

end cubic_function_extrema_l2491_249126


namespace second_year_percentage_approx_l2491_249149

def numeric_methods_students : ℕ := 240
def automatic_control_students : ℕ := 423
def both_subjects_students : ℕ := 134
def total_faculty_students : ℕ := 663

def second_year_students : ℕ := numeric_methods_students + automatic_control_students - both_subjects_students

def percentage_second_year : ℚ := (second_year_students : ℚ) / (total_faculty_students : ℚ) * 100

theorem second_year_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_second_year - 79.79| < ε :=
sorry

end second_year_percentage_approx_l2491_249149


namespace quadrilateral_second_offset_l2491_249164

/-- Given a quadrilateral with one diagonal of 50 cm, one offset of 10 cm, and an area of 450 cm^2,
    prove that the length of the second offset is 8 cm. -/
theorem quadrilateral_second_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) (offset2 : ℝ) :
  diagonal = 50 → offset1 = 10 → area = 450 →
  area = 1/2 * diagonal * (offset1 + offset2) →
  offset2 = 8 := by sorry

end quadrilateral_second_offset_l2491_249164


namespace train_length_proof_l2491_249100

/-- The length of each train in meters -/
def train_length : ℝ := 62.5

/-- The speed of the faster train in km/hr -/
def fast_train_speed : ℝ := 46

/-- The speed of the slower train in km/hr -/
def slow_train_speed : ℝ := 36

/-- The time taken for the faster train to completely pass the slower train in seconds -/
def overtake_time : ℝ := 45

theorem train_length_proof :
  let relative_speed := (fast_train_speed - slow_train_speed) * 1000 / 3600
  let distance_covered := relative_speed * overtake_time
  2 * train_length = distance_covered := by
  sorry

end train_length_proof_l2491_249100


namespace equilateral_triangle_perimeter_l2491_249160

theorem equilateral_triangle_perimeter (side_length : ℝ) (h : side_length = 7) :
  3 * side_length = 21 := by
  sorry

end equilateral_triangle_perimeter_l2491_249160


namespace point_transformation_l2491_249187

def rotate_z (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def initial_point : ℝ × ℝ × ℝ := (2, 2, -1)

def final_point : ℝ × ℝ × ℝ := (-2, 2, -1)

theorem point_transformation :
  (reflect_xy ∘ rotate_z ∘ reflect_yz ∘ reflect_xy ∘ rotate_z) initial_point = final_point := by
  sorry

end point_transformation_l2491_249187


namespace f_properties_l2491_249127

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for monotonicity intervals and extreme values
theorem f_properties :
  (∀ x < -1, HasDerivAt f (f x) x ∧ 0 < (deriv f x)) ∧
  (∀ x ∈ Set.Ioo (-1) 1, HasDerivAt f (f x) x ∧ (deriv f x) < 0) ∧
  (∀ x > 1, HasDerivAt f (f x) x ∧ 0 < (deriv f x)) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≥ -18) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≤ 2) ∧
  (f (-3) = -18) ∧
  (f (-1) = 2 ∨ f 2 = 2) := by
  sorry

#check f_properties

end f_properties_l2491_249127


namespace circle_equation_solution_l2491_249115

theorem circle_equation_solution :
  ∃! (x y : ℝ), (x - 12)^2 + (y - 13)^2 + (x - y)^2 = 1/3 ∧ 
  x = 37/3 ∧ y = 38/3 := by
sorry

end circle_equation_solution_l2491_249115


namespace first_part_games_l2491_249159

/-- Prove that the number of games in the first part of the season is 100 -/
theorem first_part_games (total_games : ℕ) (first_win_rate remaining_win_rate overall_win_rate : ℚ) : 
  total_games = 175 →
  first_win_rate = 85/100 →
  remaining_win_rate = 1/2 →
  overall_win_rate = 7/10 →
  ∃ (x : ℕ), x = 100 ∧ 
    first_win_rate * x + remaining_win_rate * (total_games - x) = overall_win_rate * total_games :=
by sorry

end first_part_games_l2491_249159


namespace octahedron_intersection_area_l2491_249162

/-- Represents a regular octahedron -/
structure RegularOctahedron where
  side_length : ℝ

/-- Represents the hexagonal intersection formed by a plane cutting the octahedron -/
structure HexagonalIntersection where
  octahedron : RegularOctahedron

/-- The area of the hexagonal intersection -/
def intersection_area (h : HexagonalIntersection) : ℝ := sorry

theorem octahedron_intersection_area 
  (o : RegularOctahedron)
  (h : HexagonalIntersection)
  (h_octahedron : h.octahedron = o)
  (side_length_eq : o.side_length = 2) :
  intersection_area h = 9 * Real.sqrt 3 / 8 := by sorry

end octahedron_intersection_area_l2491_249162


namespace initial_walnut_count_l2491_249138

/-- The number of walnut trees initially in the park -/
def initial_walnut_trees : ℕ := sorry

/-- The number of walnut trees cut down -/
def cut_trees : ℕ := 13

/-- The number of walnut trees remaining after cutting -/
def remaining_trees : ℕ := 29

/-- The number of orange trees in the park -/
def orange_trees : ℕ := 12

theorem initial_walnut_count :
  initial_walnut_trees = remaining_trees + cut_trees :=
by sorry

end initial_walnut_count_l2491_249138


namespace tea_bags_count_l2491_249150

/-- The number of tea bags in a box -/
def n : ℕ+ := sorry

/-- The number of cups of tea made from Natasha's box -/
def natasha_cups : ℕ := 41

/-- The number of cups of tea made from Inna's box -/
def inna_cups : ℕ := 58

/-- Theorem stating that the number of tea bags in the box is 20 -/
theorem tea_bags_count :
  (2 * n ≤ natasha_cups ∧ natasha_cups ≤ 3 * n) ∧
  (2 * n ≤ inna_cups ∧ inna_cups ≤ 3 * n) →
  n = 20 := by sorry

end tea_bags_count_l2491_249150


namespace sequence_formula_l2491_249182

theorem sequence_formula (n : ℕ) : 
  let a : ℕ → ℕ := λ k => 2^k + 1
  (a 1 = 3) ∧ (a 2 = 5) ∧ (a 3 = 9) ∧ (a 4 = 17) ∧ (a 5 = 33) :=
by sorry

end sequence_formula_l2491_249182


namespace range_of_m_l2491_249170

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m :
  (∀ m : ℝ, ¬(p m ∧ q m)) →
  (∀ m : ℝ, q m) →
  ∀ m : ℝ, 1 < m ∧ m < 2 :=
sorry

end range_of_m_l2491_249170


namespace jar_problem_l2491_249106

theorem jar_problem (total_jars small_jars : ℕ) 
  (small_capacity large_capacity : ℕ) (total_capacity : ℕ) :
  total_jars = 100 →
  small_jars = 62 →
  small_capacity = 3 →
  large_capacity = 5 →
  total_capacity = 376 →
  ∃ large_jars : ℕ, 
    small_jars + large_jars = total_jars ∧
    small_jars * small_capacity + large_jars * large_capacity = total_capacity ∧
    large_jars = 38 :=
by sorry

end jar_problem_l2491_249106


namespace max_quotient_value_l2491_249135

theorem max_quotient_value (a b : ℝ) (ha : 200 ≤ a ∧ a ≤ 400) (hb : 600 ≤ b ∧ b ≤ 1200) :
  (∀ x y, 200 ≤ x ∧ x ≤ 400 → 600 ≤ y ∧ y ≤ 1200 → y / x ≤ b / a) →
  b / a = 6 :=
by sorry

end max_quotient_value_l2491_249135


namespace forty_percent_of_sixty_minus_four_fifths_of_twenty_five_l2491_249174

theorem forty_percent_of_sixty_minus_four_fifths_of_twenty_five :
  (40 / 100 * 60) - (4 / 5 * 25) = 4 := by
  sorry

end forty_percent_of_sixty_minus_four_fifths_of_twenty_five_l2491_249174


namespace parabola_standard_equation_l2491_249140

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis,
    and passing through the point (-4, 4) has the standard equation y² = -4x. -/
theorem parabola_standard_equation :
  ∀ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ y^2 = -4*x) →  -- Standard equation of the parabola
  f 0 = 0 →                            -- Vertex at the origin
  (∀ x : ℝ, f x = f (-x)) →            -- Axis of symmetry along x-axis
  f (-4) = 4 →                         -- Passes through (-4, 4)
  ∀ x y : ℝ, f x = y ↔ y^2 = -4*x :=   -- Conclusion: standard equation
by sorry

end parabola_standard_equation_l2491_249140


namespace goat_kangaroo_ratio_l2491_249180

theorem goat_kangaroo_ratio : 
  ∀ (num_goats : ℕ), 
    (2 * 23 + 4 * num_goats = 322) → 
    (num_goats : ℚ) / 23 = 3 / 1 :=
by
  sorry

end goat_kangaroo_ratio_l2491_249180


namespace sin_taylor_expansion_at_3_l2491_249186

open Complex

/-- Taylor series expansion of sine function around z = 3 -/
theorem sin_taylor_expansion_at_3 (z : ℂ) : 
  sin z = (sin 3 * (∑' n, ((-1)^n / (2*n).factorial : ℂ) * (z - 3)^(2*n))) + 
          (cos 3 * (∑' n, ((-1)^n / (2*n + 1).factorial : ℂ) * (z - 3)^(2*n + 1))) := by
  sorry

end sin_taylor_expansion_at_3_l2491_249186


namespace simple_interest_double_rate_l2491_249179

/-- The rate of interest for simple interest when a sum doubles in 10 years -/
theorem simple_interest_double_rate : 
  ∀ (principal : ℝ) (rate : ℝ),
  principal > 0 →
  principal * (1 + rate * 10) = 2 * principal →
  rate = 1 / 10 := by
  sorry

end simple_interest_double_rate_l2491_249179


namespace max_tangent_segment_length_l2491_249114

/-- Given a triangle ABC with perimeter 2p, the maximum length of a segment
    parallel to BC and tangent to the inscribed circle is p/4, and this
    maximum is achieved when BC = p/2. -/
theorem max_tangent_segment_length (p : ℝ) (h : p > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 2 * p ∧
    (∀ (x y z : ℝ),
      x > 0 → y > 0 → z > 0 → x + y + z = 2 * p →
      x * (p - x) / p ≤ p / 4) ∧
    a * (p - a) / p = p / 4 ∧
    a = p / 2 := by
  sorry


end max_tangent_segment_length_l2491_249114


namespace inscribed_square_side_length_l2491_249173

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  de : ℝ
  ef : ℝ
  df : ℝ
  de_eq : de = 5
  ef_eq : ef = 12
  df_eq : df = 13
  right_angle : de^2 + ef^2 = df^2

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_df : side_length ≤ t.df
  on_de : side_length ≤ t.de
  on_ef : side_length ≤ t.ef

/-- The theorem stating the side length of the inscribed square -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 10140 / 229 := by
  sorry

end inscribed_square_side_length_l2491_249173


namespace painting_cost_in_cny_l2491_249110

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 5

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 8

/-- Cost of the painting in Namibian dollars -/
def painting_cost_nad : ℝ := 150

/-- Theorem stating the cost of the painting in Chinese yuan -/
theorem painting_cost_in_cny :
  (painting_cost_nad / usd_to_nad) * usd_to_cny = 240 := by
  sorry

end painting_cost_in_cny_l2491_249110


namespace expression_simplification_l2491_249145

theorem expression_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 - 11*x + 13*x^2 - 15 + 17*x + 19*x^2 = 25*x^2 + x - 3 :=
by sorry

end expression_simplification_l2491_249145


namespace complete_square_expression_l2491_249177

theorem complete_square_expression (y : ℝ) : 
  ∃ (k : ℤ) (b : ℝ), y^2 + 16*y + 60 = (y + b)^2 + k ∧ k = -4 := by
  sorry

end complete_square_expression_l2491_249177


namespace system_solution_quadratic_expression_l2491_249129

theorem system_solution_quadratic_expression :
  ∀ x y z : ℚ,
  (2 * x + 3 * y + z = 20) →
  (x + 2 * y + 3 * z = 26) →
  (3 * x + y + 2 * z = 29) →
  ∃ k : ℚ, 12 * x^2 + 22 * x * y + 12 * y^2 + 12 * x * z + 12 * y * z + 12 * z^2 = k :=
by
  sorry


end system_solution_quadratic_expression_l2491_249129


namespace pistachio_price_per_can_l2491_249119

/-- The price of a can of pistachios given James' consumption habits and weekly spending -/
theorem pistachio_price_per_can 
  (can_size : ℝ) 
  (consumption_per_5_days : ℝ) 
  (weekly_spending : ℝ) 
  (h1 : can_size = 5) 
  (h2 : consumption_per_5_days = 30) 
  (h3 : weekly_spending = 84) : 
  weekly_spending / ((7 / 5) * consumption_per_5_days / can_size) = 10 := by
sorry

end pistachio_price_per_can_l2491_249119


namespace first_hour_distance_correct_l2491_249198

/-- The distance traveled by a car in the first hour, given that its speed increases by 2 km/h every hour and it travels 492 km in 12 hours -/
def first_hour_distance : ℝ :=
  let speed_increase : ℝ := 2
  let total_hours : ℕ := 12
  let total_distance : ℝ := 492
  30

/-- Theorem stating that the first hour distance is correct -/
theorem first_hour_distance_correct :
  let speed_increase : ℝ := 2
  let total_hours : ℕ := 12
  let total_distance : ℝ := 492
  (first_hour_distance + total_hours * (total_hours - 1) / 2 * speed_increase) * total_hours / 2 = total_distance :=
by
  sorry

#eval first_hour_distance

end first_hour_distance_correct_l2491_249198


namespace average_rainfall_virginia_l2491_249196

theorem average_rainfall_virginia (march april may june july : ℝ) 
  (h_march : march = 3.79)
  (h_april : april = 4.5)
  (h_may : may = 3.95)
  (h_june : june = 3.09)
  (h_july : july = 4.67) :
  (march + april + may + june + july) / 5 = 4 := by
  sorry

end average_rainfall_virginia_l2491_249196


namespace smallest_square_side_length_l2491_249191

theorem smallest_square_side_length :
  ∀ (n : ℕ),
  (∃ (a b c d : ℕ),
    n * n = a + 4 * b + 9 * c ∧
    14 = a + b + c ∧
    a ≥ 10 ∧ b ≥ 3 ∧ c ≥ 1) →
  n ≥ 6 :=
by sorry

end smallest_square_side_length_l2491_249191


namespace quadratic_solution_l2491_249169

theorem quadratic_solution (c : ℝ) : 
  (18^2 + 12*18 + c = 0) → 
  (∃ x : ℝ, x^2 + 12*x + c = 0 ∧ x ≠ 18) → 
  ((-30)^2 + 12*(-30) + c = 0) := by
sorry

end quadratic_solution_l2491_249169


namespace remainder_problem_l2491_249155

theorem remainder_problem (t : ℕ) :
  let n : ℤ := 209 * t + 23
  (n % 19 = 4) ∧ (n % 11 = 1) := by
sorry

end remainder_problem_l2491_249155


namespace range_of_a_l2491_249195

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log (x / 2) - (3 * x - 6) / (x + 1)

noncomputable def g (x t a : ℝ) : ℝ := (x - t)^2 + (Real.log x - a * t)^2

theorem range_of_a :
  ∀ a : ℝ,
  (∀ x₁ : ℝ, x₁ > 1 → ∃ t x₂ : ℝ, x₂ > 0 ∧ f x₁ ≥ g x₂ t a) ↔
  a ≤ 1 / Real.exp 1 :=
sorry

end range_of_a_l2491_249195
