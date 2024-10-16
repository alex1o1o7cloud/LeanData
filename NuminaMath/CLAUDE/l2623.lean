import Mathlib

namespace NUMINAMATH_CALUDE_balloon_height_is_9482_l2623_262335

/-- Calculates the maximum height a balloon can fly given the following parameters:
    * total_money: The total amount of money available
    * sheet_cost: The cost of the balloon sheet
    * rope_cost: The cost of the rope
    * propane_cost: The cost of the propane tank and burner
    * helium_cost_per_oz: The cost of helium per ounce
    * height_per_oz: The height gain per ounce of helium
-/
def max_balloon_height (total_money sheet_cost rope_cost propane_cost helium_cost_per_oz height_per_oz : ℚ) : ℚ :=
  let remaining_money := total_money - (sheet_cost + rope_cost + propane_cost)
  let helium_oz := remaining_money / helium_cost_per_oz
  helium_oz * height_per_oz

/-- Theorem stating that given the specific conditions in the problem,
    the maximum height the balloon can fly is 9482 feet. -/
theorem balloon_height_is_9482 :
  max_balloon_height 200 42 18 14 1.5 113 = 9482 := by
  sorry

end NUMINAMATH_CALUDE_balloon_height_is_9482_l2623_262335


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2623_262333

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem absolute_value_inequality
  (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_f_1 : f 1 = 0)
  (h_functional_equation : ∀ x y, x > 0 → y > 0 → f x + f y = f (x * y)) :
  ∀ x y, 0 < x → x < y → y < 1 → |f x| > |f y| :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2623_262333


namespace NUMINAMATH_CALUDE_dress_costs_sum_l2623_262397

/-- The cost of dresses for four ladies -/
def dress_costs (pauline_cost ida_cost jean_cost patty_cost : ℕ) : Prop :=
  pauline_cost = 30 ∧
  jean_cost = pauline_cost - 10 ∧
  ida_cost = jean_cost + 30 ∧
  patty_cost = ida_cost + 10

/-- The total cost of all dresses -/
def total_cost (pauline_cost ida_cost jean_cost patty_cost : ℕ) : ℕ :=
  pauline_cost + ida_cost + jean_cost + patty_cost

/-- Theorem: The total cost of all dresses is $160 -/
theorem dress_costs_sum :
  ∀ (pauline_cost ida_cost jean_cost patty_cost : ℕ),
  dress_costs pauline_cost ida_cost jean_cost patty_cost →
  total_cost pauline_cost ida_cost jean_cost patty_cost = 160 := by
  sorry

end NUMINAMATH_CALUDE_dress_costs_sum_l2623_262397


namespace NUMINAMATH_CALUDE_smallest_positive_time_for_104_degrees_l2623_262349

def temperature (t : ℝ) : ℝ := -t^2 + 16*t + 40

theorem smallest_positive_time_for_104_degrees :
  let t := 8 + 8 * Real.sqrt 2
  (∀ s, s > 0 ∧ temperature s = 104 → s ≥ t) ∧ temperature t = 104 ∧ t > 0 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_time_for_104_degrees_l2623_262349


namespace NUMINAMATH_CALUDE_distance_for_specific_triangle_l2623_262365

/-- A right-angled triangle with sides a, b, and c --/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

/-- The distance between the centers of the inscribed and circumscribed circles of a right triangle --/
def distance_between_centers (t : RightTriangle) : ℝ :=
  sorry

theorem distance_for_specific_triangle :
  let t : RightTriangle := ⟨8, 15, 17, by norm_num⟩
  distance_between_centers t = Real.sqrt 85 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_for_specific_triangle_l2623_262365


namespace NUMINAMATH_CALUDE_morning_run_distance_l2623_262392

/-- Represents a person's daily activities and distances --/
structure DailyActivities where
  n : ℕ  -- number of stores visited
  x : ℝ  -- morning run distance
  total_distance : ℝ  -- total distance for the day
  bike_distance : ℝ  -- evening bike ride distance

/-- Theorem stating the relationship between morning run distance and other factors --/
theorem morning_run_distance (d : DailyActivities) 
  (h1 : d.total_distance = 18) 
  (h2 : d.bike_distance = 12) 
  (h3 : d.total_distance = d.x + 2 * d.n * d.x + d.bike_distance) :
  d.x = 6 / (1 + 2 * d.n) := by
  sorry

end NUMINAMATH_CALUDE_morning_run_distance_l2623_262392


namespace NUMINAMATH_CALUDE_tangent_line_equation_intersecting_line_equation_l2623_262309

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define a line passing through point P(-2, 0)
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Theorem for tangent line
theorem tangent_line_equation :
  ∀ k : ℝ, (∀ x y : ℝ, circle_C x y → line_through_P k x y → (x = -2 ∨ y = (3/4)*x + 3/2)) ∧
            (∀ x y : ℝ, (x = -2 ∨ y = (3/4)*x + 3/2) → line_through_P k x y → 
             (∃! p : ℝ × ℝ, circle_C p.1 p.2 ∧ line_through_P k p.1 p.2)) :=
sorry

-- Theorem for intersecting line with chord length 2√2
theorem intersecting_line_equation :
  ∀ k : ℝ, (∀ x y : ℝ, circle_C x y → line_through_P k x y → 
            (x - y + 2 = 0 ∨ 7*x - y + 14 = 0)) ∧
           (∀ x y : ℝ, (x - y + 2 = 0 ∨ 7*x - y + 14 = 0) → line_through_P k x y → 
            (∃ A B : ℝ × ℝ, circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 
             line_through_P k A.1 A.2 ∧ line_through_P k B.1 B.2 ∧
             (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_intersecting_line_equation_l2623_262309


namespace NUMINAMATH_CALUDE_A_inter_B_empty_A_union_B_complement_A_inter_complement_B_empty_complement_A_union_complement_B_eq_U_l2623_262351

-- Define the universal set U
def U : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}

-- Define set A
def A : Set ℝ := {x | -5 ≤ x ∧ x < -1}

-- Define set B
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Theorem for the intersection of A and B
theorem A_inter_B_empty : A ∩ B = ∅ := by sorry

-- Theorem for the union of A and B
theorem A_union_B : A ∪ B = {x | -5 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for the intersection of complements of A and B
theorem complement_A_inter_complement_B_empty : (U \ A) ∩ (U \ B) = ∅ := by sorry

-- Theorem for the union of complements of A and B
theorem complement_A_union_complement_B_eq_U : (U \ A) ∪ (U \ B) = U := by sorry

end NUMINAMATH_CALUDE_A_inter_B_empty_A_union_B_complement_A_inter_complement_B_empty_complement_A_union_complement_B_eq_U_l2623_262351


namespace NUMINAMATH_CALUDE_average_of_21_multiples_of_17_l2623_262317

/-- The average of the first n multiples of a number -/
def average_of_multiples (n : ℕ) (x : ℕ) : ℚ :=
  (n * x * (n + 1)) / (2 * n)

/-- Theorem: The average of the first 21 multiples of 17 is 187 -/
theorem average_of_21_multiples_of_17 : 
  average_of_multiples 21 17 = 187 := by
  sorry

end NUMINAMATH_CALUDE_average_of_21_multiples_of_17_l2623_262317


namespace NUMINAMATH_CALUDE_assembly_line_production_rate_l2623_262336

theorem assembly_line_production_rate 
  (initial_rate : ℝ) 
  (initial_order : ℝ) 
  (second_order : ℝ) 
  (average_output : ℝ) 
  (h1 : initial_rate = 90) 
  (h2 : initial_order = 60) 
  (h3 : second_order = 60) 
  (h4 : average_output = 72) : 
  ∃ (reduced_rate : ℝ), 
    reduced_rate = 60 ∧ 
    (initial_order / initial_rate + second_order / reduced_rate) * average_output = initial_order + second_order :=
by sorry

end NUMINAMATH_CALUDE_assembly_line_production_rate_l2623_262336


namespace NUMINAMATH_CALUDE_sin_405_plus_cos_neg_270_l2623_262373

theorem sin_405_plus_cos_neg_270 : 
  Real.sin (405 * π / 180) + Real.cos (-270 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_plus_cos_neg_270_l2623_262373


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2623_262316

/-- 
Given a natural number n, such that in the expansion of (x³ + 1/x²)^n,
the coefficient of the fourth term is the largest,
prove that the coefficient of the term with x³ is 20.
-/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (∃ k : ℕ, k = 3 ∧ 
    ∀ m : ℕ, m ≠ k → 
      Nat.choose n k ≥ Nat.choose n m) → 
  (∃ r : ℕ, Nat.choose n r * (3 * n - 5 * r) = 3 ∧ 
    Nat.choose n r = 20) := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2623_262316


namespace NUMINAMATH_CALUDE_smallest_sticker_collection_l2623_262353

theorem smallest_sticker_collection (S : ℕ) : 
  S > 2 →
  S % 5 = 2 →
  S % 11 = 2 →
  S % 13 = 2 →
  (∀ T : ℕ, T > 2 ∧ T % 5 = 2 ∧ T % 11 = 2 ∧ T % 13 = 2 → S ≤ T) →
  S = 717 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sticker_collection_l2623_262353


namespace NUMINAMATH_CALUDE_equation_solution_l2623_262332

theorem equation_solution : 
  ∃ x : ℝ, x ≠ 2 ∧ (3 / (x - 2) = 2 + x / (2 - x)) ↔ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2623_262332


namespace NUMINAMATH_CALUDE_common_point_for_gp_lines_l2623_262341

/-- A line in the form ax + by = c where a, b, c form a geometric progression -/
structure GPLine where
  a : ℝ
  r : ℝ
  h_r_nonzero : r ≠ 0

/-- The equation of a GPLine -/
def GPLine.equation (l : GPLine) (x y : ℝ) : Prop :=
  l.a * x + (l.a * l.r) * y = l.a * l.r^2

theorem common_point_for_gp_lines :
  ∀ (l : GPLine), l.equation 1 0 :=
sorry

end NUMINAMATH_CALUDE_common_point_for_gp_lines_l2623_262341


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2623_262384

theorem arithmetic_calculation : (-0.5) - (-3.2) + 2.8 - 6.5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2623_262384


namespace NUMINAMATH_CALUDE_gcd_2146_1813_l2623_262331

theorem gcd_2146_1813 : Nat.gcd 2146 1813 = 37 := by sorry

end NUMINAMATH_CALUDE_gcd_2146_1813_l2623_262331


namespace NUMINAMATH_CALUDE_complex_abs_one_plus_i_over_i_l2623_262356

theorem complex_abs_one_plus_i_over_i : Complex.abs ((1 + Complex.I) / Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_one_plus_i_over_i_l2623_262356


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2623_262399

/-- Predicate to check if exactly one of three numbers is negative -/
def exactlyOneNegative (a b c : ℝ) : Prop :=
  (a < 0 ∧ b > 0 ∧ c > 0) ∨ (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a > 0 ∧ b > 0 ∧ c < 0)

/-- The main theorem stating that there exists exactly one solution -/
theorem unique_solution_exists :
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a = Real.sqrt (b * c) ∧
    b = Real.sqrt (c * a) ∧
    c = Real.sqrt (a * b) ∧
    exactlyOneNegative a b c :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2623_262399


namespace NUMINAMATH_CALUDE_janet_muffins_count_l2623_262346

theorem janet_muffins_count :
  ∀ (muffin_cost : ℚ) (paid : ℚ) (change : ℚ),
    muffin_cost = 75 / 100 →
    paid = 20 →
    change = 11 →
    (paid - change) / muffin_cost = 12 :=
by sorry

end NUMINAMATH_CALUDE_janet_muffins_count_l2623_262346


namespace NUMINAMATH_CALUDE_product_of_numbers_l2623_262357

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 24) (sum_squares_eq : x^2 + y^2 = 400) : x * y = 88 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2623_262357


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2623_262339

-- Define the radical conjugate
def radical_conjugate (a b : ℝ) : ℝ := a + b

-- State the theorem
theorem sum_with_radical_conjugate :
  let x := 8 - Real.sqrt 1369
  x + radical_conjugate 8 (Real.sqrt 1369) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2623_262339


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l2623_262368

theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) > 0) 
  (h2 : Real.cos α < 0) : 
  π < α ∧ α < 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l2623_262368


namespace NUMINAMATH_CALUDE_constant_term_value_l2623_262337

/-- The binomial expansion of (x - 2/x)^8 has its maximum coefficient in the 5th term -/
def max_coeff_5th_term (x : ℝ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ 8 → Nat.choose 8 4 ≥ Nat.choose 8 k

/-- The constant term in the binomial expansion of (x - 2/x)^8 -/
def constant_term (x : ℝ) : ℤ :=
  Nat.choose 8 4 * (-2)^4

theorem constant_term_value (x : ℝ) :
  max_coeff_5th_term x → constant_term x = 1120 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l2623_262337


namespace NUMINAMATH_CALUDE_teacher_arrangement_count_teacher_arrangement_proof_l2623_262302

theorem teacher_arrangement_count : Nat → Nat → Nat
  | n, k => Nat.choose n k

theorem teacher_arrangement_proof :
  teacher_arrangement_count 22 5 = 26334 := by
  sorry

end NUMINAMATH_CALUDE_teacher_arrangement_count_teacher_arrangement_proof_l2623_262302


namespace NUMINAMATH_CALUDE_fran_required_speed_l2623_262304

/-- Calculates the required average speed for Fran to cover the same distance as Joann -/
theorem fran_required_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5) : 
  (joann_speed * joann_time) / fran_time = 120 / 7 := by
  sorry

#check fran_required_speed

end NUMINAMATH_CALUDE_fran_required_speed_l2623_262304


namespace NUMINAMATH_CALUDE_union_covers_reals_l2623_262334

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -3 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

-- State the theorem
theorem union_covers_reals (a : ℝ) : A ∪ B a = Set.univ → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l2623_262334


namespace NUMINAMATH_CALUDE_total_sums_attempted_l2623_262311

/-- Given a student's math problem attempt results, calculate the total number of sums attempted. -/
theorem total_sums_attempted (right_sums wrong_sums : ℕ) 
  (h1 : wrong_sums = 2 * right_sums) 
  (h2 : right_sums = 16) : 
  right_sums + wrong_sums = 48 :=
by sorry

end NUMINAMATH_CALUDE_total_sums_attempted_l2623_262311


namespace NUMINAMATH_CALUDE_smallest_k_value_l2623_262371

theorem smallest_k_value : ∃ (k : ℕ), k > 0 ∧
  (∀ (k' : ℕ), k' > 0 →
    (∃ (n : ℕ), n > 0 ∧ 2000 < n ∧ n < 3000 ∧
      (∀ (i : ℕ), 2 ≤ i ∧ i ≤ k' → n % i = i - 1)) →
    k ≤ k') ∧
  k = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_value_l2623_262371


namespace NUMINAMATH_CALUDE_geometric_ratio_in_arithmetic_sequence_l2623_262342

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- State the theorem
theorem geometric_ratio_in_arithmetic_sequence
  (a₁ d : ℝ) (h : d ≠ 0) :
  let a := arithmetic_sequence a₁ d
  (a 2) * (a 6) = (a 3)^2 →
  (a 3) / (a 2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_ratio_in_arithmetic_sequence_l2623_262342


namespace NUMINAMATH_CALUDE_university_box_cost_l2623_262352

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dim : BoxDimensions) : ℕ :=
  dim.length * dim.width * dim.height

/-- Calculates the number of boxes needed given the total volume and box volume -/
def boxesNeeded (totalVolume boxVolume : ℕ) : ℕ :=
  (totalVolume + boxVolume - 1) / boxVolume

/-- Calculates the total cost given the number of boxes and cost per box -/
def totalCost (numBoxes : ℕ) (costPerBox : ℚ) : ℚ :=
  (numBoxes : ℚ) * costPerBox

/-- Theorem stating the minimum amount the university must spend on boxes -/
theorem university_box_cost
  (boxDim : BoxDimensions)
  (costPerBox : ℚ)
  (totalVolume : ℕ)
  (h1 : boxDim = ⟨20, 20, 15⟩)
  (h2 : costPerBox = 6/5)
  (h3 : totalVolume = 3060000) :
  totalCost (boxesNeeded totalVolume (boxVolume boxDim)) costPerBox = 612 := by
  sorry


end NUMINAMATH_CALUDE_university_box_cost_l2623_262352


namespace NUMINAMATH_CALUDE_smallest_square_partition_l2623_262324

theorem smallest_square_partition : ∃ (n : ℕ),
  (n > 0) ∧ 
  (∃ (a b c : ℕ), 
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
    (a + b + c = 12) ∧ 
    (a ≥ 9) ∧
    (n^2 = a * 1^2 + b * 2^2 + c * 3^2)) ∧
  (∀ (m : ℕ), m < n → 
    ¬(∃ (a b c : ℕ),
      (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
      (a + b + c = 12) ∧
      (a ≥ 9) ∧
      (m^2 = a * 1^2 + b * 2^2 + c * 3^2))) ∧
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l2623_262324


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_l2623_262350

theorem modulo_equivalence_unique : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15725 [MOD 16] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_l2623_262350


namespace NUMINAMATH_CALUDE_smallest_triangle_perimeter_l2623_262377

theorem smallest_triangle_perimeter :
  ∀ a b c : ℕ,
  a ≥ 5 →
  b = a + 1 →
  c = b + 1 →
  a + b > c →
  a + c > b →
  b + c > a →
  ∀ x y z : ℕ,
  x ≥ 5 →
  y = x + 1 →
  z = y + 1 →
  x + y > z →
  x + z > y →
  y + z > x →
  a + b + c ≤ x + y + z :=
by sorry

end NUMINAMATH_CALUDE_smallest_triangle_perimeter_l2623_262377


namespace NUMINAMATH_CALUDE_job_arrangements_l2623_262300

/-- The number of ways to arrange n distinct candidates into k distinct jobs,
    where each job requires exactly one person and each person can take only one job. -/
def arrangements (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

/-- There are 3 different jobs, each requiring only one person,
    and each person taking on only one job.
    There are 4 candidates available for selection. -/
theorem job_arrangements : arrangements 4 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_job_arrangements_l2623_262300


namespace NUMINAMATH_CALUDE_cubic_root_form_l2623_262320

theorem cubic_root_form (x : ℝ) (hx : x > 0) (hcubic : x^3 - 4*x^2 - 2*x - Real.sqrt 3 = 0) :
  ∃ (a b : ℝ), x = a + b * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_form_l2623_262320


namespace NUMINAMATH_CALUDE_special_function_properties_l2623_262313

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧
  (∀ x y : ℝ, f (x + y) = f x + f y + f x * f y) ∧
  (∀ x : ℝ, x > 0 → f x > 0)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 0 = 0) ∧
  (∀ n : ℕ, f (n + 1) + 1 = 2 * (f n + 1)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → f (x + y) > f x) :=
by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l2623_262313


namespace NUMINAMATH_CALUDE_problem_solution_l2623_262361

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 119) : x = 39 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2623_262361


namespace NUMINAMATH_CALUDE_rectangles_count_l2623_262381

/-- A structure representing a square divided into rectangles -/
structure DividedSquare where
  k : ℕ  -- number of rectangles intersected by a vertical line
  l : ℕ  -- number of rectangles intersected by a horizontal line

/-- The total number of rectangles in a divided square -/
def total_rectangles (sq : DividedSquare) : ℕ := sq.k * sq.l

/-- Theorem stating that the total number of rectangles is k * l -/
theorem rectangles_count (sq : DividedSquare) : 
  total_rectangles sq = sq.k * sq.l := by sorry

end NUMINAMATH_CALUDE_rectangles_count_l2623_262381


namespace NUMINAMATH_CALUDE_short_trees_planted_is_57_l2623_262326

/-- Represents the number of trees in the park --/
structure ParkTrees where
  initialShortTrees : ℕ
  finalShortTrees : ℕ

/-- Calculates the number of short trees planted --/
def shortTreesPlanted (park : ParkTrees) : ℕ :=
  park.finalShortTrees - park.initialShortTrees

/-- Theorem: The number of short trees planted is 57 --/
theorem short_trees_planted_is_57 (park : ParkTrees) 
  (h1 : park.initialShortTrees = 41)
  (h2 : park.finalShortTrees = 98) : 
  shortTreesPlanted park = 57 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_planted_is_57_l2623_262326


namespace NUMINAMATH_CALUDE_min_k_10_l2623_262366

-- Define a stringent function
def Stringent (h : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 ∧ y > 0 → h x + h y > 2 * y^2

-- Define the sum of k from 1 to 15
def SumK (k : ℕ → ℤ) : ℤ :=
  (List.range 15).map (fun i => k (i + 1)) |> List.sum

-- Theorem statement
theorem min_k_10 (k : ℕ → ℤ) (hk : Stringent k) 
  (hmin : ∀ j : ℕ → ℤ, Stringent j → SumK k ≤ SumK j) : 
  k 10 ≥ 120 := by
  sorry

end NUMINAMATH_CALUDE_min_k_10_l2623_262366


namespace NUMINAMATH_CALUDE_new_model_count_l2623_262380

/-- Given an initial cost per model and the ability to buy a certain number of models,
    calculate the new number of models that can be purchased after a price increase. -/
theorem new_model_count (initial_cost new_cost : ℚ) (initial_count : ℕ) : 
  initial_cost > 0 →
  new_cost > 0 →
  initial_count > 0 →
  (initial_cost * initial_count) / new_cost = 27 →
  initial_cost = 0.45 →
  new_cost = 0.50 →
  initial_count = 30 →
  ⌊(initial_cost * initial_count) / new_cost⌋ = 27 := by
sorry

#eval (0.45 * 30) / 0.50

end NUMINAMATH_CALUDE_new_model_count_l2623_262380


namespace NUMINAMATH_CALUDE_ellipse_condition_l2623_262306

def is_ellipse (k : ℝ) : Prop :=
  1 < k ∧ k < 5 ∧ k ≠ 3

theorem ellipse_condition (k : ℝ) :
  (is_ellipse k → (1 < k ∧ k < 5)) ∧
  ¬(1 < k ∧ k < 5 → is_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2623_262306


namespace NUMINAMATH_CALUDE_equation_solution_l2623_262378

theorem equation_solution : ∃ y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2623_262378


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l2623_262343

theorem fraction_sum_proof : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l2623_262343


namespace NUMINAMATH_CALUDE_swim_time_calculation_l2623_262396

/-- 
Given a person's swimming speed in still water, the speed of the water current,
and the time taken to swim with the current for a certain distance,
calculate the time taken to swim back against the current for the same distance.
-/
theorem swim_time_calculation (still_speed water_speed with_current_time : ℝ) 
  (still_speed_pos : still_speed > 0)
  (water_speed_pos : water_speed > 0)
  (with_current_time_pos : with_current_time > 0)
  (h_still_speed : still_speed = 16)
  (h_water_speed : water_speed = 8)
  (h_with_current_time : with_current_time = 1.5) :
  let against_current_speed := still_speed - water_speed
  let with_current_speed := still_speed + water_speed
  let distance := with_current_speed * with_current_time
  let against_current_time := distance / against_current_speed
  against_current_time = 4.5 := by
sorry

end NUMINAMATH_CALUDE_swim_time_calculation_l2623_262396


namespace NUMINAMATH_CALUDE_multiply_72519_and_9999_l2623_262329

theorem multiply_72519_and_9999 : 72519 * 9999 = 724817481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_and_9999_l2623_262329


namespace NUMINAMATH_CALUDE_salt_addition_problem_l2623_262383

theorem salt_addition_problem (x : ℝ) (salt_added : ℝ) : 
  x = 104.99999999999997 →
  let initial_salt := 0.2 * x
  let water_after_evaporation := 0.75 * x
  let volume_after_evaporation := water_after_evaporation + initial_salt
  let final_volume := volume_after_evaporation + 7 + salt_added
  let final_salt := initial_salt + salt_added
  (final_salt / final_volume = 1/3) →
  salt_added = 11.375 := by
sorry

#eval (11.375 : Float)

end NUMINAMATH_CALUDE_salt_addition_problem_l2623_262383


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2623_262395

/-- Calculate the number of games in a chess tournament --/
def tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- The number of players in the tournament --/
def num_players : ℕ := 12

/-- The number of times each pair of players compete --/
def games_per_pair : ℕ := 2

theorem chess_tournament_games :
  tournament_games num_players * games_per_pair = 264 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2623_262395


namespace NUMINAMATH_CALUDE_daughter_and_child_weight_l2623_262358

/-- The combined weight of a daughter and her child given specific family weight conditions -/
theorem daughter_and_child_weight (total_weight mother_weight daughter_weight child_weight : ℝ) :
  total_weight = mother_weight + daughter_weight + child_weight →
  child_weight = (1 / 5 : ℝ) * mother_weight →
  daughter_weight = 48 →
  total_weight = 120 →
  daughter_weight + child_weight = 60 :=
by
  sorry

#check daughter_and_child_weight

end NUMINAMATH_CALUDE_daughter_and_child_weight_l2623_262358


namespace NUMINAMATH_CALUDE_cat_food_finished_l2623_262303

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day after a given number of days -/
def dayAfter (d : Day) (n : ℕ) : Day :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (match d with
    | Day.Monday => Day.Tuesday
    | Day.Tuesday => Day.Wednesday
    | Day.Wednesday => Day.Thursday
    | Day.Thursday => Day.Friday
    | Day.Friday => Day.Saturday
    | Day.Saturday => Day.Sunday
    | Day.Sunday => Day.Monday) n

/-- The amount of food consumed by the cat per day -/
def dailyConsumption : ℚ := 1/5 + 1/6

/-- The total amount of food in the box -/
def totalFood : ℚ := 10

/-- Theorem stating when the cat will finish the food -/
theorem cat_food_finished :
  ∃ (n : ℕ), n * dailyConsumption > totalFood ∧
  (n - 1) * dailyConsumption ≤ totalFood ∧
  dayAfter Day.Monday (n - 1) = Day.Wednesday :=
by sorry


end NUMINAMATH_CALUDE_cat_food_finished_l2623_262303


namespace NUMINAMATH_CALUDE_square_side_length_l2623_262372

theorem square_side_length (s : ℝ) (h : s > 0) : s^2 = 6 * (4 * s) → s = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2623_262372


namespace NUMINAMATH_CALUDE_lost_ship_depth_l2623_262363

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
def depth_of_lost_ship (descent_rate : ℝ) (time_taken : ℝ) : ℝ :=
  descent_rate * time_taken

/-- Theorem: The depth of the lost ship is 3600 feet. -/
theorem lost_ship_depth :
  let descent_rate : ℝ := 60
  let time_taken : ℝ := 60
  depth_of_lost_ship descent_rate time_taken = 3600 := by
sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l2623_262363


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l2623_262359

theorem sandcastle_height_difference (miki_height sister_height : ℝ) 
  (h1 : miki_height = 0.83)
  (h2 : sister_height = 0.5) :
  miki_height - sister_height = 0.33 := by sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l2623_262359


namespace NUMINAMATH_CALUDE_sum_multiple_of_five_l2623_262390

theorem sum_multiple_of_five (a b : ℤ) (ha : ∃ m : ℤ, a = 5 * m) (hb : ∃ n : ℤ, b = 10 * n) :
  ∃ k : ℤ, a + b = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_multiple_of_five_l2623_262390


namespace NUMINAMATH_CALUDE_optimal_line_and_minimum_value_l2623_262315

/-- A line passing through the origin with positive slope -/
structure PositiveSlopeLine where
  slope : ℝ
  positive : slope > 0

/-- A circle in the first quadrant -/
structure FirstQuadrantCircle where
  center : ℝ × ℝ
  radius : ℝ
  in_first_quadrant : center.1 ≥ 0 ∧ center.2 ≥ 0

/-- Predicate for two circles touching a line at the same point -/
def circles_touch_line_at_same_point (C1 C2 : FirstQuadrantCircle) (l : PositiveSlopeLine) : Prop :=
  ∃ (x y : ℝ), (y = l.slope * x) ∧
    ((x - C1.center.1)^2 + (y - C1.center.2)^2 = C1.radius^2) ∧
    ((x - C2.center.1)^2 + (y - C2.center.2)^2 = C2.radius^2)

/-- Predicate for a circle touching the x-axis at (1, 0) -/
def circle_touches_x_axis_at_one (C : FirstQuadrantCircle) : Prop :=
  C.center.1 = 1 ∧ C.center.2 = C.radius

/-- Predicate for a circle touching the y-axis -/
def circle_touches_y_axis (C : FirstQuadrantCircle) : Prop :=
  C.center.1 = C.radius

/-- Main theorem -/
theorem optimal_line_and_minimum_value
  (C1 C2 : FirstQuadrantCircle)
  (h1 : circle_touches_x_axis_at_one C1)
  (h2 : circle_touches_y_axis C2)
  (h3 : ∀ l : PositiveSlopeLine, circles_touch_line_at_same_point C1 C2 l) :
  ∃ (l : PositiveSlopeLine),
    l.slope = 4/3 ∧
    (∀ l' : PositiveSlopeLine, 8 * C1.radius + 9 * C2.radius ≤ 8 * C1.radius + 9 * C2.radius) ∧
    8 * C1.radius + 9 * C2.radius = 7 :=
  sorry

end NUMINAMATH_CALUDE_optimal_line_and_minimum_value_l2623_262315


namespace NUMINAMATH_CALUDE_length_QR_is_4_l2623_262369

-- Define the points and circles
structure Point := (x : ℝ) (y : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Define the given conditions
axiom P : Point
axiom Q : Point
axiom circle_P : Circle
axiom circle_Q : Circle
axiom R : Point

-- Radii of circles
axiom radius_P : circle_P.radius = 7
axiom radius_Q : circle_Q.radius = 4

-- Circles are externally tangent
axiom externally_tangent : 
  Real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2) = circle_P.radius + circle_Q.radius

-- R is on the line tangent to both circles
axiom R_on_tangent_line : 
  ∃ (S T : Point),
    (Real.sqrt ((S.x - P.x)^2 + (S.y - P.y)^2) = circle_P.radius) ∧
    (Real.sqrt ((T.x - Q.x)^2 + (T.y - Q.y)^2) = circle_Q.radius) ∧
    (R.x - S.x) * (T.y - S.y) = (R.y - S.y) * (T.x - S.x)

-- R is on ray PQ
axiom R_on_ray_PQ :
  ∃ (t : ℝ), t ≥ 0 ∧ R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)

-- Vertical line from Q is tangent to circle at P
axiom vertical_tangent :
  ∃ (U : Point),
    U.x = P.x ∧ 
    Real.sqrt ((U.x - P.x)^2 + (U.y - P.y)^2) = circle_P.radius ∧
    U.x - Q.x = 0

-- Theorem to prove
theorem length_QR_is_4 :
  Real.sqrt ((R.x - Q.x)^2 + (R.y - Q.y)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_length_QR_is_4_l2623_262369


namespace NUMINAMATH_CALUDE_ap_terms_count_l2623_262398

theorem ap_terms_count (a d : ℚ) (n : ℕ) (h_even : Even n) :
  (n / 2 : ℚ) * (2 * a + (n - 2) * d) = 30 →
  (n / 2 : ℚ) * (2 * a + n * d) = 50 →
  (n - 1) * d = 15 →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ap_terms_count_l2623_262398


namespace NUMINAMATH_CALUDE_video_games_spent_l2623_262322

def total_allowance : ℚ := 50

def books_fraction : ℚ := 1/4
def snacks_fraction : ℚ := 2/5
def apps_fraction : ℚ := 1/5

def books_spent : ℚ := total_allowance * books_fraction
def snacks_spent : ℚ := total_allowance * snacks_fraction
def apps_spent : ℚ := total_allowance * apps_fraction

theorem video_games_spent :
  total_allowance - (books_spent + snacks_spent + apps_spent) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_video_games_spent_l2623_262322


namespace NUMINAMATH_CALUDE_linear_equation_with_solution_l2623_262308

/-- A linear equation with two variables that has a specific solution -/
theorem linear_equation_with_solution :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a * x + b * y = c ↔ x = -3 ∧ y = 1) ∧
    a ≠ 0 ∧ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_with_solution_l2623_262308


namespace NUMINAMATH_CALUDE_toy_cost_price_l2623_262387

/-- The cost price of a toy -/
def cost_price : ℕ := sorry

/-- The number of toys sold -/
def toys_sold : ℕ := 18

/-- The total selling price of all toys -/
def total_selling_price : ℕ := 16800

/-- The number of toys whose cost price equals the gain -/
def toys_equal_to_gain : ℕ := 3

theorem toy_cost_price : 
  cost_price * (toys_sold + toys_equal_to_gain) = total_selling_price ∧ 
  cost_price = 800 := by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l2623_262387


namespace NUMINAMATH_CALUDE_three_digit_number_solution_l2623_262355

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Convert a repeating decimal of the form 0.ab̅ab to a fraction -/
def repeating_decimal_to_fraction (a b : Digit) : Rat :=
  (10 * a.val + b.val : Rat) / 99

/-- Convert a repeating decimal of the form 0.abc̅abc to a fraction -/
def repeating_decimal_to_fraction_3 (a b c : Digit) : Rat :=
  (100 * a.val + 10 * b.val + c.val : Rat) / 999

/-- The main theorem -/
theorem three_digit_number_solution (c d e : Digit) :
  repeating_decimal_to_fraction c d + repeating_decimal_to_fraction_3 c d e = 44 / 99 →
  (c.val * 100 + d.val * 10 + e.val : Nat) = 400 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_solution_l2623_262355


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2623_262312

/-- A regular polygon with perimeter 180 and side length 15 has 12 sides -/
theorem regular_polygon_sides (P : ℝ) (s : ℝ) (h1 : P = 180) (h2 : s = 15) :
  P / s = 12 := by
  sorry

#check regular_polygon_sides

end NUMINAMATH_CALUDE_regular_polygon_sides_l2623_262312


namespace NUMINAMATH_CALUDE_bonus_calculation_l2623_262391

/-- Represents the initial bonus amount -/
def initial_bonus : ℝ := sorry

/-- Represents the value of stocks after one year -/
def final_value : ℝ := 1350

theorem bonus_calculation : 
  (2 * (initial_bonus / 3) + 2 * (initial_bonus / 3) + (initial_bonus / 3) / 2) = final_value →
  initial_bonus = 900 := by
  sorry

end NUMINAMATH_CALUDE_bonus_calculation_l2623_262391


namespace NUMINAMATH_CALUDE_equidistant_point_is_perpendicular_bisector_intersection_l2623_262348

-- Define a triangle in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point in a 2D plane
def Point := ℝ × ℝ

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define a perpendicular bisector of a line segment
def perpendicularBisector (p1 p2 : Point) : Set Point := sorry

-- Define the intersection of three sets
def intersectionOfThree (s1 s2 s3 : Set Point) : Set Point := sorry

-- Theorem statement
theorem equidistant_point_is_perpendicular_bisector_intersection (t : Triangle) :
  ∃ (p : Point),
    (distance p t.A = distance p t.B ∧ distance p t.B = distance p t.C) ↔
    p ∈ intersectionOfThree
      (perpendicularBisector t.A t.B)
      (perpendicularBisector t.B t.C)
      (perpendicularBisector t.C t.A) :=
sorry

end NUMINAMATH_CALUDE_equidistant_point_is_perpendicular_bisector_intersection_l2623_262348


namespace NUMINAMATH_CALUDE_lever_force_calculation_l2623_262347

/-- Represents the force required to move an object with a lever -/
structure LeverForce where
  length : ℝ
  force : ℝ

/-- The inverse relationship between force and lever length -/
def inverse_relationship (k : ℝ) (lf : LeverForce) : Prop :=
  lf.force * lf.length = k

theorem lever_force_calculation (k : ℝ) (lf1 lf2 : LeverForce) :
  inverse_relationship k lf1 →
  inverse_relationship k lf2 →
  lf1.length = 12 →
  lf1.force = 200 →
  lf2.length = 8 →
  lf2.force = 300 :=
by
  sorry

#check lever_force_calculation

end NUMINAMATH_CALUDE_lever_force_calculation_l2623_262347


namespace NUMINAMATH_CALUDE_cubic_three_zeros_l2623_262393

/-- The cubic function f(x) = x^3 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- Theorem: The cubic function f(x) = x^3 + ax + 2 has exactly 3 real zeros if and only if a < -3 -/
theorem cubic_three_zeros (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_zeros_l2623_262393


namespace NUMINAMATH_CALUDE_mike_book_count_l2623_262388

/-- The number of books Tim has -/
def tim_books : ℕ := 22

/-- The total number of books Tim and Mike have together -/
def total_books : ℕ := 42

/-- The number of books Mike has -/
def mike_books : ℕ := total_books - tim_books

theorem mike_book_count : mike_books = 20 := by
  sorry

end NUMINAMATH_CALUDE_mike_book_count_l2623_262388


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2623_262375

theorem rationalize_denominator : 
  1 / (2 - Real.sqrt 2) = (2 + Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2623_262375


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_sold_l2623_262325

/-- Proves that the number of gem stone necklaces sold is 3 -/
theorem gem_stone_necklaces_sold (bead_necklaces : ℕ) (price_per_necklace : ℕ) (total_earnings : ℕ) :
  bead_necklaces = 4 →
  price_per_necklace = 3 →
  total_earnings = 21 →
  total_earnings = price_per_necklace * (bead_necklaces + 3) :=
by sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_sold_l2623_262325


namespace NUMINAMATH_CALUDE_older_brother_height_l2623_262379

theorem older_brother_height
  (younger_height : ℝ)
  (your_height : ℝ)
  (older_height : ℝ)
  (h1 : younger_height = 1.1)
  (h2 : your_height = younger_height + 0.2)
  (h3 : older_height = your_height + 0.1) :
  older_height = 1.4 := by
sorry

end NUMINAMATH_CALUDE_older_brother_height_l2623_262379


namespace NUMINAMATH_CALUDE_f_monotonicity_g_maximum_l2623_262321

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * x - 1
def g (x : ℝ) : ℝ := Real.log x - x

theorem f_monotonicity :
  (a ≤ 0 → ∀ x y, x < y → f a x < f a y) ∧
  (a > 0 → (∀ x y, Real.log a < x ∧ x < y → f a x < f a y) ∧
           (∀ x y, x < y ∧ y < Real.log a → f a x > f a y)) :=
sorry

theorem g_maximum :
  ∀ x > 0, g x ≤ g 1 ∧ g 1 = -1 :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_g_maximum_l2623_262321


namespace NUMINAMATH_CALUDE_cube_edge_length_l2623_262314

/-- The length of one edge of a cube given the sum of all edge lengths -/
theorem cube_edge_length (sum_of_edges : ℝ) (h : sum_of_edges = 144) : 
  sum_of_edges / 12 = 12 := by
  sorry

#check cube_edge_length

end NUMINAMATH_CALUDE_cube_edge_length_l2623_262314


namespace NUMINAMATH_CALUDE_binary_110011_is_51_l2623_262386

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_is_51_l2623_262386


namespace NUMINAMATH_CALUDE_product_of_max_and_min_l2623_262389

def numbers : List ℕ := [10, 11, 12]

theorem product_of_max_and_min : 
  (List.maximum numbers).get! * (List.minimum numbers).get! = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_max_and_min_l2623_262389


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2623_262374

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => x^3 - 8
  let g : ℝ → ℝ := λ x => 16 * (x + 1)^(1/3)
  ∀ x : ℝ, f x = g x ↔ x = -2 ∨ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2623_262374


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l2623_262354

/-- The perimeter of pentagon FGHIJ is 6, given that FG = GH = HI = IJ = 1 -/
theorem pentagon_perimeter (F G H I J : ℝ × ℝ) : 
  (dist F G = 1) → (dist G H = 1) → (dist H I = 1) → (dist I J = 1) →
  dist F G + dist G H + dist H I + dist I J + dist J F = 6 :=
by sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_l2623_262354


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l2623_262344

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (8/5) * x^2 - (18/5) * x - 1/5

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-1) = 5 ∧ q 2 = -1 ∧ q 4 = 11 := by
  sorry

#eval q (-1)
#eval q 2
#eval q 4

end NUMINAMATH_CALUDE_q_satisfies_conditions_l2623_262344


namespace NUMINAMATH_CALUDE_parabola_translation_l2623_262345

/-- Given a parabola y = -2x^2, prove that translating it upwards by 1 unit
    and to the right by 2 units results in the equation y = -2(x-2)^2 + 1 -/
theorem parabola_translation (x y : ℝ) :
  (y = -2 * x^2) →
  (y + 1 = -2 * ((x - 2)^2) + 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2623_262345


namespace NUMINAMATH_CALUDE_angle_c_measure_l2623_262360

theorem angle_c_measure (A B C : ℝ) (h : A + B = 90) : C = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_c_measure_l2623_262360


namespace NUMINAMATH_CALUDE_tutor_reunion_proof_l2623_262370

/-- The number of school days until all tutors work together again -/
def tutor_reunion_days : ℕ := 360

/-- Elisa's work schedule (every 5th day) -/
def elisa_schedule : ℕ := 5

/-- Frank's work schedule (every 6th day) -/
def frank_schedule : ℕ := 6

/-- Giselle's work schedule (every 8th day) -/
def giselle_schedule : ℕ := 8

/-- Hector's work schedule (every 9th day) -/
def hector_schedule : ℕ := 9

theorem tutor_reunion_proof :
  Nat.lcm elisa_schedule (Nat.lcm frank_schedule (Nat.lcm giselle_schedule hector_schedule)) = tutor_reunion_days :=
by sorry

end NUMINAMATH_CALUDE_tutor_reunion_proof_l2623_262370


namespace NUMINAMATH_CALUDE_least_four_digit_divisible_by_2_3_5_7_l2623_262362

theorem least_four_digit_divisible_by_2_3_5_7 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 1050 → ¬(2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n)) ∧
  (1050 ≥ 1000) ∧
  (2 ∣ 1050) ∧ (3 ∣ 1050) ∧ (5 ∣ 1050) ∧ (7 ∣ 1050) :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_divisible_by_2_3_5_7_l2623_262362


namespace NUMINAMATH_CALUDE_yoongi_result_l2623_262310

theorem yoongi_result (x : ℝ) (h : x / 10 = 6) : x - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_result_l2623_262310


namespace NUMINAMATH_CALUDE_joe_first_lift_weight_l2623_262364

theorem joe_first_lift_weight (first_lift second_lift : ℝ) 
  (total_weight : first_lift + second_lift = 800)
  (lift_relation : 3 * first_lift = 2 * second_lift + 450) :
  first_lift = 410 := by
sorry

end NUMINAMATH_CALUDE_joe_first_lift_weight_l2623_262364


namespace NUMINAMATH_CALUDE_remainder_theorem_l2623_262376

theorem remainder_theorem : ∃ q : ℕ, 2^160 + 160 = q * (2^80 + 2^40 + 1) + 160 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2623_262376


namespace NUMINAMATH_CALUDE_complement_of_M_l2623_262382

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | |x - 1| ≤ 2}

-- State the theorem
theorem complement_of_M : (U \ M) = {x | x < -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2623_262382


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_from_equation_l2623_262340

/-- Given a triangle ABC with side lengths a, b, and c satisfying the equation
    √(c² - a² - b²) + |a - b| = 0, prove that ABC is an isosceles right triangle. -/
theorem isosceles_right_triangle_from_equation 
  (a b c : ℝ) 
  (triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h : Real.sqrt (c^2 - a^2 - b^2) + |a - b| = 0) : 
  a = b ∧ c^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_from_equation_l2623_262340


namespace NUMINAMATH_CALUDE_stating_max_segments_proof_l2623_262307

/-- 
Given an equilateral triangle with side length n, divided into n^2 smaller 
equilateral triangles with side length 1, this function returns the maximum 
number of segments that can be chosen such that no three chosen segments 
form a triangle.
-/
def max_segments (n : ℕ) : ℕ :=
  n * (n + 1)

/-- 
Theorem stating that the maximum number of segments that can be chosen 
such that no three chosen segments form a triangle is n(n+1).
-/
theorem max_segments_proof (n : ℕ) : 
  max_segments n = n * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_stating_max_segments_proof_l2623_262307


namespace NUMINAMATH_CALUDE_base7_divisibility_l2623_262385

/-- Converts a base 7 number of the form 25y3₇ to base 10 -/
def base7ToBase10 (y : ℕ) : ℕ := 2 * 7^3 + 5 * 7^2 + y * 7 + 3

/-- Checks if a number is divisible by 19 -/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem base7_divisibility (y : ℕ) : 
  y < 7 → (isDivisibleBy19 (base7ToBase10 y) ↔ y = 3) := by sorry

end NUMINAMATH_CALUDE_base7_divisibility_l2623_262385


namespace NUMINAMATH_CALUDE_hooligan_theorem_l2623_262330

-- Define the universe
variable (Person : Type)

-- Define predicates
variable (isHooligan : Person → Prop)
variable (hasBeatlesHaircut : Person → Prop)
variable (hasRudeDemeanor : Person → Prop)

-- State the theorem
theorem hooligan_theorem 
  (exists_beatles_hooligan : ∃ x, isHooligan x ∧ hasBeatlesHaircut x)
  (all_hooligans_rude : ∀ y, isHooligan y → hasRudeDemeanor y) :
  (∃ z, isHooligan z ∧ hasRudeDemeanor z ∧ hasBeatlesHaircut z) ∧
  ¬(∀ w, isHooligan w ∧ hasRudeDemeanor w → hasBeatlesHaircut w) :=
by sorry

end NUMINAMATH_CALUDE_hooligan_theorem_l2623_262330


namespace NUMINAMATH_CALUDE_box_fillable_with_gamma_bricks_l2623_262323

/-- Represents a Γ-shape brick composed of three 1×1×1 cubes -/
structure GammaBrick :=
  (shape : Fin 3 → Fin 3 → Fin 3 → Bool)

/-- Represents a box with dimensions m × n × k -/
structure Box (m n k : ℕ) :=
  (filled : Fin m → Fin n → Fin k → Bool)

/-- Predicate to check if a box is completely filled with Γ-shape bricks -/
def is_filled_with_gamma_bricks (m n k : ℕ) (box : Box m n k) : Prop :=
  ∀ (i : Fin m) (j : Fin n) (l : Fin k), box.filled i j l = true

/-- Theorem stating that any box with dimensions m, n, k > 1 can be filled with Γ-shape bricks -/
theorem box_fillable_with_gamma_bricks (m n k : ℕ) (hm : m > 1) (hn : n > 1) (hk : k > 1) :
  ∃ (box : Box m n k), is_filled_with_gamma_bricks m n k box :=
sorry

end NUMINAMATH_CALUDE_box_fillable_with_gamma_bricks_l2623_262323


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2623_262367

theorem quadratic_equation_solution (x b : ℝ) : 
  3 * x^2 - b * x + 3 = 0 → x = 1 → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2623_262367


namespace NUMINAMATH_CALUDE_circle_radius_increase_l2623_262301

theorem circle_radius_increase (r₁ r₂ : ℝ) : 
  2 * Real.pi * r₁ = 30 → 2 * Real.pi * r₂ = 40 → r₂ - r₁ = 5 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_l2623_262301


namespace NUMINAMATH_CALUDE_divisible_by_six_l2623_262327

theorem divisible_by_six (n : ℕ) : ∃ k : ℤ, (2 * n^3 + 9 * n^2 + 13 * n : ℤ) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l2623_262327


namespace NUMINAMATH_CALUDE_smallest_y_value_l2623_262338

theorem smallest_y_value (y : ℝ) : 
  (2 * y^2 + 7 * y + 3 = 5) → (y ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_value_l2623_262338


namespace NUMINAMATH_CALUDE_function_symmetry_l2623_262318

/-- Given a function f and a real number a, 
    if f(x) = x³cos(x) + 1 and f(a) = 11, then f(-a) = -9 -/
theorem function_symmetry (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f x = x^3 * Real.cos x + 1) 
    (h2 : f a = 11) : 
  f (-a) = -9 := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l2623_262318


namespace NUMINAMATH_CALUDE_system_solution_existence_l2623_262305

/-- The system of equations has at least one solution if and only if b ≥ -2√2 - 1/4 -/
theorem system_solution_existence (b : ℝ) :
  (∃ a x y : ℝ, y = b - x^2 ∧ x^2 + y^2 + 2*a^2 = 4 - 2*a*(x + y)) ↔
  b ≥ -2 * Real.sqrt 2 - 1/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_existence_l2623_262305


namespace NUMINAMATH_CALUDE_unfoldable_cylinder_volume_l2623_262394

/-- A cylinder with a lateral surface that unfolds into a rectangle -/
structure UnfoldableCylinder where
  rectangle_length : ℝ
  rectangle_width : ℝ

/-- The volume of an unfoldable cylinder -/
def cylinder_volume (c : UnfoldableCylinder) : Set ℝ :=
  { v | ∃ (r h : ℝ), 
    ((2 * Real.pi * r = c.rectangle_length ∧ h = c.rectangle_width) ∨
     (2 * Real.pi * r = c.rectangle_width ∧ h = c.rectangle_length)) ∧
    v = Real.pi * r^2 * h }

/-- Theorem: The volume of a cylinder with lateral surface unfolding to a 4π by 1 rectangle is either 4π or 1 -/
theorem unfoldable_cylinder_volume :
  let c := UnfoldableCylinder.mk (4 * Real.pi) 1
  cylinder_volume c = {4 * Real.pi, 1} := by
  sorry

end NUMINAMATH_CALUDE_unfoldable_cylinder_volume_l2623_262394


namespace NUMINAMATH_CALUDE_score_correction_effect_l2623_262328

def class_size : ℕ := 50
def initial_average : ℝ := 70
def initial_variance : ℝ := 102
def incorrect_score1 : ℝ := 50
def correct_score1 : ℝ := 80
def incorrect_score2 : ℝ := 90
def correct_score2 : ℝ := 60

theorem score_correction_effect :
  let total_score := class_size * initial_average
  let corrected_total_score := total_score - incorrect_score1 - incorrect_score2 + correct_score1 + correct_score2
  let corrected_average := corrected_total_score / class_size
  let variance_contribution_change := (correct_score1 - initial_average)^2 + (correct_score2 - initial_average)^2 -
                                      (incorrect_score1 - initial_average)^2 - (incorrect_score2 - initial_average)^2
  let corrected_variance := initial_variance - variance_contribution_change / class_size
  corrected_average = initial_average ∧ corrected_variance = 90 := by
  sorry

end NUMINAMATH_CALUDE_score_correction_effect_l2623_262328


namespace NUMINAMATH_CALUDE_triangle_side_length_l2623_262319

theorem triangle_side_length (b c : ℝ) (h1 : b^2 - 7*b + 11 = 0) (h2 : c^2 - 7*c + 11 = 0) : 
  let a := Real.sqrt ((b^2 + c^2) - b*c)
  (a = 4) ∧ (b + c = 7) ∧ (b*c = 11) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2623_262319
