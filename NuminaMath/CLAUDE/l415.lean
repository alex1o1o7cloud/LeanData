import Mathlib

namespace NUMINAMATH_CALUDE_sample_size_is_176_l415_41571

/-- Represents the number of students in a stratum -/
structure Stratum where
  size : ℕ

/-- Represents a sample taken from a stratum -/
structure Sample where
  size : ℕ

/-- Calculates the total sample size for stratified sampling -/
def stratifiedSampleSize (male : Stratum) (female : Stratum) (femaleSample : Sample) : ℕ :=
  let maleSampleSize := (male.size * femaleSample.size) / female.size
  maleSampleSize + femaleSample.size

/-- Theorem: The total sample size is 176 given the specified conditions -/
theorem sample_size_is_176
  (male : Stratum)
  (female : Stratum)
  (femaleSample : Sample)
  (h1 : male.size = 1200)
  (h2 : female.size = 1000)
  (h3 : femaleSample.size = 80) :
  stratifiedSampleSize male female femaleSample = 176 := by
  sorry

#check sample_size_is_176

end NUMINAMATH_CALUDE_sample_size_is_176_l415_41571


namespace NUMINAMATH_CALUDE_river_road_cars_l415_41587

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 3 →
  buses = cars - 40 →
  cars = 60 := by
sorry

end NUMINAMATH_CALUDE_river_road_cars_l415_41587


namespace NUMINAMATH_CALUDE_unique_number_with_digit_sum_14_l415_41549

/-- Converts a decimal number to its octal representation -/
def toOctal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Sums the digits of a natural number in base 10 -/
def sumDigits (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (acc : ℕ) :=
    if m = 0 then acc
    else aux (m / 10) (acc + m % 10)
  aux n 0

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem unique_number_with_digit_sum_14 :
  ∃! n : ℕ,
    n > 0 ∧
    n < 1000 ∧
    (toOctal n).length = 3 ∧
    sumDigits n = 14 ∧
    sumList (toOctal n) = 14 ∧
    n = 455 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_digit_sum_14_l415_41549


namespace NUMINAMATH_CALUDE_midpoint_specific_segment_l415_41504

/-- The midpoint of a line segment in polar coordinates -/
def midpoint_polar (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_specific_segment :
  let p₁ : ℝ × ℝ := (5, π/6)
  let p₂ : ℝ × ℝ := (5, -π/6)
  let m : ℝ × ℝ := midpoint_polar p₁.1 p₁.2 p₂.1 p₂.2
  m.1 > 0 ∧ 0 ≤ m.2 ∧ m.2 < 2*π ∧ m = (5*Real.sqrt 3/2, π/6) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_specific_segment_l415_41504


namespace NUMINAMATH_CALUDE_emily_unanswered_questions_l415_41572

def total_questions : ℕ := 50
def new_score : ℕ := 120
def old_score : ℕ := 95

def scoring_systems (c w u : ℕ) : Prop :=
  (6 * c + u = new_score) ∧
  (50 + 3 * c - 2 * w = old_score) ∧
  (c + w + u = total_questions)

theorem emily_unanswered_questions :
  ∃ (c w u : ℕ), scoring_systems c w u ∧ u = 37 :=
by sorry

end NUMINAMATH_CALUDE_emily_unanswered_questions_l415_41572


namespace NUMINAMATH_CALUDE_value_two_std_dev_below_mean_l415_41567

-- Define the properties of the normal distribution
def mean : ℝ := 16.2
def std_dev : ℝ := 2.3

-- Define the value we're looking for
def value : ℝ := mean - 2 * std_dev

-- Theorem stating that the value is 11.6
theorem value_two_std_dev_below_mean :
  value = 11.6 := by sorry

end NUMINAMATH_CALUDE_value_two_std_dev_below_mean_l415_41567


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l415_41599

/-- Given two vectors a and b in R², prove that if they are parallel and a = (-1, 3) and b = (1, t), then t = -3. -/
theorem parallel_vectors_condition (a b : ℝ × ℝ) (t : ℝ) : 
  a = (-1, 3) → b = (1, t) → (∃ (k : ℝ), a = k • b) → t = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l415_41599


namespace NUMINAMATH_CALUDE_cubic_root_equation_l415_41564

theorem cubic_root_equation : 2 / (2 - Real.rpow 3 (1/3)) = 2 * (2 + Real.rpow 3 (1/3)) * (4 + Real.rpow 9 (1/3)) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_l415_41564


namespace NUMINAMATH_CALUDE_triangle_height_l415_41537

/-- Given a triangle with angles α, β, γ and side c, mc is the height corresponding to side c -/
theorem triangle_height (α β γ c mc : ℝ) (h_angles : α + β + γ = Real.pi) 
  (h_positive : 0 < c ∧ 0 < α ∧ 0 < β ∧ 0 < γ) :
  mc = (c * Real.sin α * Real.sin β) / Real.sin γ :=
sorry


end NUMINAMATH_CALUDE_triangle_height_l415_41537


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l415_41568

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 + I) :
  (z - 2) / z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l415_41568


namespace NUMINAMATH_CALUDE_f_properties_l415_41580

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1 - 2 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f x ≤ 2) ∧
  (∀ α, 0 < α ∧ α < Real.pi / 3 → (f α = 2 → α = Real.pi / 6)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l415_41580


namespace NUMINAMATH_CALUDE_probability_two_females_l415_41514

/-- The probability of selecting 2 female students from a group of 5 students (2 males and 3 females) -/
theorem probability_two_females (total_students : Nat) (male_students : Nat) (female_students : Nat) 
  (group_size : Nat) : 
  total_students = 5 → 
  male_students = 2 → 
  female_students = 3 → 
  group_size = 2 → 
  (Nat.choose female_students group_size : Rat) / (Nat.choose total_students group_size : Rat) = 3/10 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_females_l415_41514


namespace NUMINAMATH_CALUDE_eight_percent_of_1200_is_96_l415_41543

theorem eight_percent_of_1200_is_96 : 
  (8 / 100) * 1200 = 96 := by sorry

end NUMINAMATH_CALUDE_eight_percent_of_1200_is_96_l415_41543


namespace NUMINAMATH_CALUDE_pentagon_c_y_coordinate_l415_41501

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculates the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Checks if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

/-- The y-coordinate of vertex C in the given pentagon is 21 -/
theorem pentagon_c_y_coordinate :
  ∀ (p : Pentagon),
    p.A = (0, 0) →
    p.B = (0, 5) →
    p.D = (5, 5) →
    p.E = (5, 0) →
    hasVerticalSymmetry p →
    pentagonArea p = 65 →
    p.C.2 = 21 := by sorry

end NUMINAMATH_CALUDE_pentagon_c_y_coordinate_l415_41501


namespace NUMINAMATH_CALUDE_probability_twelve_rolls_eight_sided_die_l415_41592

/-- The probability of rolling an eight-sided die 12 times, where the first 11 rolls are all
    different from their immediate predecessors, and the 12th roll matches the 11th roll. -/
def probability_twelve_rolls (n : ℕ) : ℚ :=
  if n = 8 then
    (7 : ℚ)^10 / 8^11
  else
    0

/-- Theorem stating that the probability of the described event with an eight-sided die
    is equal to 7^10 / 8^11. -/
theorem probability_twelve_rolls_eight_sided_die :
  probability_twelve_rolls 8 = (7 : ℚ)^10 / 8^11 :=
by sorry

end NUMINAMATH_CALUDE_probability_twelve_rolls_eight_sided_die_l415_41592


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l415_41531

/-- The probability of selecting two non-defective pens from a box of 12 pens, where 6 are defective -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 12)
  (h2 : defective_pens = 6) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5 / 22 := by
  sorry

#check prob_two_non_defective_pens

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l415_41531


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l415_41528

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {-1, 0, 1, 2, 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 2}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l415_41528


namespace NUMINAMATH_CALUDE_zach_scored_42_points_l415_41521

def ben_points : ℝ := 21.0
def total_points : ℝ := 63

def zach_points : ℝ := total_points - ben_points

theorem zach_scored_42_points :
  zach_points = 42 := by sorry

end NUMINAMATH_CALUDE_zach_scored_42_points_l415_41521


namespace NUMINAMATH_CALUDE_product_of_sums_equals_x_l415_41576

theorem product_of_sums_equals_x : ∃ X : ℕ,
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) * (3 + 2) = X := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_x_l415_41576


namespace NUMINAMATH_CALUDE_property_one_property_two_property_three_f_satisfies_all_properties_l415_41507

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- Property 1: f(x₁x₂) = f(x₁)f(x₂)
theorem property_one : ∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂ := by sorry

-- Property 2: For x ∈ (0, +∞), f'(x) > 0
theorem property_two : ∀ x : ℝ, x > 0 → (deriv f) x > 0 := by sorry

-- Property 3: f'(x) is an odd function
theorem property_three : ∀ x : ℝ, (deriv f) (-x) = -(deriv f) x := by sorry

-- Main theorem: f(x) = x² satisfies all three properties
theorem f_satisfies_all_properties : 
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (∀ x : ℝ, x > 0 → (deriv f) x > 0) ∧ 
  (∀ x : ℝ, (deriv f) (-x) = -(deriv f) x) := by sorry

end NUMINAMATH_CALUDE_property_one_property_two_property_three_f_satisfies_all_properties_l415_41507


namespace NUMINAMATH_CALUDE_truck_wash_price_l415_41523

/-- Proves that the price of a truck wash is $6 given the conditions of Laura's carwash --/
theorem truck_wash_price (car_price : ℕ) (suv_price : ℕ) (total_raised : ℕ) 
  (num_cars num_suvs num_trucks : ℕ) :
  car_price = 5 →
  suv_price = 7 →
  num_cars = 7 →
  num_suvs = 5 →
  num_trucks = 5 →
  total_raised = 100 →
  ∃ (truck_price : ℕ), 
    truck_price = 6 ∧ 
    car_price * num_cars + suv_price * num_suvs + truck_price * num_trucks = total_raised :=
by sorry

end NUMINAMATH_CALUDE_truck_wash_price_l415_41523


namespace NUMINAMATH_CALUDE_constant_regular_cells_problem_solution_l415_41562

/-- Represents the number of regular cells capable of division after a given number of days -/
def regular_cells (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells

/-- Theorem stating that the number of regular cells remains constant -/
theorem constant_regular_cells (initial_cells : ℕ) (days : ℕ) :
  regular_cells initial_cells days = initial_cells :=
by sorry

/-- The specific case for the problem with 4 initial cells and 10 days -/
theorem problem_solution :
  regular_cells 4 10 = 4 :=
by sorry

end NUMINAMATH_CALUDE_constant_regular_cells_problem_solution_l415_41562


namespace NUMINAMATH_CALUDE_linear_equation_solution_l415_41541

theorem linear_equation_solution (a b : ℝ) : 
  (2 : ℝ) * a + (-1 : ℝ) * b = 2 → 2 * a - b - 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l415_41541


namespace NUMINAMATH_CALUDE_negative_three_inequality_l415_41534

theorem negative_three_inequality (a b : ℝ) (h : a < b) : -3*a > -3*b := by
  sorry

end NUMINAMATH_CALUDE_negative_three_inequality_l415_41534


namespace NUMINAMATH_CALUDE_sum_fraction_equality_l415_41539

theorem sum_fraction_equality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ k ∈ ({2, 3, 4, 5, 6} : Set ℕ), 
    (a₁ / (k^2 + 1) + a₂ / (k^2 + 2) + a₃ / (k^2 + 3) + a₄ / (k^2 + 4) + a₅ / (k^2 + 5)) = 1 / k^2) :
  a₁ / 2 + a₂ / 3 + a₃ / 4 + a₄ / 5 + a₅ / 6 = 57 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_equality_l415_41539


namespace NUMINAMATH_CALUDE_tip_percentage_is_15_percent_l415_41598

def lunch_cost : ℝ := 50.50
def total_spent : ℝ := 58.075

theorem tip_percentage_is_15_percent :
  (total_spent - lunch_cost) / lunch_cost * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_tip_percentage_is_15_percent_l415_41598


namespace NUMINAMATH_CALUDE_max_x_coordinate_P_max_x_coordinate_P_achieved_l415_41546

/-- The maximum x-coordinate of point P on line OA, where A is on the ellipse x²/16 + y²/4 = 1 and OA · OP = 6 -/
theorem max_x_coordinate_P (A : ℝ × ℝ) (P : ℝ × ℝ) : 
  (A.1^2 / 16 + A.2^2 / 4 = 1) →  -- A is on the ellipse
  (∃ t : ℝ, P = (t * A.1, t * A.2)) →  -- P is on the line OA
  (A.1 * P.1 + A.2 * P.2 = 6) →  -- OA · OP = 6
  P.1 ≤ Real.sqrt 3 := by
sorry

/-- The maximum x-coordinate of point P is achieved -/
theorem max_x_coordinate_P_achieved (A : ℝ × ℝ) : 
  (A.1^2 / 16 + A.2^2 / 4 = 1) →  -- A is on the ellipse
  ∃ P : ℝ × ℝ, 
    (∃ t : ℝ, P = (t * A.1, t * A.2)) ∧  -- P is on the line OA
    (A.1 * P.1 + A.2 * P.2 = 6) ∧  -- OA · OP = 6
    P.1 = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_x_coordinate_P_max_x_coordinate_P_achieved_l415_41546


namespace NUMINAMATH_CALUDE_equation_equivalence_l415_41579

theorem equation_equivalence (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  1 + 1/x + 2*(x+1)/(x*y) + 3*(x+1)*(y+2)/(x*y*z) + 4*(x+1)*(y+2)*(z+3)/(x*y*z*w) = 0 ↔
  (1 + 1/x) * (1 + 2/y) * (1 + 3/z) * (1 + 4/w) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l415_41579


namespace NUMINAMATH_CALUDE_f_nonnegative_l415_41593

/-- Definition of the function f --/
def f (A B C a b c : ℝ) : ℝ :=
  A * (a^3 + b^3 + c^3) + B * (a^2*b + b^2*c + c^2*a + a*b^2 + b*c^2 + c*a^2) + C * a * b * c

/-- Triangle inequality --/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem --/
theorem f_nonnegative (A B C : ℝ) :
  (f A B C 1 1 1 ≥ 0) →
  (f A B C 1 1 0 ≥ 0) →
  (f A B C 2 1 1 ≥ 0) →
  ∀ a b c : ℝ, is_triangle a b c → f A B C a b c ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_f_nonnegative_l415_41593


namespace NUMINAMATH_CALUDE_quadratic_minimum_l415_41510

theorem quadratic_minimum (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 18 * x + 7
  ∀ y : ℝ, f x ≤ f y ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l415_41510


namespace NUMINAMATH_CALUDE_january_oil_bill_l415_41500

theorem january_oil_bill (january_bill february_bill : ℚ) : 
  (february_bill / january_bill = 3 / 2) →
  ((february_bill + 20) / january_bill = 5 / 3) →
  january_bill = 120 := by
sorry

end NUMINAMATH_CALUDE_january_oil_bill_l415_41500


namespace NUMINAMATH_CALUDE_misread_number_correction_l415_41536

theorem misread_number_correction (n : ℕ) (incorrect_avg correct_avg incorrect_number : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 19)
  (h3 : correct_avg = 24)
  (h4 : incorrect_number = 26) :
  ∃ (correct_number : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * incorrect_avg = correct_number - incorrect_number ∧
    correct_number = 76 := by
  sorry

end NUMINAMATH_CALUDE_misread_number_correction_l415_41536


namespace NUMINAMATH_CALUDE_inequality_solution_l415_41513

theorem inequality_solution : 
  let S : Set ℚ := {-3, -1/2, 1/3, 2}
  ∀ x ∈ S, 2*(x-1)+3 < 0 ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l415_41513


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l415_41597

theorem triangle_abc_proof (a b c A B C S_ΔABC : Real) 
  (h1 : a = Real.sqrt 3)
  (h2 : b = Real.sqrt 2)
  (h3 : A = π / 3)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π)
  (h8 : Real.sin A / a = Real.sin B / b)
  (h9 : S_ΔABC = (1 / 2) * a * b * Real.sin C) :
  B = π / 4 ∧ S_ΔABC = (3 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l415_41597


namespace NUMINAMATH_CALUDE_equation_equivalence_l415_41505

theorem equation_equivalence (x : ℝ) : 
  (x^2 + x + 1) * (3*x + 4) * (-7*x + 2) * (2*x - Real.sqrt 5) * (-12*x - 16) = 0 ↔ 
  (3*x + 4 = 0 ∨ -7*x + 2 = 0 ∨ 2*x - Real.sqrt 5 = 0 ∨ -12*x - 16 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l415_41505


namespace NUMINAMATH_CALUDE_bridget_profit_l415_41506

def total_loaves : ℕ := 60
def morning_price : ℚ := 3
def afternoon_price : ℚ := 2
def late_price : ℚ := 3/2
def production_cost : ℚ := 4/5

def morning_sales : ℕ := total_loaves / 3
def afternoon_sales : ℕ := ((total_loaves - morning_sales) * 3) / 4
def late_sales : ℕ := total_loaves - morning_sales - afternoon_sales

def total_revenue : ℚ := 
  morning_sales * morning_price + 
  afternoon_sales * afternoon_price + 
  late_sales * late_price

def total_cost : ℚ := total_loaves * production_cost

def profit : ℚ := total_revenue - total_cost

theorem bridget_profit : profit = 87 := by sorry

end NUMINAMATH_CALUDE_bridget_profit_l415_41506


namespace NUMINAMATH_CALUDE_students_passed_l415_41542

def total_students : ℕ := 450

def failed_breakup : ℕ := (5 * total_students) / 12

def remaining_after_breakup : ℕ := total_students - failed_breakup

def no_show : ℕ := (7 * remaining_after_breakup) / 15

def remaining_after_no_show : ℕ := remaining_after_breakup - no_show

def penalized : ℕ := 45

def remaining_after_penalty : ℕ := remaining_after_no_show - penalized

def bonus_but_failed : ℕ := remaining_after_penalty / 8

theorem students_passed :
  total_students - failed_breakup - no_show - penalized - bonus_but_failed = 84 := by
  sorry

end NUMINAMATH_CALUDE_students_passed_l415_41542


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l415_41551

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l415_41551


namespace NUMINAMATH_CALUDE_min_colors_correct_min_colors_is_minimum_l415_41520

-- Define a function that returns the minimum number of colors needed
def min_colors (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Edge case: no keys
  | 1 => 1
  | 2 => 2
  | _ => 3

-- Theorem statement
theorem min_colors_correct (n : ℕ) :
  min_colors n = 
    if n = 0 then 0
    else if n = 1 then 1
    else if n = 2 then 2
    else 3 :=
by sorry

-- Theorem stating that this is indeed the minimum
theorem min_colors_is_minimum (n : ℕ) :
  ∀ (m : ℕ), m < min_colors n → ¬(∃ (coloring : Fin n → Fin m), ∀ (i j : Fin n), i ≠ j → coloring i ≠ coloring j) :=
by sorry

end NUMINAMATH_CALUDE_min_colors_correct_min_colors_is_minimum_l415_41520


namespace NUMINAMATH_CALUDE_waiter_remaining_customers_l415_41529

theorem waiter_remaining_customers 
  (initial_customers : Real) 
  (first_group_left : Real) 
  (second_group_left : Real) 
  (h1 : initial_customers = 36.0)
  (h2 : first_group_left = 19.0)
  (h3 : second_group_left = 14.0) : 
  initial_customers - first_group_left - second_group_left = 3.0 := by
sorry

end NUMINAMATH_CALUDE_waiter_remaining_customers_l415_41529


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l415_41526

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l415_41526


namespace NUMINAMATH_CALUDE_shopper_fraction_l415_41558

theorem shopper_fraction (total_shoppers : ℕ) (checkout_shoppers : ℕ) 
  (h1 : total_shoppers = 480) 
  (h2 : checkout_shoppers = 180) : 
  (total_shoppers - checkout_shoppers : ℚ) / total_shoppers = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_shopper_fraction_l415_41558


namespace NUMINAMATH_CALUDE_willy_tv_series_completion_time_l415_41517

/-- The number of days required to finish a TV series -/
def days_to_finish (seasons : ℕ) (episodes_per_season : ℕ) (episodes_per_day : ℕ) : ℕ :=
  (seasons * episodes_per_season) / episodes_per_day

/-- Theorem: It takes 30 days to finish the TV series under given conditions -/
theorem willy_tv_series_completion_time :
  days_to_finish 3 20 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_willy_tv_series_completion_time_l415_41517


namespace NUMINAMATH_CALUDE_village_population_l415_41511

theorem village_population (population_95_percent : ℝ) (h : population_95_percent = 57200) :
  ∃ total_population : ℕ, 
    (↑total_population : ℝ) ≥ population_95_percent / 0.95 ∧ 
    (↑total_population : ℝ) < population_95_percent / 0.95 + 1 ∧
    total_population = 60211 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l415_41511


namespace NUMINAMATH_CALUDE_wall_building_time_l415_41519

/-- Given that 8 persons can build a 140 m long wall in 42 days, 
    prove that 30 persons can complete a 100 m long wall in 8 days -/
theorem wall_building_time 
  (persons_initial : ℕ) 
  (length_initial : ℕ) 
  (days_initial : ℕ) 
  (persons_new : ℕ) 
  (length_new : ℕ) 
  (h1 : persons_initial = 8) 
  (h2 : length_initial = 140) 
  (h3 : days_initial = 42) 
  (h4 : persons_new = 30) 
  (h5 : length_new = 100) : 
  (persons_initial * days_initial * length_new) / (persons_new * length_initial) = 8 := by
  sorry

end NUMINAMATH_CALUDE_wall_building_time_l415_41519


namespace NUMINAMATH_CALUDE_no_triangle_two_right_angles_l415_41553

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_is_180 : a + b + c = 180

-- Theorem: No triangle can have two right angles
theorem no_triangle_two_right_angles :
  ∀ t : Triangle, ¬(t.a = 90 ∧ t.b = 90 ∨ t.a = 90 ∧ t.c = 90 ∨ t.b = 90 ∧ t.c = 90) :=
by
  sorry

end NUMINAMATH_CALUDE_no_triangle_two_right_angles_l415_41553


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l415_41583

def number_to_convert : ℝ := 280000

theorem scientific_notation_proof :
  number_to_convert = 2.8 * (10 : ℝ)^5 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l415_41583


namespace NUMINAMATH_CALUDE_min_side_length_l415_41552

theorem min_side_length (EF HG : ℝ) (EG HF : ℝ) (h1 : EF = 7) (h2 : EG = 15) (h3 : HG = 10) (h4 : HF = 25) :
  ∀ FG : ℝ, (FG > EG - EF ∧ FG > HF - HG) → FG ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l415_41552


namespace NUMINAMATH_CALUDE_power_multiplication_l415_41544

theorem power_multiplication (a : ℝ) : -a^4 * a^3 = -a^7 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l415_41544


namespace NUMINAMATH_CALUDE_digit_125_of_4_div_7_l415_41547

/-- The decimal representation of 4/7 has a 6-digit repeating sequence -/
def repeating_sequence_length : ℕ := 6

/-- The 125th digit after the decimal point in 4/7 -/
def target_digit : ℕ := 125

/-- The function that returns the nth digit in the decimal expansion of 4/7 -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_125_of_4_div_7 : nth_digit target_digit = 2 := by sorry

end NUMINAMATH_CALUDE_digit_125_of_4_div_7_l415_41547


namespace NUMINAMATH_CALUDE_problem_statement_l415_41573

theorem problem_statement (x y : ℝ) (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) : 
  |x| + |y| = 3/2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l415_41573


namespace NUMINAMATH_CALUDE_buffy_whiskers_l415_41596

/-- The number of whiskers for each cat -/
structure CatWhiskers where
  juniper : ℕ
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ

/-- The conditions for the cat whiskers problem -/
def catWhiskersConditions (c : CatWhiskers) : Prop :=
  c.juniper = 12 ∧
  c.puffy = 3 * c.juniper ∧
  c.scruffy = 2 * c.puffy ∧
  c.buffy = (c.juniper + c.puffy + c.scruffy) / 3

/-- Theorem stating that given the conditions, Buffy has 40 whiskers -/
theorem buffy_whiskers (c : CatWhiskers) (h : catWhiskersConditions c) : c.buffy = 40 := by
  sorry

end NUMINAMATH_CALUDE_buffy_whiskers_l415_41596


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l415_41512

open Real

theorem min_shift_for_symmetry (x : ℝ) :
  let f (x : ℝ) := cos (2 * x) + Real.sqrt 3 * sin (2 * x)
  ∃ m : ℝ, m > 0 ∧ 
    (∀ x, f (x + m) = f (-x + m)) ∧
    (∀ m' : ℝ, m' > 0 ∧ (∀ x, f (x + m') = f (-x + m')) → m ≤ m') ∧
    m = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l415_41512


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l415_41527

/-- Given a quadrilateral in the Cartesian plane, the sum of the x-coordinates
    of the midpoints of its sides is equal to the sum of the x-coordinates of its vertices. -/
theorem midpoint_sum_equals_vertex_sum (p q r s : ℝ) :
  let vertex_sum := p + q + r + s
  let midpoint_sum := (p + q) / 2 + (q + r) / 2 + (r + s) / 2 + (s + p) / 2
  midpoint_sum = vertex_sum := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l415_41527


namespace NUMINAMATH_CALUDE_g_inequality_l415_41502

def g (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem g_inequality : g (3/2) < g 0 ∧ g 0 < g 3 := by
  sorry

end NUMINAMATH_CALUDE_g_inequality_l415_41502


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l415_41525

/-- The trajectory of the midpoint of a line segment PQ, where P moves on the unit circle and Q is fixed at (3,0) -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x_p y_p : ℝ, x_p^2 + y_p^2 = 1 ∧ x = (x_p + 3)/2 ∧ y = y_p/2) → 
  (2*x - 3)^2 + 4*y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l415_41525


namespace NUMINAMATH_CALUDE_total_pebbles_is_50_l415_41584

/-- Represents the number of pebbles of each color and the total --/
structure PebbleCounts where
  white : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ
  total : ℕ

/-- Defines the conditions of the pebble problem --/
def pebble_problem (p : PebbleCounts) : Prop :=
  p.white = 20 ∧
  p.red = p.white / 2 ∧
  p.blue = p.red / 3 ∧
  p.green = p.blue + 5 ∧
  p.red = p.total / 5 ∧
  p.total = p.white + p.red + p.blue + p.green

/-- Theorem stating that the total number of pebbles is 50 --/
theorem total_pebbles_is_50 :
  ∃ p : PebbleCounts, pebble_problem p ∧ p.total = 50 :=
by sorry

end NUMINAMATH_CALUDE_total_pebbles_is_50_l415_41584


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l415_41559

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: For a geometric sequence where a_4 + a_8 = -2, 
    the value of a_6(a_2 + 2a_6 + a_10) is equal to 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : geometric_sequence a) 
    (h_sum : a 4 + a 8 = -2) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l415_41559


namespace NUMINAMATH_CALUDE_no_integer_solutions_l415_41595

theorem no_integer_solutions : ¬ ∃ (x y z : ℤ), 
  (x^2 - 2*x*y + 3*y^2 - 2*z^2 = 25) ∧ 
  (-x^2 + 4*y*z + 3*z^2 = 55) ∧ 
  (x^2 + 3*x*y - y^2 + 7*z^2 = 130) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l415_41595


namespace NUMINAMATH_CALUDE_reflection_sum_l415_41516

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (8, 7), then m + b = 11 -/
theorem reflection_sum (m b : ℝ) : 
  (∀ (x y : ℝ), y = m * x + b → 
    (x - 2) * (x - 8) + (y - 3) * (y - 7) = 0 ∧ 
    (x - 5) * (1 + m * m) = m * (y - 5)) → 
  m + b = 11 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_l415_41516


namespace NUMINAMATH_CALUDE_pies_sold_per_day_l415_41586

/-- Given a restaurant that sells pies every day for a week and sells 56 pies in total,
    prove that the number of pies sold each day is 8. -/
theorem pies_sold_per_day (total_pies : ℕ) (days_in_week : ℕ) 
  (h1 : total_pies = 56) 
  (h2 : days_in_week = 7) :
  total_pies / days_in_week = 8 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_per_day_l415_41586


namespace NUMINAMATH_CALUDE_min_value_quadratic_l415_41590

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧ 
  ∃ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 25 = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l415_41590


namespace NUMINAMATH_CALUDE_intersection_when_a_is_5_intersection_equals_A_iff_l415_41524

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + (5-a)*x - 5*a ≤ 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 6}

-- Define the complement of B
def complement_B : Set ℝ := {x | x < -3 ∨ 6 < x}

-- Theorem 1
theorem intersection_when_a_is_5 :
  A 5 ∩ complement_B = {x | -5 ≤ x ∧ x < -3} := by sorry

-- Theorem 2
theorem intersection_equals_A_iff (a : ℝ) :
  A a ∩ complement_B = A a ↔ a < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_5_intersection_equals_A_iff_l415_41524


namespace NUMINAMATH_CALUDE_white_balls_count_l415_41569

theorem white_balls_count (total : ℕ) (red : ℕ) (white : ℕ) : 
  red = 8 →
  red + white = total →
  (5 : ℚ) / 6 * total = white →
  white = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l415_41569


namespace NUMINAMATH_CALUDE_card_area_reduction_l415_41589

theorem card_area_reduction (initial_width initial_height : ℝ) 
  (h1 : initial_width = 10 ∧ initial_height = 8)
  (h2 : ∃ (reduced_side : ℝ), reduced_side = initial_width - 2 ∨ reduced_side = initial_height - 2)
  (h3 : ∃ (unreduced_side : ℝ), (reduced_side = initial_width - 2 → unreduced_side = initial_height) ∧
                                (reduced_side = initial_height - 2 → unreduced_side = initial_width))
  (h4 : reduced_side * unreduced_side = 64) :
  (initial_width - 2) * initial_height = 60 ∨ initial_width * (initial_height - 2) = 60 :=
sorry

end NUMINAMATH_CALUDE_card_area_reduction_l415_41589


namespace NUMINAMATH_CALUDE_angle_E_measure_l415_41530

structure Parallelogram where
  E : Real
  F : Real
  G : Real
  H : Real

def external_angle (p : Parallelogram) : Real := 50

theorem angle_E_measure (p : Parallelogram) :
  external_angle p = 50 → p.E = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_E_measure_l415_41530


namespace NUMINAMATH_CALUDE_cos_six_arccos_one_fourth_l415_41548

theorem cos_six_arccos_one_fourth : 
  Real.cos (6 * Real.arccos (1/4)) = -7/128 := by
  sorry

end NUMINAMATH_CALUDE_cos_six_arccos_one_fourth_l415_41548


namespace NUMINAMATH_CALUDE_custom_operation_equation_solution_l415_41566

-- Define the custom operation
def star (a b : ℝ) : ℝ := 4 * a * b

-- Theorem statement
theorem custom_operation_equation_solution :
  ∀ x : ℝ, star x x + 2 * (star 1 x) - star 2 2 = 0 → x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equation_solution_l415_41566


namespace NUMINAMATH_CALUDE_classroom_count_l415_41538

theorem classroom_count (girls boys : ℕ) (h1 : girls * 4 = boys * 3) (h2 : boys = 28) : 
  girls + boys = 49 := by
sorry

end NUMINAMATH_CALUDE_classroom_count_l415_41538


namespace NUMINAMATH_CALUDE_total_boxes_is_6200_l415_41533

/-- The number of boxes in Warehouse D -/
def warehouse_d : ℕ := 800

/-- The number of boxes in Warehouse C -/
def warehouse_c : ℕ := warehouse_d - 200

/-- The number of boxes in Warehouse B -/
def warehouse_b : ℕ := 2 * warehouse_c

/-- The number of boxes in Warehouse A -/
def warehouse_a : ℕ := 3 * warehouse_b

/-- The total number of boxes in all four warehouses -/
def total_boxes : ℕ := warehouse_a + warehouse_b + warehouse_c + warehouse_d

theorem total_boxes_is_6200 : total_boxes = 6200 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_is_6200_l415_41533


namespace NUMINAMATH_CALUDE_expression_value_l415_41575

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  (x^5 + 3*y^3) / 9 = 141 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l415_41575


namespace NUMINAMATH_CALUDE_circles_intersect_l415_41535

/-- Circle C₁ with equation x² + y² + 2x + 2y - 2 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 2 = 0

/-- Circle C₂ with equation x² + y² - 4x - 2y + 1 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The circles C₁ and C₂ are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l415_41535


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_9_l415_41550

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem smallest_perfect_square_divisible_by_4_and_9 :
  ∀ n : ℕ, n > 0 → is_perfect_square n → n % 4 = 0 → n % 9 = 0 → n ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_9_l415_41550


namespace NUMINAMATH_CALUDE_marble_collection_total_l415_41554

theorem marble_collection_total (r : ℝ) (b : ℝ) (g : ℝ) : 
  r > 0 → 
  r = 1.3 * b → 
  g = 1.5 * r → 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ abs ((r + b + g) / r - 3.27) < ε :=
sorry

end NUMINAMATH_CALUDE_marble_collection_total_l415_41554


namespace NUMINAMATH_CALUDE_construction_cost_equation_l415_41577

/-- The cost of land per square meter that satisfies the construction cost equation -/
def land_cost_per_sqm : ℝ := 50

/-- The cost of bricks per 1000 bricks -/
def brick_cost_per_1000 : ℝ := 100

/-- The cost of roof tiles per tile -/
def roof_tile_cost : ℝ := 10

/-- The required land area in square meters -/
def required_land_area : ℝ := 2000

/-- The required number of bricks -/
def required_bricks : ℝ := 10000

/-- The required number of roof tiles -/
def required_roof_tiles : ℝ := 500

/-- The total construction cost -/
def total_construction_cost : ℝ := 106000

theorem construction_cost_equation :
  land_cost_per_sqm * required_land_area +
  brick_cost_per_1000 * (required_bricks / 1000) +
  roof_tile_cost * required_roof_tiles =
  total_construction_cost :=
sorry

end NUMINAMATH_CALUDE_construction_cost_equation_l415_41577


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l415_41591

/-- Calculate the area of a triangle given its vertices' coordinates -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The coordinates of points X, Y, and Z -/
def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

/-- The ratio of areas between triangles XYZ and ABC -/
def areaRatio : ℝ := 0.1111111111111111

theorem area_of_triangle_ABC : 
  ∃ (A B C : ℝ × ℝ), 
    triangleArea X.1 X.2 Y.1 Y.2 Z.1 Z.2 / triangleArea A.1 A.2 B.1 B.2 C.1 C.2 = areaRatio ∧ 
    triangleArea A.1 A.2 B.1 B.2 C.1 C.2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l415_41591


namespace NUMINAMATH_CALUDE_fraction_cube_equality_l415_41532

theorem fraction_cube_equality : (45000 ^ 3) / (15000 ^ 3) = 27 := by sorry

end NUMINAMATH_CALUDE_fraction_cube_equality_l415_41532


namespace NUMINAMATH_CALUDE_willie_cream_purchase_l415_41509

/-- The amount of cream Willie needs to buy given the total required amount and the amount he already has. -/
def cream_to_buy (total_required : ℕ) (available : ℕ) : ℕ :=
  total_required - available

/-- Theorem stating that Willie needs to buy 151 lbs. of cream. -/
theorem willie_cream_purchase : cream_to_buy 300 149 = 151 := by
  sorry

end NUMINAMATH_CALUDE_willie_cream_purchase_l415_41509


namespace NUMINAMATH_CALUDE_train_length_calculation_l415_41518

/-- Represents a train with its length and the time it takes to cross two platforms -/
structure Train where
  length : ℝ
  time_platform1 : ℝ
  time_platform2 : ℝ

/-- The length of the first platform in meters -/
def platform1_length : ℝ := 120

/-- The length of the second platform in meters -/
def platform2_length : ℝ := 250

/-- Theorem stating that a train crossing two platforms of given lengths in specific times has a specific length -/
theorem train_length_calculation (t : Train) 
  (h1 : t.time_platform1 = 15) 
  (h2 : t.time_platform2 = 20) : 
  t.length = 270 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l415_41518


namespace NUMINAMATH_CALUDE_simplify_sum_of_fractions_l415_41515

theorem simplify_sum_of_fractions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hsum : x + y + z = 3) :
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2) =
  1 / (9 - 2*y*z) + 1 / (9 - 2*x*z) + 1 / (9 - 2*x*y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sum_of_fractions_l415_41515


namespace NUMINAMATH_CALUDE_january_book_sales_l415_41557

/-- Proves that the number of books sold in January is 15, given the sales in February and March,
    and the average sales across all three months. -/
theorem january_book_sales (february_sales march_sales : ℕ) (average_sales : ℚ)
  (h1 : february_sales = 16)
  (h2 : march_sales = 17)
  (h3 : average_sales = 16)
  (h4 : (january_sales + february_sales + march_sales : ℚ) / 3 = average_sales) :
  january_sales = 15 := by
  sorry

end NUMINAMATH_CALUDE_january_book_sales_l415_41557


namespace NUMINAMATH_CALUDE_square_of_negative_double_product_l415_41563

theorem square_of_negative_double_product (x y : ℝ) : (-2 * x * y)^2 = 4 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_double_product_l415_41563


namespace NUMINAMATH_CALUDE_approximately_200_men_joined_l415_41545

-- Define the initial number of men
def initial_men : ℕ := 1000

-- Define the initial duration of provisions in days
def initial_duration : ℚ := 20

-- Define the new duration of provisions in days
def new_duration : ℚ := 167/10  -- 16.67 as a rational number

-- Define a function to calculate the number of men who joined
def men_joined : ℚ := 
  (initial_men * initial_duration / new_duration) - initial_men

-- Theorem statement
theorem approximately_200_men_joined : 
  199 ≤ men_joined ∧ men_joined < 201 := by
  sorry


end NUMINAMATH_CALUDE_approximately_200_men_joined_l415_41545


namespace NUMINAMATH_CALUDE_ellipse_tangent_perpendicular_l415_41581

/-- Two ellipses with equations x²/a² + y²/b² = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

def is_on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def is_tangent_to_ellipse (l : Line) (e : Ellipse) (p : Point) : Prop :=
  is_on_ellipse p e ∧ l.m = -p.x * e.b^2 / (p.y * e.a^2)

def intersect_line_ellipse (l : Line) (e : Ellipse) : Set Point :=
  {p : Point | is_on_ellipse p e ∧ p.y = l.m * p.x + l.c}

def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.m * l2.m = -1

theorem ellipse_tangent_perpendicular 
  (e1 e2 : Ellipse) 
  (p : Point) 
  (l : Line) 
  (a b q : Point) :
  e1.a^2 - e1.b^2 = e2.a^2 - e2.b^2 →  -- shared foci condition
  is_tangent_to_ellipse l e1 p →
  a ∈ intersect_line_ellipse l e2 →
  b ∈ intersect_line_ellipse l e2 →
  is_tangent_to_ellipse (Line.mk ((q.y - a.y) / (q.x - a.x)) (q.y - (q.y - a.y) / (q.x - a.x) * q.x)) e2 a →
  is_tangent_to_ellipse (Line.mk ((q.y - b.y) / (q.x - b.x)) (q.y - (q.y - b.y) / (q.x - b.x) * q.x)) e2 b →
  are_perpendicular 
    (Line.mk ((q.y - p.y) / (q.x - p.x)) (q.y - (q.y - p.y) / (q.x - p.x) * q.x))
    (Line.mk ((b.y - a.y) / (b.x - a.x)) (b.y - (b.y - a.y) / (b.x - a.x) * b.x)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_perpendicular_l415_41581


namespace NUMINAMATH_CALUDE_remainder_2015_div_28_l415_41565

theorem remainder_2015_div_28 : 2015 % 28 = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2015_div_28_l415_41565


namespace NUMINAMATH_CALUDE_t_leq_s_l415_41578

theorem t_leq_s (a b t s : ℝ) (ht : t = a + 2*b) (hs : s = a + b^2 + 1) : t ≤ s := by
  sorry

end NUMINAMATH_CALUDE_t_leq_s_l415_41578


namespace NUMINAMATH_CALUDE_equation_solution_l415_41555

theorem equation_solution (x y : ℝ) : 
  y = 3 * x + 1 →
  4 * y^2 + 2 * y + 5 = 3 * (8 * x^2 + 2 * y + 3) →
  x = (-3 + Real.sqrt 21) / 6 ∨ x = (-3 - Real.sqrt 21) / 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l415_41555


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l415_41561

theorem triangle_angle_problem (a b c : ℝ) (A : ℝ) :
  b = c →
  a^2 = 2 * b^2 * (1 - Real.sin A) →
  A = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l415_41561


namespace NUMINAMATH_CALUDE_log_weight_when_cut_l415_41588

/-- Given a log of length 20 feet that weighs 150 pounds per linear foot,
    prove that when cut in half, each piece weighs 1500 pounds. -/
theorem log_weight_when_cut (log_length : ℝ) (weight_per_foot : ℝ) :
  log_length = 20 →
  weight_per_foot = 150 →
  (log_length / 2) * weight_per_foot = 1500 := by
  sorry

end NUMINAMATH_CALUDE_log_weight_when_cut_l415_41588


namespace NUMINAMATH_CALUDE_product_equals_root_fraction_l415_41508

theorem product_equals_root_fraction (a b c : ℝ) :
  a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1) →
  6 * 15 * 7 = (3 / 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_product_equals_root_fraction_l415_41508


namespace NUMINAMATH_CALUDE_parallelogram_area_l415_41556

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 inches and 20 inches is 100√3 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) : 
  a = 10 → b = 20 → θ = 150 * π / 180 → 
  a * b * Real.sin (π - θ) = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l415_41556


namespace NUMINAMATH_CALUDE_ratio_of_sums_l415_41503

theorem ratio_of_sums (p q r u v w : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
  (pos_u : 0 < u) (pos_v : 0 < v) (pos_w : 0 < w)
  (sum_squares_pqr : p^2 + q^2 + r^2 = 49)
  (sum_squares_uvw : u^2 + v^2 + w^2 = 64)
  (sum_products : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l415_41503


namespace NUMINAMATH_CALUDE_meeting_probability_l415_41574

/-- Xiaocong's arrival time at Wuhan Station -/
def xiaocong_arrival : ℝ := 13.5

/-- Duration of Xiaocong's rest at Wuhan Station -/
def xiaocong_rest : ℝ := 1

/-- Earliest possible arrival time for Xiaoming -/
def xiaoming_earliest : ℝ := 14

/-- Latest possible arrival time for Xiaoming -/
def xiaoming_latest : ℝ := 15

/-- Xiaoming's train departure time -/
def xiaoming_departure : ℝ := 15.5

/-- The probability of Xiaocong and Xiaoming meeting at Wuhan Station -/
theorem meeting_probability : ℝ := by sorry

end NUMINAMATH_CALUDE_meeting_probability_l415_41574


namespace NUMINAMATH_CALUDE_lacy_correct_percentage_l415_41570

theorem lacy_correct_percentage (x : ℝ) (h : x > 0) :
  let total_problems := 6 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  (correct_problems / total_problems) * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_lacy_correct_percentage_l415_41570


namespace NUMINAMATH_CALUDE_function_inequality_l415_41522

/-- Given a function f: ℝ → ℝ satisfying certain conditions, 
    prove that the set of x where f(x) > 1/e^x is (ln 3, +∞) -/
theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, (deriv f) x > -f x) 
  (h2 : f (Real.log 3) = 1/3) :
  {x : ℝ | f x > Real.exp (-x)} = Set.Ioi (Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l415_41522


namespace NUMINAMATH_CALUDE_unique_number_l415_41585

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  Odd n ∧ 
  n % 9 = 0 ∧ 
  is_perfect_square (digit_product n) :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l415_41585


namespace NUMINAMATH_CALUDE_square_sum_sqrt_difference_and_sum_l415_41594

theorem square_sum_sqrt_difference_and_sum (x₁ x₂ : ℝ) :
  x₁ = Real.sqrt 3 - Real.sqrt 2 →
  x₂ = Real.sqrt 3 + Real.sqrt 2 →
  x₁^2 + x₂^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_square_sum_sqrt_difference_and_sum_l415_41594


namespace NUMINAMATH_CALUDE_beach_relaxation_l415_41582

/-- The number of people left relaxing on the beach -/
def people_left_relaxing (row1_initial : ℕ) (row1_left : ℕ) (row2_initial : ℕ) (row2_left : ℕ) (row3 : ℕ) : ℕ :=
  (row1_initial - row1_left) + (row2_initial - row2_left) + row3

/-- Theorem stating the number of people left relaxing on the beach -/
theorem beach_relaxation : 
  people_left_relaxing 24 3 20 5 18 = 54 := by
  sorry

end NUMINAMATH_CALUDE_beach_relaxation_l415_41582


namespace NUMINAMATH_CALUDE_circle_radius_doubled_l415_41540

theorem circle_radius_doubled (r n : ℝ) : 
  (2 * π * (r + n) = 2 * (2 * π * r)) → r = n :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_doubled_l415_41540


namespace NUMINAMATH_CALUDE_art_club_committee_probability_l415_41560

def art_club_size : ℕ := 24
def boys_count : ℕ := 12
def girls_count : ℕ := 12
def committee_size : ℕ := 5

theorem art_club_committee_probability :
  let total_combinations := Nat.choose art_club_size committee_size
  let all_boys_or_all_girls := 2 * Nat.choose boys_count committee_size
  (total_combinations - all_boys_or_all_girls : ℚ) / total_combinations = 3427 / 3542 := by
  sorry

end NUMINAMATH_CALUDE_art_club_committee_probability_l415_41560
