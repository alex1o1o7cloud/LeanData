import Mathlib

namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1181_118194

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 
    (Real.rpow (1 - 2 * x^3 * Real.sin (5 / x)) (1/3)) - 1 + x
  else 
    0

theorem derivative_f_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1181_118194


namespace NUMINAMATH_CALUDE_decimal_units_count_l1181_118115

theorem decimal_units_count :
  (∃ n : ℕ, n * (1 / 10 : ℚ) = (19 / 10 : ℚ) ∧ n = 19) ∧
  (∃ m : ℕ, m * (1 / 100 : ℚ) = (8 / 10 : ℚ) ∧ m = 80) :=
by sorry

end NUMINAMATH_CALUDE_decimal_units_count_l1181_118115


namespace NUMINAMATH_CALUDE_binary_1101001_is_105_and_odd_l1181_118199

-- Define the binary number as a list of bits
def binary_number : List Nat := [1, 1, 0, 1, 0, 0, 1]

-- Function to convert binary to decimal
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Theorem statement
theorem binary_1101001_is_105_and_odd :
  (binary_to_decimal binary_number = 105) ∧ (105 % 2 = 1) := by
  sorry

#eval binary_to_decimal binary_number
#eval 105 % 2

end NUMINAMATH_CALUDE_binary_1101001_is_105_and_odd_l1181_118199


namespace NUMINAMATH_CALUDE_green_balls_count_l1181_118176

theorem green_balls_count (total : ℕ) (blue : ℕ) : 
  total = 40 → 
  blue = 11 → 
  ∃ (red green : ℕ), 
    red = 2 * blue ∧ 
    green = total - (red + blue) ∧ 
    green = 7 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l1181_118176


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1181_118166

theorem no_integer_solutions (n : ℤ) : ¬ ∃ x : ℤ, x^2 - 16*n*x + 7^5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1181_118166


namespace NUMINAMATH_CALUDE_smallest_six_consecutive_number_max_six_consecutive_with_perfect_square_F_l1181_118191

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a FourDigitNumber has distinct non-zero digits -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  0 < a ∧ a ≤ 9 ∧
  0 < b ∧ b ≤ 9 ∧
  0 < c ∧ c ≤ 9 ∧
  0 < d ∧ d ≤ 9 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Checks if a FourDigitNumber is a "six-consecutive number" -/
def isSixConsecutive (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  (a + b) * (c + d) = 60

/-- Calculates F(M) for a FourDigitNumber -/
def F (n : FourDigitNumber) : Int :=
  let (a, b, c, d) := n
  (a * 10 + d) - (b * 10 + c) - ((a * 10 + c) - (b * 10 + d))

/-- Converts a FourDigitNumber to its integer representation -/
def toInt (n : FourDigitNumber) : Nat :=
  let (a, b, c, d) := n
  a * 1000 + b * 100 + c * 10 + d

theorem smallest_six_consecutive_number :
  ∃ (M : FourDigitNumber),
    isValidFourDigitNumber M ∧
    isSixConsecutive M ∧
    (∀ (N : FourDigitNumber),
      isValidFourDigitNumber N → isSixConsecutive N →
      toInt M ≤ toInt N) ∧
    toInt M = 1369 := by sorry

theorem max_six_consecutive_with_perfect_square_F :
  ∃ (N : FourDigitNumber),
    isValidFourDigitNumber N ∧
    isSixConsecutive N ∧
    (∃ (k : Nat), F N = k * k) ∧
    (∀ (M : FourDigitNumber),
      isValidFourDigitNumber M → isSixConsecutive M →
      (∃ (j : Nat), F M = j * j) →
      toInt M ≤ toInt N) ∧
    toInt N = 9613 := by sorry

end NUMINAMATH_CALUDE_smallest_six_consecutive_number_max_six_consecutive_with_perfect_square_F_l1181_118191


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1181_118167

theorem smallest_dual_base_representation : ∃ (c d : ℕ), 
  c > 3 ∧ d > 3 ∧ 
  3 * c + 4 = 19 ∧ 
  4 * d + 3 = 19 ∧
  (∀ (x c' d' : ℕ), c' > 3 → d' > 3 → 3 * c' + 4 = x → 4 * d' + 3 = x → x ≥ 19) := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1181_118167


namespace NUMINAMATH_CALUDE_new_average_age_l1181_118174

theorem new_average_age (n : ℕ) (original_avg : ℚ) (new_person_age : ℕ) : 
  n = 8 → original_avg = 14 → new_person_age = 32 → 
  (n * original_avg + new_person_age : ℚ) / (n + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l1181_118174


namespace NUMINAMATH_CALUDE_max_distance_between_sine_cosine_curves_l1181_118155

theorem max_distance_between_sine_cosine_curves : 
  ∃ (C : ℝ), C = (Real.sqrt 3 / 2) * Real.sqrt 2 ∧ 
  ∀ (x : ℝ), |Real.sin (x + π/6) - 2 * Real.cos x| ≤ C ∧
  ∃ (a : ℝ), |Real.sin (a + π/6) - 2 * Real.cos a| = C :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_sine_cosine_curves_l1181_118155


namespace NUMINAMATH_CALUDE_system_a_l1181_118165

theorem system_a (x : Fin 100 → ℝ) 
  (h : ∀ i : Fin 100, x i + x ((i + 1) % 100) + x ((i + 2) % 100) = 0) :
  ∀ i : Fin 100, x i = 0 := by
sorry

end NUMINAMATH_CALUDE_system_a_l1181_118165


namespace NUMINAMATH_CALUDE_wire_cutting_l1181_118163

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 70 →
  ratio = 3 / 7 →
  shorter_length + (shorter_length / ratio) = total_length →
  shorter_length = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l1181_118163


namespace NUMINAMATH_CALUDE_locus_of_M_constant_ratio_l1181_118193

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define a point on the ellipse
variable (P : ℝ × ℝ)
axiom P_on_ellipse : ellipse P.1 P.2

-- Define point M
def M (P : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define point N
def N (P : ℝ × ℝ) : ℝ × ℝ := sorry

theorem locus_of_M (P : ℝ × ℝ) (h : ellipse P.1 P.2) : 
  (M P).1 = -8 := by sorry

theorem constant_ratio (P : ℝ × ℝ) (h : ellipse P.1 P.2) :
  ‖N P - F₁‖ / ‖M P - F₁‖ = 1/2 := by sorry

end NUMINAMATH_CALUDE_locus_of_M_constant_ratio_l1181_118193


namespace NUMINAMATH_CALUDE_acme_profit_calculation_l1181_118102

def initial_outlay : ℝ := 12450
def manufacturing_cost_per_set : ℝ := 20.75
def selling_price_per_set : ℝ := 50
def marketing_expense_rate : ℝ := 0.05
def shipping_cost_rate : ℝ := 0.03
def number_of_sets : ℕ := 950

def revenue : ℝ := selling_price_per_set * number_of_sets
def total_manufacturing_cost : ℝ := initial_outlay + manufacturing_cost_per_set * number_of_sets
def additional_variable_costs : ℝ := (marketing_expense_rate + shipping_cost_rate) * revenue

def profit : ℝ := revenue - total_manufacturing_cost - additional_variable_costs

theorem acme_profit_calculation : profit = 11537.50 := by
  sorry

end NUMINAMATH_CALUDE_acme_profit_calculation_l1181_118102


namespace NUMINAMATH_CALUDE_fidos_yard_area_fraction_l1181_118161

theorem fidos_yard_area_fraction :
  let square_side : ℝ := 2  -- Arbitrary side length
  let circle_radius : ℝ := 1  -- Half of the square side
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  circle_area / square_area = π :=
by sorry

end NUMINAMATH_CALUDE_fidos_yard_area_fraction_l1181_118161


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l1181_118110

/-- Calculates the total compensation for a bus driver based on hours worked and pay rates -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_percentage : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 16)
  (h2 : regular_hours = 40)
  (h3 : overtime_percentage = 0.75)
  (h4 : total_hours = 50) :
  let overtime_rate := regular_rate * (1 + overtime_percentage)
  let overtime_hours := total_hours - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  regular_pay + overtime_pay = 920 := by
sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l1181_118110


namespace NUMINAMATH_CALUDE_erik_pie_amount_l1181_118192

theorem erik_pie_amount (frank_pie : ℝ) (erik_extra : ℝ) 
  (h1 : frank_pie = 0.3333333333333333)
  (h2 : erik_extra = 0.3333333333333333) :
  frank_pie + erik_extra = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_erik_pie_amount_l1181_118192


namespace NUMINAMATH_CALUDE_upper_limit_of_set_W_l1181_118157

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def set_W (upper_bound : ℕ) : Set ℕ :=
  {n : ℕ | n > 10 ∧ n ≤ upper_bound ∧ is_prime n}

theorem upper_limit_of_set_W (upper_bound : ℕ) :
  (∃ (w : Set ℕ), w = set_W upper_bound ∧ 
   (∃ (max min : ℕ), max ∈ w ∧ min ∈ w ∧ 
    (∀ x ∈ w, x ≤ max ∧ x ≥ min) ∧ max - min = 12)) →
  upper_bound = 23 :=
sorry

end NUMINAMATH_CALUDE_upper_limit_of_set_W_l1181_118157


namespace NUMINAMATH_CALUDE_min_value_fraction_l1181_118169

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x - y + 2*z = 0) : 
  ∃ (m : ℝ), m = 8 ∧ ∀ k, k = y^2/(x*z) → k ≥ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1181_118169


namespace NUMINAMATH_CALUDE_circle_radius_zero_l1181_118181

/-- The radius of a circle given by the equation 4x^2 + 8x + 4y^2 - 16y + 20 = 0 is 0 -/
theorem circle_radius_zero (x y : ℝ) : 
  4*x^2 + 8*x + 4*y^2 - 16*y + 20 = 0 → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l1181_118181


namespace NUMINAMATH_CALUDE_sandy_work_hours_l1181_118117

/-- Sandy's work problem -/
theorem sandy_work_hours (hourly_rate : ℚ) (friday_hours : ℚ) (saturday_hours : ℚ) (total_earnings : ℚ) :
  hourly_rate = 15 →
  friday_hours = 10 →
  saturday_hours = 6 →
  total_earnings = 450 →
  (total_earnings - (friday_hours + saturday_hours) * hourly_rate) / hourly_rate = 14 :=
by sorry

end NUMINAMATH_CALUDE_sandy_work_hours_l1181_118117


namespace NUMINAMATH_CALUDE_constant_b_proof_l1181_118121

theorem constant_b_proof (a b c : ℝ) : 
  (∀ x : ℝ, (3 * x^2 - 2 * x + 4) * (a * x^2 + b * x + c) = 
    6 * x^4 - 5 * x^3 + 11 * x^2 - 8 * x + 16) → 
  b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_constant_b_proof_l1181_118121


namespace NUMINAMATH_CALUDE_total_jelly_beans_l1181_118106

/-- The number of jelly beans needed to fill a large drinking glass -/
def large_glass_beans : ℕ := 50

/-- The number of jelly beans needed to fill a small drinking glass -/
def small_glass_beans : ℕ := large_glass_beans / 2

/-- The number of large drinking glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small drinking glasses -/
def num_small_glasses : ℕ := 3

/-- Theorem stating the total number of jelly beans needed to fill all glasses -/
theorem total_jelly_beans :
  num_large_glasses * large_glass_beans + num_small_glasses * small_glass_beans = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_jelly_beans_l1181_118106


namespace NUMINAMATH_CALUDE_unique_solution_l1181_118118

-- Define the equation
def equation (x : ℝ) : Prop :=
  2021 * x = 2022 * (x^2021)^(1/2021) - 1

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, x ≥ 0 ∧ equation x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1181_118118


namespace NUMINAMATH_CALUDE_olympic_mascot_problem_l1181_118109

theorem olympic_mascot_problem (total_items wholesale_cost : ℕ) 
  (wholesale_price_A wholesale_price_B : ℕ) 
  (retail_price_A retail_price_B : ℕ) (min_profit : ℕ) :
  total_items = 100 ∧ 
  wholesale_cost = 5650 ∧
  wholesale_price_A = 60 ∧ 
  wholesale_price_B = 50 ∧
  retail_price_A = 80 ∧
  retail_price_B = 60 ∧
  min_profit = 1400 →
  (∃ (num_A num_B : ℕ),
    num_A + num_B = total_items ∧
    num_A * wholesale_price_A + num_B * wholesale_price_B = wholesale_cost ∧
    num_A = 65 ∧ num_B = 35) ∧
  (∃ (min_A : ℕ),
    min_A ≥ 40 ∧
    ∀ (num_A : ℕ),
      num_A ≥ min_A →
      (num_A * (retail_price_A - wholesale_price_A) + 
       (total_items - num_A) * (retail_price_B - wholesale_price_B)) ≥ min_profit) :=
by sorry

end NUMINAMATH_CALUDE_olympic_mascot_problem_l1181_118109


namespace NUMINAMATH_CALUDE_chernomor_salary_manipulation_l1181_118105

/-- Represents a salary proposal for a single month -/
structure SalaryProposal where
  warrior_salaries : Fin 33 → ℝ
  chernomor_salary : ℝ

/-- The voting function: returns true if the majority of warriors vote in favor -/
def majority_vote (current : SalaryProposal) (proposal : SalaryProposal) : Prop :=
  (Finset.filter (fun i => proposal.warrior_salaries i > current.warrior_salaries i) Finset.univ).card > 16

/-- The theorem stating that Chernomor can achieve his goal -/
theorem chernomor_salary_manipulation :
  ∃ (initial : SalaryProposal) (proposals : Fin 36 → SalaryProposal),
    (∀ i : Fin 35, majority_vote (proposals i) (proposals (i + 1))) ∧
    (proposals 35).chernomor_salary = 10 * initial.chernomor_salary ∧
    (∀ j : Fin 33, (proposals 35).warrior_salaries j ≤ initial.warrior_salaries j / 10) :=
sorry

end NUMINAMATH_CALUDE_chernomor_salary_manipulation_l1181_118105


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l1181_118139

theorem student_multiplication_problem (x y : ℝ) : 
  x = 127 → x * y - 152 = 102 → y = 2 := by
sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l1181_118139


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l1181_118148

theorem same_solution_implies_c_value (x c : ℝ) : 
  (3 * x + 8 = 5) ∧ (c * x - 15 = -3) → c = -12 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l1181_118148


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l1181_118133

theorem reciprocal_of_sum : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l1181_118133


namespace NUMINAMATH_CALUDE_kindergarten_attendance_l1181_118129

/-- Calculates the total number of students present in two kindergarten sessions -/
def total_students (morning_registered : Nat) (morning_absent : Nat) 
                   (afternoon_registered : Nat) (afternoon_absent : Nat) : Nat :=
  (morning_registered - morning_absent) + (afternoon_registered - afternoon_absent)

/-- Theorem: The total number of students present over two kindergarten sessions is 42 -/
theorem kindergarten_attendance : 
  total_students 25 3 24 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_attendance_l1181_118129


namespace NUMINAMATH_CALUDE_inequality_proof_l1181_118183

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + b * c) / a + (1 + c * a) / b + (1 + a * b) / c > 
  Real.sqrt (a^2 + 2) + Real.sqrt (b^2 + 2) + Real.sqrt (c^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1181_118183


namespace NUMINAMATH_CALUDE_fiona_reaches_food_l1181_118119

/-- Represents a lily pad --/
structure LilyPad :=
  (number : ℕ)

/-- Represents Fiona the frog --/
structure Frog :=
  (position : LilyPad)

/-- Represents the probability of a jump --/
def JumpProbability : ℚ := 1/3

/-- The total number of lily pads --/
def TotalPads : ℕ := 16

/-- The position of the first predator --/
def Predator1 : LilyPad := ⟨4⟩

/-- The position of the second predator --/
def Predator2 : LilyPad := ⟨9⟩

/-- The position of the food --/
def FoodPosition : LilyPad := ⟨14⟩

/-- Fiona's starting position --/
def StartPosition : LilyPad := ⟨0⟩

/-- Function to calculate the probability of Fiona reaching the food --/
noncomputable def probabilityToReachFood (f : Frog) : ℚ :=
  sorry

theorem fiona_reaches_food :
  probabilityToReachFood ⟨StartPosition⟩ = 52/59049 :=
sorry

end NUMINAMATH_CALUDE_fiona_reaches_food_l1181_118119


namespace NUMINAMATH_CALUDE_seashells_given_theorem_l1181_118116

/-- The number of seashells Sam gave to Joan -/
def seashells_given_to_joan (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Theorem stating that the number of seashells Sam gave to Joan
    is the difference between his initial and current number of seashells -/
theorem seashells_given_theorem (initial_seashells current_seashells : ℕ) 
  (h : initial_seashells ≥ current_seashells) :
  seashells_given_to_joan initial_seashells current_seashells = 
  initial_seashells - current_seashells :=
by
  sorry

#eval seashells_given_to_joan 35 17  -- Should output 18

end NUMINAMATH_CALUDE_seashells_given_theorem_l1181_118116


namespace NUMINAMATH_CALUDE_shooting_probability_l1181_118150

/-- The probability of hitting a shot -/
def shooting_accuracy : ℚ := 9/10

/-- The probability of hitting two consecutive shots -/
def two_consecutive_hits : ℚ := 1/2

/-- The probability of hitting the next shot given that the first shot was hit -/
def next_shot_probability : ℚ := 5/9

theorem shooting_probability :
  shooting_accuracy = 9/10 →
  two_consecutive_hits = 1/2 →
  next_shot_probability = two_consecutive_hits / shooting_accuracy :=
by sorry

end NUMINAMATH_CALUDE_shooting_probability_l1181_118150


namespace NUMINAMATH_CALUDE_bamboo_sections_volume_l1181_118113

theorem bamboo_sections_volume (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →
  a 1 + a 2 + a 3 = 3.9 →
  a 6 + a 7 + a 8 + a 9 = 3 →
  a 4 + a 5 = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_bamboo_sections_volume_l1181_118113


namespace NUMINAMATH_CALUDE_average_of_tenths_and_thousandths_l1181_118180

theorem average_of_tenths_and_thousandths :
  let a : ℚ := 4/10  -- 4 tenths
  let b : ℚ := 5/1000  -- 5 thousandths
  (a + b) / 2 = 2025/10000 := by
sorry

end NUMINAMATH_CALUDE_average_of_tenths_and_thousandths_l1181_118180


namespace NUMINAMATH_CALUDE_gcd_960_1632_l1181_118130

theorem gcd_960_1632 : Nat.gcd 960 1632 = 96 := by
  sorry

end NUMINAMATH_CALUDE_gcd_960_1632_l1181_118130


namespace NUMINAMATH_CALUDE_proposition_p_or_q_is_true_l1181_118128

open Real

theorem proposition_p_or_q_is_true :
  (∀ x > 0, exp x > 1 + x) ∨
  (∀ f : ℝ → ℝ, (∀ x, f x + 2 = -(f (-x) + 2)) → 
   ∀ x, f (x - 0) + 0 = f (-(x - 0)) + 4) := by sorry

end NUMINAMATH_CALUDE_proposition_p_or_q_is_true_l1181_118128


namespace NUMINAMATH_CALUDE_xiao_ming_error_l1181_118171

theorem xiao_ming_error (x : ℝ) : 
  (x + 1) / 2 - 1 = (x - 2) / 3 → 
  3 * (x + 1) - 1 ≠ 2 * (x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_error_l1181_118171


namespace NUMINAMATH_CALUDE_geometry_propositions_l1181_118142

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- State the theorem
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (¬ ∀ (m n : Line) (α β : Plane), 
    parallel_line_plane m α → 
    parallel_line_plane n β → 
    parallel_plane_plane α β → 
    parallel_line_line m n) ∧ 
  (∀ (m n : Line) (α β : Plane), 
    perpendicular_line_plane m α → 
    perpendicular_line_plane n β → 
    perpendicular_plane_plane α β → 
    perpendicular_line_line m n) ∧ 
  (¬ ∀ (m n : Line) (α : Plane), 
    parallel_line_plane m α → 
    parallel_line_line m n → 
    parallel_line_plane n α) ∧ 
  (∀ (m n : Line) (α β : Plane), 
    parallel_plane_plane α β → 
    perpendicular_line_plane m α → 
    parallel_line_plane n β → 
    perpendicular_line_line m n) := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1181_118142


namespace NUMINAMATH_CALUDE_third_day_temperature_l1181_118177

/-- Given the average temperature of three days and the temperatures of two of those days,
    calculate the temperature of the third day. -/
theorem third_day_temperature
  (avg_temp : ℚ)
  (day1_temp : ℚ)
  (day3_temp : ℚ)
  (h1 : avg_temp = -7)
  (h2 : day1_temp = -14)
  (h3 : day3_temp = 1)
  : (3 * avg_temp - day1_temp - day3_temp : ℚ) = -8 := by
  sorry

end NUMINAMATH_CALUDE_third_day_temperature_l1181_118177


namespace NUMINAMATH_CALUDE_peters_erasers_l1181_118104

/-- Peter's erasers problem -/
theorem peters_erasers (initial_erasers additional_erasers : ℕ) :
  initial_erasers = 8 →
  additional_erasers = 3 →
  initial_erasers + additional_erasers = 11 := by
  sorry

end NUMINAMATH_CALUDE_peters_erasers_l1181_118104


namespace NUMINAMATH_CALUDE_f_derivative_at_negative_one_l1181_118145

noncomputable def f (x : ℝ) : ℝ := -x^3 + 1/x

theorem f_derivative_at_negative_one :
  (deriv f) (-1) = -4 :=
sorry

end NUMINAMATH_CALUDE_f_derivative_at_negative_one_l1181_118145


namespace NUMINAMATH_CALUDE_library_book_distribution_l1181_118101

theorem library_book_distribution (total_books : ℕ) 
  (day1_students day2_students day3_students day4_students : ℕ) : 
  total_books = 120 →
  day1_students = 4 →
  day2_students = 5 →
  day3_students = 6 →
  day4_students = 9 →
  total_books / (day1_students + day2_students + day3_students + day4_students) = 5 :=
by
  sorry

#check library_book_distribution

end NUMINAMATH_CALUDE_library_book_distribution_l1181_118101


namespace NUMINAMATH_CALUDE_empty_solution_set_l1181_118144

theorem empty_solution_set : ∀ x : ℝ, ¬(2 * x - x^2 > 5) := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_l1181_118144


namespace NUMINAMATH_CALUDE_log_10_2_bounds_l1181_118198

theorem log_10_2_bounds :
  let log_10 (x : ℝ) := Real.log x / Real.log 10
  10^3 = 1000 ∧ 10^4 = 10000 ∧ 2^9 = 512 ∧ 2^14 = 16384 →
  2/7 < log_10 2 ∧ log_10 2 < 1/3 := by sorry

end NUMINAMATH_CALUDE_log_10_2_bounds_l1181_118198


namespace NUMINAMATH_CALUDE_balls_after_2023_steps_l1181_118187

-- Define a function to convert a number to base-8
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

-- Define a function to sum the digits of a number in a given base
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

-- Theorem statement
theorem balls_after_2023_steps :
  sumDigits (toBase8 2023) = 21 := by
  sorry

end NUMINAMATH_CALUDE_balls_after_2023_steps_l1181_118187


namespace NUMINAMATH_CALUDE_pet_store_kittens_l1181_118151

/-- The number of kittens initially at the pet store -/
def initial_kittens : ℕ := 6

/-- The number of puppies initially at the pet store -/
def initial_puppies : ℕ := 7

/-- The number of puppies sold -/
def puppies_sold : ℕ := 2

/-- The number of kittens sold -/
def kittens_sold : ℕ := 3

/-- The number of pets remaining after the sale -/
def remaining_pets : ℕ := 8

theorem pet_store_kittens :
  initial_puppies - puppies_sold + (initial_kittens - kittens_sold) = remaining_pets :=
by sorry

end NUMINAMATH_CALUDE_pet_store_kittens_l1181_118151


namespace NUMINAMATH_CALUDE_mrs_hilt_reading_l1181_118164

theorem mrs_hilt_reading (num_books : ℕ) (chapters_per_book : ℕ) (total_chapters : ℕ) : 
  num_books = 4 → chapters_per_book = 17 → total_chapters = num_books * chapters_per_book → total_chapters = 68 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_reading_l1181_118164


namespace NUMINAMATH_CALUDE_wider_bolt_width_l1181_118160

theorem wider_bolt_width (a b : ℕ) (h1 : a = 45) (h2 : b > a) (h3 : Nat.gcd a b = 15) : 
  (∀ c : ℕ, c > a ∧ Nat.gcd a c = 15 → b ≤ c) → b = 60 := by
  sorry

end NUMINAMATH_CALUDE_wider_bolt_width_l1181_118160


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1181_118132

def U : Set ℕ := {x : ℕ | x > 0 ∧ (x - 6) * (x + 1) ≤ 0}

def A : Set ℕ := {1, 2, 4}

theorem complement_of_A_in_U :
  (U \ A) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1181_118132


namespace NUMINAMATH_CALUDE_one_hundred_twentieth_letter_l1181_118162

def letter_pattern (n : ℕ) : Char :=
  match n % 4 with
  | 0 => 'D'
  | 1 => 'A'
  | 2 => 'B'
  | 3 => 'C'
  | _ => 'D'  -- This case is unreachable, but Lean requires it for exhaustiveness

theorem one_hundred_twentieth_letter :
  letter_pattern 120 = 'D' := by
  sorry

end NUMINAMATH_CALUDE_one_hundred_twentieth_letter_l1181_118162


namespace NUMINAMATH_CALUDE_sin_30_degrees_l1181_118186

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l1181_118186


namespace NUMINAMATH_CALUDE_billys_age_l1181_118125

/-- Given that the sum of Billy's and Joe's ages is 60 and Billy is three times as old as Joe,
    prove that Billy is 45 years old. -/
theorem billys_age (billy joe : ℕ) 
    (sum_condition : billy + joe = 60)
    (age_ratio : billy = 3 * joe) : 
  billy = 45 := by
  sorry

end NUMINAMATH_CALUDE_billys_age_l1181_118125


namespace NUMINAMATH_CALUDE_carlas_chickens_l1181_118124

theorem carlas_chickens (initial_chickens : ℕ) : 
  (initial_chickens : ℝ) - 0.4 * initial_chickens + 10 * (0.4 * initial_chickens) = 1840 →
  initial_chickens = 400 := by
  sorry

end NUMINAMATH_CALUDE_carlas_chickens_l1181_118124


namespace NUMINAMATH_CALUDE_problem_statement_l1181_118123

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = ({0, a^2, a+b} : Set ℝ) → 
  a^2009 + b^2009 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1181_118123


namespace NUMINAMATH_CALUDE_adam_miles_l1181_118138

/-- Adam ran 25 miles more than Katie, and Katie ran 10 miles. -/
theorem adam_miles (katie_miles : ℕ) (adam_miles : ℕ) : 
  katie_miles = 10 → adam_miles = katie_miles + 25 → adam_miles = 35 := by
  sorry

end NUMINAMATH_CALUDE_adam_miles_l1181_118138


namespace NUMINAMATH_CALUDE_complete_solution_set_l1181_118153

def S : Set (ℕ × ℕ × ℕ) :=
  {(4, 33, 30), (32, 9, 30), (40, 9, 18), (12, 31, 30), (24, 23, 30), (4, 15, 22), (36, 15, 42)}

def is_solution (t : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := t
  a^2 + b^2 + c^2 = 2005 ∧ 0 < a ∧ a ≤ b ∧ b ≤ c

theorem complete_solution_set :
  ∀ (a b c : ℕ), is_solution (a, b, c) ↔ (a, b, c) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_complete_solution_set_l1181_118153


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1181_118147

/-- Represents the cost and profit scenario for cloth selling --/
structure ClothSelling where
  total_length : ℕ
  first_half : ℕ
  second_half : ℕ
  total_price : ℚ
  profit_first : ℚ
  profit_second : ℚ

/-- The theorem stating the cost price per meter if it's the same for both halves --/
theorem cloth_cost_price (cs : ClothSelling)
  (h_total : cs.total_length = 120)
  (h_half : cs.first_half = cs.second_half)
  (h_length : cs.first_half + cs.second_half = cs.total_length)
  (h_price : cs.total_price = 15360)
  (h_profit1 : cs.profit_first = 1/10)
  (h_profit2 : cs.profit_second = 1/5)
  (h_equal_cost : ∃ (c : ℚ), 
    cs.first_half * (1 + cs.profit_first) * c + 
    cs.second_half * (1 + cs.profit_second) * c = cs.total_price) :
  ∃ (c : ℚ), c = 11130 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l1181_118147


namespace NUMINAMATH_CALUDE_train_length_calculation_l1181_118120

/-- The length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 27) 
  (h2 : man_speed = 6) 
  (h3 : passing_time = 11.999040076793857) : 
  ∃ (length : ℝ), abs (length - 110) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l1181_118120


namespace NUMINAMATH_CALUDE_fraction_of_male_birds_l1181_118100

theorem fraction_of_male_birds
  (total : ℕ)
  (h1 : total > 0)
  (robins : ℕ)
  (bluejays : ℕ)
  (h2 : robins = (2 * total) / 5)
  (h3 : bluejays = total - robins)
  (female_robins : ℕ)
  (h4 : female_robins = robins / 3)
  (female_bluejays : ℕ)
  (h5 : female_bluejays = (2 * bluejays) / 3)
  (male_birds : ℕ)
  (h6 : male_birds = total - female_robins - female_bluejays) :
  male_birds = (7 * total) / 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_male_birds_l1181_118100


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_1_any_4m_plus_1_has_prime_factor_4k_plus_1_infinitely_many_primes_4k_plus_1_from_divisibility_l1181_118108

theorem infinitely_many_primes_4k_plus_1 :
  ∀ (S : Set Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 1) →
  (∀ n, ∃ p ∈ S, p > n) :=
by
  sorry

theorem any_4m_plus_1_has_prime_factor_4k_plus_1 :
  ∀ m : Nat, ∃ p : Nat, Nat.Prime p ∧ (∃ k : Nat, p = 4*k + 1) ∧ p ∣ (4*m + 1) :=
by
  sorry

theorem infinitely_many_primes_4k_plus_1_from_divisibility 
  (h : ∀ m : Nat, ∃ p : Nat, Nat.Prime p ∧ (∃ k : Nat, p = 4*k + 1) ∧ p ∣ (4*m + 1)) :
  ∀ (S : Set Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 1) →
  (∀ n, ∃ p ∈ S, p > n) :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_1_any_4m_plus_1_has_prime_factor_4k_plus_1_infinitely_many_primes_4k_plus_1_from_divisibility_l1181_118108


namespace NUMINAMATH_CALUDE_smallest_consecutive_product_seven_consecutive_product_seven_is_smallest_l1181_118146

theorem smallest_consecutive_product (n : ℕ) : n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 5040 → n ≥ 7 :=
by sorry

theorem seven_consecutive_product : 7 * 8 * 9 * 10 = 5040 :=
by sorry

theorem seven_is_smallest : ∃ (n : ℕ), n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 5040 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_product_seven_consecutive_product_seven_is_smallest_l1181_118146


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1181_118168

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 
    and asymptotes y = ±(√3/2)x is √7/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b/a = Real.sqrt 3 / 2) :
  let e := Real.sqrt (a^2 + b^2) / a
  e = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1181_118168


namespace NUMINAMATH_CALUDE_smallest_a_is_eight_l1181_118184

-- Define the polynomial function
def f (a x : ℤ) : ℤ := x^4 + a^2 + 2*a*x

-- Define what it means for a number to be composite
def is_composite (n : ℤ) : Prop := ∃ m k : ℤ, m > 1 ∧ k > 1 ∧ n = m * k

-- State the theorem
theorem smallest_a_is_eight :
  (∀ x : ℤ, is_composite (f 8 x)) ∧
  (∀ a : ℤ, 0 < a → a < 8 → ∃ x : ℤ, ¬ is_composite (f a x)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_eight_l1181_118184


namespace NUMINAMATH_CALUDE_triangles_in_regular_decagon_l1181_118189

def regular_decagon_vertices : ℕ := 10

theorem triangles_in_regular_decagon :
  (regular_decagon_vertices.choose 3) = 120 :=
by sorry

end NUMINAMATH_CALUDE_triangles_in_regular_decagon_l1181_118189


namespace NUMINAMATH_CALUDE_graph_horizontal_shift_l1181_118114

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define a point (x, y) on the original graph
variable (x y : ℝ)

-- Define the horizontal shift
def h : ℝ := 2

-- Theorem stating that y = f(x + 2) is equivalent to shifting the graph of y = f(x) 2 units left
theorem graph_horizontal_shift :
  y = f (x + h) ↔ y = f ((x + h) - h) :=
sorry

end NUMINAMATH_CALUDE_graph_horizontal_shift_l1181_118114


namespace NUMINAMATH_CALUDE_athlete_arrangement_and_allocation_l1181_118196

/-- The number of male athletes -/
def num_male_athletes : ℕ := 4

/-- The number of female athletes -/
def num_female_athletes : ℕ := 3

/-- The total number of athletes -/
def total_athletes : ℕ := num_male_athletes + num_female_athletes

/-- The number of ways to arrange the athletes with all female athletes together -/
def arrangement_count : ℕ := (Nat.factorial (num_male_athletes + 1)) * (Nat.factorial num_female_athletes)

/-- The number of ways to allocate male athletes to two venues -/
def allocation_count : ℕ := Nat.choose num_male_athletes 1 + Nat.choose num_male_athletes 2

theorem athlete_arrangement_and_allocation :
  arrangement_count = 720 ∧ allocation_count = 10 := by sorry


end NUMINAMATH_CALUDE_athlete_arrangement_and_allocation_l1181_118196


namespace NUMINAMATH_CALUDE_no_integer_root_l1181_118197

theorem no_integer_root (q : ℤ) : ¬ ∃ x : ℤ, x^2 + 7*x - 14*(q^2 + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_root_l1181_118197


namespace NUMINAMATH_CALUDE_second_triangle_base_l1181_118185

/-- Given two triangles where the second has double the area of the first,
    prove that the base of the second triangle is 20 cm. -/
theorem second_triangle_base
  (b1 : ℝ) (h1 : ℝ) (h2 : ℝ)
  (base1 : b1 = 15)
  (height1 : h1 = 12)
  (height2 : h2 = 18)
  (area_relation : b1 * h1 = h2 * (b1 * h1 / 30)) :
  h2 * (b1 * h1 / 30) / h2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_triangle_base_l1181_118185


namespace NUMINAMATH_CALUDE_one_third_minus_decimal_approx_l1181_118126

theorem one_third_minus_decimal_approx : 
  (1 : ℚ) / 3 - 33333333 / 100000000 = 1 / (3 * 100000000) := by sorry

end NUMINAMATH_CALUDE_one_third_minus_decimal_approx_l1181_118126


namespace NUMINAMATH_CALUDE_power_equation_solution_l1181_118149

theorem power_equation_solution (m : ℕ) : 2^m = 2 * 16^2 * 4^3 → m = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1181_118149


namespace NUMINAMATH_CALUDE_morning_routine_time_l1181_118188

def skincare_routine_time : ℕ := 2 + 3 + 3 + 4 + 1 + 3 + 2 + 5 + 2 + 2 + 1

def makeup_time : ℕ := 30

def hair_styling_time : ℕ := 20

theorem morning_routine_time :
  skincare_routine_time + makeup_time + hair_styling_time = 78 := by
  sorry

end NUMINAMATH_CALUDE_morning_routine_time_l1181_118188


namespace NUMINAMATH_CALUDE_sum_not_prime_l1181_118182

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b + c + d = x * y :=
sorry

end NUMINAMATH_CALUDE_sum_not_prime_l1181_118182


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1181_118152

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧
  ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1181_118152


namespace NUMINAMATH_CALUDE_geometric_identity_l1181_118154

theorem geometric_identity 
  (a b c p x : ℝ) 
  (h1 : a + b + c = 2 * p) 
  (h2 : x = (b^2 + c^2 - a^2) / (2 * c)) 
  (h3 : c ≠ 0) : 
  b^2 - x^2 = (4 / c^2) * (p * (p - a) * (p - b) * (p - c)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_identity_l1181_118154


namespace NUMINAMATH_CALUDE_angle_C_measure_l1181_118158

/-- Represents a hexagon CALCUL with specific angle properties -/
structure Hexagon where
  -- Angles of the hexagon
  A : ℝ
  C : ℝ
  L : ℝ
  U : ℝ
  -- Conditions
  angle_sum : A + C + L + U + L + C = 720
  C_eq_L_eq_U : C = L ∧ L = U
  A_eq_L_eq_C : A = L ∧ L = C
  A_L_supplementary : A + L = 180

/-- The measure of angle C in the hexagon CALCUL is 120° -/
theorem angle_C_measure (h : Hexagon) : h.C = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l1181_118158


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l1181_118159

theorem geometric_series_second_term 
  (r : ℚ) 
  (sum : ℚ) 
  (h1 : r = 1 / 4) 
  (h2 : sum = 10) : 
  let a := sum * (1 - r)
  a * r = 15 / 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l1181_118159


namespace NUMINAMATH_CALUDE_octahedron_triangle_count_l1181_118143

/-- A regular octahedron -/
structure RegularOctahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  vertex_count : vertices.card = 6
  edge_count : edges.card = 12
  edge_validity : ∀ e ∈ edges, e.1 ≠ e.2 ∧ e.1 ∈ vertices ∧ e.2 ∈ vertices

/-- A triangle on the octahedron -/
structure OctahedronTriangle (O : RegularOctahedron) where
  vertices : Finset (Fin 6)
  vertex_count : vertices.card = 3
  vertex_validity : vertices ⊆ O.vertices
  edge_shared : ∃ e ∈ O.edges, (e.1 ∈ vertices ∧ e.2 ∈ vertices)

/-- The set of all valid triangles on the octahedron -/
def validTriangles (O : RegularOctahedron) : Set (OctahedronTriangle O) :=
  {t | t.vertices ⊆ O.vertices ∧ ∃ e ∈ O.edges, (e.1 ∈ t.vertices ∧ e.2 ∈ t.vertices)}

theorem octahedron_triangle_count (O : RegularOctahedron) :
  (validTriangles O).ncard = 12 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_triangle_count_l1181_118143


namespace NUMINAMATH_CALUDE_smallest_integer_negative_quadratic_l1181_118140

theorem smallest_integer_negative_quadratic :
  ∃ (n : ℤ), (∀ (m : ℤ), m^2 - 11*m + 28 < 0 → n ≤ m) ∧ (n^2 - 11*n + 28 < 0) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_negative_quadratic_l1181_118140


namespace NUMINAMATH_CALUDE_wang_elevator_problem_l1181_118127

def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.2

theorem wang_elevator_problem :
  (List.sum floor_movements = 0) ∧
  (List.sum (List.map (λ x => floor_height * electricity_per_meter * |x|) floor_movements) = 33.6) := by
  sorry

end NUMINAMATH_CALUDE_wang_elevator_problem_l1181_118127


namespace NUMINAMATH_CALUDE_sequence_periodicity_l1181_118141

def is_periodic (a : ℕ → ℕ) : Prop :=
  ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), a (n + p) = a n

theorem sequence_periodicity (a : ℕ → ℕ) 
  (h1 : ∀ n, a n < 1988)
  (h2 : ∀ m n, (a m + a n) % a (m + n) = 0) :
  is_periodic a := by
  sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l1181_118141


namespace NUMINAMATH_CALUDE_cubic_function_coefficients_l1181_118131

/-- Given a cubic function f(x) = ax³ - 4x² + bx - 3, 
    if f(1) = 3 and f(-2) = -47, then a = 4/3 and b = 26/3 -/
theorem cubic_function_coefficients (a b : ℚ) : 
  let f : ℚ → ℚ := λ x => a * x^3 - 4 * x^2 + b * x - 3
  (f 1 = 3 ∧ f (-2) = -47) → a = 4/3 ∧ b = 26/3 := by
  sorry


end NUMINAMATH_CALUDE_cubic_function_coefficients_l1181_118131


namespace NUMINAMATH_CALUDE_triangle_vector_properties_l1181_118190

/-- Given a triangle ABC with internal angles A, B, C, this theorem proves
    properties related to vectors m and n, and the side lengths of the triangle. -/
theorem triangle_vector_properties (A B C : Real) (m n : Real × Real) :
  let m : Real × Real := (2 * Real.sqrt 3, 1)
  let n : Real × Real := (Real.cos (A / 2) ^ 2, Real.sin A)
  C = 2 * Real.pi / 3 →
  ‖(1, 0) - (Real.cos A, Real.sin A)‖ = 3 →
  (A = Real.pi / 2 → ‖n‖ = Real.sqrt 5 / 2) ∧
  (∀ θ, m.1 * (Real.cos (θ / 2) ^ 2) + m.2 * Real.sin θ ≤ m.1 * (Real.cos (Real.pi / 12) ^ 2) + m.2 * Real.sin (Real.pi / 6)) ∧
  (‖(Real.cos (Real.pi / 6), Real.sin (Real.pi / 6)) - (Real.cos (5 * Real.pi / 6), Real.sin (5 * Real.pi / 6))‖ = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_vector_properties_l1181_118190


namespace NUMINAMATH_CALUDE_simplify_expression_l1181_118136

theorem simplify_expression (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ -1) :
  ((a^2 + 1) / a - 2) / ((a^2 - 1) / (a^2 + a)) = a - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1181_118136


namespace NUMINAMATH_CALUDE_lindas_savings_l1181_118137

theorem lindas_savings (savings : ℝ) : 
  savings > 0 →
  (0.9 * (3/8) * savings) + (0.85 * (1/4) * savings) + 450 = savings →
  savings = 1000 := by
sorry

end NUMINAMATH_CALUDE_lindas_savings_l1181_118137


namespace NUMINAMATH_CALUDE_hexagon_dimension_theorem_l1181_118170

/-- Represents a hexagon with dimension y -/
structure Hexagon :=
  (y : ℝ)

/-- Represents a rectangle with length and width -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)

/-- Represents a square with side length -/
structure Square :=
  (side : ℝ)

/-- The theorem stating that for an 8x18 rectangle cut into two congruent hexagons 
    that can be repositioned to form a square, the dimension y of the hexagon is 6 -/
theorem hexagon_dimension_theorem (rect : Rectangle) (hex1 hex2 : Hexagon) (sq : Square) :
  rect.length = 18 ∧ 
  rect.width = 8 ∧
  hex1 = hex2 ∧
  rect.length * rect.width = sq.side * sq.side →
  hex1.y = 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_dimension_theorem_l1181_118170


namespace NUMINAMATH_CALUDE_special_divisor_property_implies_prime_l1181_118156

theorem special_divisor_property_implies_prime (n : ℕ) (h1 : n > 1)
  (h2 : ∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) :
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_special_divisor_property_implies_prime_l1181_118156


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_l1181_118175

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum :
  unitsDigit (sumFactorials 15) = 3 :=
by
  sorry

/- Hint: You may want to use the following lemma -/
lemma units_digit_factorial_ge_5 (n : ℕ) (h : n ≥ 5) :
  unitsDigit (factorial n) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_l1181_118175


namespace NUMINAMATH_CALUDE_gordons_lighter_bag_weight_l1181_118122

/-- 
Given:
- Trace has 5 shopping bags
- Gordon has 2 shopping bags
- Trace's 5 bags weigh the same as Gordon's 2 bags
- One of Gordon's bags weighs 7 pounds
- Each of Trace's bags weighs 2 pounds

Prove that Gordon's lighter bag weighs 3 pounds.
-/
theorem gordons_lighter_bag_weight :
  ∀ (trace_bags gordon_bags : ℕ) 
    (trace_bag_weight gordon_heavy_bag_weight : ℝ)
    (total_trace_weight total_gordon_weight : ℝ),
  trace_bags = 5 →
  gordon_bags = 2 →
  trace_bag_weight = 2 →
  gordon_heavy_bag_weight = 7 →
  total_trace_weight = trace_bags * trace_bag_weight →
  total_gordon_weight = gordon_heavy_bag_weight + (total_trace_weight - gordon_heavy_bag_weight) →
  total_trace_weight = total_gordon_weight →
  (total_trace_weight - gordon_heavy_bag_weight) = 3 :=
by sorry

end NUMINAMATH_CALUDE_gordons_lighter_bag_weight_l1181_118122


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_8_l1181_118103

theorem largest_integer_less_than_100_with_remainder_5_mod_8 :
  ∃ (n : ℕ), n < 100 ∧ n % 8 = 5 ∧ ∀ (m : ℕ), m < 100 → m % 8 = 5 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_8_l1181_118103


namespace NUMINAMATH_CALUDE_polynomial_solution_l1181_118179

theorem polynomial_solution (a b c : ℤ) : 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (∀ X : ℤ, a^3 + a*a*X + b*X + c = a^3) ∧
  (∀ X : ℤ, b^3 + a*b*X + b*X + c = b^3) →
  a = 1 ∧ b = -1 ∧ c = -2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_solution_l1181_118179


namespace NUMINAMATH_CALUDE_almost_order_lineup_correct_almost_order_lineup_10_l1181_118135

/-- Represents the number of ways to line up n people in almost-order -/
def almost_order_lineup (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 3 => almost_order_lineup (n + 1) + almost_order_lineup (n + 2)

/-- The height difference between consecutive people -/
def height_diff : ℕ := 5

/-- The maximum allowed height difference for almost-order -/
def max_height_diff : ℕ := 8

/-- The height of the shortest person -/
def min_height : ℕ := 140

theorem almost_order_lineup_correct (n : ℕ) :
  (∀ i j, i < j → j ≤ n → min_height + i * height_diff ≤ min_height + j * height_diff + max_height_diff) →
  almost_order_lineup n = if n ≤ 2 then n else almost_order_lineup (n - 1) + almost_order_lineup (n - 2) :=
sorry

theorem almost_order_lineup_10 : almost_order_lineup 10 = 89 :=
sorry

end NUMINAMATH_CALUDE_almost_order_lineup_correct_almost_order_lineup_10_l1181_118135


namespace NUMINAMATH_CALUDE_no_valid_x_l1181_118178

/-- A circle in the xy-plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is on a circle --/
def is_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem no_valid_x : 
  ∀ x : ℝ, ¬∃ (c : Circle), 
    c.center = (15, 0) ∧ 
    c.radius = 15 ∧
    is_on_circle c (x, 18) ∧ 
    is_on_circle c (x, -18) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_x_l1181_118178


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l1181_118195

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parallelogram in 2D space -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.x - p3.x) + (p2.y - p1.y) * (p4.y - p3.y) = 0

theorem parallelogram_vertex_sum (ABCD : Parallelogram) : 
  ABCD.A = Point.mk (-1) 2 →
  ABCD.B = Point.mk 3 (-1) →
  ABCD.D = Point.mk 5 7 →
  isPerpendicular ABCD.A ABCD.B ABCD.B ABCD.C →
  ABCD.C.x + ABCD.C.y = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l1181_118195


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1181_118134

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1181_118134


namespace NUMINAMATH_CALUDE_candy_distribution_l1181_118173

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 15 →
  num_bags = 5 →
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1181_118173


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1181_118107

theorem quadratic_roots_problem (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) →
  a > b →
  a - b = 8 →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1181_118107


namespace NUMINAMATH_CALUDE_base10_115_eq_base11_A5_l1181_118112

/-- Converts a digit to its character representation in base 11 --/
def toBase11Char (d : ℕ) : Char :=
  if d < 10 then Char.ofNat (d + 48) else 'A'

/-- Converts a natural number to its base 11 representation --/
def toBase11 (n : ℕ) : String :=
  if n < 11 then String.mk [toBase11Char n]
  else toBase11 (n / 11) ++ String.mk [toBase11Char (n % 11)]

/-- Theorem stating that 115 in base 10 is equivalent to A5 in base 11 --/
theorem base10_115_eq_base11_A5 : toBase11 115 = "A5" := by
  sorry

end NUMINAMATH_CALUDE_base10_115_eq_base11_A5_l1181_118112


namespace NUMINAMATH_CALUDE_chocolate_bar_game_l1181_118172

/-- Represents the dimensions of a chocolate bar -/
structure ChocolateBar where
  m : ℕ
  n : ℕ

/-- Predicate to check if Ben has a winning strategy for given dimensions -/
def BenWins (bar : ChocolateBar) : Prop :=
  ∃ (a k : ℕ), a ≥ 2 ∧ k ≥ 0 ∧
    ((bar.m = a - 1 ∧ bar.n = 2^k * a - 1) ∨
     (bar.m = 2^k * a - 1 ∧ bar.n = a - 1))

/-- The main theorem stating the conditions for Ben's winning strategy -/
theorem chocolate_bar_game (bar : ChocolateBar) :
  BenWins bar ↔ 
  ∃ (a k : ℕ), a ≥ 2 ∧ k ≥ 0 ∧
    ((bar.m = a - 1 ∧ bar.n = 2^k * a - 1) ∨
     (bar.m = 2^k * a - 1 ∧ bar.n = a - 1)) := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_game_l1181_118172


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1181_118111

theorem restaurant_bill_calculation :
  let num_bankers : ℕ := 4
  let num_clients : ℕ := 5
  let total_people : ℕ := num_bankers + num_clients
  let cost_per_person : ℚ := 70
  let gratuity_rate : ℚ := 0.20
  let pre_gratuity_total : ℚ := total_people * cost_per_person
  let gratuity_amount : ℚ := pre_gratuity_total * gratuity_rate
  let total_bill : ℚ := pre_gratuity_total + gratuity_amount
  total_bill = 756 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1181_118111
