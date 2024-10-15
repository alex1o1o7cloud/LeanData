import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l3546_354699

noncomputable def f (a : ℝ) (θ : ℝ) (x : ℝ) : ℝ :=
  (a + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

theorem problem_solution (a θ α : ℝ) :
  (∀ x, f a θ x = -f a θ (-x)) →  -- f is an odd function
  f a θ (π/4) = 0 →
  θ ∈ Set.Ioo 0 π →
  f a θ (α/4) = -2/5 →
  α ∈ Set.Ioo (π/2) π →
  (a = -1 ∧ θ = π/2 ∧ Real.sin (α + π/3) = (4 - 3 * Real.sqrt 3) / 10) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3546_354699


namespace NUMINAMATH_CALUDE_cans_purchased_theorem_l3546_354639

/-- The number of cans that can be purchased given the conditions -/
def cans_purchased (N P T : ℚ) : ℚ :=
  5 * N * (T - 1) / P

/-- Theorem stating the number of cans that can be purchased under given conditions -/
theorem cans_purchased_theorem (N P T : ℚ) 
  (h_positive : N > 0 ∧ P > 0 ∧ T > 1) 
  (h_N_P : N / P > 0) -- N cans can be purchased for P quarters
  (h_dollar_worth : (1 : ℚ) = 5 / 4) -- 1 dollar is worth 5 quarters
  (h_fee : (1 : ℚ) > 0) -- There is a 1 dollar fee per transaction
  : cans_purchased N P T = 5 * N * (T - 1) / P :=
sorry

end NUMINAMATH_CALUDE_cans_purchased_theorem_l3546_354639


namespace NUMINAMATH_CALUDE_salad_dressing_ratio_l3546_354671

theorem salad_dressing_ratio (bowl_capacity : ℝ) (oil_density : ℝ) (vinegar_density : ℝ) 
  (total_weight : ℝ) (oil_fraction : ℝ) :
  bowl_capacity = 150 →
  oil_fraction = 2/3 →
  oil_density = 5 →
  vinegar_density = 4 →
  total_weight = 700 →
  (total_weight - oil_fraction * bowl_capacity * oil_density) / vinegar_density / bowl_capacity = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_salad_dressing_ratio_l3546_354671


namespace NUMINAMATH_CALUDE_calculate_expression_l3546_354661

theorem calculate_expression : 75 * 1313 - 25 * 1313 = 65750 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3546_354661


namespace NUMINAMATH_CALUDE_oz_language_lost_words_l3546_354606

/-- Represents the number of letters in the Oz alphabet -/
def alphabet_size : ℕ := 65

/-- Represents the number of letters in a word (either 1 or 2) -/
def word_length : Fin 2 → ℕ
| 0 => 1
| 1 => 2

/-- Calculates the number of words lost when one letter is forbidden -/
def words_lost (n : ℕ) : ℕ :=
  1 + n + n

/-- Theorem stating that forbidding one letter in the Oz language results in 131 lost words -/
theorem oz_language_lost_words :
  words_lost alphabet_size = 131 := by
  sorry

end NUMINAMATH_CALUDE_oz_language_lost_words_l3546_354606


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l3546_354665

-- Define the types
variable (Quadrilateral : Type)
variable (isRhombus : Quadrilateral → Prop)
variable (isParallelogram : Quadrilateral → Prop)

-- Define the original statement
axiom original_statement : ∀ q : Quadrilateral, isRhombus q → isParallelogram q

-- State the theorem to be proved
theorem converse_and_inverse_false :
  (∀ q : Quadrilateral, isParallelogram q → isRhombus q) = False ∧
  (∀ q : Quadrilateral, ¬isRhombus q → ¬isParallelogram q) = False :=
by sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l3546_354665


namespace NUMINAMATH_CALUDE_binary_1101_equals_base5_23_l3546_354684

/-- Converts a binary number to decimal --/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to base-5 --/
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The binary representation of 1101 --/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_equals_base5_23 :
  decimal_to_base5 (binary_to_decimal binary_1101) = [2, 3] := by
  sorry

#eval binary_to_decimal binary_1101
#eval decimal_to_base5 (binary_to_decimal binary_1101)

end NUMINAMATH_CALUDE_binary_1101_equals_base5_23_l3546_354684


namespace NUMINAMATH_CALUDE_power_of_power_l3546_354662

theorem power_of_power (a : ℝ) : (a^3)^4 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3546_354662


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3546_354645

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- Define the interval
def interval : Set ℝ := { x | 2 ≤ x ∧ x ≤ 6 }

-- State the theorem
theorem max_min_values_of_f :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 8) ∧
  (∃ x ∈ interval, f x = -1) :=
sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3546_354645


namespace NUMINAMATH_CALUDE_unique_function_existence_l3546_354628

def is_valid_function (f : ℕ → ℝ) : Prop :=
  (∀ x : ℕ, f x > 0) ∧
  (∀ a b : ℕ, f (a + b) = f a * f b) ∧
  (f 2 = 4)

theorem unique_function_existence : 
  ∃! f : ℕ → ℝ, is_valid_function f ∧ ∀ x : ℕ, f x = 2^x :=
sorry

end NUMINAMATH_CALUDE_unique_function_existence_l3546_354628


namespace NUMINAMATH_CALUDE_two_forty_is_eighty_supplement_of_half_forty_is_onesixtey_half_forty_is_twenty_l3546_354605

-- Define the given angle
def given_angle : ℝ := 40

-- Theorem 1: Two 40° angles form an 80° angle
theorem two_forty_is_eighty : given_angle + given_angle = 80 := by sorry

-- Theorem 2: The supplement of half of a 40° angle is 160°
theorem supplement_of_half_forty_is_onesixtey : 180 - (given_angle / 2) = 160 := by sorry

-- Theorem 3: Half of a 40° angle is 20°
theorem half_forty_is_twenty : given_angle / 2 = 20 := by sorry

end NUMINAMATH_CALUDE_two_forty_is_eighty_supplement_of_half_forty_is_onesixtey_half_forty_is_twenty_l3546_354605


namespace NUMINAMATH_CALUDE_family_savings_theorem_l3546_354689

/-- Represents the monthly financial data of Ivan Tsarevich's family -/
structure FamilyFinances where
  ivan_salary : ℝ
  vasilisa_salary : ℝ
  mother_salary : ℝ
  father_salary : ℝ
  son_scholarship : ℝ
  monthly_expenses : ℝ
  tax_rate : ℝ

def calculate_net_income (gross_income : ℝ) (tax_rate : ℝ) : ℝ :=
  gross_income * (1 - tax_rate)

def calculate_total_net_income (f : FamilyFinances) : ℝ :=
  calculate_net_income f.ivan_salary f.tax_rate +
  calculate_net_income f.vasilisa_salary f.tax_rate +
  calculate_net_income f.mother_salary f.tax_rate +
  calculate_net_income f.father_salary f.tax_rate +
  f.son_scholarship

def calculate_monthly_savings (f : FamilyFinances) : ℝ :=
  calculate_total_net_income f - f.monthly_expenses

theorem family_savings_theorem (f : FamilyFinances) 
  (h1 : f.ivan_salary = 55000)
  (h2 : f.vasilisa_salary = 45000)
  (h3 : f.mother_salary = 18000)
  (h4 : f.father_salary = 20000)
  (h5 : f.son_scholarship = 3000)
  (h6 : f.monthly_expenses = 74000)
  (h7 : f.tax_rate = 0.13) :
  calculate_monthly_savings f = 49060 ∧
  calculate_monthly_savings { f with 
    mother_salary := 10000,
    son_scholarship := 3000 } = 43400 ∧
  calculate_monthly_savings { f with 
    mother_salary := 10000,
    son_scholarship := 16050 } = 56450 := by
  sorry

#check family_savings_theorem

end NUMINAMATH_CALUDE_family_savings_theorem_l3546_354689


namespace NUMINAMATH_CALUDE_unique_solution_l3546_354631

/-- The equation holds for all real x -/
def equation_holds (k : ℕ) : Prop :=
  ∀ x : ℝ, (Real.sin x)^k * Real.sin (k * x) + (Real.cos x)^k * Real.cos (k * x) = (Real.cos (2 * x))^k

/-- k = 3 is the only positive integer solution -/
theorem unique_solution :
  ∃! k : ℕ, k > 0 ∧ equation_holds k :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3546_354631


namespace NUMINAMATH_CALUDE_rudy_total_running_time_l3546_354651

/-- Calculates the total running time for Rudy given his running segments -/
theorem rudy_total_running_time :
  let segment1 : ℝ := 5 * 10  -- 5 miles at 10 minutes per mile
  let segment2 : ℝ := 4 * 9.5 -- 4 miles at 9.5 minutes per mile
  let segment3 : ℝ := 3 * 8.5 -- 3 miles at 8.5 minutes per mile
  let segment4 : ℝ := 2 * 12  -- 2 miles at 12 minutes per mile
  segment1 + segment2 + segment3 + segment4 = 137.5 := by
sorry

end NUMINAMATH_CALUDE_rudy_total_running_time_l3546_354651


namespace NUMINAMATH_CALUDE_equation_solution_l3546_354680

theorem equation_solution :
  ∃ x : ℚ, (x + 2 ≠ 0 ∧ 3 - x ≠ 0) ∧
  ((3 * x - 5) / (x + 2) + (3 * x - 9) / (3 - x) = 2) ∧
  x = -15 / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3546_354680


namespace NUMINAMATH_CALUDE_gamma_less_than_delta_l3546_354663

open Real

theorem gamma_less_than_delta (α β γ δ : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2)
  (h4 : 0 < γ) (h5 : γ < π/2)
  (h6 : 0 < δ) (h7 : δ < π/2)
  (h8 : tan γ = (tan α + tan β) / 2)
  (h9 : 1/cos δ = (1/cos α + 1/cos β) / 2) :
  γ < δ := by
sorry


end NUMINAMATH_CALUDE_gamma_less_than_delta_l3546_354663


namespace NUMINAMATH_CALUDE_no_tetrahedron_with_given_edges_l3546_354609

/-- Represents a tetrahedron with three pairs of opposite edges --/
structure Tetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ

/-- Checks if a tetrahedron with given edge lengths can exist --/
def tetrahedronExists (t : Tetrahedron) : Prop :=
  t.edge1 > 0 ∧ t.edge2 > 0 ∧ t.edge3 > 0 ∧
  t.edge1^2 + t.edge2^2 > t.edge3^2 ∧
  t.edge1^2 + t.edge3^2 > t.edge2^2 ∧
  t.edge2^2 + t.edge3^2 > t.edge1^2

/-- Theorem stating that a tetrahedron with the given edge lengths does not exist --/
theorem no_tetrahedron_with_given_edges :
  ¬ ∃ (t : Tetrahedron), t.edge1 = 12 ∧ t.edge2 = 12.5 ∧ t.edge3 = 13 ∧ tetrahedronExists t :=
by sorry


end NUMINAMATH_CALUDE_no_tetrahedron_with_given_edges_l3546_354609


namespace NUMINAMATH_CALUDE_range_x_and_a_l3546_354626

def P (x a : ℝ) : Prop := -x^2 + 4*a*x - 3*a^2 > 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem range_x_and_a (a : ℝ) (h : a > 0) :
  (∀ x, P x 1 ∧ q x → x > 2 ∧ x < 3) ∧
  (∀ a, (∀ x, 2 < x ∧ x < 3 → a < x ∧ x < 3*a) ↔ 1 ≤ a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_range_x_and_a_l3546_354626


namespace NUMINAMATH_CALUDE_work_increase_with_absences_l3546_354667

/-- Given a total amount of work W and p persons, prove that when 1/3 of the persons are absent,
    the increase in work for each remaining person is W/(2p). -/
theorem work_increase_with_absences (W p : ℝ) (h₁ : W > 0) (h₂ : p > 0) :
  let initial_work_per_person := W / p
  let remaining_persons := (2 / 3) * p
  let new_work_per_person := W / remaining_persons
  new_work_per_person - initial_work_per_person = W / (2 * p) :=
by sorry

end NUMINAMATH_CALUDE_work_increase_with_absences_l3546_354667


namespace NUMINAMATH_CALUDE_part_one_part_two_l3546_354625

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ 2 * m}

-- Theorem for part 1
theorem part_one :
  A ∩ (U \ B 3) = {x | 0 ≤ x ∧ x < 1} := by sorry

-- Theorem for part 2
theorem part_two :
  ∀ m : ℝ, A ∪ B m = B m ↔ 3/2 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3546_354625


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3546_354660

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, a * x + y - 1 - a = 0 ↔ x - (1/2) * y = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3546_354660


namespace NUMINAMATH_CALUDE_brick_length_calculation_l3546_354691

theorem brick_length_calculation (courtyard_length : Real) (courtyard_width : Real)
  (brick_width : Real) (total_bricks : Nat) :
  courtyard_length = 18 ∧ 
  courtyard_width = 12 ∧
  brick_width = 0.06 ∧
  total_bricks = 30000 →
  ∃ brick_length : Real,
    brick_length = 0.12 ∧
    courtyard_length * courtyard_width * 10000 = total_bricks * brick_length * brick_width :=
by sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l3546_354691


namespace NUMINAMATH_CALUDE_max_cookies_per_student_l3546_354611

/-- Proves the maximum number of cookies a single student can take in a class -/
theorem max_cookies_per_student
  (num_students : ℕ)
  (mean_cookies : ℕ)
  (h_num_students : num_students = 25)
  (h_mean_cookies : mean_cookies = 4)
  (h_min_cookie : ∀ student, student ≥ 1) :
  (num_students * mean_cookies) - (num_students - 1) = 76 := by
sorry

end NUMINAMATH_CALUDE_max_cookies_per_student_l3546_354611


namespace NUMINAMATH_CALUDE_repair_shop_earnings_l3546_354668

/-- Calculates the total earnings for a repair shop given the number of repairs and their costs. -/
def total_earnings (phone_repairs laptop_repairs computer_repairs : ℕ) 
  (phone_cost laptop_cost computer_cost : ℕ) : ℕ :=
  phone_repairs * phone_cost + laptop_repairs * laptop_cost + computer_repairs * computer_cost

/-- Theorem stating that the total earnings for the given repairs and costs is $121. -/
theorem repair_shop_earnings : 
  total_earnings 5 2 2 11 15 18 = 121 := by
  sorry

end NUMINAMATH_CALUDE_repair_shop_earnings_l3546_354668


namespace NUMINAMATH_CALUDE_parabola_function_expression_l3546_354610

-- Define the parabola function
def parabola (a : ℝ) (x : ℝ) : ℝ := a * (x + 3)^2 + 2

-- State the theorem
theorem parabola_function_expression :
  ∃ a : ℝ, 
    (parabola a (-3) = 2) ∧ 
    (parabola a 1 = -14) ∧
    (∀ x : ℝ, parabola a x = -(x + 3)^2 + 2) := by
  sorry


end NUMINAMATH_CALUDE_parabola_function_expression_l3546_354610


namespace NUMINAMATH_CALUDE_inequality_solution_l3546_354686

theorem inequality_solution (x : ℝ) : 2 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ x ∈ Set.Ioo (35 / 13) (10 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3546_354686


namespace NUMINAMATH_CALUDE_solution_count_l3546_354677

-- Define the function f
def f (n : ℤ) : ℤ := ⌈(149 * n : ℚ) / 150⌉ - ⌊(150 * n : ℚ) / 151⌋

-- State the theorem
theorem solution_count : 
  (∃ (S : Finset ℤ), (∀ n ∈ S, 1 + ⌊(150 * n : ℚ) / 151⌋ = ⌈(149 * n : ℚ) / 150⌉) ∧ S.card = 15150) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l3546_354677


namespace NUMINAMATH_CALUDE_stock_worth_l3546_354654

def total_modules : ℕ := 11
def cheap_modules : ℕ := 10
def expensive_modules : ℕ := total_modules - cheap_modules
def cheap_cost : ℚ := 3.5
def expensive_cost : ℚ := 10

def total_worth : ℚ := cheap_modules * cheap_cost + expensive_modules * expensive_cost

theorem stock_worth : total_worth = 45 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_l3546_354654


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l3546_354633

theorem min_value_of_expression (x y : ℝ) : 
  (x^2*y - 1)^2 + (x^2 + y)^2 ≥ 1 :=
sorry

theorem min_value_achievable : 
  ∃ x y : ℝ, (x^2*y - 1)^2 + (x^2 + y)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l3546_354633


namespace NUMINAMATH_CALUDE_import_tax_problem_l3546_354617

theorem import_tax_problem (tax_rate : ℝ) (tax_threshold : ℝ) (tax_paid : ℝ) (total_value : ℝ) : 
  tax_rate = 0.07 →
  tax_threshold = 1000 →
  tax_paid = 109.90 →
  tax_paid = tax_rate * (total_value - tax_threshold) →
  total_value = 2570 := by
  sorry

end NUMINAMATH_CALUDE_import_tax_problem_l3546_354617


namespace NUMINAMATH_CALUDE_yellow_balls_percentage_l3546_354632

/-- The percentage of yellow balls in a collection of colored balls. -/
def percentage_yellow_balls (yellow brown blue green : ℕ) : ℚ :=
  (yellow : ℚ) / ((yellow + brown + blue + green : ℕ) : ℚ) * 100

/-- Theorem stating that the percentage of yellow balls is 25% given the specific numbers. -/
theorem yellow_balls_percentage :
  percentage_yellow_balls 75 120 45 60 = 25 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_percentage_l3546_354632


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l3546_354615

/-- The greatest distance between centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 16)
  (h_height : rectangle_height = 20)
  (h_diameter : circle_diameter = 8)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 52 ∧
    ∀ (d' : ℝ), d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 ∧
        d' = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l3546_354615


namespace NUMINAMATH_CALUDE_pigeonhole_friends_l3546_354636

/-- Represents a class of students -/
structure ClassOfStudents where
  n : ℕ  -- number of students
  h : n > 0  -- ensures the class is not empty

/-- Represents the number of friends each student has -/
def FriendCount (c : ClassOfStudents) := Fin c.n → ℕ

/-- The property that if a student has 0 friends, no one has n-1 friends -/
def ValidFriendCount (c : ClassOfStudents) (f : FriendCount c) : Prop :=
  (∃ i, f i = 0) → ∀ j, f j < c.n - 1

theorem pigeonhole_friends (c : ClassOfStudents) (f : FriendCount c) 
    (hf : ValidFriendCount c f) : 
    ∃ i j, i ≠ j ∧ f i = f j :=
  sorry


end NUMINAMATH_CALUDE_pigeonhole_friends_l3546_354636


namespace NUMINAMATH_CALUDE_square_inequality_l3546_354670

theorem square_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 2 / (x - y)) : x^2 > y^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l3546_354670


namespace NUMINAMATH_CALUDE_closest_value_is_112_l3546_354616

def original_value : ℝ := 50.5
def increase_percentage : ℝ := 0.05
def additional_value : ℝ := 0.15
def multiplier : ℝ := 2.1

def options : List ℝ := [105, 110, 112, 115, 120]

def calculated_value : ℝ := multiplier * ((original_value * (1 + increase_percentage)) + additional_value)

theorem closest_value_is_112 : 
  (options.argmin (λ x => |x - calculated_value|)) = some 112 := by sorry

end NUMINAMATH_CALUDE_closest_value_is_112_l3546_354616


namespace NUMINAMATH_CALUDE_bracelet_count_l3546_354622

/-- Calculates the number of sets that can be made from a given number of beads -/
def sets_from_beads (beads : ℕ) : ℕ := beads / 2

/-- Represents the number of beads Nancy and Rose have -/
structure BeadCounts where
  metal : ℕ
  pearl : ℕ
  crystal : ℕ
  stone : ℕ

/-- Calculates the maximum number of bracelets that can be made -/
def max_bracelets (counts : BeadCounts) : ℕ :=
  min (min (sets_from_beads counts.metal) (sets_from_beads counts.pearl))
      (min (sets_from_beads counts.crystal) (sets_from_beads counts.stone))

theorem bracelet_count (counts : BeadCounts)
  (h1 : counts.metal = 40)
  (h2 : counts.pearl = 60)
  (h3 : counts.crystal = 20)
  (h4 : counts.stone = 40) :
  max_bracelets counts = 10 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_count_l3546_354622


namespace NUMINAMATH_CALUDE_collinear_vectors_magnitude_l3546_354634

/-- Given two planar vectors a and b that are collinear and have a negative dot product,
    prove that the magnitude of b is 2√2. -/
theorem collinear_vectors_magnitude (m : ℝ) :
  let a : ℝ × ℝ := (2 * m + 1, 3)
  let b : ℝ × ℝ := (2, m)
  (∃ (k : ℝ), a = k • b) →  -- collinearity condition
  (a.1 * b.1 + a.2 * b.2 < 0) →  -- dot product condition
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_magnitude_l3546_354634


namespace NUMINAMATH_CALUDE_remainder_theorem_l3546_354612

-- Define the polynomial p(x) = x^4 - 2x^2 + 4x - 5
def p (x : ℝ) : ℝ := x^4 - 2*x^2 + 4*x - 5

-- State the theorem
theorem remainder_theorem : 
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ), p x = (x - 1) * q x + (-2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3546_354612


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3546_354675

theorem trigonometric_simplification (α : ℝ) : 
  (2 * Real.sin (π - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3546_354675


namespace NUMINAMATH_CALUDE_children_not_enrolled_l3546_354658

theorem children_not_enrolled (total children_basketball children_robotics children_both : ℕ) 
  (h_total : total = 150)
  (h_basketball : children_basketball = 85)
  (h_robotics : children_robotics = 58)
  (h_both : children_both = 18) :
  total - (children_basketball + children_robotics - children_both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_children_not_enrolled_l3546_354658


namespace NUMINAMATH_CALUDE_union_equals_interval_l3546_354608

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 5*x - 6 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}

-- Define the open interval (-6, 6)
def openInterval : Set ℝ := Ioo (-6) 6

-- Theorem statement
theorem union_equals_interval : A ∪ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_union_equals_interval_l3546_354608


namespace NUMINAMATH_CALUDE_dot_product_condition_l3546_354644

/-- Given vectors a and b, if a · (2a - b) = 0, then k = 12 -/
theorem dot_product_condition (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (2, 1))
  (h2 : b = (-1, k))
  (h3 : a • (2 • a - b) = 0) :
  k = 12 := by sorry

end NUMINAMATH_CALUDE_dot_product_condition_l3546_354644


namespace NUMINAMATH_CALUDE_fraction_equality_l3546_354630

theorem fraction_equality (q r s t v : ℝ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : v / t = 4)
  (h4 : s / v = 1 / 3) :
  t / q = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3546_354630


namespace NUMINAMATH_CALUDE_kevin_cards_l3546_354650

theorem kevin_cards (initial_cards lost_cards : ℝ) 
  (h1 : initial_cards = 47.0)
  (h2 : lost_cards = 7.0) :
  initial_cards - lost_cards = 40.0 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l3546_354650


namespace NUMINAMATH_CALUDE_elaines_rent_percentage_l3546_354664

/-- Proves that given the conditions in the problem, Elaine spent 20% of her annual earnings on rent last year. -/
theorem elaines_rent_percentage (E : ℝ) (P : ℝ) : 
  E > 0 → -- Elaine's earnings last year (assumed positive)
  0.30 * (1.35 * E) = 2.025 * (P / 100 * E) → -- Condition relating this year's and last year's rent
  P = 20 := by sorry

end NUMINAMATH_CALUDE_elaines_rent_percentage_l3546_354664


namespace NUMINAMATH_CALUDE_janice_overtime_shifts_l3546_354672

/-- Proves that Janice worked 3 overtime shifts given her work schedule and earnings --/
theorem janice_overtime_shifts :
  let regular_days : ℕ := 5
  let regular_daily_pay : ℕ := 30
  let overtime_pay : ℕ := 15
  let total_earnings : ℕ := 195
  let regular_earnings := regular_days * regular_daily_pay
  let overtime_earnings := total_earnings - regular_earnings
  overtime_earnings / overtime_pay = 3 := by sorry

end NUMINAMATH_CALUDE_janice_overtime_shifts_l3546_354672


namespace NUMINAMATH_CALUDE_simplify_fraction_l3546_354685

theorem simplify_fraction : 5 * (14 / 3) * (9 / -42) = -5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3546_354685


namespace NUMINAMATH_CALUDE_minimum_pieces_to_capture_all_l3546_354698

/-- Represents a rhombus-shaped game board -/
structure RhombusBoard where
  angle : ℝ
  side_divisions : ℕ

/-- Represents a piece on the game board -/
structure GamePiece where
  position : ℕ × ℕ

/-- Represents the set of cells captured by a piece -/
def captured_cells (board : RhombusBoard) (piece : GamePiece) : Set (ℕ × ℕ) :=
  sorry

/-- The total number of cells on the board -/
def total_cells (board : RhombusBoard) : ℕ :=
  sorry

/-- Checks if a set of pieces captures all cells on the board -/
def captures_all_cells (board : RhombusBoard) (pieces : List GamePiece) : Prop :=
  sorry

theorem minimum_pieces_to_capture_all (board : RhombusBoard)
  (h1 : board.angle = 60)
  (h2 : board.side_divisions = 9) :
  ∃ (pieces : List GamePiece),
    pieces.length = 6 ∧
    captures_all_cells board pieces ∧
    ∀ (other_pieces : List GamePiece),
      captures_all_cells board other_pieces →
      other_pieces.length ≥ 6 :=
  sorry

end NUMINAMATH_CALUDE_minimum_pieces_to_capture_all_l3546_354698


namespace NUMINAMATH_CALUDE_alex_trips_l3546_354642

def savings : ℝ := 14500
def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def grocery_percentage : ℝ := 0.05
def grocery_value : ℝ := 800

def earnings_per_trip : ℝ := trip_charge + grocery_percentage * grocery_value

theorem alex_trips : 
  ∃ n : ℕ, (n : ℝ) * earnings_per_trip ≥ car_cost - savings ∧ 
  ∀ m : ℕ, (m : ℝ) * earnings_per_trip ≥ car_cost - savings → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_alex_trips_l3546_354642


namespace NUMINAMATH_CALUDE_urn_probability_l3546_354641

/-- Represents the color of a ball -/
inductive Color
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a sequence of ball draws -/
def DrawSequence := List Color

/-- The initial state of the urn -/
def initial_state : UrnState := ⟨2, 1⟩

/-- The number of operations performed -/
def num_operations : ℕ := 5

/-- The final number of balls in the urn -/
def final_total_balls : ℕ := 8

/-- The desired final state of the urn -/
def target_state : UrnState := ⟨3, 3⟩

/-- Calculates the probability of a specific draw sequence -/
def sequence_probability (seq : DrawSequence) : ℚ :=
  sorry

/-- Calculates the number of valid sequences that result in the target state -/
def num_valid_sequences : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability : 
  (num_valid_sequences * sequence_probability (List.replicate num_operations Color.Red)) = 8 / 105 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_l3546_354641


namespace NUMINAMATH_CALUDE_sneakers_cost_l3546_354683

/-- The cost of sneakers calculated from lawn mowing charges -/
theorem sneakers_cost (charge_per_yard : ℝ) (yards_to_cut : ℕ) : 
  charge_per_yard * (yards_to_cut : ℝ) = 12.90 :=
by
  sorry

#check sneakers_cost 2.15 6

end NUMINAMATH_CALUDE_sneakers_cost_l3546_354683


namespace NUMINAMATH_CALUDE_equal_squares_count_l3546_354653

/-- Represents a square grid -/
structure Grid (n : ℕ) where
  cells : Fin n → Fin n → Bool

/-- Counts squares with equal black and white cells in a 5x5 grid -/
def count_equal_squares (g : Grid 5) : ℕ :=
  let valid_2x2 : ℕ := 14  -- 16 total - 2 invalid
  let valid_4x4 : ℕ := 2
  valid_2x2 + valid_4x4

/-- Theorem: The number of squares with equal black and white cells is 16 -/
theorem equal_squares_count (g : Grid 5) : count_equal_squares g = 16 := by
  sorry

end NUMINAMATH_CALUDE_equal_squares_count_l3546_354653


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l3546_354647

/-- Given a jar with blue, red, and yellow marbles, this theorem proves
    the number of yellow marbles, given the number of blue and red marbles
    and the probability of picking a yellow marble. -/
theorem yellow_marbles_count
  (blue : ℕ) (red : ℕ) (prob_yellow : ℚ)
  (h_blue : blue = 7)
  (h_red : red = 11)
  (h_prob : prob_yellow = 1/4) :
  ∃ (yellow : ℕ), yellow = 6 ∧
    prob_yellow = yellow / (blue + red + yellow) :=
by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l3546_354647


namespace NUMINAMATH_CALUDE_max_value_theorem_min_value_theorem_l3546_354604

/-- Given a > b > 0 and 7a² + 8ab + 4b² = 24, the maximum value of 3a + 2b occurs when b = √2/2 -/
theorem max_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 7 * a^2 + 8 * a * b + 4 * b^2 = 24) :
  (∀ a' b', a' > b' ∧ b' > 0 ∧ 7 * a'^2 + 8 * a' * b' + 4 * b'^2 = 24 → 3 * a' + 2 * b' ≤ 3 * a + 2 * b) →
  b = Real.sqrt 2 / 2 :=
sorry

/-- Given a > b > 0 and 1/(a - b) + 1/b = 1, the minimum value of a + 3b is 9 -/
theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 1 / (a - b) + 1 / b = 1) :
  a + 3 * b ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_min_value_theorem_l3546_354604


namespace NUMINAMATH_CALUDE_uncrossed_numbers_count_l3546_354623

theorem uncrossed_numbers_count : 
  let total_numbers := 1000
  let gcd_value := Nat.gcd 1000 15
  let crossed_out := (total_numbers - 1) / gcd_value + 1
  total_numbers - crossed_out = 800 := by
  sorry

end NUMINAMATH_CALUDE_uncrossed_numbers_count_l3546_354623


namespace NUMINAMATH_CALUDE_prob_five_largest_l3546_354637

def card_set : Finset ℕ := Finset.range 6

def selection_size : ℕ := 4

def prob_not_select_6 : ℚ :=
  (5 : ℚ) / 6 * 4 / 5 * 3 / 4 * 2 / 3

def prob_not_select_5_or_6 : ℚ :=
  (4 : ℚ) / 6 * 3 / 5 * 2 / 4 * 1 / 3

theorem prob_five_largest (card_set : Finset ℕ) (selection_size : ℕ) 
  (prob_not_select_6 : ℚ) (prob_not_select_5_or_6 : ℚ) :
  card_set = Finset.range 6 →
  selection_size = 4 →
  prob_not_select_6 = (5 : ℚ) / 6 * 4 / 5 * 3 / 4 * 2 / 3 →
  prob_not_select_5_or_6 = (4 : ℚ) / 6 * 3 / 5 * 2 / 4 * 1 / 3 →
  prob_not_select_6 - prob_not_select_5_or_6 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_largest_l3546_354637


namespace NUMINAMATH_CALUDE_chrome_parts_total_l3546_354679

/-- Represents the number of machines of type A -/
def a : ℕ := sorry

/-- Represents the number of machines of type B -/
def b : ℕ := sorry

/-- The total number of machines -/
def total_machines : ℕ := 21

/-- The total number of steel parts -/
def total_steel : ℕ := 50

/-- The number of steel parts in a type A machine -/
def steel_parts_A : ℕ := 3

/-- The number of steel parts in a type B machine -/
def steel_parts_B : ℕ := 2

/-- The number of chrome parts in a type A machine -/
def chrome_parts_A : ℕ := 2

/-- The number of chrome parts in a type B machine -/
def chrome_parts_B : ℕ := 4

theorem chrome_parts_total : 
  a + b = total_machines ∧ 
  steel_parts_A * a + steel_parts_B * b = total_steel →
  chrome_parts_A * a + chrome_parts_B * b = 68 := by
  sorry

end NUMINAMATH_CALUDE_chrome_parts_total_l3546_354679


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l3546_354696

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoid (a b : ℝ) :=
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_lt_b : a < b)

/-- Properties of the isosceles trapezoid -/
def trapezoid_properties (a b : ℝ) (t : IsoscelesTrapezoid a b) :=
  let AB := (a + b) / 2
  let BH := Real.sqrt (a * b)
  let BP := 2 * a * b / (a + b)
  let DF := Real.sqrt ((a^2 + b^2) / 2)
  (AB = (a + b) / 2) ∧
  (BH = Real.sqrt (a * b)) ∧
  (BP = 2 * a * b / (a + b)) ∧
  (DF = Real.sqrt ((a^2 + b^2) / 2)) ∧
  (BP < BH) ∧ (BH < AB) ∧ (AB < DF)

/-- Theorem stating the properties of the isosceles trapezoid -/
theorem isosceles_trapezoid_theorem (a b : ℝ) (t : IsoscelesTrapezoid a b) :
  trapezoid_properties a b t := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l3546_354696


namespace NUMINAMATH_CALUDE_quadratic_root_scaling_l3546_354629

theorem quadratic_root_scaling (a b c n : ℝ) (h : a ≠ 0) :
  let original_eq := fun x : ℝ => a * x^2 + b * x + c
  let scaled_eq := fun x : ℝ => a * x^2 + n * b * x + n^2 * c
  let roots := { x : ℝ | original_eq x = 0 }
  let scaled_roots := { x : ℝ | ∃ y ∈ roots, x = n * y }
  scaled_roots = { x : ℝ | scaled_eq x = 0 } :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_scaling_l3546_354629


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3546_354648

def is_perfect_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^4

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_n_satisfying_conditions : 
  (∀ n : ℕ, n > 0 ∧ n < 2000 → ¬(is_perfect_fourth_power (5*n) ∧ is_perfect_cube (4*n))) ∧
  (is_perfect_fourth_power (5*2000) ∧ is_perfect_cube (4*2000)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3546_354648


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l3546_354618

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 19500) → x = 53800 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l3546_354618


namespace NUMINAMATH_CALUDE_max_profit_l3546_354690

noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 300 * x
  else if x ≥ 40 then (901 * x^2 - 9450 * x + 10000) / x
  else 0

noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then -10 * x^2 + 600 * x - 260
  else if x ≥ 40 then (-x^2 + 9190 * x - 10000) / x
  else 0

theorem max_profit (x : ℝ) :
  (∀ y, y > 0 → W y ≤ W 100) ∧ W 100 = 8990 := by sorry

end NUMINAMATH_CALUDE_max_profit_l3546_354690


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l3546_354627

theorem multiply_and_simplify (x y z : ℝ) :
  (3 * x^2 * z - 7 * y^3) * (9 * x^4 * z^2 + 21 * x^2 * y * z^3 + 49 * y^6) = 27 * x^6 * z^3 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l3546_354627


namespace NUMINAMATH_CALUDE_sum_x_y_equals_ten_l3546_354682

theorem sum_x_y_equals_ten (x y : ℝ) 
  (h1 : |x| - x + y = 6)
  (h2 : x + |y| + y = 16) :
  x + y = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_ten_l3546_354682


namespace NUMINAMATH_CALUDE_poles_inside_base_l3546_354687

/-- A non-convex polygon representing the fence -/
structure Fence where
  isNonConvex : Bool

/-- A power line with poles -/
structure PowerLine where
  totalPoles : Nat

/-- A spy walking around the fence -/
structure Spy where
  totalCount : Nat

/-- The secret base surrounded by the fence -/
structure Base where
  polesInside : Nat

/-- Theorem stating the number of poles inside the base -/
theorem poles_inside_base 
  (fence : Fence) 
  (powerLine : PowerLine)
  (spy : Spy) :
  fence.isNonConvex = true →
  powerLine.totalPoles = 36 →
  spy.totalCount = 2015 →
  ∃ (base : Base), base.polesInside = 1 := by
  sorry

end NUMINAMATH_CALUDE_poles_inside_base_l3546_354687


namespace NUMINAMATH_CALUDE_area_of_EFGH_l3546_354657

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The configuration of rectangles forming EFGH -/
structure Configuration where
  small_rectangle : Rectangle
  large_rectangle : Rectangle

/-- The properties of the configuration as described in the problem -/
def valid_configuration (c : Configuration) : Prop :=
  c.small_rectangle.height = 6 ∧
  c.large_rectangle.width = 2 * c.small_rectangle.width ∧
  c.large_rectangle.height = 2 * c.small_rectangle.height

theorem area_of_EFGH (c : Configuration) (h : valid_configuration c) :
  c.large_rectangle.area = 144 :=
sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l3546_354657


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_19_l3546_354676

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_sum_19 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 19 → n ≤ 9910 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_19_l3546_354676


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3546_354646

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 3 + 1
  (x + 1) / x / (x - 1 / x) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3546_354646


namespace NUMINAMATH_CALUDE_smallest_w_l3546_354635

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_w : ∃ (w : ℕ), 
  w > 0 ∧
  is_factor (2^4) (1452 * w) ∧
  is_factor (3^3) (1452 * w) ∧
  is_factor (13^3) (1452 * w) ∧
  ∀ (x : ℕ), x > 0 ∧ 
    is_factor (2^4) (1452 * x) ∧
    is_factor (3^3) (1452 * x) ∧
    is_factor (13^3) (1452 * x) →
    w ≤ x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_w_l3546_354635


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3546_354669

theorem condition_neither_sufficient_nor_necessary 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (ha₁ : a₁ ≠ 0) (hb₁ : b₁ ≠ 0) (hc₁ : c₁ ≠ 0)
  (ha₂ : a₂ ≠ 0) (hb₂ : b₂ ≠ 0) (hc₂ : c₂ ≠ 0)
  (M : Set ℝ) (hM : M = {x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0})
  (N : Set ℝ) (hN : N = {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0}) :
  ¬(((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)) → (M = N)) ∧ 
  ¬((M = N) → ((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂))) :=
sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3546_354669


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l3546_354624

def f (x : ℝ) : ℝ := |x| + 1

theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l3546_354624


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3546_354652

theorem min_sum_of_squares (x y z : ℝ) : 
  (x + 5) * (y - 5) = 0 →
  (y + 5) * (z - 5) = 0 →
  (z + 5) * (x - 5) = 0 →
  x^2 + y^2 + z^2 ≥ 75 ∧ ∃ (x' y' z' : ℝ), 
    (x' + 5) * (y' - 5) = 0 ∧
    (y' + 5) * (z' - 5) = 0 ∧
    (z' + 5) * (x' - 5) = 0 ∧
    x'^2 + y'^2 + z'^2 = 75 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3546_354652


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3546_354621

theorem quadratic_minimum_value (x y : ℝ) :
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 ≥ -13/5 ∧
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 2 * y₀^2 - 6 * x₀ + 8 * y₀ + 10 = -13/5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3546_354621


namespace NUMINAMATH_CALUDE_sarah_copies_360_pages_l3546_354681

/-- The number of copies Sarah needs to make for each person -/
def copies_per_person : ℕ := 2

/-- The number of people in the meeting -/
def number_of_people : ℕ := 9

/-- The number of pages in the contract -/
def pages_per_contract : ℕ := 20

/-- The total number of pages Sarah will copy -/
def total_pages : ℕ := copies_per_person * number_of_people * pages_per_contract

/-- Theorem stating that the total number of pages Sarah will copy is 360 -/
theorem sarah_copies_360_pages : total_pages = 360 := by
  sorry

end NUMINAMATH_CALUDE_sarah_copies_360_pages_l3546_354681


namespace NUMINAMATH_CALUDE_no_solution_exists_l3546_354602

/-- A polynomial with roots -p, -p-1, -p-2, -p-3 -/
def g (p : ℕ+) (x : ℝ) : ℝ :=
  (x + p) * (x + p + 1) * (x + p + 2) * (x + p + 3)

/-- Coefficients of the expanded polynomial g -/
def a (p : ℕ+) : ℝ := 4 * p + 6
def b (p : ℕ+) : ℝ := 10 * p^2 + 15 * p + 11
def c (p : ℕ+) : ℝ := 12 * p^3 + 18 * p^2 + 22 * p + 6
def d (p : ℕ+) : ℝ := 6 * p^4 + 9 * p^3 + 20 * p^2 + 15 * p + 6

/-- Theorem stating that there is no positive integer p satisfying the given condition -/
theorem no_solution_exists : ¬ ∃ (p : ℕ+), a p + b p + c p + d p = 2056 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3546_354602


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3546_354693

theorem algebraic_expression_value (a b : ℝ) (h : 2 * a - b = 5) :
  2 * b - 4 * a + 8 = -2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3546_354693


namespace NUMINAMATH_CALUDE_tangent_parallel_to_given_line_l3546_354674

-- Define the curve
def f (x : ℝ) := x^4

-- Define the derivative of the curve
def f' (x : ℝ) := 4 * x^3

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the given line
def givenLine (x y : ℝ) : Prop := 4 * x - y + 1 = 0

-- Define parallel lines
def parallel (m₁ b₁ m₂ b₂ : ℝ) : Prop := m₁ = m₂ ∧ b₁ ≠ b₂

theorem tangent_parallel_to_given_line :
  let m := f' P.1  -- Slope of tangent line
  let b := P.2 - m * P.1  -- y-intercept of tangent line
  parallel m b 4 (-1) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_given_line_l3546_354674


namespace NUMINAMATH_CALUDE_complex_number_problem_l3546_354613

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  z₁ * (2 + Complex.I) = 5 * Complex.I →
  (∃ (r : ℝ), z₁ + z₂ = r) →
  (∃ (y : ℝ), y ≠ 0 ∧ z₁ * z₂ = y * Complex.I) →
  z₂ = -4 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3546_354613


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l3546_354655

def base_5_to_10 (b : ℕ) : ℕ := 780 * b

def base_c_to_10 (c : ℕ) : ℕ := 4 * (c + 1)

def valid_base_5_digit (b : ℕ) : Prop := 1 ≤ b ∧ b ≤ 4

def valid_base_c (c : ℕ) : Prop := c > 6

theorem smallest_sum_B_plus_c :
  ∃ (B c : ℕ),
    valid_base_5_digit B ∧
    valid_base_c c ∧
    base_5_to_10 B = base_c_to_10 c ∧
    (∀ (B' c' : ℕ),
      valid_base_5_digit B' →
      valid_base_c c' →
      base_5_to_10 B' = base_c_to_10 c' →
      B + c ≤ B' + c') ∧
    B + c = 195 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l3546_354655


namespace NUMINAMATH_CALUDE_log_equality_implies_product_one_l3546_354643

theorem log_equality_implies_product_one (M N : ℝ) 
  (h1 : (Real.log N / Real.log M)^2 = (Real.log M / Real.log N)^2)
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) :
  M * N = 1 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_product_one_l3546_354643


namespace NUMINAMATH_CALUDE_exam_students_count_l3546_354600

/-- The total number of students in an examination -/
def total_students : ℕ := 300

/-- The number of students who just passed -/
def students_just_passed : ℕ := 60

/-- The percentage of students who got first division -/
def first_division_percent : ℚ := 26 / 100

/-- The percentage of students who got second division -/
def second_division_percent : ℚ := 54 / 100

/-- Theorem stating that the total number of students is 300 -/
theorem exam_students_count :
  (students_just_passed : ℚ) / total_students = 1 - first_division_percent - second_division_percent :=
by sorry

end NUMINAMATH_CALUDE_exam_students_count_l3546_354600


namespace NUMINAMATH_CALUDE_leibniz_recursive_relation_leibniz_boundary_condition_pascal_leibniz_relation_l3546_354678

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Element in Leibniz's Triangle -/
def leibniz (n k : ℕ) : ℚ := 1 / ((n + 1 : ℚ) * (binomial n k))

/-- Theorem stating the recursive relationship in Leibniz's Triangle -/
theorem leibniz_recursive_relation (n k : ℕ) (h : 0 < k ∧ k ≤ n) :
  leibniz n (k - 1) + leibniz n k = leibniz (n - 1) (k - 1) := by sorry

/-- Theorem stating that the formula for Leibniz's Triangle satisfies its boundary condition -/
theorem leibniz_boundary_condition (n : ℕ) :
  leibniz n 0 = 1 / (n + 1 : ℚ) ∧ leibniz n n = 1 / (n + 1 : ℚ) := by sorry

/-- Main theorem relating Pascal's Triangle to Leibniz's Triangle -/
theorem pascal_leibniz_relation (n k : ℕ) (h : k ≤ n) :
  leibniz n k = 1 / ((n + 1 : ℚ) * (binomial n k : ℚ)) := by sorry

end NUMINAMATH_CALUDE_leibniz_recursive_relation_leibniz_boundary_condition_pascal_leibniz_relation_l3546_354678


namespace NUMINAMATH_CALUDE_tangent_circles_area_l3546_354638

/-- The area of the region outside a circle of radius 2 and inside two circles of radius 3
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem tangent_circles_area : Real :=
  let r₁ : Real := 2  -- radius of smaller circle
  let r₂ : Real := 3  -- radius of larger circles
  let total_area : Real := (5 * Real.pi) / 2 - 4 * Real.sqrt 5
  total_area

#check tangent_circles_area

end NUMINAMATH_CALUDE_tangent_circles_area_l3546_354638


namespace NUMINAMATH_CALUDE_slope_of_solutions_l3546_354673

theorem slope_of_solutions (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : (5 / x₁) + (4 / y₁) = 0) (h₃ : (5 / x₂) + (4 / y₂) = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_solutions_l3546_354673


namespace NUMINAMATH_CALUDE_simplify_sqrt_one_third_l3546_354692

theorem simplify_sqrt_one_third : Real.sqrt (1/3) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_one_third_l3546_354692


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sevenths_l3546_354649

theorem smallest_fraction_greater_than_five_sevenths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (5 : ℚ) / 7 < (a : ℚ) / b →
    (68 : ℚ) / 95 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sevenths_l3546_354649


namespace NUMINAMATH_CALUDE_wedding_guest_ratio_l3546_354694

def wedding_guests (bridgette_guests : ℕ) (extra_plates : ℕ) (spears_per_plate : ℕ) (total_spears : ℕ) : Prop :=
  ∃ (alex_guests : ℕ),
    (bridgette_guests + alex_guests + extra_plates) * spears_per_plate = total_spears ∧
    alex_guests * 3 = bridgette_guests * 2

theorem wedding_guest_ratio :
  wedding_guests 84 10 8 1200 :=
sorry

end NUMINAMATH_CALUDE_wedding_guest_ratio_l3546_354694


namespace NUMINAMATH_CALUDE_robin_gum_count_l3546_354697

theorem robin_gum_count (initial : Real) (additional : Real) (total : Real) : 
  initial = 18.0 → additional = 44.0 → total = initial + additional → total = 62.0 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l3546_354697


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3546_354695

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 5
  side_b : b = 12
  side_c : c = 13

/-- Square inscribed with one vertex at the right angle -/
def inscribed_square_x (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x < t.a ∧ x < t.b ∧ x / t.a = x / t.b

/-- Square inscribed with one side along the hypotenuse -/
def inscribed_square_y (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y < t.c ∧ (t.a / t.c) * y / t.a = y / t.c

/-- The main theorem -/
theorem inscribed_squares_ratio (t : RightTriangle) 
  (x y : ℝ) (hx : inscribed_square_x t x) (hy : inscribed_square_y t y) : 
  x / y = 4320 / 2873 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3546_354695


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3546_354659

theorem sufficient_not_necessary (a : ℝ) :
  (a < -1 → ∃ x₀ : ℝ, a * Real.sin x₀ + 1 < 0) ∧
  (∃ a : ℝ, a ≥ -1 ∧ ∃ x₀ : ℝ, a * Real.sin x₀ + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3546_354659


namespace NUMINAMATH_CALUDE_cost_price_articles_l3546_354603

/-- Given that the cost price of N articles equals the selling price of 50 articles,
    and the profit percentage is 10.000000000000004%, prove that N = 55. -/
theorem cost_price_articles (N : ℕ) (C S : ℝ) : 
  N * C = 50 * S →
  (S - C) / C * 100 = 10.000000000000004 →
  N = 55 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_articles_l3546_354603


namespace NUMINAMATH_CALUDE_chord_segment_ratio_l3546_354601

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a chord
structure Chord (c : Circle) where
  p1 : Point
  p2 : Point

-- Define the intersection of two chords
def chord_intersection (c : Circle) (ch1 ch2 : Chord c) : Point := sorry

-- Power of a Point theorem
axiom power_of_point (c : Circle) (ch1 ch2 : Chord c) (q : Point) :
  let x := chord_intersection c ch1 ch2
  (x.1 - q.1) * (ch1.p2.1 - q.1) = (x.1 - q.1) * (ch2.p2.1 - q.1)

-- Main theorem
theorem chord_segment_ratio (c : Circle) (ch1 ch2 : Chord c) :
  let q := chord_intersection c ch1 ch2
  let x := ch1.p1
  let y := ch1.p2
  let w := ch2.p1
  let z := ch2.p2
  (x.1 - q.1) = 5 →
  (w.1 - q.1) = 7 →
  (y.1 - q.1) / (z.1 - q.1) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_segment_ratio_l3546_354601


namespace NUMINAMATH_CALUDE_diophantus_age_problem_l3546_354656

theorem diophantus_age_problem :
  ∀ (x : ℕ),
    (x / 6 : ℚ) + (x / 12 : ℚ) + (x / 7 : ℚ) + 5 + (x / 2 : ℚ) + 4 = x →
    x = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_diophantus_age_problem_l3546_354656


namespace NUMINAMATH_CALUDE_sum_zero_ratio_negative_half_l3546_354688

theorem sum_zero_ratio_negative_half 
  (w x y z : ℝ) 
  (hw : w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z) 
  (hsum : w + x + y + z = 0) :
  (w * y + x * z) / (w^2 + x^2 + y^2 + z^2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_ratio_negative_half_l3546_354688


namespace NUMINAMATH_CALUDE_lines_parallel_iff_same_slope_diff_intercept_l3546_354666

/-- Two lines in slope-intercept form are parallel if and only if they have the same slope and different y-intercepts -/
theorem lines_parallel_iff_same_slope_diff_intercept (k₁ k₂ l₁ l₂ : ℝ) :
  (∀ x y : ℝ, y = k₁ * x + l₁ ∨ y = k₂ * x + l₂) →
  (∀ x y : ℝ, y = k₁ * x + l₁ → y = k₂ * x + l₂ → False) ↔ k₁ = k₂ ∧ l₁ ≠ l₂ :=
by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_same_slope_diff_intercept_l3546_354666


namespace NUMINAMATH_CALUDE_number_of_white_balls_l3546_354620

/-- Given the number of red and blue balls, and the relationship between red balls and the sum of blue and white balls, prove the number of white balls. -/
theorem number_of_white_balls (red blue : ℕ) (h1 : red = 60) (h2 : blue = 30) 
  (h3 : red = blue + white + 5) : white = 25 :=
by
  sorry

#check number_of_white_balls

end NUMINAMATH_CALUDE_number_of_white_balls_l3546_354620


namespace NUMINAMATH_CALUDE_solution_set_equality_l3546_354640

/-- A function f: ℝ → ℝ that is odd, monotonically increasing on (0, +∞), and f(-1) = 2 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) ∧
  (f (-1) = 2)

/-- The theorem statement -/
theorem solution_set_equality (f : ℝ → ℝ) (h : special_function f) :
  {x : ℝ | x > 0 ∧ f (x - 1) + 2 ≤ 0} = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3546_354640


namespace NUMINAMATH_CALUDE_peters_horses_feeding_days_l3546_354619

theorem peters_horses_feeding_days :
  let num_horses : ℕ := 4
  let oats_per_meal : ℕ := 4
  let oats_meals_per_day : ℕ := 2
  let grain_per_day : ℕ := 3
  let total_food : ℕ := 132
  
  let food_per_horse_per_day : ℕ := oats_per_meal * oats_meals_per_day + grain_per_day
  let total_food_per_day : ℕ := num_horses * food_per_horse_per_day
  
  total_food / total_food_per_day = 3 :=
by sorry

end NUMINAMATH_CALUDE_peters_horses_feeding_days_l3546_354619


namespace NUMINAMATH_CALUDE_expected_attempts_proof_l3546_354614

/-- The expected number of attempts to open a safe with n keys -/
def expected_attempts (n : ℕ) : ℚ :=
  (n + 1 : ℚ) / 2

/-- Theorem stating that the expected number of attempts to open a safe
    with n keys distributed sequentially to n students is (n+1)/2 -/
theorem expected_attempts_proof (n : ℕ) :
  expected_attempts n = (n + 1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_attempts_proof_l3546_354614


namespace NUMINAMATH_CALUDE_operation_result_l3546_354607

theorem operation_result (x : ℝ) : 40 + 5 * x / (180 / 3) = 41 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l3546_354607
