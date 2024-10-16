import Mathlib

namespace NUMINAMATH_CALUDE_child_support_owed_amount_l3373_337375

/-- Calculates the amount owed in child support given the specified conditions --/
def child_support_owed (
  support_rate : ℝ)
  (initial_salary : ℝ)
  (initial_years : ℕ)
  (raise_percentage : ℝ)
  (raise_years : ℕ)
  (amount_paid : ℝ) : ℝ :=
  let total_initial_income := initial_salary * initial_years
  let raised_salary := initial_salary * (1 + raise_percentage)
  let total_raised_income := raised_salary * raise_years
  let total_income := total_initial_income + total_raised_income
  let total_support_due := total_income * support_rate
  total_support_due - amount_paid

/-- Theorem stating that the amount owed in child support is $69,000 --/
theorem child_support_owed_amount : 
  child_support_owed 0.3 30000 3 0.2 4 1200 = 69000 := by
  sorry

end NUMINAMATH_CALUDE_child_support_owed_amount_l3373_337375


namespace NUMINAMATH_CALUDE_complex_power_difference_l3373_337356

theorem complex_power_difference (x : ℂ) : 
  x - 1 / x = 2 * Complex.I → x^729 - 1 / x^729 = 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_power_difference_l3373_337356


namespace NUMINAMATH_CALUDE_floor_sqrt_63_l3373_337303

theorem floor_sqrt_63 : ⌊Real.sqrt 63⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_63_l3373_337303


namespace NUMINAMATH_CALUDE_number_from_percentage_l3373_337378

theorem number_from_percentage (x : ℝ) : 0.15 * 0.30 * 0.50 * x = 117 → x = 5200 := by
  sorry

end NUMINAMATH_CALUDE_number_from_percentage_l3373_337378


namespace NUMINAMATH_CALUDE_science_club_enrollment_l3373_337368

theorem science_club_enrollment (total : ℕ) (chem : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 75)
  (h2 : chem = 45)
  (h3 : bio = 30)
  (h4 : both = 18) :
  total - (chem + bio - both) = 18 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l3373_337368


namespace NUMINAMATH_CALUDE_triangle_side_length_l3373_337334

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a = 4 →
  B = π / 3 →
  A = π / 4 →
  b = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3373_337334


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l3373_337358

theorem product_of_sum_of_squares (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 + (a*d - b*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l3373_337358


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l3373_337367

/-- Proves that the number of meters of cloth sold is 400 -/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  total_selling_price = 18000 →
  loss_per_meter = 5 →
  cost_price_per_meter = 50 →
  (total_selling_price / (cost_price_per_meter - loss_per_meter) : ℕ) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_problem_l3373_337367


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l3373_337319

/-- Given a cylindrical tank with radius 5 inches and height 6 inches, 
    this theorem proves that increasing the radius by x inches 
    results in the same volume increase as increasing the height by 2x inches 
    when x = 10/3. -/
theorem cylinder_volume_increase (x : ℝ) : x = 10/3 ↔ 
  π * (5 + x)^2 * 6 = π * 5^2 * (6 + 2*x) := by
  sorry

#check cylinder_volume_increase

end NUMINAMATH_CALUDE_cylinder_volume_increase_l3373_337319


namespace NUMINAMATH_CALUDE_correct_calculation_l3373_337310

theorem correct_calculation (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3373_337310


namespace NUMINAMATH_CALUDE_equidistant_line_from_three_parallel_lines_l3373_337342

/-- Given three parallel lines in the form Ax + By = Cᵢ, 
    this theorem states that the line Ax + By = (C₁ + 2C₂ + C₃) / 4 
    is equidistant from all three lines. -/
theorem equidistant_line_from_three_parallel_lines 
  (A B C₁ C₂ C₃ : ℝ) 
  (h_distinct₁ : C₁ ≠ C₂) 
  (h_distinct₂ : C₂ ≠ C₃) 
  (h_distinct₃ : C₁ ≠ C₃) :
  let d₁₂ := |C₂ - C₁| / Real.sqrt (A^2 + B^2)
  let d₂₃ := |C₃ - C₂| / Real.sqrt (A^2 + B^2)
  let d₁₃ := |C₃ - C₁| / Real.sqrt (A^2 + B^2)
  let M := (C₁ + 2*C₂ + C₃) / 4
  ∀ x y, A*x + B*y = M → 
    |A*x + B*y - C₁| / Real.sqrt (A^2 + B^2) = 
    |A*x + B*y - C₂| / Real.sqrt (A^2 + B^2) ∧
    |A*x + B*y - C₂| / Real.sqrt (A^2 + B^2) = 
    |A*x + B*y - C₃| / Real.sqrt (A^2 + B^2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_line_from_three_parallel_lines_l3373_337342


namespace NUMINAMATH_CALUDE_set_union_problem_l3373_337383

theorem set_union_problem (A B : Set ℕ) (m : ℕ) :
  A = {1, 2, 3, 4} →
  B = {m, 4, 7, 8} →
  A ∩ B = {1, 4} →
  A ∪ B = {1, 2, 3, 4, 7, 8} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l3373_337383


namespace NUMINAMATH_CALUDE_project_hours_theorem_l3373_337347

/-- Represents the hours worked by three people on a project -/
structure ProjectHours where
  least : ℕ
  middle : ℕ
  most : ℕ

/-- The total hours worked on the project -/
def total_hours (h : ProjectHours) : ℕ := h.least + h.middle + h.most

/-- The condition that the working times are in the ratio 1:2:3 -/
def ratio_condition (h : ProjectHours) : Prop :=
  h.middle = 2 * h.least ∧ h.most = 3 * h.least

/-- The condition that the hardest working person worked 40 hours more than the person who worked the least -/
def difference_condition (h : ProjectHours) : Prop :=
  h.most = h.least + 40

theorem project_hours_theorem (h : ProjectHours) 
  (hc1 : ratio_condition h) 
  (hc2 : difference_condition h) : 
  total_hours h = 120 := by
  sorry

#check project_hours_theorem

end NUMINAMATH_CALUDE_project_hours_theorem_l3373_337347


namespace NUMINAMATH_CALUDE_water_needed_for_growth_medium_l3373_337326

/-- Given a growth medium mixture with initial volumes of nutrient concentrate and water,
    calculate the amount of water needed for a specified total volume. -/
theorem water_needed_for_growth_medium 
  (nutrient_vol : ℝ) 
  (initial_water_vol : ℝ) 
  (total_vol : ℝ) 
  (h1 : nutrient_vol = 0.08)
  (h2 : initial_water_vol = 0.04)
  (h3 : total_vol = 1) :
  (total_vol * initial_water_vol) / (nutrient_vol + initial_water_vol) = 1/3 := by
  sorry

#check water_needed_for_growth_medium

end NUMINAMATH_CALUDE_water_needed_for_growth_medium_l3373_337326


namespace NUMINAMATH_CALUDE_reflection_of_A_across_x_axis_l3373_337338

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the original point A
def A : Point := (1, -2)

-- Define reflection across x-axis
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem to prove
theorem reflection_of_A_across_x_axis :
  reflect_x A = (1, 2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_A_across_x_axis_l3373_337338


namespace NUMINAMATH_CALUDE_max_value_theorem_l3373_337312

theorem max_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 3 * x^2 - 2 * x * y + y^2 = 6) :
  ∃ (z : ℝ), z = 9 + 3 * Real.sqrt 3 ∧ 
  ∀ (w : ℝ), w = 3 * x^2 + 2 * x * y + y^2 → w ≤ z :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3373_337312


namespace NUMINAMATH_CALUDE_no_solution_equations_l3373_337359

theorem no_solution_equations :
  (∀ x : ℝ, (|2*x| + 7 ≠ 0)) ∧
  (∀ x : ℝ, (Real.sqrt (3*x) + 2 ≠ 0)) ∧
  (∃ x : ℝ, ((x - 5)^2 = 0)) ∧
  (∃ x : ℝ, (Real.cos x - 1 = 0)) ∧
  (∃ x : ℝ, (|x| - 3 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equations_l3373_337359


namespace NUMINAMATH_CALUDE_correct_calculation_l3373_337361

theorem correct_calculation (a : ℝ) : 2 * a^3 * 3 * a^5 = 6 * a^8 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3373_337361


namespace NUMINAMATH_CALUDE_cos_nineteen_pi_fourths_l3373_337315

theorem cos_nineteen_pi_fourths : Real.cos (19 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_nineteen_pi_fourths_l3373_337315


namespace NUMINAMATH_CALUDE_certain_number_proof_l3373_337318

theorem certain_number_proof (m : ℕ) : 9999 * m = 724827405 → m = 72483 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3373_337318


namespace NUMINAMATH_CALUDE_middle_book_price_l3373_337313

/-- A sequence of 49 numbers where each number differs by 5 from its adjacent numbers -/
def IncreasingSequence (a : ℕ → ℚ) : Prop :=
  (∀ n < 48, a (n + 1) = a n + 5) ∧ 
  (∀ n, n < 49)

theorem middle_book_price
  (a : ℕ → ℚ)
  (h1 : IncreasingSequence a)
  (h2 : a 48 = 2 * (a 23 + a 24 + a 25)) :
  a 24 = 24 := by
  sorry

end NUMINAMATH_CALUDE_middle_book_price_l3373_337313


namespace NUMINAMATH_CALUDE_sample_size_eq_selected_cards_l3373_337323

/-- Represents a statistical study of student report cards -/
structure ReportCardStudy where
  totalStudents : ℕ
  selectedCards : ℕ
  h_total : totalStudents = 1000
  h_selected : selectedCards = 100
  h_selected_le_total : selectedCards ≤ totalStudents

/-- The sample size of a report card study is equal to the number of selected cards -/
theorem sample_size_eq_selected_cards (study : ReportCardStudy) : 
  study.selectedCards = 100 := by
  sorry

#check sample_size_eq_selected_cards

end NUMINAMATH_CALUDE_sample_size_eq_selected_cards_l3373_337323


namespace NUMINAMATH_CALUDE_bella_apple_consumption_l3373_337305

/-- The fraction of apples Bella consumes from what Grace picks -/
def bella_fraction : ℚ := 1 / 18

/-- The number of apples Bella eats per day -/
def bella_daily_apples : ℕ := 6

/-- The number of apples Grace has left after 6 weeks -/
def grace_remaining_apples : ℕ := 504

/-- The number of weeks in the problem -/
def weeks : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem bella_apple_consumption :
  bella_fraction = 1 / 18 :=
sorry

end NUMINAMATH_CALUDE_bella_apple_consumption_l3373_337305


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3373_337341

-- Define the function f
def f (x : ℝ) : ℝ := x * abs (x - 2)

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) : 
  f (2 - Real.log (x + 1)) > f 3 ↔ -1 < x ∧ x < Real.exp (-1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3373_337341


namespace NUMINAMATH_CALUDE_prob_sum_18_three_dice_l3373_337376

-- Define a die as having 6 faces
def die_faces : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 3

-- Define the target sum
def target_sum : ℕ := 18

-- Define the probability of rolling a specific number on a single die
def single_die_prob : ℚ := 1 / die_faces

-- Statement to prove
theorem prob_sum_18_three_dice : 
  (single_die_prob ^ num_dice : ℚ) = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_prob_sum_18_three_dice_l3373_337376


namespace NUMINAMATH_CALUDE_sum_of_first_n_natural_numbers_l3373_337380

theorem sum_of_first_n_natural_numbers (n : ℕ) : 
  (∃ (k : ℕ), k > 0 ∧ k < 10 ∧ n * (n + 1) / 2 = 111 * k) ↔ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_n_natural_numbers_l3373_337380


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3373_337370

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (a + i) * i = b + i) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3373_337370


namespace NUMINAMATH_CALUDE_total_subjects_l3373_337349

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) 
  (h1 : average_all = 79)
  (h2 : average_five = 74)
  (h3 : last_subject = 104) : 
  ∃ n : ℕ, n = 6 ∧ n * average_all = 5 * average_five + last_subject :=
by sorry

end NUMINAMATH_CALUDE_total_subjects_l3373_337349


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l3373_337381

theorem choose_four_from_nine :
  Nat.choose 9 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l3373_337381


namespace NUMINAMATH_CALUDE_floor_sum_example_l3373_337333

theorem floor_sum_example : ⌊(-13.7 : ℝ)⌋ + ⌊(13.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3373_337333


namespace NUMINAMATH_CALUDE_smallest_among_given_numbers_l3373_337362

theorem smallest_among_given_numbers : 
  let a := Real.sqrt 3
  let b := -(1/3 : ℝ)
  let c := -2
  let d := 0
  c < b ∧ c < d ∧ c < a :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_given_numbers_l3373_337362


namespace NUMINAMATH_CALUDE_triangle_area_l3373_337372

-- Define the linear functions
def f (x : ℝ) : ℝ := x - 4
def g (x : ℝ) : ℝ := -x - 4

-- Define the triangle
def Triangle := {(x, y) : ℝ × ℝ | (y = f x ∨ y = g x) ∧ y ≥ 0}

-- Theorem statement
theorem triangle_area : MeasureTheory.volume Triangle = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3373_337372


namespace NUMINAMATH_CALUDE_apartment_expenditure_difference_l3373_337393

def akeno_expenditure : ℕ := 2985
def lev_expenditure : ℕ := akeno_expenditure / 3
def extra_akeno : ℕ := 1172

theorem apartment_expenditure_difference :
  ∃ (ambrocio_expenditure : ℕ),
    ambrocio_expenditure < lev_expenditure ∧
    akeno_expenditure = lev_expenditure + ambrocio_expenditure + extra_akeno ∧
    lev_expenditure - ambrocio_expenditure = 177 :=
by sorry

end NUMINAMATH_CALUDE_apartment_expenditure_difference_l3373_337393


namespace NUMINAMATH_CALUDE_water_bottle_shape_l3373_337398

/-- Represents the volume of water in a bottle as a function of height -/
noncomputable def VolumeFunction := ℝ → ℝ

/-- A water bottle with a given height and volume function -/
structure WaterBottle where
  height : ℝ
  volume : VolumeFunction
  height_pos : height > 0

/-- The shape of a water bottle is non-linear and increases faster than linear growth -/
def IsNonLinearIncreasing (b : WaterBottle) : Prop :=
  b.volume (b.height / 2) > (1 / 2) * b.volume b.height

theorem water_bottle_shape (b : WaterBottle) 
  (h : IsNonLinearIncreasing b) : 
  ∃ (k : ℝ), k > 0 ∧ ∀ h, 0 ≤ h ∧ h ≤ b.height → b.volume h = k * h^2 :=
sorry

end NUMINAMATH_CALUDE_water_bottle_shape_l3373_337398


namespace NUMINAMATH_CALUDE_unique_right_triangle_from_medians_l3373_337300

/-- Given two positive real numbers representing the lengths of medians from the endpoints of a hypotenuse,
    there exists at most one right triangle with these medians. -/
theorem unique_right_triangle_from_medians (s_a s_b : ℝ) (h_sa : s_a > 0) (h_sb : s_b > 0) :
  ∃! (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
    s_a^2 = (b/2)^2 + (c/2)^2 ∧ s_b^2 = (a/2)^2 + (c/2)^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_from_medians_l3373_337300


namespace NUMINAMATH_CALUDE_triangle_problem_l3373_337387

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧
  a ≠ b ∧
  c = Real.sqrt 3 ∧
  Real.sqrt 3 * (Real.cos A)^2 - Real.sqrt 3 * (Real.cos B)^2 = Real.sin A * Real.cos A - Real.sin B * Real.cos B ∧
  Real.sin A = 4/5 →
  C = π/6 ∧
  1/2 * a * c * Real.sin B = (24 * Real.sqrt 3 + 18) / 25 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3373_337387


namespace NUMINAMATH_CALUDE_weight_replacement_l3373_337392

theorem weight_replacement (initial_count : ℕ) (average_increase weight_new : ℝ) :
  initial_count = 8 →
  average_increase = 4.2 →
  weight_new = 98.6 →
  ∃ weight_replaced : ℝ,
    weight_replaced = weight_new - (initial_count * average_increase) ∧
    weight_replaced = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l3373_337392


namespace NUMINAMATH_CALUDE_union_equals_reals_subset_of_complement_l3373_337301

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2*a}

-- Theorem for part (1)
theorem union_equals_reals (a : ℝ) : 
  A ∪ B a = Set.univ ↔ a ∈ Set.Iic 0 :=
sorry

-- Theorem for part (2)
theorem subset_of_complement (a : ℝ) :
  B a ⊆ (Set.univ \ A) ↔ a ∈ Set.Ici (1/2) :=
sorry

end NUMINAMATH_CALUDE_union_equals_reals_subset_of_complement_l3373_337301


namespace NUMINAMATH_CALUDE_set_B_determination_l3373_337352

theorem set_B_determination (U A B : Set ℕ) : 
  U = A ∪ B ∧ 
  U = {x : ℕ | 0 ≤ x ∧ x ≤ 10} ∧ 
  A ∩ (U \ B) = {1, 3, 5, 7} → 
  B = {0, 2, 4, 6, 8, 9, 10} := by
  sorry

end NUMINAMATH_CALUDE_set_B_determination_l3373_337352


namespace NUMINAMATH_CALUDE_select_blocks_count_l3373_337325

def grid_size : ℕ := 6
def blocks_to_select : ℕ := 4

/-- The number of ways to select 4 blocks from a 6x6 grid, 
    such that no two blocks are in the same row or column -/
def select_blocks : ℕ := Nat.choose grid_size blocks_to_select * 
                         Nat.choose grid_size blocks_to_select * 
                         Nat.factorial blocks_to_select

theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end NUMINAMATH_CALUDE_select_blocks_count_l3373_337325


namespace NUMINAMATH_CALUDE_ratio_problem_l3373_337307

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) :
  second_part = 5 →
  percent = 180 →
  first_part / second_part = percent / 100 →
  first_part = 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3373_337307


namespace NUMINAMATH_CALUDE_min_abs_product_l3373_337354

/-- Given two perpendicular lines and non-zero real parameters, prove the minimum value of their product's absolute value -/
theorem min_abs_product (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (1 : ℝ) / a^2 = (a^2 + 1) / b) : 
  ∀ x y : ℝ, |x * y| ≥ 2 → |a * b| ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_product_l3373_337354


namespace NUMINAMATH_CALUDE_john_bought_three_shirts_l3373_337336

/-- The number of dress shirts John bought -/
def num_shirts : ℕ := 3

/-- The cost of each dress shirt in dollars -/
def shirt_cost : ℚ := 20

/-- The tax rate as a percentage -/
def tax_rate : ℚ := 10

/-- The total amount John paid in dollars -/
def total_paid : ℚ := 66

/-- Theorem stating that the number of shirts John bought is correct -/
theorem john_bought_three_shirts :
  (shirt_cost * num_shirts) * (1 + tax_rate / 100) = total_paid := by
  sorry

end NUMINAMATH_CALUDE_john_bought_three_shirts_l3373_337336


namespace NUMINAMATH_CALUDE_melanie_football_games_l3373_337327

theorem melanie_football_games (total_games missed_games : ℕ) 
  (h1 : total_games = 7)
  (h2 : missed_games = 4) :
  total_games - missed_games = 3 := by
  sorry

end NUMINAMATH_CALUDE_melanie_football_games_l3373_337327


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l3373_337357

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel
  (m n : Line) (α : Plane) 
  (h_diff : m ≠ n)
  (h_m_perp : perp m α)
  (h_n_perp : perp n α) :
  para m n :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l3373_337357


namespace NUMINAMATH_CALUDE_dandelion_picking_l3373_337329

theorem dandelion_picking (billy_initial : ℕ) (george_initial : ℕ) (average : ℕ) : 
  billy_initial = 36 →
  george_initial = billy_initial / 3 →
  average = 34 →
  (billy_initial + george_initial + 2 * (average - (billy_initial + george_initial) / 2)) / 2 = average →
  average - (billy_initial + george_initial) / 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_dandelion_picking_l3373_337329


namespace NUMINAMATH_CALUDE_smallest_area_is_40_l3373_337337

/-- A rectangle with even side lengths that can be divided into squares and dominoes -/
structure CheckeredRectangle where
  width : Nat
  height : Nat
  has_square : Bool
  has_domino : Bool
  width_even : Even width
  height_even : Even height
  both_types : has_square ∧ has_domino

/-- The area of a CheckeredRectangle -/
def area (r : CheckeredRectangle) : Nat :=
  r.width * r.height

/-- Theorem stating the smallest possible area of a valid CheckeredRectangle is 40 -/
theorem smallest_area_is_40 :
  ∀ r : CheckeredRectangle, area r ≥ 40 ∧ ∃ r' : CheckeredRectangle, area r' = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_area_is_40_l3373_337337


namespace NUMINAMATH_CALUDE_triangular_prism_volume_l3373_337345

/-- Given a rectangle ABCD with dimensions as specified, prove that the volume of the
    triangular prism formed by folding is 594. -/
theorem triangular_prism_volume (A B C D P : ℝ × ℝ) : 
  let AB := 13 * Real.sqrt 3
  let BC := 12 * Real.sqrt 3
  -- A, B, C, D form a rectangle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = AB^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = BC^2 ∧
  -- P is the intersection of diagonals
  (A.1 - C.1) * (B.2 - D.2) = (A.2 - C.2) * (B.1 - D.1) ∧
  P.1 = (A.1 + C.1) / 2 ∧ P.2 = (A.2 + C.2) / 2 →
  -- Volume of the triangular prism after folding
  (1/6) * AB * BC * Real.sqrt (AB^2 + BC^2 - (AB^2 * BC^2) / (AB^2 + BC^2)) = 594 := by
  sorry


end NUMINAMATH_CALUDE_triangular_prism_volume_l3373_337345


namespace NUMINAMATH_CALUDE_sum_of_first_5n_integers_l3373_337390

theorem sum_of_first_5n_integers (n : ℕ) : 
  (4*n*(4*n+1))/2 = (n*(n+1))/2 + 210 → (5*n*(5*n+1))/2 = 465 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_5n_integers_l3373_337390


namespace NUMINAMATH_CALUDE_chinese_gcd_168_93_l3373_337314

def chinese_gcd (a b : ℕ) : ℕ := sorry

def chinese_gcd_sequence (a b : ℕ) : List (ℕ × ℕ) := sorry

theorem chinese_gcd_168_93 :
  let seq := chinese_gcd_sequence 168 93
  (57, 18) ∈ seq ∧
  (3, 18) ∈ seq ∧
  (3, 3) ∈ seq ∧
  (6, 9) ∉ seq ∧
  chinese_gcd 168 93 = 3 := by sorry

end NUMINAMATH_CALUDE_chinese_gcd_168_93_l3373_337314


namespace NUMINAMATH_CALUDE_sin_239_equals_neg_cos_31_l3373_337321

theorem sin_239_equals_neg_cos_31 (a : ℝ) (h : Real.cos (31 * π / 180) = a) :
  Real.sin (239 * π / 180) = -a := by
  sorry

end NUMINAMATH_CALUDE_sin_239_equals_neg_cos_31_l3373_337321


namespace NUMINAMATH_CALUDE_bees_flew_in_l3373_337343

/-- Given an initial number of bees in a hive and a final total number of bees after more fly in,
    this theorem proves that the number of bees that flew in is equal to the difference between
    the final total and the initial number. -/
theorem bees_flew_in (initial_bees final_bees : ℕ) : 
  initial_bees = 16 → final_bees = 24 → final_bees - initial_bees = 8 := by
  sorry

end NUMINAMATH_CALUDE_bees_flew_in_l3373_337343


namespace NUMINAMATH_CALUDE_max_k_inequality_l3373_337322

theorem max_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_condition : a^2 + b^2 + c^2 = 2*(a*b + b*c + c*a)) :
  ∃ (k : ℝ), k > 0 ∧ k = 2 ∧
  ∀ (k' : ℝ), k' > 0 →
    (1 / (k'*a*b + c^2) + 1 / (k'*b*c + a^2) + 1 / (k'*c*a + b^2) ≥ (k' + 3) / (a^2 + b^2 + c^2)) →
    k' ≤ k :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l3373_337322


namespace NUMINAMATH_CALUDE_lcm_of_72_108_2100_l3373_337348

theorem lcm_of_72_108_2100 : Nat.lcm (Nat.lcm 72 108) 2100 = 37800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_72_108_2100_l3373_337348


namespace NUMINAMATH_CALUDE_inequality_proof_l3373_337340

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * b) - Real.sqrt (c * d)| ≤ 2) : 
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3373_337340


namespace NUMINAMATH_CALUDE_expected_difference_l3373_337397

/-- The number of students in the school -/
def total_students : ℕ := 100

/-- The number of classes and teachers -/
def num_classes : ℕ := 5

/-- The distribution of students across classes -/
def class_sizes : List ℕ := [40, 40, 10, 5, 5]

/-- The expected number of students per class when choosing a teacher at random -/
def t : ℚ := (total_students : ℚ) / num_classes

/-- The expected number of students per class when choosing a student at random -/
def s : ℚ := (List.sum (List.map (fun x => x * x) class_sizes) : ℚ) / total_students

theorem expected_difference :
  t - s = -27/2 := by sorry

end NUMINAMATH_CALUDE_expected_difference_l3373_337397


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l3373_337373

-- Define the parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

-- Define the y-intercept
def y_intercept (d : ℝ) : Prop := parabola 0 = d

-- Define the x-intercepts
def x_intercepts (e f : ℝ) : Prop := parabola e = 0 ∧ parabola f = 0 ∧ e ≠ f

theorem parabola_intercepts_sum (d e f : ℝ) :
  y_intercept d → x_intercepts e f → d + e + f = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l3373_337373


namespace NUMINAMATH_CALUDE_car_payment_remainder_l3373_337394

theorem car_payment_remainder (part_payment : ℝ) (percentage : ℝ) (total_cost : ℝ) (remainder : ℝ) : 
  part_payment = 300 →
  percentage = 5 →
  part_payment = percentage / 100 * total_cost →
  remainder = total_cost - part_payment →
  remainder = 5700 := by
sorry

end NUMINAMATH_CALUDE_car_payment_remainder_l3373_337394


namespace NUMINAMATH_CALUDE_square_root_division_l3373_337382

theorem square_root_division (x : ℝ) : (Real.sqrt 5776 / x = 4) → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_l3373_337382


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l3373_337320

theorem p_or_q_is_true : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → Real.exp x ≥ 1) ∨ 
  (∃ x : ℝ, x^2 + x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l3373_337320


namespace NUMINAMATH_CALUDE_solutions_x_fourth_plus_81_l3373_337317

theorem solutions_x_fourth_plus_81 :
  {x : ℂ | x^4 + 81 = 0} = {3 + 3*I, -3 - 3*I, -3 + 3*I, 3 - 3*I} := by
  sorry

end NUMINAMATH_CALUDE_solutions_x_fourth_plus_81_l3373_337317


namespace NUMINAMATH_CALUDE_exchange_to_hundred_bills_l3373_337330

def twenty_bills : ℕ := 10
def ten_bills : ℕ := 8
def five_bills : ℕ := 4

def total_amount : ℕ := twenty_bills * 20 + ten_bills * 10 + five_bills * 5

theorem exchange_to_hundred_bills :
  (total_amount / 100 : ℕ) = 3 := by sorry

end NUMINAMATH_CALUDE_exchange_to_hundred_bills_l3373_337330


namespace NUMINAMATH_CALUDE_equation_solution_l3373_337377

theorem equation_solution (x y : ℚ) 
  (eq1 : 4 * x + y = 20) 
  (eq2 : x + 2 * y = 17) : 
  5 * x^2 + 18 * x * y + 5 * y^2 = 696 + 5/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3373_337377


namespace NUMINAMATH_CALUDE_book_purchases_l3373_337309

theorem book_purchases (people_A : ℕ) (people_B : ℕ) (people_both : ℕ) (people_only_B : ℕ) (people_only_A : ℕ) : 
  people_A = 2 * people_B →
  people_both = 500 →
  people_both = 2 * people_only_B →
  people_A = people_only_A + people_both →
  people_B = people_only_B + people_both →
  people_only_A = 1000 := by
sorry

end NUMINAMATH_CALUDE_book_purchases_l3373_337309


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l3373_337316

/-- The intersection point of the diagonals of a parallelogram with opposite vertices (2, -3) and (10, 9) is (6, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (10, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (6, 3) := by
sorry


end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l3373_337316


namespace NUMINAMATH_CALUDE_magnitude_of_w_l3373_337308

theorem magnitude_of_w (w : ℂ) (h : w^2 = -7 + 24*I) : Complex.abs w = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_w_l3373_337308


namespace NUMINAMATH_CALUDE_no_roots_implies_non_integer_l3373_337363

theorem no_roots_implies_non_integer (a b : ℝ) : 
  a ≠ b →
  (∀ x : ℝ, (x^2 + 20*a*x + 10*b) * (x^2 + 20*b*x + 10*a) ≠ 0) →
  ¬(∃ n : ℤ, 20*(b-a) = n) :=
by sorry

end NUMINAMATH_CALUDE_no_roots_implies_non_integer_l3373_337363


namespace NUMINAMATH_CALUDE_third_dimension_of_smaller_box_l3373_337389

/-- The length of the third dimension of the smaller box -/
def h : ℕ := sorry

/-- The volume of the larger box -/
def large_box_volume : ℕ := 12 * 14 * 16

/-- The volume of a single smaller box -/
def small_box_volume (h : ℕ) : ℕ := 3 * 7 * h

/-- The number of smaller boxes that fit into the larger box -/
def num_boxes : ℕ := 64

theorem third_dimension_of_smaller_box :
  (num_boxes * small_box_volume h ≤ large_box_volume) → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_third_dimension_of_smaller_box_l3373_337389


namespace NUMINAMATH_CALUDE_students_per_bus_l3373_337396

theorem students_per_bus (total_students : ℕ) (num_buses : ℕ) (students_in_cars : ℕ) :
  total_students = 375 →
  num_buses = 7 →
  students_in_cars = 4 →
  (total_students - students_in_cars) / num_buses = 53 := by
  sorry

end NUMINAMATH_CALUDE_students_per_bus_l3373_337396


namespace NUMINAMATH_CALUDE_line_properties_l3373_337350

-- Define the lines
def l1 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 1 = 0
def l2 (a x y : ℝ) : Prop := a * x + y = 1

-- Define perpendicularity
def perpendicular (a : ℝ) : Prop := Real.sqrt 3 * a + 1 = 0

-- Define angle of inclination
def angle_of_inclination (θ : ℝ) : Prop := θ = 2 * Real.pi / 3

-- Define distance from origin to line
def distance_to_origin (a : ℝ) (d : ℝ) : Prop := 
  d = 1 / Real.sqrt (a^2 + 1)

theorem line_properties (a : ℝ) :
  perpendicular a →
  angle_of_inclination (Real.arctan (-Real.sqrt 3)) ∧
  distance_to_origin a (Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l3373_337350


namespace NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l3373_337366

theorem max_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_two : a + b + c = 2) : 
  a^2 * b^3 * c^2 ≤ 128/2187 := by
sorry

theorem max_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ 
  a^2 * b^3 * c^2 > 128/2187 - ε := by
sorry

end NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l3373_337366


namespace NUMINAMATH_CALUDE_garden_area_ratio_l3373_337364

theorem garden_area_ratio :
  ∀ (L W : ℝ),
  L / W = 5 / 4 →
  L + W = 50 →
  (L * W) / (π * (W / 2)^2) = 5 / π :=
λ L W h1 h2 => by
  sorry

end NUMINAMATH_CALUDE_garden_area_ratio_l3373_337364


namespace NUMINAMATH_CALUDE_chocolate_problem_l3373_337395

theorem chocolate_problem (C S : ℝ) (N : ℕ) :
  (N * C = 77 * S) →  -- The total cost price equals the selling price of 77 chocolates
  ((S - C) / C = 4 / 7) →  -- The gain percent is 4/7
  N = 121 := by  -- Prove that N (number of chocolates bought at cost price) is 121
sorry

end NUMINAMATH_CALUDE_chocolate_problem_l3373_337395


namespace NUMINAMATH_CALUDE_card_selection_count_l3373_337374

/-- Represents a standard deck of cards with an additional special suit -/
structure Deck :=
  (standard_cards : Nat)
  (special_suit_cards : Nat)
  (ace_count : Nat)

/-- Represents the selection criteria for the cards -/
structure Selection :=
  (total_cards : Nat)
  (min_aces : Nat)
  (different_suits : Bool)

/-- Calculates the number of ways to choose cards according to the given criteria -/
def choose_cards (d : Deck) (s : Selection) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem card_selection_count (d : Deck) (s : Selection) :
  d.standard_cards = 52 →
  d.special_suit_cards = 13 →
  d.ace_count = 4 →
  s.total_cards = 5 →
  s.min_aces = 1 →
  s.different_suits = true →
  choose_cards d s = 114244 :=
sorry

end NUMINAMATH_CALUDE_card_selection_count_l3373_337374


namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l3373_337311

/-- Calculates the number of students in both band and chorus -/
def students_in_both (total : ℕ) (band : ℕ) (chorus : ℕ) (band_or_chorus : ℕ) : ℕ :=
  band + chorus - band_or_chorus

/-- Proves that the number of students in both band and chorus is 50 -/
theorem students_in_both_band_and_chorus :
  students_in_both 300 120 180 250 = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l3373_337311


namespace NUMINAMATH_CALUDE_solve_for_y_l3373_337331

theorem solve_for_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 25) (h4 : x / y = 36) : y = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3373_337331


namespace NUMINAMATH_CALUDE_rectangle_enclosure_count_l3373_337339

theorem rectangle_enclosure_count :
  let horizontal_lines : ℕ := 5
  let vertical_lines : ℕ := 5
  let choose_horizontal : ℕ := 2
  let choose_vertical : ℕ := 2
  (Nat.choose horizontal_lines choose_horizontal) * (Nat.choose vertical_lines choose_vertical) = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_enclosure_count_l3373_337339


namespace NUMINAMATH_CALUDE_smallest_number_l3373_337369

theorem smallest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -1) (hc : c = 1) (hd : d = -5) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c ∧ d ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3373_337369


namespace NUMINAMATH_CALUDE_dave_final_tickets_l3373_337399

def arcade_tickets (initial_tickets : ℕ) (candy_cost : ℕ) (beanie_cost : ℕ) (racing_game_win : ℕ) : ℕ :=
  let remaining_tickets := initial_tickets - (candy_cost + beanie_cost)
  let tickets_before_challenge := remaining_tickets + racing_game_win
  2 * tickets_before_challenge

theorem dave_final_tickets :
  arcade_tickets 11 3 5 10 = 26 := by
  sorry

end NUMINAMATH_CALUDE_dave_final_tickets_l3373_337399


namespace NUMINAMATH_CALUDE_sqrt_equality_condition_l3373_337328

theorem sqrt_equality_condition (a b c : ℕ+) :
  (Real.sqrt (a + b / (c ^ 2 : ℝ)) = a * Real.sqrt (b / (c ^ 2 : ℝ))) ↔ 
  (c ^ 2 : ℝ) = b * (a ^ 2 - 1) / a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_condition_l3373_337328


namespace NUMINAMATH_CALUDE_picture_difference_is_eight_l3373_337304

/-- The number of pictures Ralph has -/
def ralph_pictures : ℕ := 26

/-- The number of pictures Derrick has -/
def derrick_pictures : ℕ := 34

/-- The difference in the number of pictures between Derrick and Ralph -/
def picture_difference : ℕ := derrick_pictures - ralph_pictures

theorem picture_difference_is_eight : picture_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_picture_difference_is_eight_l3373_337304


namespace NUMINAMATH_CALUDE_max_value_theorem_l3373_337355

theorem max_value_theorem (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 3) :
  a^3 * b + b^3 * a ≤ 81/16 ∧ ∃ (a₀ b₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ a₀ + b₀ = 3 ∧ a₀^3 * b₀ + b₀^3 * a₀ = 81/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3373_337355


namespace NUMINAMATH_CALUDE_score_three_points_count_l3373_337379

/-- Represents the number of items to be matched -/
def n : ℕ := 4

/-- Represents the number of points awarded for a correct match -/
def correct_points : ℕ := 3

/-- Represents the number of points awarded for an incorrect match -/
def incorrect_points : ℕ := 0

/-- The total number of ways to match exactly one item correctly and the rest incorrectly -/
def ways_to_score_three_points : ℕ := n

theorem score_three_points_count :
  ways_to_score_three_points = n := by
  sorry

end NUMINAMATH_CALUDE_score_three_points_count_l3373_337379


namespace NUMINAMATH_CALUDE_savings_calculation_l3373_337365

theorem savings_calculation (total : ℚ) (furniture_fraction : ℚ) (tv_cost : ℚ) : 
  furniture_fraction = 3 / 4 →
  tv_cost = 300 →
  (1 - furniture_fraction) * total = tv_cost →
  total = 1200 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l3373_337365


namespace NUMINAMATH_CALUDE_factorization_equality_l3373_337384

theorem factorization_equality (a y : ℝ) : a^2 * y - 4 * y = y * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3373_337384


namespace NUMINAMATH_CALUDE_f_symmetry_iff_a_eq_one_l3373_337324

/-- The function f(x) defined as -|x-a| -/
def f (a : ℝ) (x : ℝ) : ℝ := -|x - a|

/-- Theorem stating that f(1+x) = f(1-x) for all x is equivalent to a = 1 -/
theorem f_symmetry_iff_a_eq_one (a : ℝ) :
  (∀ x, f a (1 + x) = f a (1 - x)) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_iff_a_eq_one_l3373_337324


namespace NUMINAMATH_CALUDE_student_count_l3373_337388

theorem student_count (avg_age : ℝ) (teacher_age : ℝ) (new_avg : ℝ) :
  avg_age = 20 →
  teacher_age = 40 →
  new_avg = avg_age + 1 →
  (∃ n : ℕ, n * avg_age + teacher_age = (n + 1) * new_avg ∧ n = 19) :=
by sorry

end NUMINAMATH_CALUDE_student_count_l3373_337388


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3373_337306

/-- Given vectors a and b, prove that a - b is perpendicular to b -/
theorem vector_perpendicular (a b : ℝ × ℝ) (h1 : a = (1, 0)) (h2 : b = (1/2, 1/2)) :
  (a - b) • b = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3373_337306


namespace NUMINAMATH_CALUDE_function_range_condition_l3373_337385

def f (a x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

theorem function_range_condition (a : ℝ) :
  (∀ x, ∃ y, f a x = y) ∧ (∀ y, ∃ x, f a x = y) ↔ a > 3 ∨ a < -1 :=
sorry

end NUMINAMATH_CALUDE_function_range_condition_l3373_337385


namespace NUMINAMATH_CALUDE_ball_probabilities_l3373_337391

/-- Represents a bag of balls -/
structure BallBag where
  white : ℕ
  red : ℕ

/-- The probability of drawing a red ball from the bag -/
def probRed (bag : BallBag) : ℚ :=
  bag.red / (bag.white + bag.red)

/-- The probability of drawing a white ball from the bag -/
def probWhite (bag : BallBag) : ℚ :=
  bag.white / (bag.white + bag.red)

/-- The bag after removing x red balls and adding x white balls -/
def adjustedBag (bag : BallBag) (x : ℕ) : BallBag :=
  { white := bag.white + x, red := bag.red - x }

theorem ball_probabilities (initialBag : BallBag) (x : ℕ) :
  initialBag.white = 8 ∧ initialBag.red = 12 →
  probRed initialBag = 3/5 ∧
  (probWhite (adjustedBag initialBag x) = 4/5 → x = 8) := by
  sorry

#check ball_probabilities

end NUMINAMATH_CALUDE_ball_probabilities_l3373_337391


namespace NUMINAMATH_CALUDE_adlai_has_two_dogs_l3373_337386

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The total number of animal legs -/
def total_legs : ℕ := 10

/-- The number of chickens Adlai has -/
def num_chickens : ℕ := 1

/-- Theorem stating that Adlai has 2 dogs -/
theorem adlai_has_two_dogs : 
  ∃ (num_dogs : ℕ), num_dogs * dog_legs + num_chickens * chicken_legs = total_legs ∧ num_dogs = 2 :=
sorry

end NUMINAMATH_CALUDE_adlai_has_two_dogs_l3373_337386


namespace NUMINAMATH_CALUDE_least_integer_b_for_quadratic_range_l3373_337353

theorem least_integer_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -10) ↔ b ≤ -10 :=
sorry

end NUMINAMATH_CALUDE_least_integer_b_for_quadratic_range_l3373_337353


namespace NUMINAMATH_CALUDE_total_snakes_l3373_337351

theorem total_snakes (boa_constrictors python rattlesnakes : ℕ) : 
  boa_constrictors = 40 →
  python = 3 * boa_constrictors →
  rattlesnakes = 40 →
  boa_constrictors + python + rattlesnakes = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_snakes_l3373_337351


namespace NUMINAMATH_CALUDE_expression_simplification_l3373_337302

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x + 3)^2 + (x + 2)*(x - 2) - x*(x + 6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3373_337302


namespace NUMINAMATH_CALUDE_range_of_a_l3373_337335

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - 1 / Real.exp x

theorem range_of_a (a : ℝ) :
  (f (a - 1) + f (2 * a^2) ≤ 0) → (-1 ≤ a ∧ a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3373_337335


namespace NUMINAMATH_CALUDE_small_room_four_painters_l3373_337360

/-- Represents the number of work-days required for a given number of painters to complete a room -/
def work_days (painters : ℕ) (room_size : ℝ) : ℝ := sorry

theorem small_room_four_painters 
  (large_room_size small_room_size : ℝ)
  (h1 : work_days 5 large_room_size = 2)
  (h2 : small_room_size = large_room_size / 2)
  : work_days 4 small_room_size = 1.25 := by sorry

end NUMINAMATH_CALUDE_small_room_four_painters_l3373_337360


namespace NUMINAMATH_CALUDE_assignment_schemes_with_girl_count_l3373_337346

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The number of people to be selected -/
def num_selected : ℕ := 3

/-- The number of tasks to be assigned -/
def num_tasks : ℕ := 3

/-- The function to calculate the number of assignment schemes -/
def assignment_schemes (b g s t : ℕ) : ℕ :=
  Nat.descFactorial (b + g) s - Nat.descFactorial b s

/-- Theorem stating that the number of assignment schemes with at least one girl is 186 -/
theorem assignment_schemes_with_girl_count :
  assignment_schemes num_boys num_girls num_selected num_tasks = 186 := by
  sorry

end NUMINAMATH_CALUDE_assignment_schemes_with_girl_count_l3373_337346


namespace NUMINAMATH_CALUDE_maria_has_four_l3373_337371

/-- Represents a player in the card game -/
inductive Player : Type
  | Maria
  | Josh
  | Laura
  | Neil
  | Eva

/-- The score of each player -/
def score (p : Player) : ℕ :=
  match p with
  | Player.Maria => 13
  | Player.Josh => 15
  | Player.Laura => 9
  | Player.Neil => 18
  | Player.Eva => 19

/-- The set of all possible cards -/
def cards : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

/-- Predicate to check if a pair of cards is valid for a player -/
def validCardPair (p : Player) (c1 c2 : ℕ) : Prop :=
  c1 ∈ cards ∧ c2 ∈ cards ∧ c1 + c2 = score p ∧ c1 ≠ c2

/-- Theorem stating that Maria must have received card number 4 -/
theorem maria_has_four :
  ∃ (c : ℕ), c ∈ cards ∧ c ≠ 4 ∧ validCardPair Player.Maria 4 c ∧
  (∀ (p : Player), p ≠ Player.Maria → ¬∃ (c1 c2 : ℕ), (c1 = 4 ∨ c2 = 4) ∧ validCardPair p c1 c2) :=
sorry

end NUMINAMATH_CALUDE_maria_has_four_l3373_337371


namespace NUMINAMATH_CALUDE_total_apples_for_bobbing_l3373_337344

/-- The number of apples in each bucket for bobbing apples. -/
def apples_per_bucket : ℕ := 9

/-- The number of buckets Mrs. Walker needs. -/
def number_of_buckets : ℕ := 7

/-- Theorem: Mrs. Walker has 63 apples for bobbing for apples. -/
theorem total_apples_for_bobbing : 
  apples_per_bucket * number_of_buckets = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_for_bobbing_l3373_337344


namespace NUMINAMATH_CALUDE_triangle_side_indeterminate_l3373_337332

/-- Given a triangle ABC with AB = 3 and AC = 2, the length of BC cannot be uniquely determined. -/
theorem triangle_side_indeterminate (A B C : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d A B = 3) → (d A C = 2) → 
  ¬∃! x : ℝ, d B C = x :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_indeterminate_l3373_337332
