import Mathlib

namespace NUMINAMATH_CALUDE_cos_5pi_4_plus_x_l1646_164655

theorem cos_5pi_4_plus_x (x : ℝ) (h : Real.sin (π/4 - x) = -1/5) : 
  Real.cos (5*π/4 + x) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_5pi_4_plus_x_l1646_164655


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_13_l1646_164621

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 3000000 + 100000 * a + 100 * b + 3

def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 13 * k

theorem seven_digit_divisible_by_13 :
  {n : ℕ | is_valid_number n ∧ is_divisible_by_13 n} =
  {3000803, 3020303, 3030703, 3050203, 3060603, 3080103, 3090503} :=
sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_13_l1646_164621


namespace NUMINAMATH_CALUDE_same_solution_d_value_l1646_164673

theorem same_solution_d_value (x : ℝ) (d : ℝ) : 
  (3 * x + 8 = 4) ∧ (d * x - 15 = -5) → d = -7.5 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_d_value_l1646_164673


namespace NUMINAMATH_CALUDE_absolute_value_problem_l1646_164680

theorem absolute_value_problem (x y : ℝ) 
  (hx : |x| = 3) 
  (hy : |y| = 2) :
  (x < y → x - y = -5 ∨ x - y = -1) ∧
  (x * y > 0 → x + y = 5) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l1646_164680


namespace NUMINAMATH_CALUDE_initial_crayons_l1646_164602

theorem initial_crayons (C : ℕ) : 
  (C : ℚ) * (3/4) * (1/2) = 18 → C = 48 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_l1646_164602


namespace NUMINAMATH_CALUDE_janet_farmland_acreage_l1646_164667

/-- Represents Janet's farm and fertilizer production system -/
structure FarmSystem where
  horses : ℕ
  fertilizer_per_horse : ℕ
  fertilizer_per_acre : ℕ
  acres_spread_per_day : ℕ
  days_to_fertilize : ℕ

/-- Calculates the total acreage of Janet's farmland -/
def total_acreage (farm : FarmSystem) : ℕ :=
  farm.acres_spread_per_day * farm.days_to_fertilize

/-- Theorem: Janet's farmland is 100 acres given the specified conditions -/
theorem janet_farmland_acreage :
  let farm := FarmSystem.mk 80 5 400 4 25
  total_acreage farm = 100 := by
  sorry


end NUMINAMATH_CALUDE_janet_farmland_acreage_l1646_164667


namespace NUMINAMATH_CALUDE_triangle_properties_l1646_164613

/-- Given a triangle with sides 8, 15, and 17, prove it's a right triangle
    and find the longest side of a similar triangle with perimeter 160 -/
theorem triangle_properties (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  (a^2 + b^2 = c^2) ∧ 
  (∃ (x : ℝ), x * (a + b + c) = 160 ∧ x * c = 68) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1646_164613


namespace NUMINAMATH_CALUDE_factorization_sum_l1646_164601

theorem factorization_sum (a b c d e f g : ℤ) :
  (∀ x y : ℝ, 16 * x^4 - 81 * y^4 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y)) →
  a + b + c + d + e + f + g = 17 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l1646_164601


namespace NUMINAMATH_CALUDE_p_and_q_true_iff_not_p_or_not_q_false_l1646_164610

theorem p_and_q_true_iff_not_p_or_not_q_false (p q : Prop) :
  (p ∧ q) ↔ ¬(¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_iff_not_p_or_not_q_false_l1646_164610


namespace NUMINAMATH_CALUDE_equation_three_roots_l1646_164671

theorem equation_three_roots :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
  (∀ x : ℝ, x ∈ s ↔ Real.sqrt (9 - x) = x^2 * Real.sqrt (9 - x)) :=
by sorry

end NUMINAMATH_CALUDE_equation_three_roots_l1646_164671


namespace NUMINAMATH_CALUDE_vectors_form_basis_l1646_164664

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : 
  LinearIndependent ℝ (fun i => if i = 0 then e₁ else e₂) ∧ 
  Submodule.span ℝ {e₁, e₂} = ⊤ :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l1646_164664


namespace NUMINAMATH_CALUDE_cube_sum_difference_l1646_164687

/-- Represents a face of a cube --/
inductive Face
| One
| Two
| Three
| Four
| Five
| Six

/-- A single small cube with numbered faces --/
structure SmallCube where
  faces : List Face
  face_count : faces.length = 6
  opposite_faces : 
    (Face.One ∈ faces ↔ Face.Two ∈ faces) ∧
    (Face.Three ∈ faces ↔ Face.Five ∈ faces) ∧
    (Face.Four ∈ faces ↔ Face.Six ∈ faces)

/-- The large 2×2×2 cube composed of small cubes --/
structure LargeCube where
  small_cubes : List SmallCube
  cube_count : small_cubes.length = 8

/-- The sum of numbers on the outer surface of the large cube --/
def outer_surface_sum (lc : LargeCube) : ℕ := sorry

/-- The maximum possible sum of numbers on the outer surface --/
def max_sum (lc : LargeCube) : ℕ := sorry

/-- The minimum possible sum of numbers on the outer surface --/
def min_sum (lc : LargeCube) : ℕ := sorry

/-- The main theorem to prove --/
theorem cube_sum_difference (lc : LargeCube) : 
  max_sum lc - min_sum lc = 24 := by sorry

end NUMINAMATH_CALUDE_cube_sum_difference_l1646_164687


namespace NUMINAMATH_CALUDE_total_chocolates_l1646_164697

theorem total_chocolates (bags : ℕ) (chocolates_per_bag : ℕ) 
  (h1 : bags = 20) (h2 : chocolates_per_bag = 156) :
  bags * chocolates_per_bag = 3120 :=
by sorry

end NUMINAMATH_CALUDE_total_chocolates_l1646_164697


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_fraction_multiplication_main_theorem_l1646_164641

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d :=
by sorry

theorem fraction_multiplication (a b c : ℚ) (n : ℕ) :
  (a * b * c : ℚ) * n = (n : ℚ) / ((1 / a) * (1 / b) * (1 / c)) :=
by sorry

theorem main_theorem : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 7 : ℚ) * 126 = 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_fraction_multiplication_main_theorem_l1646_164641


namespace NUMINAMATH_CALUDE_garden_dimensions_l1646_164669

/-- Represents a rectangular garden with walkways -/
structure Garden where
  L : ℝ  -- Length of the garden
  W : ℝ  -- Width of the garden
  w : ℝ  -- Width of the walkways
  h_L_gt_W : L > W  -- Length is greater than width

/-- The theorem representing the garden problem -/
theorem garden_dimensions (g : Garden) 
  (h1 : g.w * g.L = 228)  -- First walkway area
  (h2 : g.w * g.W = 117)  -- Second walkway area
  (h3 : g.w * g.L - g.w^2 = 219)  -- Third walkway area
  (h4 : g.w * g.L - (g.w * g.L - g.w^2) = g.w^2)  -- Difference between first and third walkway areas
  : g.L = 76 ∧ g.W = 42 ∧ g.w = 3 := by
  sorry

end NUMINAMATH_CALUDE_garden_dimensions_l1646_164669


namespace NUMINAMATH_CALUDE_simplify_fraction_l1646_164662

theorem simplify_fraction : (90 : ℚ) / 8100 = 1 / 90 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1646_164662


namespace NUMINAMATH_CALUDE_intersection_point_on_lines_unique_intersection_l1646_164628

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (39/17, 53/17)

/-- First line equation: 8x - 3y = 9 -/
def line1 (x y : ℚ) : Prop := 8*x - 3*y = 9

/-- Second line equation: 6x + 2y = 20 -/
def line2 (x y : ℚ) : Prop := 6*x + 2*y = 20

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_lines : 
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_on_lines_unique_intersection_l1646_164628


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l1646_164672

theorem pizza_slices_remaining (total_slices : ℕ) (given_to_first_group : ℕ) (given_to_second_group : ℕ) :
  total_slices = 8 →
  given_to_first_group = 3 →
  given_to_second_group = 4 →
  total_slices - (given_to_first_group + given_to_second_group) = 1 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l1646_164672


namespace NUMINAMATH_CALUDE_peters_vacation_savings_l1646_164660

/-- Peter's vacation savings problem -/
theorem peters_vacation_savings 
  (current_savings : ℕ) 
  (monthly_savings : ℕ) 
  (months_to_wait : ℕ) 
  (h1 : current_savings = 2900)
  (h2 : monthly_savings = 700)
  (h3 : months_to_wait = 3) :
  current_savings + monthly_savings * months_to_wait = 5000 :=
by sorry

end NUMINAMATH_CALUDE_peters_vacation_savings_l1646_164660


namespace NUMINAMATH_CALUDE_smaller_number_l1646_164606

theorem smaller_number (a b d x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = 2 * a / (3 * b) → x + 2 * y = d →
  min x y = a * d / (2 * a + 3 * b) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_l1646_164606


namespace NUMINAMATH_CALUDE_rectangular_field_fence_l1646_164633

theorem rectangular_field_fence (L W : ℝ) : 
  L > 0 ∧ W > 0 →  -- Positive dimensions
  L * W = 210 →    -- Area condition
  L + 2 * W = 41 → -- Fencing condition
  L = 21 :=        -- Conclusion: uncovered side length
by sorry

end NUMINAMATH_CALUDE_rectangular_field_fence_l1646_164633


namespace NUMINAMATH_CALUDE_jimmy_water_consumption_l1646_164677

/-- Represents the amount of water Jimmy drinks each time in ounces -/
def water_per_time (times_per_day : ℕ) (days : ℕ) (total_gallons : ℚ) (ounce_to_gallon : ℚ) : ℚ :=
  (total_gallons / ounce_to_gallon) / (times_per_day * days)

/-- Theorem stating that Jimmy drinks 8 ounces of water each time -/
theorem jimmy_water_consumption :
  water_per_time 8 5 (5/2) (1/128) = 8 := by
sorry

end NUMINAMATH_CALUDE_jimmy_water_consumption_l1646_164677


namespace NUMINAMATH_CALUDE_two_sector_area_l1646_164699

/-- The area of a figure formed by two sectors of a circle -/
theorem two_sector_area (r : ℝ) (θ : ℝ) : 
  r = 15 → θ = 90 → 2 * (θ / 360) * π * r^2 = 112.5 * π := by sorry

end NUMINAMATH_CALUDE_two_sector_area_l1646_164699


namespace NUMINAMATH_CALUDE_value_added_to_reach_new_average_l1646_164635

theorem value_added_to_reach_new_average (n : ℕ) (initial_avg final_avg : ℝ) (h1 : n = 15) (h2 : initial_avg = 40) (h3 : final_avg = 55) :
  ∃ x : ℝ, (n : ℝ) * initial_avg + n * x = n * final_avg ∧ x = 15 :=
by sorry

end NUMINAMATH_CALUDE_value_added_to_reach_new_average_l1646_164635


namespace NUMINAMATH_CALUDE_vehicle_value_last_year_l1646_164624

theorem vehicle_value_last_year 
  (value_this_year : ℝ) 
  (value_ratio : ℝ) 
  (h1 : value_this_year = 16000)
  (h2 : value_ratio = 0.8)
  (h3 : value_this_year = value_ratio * value_last_year) :
  value_last_year = 20000 :=
by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_last_year_l1646_164624


namespace NUMINAMATH_CALUDE_max_value_constraint_l1646_164648

theorem max_value_constraint (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) 
  (h1 : x + y - 3 ≤ 0) (h2 : 2 * x + y - 4 ≥ 0) : 
  2 * x + 3 * y ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1646_164648


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l1646_164604

theorem quiz_competition_participants (initial_participants : ℕ) : 
  (initial_participants : ℝ) * 0.4 * 0.25 = 15 → initial_participants = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l1646_164604


namespace NUMINAMATH_CALUDE_officials_selection_count_l1646_164630

/-- Represents the number of ways to choose officials from a club --/
def choose_officials (total_members : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  girls * boys * (boys - 1)

/-- Theorem: The number of ways to choose officials under given conditions is 1716 --/
theorem officials_selection_count :
  choose_officials 25 12 13 = 1716 := by
  sorry

end NUMINAMATH_CALUDE_officials_selection_count_l1646_164630


namespace NUMINAMATH_CALUDE_real_roots_iff_k_le_2_m_eq_3_and_other_root_4_l1646_164619

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : Prop := x^2 - 4*x + 2*k = 0

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop := ∃ x : ℝ, quadratic k x

-- Define the second quadratic equation
def quadratic2 (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + 3*m - 1 = 0

-- Theorem for part 1
theorem real_roots_iff_k_le_2 :
  ∀ k : ℝ, has_real_roots k ↔ k ≤ 2 :=
sorry

-- Theorem for part 2
theorem m_eq_3_and_other_root_4 :
  ∃ x : ℝ, quadratic 2 x ∧ quadratic2 3 x ∧ quadratic2 3 4 :=
sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_le_2_m_eq_3_and_other_root_4_l1646_164619


namespace NUMINAMATH_CALUDE_arrangements_count_l1646_164603

/-- Represents the number of teachers -/
def num_teachers : ℕ := 5

/-- Represents the number of days -/
def num_days : ℕ := 4

/-- Represents the number of teachers required on Monday -/
def teachers_on_monday : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of arrangements for the given scenario -/
def num_arrangements : ℕ := 
  (choose num_teachers teachers_on_monday) * (Nat.factorial (num_teachers - teachers_on_monday))

/-- Theorem stating that the number of arrangements is 60 -/
theorem arrangements_count : num_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l1646_164603


namespace NUMINAMATH_CALUDE_function_properties_l1646_164609

/-- Given a function f with the specified properties, prove its simplified form and extrema -/
theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sqrt 3 * Real.cos (ω * x) * Real.sin (ω * x) - 2 * (Real.cos (ω * x))^2 + 1
  (∀ x, f (x + π) = f x) →  -- smallest positive period is π
  (∃ g : ℝ → ℝ, ∀ x, f x = 2 * Real.sin (2 * x - π / 6)) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≥ -1) ∧
  (f 0 = -1) ∧
  (f (π / 3) = 2) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1646_164609


namespace NUMINAMATH_CALUDE_store_price_reduction_l1646_164698

theorem store_price_reduction (original_price : ℝ) (first_reduction : ℝ) :
  first_reduction > 0 →
  first_reduction < 100 →
  let second_reduction := 10
  let final_price_percentage := 82.8
  (original_price * (1 - first_reduction / 100) * (1 - second_reduction / 100)) / original_price * 100 = final_price_percentage →
  first_reduction = 8 := by
sorry

end NUMINAMATH_CALUDE_store_price_reduction_l1646_164698


namespace NUMINAMATH_CALUDE_lyle_notebook_cost_l1646_164614

/-- The cost of a pen in dollars -/
def pen_cost : ℝ := 1.50

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := 3 * pen_cost

/-- The number of notebooks Lyle wants to buy -/
def num_notebooks : ℕ := 4

/-- The total cost of notebooks Lyle will pay -/
def total_cost : ℝ := num_notebooks * notebook_cost

theorem lyle_notebook_cost : total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_lyle_notebook_cost_l1646_164614


namespace NUMINAMATH_CALUDE_student_number_problem_l1646_164616

theorem student_number_problem : ∃ x : ℤ, 2 * x - 138 = 110 ∧ x = 124 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1646_164616


namespace NUMINAMATH_CALUDE_unique_number_with_divisor_properties_l1646_164686

theorem unique_number_with_divisor_properties :
  ∀ (N p q r : ℕ) (α β γ : ℕ),
    (∃ (h_prime_p : Nat.Prime p) (h_prime_q : Nat.Prime q) (h_prime_r : Nat.Prime r),
      N = p^α * q^β * r^γ ∧
      p * q - r = 3 ∧
      p * r - q = 9 ∧
      (Nat.divisors (N / p)).card = (Nat.divisors N).card - 20 ∧
      (Nat.divisors (N / q)).card = (Nat.divisors N).card - 12 ∧
      (Nat.divisors (N / r)).card = (Nat.divisors N).card - 15) →
    N = 857500 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_divisor_properties_l1646_164686


namespace NUMINAMATH_CALUDE_circle_center_l1646_164685

/-- The center of a circle with equation x^2 - 8x + y^2 - 4y = 16 is (4, 2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 8*x + y^2 - 4*y = 16) → 
  (∃ r : ℝ, (x - 4)^2 + (y - 2)^2 = r^2) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l1646_164685


namespace NUMINAMATH_CALUDE_no_solution_implies_non_positive_product_l1646_164663

theorem no_solution_implies_non_positive_product (a b : ℝ) : 
  (∀ x : ℝ, (3*a + 8*b)*x + 7 ≠ 0) → a*b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_non_positive_product_l1646_164663


namespace NUMINAMATH_CALUDE_vegan_meal_clients_l1646_164638

theorem vegan_meal_clients (total : ℕ) (kosher : ℕ) (both : ℕ) (neither : ℕ) :
  total = 30 ∧ kosher = 8 ∧ both = 3 ∧ neither = 18 →
  ∃ vegan : ℕ, vegan = 10 ∧ vegan + (kosher - both) + neither = total :=
by sorry

end NUMINAMATH_CALUDE_vegan_meal_clients_l1646_164638


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1646_164607

theorem quadratic_equation_solution :
  ∀ x : ℝ, x * (x - 3) = 0 ↔ x = 0 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1646_164607


namespace NUMINAMATH_CALUDE_present_ages_sum_l1646_164658

theorem present_ages_sum (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : 4 * a = 3 * b) : 
  a + b = 35 := by
  sorry

end NUMINAMATH_CALUDE_present_ages_sum_l1646_164658


namespace NUMINAMATH_CALUDE_birthday_product_difference_l1646_164693

theorem birthday_product_difference (n : ℕ) (h : n = 7) : (n + 1)^2 - n^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_birthday_product_difference_l1646_164693


namespace NUMINAMATH_CALUDE_complement_of_union_l1646_164645

def U : Set Nat := {1, 2, 3, 4, 5}
def P : Set Nat := {1, 2, 3}
def Q : Set Nat := {2, 3, 4}

theorem complement_of_union :
  (U \ (P ∪ Q)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1646_164645


namespace NUMINAMATH_CALUDE_tax_rate_problem_l1646_164652

/-- The tax rate problem in Country X -/
theorem tax_rate_problem (income : ℝ) (total_tax : ℝ) (tax_rate_above_40k : ℝ) :
  income = 50000 →
  total_tax = 8000 →
  tax_rate_above_40k = 0.2 →
  ∃ (tax_rate_below_40k : ℝ),
    tax_rate_below_40k * 40000 + tax_rate_above_40k * (income - 40000) = total_tax ∧
    tax_rate_below_40k = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_problem_l1646_164652


namespace NUMINAMATH_CALUDE_workday_meeting_percentage_l1646_164608

-- Define the workday duration in minutes
def workday_minutes : ℕ := 10 * 60

-- Define the duration of the first meeting
def first_meeting_duration : ℕ := 30

-- Define the duration of the second meeting
def second_meeting_duration : ℕ := 2 * first_meeting_duration

-- Define the duration of the third meeting
def third_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration

-- Define the total time spent in meetings
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

-- Theorem to prove
theorem workday_meeting_percentage : 
  (total_meeting_time : ℚ) / workday_minutes * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_workday_meeting_percentage_l1646_164608


namespace NUMINAMATH_CALUDE_olivia_picked_16_pieces_l1646_164612

/-- The number of pieces of paper Olivia picked up -/
def olivia_pieces : ℕ := 19 - 3

/-- The number of pieces of paper Edward picked up -/
def edward_pieces : ℕ := 3

/-- The total number of pieces of paper picked up -/
def total_pieces : ℕ := 19

theorem olivia_picked_16_pieces :
  olivia_pieces = 16 ∧ olivia_pieces + edward_pieces = total_pieces :=
sorry

end NUMINAMATH_CALUDE_olivia_picked_16_pieces_l1646_164612


namespace NUMINAMATH_CALUDE_leftover_pie_share_l1646_164683

theorem leftover_pie_share (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 12 / 13 → num_people = 4 → total_pie / num_people = 3 / 13 := by
  sorry

end NUMINAMATH_CALUDE_leftover_pie_share_l1646_164683


namespace NUMINAMATH_CALUDE_circle_equation_l1646_164668

theorem circle_equation (x y : ℝ) : 
  (∃ c : ℝ, x^2 + (y - c)^2 = 1 ∧ 1^2 + (2 - c)^2 = 1) → 
  x^2 + (y - 2)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l1646_164668


namespace NUMINAMATH_CALUDE_move_point_right_l1646_164679

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally by a given distance -/
def moveHorizontally (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

theorem move_point_right : 
  let A : Point := { x := -2, y := 3 }
  let movedA : Point := moveHorizontally A 2
  movedA = { x := 0, y := 3 } := by
  sorry


end NUMINAMATH_CALUDE_move_point_right_l1646_164679


namespace NUMINAMATH_CALUDE_no_solution_exists_l1646_164636

theorem no_solution_exists : ¬ ∃ (a b c d : ℤ), a^4 + b^4 + c^4 + 2016 = 10*d := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1646_164636


namespace NUMINAMATH_CALUDE_equation_solution_l1646_164627

theorem equation_solution :
  ∃! y : ℚ, 7 * (4 * y + 3) - 3 = -3 * (2 - 5 * y) ∧ y = -24 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1646_164627


namespace NUMINAMATH_CALUDE_paving_stone_length_l1646_164632

/-- Given a rectangular courtyard and paving stones with specific properties,
    prove that the length of each paving stone is 4 meters. -/
theorem paving_stone_length
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (num_stones : ℕ)
  (stone_width : ℝ)
  (h1 : courtyard_length = 40)
  (h2 : courtyard_width = 20)
  (h3 : num_stones = 100)
  (h4 : stone_width = 2)
  : ∃ (stone_length : ℝ), stone_length = 4 ∧
    courtyard_length * courtyard_width = ↑num_stones * stone_length * stone_width :=
by
  sorry


end NUMINAMATH_CALUDE_paving_stone_length_l1646_164632


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1646_164629

theorem inverse_variation_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ k : ℝ, k > 0 ∧ ∀ x y, x^3 * y = k) 
  (h4 : 2^3 * 8 = x^3 * 512) : x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1646_164629


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1646_164643

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 80 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 200 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1646_164643


namespace NUMINAMATH_CALUDE_course_selection_schemes_l1646_164611

def number_of_courses : ℕ := 7
def courses_to_choose : ℕ := 4

def total_combinations : ℕ := Nat.choose number_of_courses courses_to_choose

def forbidden_combinations : ℕ := Nat.choose (number_of_courses - 2) (courses_to_choose - 2)

theorem course_selection_schemes :
  total_combinations - forbidden_combinations = 25 := by sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l1646_164611


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1646_164634

theorem min_value_sum_reciprocals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 4) :
  (1 / a + 4 / b + 9 / c) ≥ 9 ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ + b₀ + c₀ = 4 ∧ 1 / a₀ + 4 / b₀ + 9 / c₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1646_164634


namespace NUMINAMATH_CALUDE_unique_cube_root_l1646_164684

theorem unique_cube_root (M : ℕ+) : 18^3 * 50^3 = 30^3 * M^3 ↔ M = 30 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_root_l1646_164684


namespace NUMINAMATH_CALUDE_root_properties_l1646_164642

theorem root_properties : 
  (∃ x : ℝ, x^3 = -9 ∧ x = -3) ∧ 
  (∀ y : ℝ, y^2 = 9 ↔ y = 3 ∨ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_root_properties_l1646_164642


namespace NUMINAMATH_CALUDE_profit_and_max_profit_l1646_164644

def cost_price : ℝ := 12
def initial_price : ℝ := 20
def initial_quantity : ℝ := 240
def quantity_increase_rate : ℝ := 40

def profit (x : ℝ) : ℝ :=
  (initial_price - cost_price - x) * (initial_quantity + quantity_increase_rate * x)

theorem profit_and_max_profit :
  (∃ x : ℝ, profit x = 1920 ∧ x = 2) ∧
  (∃ x : ℝ, ∀ y : ℝ, profit y ≤ profit x ∧ x = 4) ∧
  (∃ x : ℝ, profit x = 2560 ∧ x = 4) := by sorry

end NUMINAMATH_CALUDE_profit_and_max_profit_l1646_164644


namespace NUMINAMATH_CALUDE_kathleen_bottle_caps_l1646_164615

/-- The number of times Kathleen went to the store last month -/
def store_visits : ℕ := 5

/-- The number of bottle caps Kathleen buys each time she goes to the store -/
def bottle_caps_per_visit : ℕ := 5

/-- The total number of bottle caps Kathleen bought last month -/
def total_bottle_caps : ℕ := store_visits * bottle_caps_per_visit

theorem kathleen_bottle_caps : total_bottle_caps = 25 := by
  sorry

end NUMINAMATH_CALUDE_kathleen_bottle_caps_l1646_164615


namespace NUMINAMATH_CALUDE_kylie_coins_l1646_164694

theorem kylie_coins (initial_coins : ℕ) (received_coins1 : ℕ) (received_coins2 : ℕ) (given_away : ℕ) :
  initial_coins = 15 →
  received_coins1 = 13 →
  received_coins2 = 8 →
  given_away = 21 →
  initial_coins + received_coins1 + received_coins2 - given_away = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_kylie_coins_l1646_164694


namespace NUMINAMATH_CALUDE_largest_integer_below_root_l1646_164646

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 6

theorem largest_integer_below_root :
  ∃ (x₀ : ℝ), f x₀ = 0 ∧
  (∀ x > 0, x < x₀ → f x < 0) ∧
  (∀ x > x₀, f x > 0) ∧
  (∀ n : ℤ, (n : ℝ) ≤ x₀ → n ≤ 4) ∧
  ((4 : ℝ) ≤ x₀) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_below_root_l1646_164646


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1646_164640

open Set

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {y | ∃ x, y = x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {y | 0 ≤ y} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1646_164640


namespace NUMINAMATH_CALUDE_simplify_expression_l1646_164695

theorem simplify_expression (a : ℝ) : (3 * a)^2 * a^5 = 9 * a^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1646_164695


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l1646_164618

/-- Given a circle with original radius r₀ > 0, prove that when the radius is
    increased by 50%, the ratio of the new circumference to the new area
    is equal to 4 / (3r₀). -/
theorem circle_ratio_after_increase (r₀ : ℝ) (h : r₀ > 0) :
  let new_radius := 1.5 * r₀
  let new_circumference := 2 * Real.pi * new_radius
  let new_area := Real.pi * new_radius ^ 2
  new_circumference / new_area = 4 / (3 * r₀) := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l1646_164618


namespace NUMINAMATH_CALUDE_square_difference_equality_l1646_164653

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1646_164653


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1646_164689

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 10 = 16 → a 4 + a 6 + a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1646_164689


namespace NUMINAMATH_CALUDE_quadratic_minimum_minimizing_x_value_l1646_164690

/-- The quadratic function f(x) = x^2 - 10x + 24 attains its minimum value when x = 5. -/
theorem quadratic_minimum : ∀ x : ℝ, (x^2 - 10*x + 24) ≥ (5^2 - 10*5 + 24) := by
  sorry

/-- The value of x that minimizes the quadratic function f(x) = x^2 - 10x + 24 is 5. -/
theorem minimizing_x_value : ∃! x : ℝ, ∀ y : ℝ, (x^2 - 10*x + 24) ≤ (y^2 - 10*y + 24) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_minimizing_x_value_l1646_164690


namespace NUMINAMATH_CALUDE_inverse_proportion_points_l1646_164678

def inverse_proportion (x y : ℝ) : Prop := y = -4 / x

theorem inverse_proportion_points :
  inverse_proportion (-2) 2 ∧
  ¬ inverse_proportion 1 4 ∧
  ¬ inverse_proportion (-2) (-2) ∧
  ¬ inverse_proportion (-4) (-1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_l1646_164678


namespace NUMINAMATH_CALUDE_solution_set_of_f_gt_zero_l1646_164665

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Theorem statement
theorem solution_set_of_f_gt_zero
  (h_even : is_even f)
  (h_monotone : is_monotone_increasing_on_nonneg f)
  (h_f_one : f 1 = 0) :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_f_gt_zero_l1646_164665


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1646_164605

theorem quadratic_always_positive 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hseq : b / a = c / b) : 
  ∀ x : ℝ, a * x^2 + b * x + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1646_164605


namespace NUMINAMATH_CALUDE_equation_solution_l1646_164661

theorem equation_solution :
  ∃ x : ℚ, x + 5/6 = 7/18 - 2/9 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1646_164661


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1646_164639

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse length
  A = (h^2) / 4 →        -- area formula for isosceles right triangle
  A = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1646_164639


namespace NUMINAMATH_CALUDE_daria_concert_money_l1646_164600

theorem daria_concert_money (total_tickets : ℕ) (ticket_cost : ℕ) (current_savings : ℕ) :
  total_tickets = 4 →
  ticket_cost = 90 →
  current_savings = 189 →
  total_tickets * ticket_cost - current_savings = 171 :=
by sorry

end NUMINAMATH_CALUDE_daria_concert_money_l1646_164600


namespace NUMINAMATH_CALUDE_range_of_a_l1646_164691

-- Define the propositions p and q
def p (x : ℝ) : Prop := |4 - x| ≤ 6
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≥ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, ¬(p x) → q x a) →
  (∃ x : ℝ, p x ∧ q x a) →
  (0 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1646_164691


namespace NUMINAMATH_CALUDE_hua_luogeng_uses_golden_ratio_l1646_164654

-- Define the possible methods for optimal selection
inductive OptimalSelectionMethod
  | GoldenRatio
  | Mean
  | Mode
  | Median

-- Define Hua Luogeng's optimal selection method
def huaLuogengMethod : OptimalSelectionMethod := OptimalSelectionMethod.GoldenRatio

-- Theorem stating that Hua Luogeng's method uses the golden ratio
theorem hua_luogeng_uses_golden_ratio :
  huaLuogengMethod = OptimalSelectionMethod.GoldenRatio := by sorry

end NUMINAMATH_CALUDE_hua_luogeng_uses_golden_ratio_l1646_164654


namespace NUMINAMATH_CALUDE_basketball_players_count_l1646_164696

def students_jumping_rope : ℕ := 6

def students_playing_basketball : ℕ := 4 * students_jumping_rope

theorem basketball_players_count : students_playing_basketball = 24 := by
  sorry

end NUMINAMATH_CALUDE_basketball_players_count_l1646_164696


namespace NUMINAMATH_CALUDE_inverse_value_of_symmetrical_function_l1646_164626

-- Define a function that is symmetrical about a point
def SymmetricalAboutPoint (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

-- Define the existence of an inverse function
def HasInverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem inverse_value_of_symmetrical_function
  (f : ℝ → ℝ)
  (h_sym : SymmetricalAboutPoint f (1, 2))
  (h_inv : HasInverse f)
  (h_f4 : f 4 = 0) :
  ∃ f_inv : ℝ → ℝ, HasInverse f ∧ f_inv 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_value_of_symmetrical_function_l1646_164626


namespace NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_for_skew_lines_l1646_164622

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the intersection relation between two planes
variable (intersect : Plane → Plane → Prop)

-- Define the skew relation between two lines
variable (skew : Line → Line → Prop)

-- Define the theorem
theorem planes_intersect_necessary_not_sufficient_for_skew_lines
  (α β : Plane) (m n : Line)
  (distinct_planes : α ≠ β)
  (m_perp_α : perpendicular m α)
  (n_perp_β : perpendicular n β) :
  (∀ m n, skew m n → intersect α β) ∧
  ¬(∀ m n, intersect α β → skew m n) :=
sorry

end NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_for_skew_lines_l1646_164622


namespace NUMINAMATH_CALUDE_square_difference_l1646_164676

theorem square_difference (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1646_164676


namespace NUMINAMATH_CALUDE_equation_solution_l1646_164692

theorem equation_solution (a : ℝ) : 
  (2*a + 4*(-1) = (-1) + 5*a) → 
  (a = -1) ∧ 
  (∀ y : ℝ, (-1)*y + 6 = 6*(-1) + 2*y → y = 4) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1646_164692


namespace NUMINAMATH_CALUDE_sine_sum_identity_l1646_164659

theorem sine_sum_identity (α β γ : ℝ) (h : α + β + γ = 0) :
  Real.sin α + Real.sin β + Real.sin γ = -4 * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_identity_l1646_164659


namespace NUMINAMATH_CALUDE_gcd_set_divisors_l1646_164682

theorem gcd_set_divisors (a b c d : ℕ+) (h1 : a * d ≠ b * c) (h2 : Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1) :
  ∃ k : ℕ, {x : ℕ | ∃ n : ℕ+, x = Nat.gcd (a * n + b) (c * n + d)} = {x : ℕ | x ∣ k} := by
  sorry


end NUMINAMATH_CALUDE_gcd_set_divisors_l1646_164682


namespace NUMINAMATH_CALUDE_initial_average_height_l1646_164651

/-- The initially calculated average height of students in a class with measurement error -/
theorem initial_average_height (n : ℕ) (incorrect_height actual_height : ℝ) (actual_average : ℝ) 
  (hn : n = 20)
  (h_incorrect : incorrect_height = 151)
  (h_actual : actual_height = 136)
  (h_average : actual_average = 174.25) :
  ∃ (initial_average : ℝ), 
    initial_average * n = actual_average * n - (incorrect_height - actual_height) ∧ 
    initial_average = 173.5 := by
  sorry


end NUMINAMATH_CALUDE_initial_average_height_l1646_164651


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l1646_164666

/-- Given a curve in polar coordinates ρ = 4sin θ, prove its equivalence to the rectangular form x² + y² - 4y = 0 -/
theorem polar_to_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
  (ρ = 4 * Real.sin θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) →
  (x^2 + y^2 - 4*y = 0) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l1646_164666


namespace NUMINAMATH_CALUDE_horner_v₂_eq_40_l1646_164623

/-- Horner's method for a polynomial of degree 6 -/
def horner (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) (x : ℝ) : ℝ :=
  a₀ + x * (a₁ + x * (a₂ + x * (a₃ + x * (a₄ + x * (a₅ + x * a₆)))))

/-- The second Horner value for a polynomial of degree 6 -/
def v₂ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) (x : ℝ) : ℝ :=
  a₂ + x * (a₃ + x * (a₄ + x * (a₅ + x * a₆))) - 
  x * (a₁ + x * (a₂ + x * (a₃ + x * (a₄ + x * (a₅ + x * a₆)))))

theorem horner_v₂_eq_40 :
  v₂ 64 (-192) 240 (-160) 60 (-12) 1 2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_horner_v₂_eq_40_l1646_164623


namespace NUMINAMATH_CALUDE_bullet_speed_difference_l1646_164617

/-- The speed difference of a bullet fired from a moving horse with wind assistance -/
theorem bullet_speed_difference
  (horse_speed : ℝ) 
  (bullet_speed : ℝ)
  (wind_speed : ℝ)
  (h1 : horse_speed = 20)
  (h2 : bullet_speed = 400)
  (h3 : wind_speed = 10) :
  (bullet_speed + horse_speed + wind_speed) - (bullet_speed - horse_speed - wind_speed) = 60 := by
  sorry


end NUMINAMATH_CALUDE_bullet_speed_difference_l1646_164617


namespace NUMINAMATH_CALUDE_maximize_product_l1646_164657

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 50) :
  x^6 * y^3 ≤ (100/3)^6 * (50/3)^3 ∧
  x^6 * y^3 = (100/3)^6 * (50/3)^3 ↔ x = 100/3 ∧ y = 50/3 :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l1646_164657


namespace NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l1646_164674

/-- Given a geometric sequence with first term 12 and second term 4,
    prove that its eighth term is 4/729. -/
theorem eighth_term_of_geometric_sequence : 
  ∀ (a : ℕ → ℚ), 
    (∀ n, a (n + 2) * a n = (a (n + 1))^2) →  -- geometric sequence condition
    a 1 = 12 →                                -- first term
    a 2 = 4 →                                 -- second term
    a 8 = 4/729 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l1646_164674


namespace NUMINAMATH_CALUDE_gcd_of_repeated_numbers_l1646_164650

def repeated_number (n : ℕ) : ℕ := 1001001001 * n

theorem gcd_of_repeated_numbers :
  ∃ (m : ℕ), m > 0 ∧ m < 1000 ∧
  (∀ (n : ℕ), n > 0 ∧ n < 1000 → Nat.gcd (repeated_number m) (repeated_number n) = 1001001001) :=
sorry

end NUMINAMATH_CALUDE_gcd_of_repeated_numbers_l1646_164650


namespace NUMINAMATH_CALUDE_unique_coin_combination_l1646_164681

/-- Represents the number of coins of each denomination -/
structure CoinCombination where
  bronze : Nat
  silver : Nat
  gold : Nat

/-- Calculates the total value of a coin combination -/
def totalValue (c : CoinCombination) : Nat :=
  c.bronze + 9 * c.silver + 81 * c.gold

/-- Calculates the total number of coins in a combination -/
def totalCoins (c : CoinCombination) : Nat :=
  c.bronze + c.silver + c.gold

/-- Checks if a coin combination is valid for the problem -/
def isValidCombination (c : CoinCombination) : Prop :=
  totalCoins c = 23 ∧ totalValue c < 700

/-- Checks if a coin combination has the minimum number of coins for its value -/
def isMinimalCombination (c : CoinCombination) : Prop :=
  ∀ c', isValidCombination c' → totalValue c' = totalValue c → totalCoins c' ≥ totalCoins c

/-- The main theorem to prove -/
theorem unique_coin_combination : 
  ∃! c : CoinCombination, isValidCombination c ∧ isMinimalCombination c ∧ totalValue c = 647 :=
sorry

end NUMINAMATH_CALUDE_unique_coin_combination_l1646_164681


namespace NUMINAMATH_CALUDE_power_inequality_l1646_164670

theorem power_inequality : 0.2^0.3 < 0.3^0.3 ∧ 0.3^0.3 < 0.3^0.2 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1646_164670


namespace NUMINAMATH_CALUDE_homework_problem_l1646_164688

/-- Given a homework assignment with a total number of problems, 
    finished problems, and remaining pages, calculate the number of 
    problems per page assuming each page has the same number of problems. -/
def problems_per_page (total : ℕ) (finished : ℕ) (pages : ℕ) : ℕ :=
  (total - finished) / pages

/-- Theorem stating that for the given homework scenario, 
    there are 7 problems per page. -/
theorem homework_problem : 
  problems_per_page 40 26 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l1646_164688


namespace NUMINAMATH_CALUDE_complex_quotient_real_l1646_164631

/-- Given complex numbers Z₁ and Z₂, where Z₁ = a + 2i and Z₂ = 3 - 4i,
    if Z₁/Z₂ is a real number, then a = -3/2 -/
theorem complex_quotient_real (a : ℝ) :
  let Z₁ : ℂ := a + 2*I
  let Z₂ : ℂ := 3 - 4*I
  (∃ (r : ℝ), Z₁ / Z₂ = r) → a = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_quotient_real_l1646_164631


namespace NUMINAMATH_CALUDE_modulus_of_z_l1646_164620

theorem modulus_of_z (z : ℂ) (h : z * (1 - Complex.I) = 2 + 4 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1646_164620


namespace NUMINAMATH_CALUDE_smallest_absolute_value_at_0_l1646_164656

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- The property that a polynomial P satisfies P(-10) = 145 and P(9) = 164 -/
def SatisfiesConditions (P : IntPolynomial) : Prop :=
  P (-10) = 145 ∧ P 9 = 164

/-- The smallest possible absolute value of P(0) for polynomials satisfying the conditions -/
def SmallestAbsoluteValueAt0 : ℕ := 25

theorem smallest_absolute_value_at_0 :
  ∀ P : IntPolynomial,
  SatisfiesConditions P →
  ∀ n : ℕ,
  n < SmallestAbsoluteValueAt0 →
  ¬(|P 0| = n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_at_0_l1646_164656


namespace NUMINAMATH_CALUDE_factorization_sum_l1646_164647

theorem factorization_sum (d e f : ℤ) :
  (∀ x : ℝ, x^2 + 20*x + 75 = (x + d)*(x + e)) →
  (∀ x : ℝ, x^2 - 22*x + 120 = (x - e)*(x - f)) →
  d + e + f = 37 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l1646_164647


namespace NUMINAMATH_CALUDE_prime_sum_and_squares_l1646_164625

theorem prime_sum_and_squares (p q r s : ℕ) : 
  p.Prime ∧ q.Prime ∧ r.Prime ∧ s.Prime ∧  -- p, q, r, s are prime
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧  -- p, q, r, s are distinct
  (p + q + r + s).Prime ∧  -- their sum is prime
  ∃ a, p^2 + q*s = a^2 ∧  -- p² + qs is a perfect square
  ∃ b, p^2 + q*r = b^2  -- p² + qr is a perfect square
  →
  ((p = 2 ∧ q = 7 ∧ r = 11 ∧ s = 3) ∨ (p = 2 ∧ q = 7 ∧ r = 3 ∧ s = 11)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_and_squares_l1646_164625


namespace NUMINAMATH_CALUDE_range_of_b_l1646_164675

/-- The curve representing a semi-circle -/
def curve (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

/-- The line intersecting the curve -/
def line (x y b : ℝ) : Prop := y = x + b

/-- The domain constraints for x and y -/
def domain_constraints (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3

/-- The theorem stating the range of b -/
theorem range_of_b :
  ∀ b : ℝ, (∃ x y : ℝ, curve x y ∧ line x y b ∧ domain_constraints x y) ↔ 
  (1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l1646_164675


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_quarter_of_fifth_of_sixth_of_120_l1646_164637

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * (b * (c * d)) = (a * b * c) * d :=
by sorry

theorem quarter_of_fifth_of_sixth_of_120 :
  (1 / 4 : ℚ) * ((1 / 5 : ℚ) * ((1 / 6 : ℚ) * 120)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_quarter_of_fifth_of_sixth_of_120_l1646_164637


namespace NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_ratio_l1646_164649

theorem larger_number_given_hcf_lcm_ratio (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 84)
  (lcm_eq : Nat.lcm a b = 21)
  (ratio : a * 4 = b) :
  max a b = 84 := by
sorry

end NUMINAMATH_CALUDE_larger_number_given_hcf_lcm_ratio_l1646_164649
