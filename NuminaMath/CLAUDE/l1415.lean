import Mathlib

namespace NUMINAMATH_CALUDE_probability_of_passing_test_probability_is_two_thirds_l1415_141573

/-- Represents the probability of passing a test given specific conditions -/
theorem probability_of_passing_test
  (total_questions : ℕ)
  (selected_questions : ℕ)
  (correct_answers : ℕ)
  (passing_threshold : ℕ)
  (h1 : total_questions = 10)
  (h2 : selected_questions = 3)
  (h3 : correct_answers = 6)
  (h4 : passing_threshold = 2)
  (h5 : passing_threshold ≤ selected_questions)
  (h6 : correct_answers ≤ total_questions) :
  ℝ :=
2/3

/-- The main theorem stating that the probability of passing the test is 2/3 -/
theorem probability_is_two_thirds
  (total_questions : ℕ)
  (selected_questions : ℕ)
  (correct_answers : ℕ)
  (passing_threshold : ℕ)
  (h1 : total_questions = 10)
  (h2 : selected_questions = 3)
  (h3 : correct_answers = 6)
  (h4 : passing_threshold = 2)
  (h5 : passing_threshold ≤ selected_questions)
  (h6 : correct_answers ≤ total_questions) :
  probability_of_passing_test total_questions selected_questions correct_answers passing_threshold h1 h2 h3 h4 h5 h6 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_passing_test_probability_is_two_thirds_l1415_141573


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l1415_141594

theorem cylinder_volume_change (r h : ℝ) (h_positive : 0 < h) (r_positive : 0 < r) : 
  π * r^2 * h = 15 → π * (3*r)^2 * (h/2) = 67.5 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l1415_141594


namespace NUMINAMATH_CALUDE_min_colors_for_grid_colors_506_sufficient_min_colors_is_506_l1415_141523

def grid_size : ℕ := 2021

theorem min_colors_for_grid (n : ℕ) : 
  (∀ (col : Fin grid_size) (color : Fin n) (i j k l : Fin grid_size),
    i < j → j < k → k < l →
    (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
      (∀ (col' : Fin grid_size), col' > col → 
        ∃ (color' : Fin n), color' ≠ color))) →
  n ≥ 506 :=
sorry

theorem colors_506_sufficient : 
  ∃ (coloring : Fin grid_size → Fin grid_size → Fin 506),
    ∀ (col : Fin grid_size) (color : Fin 506) (i j k l : Fin grid_size),
      i < j → j < k → k < l →
      (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
        (∀ (col' : Fin grid_size), col' > col → 
          ∃ (color' : Fin 506), color' ≠ color)) :=
sorry

theorem min_colors_is_506 : 
  (∃ (n : ℕ), 
    (∀ (col : Fin grid_size) (color : Fin n) (i j k l : Fin grid_size),
      i < j → j < k → k < l →
      (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
        (∀ (col' : Fin grid_size), col' > col → 
          ∃ (color' : Fin n), color' ≠ color))) ∧
    (∀ m < n, ¬(∀ (col : Fin grid_size) (color : Fin m) (i j k l : Fin grid_size),
      i < j → j < k → k < l →
      (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
        (∀ (col' : Fin grid_size), col' > col → 
          ∃ (color' : Fin m), color' ≠ color))))) →
  n = 506 :=
sorry

end NUMINAMATH_CALUDE_min_colors_for_grid_colors_506_sufficient_min_colors_is_506_l1415_141523


namespace NUMINAMATH_CALUDE_product_of_roots_product_of_roots_specific_equation_l1415_141552

theorem product_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let p := (- b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
  let q := (- b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
  p * q = c / a :=
by sorry

theorem product_of_roots_specific_equation :
  let p := (9 + Real.sqrt (81 + 4 * 36)) / 2
  let q := (9 - Real.sqrt (81 + 4 * 36)) / 2
  p * q = -36 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_product_of_roots_specific_equation_l1415_141552


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1415_141568

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 
    (Real.rpow (1 - 2 * x^3 * Real.sin (5 / x)) (1/3)) - 1 + x
  else 
    0

theorem derivative_f_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1415_141568


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1415_141571

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x + 5
  ∃ x₁ x₂ : ℝ, x₁ = 5 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1415_141571


namespace NUMINAMATH_CALUDE_shaded_fraction_of_specific_rectangle_l1415_141525

/-- Represents a rectangle divided into equal squares -/
structure DividedRectangle where
  total_squares : ℕ
  shaded_half_squares : ℕ

/-- Calculates the fraction of a divided rectangle that is shaded -/
def shaded_fraction (rect : DividedRectangle) : ℚ :=
  rect.shaded_half_squares / (2 * rect.total_squares)

theorem shaded_fraction_of_specific_rectangle : 
  ∀ (rect : DividedRectangle), 
    rect.total_squares = 6 → 
    rect.shaded_half_squares = 5 → 
    shaded_fraction rect = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_specific_rectangle_l1415_141525


namespace NUMINAMATH_CALUDE_elevator_distribution_ways_l1415_141521

/-- The number of elevators available --/
def num_elevators : ℕ := 4

/-- The number of people taking elevators --/
def num_people : ℕ := 3

/-- The number of people taking the same elevator --/
def same_elevator : ℕ := 2

/-- The number of ways to distribute people among elevators --/
def distribute_ways : ℕ := 36

/-- Theorem stating that the number of ways to distribute people among elevators is 36 --/
theorem elevator_distribution_ways :
  (num_elevators = 4) →
  (num_people = 3) →
  (same_elevator = 2) →
  (distribute_ways = 36) := by
sorry

end NUMINAMATH_CALUDE_elevator_distribution_ways_l1415_141521


namespace NUMINAMATH_CALUDE_stock_price_change_l1415_141547

theorem stock_price_change (P1 P2 D : ℝ) (h1 : D = 0.18 * P1) (h2 : D = 0.12 * P2) :
  P2 = 1.5 * P1 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l1415_141547


namespace NUMINAMATH_CALUDE_age_difference_l1415_141535

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 17) : A = C + 17 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1415_141535


namespace NUMINAMATH_CALUDE_women_average_age_l1415_141581

theorem women_average_age (n : ℕ) (A : ℝ) (W₁ W₂ : ℝ) :
  n = 7 ∧ 
  (n * A - 26 - 30 + W₁ + W₂) / n = A + 4 →
  (W₁ + W₂) / 2 = 42 := by
sorry

end NUMINAMATH_CALUDE_women_average_age_l1415_141581


namespace NUMINAMATH_CALUDE_power_of_negative_two_a_cubed_l1415_141510

theorem power_of_negative_two_a_cubed (a : ℝ) : (-2 * a^3)^3 = -8 * a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_two_a_cubed_l1415_141510


namespace NUMINAMATH_CALUDE_fraction_inequality_l1415_141541

theorem fraction_inequality (x y : ℝ) (h : x / y = 3 / 4) :
  (2 * x + y) / y ≠ 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1415_141541


namespace NUMINAMATH_CALUDE_lowest_price_after_discounts_l1415_141565

/-- Calculates the lowest possible price of a product after applying regular and sale discounts -/
theorem lowest_price_after_discounts 
  (msrp : ℝ)
  (max_regular_discount : ℝ)
  (sale_discount : ℝ)
  (h1 : msrp = 40)
  (h2 : max_regular_discount = 0.3)
  (h3 : sale_discount = 0.2)
  : ∃ (lowest_price : ℝ), lowest_price = 22.4 :=
by
  sorry

#check lowest_price_after_discounts

end NUMINAMATH_CALUDE_lowest_price_after_discounts_l1415_141565


namespace NUMINAMATH_CALUDE_time_after_duration_sum_l1415_141518

/-- Represents time on a 12-hour digital clock -/
structure Time12 where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds a duration to a given time and returns the resulting time on a 12-hour clock -/
def addDuration (start : Time12) (hours minutes seconds : Nat) : Time12 :=
  sorry

/-- Converts a Time12 to the sum of its components -/
def timeSum (t : Time12) : Nat :=
  t.hours + t.minutes + t.seconds

theorem time_after_duration_sum :
  let start := Time12.mk 3 0 0
  let result := addDuration start 307 58 59
  timeSum result = 127 := by
  sorry

end NUMINAMATH_CALUDE_time_after_duration_sum_l1415_141518


namespace NUMINAMATH_CALUDE_factorization_proof_l1415_141516

theorem factorization_proof (a b : ℝ) : 4 * a^2 * (a - b) - (a - b) = (a - b) * (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1415_141516


namespace NUMINAMATH_CALUDE_equilateral_roots_ratio_l1415_141536

theorem equilateral_roots_ratio (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  z₂ = (Complex.exp (2*Real.pi*Complex.I/3)) * z₁ → 
  a^2 / b = 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_roots_ratio_l1415_141536


namespace NUMINAMATH_CALUDE_factor_implies_k_equals_five_l1415_141558

theorem factor_implies_k_equals_five (m k : ℤ) : 
  (∃ (A B : ℤ), m^3 - k*m^2 - 24*m + 16 = (m^2 - 8*m) * (A*m + B)) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_k_equals_five_l1415_141558


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_relation_l1415_141506

theorem cubic_expansion_coefficient_relation :
  ∀ (a₀ a₁ a₂ a₃ : ℝ),
  (∀ x : ℝ, (2*x + Real.sqrt 3)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_relation_l1415_141506


namespace NUMINAMATH_CALUDE_right_handed_players_count_l1415_141538

theorem right_handed_players_count (total_players throwers : ℕ) : 
  total_players = 70 →
  throwers = 34 →
  throwers ≤ total_players →
  (total_players - throwers) % 3 = 0 →
  (∃ (right_handed : ℕ), 
    right_handed = throwers + 2 * ((total_players - throwers) / 3) ∧
    right_handed = 58) := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l1415_141538


namespace NUMINAMATH_CALUDE_total_distance_is_11500_l1415_141504

/-- A right-angled triangle with sides XY, YZ, and ZX -/
structure RightTriangle where
  XY : ℝ
  ZX : ℝ
  YZ : ℝ
  right_angle : YZ^2 + ZX^2 = XY^2

/-- The total distance traveled in the triangle -/
def total_distance (t : RightTriangle) : ℝ :=
  t.XY + t.YZ + t.ZX

/-- Theorem: The total distance traveled in the given triangle is 11500 km -/
theorem total_distance_is_11500 :
  ∃ t : RightTriangle, t.XY = 5000 ∧ t.ZX = 4000 ∧ total_distance t = 11500 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_11500_l1415_141504


namespace NUMINAMATH_CALUDE_square_difference_l1415_141554

theorem square_difference : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by sorry

end NUMINAMATH_CALUDE_square_difference_l1415_141554


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l1415_141593

/-- Represents a rectangle with length and width --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def areaOfRectangle (r : Rectangle) : ℝ :=
  r.length * r.width

theorem equal_area_rectangles (carol_rect jordan_rect : Rectangle) :
  carol_rect.length = 5 →
  carol_rect.width = 24 →
  jordan_rect.length = 12 →
  areaOfRectangle carol_rect = areaOfRectangle jordan_rect →
  jordan_rect.width = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_equal_area_rectangles_l1415_141593


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_l1415_141560

theorem sum_of_squares_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_of_roots_specific :
  let a : ℝ := 10
  let b : ℝ := 15
  let c : ℝ := -20
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = 25/4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_l1415_141560


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1415_141599

/-- Given a circle with equation x^2 + y^2 = 1 and a line of symmetry x - y - 2 = 0,
    the equation of the symmetric circle is (x-2)^2 + (y+2)^2 = 1 -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 = 1) →
  (x - y - 2 = 0) →
  (∃ (x' y' : ℝ), (x' - 2)^2 + (y' + 2)^2 = 1 ∧
    (∀ (p q : ℝ), (p - q - 2 = 0) → 
      ((x - p)^2 + (y - q)^2 = (x' - p)^2 + (y' - q)^2))) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l1415_141599


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1415_141533

theorem profit_percentage_calculation (selling_price cost_price : ℝ) 
  (h1 : selling_price = 290)
  (h2 : cost_price = 241.67) : 
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1415_141533


namespace NUMINAMATH_CALUDE_solve_equation_l1415_141559

theorem solve_equation (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1415_141559


namespace NUMINAMATH_CALUDE_solution_range_l1415_141556

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ (2 * x + m) / (x - 1) = 1) → m > -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l1415_141556


namespace NUMINAMATH_CALUDE_product_mod_450_l1415_141584

theorem product_mod_450 : (2011 * 1537) % 450 = 307 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_450_l1415_141584


namespace NUMINAMATH_CALUDE_james_arthur_muffin_ratio_l1415_141597

theorem james_arthur_muffin_ratio :
  let arthur_muffins : ℕ := 115
  let james_muffins : ℕ := 1380
  (james_muffins : ℚ) / (arthur_muffins : ℚ) = 12 := by sorry

end NUMINAMATH_CALUDE_james_arthur_muffin_ratio_l1415_141597


namespace NUMINAMATH_CALUDE_work_time_ratio_l1415_141549

-- Define the time taken by A to finish the work
def time_A : ℝ := 4

-- Define the combined work rate of A and B
def combined_work_rate : ℝ := 0.75

-- Define the time taken by B to finish the work
def time_B : ℝ := 2

-- Theorem statement
theorem work_time_ratio :
  (1 / time_A + 1 / time_B = combined_work_rate) →
  (time_B / time_A = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_work_time_ratio_l1415_141549


namespace NUMINAMATH_CALUDE_force_system_ratio_l1415_141532

/-- Two forces acting on a material point at a right angle -/
structure ForceSystem where
  f1 : ℝ
  f2 : ℝ
  resultant : ℝ

/-- The magnitudes form an arithmetic progression -/
def is_arithmetic_progression (fs : ForceSystem) : Prop :=
  ∃ (d : ℝ), fs.f2 = fs.f1 + d ∧ fs.resultant = fs.f1 + 2*d

/-- The forces act at a right angle -/
def forces_at_right_angle (fs : ForceSystem) : Prop :=
  fs.resultant^2 = fs.f1^2 + fs.f2^2

/-- The ratio of the magnitudes of the forces is 3:4 -/
def force_ratio_is_3_to_4 (fs : ForceSystem) : Prop :=
  3 * fs.f2 = 4 * fs.f1

theorem force_system_ratio (fs : ForceSystem) 
  (h1 : is_arithmetic_progression fs) 
  (h2 : forces_at_right_angle fs) : 
  force_ratio_is_3_to_4 fs :=
sorry

end NUMINAMATH_CALUDE_force_system_ratio_l1415_141532


namespace NUMINAMATH_CALUDE_pool_count_l1415_141566

/-- The number of pools in two stores -/
def total_pools (store_a : ℕ) (store_b : ℕ) : ℕ :=
  store_a + store_b

/-- Theorem stating the total number of pools given the conditions -/
theorem pool_count : 
  let store_a := 200
  let store_b := 3 * store_a
  total_pools store_a store_b = 800 := by
sorry

end NUMINAMATH_CALUDE_pool_count_l1415_141566


namespace NUMINAMATH_CALUDE_smallest_number_greater_than_0_4_l1415_141586

theorem smallest_number_greater_than_0_4 (S : Set ℝ) : 
  S = {0.8, 1/2, 0.3, 1/3} → 
  (∃ x ∈ S, x > 0.4 ∧ ∀ y ∈ S, y > 0.4 → x ≤ y) → 
  (1/2 ∈ S ∧ 1/2 > 0.4 ∧ ∀ y ∈ S, y > 0.4 → 1/2 ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_greater_than_0_4_l1415_141586


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_l1415_141529

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the upper vertex M of the ellipse
def upper_vertex (M : ℝ × ℝ) : Prop := 
  M.1 = 0 ∧ M.2 = 1

-- Define points A and B on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := 
  ellipse_C P.1 P.2

-- Define the slopes of lines MA and MB
def slopes_sum_2 (k₁ k₂ : ℝ) : Prop := 
  k₁ + k₂ = 2

-- Theorem statement
theorem ellipse_fixed_point 
  (M A B : ℝ × ℝ) 
  (k₁ k₂ : ℝ) 
  (hM : upper_vertex M) 
  (hA : on_ellipse A) 
  (hB : on_ellipse B) 
  (hk : slopes_sum_2 k₁ k₂) :
  ∃ (t : ℝ), A.1 * t + A.2 * (1 - t) = -1 ∧ 
             B.1 * t + B.2 * (1 - t) = -1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_l1415_141529


namespace NUMINAMATH_CALUDE_inverse_tan_product_range_l1415_141574

/-- Given an acute-angled triangle ABC where b^2 - a^2 = ac, 
    prove that 1 / (tan A * tan B) is in the open interval (0, 1) -/
theorem inverse_tan_product_range (A B C : ℝ) (a b c : ℝ) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sides : b^2 - a^2 = a*c) :
  0 < (1 : ℝ) / (Real.tan A * Real.tan B) ∧ (1 : ℝ) / (Real.tan A * Real.tan B) < 1 :=
sorry

end NUMINAMATH_CALUDE_inverse_tan_product_range_l1415_141574


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1415_141561

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), (2*a*x - b*y + 2 = 0 ∧ 
                 x^2 + y^2 + 2*x - 4*y + 1 = 0) ∧
   (∃ (x1 y1 x2 y2 : ℝ), 
      (2*a*x1 - b*y1 + 2 = 0 ∧ x1^2 + y1^2 + 2*x1 - 4*y1 + 1 = 0) ∧
      (2*a*x2 - b*y2 + 2 = 0 ∧ x2^2 + y2^2 + 2*x2 - 4*y2 + 1 = 0) ∧
      ((x1 - x2)^2 + (y1 - y2)^2 = 16))) →
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 1/a' + 1/b' ≥ 1/a + 1/b) →
  1/a + 1/b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1415_141561


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l1415_141580

theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 3) - 5
  f (3/2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l1415_141580


namespace NUMINAMATH_CALUDE_shaded_area_comparison_l1415_141539

/-- Represents a square divided into smaller squares -/
structure DividedSquare where
  total_divisions : ℕ
  shaded_divisions : ℕ

/-- The three squares described in the problem -/
def square_I : DividedSquare := { total_divisions := 16, shaded_divisions := 4 }
def square_II : DividedSquare := { total_divisions := 64, shaded_divisions := 16 }
def square_III : DividedSquare := { total_divisions := 16, shaded_divisions := 8 }

/-- Calculates the shaded area ratio of a divided square -/
def shaded_area_ratio (s : DividedSquare) : ℚ :=
  s.shaded_divisions / s.total_divisions

/-- Theorem stating the equality of shaded areas for squares I and II, and the difference for square III -/
theorem shaded_area_comparison :
  shaded_area_ratio square_I = shaded_area_ratio square_II ∧
  shaded_area_ratio square_I ≠ shaded_area_ratio square_III ∧
  shaded_area_ratio square_II ≠ shaded_area_ratio square_III := by
  sorry

#eval shaded_area_ratio square_I
#eval shaded_area_ratio square_II
#eval shaded_area_ratio square_III

end NUMINAMATH_CALUDE_shaded_area_comparison_l1415_141539


namespace NUMINAMATH_CALUDE_total_oranges_l1415_141595

def oranges_per_box : ℝ := 10.0
def boxes_packed : ℝ := 2650.0

theorem total_oranges : oranges_per_box * boxes_packed = 26500.0 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_l1415_141595


namespace NUMINAMATH_CALUDE_annie_milkshakes_l1415_141555

/-- The number of milkshakes Annie bought -/
def milkshakes : ℕ := sorry

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℕ := 4

/-- The cost of a milkshake in dollars -/
def milkshake_cost : ℕ := 5

/-- The number of hamburgers Annie bought -/
def hamburgers_bought : ℕ := 8

/-- Annie's initial amount of money in dollars -/
def initial_money : ℕ := 132

/-- Annie's remaining money after purchases in dollars -/
def remaining_money : ℕ := 70

theorem annie_milkshakes :
  milkshakes = 6 ∧
  initial_money = remaining_money + hamburgers_bought * hamburger_cost + milkshakes * milkshake_cost :=
by sorry

end NUMINAMATH_CALUDE_annie_milkshakes_l1415_141555


namespace NUMINAMATH_CALUDE_union_complement_problem_l1415_141512

theorem union_complement_problem (U A B : Set Char) : 
  U = {'a', 'b', 'c', 'd', 'e'} →
  A = {'b', 'c', 'd'} →
  B = {'b', 'e'} →
  B ∪ (U \ A) = {'a', 'b', 'e'} := by
sorry

end NUMINAMATH_CALUDE_union_complement_problem_l1415_141512


namespace NUMINAMATH_CALUDE_inequality_proof_l1415_141550

theorem inequality_proof (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h : x₁^2 + x₂^2 + x₃^2 ≤ 1) :
  (x₁*y₁ + x₂*y₂ + x₃*y₃ - 1)^2 ≥ (x₁^2 + x₂^2 + x₃^2 - 1)*(y₁^2 + y₂^2 + y₃^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1415_141550


namespace NUMINAMATH_CALUDE_probability_two_females_l1415_141543

/-- The probability of selecting two female contestants out of 7 total contestants 
    (4 female, 3 male) when choosing 2 contestants at random -/
theorem probability_two_females (total : Nat) (females : Nat) (chosen : Nat) 
    (h1 : total = 7) 
    (h2 : females = 4) 
    (h3 : chosen = 2) : 
    (Nat.choose females chosen : Rat) / (Nat.choose total chosen : Rat) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_l1415_141543


namespace NUMINAMATH_CALUDE_mba_committee_size_l1415_141531

/-- Represents the number of second-year MBAs -/
def total_mbas : ℕ := 6

/-- Represents the number of committees -/
def num_committees : ℕ := 2

/-- Represents the probability that Jane and Albert are on the same committee -/
def same_committee_prob : ℚ := 2/5

/-- Represents the number of members in each committee -/
def committee_size : ℕ := total_mbas / num_committees

theorem mba_committee_size :
  (committee_size = 3) ∧
  (same_committee_prob = (committee_size - 1 : ℚ) / (total_mbas - 1 : ℚ)) :=
sorry

end NUMINAMATH_CALUDE_mba_committee_size_l1415_141531


namespace NUMINAMATH_CALUDE_mike_changed_tires_on_ten_cars_l1415_141598

/-- The number of cars Mike changed tires on -/
def num_cars (total_tires num_motorcycles tires_per_motorcycle tires_per_car : ℕ) : ℕ :=
  (total_tires - num_motorcycles * tires_per_motorcycle) / tires_per_car

theorem mike_changed_tires_on_ten_cars :
  num_cars 64 12 2 4 = 10 := by sorry

end NUMINAMATH_CALUDE_mike_changed_tires_on_ten_cars_l1415_141598


namespace NUMINAMATH_CALUDE_solve_equation_l1415_141526

theorem solve_equation (x : ℝ) : 3 * x + 36 = 48 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1415_141526


namespace NUMINAMATH_CALUDE_world_grain_supply_l1415_141524

/-- World grain supply problem -/
theorem world_grain_supply :
  let world_grain_demand : ℝ := 2400000
  let supply_ratio : ℝ := 0.75
  let world_grain_supply : ℝ := supply_ratio * world_grain_demand
  world_grain_supply = 1800000 := by
  sorry

end NUMINAMATH_CALUDE_world_grain_supply_l1415_141524


namespace NUMINAMATH_CALUDE_min_sum_of_product_72_l1415_141544

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) : 
  (∀ x y : ℤ, x * y = 72 → a + b ≤ x + y) ∧ (a + b = -17) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_72_l1415_141544


namespace NUMINAMATH_CALUDE_binomial_square_equivalence_l1415_141572

theorem binomial_square_equivalence (x y : ℝ) : 
  (-x - y) * (-x + y) = (-x - y)^2 := by sorry

end NUMINAMATH_CALUDE_binomial_square_equivalence_l1415_141572


namespace NUMINAMATH_CALUDE_division_problem_l1415_141588

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 141 →
  quotient = 8 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1415_141588


namespace NUMINAMATH_CALUDE_unique_prime_pair_l1415_141578

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Nat.Prime (p + q) ∧ 
  Nat.Prime (p^2 + q^2 - q) ∧ 
  p = 3 ∧ 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l1415_141578


namespace NUMINAMATH_CALUDE_rectangle_arrangement_perimeters_l1415_141508

/- Define the properties of the rectangles -/
def identical_rectangles (l w : ℝ) : Prop := l = 2 * w

/- Define the first arrangement's perimeter -/
def first_arrangement_perimeter (l w : ℝ) : ℝ := 3 * l + 4 * w

/- Define the second arrangement's perimeter -/
def second_arrangement_perimeter (l w : ℝ) : ℝ := 6 * l + 2 * w

/- Theorem statement -/
theorem rectangle_arrangement_perimeters (l w : ℝ) :
  identical_rectangles l w →
  first_arrangement_perimeter l w = 20 →
  second_arrangement_perimeter l w = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_perimeters_l1415_141508


namespace NUMINAMATH_CALUDE_mono_increasing_range_l1415_141582

/-- A function f is monotonically increasing on ℝ -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem mono_increasing_range (f : ℝ → ℝ) (h : MonoIncreasing f) :
  ∀ m : ℝ, f (2 * m - 3) > f (-m) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_mono_increasing_range_l1415_141582


namespace NUMINAMATH_CALUDE_profit_starts_third_year_average_profit_plan_more_effective_l1415_141575

/-- Represents a fishing company's financial situation -/
structure FishingCompany where
  initialCost : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrement : ℕ
  annualIncome : ℕ

/-- Calculates the cumulative expenses after n years -/
def cumulativeExpenses (company : FishingCompany) (n : ℕ) : ℕ :=
  company.initialCost + n * company.firstYearExpenses + (n * (n - 1) / 2) * company.annualExpenseIncrement

/-- Calculates the cumulative income after n years -/
def cumulativeIncome (company : FishingCompany) (n : ℕ) : ℕ :=
  n * company.annualIncome

/-- Determines if the company is profitable after n years -/
def isProfitable (company : FishingCompany) (n : ℕ) : Prop :=
  cumulativeIncome company n > cumulativeExpenses company n

/-- Represents the two selling plans -/
inductive SellingPlan
  | AverageProfit
  | TotalNetIncome

/-- Theorem: The company begins to profit in the third year -/
theorem profit_starts_third_year (company : FishingCompany) 
  (h1 : company.initialCost = 490000)
  (h2 : company.firstYearExpenses = 60000)
  (h3 : company.annualExpenseIncrement = 20000)
  (h4 : company.annualIncome = 250000) :
  isProfitable company 3 ∧ ¬isProfitable company 2 :=
sorry

/-- Theorem: The average annual profit plan is more cost-effective -/
theorem average_profit_plan_more_effective (company : FishingCompany) 
  (h1 : company.initialCost = 490000)
  (h2 : company.firstYearExpenses = 60000)
  (h3 : company.annualExpenseIncrement = 20000)
  (h4 : company.annualIncome = 250000) :
  ∃ (n m : ℕ), 
    (∀ k, cumulativeIncome company k - cumulativeExpenses company k + 180000 ≤ n) ∧
    (∀ k, cumulativeIncome company k - cumulativeExpenses company k + 90000 ≤ m) ∧
    n > m :=
sorry

end NUMINAMATH_CALUDE_profit_starts_third_year_average_profit_plan_more_effective_l1415_141575


namespace NUMINAMATH_CALUDE_product_not_divisible_by_72_l1415_141503

def S : Finset Nat := {4, 8, 18, 28, 36, 49, 56}

theorem product_not_divisible_by_72 (a b : Nat) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  ¬(72 ∣ a * b) := by
  sorry

#check product_not_divisible_by_72

end NUMINAMATH_CALUDE_product_not_divisible_by_72_l1415_141503


namespace NUMINAMATH_CALUDE_square_room_perimeter_l1415_141591

theorem square_room_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 500 → perimeter = 40 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_square_room_perimeter_l1415_141591


namespace NUMINAMATH_CALUDE_product_difference_equals_2019_l1415_141577

theorem product_difference_equals_2019 : 672 * 673 * 674 - 671 * 673 * 675 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equals_2019_l1415_141577


namespace NUMINAMATH_CALUDE_crosswalk_distance_l1415_141546

/-- Given a parallelogram with one side of length 20 feet, height of 60 feet,
    and another side of length 80 feet, the distance between the 20-foot side
    and the 80-foot side is 15 feet. -/
theorem crosswalk_distance (side1 side2 height : ℝ) : 
  side1 = 20 → side2 = 80 → height = 60 → 
  (side1 * height) / side2 = 15 := by sorry

end NUMINAMATH_CALUDE_crosswalk_distance_l1415_141546


namespace NUMINAMATH_CALUDE_sum_of_rationals_is_rational_l1415_141589

theorem sum_of_rationals_is_rational (r₁ r₂ : ℚ) : ∃ (q : ℚ), r₁ + r₂ = q := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rationals_is_rational_l1415_141589


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1415_141505

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  ∃ (min : ℝ), min = (3 / 2 + Real.sqrt 2) ∧ ∀ z w, z > 0 → w > 0 → 2 * z + w = 2 → 1 / z + 1 / w ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1415_141505


namespace NUMINAMATH_CALUDE_train_journey_theorem_l1415_141585

/-- Represents the train's journey with two potential accident scenarios -/
theorem train_journey_theorem (D v : ℝ) : 
  (D > 0) →  -- Distance is positive
  (v > 0) →  -- Speed is positive
  -- First accident scenario
  (2 + 1 + (3 * (D - 2*v)) / (2*v) = D/v + 4) → 
  -- Second accident scenario
  (2.5 + 120/v + (6 * (D - 2*v - 120)) / (5*v) = D/v + 3.5) → 
  -- The distance D is one of the given choices
  (D = 420 ∨ D = 480 ∨ D = 540 ∨ D = 600 ∨ D = 660) :=
by sorry


end NUMINAMATH_CALUDE_train_journey_theorem_l1415_141585


namespace NUMINAMATH_CALUDE_total_cents_l1415_141583

-- Define the values in cents
def lance_cents : ℕ := 70
def margaret_cents : ℕ := 75 -- three-fourths of a dollar
def guy_cents : ℕ := 50 + 10 -- two quarters and a dime
def bill_cents : ℕ := 6 * 10 -- six dimes

-- Theorem to prove
theorem total_cents : 
  lance_cents + margaret_cents + guy_cents + bill_cents = 265 := by
  sorry

end NUMINAMATH_CALUDE_total_cents_l1415_141583


namespace NUMINAMATH_CALUDE_smallest_sum_is_337_dice_sum_theorem_l1415_141501

/-- Represents a set of symmetrical dice --/
structure DiceSet where
  num_dice : ℕ
  max_sum : ℕ
  min_sum : ℕ

/-- The property that the dice set can achieve a sum of 2022 --/
def can_sum_2022 (d : DiceSet) : Prop :=
  d.max_sum = 2022

/-- The property that each die is symmetrical (6-sided) --/
def symmetrical_dice (d : DiceSet) : Prop :=
  d.max_sum = 6 * d.num_dice ∧ d.min_sum = d.num_dice

/-- The theorem stating that the smallest possible sum is 337 --/
theorem smallest_sum_is_337 (d : DiceSet) 
  (h1 : can_sum_2022 d) 
  (h2 : symmetrical_dice d) : 
  d.min_sum = 337 := by
  sorry

/-- The main theorem combining all conditions --/
theorem dice_sum_theorem (d : DiceSet) 
  (h1 : can_sum_2022 d) 
  (h2 : symmetrical_dice d) : 
  ∃ (p : ℝ), p > 0 ∧ 
    (∃ (sum : ℕ), sum = 2022 ∧ sum ≤ d.max_sum) ∧
    (∃ (min_sum : ℕ), min_sum = 337 ∧ min_sum = d.min_sum) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_is_337_dice_sum_theorem_l1415_141501


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1415_141517

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of six coins -/
structure CoinSet :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (half_dollar : CoinFlip)
  (dollar : CoinFlip)

/-- The condition for a successful outcome -/
def successful_outcome (cs : CoinSet) : Prop :=
  (cs.penny = cs.nickel) ∧
  (cs.dime = cs.quarter) ∧ (cs.quarter = cs.half_dollar)

/-- The total number of possible outcomes -/
def total_outcomes : Nat := 64

/-- The number of successful outcomes -/
def successful_outcomes : Nat := 16

/-- The theorem to be proved -/
theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1415_141517


namespace NUMINAMATH_CALUDE_marbles_selection_count_l1415_141567

def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def marbles_to_choose : ℕ := 5

theorem marbles_selection_count :
  (special_marbles * (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - 1))) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_marbles_selection_count_l1415_141567


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l1415_141590

theorem parabola_point_comparison 
  (m : ℝ) (t x₁ x₂ y₁ y₂ : ℝ) 
  (h_m : m > 0)
  (h_x₁ : t < x₁ ∧ x₁ < t + 1)
  (h_x₂ : t + 2 < x₂ ∧ x₂ < t + 3)
  (h_y₁ : y₁ = m * x₁^2 - 2 * m * x₁ + 1)
  (h_y₂ : y₂ = m * x₂^2 - 2 * m * x₂ + 1)
  (h_t : t ≥ 1) :
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l1415_141590


namespace NUMINAMATH_CALUDE_box_volume_correct_l1415_141528

/-- The volume of an open box created from a rectangular sheet -/
def boxVolume (L W S : ℝ) : ℝ := (L - 2*S) * (W - 2*S) * S

/-- Theorem stating that the boxVolume function correctly calculates the volume of the open box -/
theorem box_volume_correct (L W S : ℝ) (hL : L > 0) (hW : W > 0) (hS : 0 < S ∧ S < L/2 ∧ S < W/2) : 
  boxVolume L W S = (L - 2*S) * (W - 2*S) * S :=
sorry

end NUMINAMATH_CALUDE_box_volume_correct_l1415_141528


namespace NUMINAMATH_CALUDE_sum_in_B_l1415_141562

def A : Set ℤ := {x | ∃ k, x = 2 * k}
def B : Set ℤ := {x | ∃ k, x = 2 * k + 1}
def C : Set ℤ := {x | ∃ k, x = 4 * k + 1}

theorem sum_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end NUMINAMATH_CALUDE_sum_in_B_l1415_141562


namespace NUMINAMATH_CALUDE_sister_watermelons_count_l1415_141545

/-- The number of watermelons Danny brings -/
def danny_watermelons : ℕ := 3

/-- The number of slices Danny cuts each watermelon into -/
def danny_slices_per_watermelon : ℕ := 10

/-- The number of slices Danny's sister cuts each watermelon into -/
def sister_slices_per_watermelon : ℕ := 15

/-- The total number of watermelon slices at the picnic -/
def total_slices : ℕ := 45

/-- The number of watermelons Danny's sister brings -/
def sister_watermelons : ℕ := 1

theorem sister_watermelons_count : sister_watermelons = 
  (total_slices - danny_watermelons * danny_slices_per_watermelon) / sister_slices_per_watermelon := by
  sorry

end NUMINAMATH_CALUDE_sister_watermelons_count_l1415_141545


namespace NUMINAMATH_CALUDE_function_inequality_l1415_141576

open Real

theorem function_inequality (f : ℝ → ℝ) (a : ℝ) (h_cont : Continuous f) (h_pos : a > 0) 
  (h_fa : f a = 1) (h_ineq : ∀ x y, x > 0 → y > 0 → f x * f y + f (a / x) * f (a / y) ≤ 2 * f (x * y)) :
  ∀ x y, x > 0 → y > 0 → f x * f y ≤ f (x * y) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1415_141576


namespace NUMINAMATH_CALUDE_monotonic_f_range_a_l1415_141596

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^2 + a*x + a/4 else a^x

/-- Theorem stating the range of a for monotonically increasing f(x) -/
theorem monotonic_f_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_f_range_a_l1415_141596


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1415_141507

theorem fraction_sum_equals_decimal : 
  (1 : ℚ) / 10 + 9 / 100 + 9 / 1000 + 7 / 10000 = 0.1997 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1415_141507


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1415_141570

/-- The quadratic function f(x) with parameter t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^2 + t*x - t

/-- f has a root for a given t -/
def has_root (t : ℝ) : Prop := ∃ x, f t x = 0

theorem sufficient_not_necessary_condition :
  (∀ t ≥ 0, has_root t) ∧ (∃ t < 0, has_root t) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1415_141570


namespace NUMINAMATH_CALUDE_rectangles_with_one_gray_count_l1415_141587

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the count of different types of cells in the grid -/
structure CellCount :=
  (total_gray : ℕ)
  (interior_gray : ℕ)
  (edge_gray : ℕ)

/-- Calculates the number of rectangles containing exactly one gray cell -/
def count_rectangles_with_one_gray (g : Grid) (c : CellCount) : ℕ :=
  c.interior_gray * 4 + c.edge_gray * 8

/-- The main theorem stating the number of rectangles with one gray cell -/
theorem rectangles_with_one_gray_count 
  (g : Grid) 
  (c : CellCount) 
  (h1 : g.rows = 5) 
  (h2 : g.cols = 22) 
  (h3 : c.total_gray = 40) 
  (h4 : c.interior_gray = 36) 
  (h5 : c.edge_gray = 4) :
  count_rectangles_with_one_gray g c = 176 := by
  sorry

#check rectangles_with_one_gray_count

end NUMINAMATH_CALUDE_rectangles_with_one_gray_count_l1415_141587


namespace NUMINAMATH_CALUDE_simplify_trig_fraction_l1415_141511

theorem simplify_trig_fraction (x : ℝ) : 
  (1 - Real.sin x - Real.cos x) / (1 - Real.sin x + Real.cos x) = -Real.tan (x / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_fraction_l1415_141511


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l1415_141500

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

-- Define the conditions
def symmetric_about_negative_one (f : ℝ → ℝ) : Prop :=
  ∀ k : ℝ, f (-1 + k) = f (-1 - k)

def y_intercept_at_one (f : ℝ → ℝ) : Prop :=
  f 0 = 1

def x_axis_intercept_length (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 2 * Real.sqrt 2

-- Main theorem
theorem quadratic_function_unique_form (f : ℝ → ℝ) :
  quadratic_function f →
  symmetric_about_negative_one f →
  y_intercept_at_one f →
  x_axis_intercept_length f →
  ∀ x, f x = -x^2 - 2*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_form_l1415_141500


namespace NUMINAMATH_CALUDE_closest_to_sqrt_difference_l1415_141537

theorem closest_to_sqrt_difference : 
  let diff := Real.sqrt 145 - Real.sqrt 141
  ∀ x ∈ ({0.19, 0.20, 0.21, 0.22} : Set ℝ), 
    |diff - 0.18| < |diff - x| := by
  sorry

end NUMINAMATH_CALUDE_closest_to_sqrt_difference_l1415_141537


namespace NUMINAMATH_CALUDE_sallys_nickels_l1415_141520

/-- The number of nickels Sally has after receiving some from her parents -/
def total_nickels (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem: Sally's total nickels equals the sum of her initial nickels and those received from parents -/
theorem sallys_nickels (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) :
  total_nickels initial from_dad from_mom = initial + from_dad + from_mom := by
  sorry

end NUMINAMATH_CALUDE_sallys_nickels_l1415_141520


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l1415_141542

theorem incorrect_average_calculation (n : ℕ) (incorrect_num correct_num : ℝ) (correct_avg : ℝ) :
  n = 10 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 65 ∧ 
  correct_avg = 50 →
  ∃ (other_sum : ℝ),
    (other_sum + correct_num) / n = correct_avg ∧
    (other_sum + incorrect_num) / n = 46 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l1415_141542


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l1415_141563

theorem shirt_price_calculation (num_shirts : ℕ) (discount_rate : ℚ) (total_paid : ℚ) :
  num_shirts = 6 →
  discount_rate = 1/5 →
  total_paid = 240 →
  ∃ (regular_price : ℚ), regular_price = 50 ∧ 
    num_shirts * (regular_price * (1 - discount_rate)) = total_paid := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l1415_141563


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l1415_141553

theorem solution_to_system_of_equations :
  ∃ x y : ℚ, x + 2*y = 3 ∧ 9*x - 8*y = 5 ∧ x = 17/13 ∧ y = 11/13 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l1415_141553


namespace NUMINAMATH_CALUDE_range_of_a_l1415_141557

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, p x → q x a) →
  (∃ x : ℝ, q x a ∧ ¬p x) →
  (0 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1415_141557


namespace NUMINAMATH_CALUDE_fuel_cost_difference_l1415_141509

-- Define the parameters
def num_vans : ℝ := 6.0
def num_buses : ℝ := 8
def people_per_van : ℝ := 6
def people_per_bus : ℝ := 18
def van_distance : ℝ := 120
def bus_distance : ℝ := 150
def van_efficiency : ℝ := 20
def bus_efficiency : ℝ := 6
def van_fuel_cost : ℝ := 2.5
def bus_fuel_cost : ℝ := 3

-- Define the theorem
theorem fuel_cost_difference : 
  let van_total_distance := num_vans * van_distance
  let bus_total_distance := num_buses * bus_distance
  let van_fuel_consumed := van_total_distance / van_efficiency
  let bus_fuel_consumed := bus_total_distance / bus_efficiency
  let van_total_cost := van_fuel_consumed * van_fuel_cost
  let bus_total_cost := bus_fuel_consumed * bus_fuel_cost
  bus_total_cost - van_total_cost = 510 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_difference_l1415_141509


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1415_141569

theorem degree_to_radian_conversion (π : Real) : 
  ((-300 : Real) * (π / 180)) = (-5 * π / 3) := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1415_141569


namespace NUMINAMATH_CALUDE_lottery_probability_l1415_141515

def sharpBallCount : ℕ := 30
def prizeBallCount : ℕ := 50
def prizeBallsDrawn : ℕ := 6

theorem lottery_probability :
  (1 : ℚ) / sharpBallCount * (1 : ℚ) / (Nat.choose prizeBallCount prizeBallsDrawn) = 1 / 476721000 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l1415_141515


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1415_141534

theorem max_value_of_expression (a b c : ℝ) (h : a + 3 * b + c = 6) :
  (∀ x y z : ℝ, x + 3 * y + z = 6 → a * b + a * c + b * c ≥ x * y + x * z + y * z) →
  a * b + a * c + b * c = 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1415_141534


namespace NUMINAMATH_CALUDE_triangles_in_decagon_l1415_141551

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def numTrianglesInDecagon : ℕ := 120

/-- A regular decagon has 10 vertices -/
def numVerticesInDecagon : ℕ := 10

theorem triangles_in_decagon :
  numTrianglesInDecagon = (numVerticesInDecagon.choose 3) := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_decagon_l1415_141551


namespace NUMINAMATH_CALUDE_jeff_cabinet_count_l1415_141540

/-- The number of cabinets Jeff has after installation and removal -/
def total_cabinets : ℕ :=
  let initial_cabinets := 3
  let counters_with_double := 4
  let cabinets_per_double_counter := 2 * initial_cabinets
  let additional_cabinets := [3, 5, 7]
  let cabinets_to_remove := 2

  initial_cabinets + 
  counters_with_double * cabinets_per_double_counter + 
  additional_cabinets.sum - 
  cabinets_to_remove

theorem jeff_cabinet_count : total_cabinets = 37 := by
  sorry

end NUMINAMATH_CALUDE_jeff_cabinet_count_l1415_141540


namespace NUMINAMATH_CALUDE_fraction_equality_l1415_141579

theorem fraction_equality (x : ℚ) (c : ℚ) (h1 : c ≠ 0) (h2 : c ≠ 3) :
  (4 + x) / (5 + x) = c / (3 * c) → x = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1415_141579


namespace NUMINAMATH_CALUDE_thirty_first_never_sunday_l1415_141527

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the months of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

/-- The number of days in each month -/
def daysInMonth (m : Month) (isLeapYear : Bool) : Nat :=
  match m with
  | Month.February => if isLeapYear then 29 else 28
  | Month.April | Month.June | Month.September | Month.November => 30
  | _ => 31

/-- The theorem stating that 31 is the only date that can never be a Sunday -/
theorem thirty_first_never_sunday :
  ∃! (date : Nat), date > 0 ∧ date ≤ 31 ∧
  ∀ (year : Nat) (m : Month),
    daysInMonth m (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) ≥ date →
    ∃ (dow : DayOfWeek), dow ≠ DayOfWeek.Sunday :=
by
  sorry

end NUMINAMATH_CALUDE_thirty_first_never_sunday_l1415_141527


namespace NUMINAMATH_CALUDE_negative_fraction_greater_than_negative_decimal_l1415_141522

theorem negative_fraction_greater_than_negative_decimal : -3/4 > -0.8 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_greater_than_negative_decimal_l1415_141522


namespace NUMINAMATH_CALUDE_train_speed_l1415_141564

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 441) (h2 : time = 21) :
  length / time = 21 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1415_141564


namespace NUMINAMATH_CALUDE_andrew_age_proof_l1415_141530

/-- Andrew's age in years -/
def andrew_age : ℚ := 30 / 7

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℚ := 15 * andrew_age

theorem andrew_age_proof :
  (grandfather_age - andrew_age = 60) ∧ (grandfather_age = 15 * andrew_age) → andrew_age = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_andrew_age_proof_l1415_141530


namespace NUMINAMATH_CALUDE_pool_visitors_l1415_141502

theorem pool_visitors (total_earned : ℚ) (cost_per_person : ℚ) (amount_left : ℚ) 
  (h1 : total_earned = 30)
  (h2 : cost_per_person = 5/2)
  (h3 : amount_left = 5) :
  (total_earned - amount_left) / cost_per_person = 10 := by
  sorry

end NUMINAMATH_CALUDE_pool_visitors_l1415_141502


namespace NUMINAMATH_CALUDE_quadratic_roots_relations_l1415_141514

/-- Given complex numbers a, b, c satisfying certain conditions, prove specific algebraic relations -/
theorem quadratic_roots_relations (a b c : ℂ) 
  (h1 : a + b ≠ 0)
  (h2 : b + c ≠ 0)
  (h3 : c + a ≠ 0)
  (h4 : ∀ (x : ℂ), (x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0) → 
    ∃ (y : ℂ), y^2 + a*y + b = 0 ∧ y^2 + b*y + c = 0 ∧ x = -y)
  (h5 : ∀ (x : ℂ), (x^2 + b*x + c = 0 ∧ x^2 + c*x + a = 0) → 
    ∃ (y : ℂ), y^2 + b*y + c = 0 ∧ y^2 + c*y + a = 0 ∧ x = -y)
  (h6 : ∀ (x : ℂ), (x^2 + c*x + a = 0 ∧ x^2 + a*x + b = 0) → 
    ∃ (y : ℂ), y^2 + c*y + a = 0 ∧ y^2 + a*y + b = 0 ∧ x = -y) :
  a^2 + b^2 + c^2 = 18 ∧ 
  a^2*b + b^2*c + c^2*a = 27 ∧ 
  a^3*b^2 + b^3*c^2 + c^3*a^2 = -162 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relations_l1415_141514


namespace NUMINAMATH_CALUDE_special_multiples_count_l1415_141519

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def count_special_multiples (n : ℕ) : ℕ :=
  count_multiples n 3 + count_multiples n 4 - count_multiples n 6

theorem special_multiples_count :
  count_special_multiples 3000 = 1250 := by sorry

end NUMINAMATH_CALUDE_special_multiples_count_l1415_141519


namespace NUMINAMATH_CALUDE_circle_equation_l1415_141548

-- Define the circle ⊙C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def tangent_to_x_axis (c : Circle) : Prop :=
  c.center.2 = c.radius

def center_on_line (c : Circle) : Prop :=
  3 * c.center.1 = c.center.2

def intersects_line (c : Circle) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    A.1 - A.2 = 0 ∧ B.1 - B.2 = 0 ∧
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2

def triangle_area (c : Circle) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    A.1 - A.2 = 0 ∧ B.1 - B.2 = 0 ∧
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2 ∧
    1/2 * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * 
    (|c.center.1 - c.center.2| / Real.sqrt 2) = Real.sqrt 14

-- Theorem statement
theorem circle_equation (c : Circle) :
  tangent_to_x_axis c →
  center_on_line c →
  intersects_line c →
  triangle_area c →
  ((c.center.1 = 1 ∧ c.center.2 = 3 ∧ c.radius = 3) ∨
   (c.center.1 = -1 ∧ c.center.2 = -3 ∧ c.radius = 3)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1415_141548


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1415_141592

theorem inequality_solution_set (x : ℝ) :
  -x^2 + 4*x + 5 < 0 ↔ x > 5 ∨ x < -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1415_141592


namespace NUMINAMATH_CALUDE_trigonometric_sum_l1415_141513

theorem trigonometric_sum (θ φ : Real) 
  (h : (Real.cos θ)^6 / (Real.cos φ)^2 + (Real.sin θ)^6 / (Real.sin φ)^2 = 1) :
  (Real.sin φ)^6 / (Real.sin θ)^2 + (Real.cos φ)^6 / (Real.cos θ)^2 = (1 + (Real.cos (2 * φ))^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_l1415_141513
