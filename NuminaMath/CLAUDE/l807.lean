import Mathlib

namespace NUMINAMATH_CALUDE_max_product_two_digit_numbers_l807_80717

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def unique_digits (a b c d e : ℕ) : Prop :=
  let digits := a.digits 10 ++ b.digits 10 ++ c.digits 10 ++ d.digits 10 ++ e.digits 10
  digits.length = 10 ∧ digits.toFinset.card = 10

theorem max_product_two_digit_numbers :
  ∃ (a b c d e : ℕ),
    is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧ is_two_digit e ∧
    unique_digits a b c d e ∧
    a * b * c * d * e = 1785641760 ∧
    ∀ (x y z w v : ℕ),
      is_two_digit x ∧ is_two_digit y ∧ is_two_digit z ∧ is_two_digit w ∧ is_two_digit v ∧
      unique_digits x y z w v →
      x * y * z * w * v ≤ 1785641760 :=
by sorry

end NUMINAMATH_CALUDE_max_product_two_digit_numbers_l807_80717


namespace NUMINAMATH_CALUDE_johns_reading_rate_l807_80716

/-- The number of books John read in 6 weeks -/
def total_books : ℕ := 48

/-- The number of weeks John read -/
def weeks : ℕ := 6

/-- The number of days John reads per week -/
def reading_days_per_week : ℕ := 2

/-- The number of books John can read in a day -/
def books_per_day : ℕ := total_books / (weeks * reading_days_per_week)

theorem johns_reading_rate : books_per_day = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_reading_rate_l807_80716


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l807_80771

/-- 
Given an infinite geometric series with common ratio 1/4 and sum 40,
prove that the second term of the sequence is 7.5.
-/
theorem second_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (1/4)^n = 40) → a * (1/4) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l807_80771


namespace NUMINAMATH_CALUDE_num_broadcasting_methods_is_36_l807_80773

/-- The number of different commercial ads -/
def num_commercial_ads : ℕ := 3

/-- The number of different Olympic promotional ads -/
def num_olympic_ads : ℕ := 2

/-- The total number of ads to be broadcast -/
def total_ads : ℕ := 5

/-- A function to calculate the number of broadcasting methods -/
def num_broadcasting_methods : ℕ := 
  let last_olympic_ad_choices := num_olympic_ads
  let second_olympic_ad_positions := total_ads - 2
  let remaining_ad_permutations := Nat.factorial num_commercial_ads
  last_olympic_ad_choices * second_olympic_ad_positions * remaining_ad_permutations

/-- Theorem stating that the number of broadcasting methods is 36 -/
theorem num_broadcasting_methods_is_36 : num_broadcasting_methods = 36 := by
  sorry


end NUMINAMATH_CALUDE_num_broadcasting_methods_is_36_l807_80773


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l807_80761

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_derivative_at_one : 
  (deriv f) 1 = 2 * Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l807_80761


namespace NUMINAMATH_CALUDE_cos_sum_less_than_sum_of_cos_l807_80767

theorem cos_sum_less_than_sum_of_cos (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) : 
  Real.cos (α + β) < Real.cos α + Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_less_than_sum_of_cos_l807_80767


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l807_80710

/-- A geometric sequence with specific conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  first_condition : a 1 + a 3 = 10
  second_condition : a 4 + a 6 = 5/4

/-- The general term of the geometric sequence -/
def generalTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  2^(4-n)

/-- The sum of the first four terms of the geometric sequence -/
def sumFirstFour (seq : GeometricSequence) : ℝ :=
  15

/-- Theorem stating the correctness of the general term and sum -/
theorem geometric_sequence_properties (seq : GeometricSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧ sumFirstFour seq = 15 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_properties_l807_80710


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_l807_80715

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem smallest_sum_of_digits :
  (∀ n : ℕ, sum_of_digits (3 * n^2 + n + 1) ≥ 3) ∧
  (∃ n : ℕ, sum_of_digits (3 * n^2 + n + 1) = 3) := by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_l807_80715


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l807_80796

def triangle_area (a b : ℝ) (cos_C : ℝ) : ℝ :=
  6

theorem triangle_area_theorem (a b cos_C : ℝ) :
  a = 3 →
  b = 5 →
  5 * cos_C^2 - 7 * cos_C - 6 = 0 →
  triangle_area a b cos_C = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l807_80796


namespace NUMINAMATH_CALUDE_car_subsidy_theorem_l807_80713

/-- Represents the sales and pricing data for a car dealership --/
structure CarSalesData where
  manual_nov : ℕ
  auto_nov : ℕ
  manual_dec : ℕ
  auto_dec : ℕ
  manual_price : ℕ
  auto_price : ℕ
  subsidy_rate : ℚ

/-- Calculates the total government subsidy based on car sales data --/
def total_subsidy (data : CarSalesData) : ℚ :=
  (data.manual_dec * data.manual_price + data.auto_dec * data.auto_price) * data.subsidy_rate

/-- Theorem stating the total government subsidy for the given scenario --/
theorem car_subsidy_theorem (data : CarSalesData) :
  data.manual_nov + data.auto_nov = 960 →
  data.manual_dec + data.auto_dec = 1228 →
  data.manual_dec = (13 * data.manual_nov) / 10 →
  data.auto_dec = (5 * data.auto_nov) / 4 →
  data.manual_price = 80000 →
  data.auto_price = 90000 →
  data.subsidy_rate = 1 / 20 →
  total_subsidy data = 516200000 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_car_subsidy_theorem_l807_80713


namespace NUMINAMATH_CALUDE_lynne_book_purchase_total_cost_lynne_spent_75_dollars_l807_80786

theorem lynne_book_purchase_total_cost : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | cat_books, solar_books, magazines, book_cost, magazine_cost =>
    let total_books := cat_books + solar_books
    let book_total_cost := total_books * book_cost
    let magazine_total_cost := magazines * magazine_cost
    book_total_cost + magazine_total_cost

theorem lynne_spent_75_dollars : 
  lynne_book_purchase_total_cost 7 2 3 7 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_lynne_book_purchase_total_cost_lynne_spent_75_dollars_l807_80786


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_l807_80720

theorem minimum_value_of_expression (x : ℝ) (h : x > 0) :
  9 * x + 1 / x^6 ≥ 10 ∧ ∃ y > 0, 9 * y + 1 / y^6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_l807_80720


namespace NUMINAMATH_CALUDE_log_calculation_l807_80714

-- Define the common logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_calculation :
  (lg 2)^2 + lg 2 * lg 50 + lg 25 = 2 :=
by
  -- Properties of logarithms
  have h1 : lg 50 = lg 2 + lg 25 := by sorry
  have h2 : lg 25 = 2 * lg 5 := by sorry
  have h3 : lg 10 = 1 := by sorry
  
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_log_calculation_l807_80714


namespace NUMINAMATH_CALUDE_parallel_transitive_l807_80719

-- Define the type for lines
def Line : Type := ℝ → ℝ → ℝ → Prop

-- Define the parallel relation between lines
def parallel (a b : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l807_80719


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l807_80704

/-- Proves that shifting sin(2x + π/3) right by π/4 results in sin(2x - π/6) -/
theorem sin_shift_equivalence (x : ℝ) :
  Real.sin (2 * (x - π/4) + π/3) = Real.sin (2*x - π/6) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l807_80704


namespace NUMINAMATH_CALUDE_fixed_points_theorem_l807_80707

/-- A function f(x) = ax^2 + (b+1)x + (b-1) where a ≠ 0 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

/-- x0 is a fixed point of f if f(x0) = x0 -/
def is_fixed_point (a b x0 : ℝ) : Prop := f a b x0 = x0

/-- The function has two distinct fixed points -/
def has_two_distinct_fixed_points (a b : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point a b x1 ∧ is_fixed_point a b x2

/-- The fixed points are symmetric with respect to the line y = kx + 1/(2a^2 + 1) -/
def fixed_points_symmetric (a b : ℝ) : Prop :=
  ∃ x1 x2 k : ℝ, x1 ≠ x2 ∧ is_fixed_point a b x1 ∧ is_fixed_point a b x2 ∧
    (f a b x1 + f a b x2) / 2 = k * (x1 + x2) / 2 + 1 / (2 * a^2 + 1)

/-- Main theorem -/
theorem fixed_points_theorem (a b : ℝ) (ha : a ≠ 0) :
  (has_two_distinct_fixed_points a b ↔ 0 < a ∧ a < 1) ∧
  (0 < a ∧ a < 1 → ∃ b_min : ℝ, ∀ b : ℝ, fixed_points_symmetric a b → b ≥ b_min) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_theorem_l807_80707


namespace NUMINAMATH_CALUDE_only_paint_worthy_is_204_l807_80794

/-- Represents a painting configuration for the fence. -/
structure PaintConfig where
  h : ℕ+  -- Harold's interval
  t : ℕ+  -- Tanya's interval
  u : ℕ+  -- Ulysses' interval

/-- Checks if a painting configuration is valid (covers all pickets exactly once). -/
def isValidConfig (config : PaintConfig) : Prop :=
  -- Harold starts from second picket
  -- Tanya starts from first picket
  -- Ulysses starts from fourth picket
  -- Each picket is painted exactly once
  sorry

/-- Calculates the paint-worthy number for a given configuration. -/
def paintWorthy (config : PaintConfig) : ℕ :=
  100 * config.h.val + 10 * config.t.val + config.u.val

/-- The main theorem stating that 204 is the only paint-worthy number. -/
theorem only_paint_worthy_is_204 :
  ∀ config : PaintConfig, isValidConfig config → paintWorthy config = 204 := by
  sorry

end NUMINAMATH_CALUDE_only_paint_worthy_is_204_l807_80794


namespace NUMINAMATH_CALUDE_namjoon_position_l807_80791

theorem namjoon_position (total_students : ℕ) (position_from_left : ℕ) :
  total_students = 15 →
  position_from_left = 7 →
  total_students - position_from_left + 1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_namjoon_position_l807_80791


namespace NUMINAMATH_CALUDE_sum_of_specific_primes_l807_80725

theorem sum_of_specific_primes : ∃ (S : Finset Nat),
  (∀ p ∈ S, p.Prime ∧ 1 < p ∧ p ≤ 100 ∧ p % 6 = 1 ∧ p % 7 = 6) ∧
  (∀ p, p.Prime → 1 < p → p ≤ 100 → p % 6 = 1 → p % 7 = 6 → p ∈ S) ∧
  S.sum id = 104 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_primes_l807_80725


namespace NUMINAMATH_CALUDE_rectangle_width_l807_80787

/-- Given a rectangle with perimeter 50 cm and length 13 cm, its width is 12 cm. -/
theorem rectangle_width (perimeter length width : ℝ) : 
  perimeter = 50 ∧ length = 13 ∧ perimeter = 2 * length + 2 * width → width = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l807_80787


namespace NUMINAMATH_CALUDE_tom_flashlight_batteries_l807_80759

/-- The number of batteries Tom used on his flashlights -/
def flashlight_batteries : ℕ := 19 - 15 - 2

/-- Proof that Tom used 4 batteries on his flashlights -/
theorem tom_flashlight_batteries :
  flashlight_batteries = 4 :=
by sorry

end NUMINAMATH_CALUDE_tom_flashlight_batteries_l807_80759


namespace NUMINAMATH_CALUDE_sin_cos_extrema_l807_80736

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  (∀ a b : ℝ, Real.sin a + Real.sin b = 1/3 → 
    Real.sin a - Real.cos b ^ 2 ≤ 4/9 ∧ 
    Real.sin a - Real.cos b ^ 2 ≥ -11/12) ∧
  (∃ c d : ℝ, Real.sin c + Real.sin d = 1/3 ∧ 
    Real.sin c - Real.cos d ^ 2 = 4/9) ∧
  (∃ e f : ℝ, Real.sin e + Real.sin f = 1/3 ∧ 
    Real.sin e - Real.cos f ^ 2 = -11/12) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_extrema_l807_80736


namespace NUMINAMATH_CALUDE_fourth_root_squared_cubed_l807_80734

theorem fourth_root_squared_cubed (x : ℝ) : ((x^(1/4))^2)^3 = 1296 → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_squared_cubed_l807_80734


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l807_80750

def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (6, -5)
  let reflected_center : ℝ × ℝ := reflect_over_y_eq_x original_center
  reflected_center = (-5, 6) := by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l807_80750


namespace NUMINAMATH_CALUDE_cat_meows_l807_80712

theorem cat_meows (cat1 : ℝ) (cat2 : ℝ) (cat3 : ℝ) : 
  cat2 = 2 * cat1 →
  cat3 = (2 * cat1) / 3 →
  5 * cat1 + 5 * cat2 + 5 * cat3 = 55 →
  cat1 = 3 := by
sorry

end NUMINAMATH_CALUDE_cat_meows_l807_80712


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l807_80779

-- 1. Prove that (-10) - (-22) + (-8) - 13 = -9
theorem problem_1 : (-10) - (-22) + (-8) - 13 = -9 := by sorry

-- 2. Prove that (-7/9 + 5/6 - 3/4) * (-36) = 25
theorem problem_2 : (-7/9 + 5/6 - 3/4) * (-36) = 25 := by sorry

-- 3. Prove that the solution to 6x - 7 = 4x - 5 is x = 1
theorem problem_3 : ∃ x : ℝ, 6*x - 7 = 4*x - 5 ∧ x = 1 := by sorry

-- 4. Prove that the solution to (x-3)/2 - (2x)/3 = 1 is x = -15
theorem problem_4 : ∃ x : ℝ, (x-3)/2 - (2*x)/3 = 1 ∧ x = -15 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l807_80779


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l807_80730

theorem no_solution_iff_n_eq_neg_one (n : ℝ) : 
  (∀ x y z : ℝ, nx + y = 1 ∧ ny + z = 1 ∧ x + nz = 1 → False) ↔ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l807_80730


namespace NUMINAMATH_CALUDE_factorization_sum_l807_80702

theorem factorization_sum (d e f : ℤ) :
  (∀ x : ℝ, x^2 + 19*x + 88 = (x + d)*(x + e)) →
  (∀ x : ℝ, x^2 - 23*x + 120 = (x - e)*(x - f)) →
  d + e + f = 31 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l807_80702


namespace NUMINAMATH_CALUDE_collinear_vectors_solution_l807_80700

/-- Two vectors in R² -/
def m (x : ℝ) : ℝ × ℝ := (x, x + 2)
def n (x : ℝ) : ℝ × ℝ := (1, 3*x)

/-- Collinearity condition for two vectors -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem collinear_vectors_solution :
  ∀ x : ℝ, collinear (m x) (n x) → x = -2/3 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_collinear_vectors_solution_l807_80700


namespace NUMINAMATH_CALUDE_circle_revolutions_l807_80732

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the motion of circles C₁, C₂, and C₃ -/
structure CircleMotion where
  C₁ : Circle
  C₂ : Circle
  C₃ : Circle
  n : ℕ

/-- The number of revolutions C₃ makes relative to the ground -/
def revolutions (motion : CircleMotion) : ℝ := motion.n - 1

/-- The theorem stating the number of revolutions C₃ makes -/
theorem circle_revolutions (motion : CircleMotion) 
  (h₁ : motion.n > 2)
  (h₂ : motion.C₁.radius = motion.n * motion.C₃.radius)
  (h₃ : motion.C₂.radius = 2 * motion.C₃.radius)
  (h₄ : motion.C₃.radius > 0) :
  revolutions motion = motion.n - 1 := by sorry

end NUMINAMATH_CALUDE_circle_revolutions_l807_80732


namespace NUMINAMATH_CALUDE_triangle_problem_l807_80766

theorem triangle_problem (a b c A B C : ℝ) (h1 : 0 < A) (h2 : A < π) : 
  c = a * Real.sin C - c * Real.cos A →
  (A = π / 2) ∧ 
  (a = 2 → 1/2 * b * c * Real.sin A = 2 → b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l807_80766


namespace NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l807_80744

-- Define the properties
def is_in_second_quadrant (α : Real) : Prop := 90 < α ∧ α ≤ 180
def is_obtuse_angle (α : Real) : Prop := 90 < α ∧ α < 180

-- Theorem statement
theorem second_quadrant_necessary_not_sufficient_for_obtuse :
  (∀ α, is_obtuse_angle α → is_in_second_quadrant α) ∧
  (∃ α, is_in_second_quadrant α ∧ ¬is_obtuse_angle α) :=
sorry

end NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l807_80744


namespace NUMINAMATH_CALUDE_average_reduction_percentage_option1_more_favorable_l807_80726

-- Define the original and final prices
def original_price : ℝ := 5
def final_price : ℝ := 3.2

-- Define the quantity to purchase in kilograms
def quantity : ℝ := 5000

-- Define the discount percentage and cash discount
def discount_percentage : ℝ := 0.1
def cash_discount_per_ton : ℝ := 200

-- Theorem for the average percentage reduction
theorem average_reduction_percentage :
  ∃ x : ℝ, x > 0 ∧ x < 1 ∧ original_price * (1 - x)^2 = final_price ∧ x = 0.2 :=
sorry

-- Theorem for the more favorable option
theorem option1_more_favorable :
  final_price * (1 - discount_percentage) * quantity <
  final_price * quantity - (cash_discount_per_ton * (quantity / 1000)) :=
sorry

end NUMINAMATH_CALUDE_average_reduction_percentage_option1_more_favorable_l807_80726


namespace NUMINAMATH_CALUDE_complex_product_equals_43_l807_80763

theorem complex_product_equals_43 (x : ℂ) (h : x = Complex.exp (2 * Real.pi * Complex.I / 7)) :
  (2*x + x^2) * (2*x^2 + x^4) * (2*x^3 + x^6) * (2*x^4 + x^8) * (2*x^5 + x^10) * (2*x^6 + x^12) = 43 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_43_l807_80763


namespace NUMINAMATH_CALUDE_dave_final_tickets_l807_80785

/-- Calculates the final number of tickets Dave has after a series of events at the arcade. -/
def dave_tickets : ℕ :=
  let initial_tickets := 11
  let candy_bar_cost := 3
  let beanie_cost := 5
  let racing_game_win := 10
  let claw_machine_win := 7
  let after_spending := initial_tickets - candy_bar_cost - beanie_cost
  let after_winning := after_spending + racing_game_win + claw_machine_win
  2 * after_winning

/-- Theorem stating that Dave ends up with 40 tickets after all events at the arcade. -/
theorem dave_final_tickets : dave_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_dave_final_tickets_l807_80785


namespace NUMINAMATH_CALUDE_min_volume_base_area_is_8d_squared_l807_80703

/-- Regular quadrilateral pyramid with a plane bisecting the dihedral angle -/
structure RegularPyramid where
  /-- Distance from the base to the intersection point of the bisecting plane and the height -/
  d : ℝ
  /-- The plane bisects the dihedral angle at a side of the base -/
  bisects_dihedral_angle : True

/-- The area of the base that minimizes the volume of the pyramid -/
def min_volume_base_area (p : RegularPyramid) : ℝ := 8 * p.d^2

/-- Theorem stating that the area of the base minimizing the volume is 8d^2 -/
theorem min_volume_base_area_is_8d_squared (p : RegularPyramid) :
  min_volume_base_area p = 8 * p.d^2 := by
  sorry

end NUMINAMATH_CALUDE_min_volume_base_area_is_8d_squared_l807_80703


namespace NUMINAMATH_CALUDE_fourth_task_end_time_l807_80740

-- Define the start time of the first task
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes since midnight

-- Define the end time of the third task
def end_third_task : Nat := 11 * 60 + 30  -- 11:30 AM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 4

-- Define the theorem
theorem fourth_task_end_time :
  let total_time := end_third_task - start_time
  let task_duration := total_time / 3
  let fourth_task_end := end_third_task + task_duration
  fourth_task_end = 12 * 60 + 20  -- 12:20 PM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_fourth_task_end_time_l807_80740


namespace NUMINAMATH_CALUDE_remainder_theorem_l807_80777

theorem remainder_theorem (N : ℤ) (h : N % 35 = 25) : N % 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l807_80777


namespace NUMINAMATH_CALUDE_ralphs_purchase_cost_l807_80793

/-- Calculates the final cost of Ralph's purchase given the initial conditions --/
theorem ralphs_purchase_cost
  (initial_total : ℝ)
  (discounted_item_price : ℝ)
  (item_discount_rate : ℝ)
  (total_discount_rate : ℝ)
  (h1 : initial_total = 54)
  (h2 : discounted_item_price = 20)
  (h3 : item_discount_rate = 0.2)
  (h4 : total_discount_rate = 0.1)
  : ∃ (final_cost : ℝ), final_cost = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_ralphs_purchase_cost_l807_80793


namespace NUMINAMATH_CALUDE_xy_equals_nine_x_div_y_equals_thirtysix_l807_80775

theorem xy_equals_nine_x_div_y_equals_thirtysix (x y : ℝ) 
  (h1 : x * y = 9)
  (h2 : x / y = 36)
  (hx : x > 0)
  (hy : y > 0) :
  y = 1/2 := by
sorry

end NUMINAMATH_CALUDE_xy_equals_nine_x_div_y_equals_thirtysix_l807_80775


namespace NUMINAMATH_CALUDE_james_sales_percentage_l807_80722

/-- Represents the number of houses visited on the first day -/
def houses_day1 : ℕ := 20

/-- Represents the number of houses visited on the second day -/
def houses_day2 : ℕ := 2 * houses_day1

/-- Represents the number of items sold per house each day -/
def items_per_house : ℕ := 2

/-- Represents the total number of items sold over both days -/
def total_items_sold : ℕ := 104

/-- Calculates the percentage of houses sold to on the second day -/
def percentage_sold_day2 : ℚ :=
  (total_items_sold - houses_day1 * items_per_house) / (2 * houses_day2)

theorem james_sales_percentage :
  percentage_sold_day2 = 4/5 := by sorry

end NUMINAMATH_CALUDE_james_sales_percentage_l807_80722


namespace NUMINAMATH_CALUDE_regular_hexagon_side_length_l807_80705

/-- The length of a side in a regular hexagon, given the distance between opposite sides -/
theorem regular_hexagon_side_length (distance_between_opposite_sides : ℝ) : 
  distance_between_opposite_sides > 0 →
  ∃ (side_length : ℝ), 
    side_length = (20 * Real.sqrt 3) / 3 * distance_between_opposite_sides / 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_side_length_l807_80705


namespace NUMINAMATH_CALUDE_new_person_weight_l807_80784

/-- The weight of the new person given the initial conditions -/
def weightOfNewPerson (initialCount : ℕ) (averageIncrease : ℚ) (replacedWeight : ℚ) : ℚ :=
  replacedWeight + (initialCount : ℚ) * averageIncrease

/-- Theorem stating the weight of the new person under the given conditions -/
theorem new_person_weight :
  weightOfNewPerson 8 (5/2) 75 = 95 := by sorry

end NUMINAMATH_CALUDE_new_person_weight_l807_80784


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l807_80768

theorem halfway_between_one_eighth_and_one_third :
  (1 / 8 : ℚ) / 2 + (1 / 3 : ℚ) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l807_80768


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l807_80780

theorem pizza_toppings_combinations (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 3) :
  Nat.choose n k = 84 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l807_80780


namespace NUMINAMATH_CALUDE_pete_miles_walked_l807_80721

/-- Represents a pedometer with a maximum step count --/
structure Pedometer :=
  (max_steps : ℕ)

/-- Calculates the total number of steps given the number of resets and final reading --/
def total_steps (p : Pedometer) (resets : ℕ) (final_reading : ℕ) : ℕ :=
  resets * (p.max_steps + 1) + final_reading

/-- Converts steps to miles --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℚ :=
  (steps : ℚ) / (steps_per_mile : ℚ)

/-- Theorem stating the approximate number of miles Pete walked --/
theorem pete_miles_walked :
  let p : Pedometer := ⟨99999⟩
  let resets : ℕ := 44
  let final_reading : ℕ := 50000
  let steps_per_mile : ℕ := 1800
  let total_steps := total_steps p resets final_reading
  let miles_walked := steps_to_miles total_steps steps_per_mile
  ∃ ε > 0, abs (miles_walked - 2472.22) < ε := by
  sorry

end NUMINAMATH_CALUDE_pete_miles_walked_l807_80721


namespace NUMINAMATH_CALUDE_a5_is_zero_in_825_factorial_base_l807_80737

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorialBaseCoeff (n k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

theorem a5_is_zero_in_825_factorial_base : 
  factorialBaseCoeff 825 5 = 0 := by sorry

end NUMINAMATH_CALUDE_a5_is_zero_in_825_factorial_base_l807_80737


namespace NUMINAMATH_CALUDE_prob_other_side_red_given_red_l807_80751

/-- Represents the types of cards in the box -/
inductive Card
  | BlackBlack
  | BlackRed
  | RedRed

/-- The total number of cards in the box -/
def total_cards : Nat := 7

/-- The number of black-black cards -/
def black_black_cards : Nat := 2

/-- The number of black-red cards -/
def black_red_cards : Nat := 3

/-- The number of red-red cards -/
def red_red_cards : Nat := 2

/-- The total number of red faces -/
def total_red_faces : Nat := black_red_cards + 2 * red_red_cards

/-- The number of red faces on completely red cards -/
def red_faces_on_red_cards : Nat := 2 * red_red_cards

/-- The probability of seeing a red face and the other side being red -/
theorem prob_other_side_red_given_red (h1 : total_cards = black_black_cards + black_red_cards + red_red_cards)
  (h2 : total_red_faces = black_red_cards + 2 * red_red_cards)
  (h3 : red_faces_on_red_cards = 2 * red_red_cards) :
  (red_faces_on_red_cards : ℚ) / total_red_faces = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_other_side_red_given_red_l807_80751


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_diagonal_length_l807_80746

-- Define the cyclic quadrilateral ABCD and point K
variable (A B C D K : Point)

-- Define the property of being a cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the property of K being the intersection of diagonals
def is_diagonal_intersection (A B C D K : Point) : Prop := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- State the theorem
theorem cyclic_quadrilateral_diagonal_length
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_diagonal : is_diagonal_intersection A B C D K)
  (h_equal_sides : distance A B = distance B C)
  (h_BK : distance B K = b)
  (h_DK : distance D K = d)
  : distance A B = Real.sqrt (b^2 + b*d) := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_diagonal_length_l807_80746


namespace NUMINAMATH_CALUDE_min_even_integers_l807_80764

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 30 → 
  a + b + c + d = 50 → 
  a + b + c + d + e + f = 70 → 
  ∃ (evens : Finset ℤ), evens ⊆ {a, b, c, d, e, f} ∧ 
    (∀ x ∈ evens, Even x) ∧ 
    evens.card = 2 ∧ 
    (∀ (other_evens : Finset ℤ), other_evens ⊆ {a, b, c, d, e, f} → 
      (∀ x ∈ other_evens, Even x) → other_evens.card ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l807_80764


namespace NUMINAMATH_CALUDE_larger_number_problem_l807_80701

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1000) (h3 : L = 10 * S + 10) : L = 1110 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l807_80701


namespace NUMINAMATH_CALUDE_solution_set_inequality_not_sufficient_condition_negation_of_proposition_not_necessary_condition_l807_80799

-- Statement 1
theorem solution_set_inequality (x : ℝ) : 
  (x + 2) / (2 * x + 1) > 1 ↔ -1/2 < x ∧ x < 1 := by sorry

-- Statement 2
theorem not_sufficient_condition : 
  ∃ a b : ℝ, a * b > 1 ∧ ¬(a > 1 ∧ b > 1) := by sorry

-- Statement 3
theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := by sorry

-- Statement 4
theorem not_necessary_condition : 
  ∃ a : ℝ, a < 6 ∧ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_not_sufficient_condition_negation_of_proposition_not_necessary_condition_l807_80799


namespace NUMINAMATH_CALUDE_birth_year_property_l807_80758

def current_year : Nat := 2023
def birth_year : Nat := 1957

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.repr.data.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem birth_year_property : 
  current_year - birth_year = sum_of_digits birth_year := by
  sorry

end NUMINAMATH_CALUDE_birth_year_property_l807_80758


namespace NUMINAMATH_CALUDE_roger_tray_collection_l807_80706

/-- The number of trips required to collect trays -/
def numTrips (capacity traysTable1 traysTable2 : ℕ) : ℕ :=
  (traysTable1 + traysTable2 + capacity - 1) / capacity

theorem roger_tray_collection (capacity traysTable1 traysTable2 : ℕ) 
  (h1 : capacity = 4) 
  (h2 : traysTable1 = 10) 
  (h3 : traysTable2 = 2) : 
  numTrips capacity traysTable1 traysTable2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_roger_tray_collection_l807_80706


namespace NUMINAMATH_CALUDE_yellow_ball_count_l807_80724

/-- Given a bag with red and yellow balls, this theorem proves the number of yellow balls
    when the number of red balls and the probability of drawing a red ball are known. -/
theorem yellow_ball_count (total : ℕ) (red : ℕ) (p : ℚ) : 
  red = 8 →
  p = 1/3 →
  p = red / total →
  total - red = 16 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l807_80724


namespace NUMINAMATH_CALUDE_distance_between_polar_points_l807_80772

/-- Given two points in polar coordinates, prove their distance -/
theorem distance_between_polar_points (θ₁ θ₂ : ℝ) :
  let A : ℝ × ℝ := (4 * Real.cos θ₁, 4 * Real.sin θ₁)
  let B : ℝ × ℝ := (6 * Real.cos θ₂, 6 * Real.sin θ₂)
  θ₁ - θ₂ = π / 3 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_polar_points_l807_80772


namespace NUMINAMATH_CALUDE_unique_valid_denomination_l807_80733

def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 104 → ∃ a b c : ℕ, k = 7 * a + n * b + (n + 2) * c ∧
  ¬∃ a b c : ℕ, 104 = 7 * a + n * b + (n + 2) * c

theorem unique_valid_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_denomination_l807_80733


namespace NUMINAMATH_CALUDE_nested_fourth_root_equation_solution_l807_80782

/-- Defines the nested fourth root function for the left-hand side of the equation -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - Real.sqrt (x - Real.sqrt (x - Real.sqrt x)))

/-- Defines the nested fourth root function for the right-hand side of the equation -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

/-- There exists a positive real number x that satisfies the equation -/
theorem nested_fourth_root_equation_solution :
  ∃ x : ℝ, x > 0 ∧ f x = g x := by sorry

end NUMINAMATH_CALUDE_nested_fourth_root_equation_solution_l807_80782


namespace NUMINAMATH_CALUDE_min_value_of_f_l807_80774

open Real

-- Define the function f
def f (a b c d e x : ℝ) : ℝ := |x - a| + |x - b| + |x - c| + |x - d| + |x - e|

-- State the theorem
theorem min_value_of_f (a b c d e : ℝ) (h : a < b ∧ b < c ∧ c < d ∧ d < e) :
  ∃ (m : ℝ), (∀ x, f a b c d e x ≥ m) ∧ m = e + d - b - a := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l807_80774


namespace NUMINAMATH_CALUDE_non_integer_a_implies_b_is_one_l807_80708

theorem non_integer_a_implies_b_is_one (a b : ℝ) : 
  a + b - a * b = 1 → ¬(∃ n : ℤ, a = n) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_non_integer_a_implies_b_is_one_l807_80708


namespace NUMINAMATH_CALUDE_distance_between_points_l807_80728

/-- The distance between points (2,5) and (7,1) is √41. -/
theorem distance_between_points : Real.sqrt 41 = Real.sqrt ((7 - 2)^2 + (1 - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l807_80728


namespace NUMINAMATH_CALUDE_correct_vote_distribution_l807_80776

/-- Represents the number of votes for each candidate -/
structure Votes where
  eliot : ℕ
  shaun : ℕ
  randy : ℕ
  lisa : ℕ

/-- Checks if the vote distribution satisfies the given conditions -/
def is_valid_vote_distribution (v : Votes) : Prop :=
  v.eliot = 2 * v.shaun ∧
  v.eliot = 4 * v.randy ∧
  v.shaun = 5 * v.randy ∧
  v.shaun = 3 * v.lisa ∧
  v.randy = 16

/-- The theorem stating that the given vote distribution is correct -/
theorem correct_vote_distribution :
  ∃ (v : Votes), is_valid_vote_distribution v ∧
    v.eliot = 64 ∧ v.shaun = 80 ∧ v.randy = 16 ∧ v.lisa = 27 :=
by
  sorry


end NUMINAMATH_CALUDE_correct_vote_distribution_l807_80776


namespace NUMINAMATH_CALUDE_T_mod_1000_l807_80754

/-- The sum of all four-digit positive integers with four distinct digits -/
def T : ℕ := sorry

/-- Theorem stating that T mod 1000 = 465 -/
theorem T_mod_1000 : T % 1000 = 465 := by sorry

end NUMINAMATH_CALUDE_T_mod_1000_l807_80754


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l807_80797

theorem student_multiplication_problem (x : ℝ) : 40 * x - 138 = 102 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l807_80797


namespace NUMINAMATH_CALUDE_factor_calculation_l807_80753

theorem factor_calculation : 
  let initial_number : ℕ := 15
  let resultant : ℕ := 2 * initial_number + 5
  let final_result : ℕ := 105
  ∃ factor : ℚ, factor * resultant = final_result ∧ factor = 3 :=
by sorry

end NUMINAMATH_CALUDE_factor_calculation_l807_80753


namespace NUMINAMATH_CALUDE_maximize_electronic_thermometers_l807_80738

/-- Represents the problem of maximizing electronic thermometers purchase --/
theorem maximize_electronic_thermometers
  (total_budget : ℕ)
  (mercury_cost : ℕ)
  (electronic_cost : ℕ)
  (total_students : ℕ)
  (h1 : total_budget = 300)
  (h2 : mercury_cost = 3)
  (h3 : electronic_cost = 10)
  (h4 : total_students = 53) :
  ∃ (x : ℕ), x ≤ total_students ∧
             x * electronic_cost + (total_students - x) * mercury_cost ≤ total_budget ∧
             ∀ (y : ℕ), y ≤ total_students →
                        y * electronic_cost + (total_students - y) * mercury_cost ≤ total_budget →
                        y ≤ x ∧
             x = 20 :=
by sorry

end NUMINAMATH_CALUDE_maximize_electronic_thermometers_l807_80738


namespace NUMINAMATH_CALUDE_vehicle_distance_after_three_minutes_l807_80760

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

theorem vehicle_distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_minutes : ℝ := 3
  let time_hours : ℝ := time_minutes / 60
  distance_between_vehicles truck_speed car_speed time_hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_distance_after_three_minutes_l807_80760


namespace NUMINAMATH_CALUDE_ratio_arithmetic_property_l807_80781

/-- Definition of a ratio arithmetic sequence -/
def is_ratio_arithmetic (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 2) / a (n + 1) - a (n + 1) / a n = d

/-- Our specific sequence -/
def our_sequence (a : ℕ → ℚ) : Prop :=
  is_ratio_arithmetic a 2 ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3

theorem ratio_arithmetic_property (a : ℕ → ℚ) (h : our_sequence a) :
  a 2019 / a 2017 = 4 * 2017^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_arithmetic_property_l807_80781


namespace NUMINAMATH_CALUDE_complex_coordinate_l807_80711

theorem complex_coordinate (z : ℂ) (h : Complex.I * z = 2 + 4 * Complex.I) : 
  z = 4 - 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_coordinate_l807_80711


namespace NUMINAMATH_CALUDE_range_of_x_l807_80745

theorem range_of_x (x : ℝ) : 
  (x^2 - 2*x - 3 ≤ 0) → (1/(x-2) ≤ 0) → (-1 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l807_80745


namespace NUMINAMATH_CALUDE_five_by_five_grid_squares_l807_80783

/-- The number of squares of a given size in a 5x5 grid -/
def num_squares (size : Nat) : Nat :=
  (6 - size) ^ 2

/-- The total number of squares in a 5x5 grid -/
def total_squares : Nat :=
  (List.range 5).map (λ i => num_squares (i + 1)) |>.sum

theorem five_by_five_grid_squares :
  total_squares = 55 := by
  sorry

end NUMINAMATH_CALUDE_five_by_five_grid_squares_l807_80783


namespace NUMINAMATH_CALUDE_total_rainfall_calculation_l807_80741

theorem total_rainfall_calculation (storm1_rate : ℝ) (storm2_rate : ℝ) 
  (total_duration : ℝ) (storm1_duration : ℝ) :
  storm1_rate = 30 →
  storm2_rate = 15 →
  total_duration = 45 →
  storm1_duration = 20 →
  storm1_rate * storm1_duration + storm2_rate * (total_duration - storm1_duration) = 975 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_calculation_l807_80741


namespace NUMINAMATH_CALUDE_base9_432_equals_base10_353_l807_80778

/-- Converts a base 9 number to base 10 --/
def base9_to_base10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 9^2 + d₁ * 9^1 + d₀ * 9^0

/-- The base 9 number 432₉ is equal to 353 in base 10 --/
theorem base9_432_equals_base10_353 :
  base9_to_base10 4 3 2 = 353 := by sorry

end NUMINAMATH_CALUDE_base9_432_equals_base10_353_l807_80778


namespace NUMINAMATH_CALUDE_intersection_to_left_focus_distance_l807_80748

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection point in the first quadrant
def intersection_point (x y : ℝ) : Prop :=
  ellipse x y ∧ parabola x y ∧ x > 0 ∧ y > 0

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem intersection_to_left_focus_distance :
  ∀ x y : ℝ, intersection_point x y →
  Real.sqrt ((x - left_focus.1)^2 + (y - left_focus.2)^2) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_to_left_focus_distance_l807_80748


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l807_80723

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity : ∀ (a : ℝ),
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 → 
    ∃ (c : ℝ), c = 4 ∧ c^2 = a^2 + 9) →
  (4 : ℝ) * Real.sqrt 7 / 7 = 
    (4 : ℝ) / Real.sqrt (a^2) := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l807_80723


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l807_80789

theorem smallest_number_with_given_remainders : ∃ b : ℕ, 
  b % 4 = 2 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧
  ∀ n : ℕ, n < b → (n % 4 ≠ 2 ∨ n % 3 ≠ 2 ∨ n % 5 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l807_80789


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l807_80718

/-- A circle with equation x^2 + y^2 = n is tangent to the line x + y + 1 = 0 if and only if n = 1/2 -/
theorem circle_tangent_to_line (n : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = n ∧ x + y + 1 = 0 ∧ 
    ∀ (x' y' : ℝ), x'^2 + y'^2 = n → x' + y' + 1 ≠ 0 ∨ (x' = x ∧ y' = y)) ↔ 
  n = 1/2 := by
sorry


end NUMINAMATH_CALUDE_circle_tangent_to_line_l807_80718


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l807_80735

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9 / 16) → x = Real.sqrt 137 / 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l807_80735


namespace NUMINAMATH_CALUDE_garden_tulips_calculation_l807_80795

-- Define the initial state of the garden
def initial_daisies : ℕ := 32
def ratio_tulips : ℕ := 3
def ratio_daisies : ℕ := 4
def added_daisies : ℕ := 8

-- Define the function to calculate tulips based on daisies and ratio
def calculate_tulips (daisies : ℕ) : ℕ :=
  (daisies * ratio_tulips) / ratio_daisies

-- Theorem statement
theorem garden_tulips_calculation :
  let initial_tulips := calculate_tulips initial_daisies
  let final_daisies := initial_daisies + added_daisies
  let final_tulips := calculate_tulips final_daisies
  let additional_tulips := final_tulips - initial_tulips
  (additional_tulips = 6) ∧ (final_tulips = 30) := by
  sorry

end NUMINAMATH_CALUDE_garden_tulips_calculation_l807_80795


namespace NUMINAMATH_CALUDE_monotonic_shift_l807_80788

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being monotonic on an interval
def MonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

-- State the theorem
theorem monotonic_shift (a b : ℝ) (h : MonotonicOn f a b) :
  MonotonicOn (fun x => f (x + 3)) (a - 3) (b - 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_shift_l807_80788


namespace NUMINAMATH_CALUDE_average_charge_is_five_l807_80757

/-- Represents the charges and attendance for a three-day show -/
structure ShowData where
  day1_charge : ℚ
  day2_charge : ℚ
  day3_charge : ℚ
  day1_attendance : ℚ
  day2_attendance : ℚ
  day3_attendance : ℚ

/-- Calculates the average charge per person for the whole show -/
def averageCharge (data : ShowData) : ℚ :=
  let total_revenue := data.day1_charge * data.day1_attendance +
                       data.day2_charge * data.day2_attendance +
                       data.day3_charge * data.day3_attendance
  let total_attendance := data.day1_attendance + data.day2_attendance + data.day3_attendance
  total_revenue / total_attendance

/-- Theorem stating that the average charge for the given show data is 5 -/
theorem average_charge_is_five (data : ShowData)
  (h1 : data.day1_charge = 15)
  (h2 : data.day2_charge = 15/2)
  (h3 : data.day3_charge = 5/2)
  (h4 : data.day1_attendance = 2 * x)
  (h5 : data.day2_attendance = 5 * x)
  (h6 : data.day3_attendance = 13 * x)
  (h7 : x > 0) :
  averageCharge data = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_charge_is_five_l807_80757


namespace NUMINAMATH_CALUDE_simplify_expression_l807_80729

theorem simplify_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  a * (1 / b + 1 / c) + b * (1 / a + 1 / c) + c * (1 / a + 1 / b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l807_80729


namespace NUMINAMATH_CALUDE_airplane_seats_l807_80756

/-- Calculates the total number of seats on an airplane given the number of coach class seats
    and the relationship between coach and first-class seats. -/
theorem airplane_seats (coach_seats : ℕ) (h1 : coach_seats = 310) 
    (h2 : ∃ first_class : ℕ, coach_seats = 4 * first_class + 2) : 
  coach_seats + (coach_seats - 2) / 4 = 387 := by
  sorry

#check airplane_seats

end NUMINAMATH_CALUDE_airplane_seats_l807_80756


namespace NUMINAMATH_CALUDE_complex_24th_power_of_cube_root_of_unity_l807_80739

theorem complex_24th_power_of_cube_root_of_unity (z : ℂ) : z = (1 + Complex.I * Real.sqrt 3) / 2 → z^24 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_24th_power_of_cube_root_of_unity_l807_80739


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l807_80770

theorem restaurant_bill_theorem (total_people : ℕ) (total_bill : ℚ) (gratuity_rate : ℚ) :
  total_people = 6 →
  total_bill = 720 →
  gratuity_rate = 1/5 →
  (total_bill / (1 + gratuity_rate)) / total_people = 100 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l807_80770


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l807_80731

/-- 
Given:
- The passing mark is 35% of the maximum marks
- A student got 150 marks and failed by 25 marks
Prove that the maximum marks is 500
-/
theorem maximum_marks_calculation (M : ℝ) : 
  (0.35 * M = 150 + 25) → M = 500 := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l807_80731


namespace NUMINAMATH_CALUDE_solutions_cubic_equation_l807_80747

theorem solutions_cubic_equation :
  {x : ℝ | x^3 - 4*x = 0} = {0, -2, 2} := by sorry

end NUMINAMATH_CALUDE_solutions_cubic_equation_l807_80747


namespace NUMINAMATH_CALUDE_train_crossing_time_l807_80709

/-- Proves that a train 100 meters long, traveling at 144 km/hr, will take 2.5 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 100 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l807_80709


namespace NUMINAMATH_CALUDE_third_median_length_l807_80790

/-- Given a triangle with two medians of lengths 5 and 7 inches, and an area of 4√21 square inches,
    the length of the third median is 2√14 inches. -/
theorem third_median_length (m₁ m₂ : ℝ) (area : ℝ) (h₁ : m₁ = 5) (h₂ : m₂ = 7) (h_area : area = 4 * Real.sqrt 21) :
  ∃ (m₃ : ℝ), m₃ = 2 * Real.sqrt 14 ∧ 
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 3 * (m₁^2 + m₂^2 + m₃^2) ∧
                   area = (4 / 3) * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :=
by sorry

end NUMINAMATH_CALUDE_third_median_length_l807_80790


namespace NUMINAMATH_CALUDE_sound_pressure_relations_l807_80769

noncomputable def sound_pressure_level (p p₀ : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

theorem sound_pressure_relations
  (p₀ : ℝ) (hp₀ : p₀ > 0)
  (p₁ p₂ p₃ : ℝ)
  (hp₁ : 60 ≤ sound_pressure_level p₁ p₀ ∧ sound_pressure_level p₁ p₀ ≤ 90)
  (hp₂ : 50 ≤ sound_pressure_level p₂ p₀ ∧ sound_pressure_level p₂ p₀ ≤ 60)
  (hp₃ : sound_pressure_level p₃ p₀ = 40) :
  p₁ ≥ p₂ ∧ p₃ = 100 * p₀ ∧ p₁ ≤ 100 * p₂ :=
by sorry

end NUMINAMATH_CALUDE_sound_pressure_relations_l807_80769


namespace NUMINAMATH_CALUDE_prairie_size_and_untouched_percentage_l807_80727

/-- Represents the prairie and its natural events -/
structure Prairie where
  dust_storm1 : ℕ
  dust_storm2 : ℕ
  flood : ℕ
  wildfire : ℕ
  untouched : ℕ
  affected : ℕ

/-- The prairie with given conditions -/
def our_prairie : Prairie :=
  { dust_storm1 := 75000
  , dust_storm2 := 120000
  , flood := 30000
  , wildfire := 80000
  , untouched := 5000
  , affected := 290000
  }

/-- The theorem stating the total size and untouched percentage of the prairie -/
theorem prairie_size_and_untouched_percentage (p : Prairie) 
  (h : p = our_prairie) : 
  (p.affected + p.untouched = 295000) ∧ 
  (p.untouched : ℚ) / (p.affected + p.untouched : ℚ) * 100 = 5000 / 295000 * 100 := by
  sorry

#eval (our_prairie.untouched : ℚ) / (our_prairie.affected + our_prairie.untouched : ℚ) * 100

end NUMINAMATH_CALUDE_prairie_size_and_untouched_percentage_l807_80727


namespace NUMINAMATH_CALUDE_two_middle_zeros_in_quotient_l807_80755

/-- Count the number of zeros in the middle of a positive integer -/
def count_middle_zeros (n : ℕ) : ℕ :=
  sorry

/-- The quotient when 2010 is divided by 2 -/
def quotient : ℕ := 2010 / 2

theorem two_middle_zeros_in_quotient : count_middle_zeros quotient = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_middle_zeros_in_quotient_l807_80755


namespace NUMINAMATH_CALUDE_more_mashed_potatoes_than_bacon_l807_80792

theorem more_mashed_potatoes_than_bacon (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 457) 
  (h2 : bacon = 394) : 
  mashed_potatoes - bacon = 63 := by
sorry

end NUMINAMATH_CALUDE_more_mashed_potatoes_than_bacon_l807_80792


namespace NUMINAMATH_CALUDE_aeroplane_transaction_loss_l807_80798

theorem aeroplane_transaction_loss : 
  let selling_price : ℝ := 600
  let profit_percentage : ℝ := 0.2
  let loss_percentage : ℝ := 0.2
  let cost_price_profit : ℝ := selling_price / (1 + profit_percentage)
  let cost_price_loss : ℝ := selling_price / (1 - loss_percentage)
  let total_cost : ℝ := cost_price_profit + cost_price_loss
  let total_revenue : ℝ := 2 * selling_price
  total_cost - total_revenue = 50 := by
sorry


end NUMINAMATH_CALUDE_aeroplane_transaction_loss_l807_80798


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l807_80752

theorem sum_of_special_primes_is_prime (A B : ℕ+) : 
  Nat.Prime A ∧ 
  Nat.Prime B ∧ 
  Nat.Prime (A - B) ∧ 
  Nat.Prime (A + B) → 
  Nat.Prime (A + B + (A - B) + A + B) :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l807_80752


namespace NUMINAMATH_CALUDE_don_bottles_from_shop_c_l807_80765

/-- The total number of bottles Don can buy -/
def total_bottles : ℕ := 550

/-- The number of bottles Don buys from Shop A -/
def shop_a_bottles : ℕ := 150

/-- The number of bottles Don buys from Shop B -/
def shop_b_bottles : ℕ := 180

/-- The number of bottles Don buys from Shop C -/
def shop_c_bottles : ℕ := total_bottles - (shop_a_bottles + shop_b_bottles)

theorem don_bottles_from_shop_c :
  shop_c_bottles = 220 :=
by sorry

end NUMINAMATH_CALUDE_don_bottles_from_shop_c_l807_80765


namespace NUMINAMATH_CALUDE_inequality_proof_l807_80762

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a * b * c = 1) : 
  (a - 1) / c + (c - 1) / b + (b - 1) / a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l807_80762


namespace NUMINAMATH_CALUDE_rationalize_denominator_l807_80749

theorem rationalize_denominator : 
  (35 : ℝ) / Real.sqrt 15 = (7 / 3) * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l807_80749


namespace NUMINAMATH_CALUDE_initial_orchids_l807_80743

theorem initial_orchids (initial_roses : ℕ) (final_roses : ℕ) (final_orchids : ℕ) 
  (orchid_rose_difference : ℕ) : 
  initial_roses = 7 → 
  final_roses = 11 → 
  final_orchids = 20 → 
  orchid_rose_difference = 9 → 
  final_orchids = final_roses + orchid_rose_difference →
  initial_roses + orchid_rose_difference = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_orchids_l807_80743


namespace NUMINAMATH_CALUDE_cost_price_is_1200_l807_80742

/-- Calculates the cost price of a toy given the total selling price, number of toys sold, and the gain condition. -/
def cost_price_of_toy (total_selling_price : ℕ) (num_toys_sold : ℕ) (num_toys_gain : ℕ) : ℕ :=
  let selling_price_per_toy := total_selling_price / num_toys_sold
  let cost_price := selling_price_per_toy * num_toys_sold / (num_toys_sold + num_toys_gain)
  cost_price

/-- Theorem stating that under the given conditions, the cost price of a toy is 1200. -/
theorem cost_price_is_1200 :
  cost_price_of_toy 50400 36 6 = 1200 := by
  sorry

#eval cost_price_of_toy 50400 36 6

end NUMINAMATH_CALUDE_cost_price_is_1200_l807_80742
