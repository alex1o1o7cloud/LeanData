import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l78_7836

theorem imaginary_part_of_complex_number :
  let z : ℂ := 3 - 2 * I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l78_7836


namespace NUMINAMATH_CALUDE_odd_function_iff_m_and_n_l78_7829

def f (m n x : ℝ) : ℝ := (m^2 - 1) * x^2 + (m - 1) * x + n + 2

theorem odd_function_iff_m_and_n (m n : ℝ) :
  (∀ x, f m n (-x) = -f m n x) ↔ ((m = 1 ∨ m = -1) ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_odd_function_iff_m_and_n_l78_7829


namespace NUMINAMATH_CALUDE_remainder_mod_five_l78_7805

theorem remainder_mod_five : (9^6 + 8^8 + 7^9) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_five_l78_7805


namespace NUMINAMATH_CALUDE_prop_logic_l78_7823

theorem prop_logic (p q : Prop) (h : ¬(¬p ∨ ¬q)) : (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_prop_logic_l78_7823


namespace NUMINAMATH_CALUDE_square_minimizes_diagonal_l78_7865

/-- A parallelogram with side lengths and angles -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  angle : ℝ
  area : ℝ

/-- The length of the larger diagonal of a parallelogram -/
def largerDiagonal (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem: Among all parallelograms with a given area, the square has the smallest larger diagonal -/
theorem square_minimizes_diagonal {A : ℝ} (h : A > 0) :
  ∀ p : Parallelogram, p.area = A →
    largerDiagonal p ≥ largerDiagonal { side1 := Real.sqrt A, side2 := Real.sqrt A, angle := π/2, area := A } :=
  sorry

end NUMINAMATH_CALUDE_square_minimizes_diagonal_l78_7865


namespace NUMINAMATH_CALUDE_percentage_difference_difference_is_twelve_l78_7821

theorem percentage_difference : ℝ → Prop :=
  let percent_of_40 := (80 / 100) * 40
  let fraction_of_25 := (4 / 5) * 25
  λ x => percent_of_40 - fraction_of_25 = x

theorem difference_is_twelve : percentage_difference 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_difference_is_twelve_l78_7821


namespace NUMINAMATH_CALUDE_jellybean_count_l78_7819

/-- The number of jellybeans in a bag with black, green, and orange beans -/
def total_jellybeans (black green orange : ℕ) : ℕ := black + green + orange

/-- Theorem: The total number of jellybeans in the bag is 27 -/
theorem jellybean_count :
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  orange = green - 1 →
  total_jellybeans black green orange = 27 := by
sorry

end NUMINAMATH_CALUDE_jellybean_count_l78_7819


namespace NUMINAMATH_CALUDE_cube_root_equality_l78_7877

theorem cube_root_equality : 
  (2016^2 + 2016 * 2017 + 2017^2 + 2016^3 : ℝ)^(1/3) = 2017 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equality_l78_7877


namespace NUMINAMATH_CALUDE_remainder_theorem_l78_7813

/-- A polynomial p(x) satisfying p(2) = 4 and p(5) = 10 -/
def p : Polynomial ℝ :=
  sorry

theorem remainder_theorem (h1 : p.eval 2 = 4) (h2 : p.eval 5 = 10) :
  ∃ q : Polynomial ℝ, p = q * ((X - 2) * (X - 5)) + (2 * X) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l78_7813


namespace NUMINAMATH_CALUDE_parabola_intersection_l78_7869

theorem parabola_intersection (m : ℝ) (h : m > 0) :
  let f (x : ℝ) := x^2 + 2*m*x - (5/4)*m^2
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 6 → m = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l78_7869


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l78_7846

/-- Proves that the average salary of non-technician workers is 6000, given the conditions of the workshop --/
theorem workshop_salary_problem (total_workers : ℕ) (avg_salary_all : ℕ) 
  (num_technicians : ℕ) (avg_salary_tech : ℕ) :
  total_workers = 49 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_tech = 20000 →
  (total_workers - num_technicians) * 
    ((total_workers * avg_salary_all - num_technicians * avg_salary_tech) / 
     (total_workers - num_technicians)) = 
  (total_workers - num_technicians) * 6000 := by
  sorry

#check workshop_salary_problem

end NUMINAMATH_CALUDE_workshop_salary_problem_l78_7846


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_times_square_l78_7804

theorem factorization_cubic_minus_linear_times_square (a b : ℝ) :
  a^3 - a*b^2 = a*(a+b)*(a-b) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_times_square_l78_7804


namespace NUMINAMATH_CALUDE_third_smallest_number_indeterminate_l78_7856

theorem third_smallest_number_indeterminate 
  (a b c d : ℕ) 
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum1 : a + b + c = 21)
  (h_sum2 : a + b + d = 27)
  (h_sum3 : a + c + d = 30) :
  ¬∃(n : ℕ), ∀(x : ℕ), (x = c) ↔ (x = n) :=
sorry

end NUMINAMATH_CALUDE_third_smallest_number_indeterminate_l78_7856


namespace NUMINAMATH_CALUDE_parcel_weight_sum_l78_7835

/-- Given three parcels with weights x, y, and z, prove that their total weight is 195 pounds
    if the sum of each pair of parcels weighs 112, 146, and 132 pounds respectively. -/
theorem parcel_weight_sum (x y z : ℝ) 
  (pair_xy : x + y = 112)
  (pair_yz : y + z = 146)
  (pair_zx : z + x = 132) :
  x + y + z = 195 := by
  sorry

end NUMINAMATH_CALUDE_parcel_weight_sum_l78_7835


namespace NUMINAMATH_CALUDE_system_solution_l78_7808

theorem system_solution (x y : ℝ) (eq1 : 2 * x + y = 7) (eq2 : x + 2 * y = 5) : x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l78_7808


namespace NUMINAMATH_CALUDE_bertolli_farm_corn_count_l78_7802

theorem bertolli_farm_corn_count :
  ∀ (tomatoes onions corn : ℕ),
    tomatoes = 2073 →
    onions = 985 →
    tomatoes + corn - onions = 5200 →
    corn = 4039 :=
by
  sorry

end NUMINAMATH_CALUDE_bertolli_farm_corn_count_l78_7802


namespace NUMINAMATH_CALUDE_estimate_larger_than_actual_l78_7824

theorem estimate_larger_than_actual (x y z : ℝ) 
  (hxy : x > y) (hy : y > 0) (hz : z > 0) : 
  (x + z) - (y - z) > x - y := by
  sorry

end NUMINAMATH_CALUDE_estimate_larger_than_actual_l78_7824


namespace NUMINAMATH_CALUDE_petya_bonus_points_l78_7860

def calculate_bonus (score : ℕ) : ℕ :=
  if score < 1000 then
    (score * 20) / 100
  else if score < 2000 then
    200 + ((score - 1000) * 30) / 100
  else
    200 + 300 + ((score - 2000) * 50) / 100

theorem petya_bonus_points : calculate_bonus 2370 = 685 := by
  sorry

end NUMINAMATH_CALUDE_petya_bonus_points_l78_7860


namespace NUMINAMATH_CALUDE_sara_picked_37_peaches_l78_7859

/-- The number of peaches Sara picked at the orchard -/
def peaches_picked (initial_peaches final_peaches : ℕ) : ℕ :=
  final_peaches - initial_peaches

/-- Theorem: Sara picked 37 peaches at the orchard -/
theorem sara_picked_37_peaches : 
  peaches_picked 24 61 = 37 := by
  sorry

end NUMINAMATH_CALUDE_sara_picked_37_peaches_l78_7859


namespace NUMINAMATH_CALUDE_diophantine_fraction_equality_l78_7861

theorem diophantine_fraction_equality : ∃ (A B : ℤ), 
  A = 500 ∧ B = -501 ∧ (A : ℚ) / 999 + (B : ℚ) / 1001 = 1 / 999999 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_fraction_equality_l78_7861


namespace NUMINAMATH_CALUDE_power_sum_and_division_l78_7872

theorem power_sum_and_division (x y z : ℕ) : 3^128 + 8^5 / 8^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_division_l78_7872


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l78_7810

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l78_7810


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l78_7820

theorem parallel_vectors_angle (α : Real) : 
  α > 0 → 
  α < π / 2 → 
  let a : Fin 2 → Real := ![3/4, Real.sin α]
  let b : Fin 2 → Real := ![Real.cos α, 1/3]
  (∃ (k : Real), k ≠ 0 ∧ a = k • b) → 
  α = π / 12 ∨ α = 5 * π / 12 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l78_7820


namespace NUMINAMATH_CALUDE_raccoon_stall_time_l78_7893

/-- The time (in minutes) the first lock stalls the raccoons -/
def T1 : ℕ := 5

/-- The time (in minutes) the second lock stalls the raccoons -/
def T2 : ℕ := 3 * T1 - 3

/-- The time (in minutes) both locks together stall the raccoons -/
def both_locks : ℕ := 5 * T2

theorem raccoon_stall_time : both_locks = 60 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_stall_time_l78_7893


namespace NUMINAMATH_CALUDE_weight_ratio_john_to_mary_l78_7886

/-- Proves that the ratio of John's weight to Mary's weight is 5:4 given the specified conditions -/
theorem weight_ratio_john_to_mary :
  ∀ (john_weight mary_weight jamison_weight : ℕ),
    mary_weight = 160 →
    mary_weight + 20 = jamison_weight →
    john_weight + mary_weight + jamison_weight = 540 →
    (john_weight : ℚ) / mary_weight = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_weight_ratio_john_to_mary_l78_7886


namespace NUMINAMATH_CALUDE_sequences_1992_values_l78_7884

/-- Two sequences of integer numbers satisfying given conditions -/
def Sequences (a b : ℕ → ℤ) : Prop :=
  (a 0 = 0) ∧ (b 0 = 8) ∧
  (∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n + 2) ∧
  (∀ n : ℕ, b (n + 2) = 2 * b (n + 1) - b n) ∧
  (∀ n : ℕ, ∃ k : ℤ, (a n)^2 + (b n)^2 = k^2)

/-- The theorem to be proved -/
theorem sequences_1992_values (a b : ℕ → ℤ) (h : Sequences a b) :
  ((a 1992 = 31872 ∧ b 1992 = 31880) ∨ (a 1992 = -31872 ∧ b 1992 = -31864)) :=
sorry

end NUMINAMATH_CALUDE_sequences_1992_values_l78_7884


namespace NUMINAMATH_CALUDE_inequalities_proof_l78_7874

theorem inequalities_proof (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : a + b > 0) :
  (a^3 * b < a * b^3) ∧ (a / b + b / a < -2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l78_7874


namespace NUMINAMATH_CALUDE_f_min_value_f_inequality_condition_l78_7899

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + abs (x - 2)

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 7/4) :=
sorry

-- Theorem for the inequality condition
theorem f_inequality_condition (a b c : ℝ) :
  (∀ (x : ℝ), f x ≥ a^2 + 2*b^2 + 3*c^2) → a*c + 2*b*c ≤ 7/8 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_f_inequality_condition_l78_7899


namespace NUMINAMATH_CALUDE_vector_problem_l78_7815

/-- Given two vectors a and b in R^2 -/
def a (x : ℝ) : Fin 2 → ℝ := ![x, -2]
def b : Fin 2 → ℝ := ![2, 4]

/-- Parallel vectors have proportional components -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 2, v i = k * w i

/-- The squared magnitude of a vector -/
def magnitude_squared (v : Fin 2 → ℝ) : ℝ :=
  (v 0)^2 + (v 1)^2

/-- Vector addition -/
def vec_add (v w : Fin 2 → ℝ) : Fin 2 → ℝ :=
  λ i => v i + w i

theorem vector_problem (x : ℝ) :
  (parallel (a x) b → x = -1) ∧
  (magnitude_squared (vec_add (a x) b) = 13 → x = 1 ∨ x = -5) := by
  sorry


end NUMINAMATH_CALUDE_vector_problem_l78_7815


namespace NUMINAMATH_CALUDE_solve_for_m_l78_7866

theorem solve_for_m (x y m : ℝ) (h1 : x = 2) (h2 : y = m) (h3 : 3 * x + 2 * y = 10) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l78_7866


namespace NUMINAMATH_CALUDE_missing_angle_is_zero_l78_7871

/-- Represents a polygon with a missing angle -/
structure PolygonWithMissingAngle where
  n : ℕ                     -- number of sides
  sum_without_missing : ℝ   -- sum of all angles except the missing one
  missing_angle : ℝ         -- the missing angle

/-- The theorem stating that the missing angle is 0° -/
theorem missing_angle_is_zero (p : PolygonWithMissingAngle) 
  (h1 : p.sum_without_missing = 3240)
  (h2 : p.sum_without_missing + p.missing_angle = 180 * (p.n - 2)) :
  p.missing_angle = 0 := by
sorry


end NUMINAMATH_CALUDE_missing_angle_is_zero_l78_7871


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l78_7894

theorem smaller_number_in_ratio (a b c x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = a / (b ^ 2) → x + y = 2 * c → 
  min x y = (2 * a * c) / (a + b ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l78_7894


namespace NUMINAMATH_CALUDE_pool_capacity_l78_7890

theorem pool_capacity (fill_time_both : ℝ) (fill_time_first : ℝ) (additional_rate : ℝ) :
  fill_time_both = 48 →
  fill_time_first = 120 →
  additional_rate = 50 →
  ∃ (capacity : ℝ),
    capacity = 12000 ∧
    capacity / fill_time_both = capacity / fill_time_first + (capacity / fill_time_first + additional_rate) :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l78_7890


namespace NUMINAMATH_CALUDE_student_number_problem_l78_7818

theorem student_number_problem (x : ℤ) : 2 * x - 138 = 110 → x = 124 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l78_7818


namespace NUMINAMATH_CALUDE_symmetric_complex_number_l78_7885

/-- Given that z is symmetric to 2/(1-i) with respect to the imaginary axis, prove that z = -1 + i -/
theorem symmetric_complex_number (z : ℂ) : 
  (z.re = -(2 / (1 - I)).re ∧ z.im = (2 / (1 - I)).im) → z = -1 + I :=
by sorry

end NUMINAMATH_CALUDE_symmetric_complex_number_l78_7885


namespace NUMINAMATH_CALUDE_batsman_average_increase_l78_7847

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average after a new innings -/
def averageIncrease (prevStats : BatsmanStats) (newInningRuns : ℕ) : ℚ :=
  let newAverage := (prevStats.totalRuns + newInningRuns : ℚ) / (prevStats.inningsPlayed + 1 : ℚ)
  newAverage - prevStats.average

/-- Theorem: The increase in the batsman's average is 2 runs per inning -/
theorem batsman_average_increase :
  ∀ (prevStats : BatsmanStats),
    prevStats.inningsPlayed = 16 →
    averageIncrease prevStats 50 = 18 - prevStats.average →
    averageIncrease prevStats 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l78_7847


namespace NUMINAMATH_CALUDE_symmetry_implies_axis_l78_7898

/-- A function g is symmetric about x = 1.5 if g(x) = g(3-x) for all x -/
def SymmetricAbout1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for g if
    for all points (x, g(x)), the point (3-x, g(x)) is also on the graph of g -/
def IsAxisOfSymmetry1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

theorem symmetry_implies_axis (g : ℝ → ℝ) :
  SymmetricAbout1_5 g → IsAxisOfSymmetry1_5 g :=
by
  sorry

#check symmetry_implies_axis

end NUMINAMATH_CALUDE_symmetry_implies_axis_l78_7898


namespace NUMINAMATH_CALUDE_akeno_extra_expenditure_l78_7895

def akeno_expenditure : ℕ := 2985

def lev_expenditure (akeno : ℕ) : ℕ := akeno / 3

def ambrocio_expenditure (lev : ℕ) : ℕ := lev - 177

theorem akeno_extra_expenditure (akeno lev ambrocio : ℕ) 
  (h1 : akeno = akeno_expenditure)
  (h2 : lev = lev_expenditure akeno)
  (h3 : ambrocio = ambrocio_expenditure lev) :
  akeno - (lev + ambrocio) = 1172 := by
  sorry

end NUMINAMATH_CALUDE_akeno_extra_expenditure_l78_7895


namespace NUMINAMATH_CALUDE_number_pyramid_result_l78_7868

theorem number_pyramid_result : 123456 * 9 + 7 = 1111111 := by
  sorry

end NUMINAMATH_CALUDE_number_pyramid_result_l78_7868


namespace NUMINAMATH_CALUDE_water_tank_capacity_l78_7851

theorem water_tank_capacity (initial_fraction : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  initial_fraction = 1/3 →
  added_amount = 5 →
  final_fraction = 2/5 →
  (initial_fraction * added_amount) / (final_fraction - initial_fraction) = 75 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l78_7851


namespace NUMINAMATH_CALUDE_soccer_ball_donation_l78_7839

/-- Calculates the total number of soccer balls donated by a public official to two schools -/
def total_soccer_balls (balls_per_class : ℕ) (num_schools : ℕ) (elementary_classes : ℕ) (middle_classes : ℕ) : ℕ :=
  balls_per_class * num_schools * (elementary_classes + middle_classes)

/-- Proves that the total number of soccer balls donated is 90 -/
theorem soccer_ball_donation : total_soccer_balls 5 2 4 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_donation_l78_7839


namespace NUMINAMATH_CALUDE_bookstore_shipment_l78_7889

theorem bookstore_shipment (displayed_percentage : ℚ) (storeroom_books : ℕ) : 
  displayed_percentage = 30 / 100 →
  storeroom_books = 210 →
  ∃ total_books : ℕ, 
    (1 - displayed_percentage) * total_books = storeroom_books ∧
    total_books = 300 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_shipment_l78_7889


namespace NUMINAMATH_CALUDE_same_color_probability_l78_7881

/-- The probability of drawing two balls of the same color from an urn -/
theorem same_color_probability (w b r : ℕ) (hw : w = 4) (hb : b = 6) (hr : r = 5) :
  let total := w + b + r
  let p_white := (w / total) * ((w - 1) / (total - 1))
  let p_black := (b / total) * ((b - 1) / (total - 1))
  let p_red := (r / total) * ((r - 1) / (total - 1))
  p_white + p_black + p_red = 31 / 105 := by
  sorry

#check same_color_probability

end NUMINAMATH_CALUDE_same_color_probability_l78_7881


namespace NUMINAMATH_CALUDE_annie_initial_money_l78_7878

def initial_money (hamburger_price burger_count milkshake_price shake_count money_left : ℕ) : ℕ :=
  hamburger_price * burger_count + milkshake_price * shake_count + money_left

theorem annie_initial_money :
  initial_money 4 8 5 6 70 = 132 := by
  sorry

end NUMINAMATH_CALUDE_annie_initial_money_l78_7878


namespace NUMINAMATH_CALUDE_first_pickup_fraction_proof_l78_7845

/-- Represents the carrying capacity of the bus -/
def bus_capacity : ℕ := 80

/-- Represents the number of people waiting at the second pickup point -/
def second_pickup_waiting : ℕ := 50

/-- Represents the number of people who couldn't board at the second pickup point -/
def unable_to_board : ℕ := 18

/-- Represents the fraction of bus capacity that entered at the first pickup point -/
def first_pickup_fraction : ℚ := 3 / 5

theorem first_pickup_fraction_proof :
  first_pickup_fraction = (bus_capacity - (second_pickup_waiting - unable_to_board)) / bus_capacity :=
by sorry

end NUMINAMATH_CALUDE_first_pickup_fraction_proof_l78_7845


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l78_7848

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, (x - y) * x^4 < 0 → x < y) ∧
  (∃ x y : ℝ, x < y ∧ (x - y) * x^4 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l78_7848


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l78_7850

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (5 * Complex.I) / (4 + 3 * Complex.I)
  Complex.im z = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l78_7850


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l78_7843

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧ (∃ a, a^2 > 1 ∧ ¬(a > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l78_7843


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l78_7801

/-- A geometric sequence with given first and fourth terms -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ a 4 = -4 ∧ ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The common ratio of the geometric sequence -/
def common_ratio (a : ℕ → ℚ) : ℚ :=
  (a 2) / (a 1)

/-- Theorem: Properties of the geometric sequence -/
theorem geometric_sequence_properties (a : ℕ → ℚ) 
  (h : geometric_sequence a) : 
  common_ratio a = -2 ∧ ∀ n : ℕ, a n = 1/2 * (-2)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l78_7801


namespace NUMINAMATH_CALUDE_mickey_horses_per_week_l78_7844

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := 7 + 3

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem stating that Mickey mounts 98 horses per week -/
theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end NUMINAMATH_CALUDE_mickey_horses_per_week_l78_7844


namespace NUMINAMATH_CALUDE_pants_and_coat_cost_l78_7830

theorem pants_and_coat_cost (p s c : ℝ) 
  (h1 : p + s = 100)
  (h2 : c = 5 * s)
  (h3 : c = 180) : 
  p + c = 244 := by
  sorry

end NUMINAMATH_CALUDE_pants_and_coat_cost_l78_7830


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l78_7873

theorem fixed_point_of_exponential_function (a : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-2) + 3
  f 2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l78_7873


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l78_7863

def A : Set ℕ := {x | x - 4 < 0}
def B : Set ℕ := {0, 1, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l78_7863


namespace NUMINAMATH_CALUDE_greatest_good_set_l78_7855

def is_good (k : ℕ) (S : Set ℕ) : Prop :=
  ∃ (color : ℕ → Fin k),
    ∀ s ∈ S, ∀ x y : ℕ, x + y = s → color x ≠ color y

theorem greatest_good_set (k : ℕ) (h : k > 1) :
  (∀ a : ℕ, is_good k {x | ∃ t, x = a + t ∧ 1 ≤ t ∧ t ≤ 2*k - 1}) ∧
  ¬(∀ a : ℕ, is_good k {x | ∃ t, x = a + t ∧ 1 ≤ t ∧ t ≤ 2*k}) :=
sorry

end NUMINAMATH_CALUDE_greatest_good_set_l78_7855


namespace NUMINAMATH_CALUDE_sequence_less_than_two_l78_7822

theorem sequence_less_than_two (a : ℕ → ℝ) :
  (∀ n, a n < 2) ↔ ¬(∃ k, a k ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_less_than_two_l78_7822


namespace NUMINAMATH_CALUDE_pool_width_is_twelve_l78_7841

/-- Represents a rectangular swimming pool with a surrounding deck -/
structure PoolWithDeck where
  poolLength : ℝ
  poolWidth : ℝ
  deckWidth : ℝ
  totalArea : ℝ

/-- Theorem stating the width of the swimming pool given specific conditions -/
theorem pool_width_is_twelve (p : PoolWithDeck)
  (h1 : p.poolLength = 10)
  (h2 : p.deckWidth = 4)
  (h3 : p.totalArea = 360)
  (h4 : (p.poolLength + 2 * p.deckWidth) * (p.poolWidth + 2 * p.deckWidth) = p.totalArea) :
  p.poolWidth = 12 := by
  sorry

end NUMINAMATH_CALUDE_pool_width_is_twelve_l78_7841


namespace NUMINAMATH_CALUDE_lori_earnings_l78_7837

/-- Calculates the total earnings for a car rental company given the number of cars,
    rental rates, and rental duration. -/
def total_earnings (red_cars white_cars : ℕ) (red_rate white_rate : ℚ) (hours : ℕ) : ℚ :=
  (red_cars * red_rate + white_cars * white_rate) * (hours * 60)

/-- Proves that given the specific conditions of Lori's car rental business,
    the total earnings are $2340. -/
theorem lori_earnings :
  total_earnings 3 2 3 2 3 = 2340 := by
  sorry

#eval total_earnings 3 2 3 2 3

end NUMINAMATH_CALUDE_lori_earnings_l78_7837


namespace NUMINAMATH_CALUDE_sum_of_powers_of_three_l78_7883

theorem sum_of_powers_of_three : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_three_l78_7883


namespace NUMINAMATH_CALUDE_courier_speed_impossibility_l78_7809

/-- Proves that it's impossible to achieve an average speed of 12 mph over 24 miles
    if two-thirds of the distance was run at 8 mph -/
theorem courier_speed_impossibility (total_distance : ℝ) (initial_speed : ℝ) (target_speed : ℝ) :
  total_distance = 24 →
  initial_speed = 8 →
  target_speed = 12 →
  ¬ ∃ (remaining_speed : ℝ),
    (2 / 3 * total_distance / initial_speed + 1 / 3 * total_distance / remaining_speed) 
    = total_distance / target_speed :=
by sorry

end NUMINAMATH_CALUDE_courier_speed_impossibility_l78_7809


namespace NUMINAMATH_CALUDE_unique_cube_prime_l78_7892

theorem unique_cube_prime (p : ℕ) : Prime p → (∃ n : ℕ, 2 * p + 1 = n ^ 3) ↔ p = 13 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_prime_l78_7892


namespace NUMINAMATH_CALUDE_x_value_proof_l78_7897

theorem x_value_proof (x y : ℝ) (h1 : x > y) 
  (h2 : x^2 * y^2 + x^2 + y^2 + 2*x*y = 40) 
  (h3 : x*y + x + y = 8) : x = 3 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l78_7897


namespace NUMINAMATH_CALUDE_inequality_of_reciprocals_l78_7831

theorem inequality_of_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_reciprocals_l78_7831


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l78_7857

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

theorem tangent_line_at_point (x₀ y₀ : ℝ) (h : y₀ = f x₀) :
  let m := (3*x₀^2 - 4*x₀ - 4)  -- Derivative of f at x₀
  (5 : ℝ) * x + y - 2 = 0 ↔ y - y₀ = m * (x - x₀) ∧ x₀ = 1 ∧ y₀ = -3 :=
by sorry

#check tangent_line_at_point

end NUMINAMATH_CALUDE_tangent_line_at_point_l78_7857


namespace NUMINAMATH_CALUDE_projection_vector_l78_7842

/-- Given two lines k and n in 2D space, prove that the vector (-6, 9) satisfies the conditions for the projection of DC onto the normal of line n. -/
theorem projection_vector : ∃ (w1 w2 : ℝ), w1 = -6 ∧ w2 = 9 ∧ w1 + w2 = 3 ∧ 
  ∃ (t s : ℝ),
    let k := λ t : ℝ => (2 + 3*t, 3 + 2*t)
    let n := λ s : ℝ => (1 + 3*s, 5 + 2*s)
    let C := k t
    let D := n s
    let normal_n := (-2, 3)
    ∃ (c : ℝ), (w1, w2) = c • normal_n :=
by sorry

end NUMINAMATH_CALUDE_projection_vector_l78_7842


namespace NUMINAMATH_CALUDE_line_parallel_implies_plane_perpendicular_l78_7896

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_implies_plane_perpendicular
  (l : Line) (m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : parallel l m) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_implies_plane_perpendicular_l78_7896


namespace NUMINAMATH_CALUDE_max_cube_volume_in_pyramid_l78_7826

/-- The maximum volume of a cube inscribed in a pyramid --/
theorem max_cube_volume_in_pyramid (base_side : ℝ) (pyramid_height : ℝ) : 
  base_side = 2 →
  pyramid_height = 3 →
  ∃ (cube_volume : ℝ), 
    cube_volume = (81 * Real.sqrt 6) / 32 ∧ 
    ∀ (other_volume : ℝ), 
      (∃ (cube_side : ℝ), 
        cube_side > 0 ∧
        other_volume = cube_side ^ 3 ∧
        cube_side * Real.sqrt 2 ≤ 3 * Real.sqrt 3 / 2) →
      other_volume ≤ cube_volume :=
by sorry

end NUMINAMATH_CALUDE_max_cube_volume_in_pyramid_l78_7826


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l78_7858

/-- Given that x and y are inversely proportional, prove that when x = -12, y = -39.0625 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 50 ∧ x₀ = 3 * y₀ ∧ x₀ * y₀ = k) :
  x = -12 → y = -39.0625 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l78_7858


namespace NUMINAMATH_CALUDE_orange_buckets_l78_7838

/-- Given three buckets of oranges with specific relationships between their quantities
    and a total number of oranges, prove that the first bucket contains 22 oranges. -/
theorem orange_buckets (b1 b2 b3 : ℕ) : 
  b2 = b1 + 17 →
  b3 = b2 - 11 →
  b1 + b2 + b3 = 89 →
  b1 = 22 := by
sorry

end NUMINAMATH_CALUDE_orange_buckets_l78_7838


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_for_a_sq_eq_one_l78_7876

theorem a_eq_one_sufficient_not_necessary_for_a_sq_eq_one :
  ∃ (a : ℝ), (a = 1 → a^2 = 1) ∧ ¬(a^2 = 1 → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_for_a_sq_eq_one_l78_7876


namespace NUMINAMATH_CALUDE_tangent_slope_point_l78_7817

theorem tangent_slope_point (x₀ : ℝ) :
  let f : ℝ → ℝ := fun x ↦ Real.exp (-x)
  let y₀ : ℝ := f x₀
  (deriv f x₀ = -2) → (x₀ = -Real.log 2 ∧ y₀ = 2) := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_point_l78_7817


namespace NUMINAMATH_CALUDE_specific_line_equation_l78_7812

/-- A line parameterized by real t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The specific parametric line from the problem -/
def specificLine : ParametricLine where
  x := λ t => 3 * t + 2
  y := λ t => 5 * t - 3

/-- The equation of a line in slope-intercept form -/
structure LineEquation where
  slope : ℝ
  intercept : ℝ

/-- Theorem stating that the specific parametric line has the given equation -/
theorem specific_line_equation :
  ∃ (t : ℝ), specificLine.y t = (5/3) * specificLine.x t - 19/3 := by
  sorry

end NUMINAMATH_CALUDE_specific_line_equation_l78_7812


namespace NUMINAMATH_CALUDE_triangle_heights_sum_ge_nine_times_inradius_l78_7807

/-- Given a triangle with heights h₁, h₂, h₃ and an inscribed circle of radius r,
    the sum of the heights is greater than or equal to 9 times the radius. -/
theorem triangle_heights_sum_ge_nine_times_inradius 
  (h₁ h₂ h₃ r : ℝ) 
  (height_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (inradius_positive : r > 0)
  (triangle_heights : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    h₁ = 2 * (a * b * c).sqrt / (a * (a + b + c)) ∧
    h₂ = 2 * (a * b * c).sqrt / (b * (a + b + c)) ∧
    h₃ = 2 * (a * b * c).sqrt / (c * (a + b + c)))
  (inradius_def : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    r = (a * b * c).sqrt / (a + b + c)) :
  h₁ + h₂ + h₃ ≥ 9 * r := by
  sorry

end NUMINAMATH_CALUDE_triangle_heights_sum_ge_nine_times_inradius_l78_7807


namespace NUMINAMATH_CALUDE_line_properties_l78_7806

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Define points A and P
def point_A : ℝ × ℝ := (3, 2)
def point_P : ℝ × ℝ := (3, 0)

-- Define the perpendicular line l₁
def line_l1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define the parallel lines l₂
def line_l2_1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def line_l2_2 (x y : ℝ) : Prop := 2 * x + y - 11 = 0

-- Theorem statement
theorem line_properties :
  (∀ x y : ℝ, line_l1 x y ↔ (x = point_A.1 ∧ y = point_A.2) ∨ 
    (∃ k : ℝ, x = point_A.1 + k ∧ y = point_A.2 - k/2)) ∧
  (∀ x y : ℝ, (line_l2_1 x y ∨ line_l2_2 x y) ↔
    (∃ k : ℝ, x = k ∧ y = -2*k + 1) ∧
    (|2 * point_P.1 + point_P.2 + 1| / Real.sqrt 5 = Real.sqrt 5 ∨
     |2 * point_P.1 + point_P.2 + 11| / Real.sqrt 5 = Real.sqrt 5)) :=
by sorry


end NUMINAMATH_CALUDE_line_properties_l78_7806


namespace NUMINAMATH_CALUDE_reciprocal_of_two_l78_7800

theorem reciprocal_of_two :
  ∃ x : ℚ, x * 2 = 1 ∧ x = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_l78_7800


namespace NUMINAMATH_CALUDE_probability_properties_l78_7880

theorem probability_properties (A B : Set Ω) (P : Set Ω → ℝ) 
  (hA : P A = 0.5) (hB : P B = 0.3) :
  (∀ h : A ∩ B = ∅, P (A ∪ B) = 0.8) ∧ 
  (∀ h : P (A ∩ B) = P A * P B, P (A ∪ B) = 0.65) ∧
  (∀ h : P (B ∩ A) / P A = 0.5, P (B ∩ Aᶜ) / P Aᶜ = 0.1) := by
  sorry

end NUMINAMATH_CALUDE_probability_properties_l78_7880


namespace NUMINAMATH_CALUDE_line_relationship_l78_7816

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_relationship (a b c : Line) 
  (h1 : skew a b) 
  (h2 : parallel c a) : 
  ¬ parallel c b := by
  sorry

end NUMINAMATH_CALUDE_line_relationship_l78_7816


namespace NUMINAMATH_CALUDE_gcd_lcm_equation_solutions_l78_7814

theorem gcd_lcm_equation_solutions :
  let S : Set (ℕ × ℕ) := {(8, 513), (513, 8), (215, 2838), (2838, 215),
                          (258, 1505), (1505, 258), (235, 2961), (2961, 235)}
  ∀ α β : ℕ, (Nat.gcd α β + Nat.lcm α β = 4 * (α + β) + 2021) ↔ (α, β) ∈ S :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_equation_solutions_l78_7814


namespace NUMINAMATH_CALUDE_vector_parallelism_l78_7887

theorem vector_parallelism (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![t, 1]
  (∃ k : ℝ, k ≠ 0 ∧ (a + b) = k • (a - b)) → t = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l78_7887


namespace NUMINAMATH_CALUDE_cricketer_score_percentage_l78_7870

/-- Calculates the percentage of runs made by running between the wickets -/
def percentage_runs_by_running (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let runs_from_boundaries := boundaries * 4
  let runs_from_sixes := sixes * 6
  let runs_by_running := total_runs - (runs_from_boundaries + runs_from_sixes)
  (runs_by_running : ℚ) / total_runs * 100

/-- Proves that the percentage of runs made by running between the wickets is approximately 60.53% -/
theorem cricketer_score_percentage :
  let result := percentage_runs_by_running 152 12 2
  ∃ ε > 0, |result - 60.53| < ε :=
sorry

end NUMINAMATH_CALUDE_cricketer_score_percentage_l78_7870


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l78_7811

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 4 - 2 * x) ↔ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l78_7811


namespace NUMINAMATH_CALUDE_proposition_variants_l78_7867

theorem proposition_variants (a b : ℝ) :
  (((a - 2 > b - 2) → (a > b)) ∧
   ((a ≤ b) → (a - 2 ≤ b - 2)) ∧
   ((a - 2 ≤ b - 2) → (a ≤ b)) ∧
   ¬((a > b) → (a - 2 ≤ b - 2))) := by
  sorry

end NUMINAMATH_CALUDE_proposition_variants_l78_7867


namespace NUMINAMATH_CALUDE_range_of_f_l78_7849

def f (x : ℕ) : ℤ := 3 * x - 1

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 4}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {2, 5, 8, 11} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l78_7849


namespace NUMINAMATH_CALUDE_distribute_cards_count_l78_7803

/-- The number of ways to distribute 6 cards into 3 envelopes -/
def distribute_cards : ℕ :=
  let n_cards := 6
  let n_envelopes := 3
  let cards_per_envelope := 2
  let ways_to_place_1_and_2 := n_envelopes
  let remaining_cards := n_cards - cards_per_envelope
  let ways_to_distribute_remaining := 6  -- This is a given fact from the problem
  ways_to_place_1_and_2 * ways_to_distribute_remaining

/-- Theorem stating that the number of ways to distribute the cards is 18 -/
theorem distribute_cards_count : distribute_cards = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribute_cards_count_l78_7803


namespace NUMINAMATH_CALUDE_sandy_turnips_count_undetermined_l78_7875

/-- Represents the number of vegetables grown by a person -/
structure VegetableCount where
  carrots : ℕ
  turnips : ℕ

/-- The given information about Sandy and Mary's vegetable growth -/
def given : Prop :=
  ∃ (sandy : VegetableCount) (mary : VegetableCount),
    sandy.carrots = 8 ∧
    mary.carrots = 6 ∧
    sandy.carrots + mary.carrots = 14

/-- The statement that Sandy's turnip count cannot be determined -/
def sandy_turnips_undetermined : Prop :=
  ∀ (n : ℕ),
    (∃ (sandy : VegetableCount) (mary : VegetableCount),
      sandy.carrots = 8 ∧
      mary.carrots = 6 ∧
      sandy.carrots + mary.carrots = 14 ∧
      sandy.turnips = n) →
    (∃ (sandy : VegetableCount) (mary : VegetableCount),
      sandy.carrots = 8 ∧
      mary.carrots = 6 ∧
      sandy.carrots + mary.carrots = 14 ∧
      sandy.turnips ≠ n)

theorem sandy_turnips_count_undetermined :
  given → sandy_turnips_undetermined :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_turnips_count_undetermined_l78_7875


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_1_real_l78_7882

theorem sqrt_2x_minus_1_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x - 1) ↔ x ≥ 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_1_real_l78_7882


namespace NUMINAMATH_CALUDE_point_position_on_line_l78_7833

/-- Given five points on a line and a point P satisfying certain conditions, prove the position of P -/
theorem point_position_on_line (a b c d : ℝ) :
  let O := (0 : ℝ)
  let A := a
  let B := b
  let C := c
  let D := d
  ∀ P, b ≤ P ∧ P ≤ c →
  (A - P) / (P - D) = (B - P) / (P - C) →
  P = (a * c - b * d) / (a - b + c - d) :=
by sorry

end NUMINAMATH_CALUDE_point_position_on_line_l78_7833


namespace NUMINAMATH_CALUDE_milburg_population_l78_7888

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

/-- The number of children in Milburg -/
def children : ℕ := 2987

/-- The total population of Milburg -/
def total_population : ℕ := grown_ups + children

theorem milburg_population : total_population = 8243 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l78_7888


namespace NUMINAMATH_CALUDE_park_tree_count_l78_7862

def park_trees (initial_maple : ℕ) (initial_poplar : ℕ) (oak : ℕ) : ℕ :=
  let planted_maple := 3 * initial_poplar
  let total_maple := initial_maple + planted_maple
  let planted_poplar := 3 * initial_poplar
  let total_poplar := initial_poplar + planted_poplar
  total_maple + total_poplar + oak

theorem park_tree_count :
  park_trees 2 5 4 = 32 := by sorry

end NUMINAMATH_CALUDE_park_tree_count_l78_7862


namespace NUMINAMATH_CALUDE_larger_number_is_84_l78_7825

theorem larger_number_is_84 (a b : ℕ+) (h1 : Nat.gcd a b = 84) (h2 : Nat.lcm a b = 21) (h3 : b = 4 * a) :
  b = 84 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_84_l78_7825


namespace NUMINAMATH_CALUDE_extended_fishing_rod_length_l78_7879

theorem extended_fishing_rod_length 
  (original_length : ℝ) 
  (increase_factor : ℝ) 
  (extended_length : ℝ) : 
  original_length = 48 → 
  increase_factor = 1.33 → 
  extended_length = original_length * increase_factor → 
  extended_length = 63.84 :=
by sorry

end NUMINAMATH_CALUDE_extended_fishing_rod_length_l78_7879


namespace NUMINAMATH_CALUDE_game_cost_calculation_l78_7834

theorem game_cost_calculation (initial_amount : ℕ) (spent_amount : ℕ) (num_games : ℕ) :
  initial_amount = 42 →
  spent_amount = 10 →
  num_games = 4 →
  num_games > 0 →
  ∃ (game_cost : ℕ), game_cost * num_games = initial_amount - spent_amount ∧ game_cost = 8 :=
by sorry

end NUMINAMATH_CALUDE_game_cost_calculation_l78_7834


namespace NUMINAMATH_CALUDE_first_digit_base_16_l78_7827

def base_4_representation : List ℕ := [2, 0, 3, 1, 3, 3, 2, 0, 1, 3, 2, 2, 2, 0, 3, 1, 2, 0, 3, 1]

def y : ℕ := (List.foldl (λ acc d => acc * 4 + d) 0 base_4_representation)

theorem first_digit_base_16 : ∃ (rest : ℕ), y = 5 * 16^rest + (y % 16^rest) ∧ y < 6 * 16^rest :=
sorry

end NUMINAMATH_CALUDE_first_digit_base_16_l78_7827


namespace NUMINAMATH_CALUDE_equation_solution_l78_7864

theorem equation_solution : ∃! x : ℝ, (4 : ℝ) ^ (x + 1) = (64 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l78_7864


namespace NUMINAMATH_CALUDE_not_all_zero_equiv_one_nonzero_l78_7832

theorem not_all_zero_equiv_one_nonzero (a b c : ℝ) :
  (¬(a = 0 ∧ b = 0 ∧ c = 0)) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_zero_equiv_one_nonzero_l78_7832


namespace NUMINAMATH_CALUDE_remainder_theorem_l78_7891

theorem remainder_theorem (n m p q r : ℤ)
  (hn : n % 18 = 10)
  (hm : m % 27 = 16)
  (hp : p % 6 = 4)
  (hq : q % 12 = 8)
  (hr : r % 3 = 2) :
  ((3*n + 2*m) - (p + q) / r) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l78_7891


namespace NUMINAMATH_CALUDE_triangle_side_sum_l78_7840

theorem triangle_side_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 60) (h3 : b = 30) (h4 : c = 90) 
  (side_opposite_30 : ℝ) (h5 : side_opposite_30 = 8 * Real.sqrt 3) :
  ∃ (other_sides_sum : ℝ), other_sides_sum = 12 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l78_7840


namespace NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l78_7853

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangleConditions (t : Triangle) : Prop :=
  t.c = 2 * Real.sqrt 3 ∧ Real.sin t.B = 2 * Real.sin t.A

-- Theorem 1: When C = π/3
theorem triangle_case1 (t : Triangle) (h : triangleConditions t) (hC : t.C = π / 3) :
  t.a = 2 ∧ t.b = 4 := by sorry

-- Theorem 2: When cos C = 1/4
theorem triangle_case2 (t : Triangle) (h : triangleConditions t) (hC : Real.cos t.C = 1 / 4) :
  (1 / 2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 15) / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l78_7853


namespace NUMINAMATH_CALUDE_product_of_w_and_x_is_zero_l78_7828

theorem product_of_w_and_x_is_zero 
  (w x y : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / y) 
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) : 
  w * x = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_w_and_x_is_zero_l78_7828


namespace NUMINAMATH_CALUDE_integer_solution_system_l78_7852

theorem integer_solution_system :
  ∀ A B C : ℤ,
  (A^2 - B^2 - C^2 = 1 ∧ B + C - A = 3) ↔
  ((A = 9 ∧ B = 8 ∧ C = 4) ∨
   (A = 9 ∧ B = 4 ∧ C = 8) ∨
   (A = -3 ∧ B = 2 ∧ C = -2) ∨
   (A = -3 ∧ B = -2 ∧ C = 2)) :=
by sorry


end NUMINAMATH_CALUDE_integer_solution_system_l78_7852


namespace NUMINAMATH_CALUDE_sets_in_borel_sigma_algebra_l78_7854

-- Define the type for infinite sequences of real numbers
def RealSequence := ℕ → ℝ

-- Define the Borel σ-algebra on ℝ^∞
def BorelSigmaAlgebra : Set (Set RealSequence) := sorry

-- Define the limsup of a sequence
def limsup (x : RealSequence) : ℝ := sorry

-- Define the limit of a sequence
def limit (x : RealSequence) : Option ℝ := sorry

-- Theorem statement
theorem sets_in_borel_sigma_algebra (a : ℝ) :
  {x : RealSequence | limsup x ≤ a} ∈ BorelSigmaAlgebra ∧
  {x : RealSequence | ∃ (l : ℝ), limit x = some l ∧ l > a} ∈ BorelSigmaAlgebra :=
sorry

end NUMINAMATH_CALUDE_sets_in_borel_sigma_algebra_l78_7854
