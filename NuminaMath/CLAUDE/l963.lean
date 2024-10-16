import Mathlib

namespace NUMINAMATH_CALUDE_expression_equals_one_l963_96394

theorem expression_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h_sum : a + b + c = 1) :
  (a^2 * b^2 / ((a^2 - b*c) * (b^2 - a*c))) +
  (a^2 * c^2 / ((a^2 - b*c) * (c^2 - a*b))) +
  (b^2 * c^2 / ((b^2 - a*c) * (c^2 - a*b))) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l963_96394


namespace NUMINAMATH_CALUDE_ana_driving_problem_l963_96331

theorem ana_driving_problem (initial_distance : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (target_average_speed : ℝ) (additional_distance : ℝ) :
  initial_distance = 20 →
  initial_speed = 40 →
  additional_speed = 70 →
  target_average_speed = 60 →
  additional_distance = 70 →
  (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / additional_speed)) = target_average_speed :=
by
  sorry

end NUMINAMATH_CALUDE_ana_driving_problem_l963_96331


namespace NUMINAMATH_CALUDE_f_2014_value_l963_96353

def N0 : Set ℕ := {n : ℕ | n ≥ 0}

def is_valid_f (f : ℕ → ℕ) : Prop :=
  f 2 = 0 ∧
  f 3 > 0 ∧
  f 6042 = 2014 ∧
  ∀ m n : ℕ, (f (m + n) - f m - f n) ∈ ({0, 1} : Set ℕ)

theorem f_2014_value (f : ℕ → ℕ) (h : is_valid_f f) : f 2014 = 671 := by
  sorry

end NUMINAMATH_CALUDE_f_2014_value_l963_96353


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l963_96337

/-- Represents the average percentage decrease in price per reduction -/
def average_decrease : ℝ := 0.25

/-- The original price of the medicine in yuan -/
def original_price : ℝ := 16

/-- The current price of the medicine in yuan -/
def current_price : ℝ := 9

/-- The number of successive price reductions -/
def num_reductions : ℕ := 2

theorem medicine_price_reduction :
  current_price = original_price * (1 - average_decrease) ^ num_reductions :=
by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l963_96337


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l963_96329

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop := (z.re = 0) ∧ (z.im ≠ 0)

/-- Given that b is a real number and (1+bi)(2+i) is a pure imaginary number, b equals 2 -/
theorem complex_product_pure_imaginary (b : ℝ) 
  (h : IsPureImaginary ((1 + b * Complex.I) * (2 + Complex.I))) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l963_96329


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_for_m_3_find_m_for_given_intersection_l963_96351

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 5 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_A_complement_B_for_m_3 : 
  A ∩ (Set.univ \ B 3) = {x | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Part 2
theorem find_m_for_given_intersection :
  ∃ m : ℝ, A ∩ B m = {x | -1 ≤ x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_for_m_3_find_m_for_given_intersection_l963_96351


namespace NUMINAMATH_CALUDE_factor_polynomial_l963_96372

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 270 * x^13 = 15 * x^7 * (5 - 18 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l963_96372


namespace NUMINAMATH_CALUDE_melissas_fabric_l963_96373

/-- The amount of fabric Melissa has given her work hours and dress requirements -/
theorem melissas_fabric (fabric_per_dress : ℝ) (hours_per_dress : ℝ) (total_work_hours : ℝ) :
  fabric_per_dress = 4 →
  hours_per_dress = 3 →
  total_work_hours = 42 →
  (total_work_hours / hours_per_dress) * fabric_per_dress = 56 := by
  sorry

end NUMINAMATH_CALUDE_melissas_fabric_l963_96373


namespace NUMINAMATH_CALUDE_olivias_carrots_l963_96310

theorem olivias_carrots (mom_carrots : ℕ) (good_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : mom_carrots = 14)
  (h2 : good_carrots = 19)
  (h3 : bad_carrots = 15) :
  good_carrots + bad_carrots - mom_carrots = 20 := by
  sorry

end NUMINAMATH_CALUDE_olivias_carrots_l963_96310


namespace NUMINAMATH_CALUDE_julie_delivered_600_newspapers_l963_96339

/-- Represents Julie's earnings and expenses --/
structure JulieFinances where
  saved : ℕ
  bikeCost : ℕ
  lawnsMowed : ℕ
  lawnRate : ℕ
  dogsWalked : ℕ
  dogRate : ℕ
  newspaperRate : ℕ
  leftover : ℕ

/-- Calculates the number of newspapers Julie delivered --/
def newspapersDelivered (j : JulieFinances) : ℕ :=
  ((j.bikeCost + j.leftover) - (j.saved + j.lawnsMowed * j.lawnRate + j.dogsWalked * j.dogRate)) / j.newspaperRate

/-- Theorem stating that Julie delivered 600 newspapers --/
theorem julie_delivered_600_newspapers :
  let j : JulieFinances := {
    saved := 1500,
    bikeCost := 2345,
    lawnsMowed := 20,
    lawnRate := 20,
    dogsWalked := 24,
    dogRate := 15,
    newspaperRate := 40,  -- in cents
    leftover := 155
  }
  newspapersDelivered j = 600 := by sorry


end NUMINAMATH_CALUDE_julie_delivered_600_newspapers_l963_96339


namespace NUMINAMATH_CALUDE_M_intersect_N_empty_l963_96347

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem M_intersect_N_empty : M ∩ (N.prod Set.univ) = ∅ := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_empty_l963_96347


namespace NUMINAMATH_CALUDE_picture_tube_consignment_l963_96359

theorem picture_tube_consignment (defective : ℕ) (prob : ℚ) (total : ℕ) : 
  defective = 5 →
  prob = 5263157894736842 / 100000000000000000 →
  (defective : ℚ) / total * (defective - 1 : ℚ) / (total - 1) = prob →
  total = 20 := by
  sorry

end NUMINAMATH_CALUDE_picture_tube_consignment_l963_96359


namespace NUMINAMATH_CALUDE_lemonade_stand_operational_cost_l963_96358

/-- Yulia's lemonade stand finances -/
def lemonade_stand_finances (net_profit babysitting_revenue lemonade_revenue : ℕ) : Prop :=
  ∃ (operational_cost : ℕ),
    net_profit + operational_cost = babysitting_revenue + lemonade_revenue ∧
    operational_cost = 34

/-- Theorem: Given Yulia's financial information, prove that her lemonade stand's operational cost is $34 -/
theorem lemonade_stand_operational_cost :
  lemonade_stand_finances 44 31 47 :=
by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_operational_cost_l963_96358


namespace NUMINAMATH_CALUDE_grace_and_henry_weight_l963_96341

/-- Given the weights of pairs of people, prove that Grace and Henry weigh 250 pounds together. -/
theorem grace_and_henry_weight
  (e f g h : ℝ)  -- Weights of Ella, Finn, Grace, and Henry
  (h1 : e + f = 280)  -- Ella and Finn weigh 280 pounds together
  (h2 : f + g = 230)  -- Finn and Grace weigh 230 pounds together
  (h3 : e + h = 300)  -- Ella and Henry weigh 300 pounds together
  : g + h = 250 := by
  sorry

end NUMINAMATH_CALUDE_grace_and_henry_weight_l963_96341


namespace NUMINAMATH_CALUDE_negative_x_over_two_abs_x_positive_l963_96379

theorem negative_x_over_two_abs_x_positive (x : ℝ) (h : x < 0) :
  -x / (2 * |x|) > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_over_two_abs_x_positive_l963_96379


namespace NUMINAMATH_CALUDE_tiffany_albums_l963_96378

theorem tiffany_albums (phone_pics camera_pics pics_per_album : ℕ) 
  (h1 : phone_pics = 7)
  (h2 : camera_pics = 13)
  (h3 : pics_per_album = 4) :
  (phone_pics + camera_pics) / pics_per_album = 5 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_albums_l963_96378


namespace NUMINAMATH_CALUDE_hyperbola_equation_l963_96311

theorem hyperbola_equation (ellipse : (ℝ × ℝ) → Prop) 
  (ellipse_eq : ∀ x y, ellipse (x, y) ↔ x^2/27 + y^2/36 = 1)
  (shared_foci : ∃ f1 f2 : ℝ × ℝ, (∀ x y, ellipse (x, y) → 
    (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 36) ∧
    (∀ x y, x^2/4 - y^2/5 = 1 → 
    (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 9))
  (point_on_hyperbola : (Real.sqrt 15)^2/4 - 4^2/5 = 1) :
  ∀ x y, x^2/4 - y^2/5 = 1 ↔ 
    ∃ f1 f2 : ℝ × ℝ, (∀ a b, ellipse (a, b) → 
    (a - f1.1)^2 + (b - f1.2)^2 - ((a - f2.1)^2 + (b - f2.2)^2) = 36) ∧
    (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 9 :=
sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l963_96311


namespace NUMINAMATH_CALUDE_share_division_l963_96356

theorem share_division (total : ℚ) (a b c : ℚ) 
  (h1 : total = 595)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) :
  c = 420 := by sorry

end NUMINAMATH_CALUDE_share_division_l963_96356


namespace NUMINAMATH_CALUDE_sum_of_proportional_values_l963_96318

theorem sum_of_proportional_values (a b c d e f : ℝ) 
  (h1 : a / b = 4 / 3)
  (h2 : c / d = 4 / 3)
  (h3 : e / f = 4 / 3)
  (h4 : b + d + f = 15) :
  a + c + e = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_proportional_values_l963_96318


namespace NUMINAMATH_CALUDE_ratio_problem_l963_96328

theorem ratio_problem (x y z : ℚ) 
  (h1 : x / y = 4 / 7) 
  (h2 : z / x = 3 / 5) : 
  (x + y) / (z + x) = 55 / 32 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l963_96328


namespace NUMINAMATH_CALUDE_base_comparison_l963_96386

theorem base_comparison (a b n : ℕ) (A_n B_n A_n_minus_1 B_n_minus_1 : ℕ) 
  (ha : a > 1) (hb : b > 1) (hn : n > 1)
  (hA : A_n > 0) (hB : B_n > 0) (hA_minus_1 : A_n_minus_1 > 0) (hB_minus_1 : B_n_minus_1 > 0)
  (hA_def : A_n = a^n + A_n_minus_1) (hB_def : B_n = b^n + B_n_minus_1) :
  (a > b) ↔ (A_n_minus_1 / A_n : ℚ) < (B_n_minus_1 / B_n : ℚ) := by
sorry

end NUMINAMATH_CALUDE_base_comparison_l963_96386


namespace NUMINAMATH_CALUDE_hot_air_balloon_balloons_l963_96305

theorem hot_air_balloon_balloons (initial_balloons : ℕ) : 
  (initial_balloons : ℚ) * (2 / 5) = 80 → initial_balloons = 200 :=
by
  sorry

#check hot_air_balloon_balloons

end NUMINAMATH_CALUDE_hot_air_balloon_balloons_l963_96305


namespace NUMINAMATH_CALUDE_product_variation_l963_96365

theorem product_variation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (5 * a) * b = 5 * (a * b) := by sorry

end NUMINAMATH_CALUDE_product_variation_l963_96365


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l963_96345

theorem ellipse_focal_distance (m : ℝ) :
  (∀ x y : ℝ, x^2/16 + y^2/m = 1) →
  (∃ c : ℝ, c > 0 ∧ c^2 = 16 - m ∧ 2*c = 2*Real.sqrt 7) →
  m = 9 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l963_96345


namespace NUMINAMATH_CALUDE_vehicle_speed_problem_l963_96384

/-- Represents the problem of determining initial and final speeds of a vehicle --/
theorem vehicle_speed_problem
  (total_distance : ℝ)
  (initial_distance : ℝ)
  (initial_time : ℝ)
  (late_time : ℝ)
  (early_time : ℝ)
  (h1 : total_distance = 280)
  (h2 : initial_distance = 112)
  (h3 : initial_time = 2)
  (h4 : late_time = 0.5)
  (h5 : early_time = 0.5)
  : ∃ (initial_speed final_speed : ℝ),
    initial_speed = initial_distance / initial_time ∧
    final_speed = (total_distance - initial_distance) / (
      (total_distance / initial_speed - late_time) - initial_time
    ) ∧
    initial_speed = 56 ∧
    final_speed = 84 := by
  sorry


end NUMINAMATH_CALUDE_vehicle_speed_problem_l963_96384


namespace NUMINAMATH_CALUDE_library_donation_l963_96366

/-- The number of books donated to the library --/
def books_donated (num_students : ℕ) (books_per_student : ℕ) (shortfall : ℕ) : ℕ :=
  num_students * books_per_student - shortfall

/-- Theorem stating the number of books donated to the library --/
theorem library_donation (num_students : ℕ) (books_per_student : ℕ) (shortfall : ℕ) :
  books_donated num_students books_per_student shortfall = 294 :=
by
  sorry

#eval books_donated 20 15 6

end NUMINAMATH_CALUDE_library_donation_l963_96366


namespace NUMINAMATH_CALUDE_parentheses_removal_equality_l963_96301

theorem parentheses_removal_equality (x : ℝ) : -(x - 2) - 2 * (x^2 + 2) = -x + 2 - 2*x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_equality_l963_96301


namespace NUMINAMATH_CALUDE_equation_solutions_l963_96390

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x^2 + 2*x)^(1/3) + (3*x^2 + 6*x - 4)^(1/3) = (x^2 + 2*x - 4)^(1/3)

-- Theorem statement
theorem equation_solutions :
  {x : ℝ | equation x} = {-2, 0} :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l963_96390


namespace NUMINAMATH_CALUDE_complement_union_problem_l963_96364

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 3, 4}
def B : Finset Nat := {2, 4}

theorem complement_union_problem :
  (U \ A) ∪ B = {2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_problem_l963_96364


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l963_96361

theorem modular_arithmetic_problem :
  ∃ (a b : ℕ), 
    (7 * a) % 60 = 1 ∧ 
    (13 * b) % 60 = 1 ∧ 
    ((3 * a + 9 * b) % 60 : ℕ) = 42 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l963_96361


namespace NUMINAMATH_CALUDE_new_shoes_count_l963_96307

theorem new_shoes_count (pairs_bought : ℕ) (shoes_per_pair : ℕ) : 
  pairs_bought = 3 → shoes_per_pair = 2 → pairs_bought * shoes_per_pair = 6 := by
  sorry

end NUMINAMATH_CALUDE_new_shoes_count_l963_96307


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_six_l963_96316

theorem largest_five_digit_divisible_by_six : ∃ n : ℕ,
  n = 99996 ∧
  n ≥ 10000 ∧
  n < 100000 ∧
  n % 6 = 0 ∧
  ∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 6 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_six_l963_96316


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l963_96303

theorem average_of_remaining_numbers 
  (total : ℝ) 
  (group1 : ℝ) 
  (group2 : ℝ) 
  (h1 : total = 6 * 3.95) 
  (h2 : group1 = 2 * 4.2) 
  (h3 : group2 = 2 * 3.85) : 
  (total - group1 - group2) / 2 = 3.8 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l963_96303


namespace NUMINAMATH_CALUDE_det_trig_matrix_l963_96392

open Real Matrix

theorem det_trig_matrix (a b : ℝ) : 
  det !![1, sin (a + b), cos a; 
         sin (a + b), 1, sin b; 
         cos a, sin b, 1] = 
  2 * sin (a + b) * sin b * cos a + sin (a + b)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_det_trig_matrix_l963_96392


namespace NUMINAMATH_CALUDE_second_artifact_time_multiple_l963_96350

/-- Represents the time spent on artifact collection in months -/
structure ArtifactTime where
  research : ℕ
  expedition : ℕ

/-- The total time spent on both artifacts in months -/
def total_time : ℕ := 10 * 12

/-- Time spent on the first artifact -/
def first_artifact : ArtifactTime := { research := 6, expedition := 2 * 12 }

/-- Calculate the total time spent on an artifact -/
def total_artifact_time (a : ArtifactTime) : ℕ := a.research + a.expedition

/-- The multiple of time taken for the second artifact compared to the first -/
def time_multiple : ℚ :=
  (total_time - total_artifact_time first_artifact) / total_artifact_time first_artifact

theorem second_artifact_time_multiple :
  time_multiple = 3 := by sorry

end NUMINAMATH_CALUDE_second_artifact_time_multiple_l963_96350


namespace NUMINAMATH_CALUDE_probability_mathematics_in_machine_l963_96382

def mathematics_letters : Finset Char := {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S'}
def machine_letters : Finset Char := {'M', 'A', 'C', 'H', 'I', 'N', 'E'}

theorem probability_mathematics_in_machine :
  (mathematics_letters.filter (λ c => c ∈ machine_letters)).card / mathematics_letters.card = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_mathematics_in_machine_l963_96382


namespace NUMINAMATH_CALUDE_break_difference_l963_96324

def work_duration : ℕ := 240
def water_break_interval : ℕ := 20
def sitting_break_interval : ℕ := 120

def water_breaks : ℕ := work_duration / water_break_interval
def sitting_breaks : ℕ := work_duration / sitting_break_interval

theorem break_difference : water_breaks - sitting_breaks = 10 := by
  sorry

end NUMINAMATH_CALUDE_break_difference_l963_96324


namespace NUMINAMATH_CALUDE_fraction_equality_l963_96314

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (5 * a + 2 * b) / (2 * a - 5 * b) = 3) : 
  (2 * a + 5 * b) / (5 * a - 2 * b) = 39 / 83 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l963_96314


namespace NUMINAMATH_CALUDE_total_profit_is_45000_l963_96312

/-- Represents the total profit earned by Tom and Jose given their investments and Jose's share of profit. -/
def total_profit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  let tom_ratio : ℕ := tom_investment * tom_months
  let jose_ratio : ℕ := jose_investment * jose_months
  let total_ratio : ℕ := tom_ratio + jose_ratio
  (jose_profit * total_ratio) / jose_ratio

/-- Theorem stating that the total profit is 45000 given the specified conditions. -/
theorem total_profit_is_45000 :
  total_profit 30000 12 45000 10 25000 = 45000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_45000_l963_96312


namespace NUMINAMATH_CALUDE_fans_with_all_items_l963_96367

def stadium_capacity : ℕ := 4800
def scarf_interval : ℕ := 80
def hat_interval : ℕ := 40
def whistle_interval : ℕ := 60

theorem fans_with_all_items :
  (stadium_capacity / (lcm scarf_interval (lcm hat_interval whistle_interval))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l963_96367


namespace NUMINAMATH_CALUDE_f_properties_l963_96377

-- Define the function f(x) = x^3 - 6x + 5
def f (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the theorem for the extreme points and the range of k
theorem f_properties :
  -- Part I: Extreme points
  (∃ (x_max x_min : ℝ), x_max = -Real.sqrt 2 ∧ x_min = Real.sqrt 2 ∧
    (∀ (x : ℝ), f x ≤ f x_max) ∧
    (∀ (x : ℝ), f x ≥ f x_min)) ∧
  -- Part II: Range of k
  (∀ (k : ℝ), (∀ (x : ℝ), x > 1 → f x ≥ k * (x - 1)) ↔ k ≤ -3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l963_96377


namespace NUMINAMATH_CALUDE_tan_fifteen_fraction_equals_sqrt_three_l963_96304

theorem tan_fifteen_fraction_equals_sqrt_three : 
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_fraction_equals_sqrt_three_l963_96304


namespace NUMINAMATH_CALUDE_proposition_equivalence_l963_96363

theorem proposition_equivalence (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, x^2 - a ≤ 0) ↔ a ≥ 9 := by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l963_96363


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_to_yaxis_l963_96355

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- A circle with center c and radius r -/
structure Circle where
  center : Point
  radius : ℝ

/-- The y-axis -/
def yAxis := {p : Point | p.x = 0}

/-- Predicate to check if a circle is tangent to the y-axis -/
def isTangentToYAxis (c : Circle) : Prop :=
  c.center.x = c.radius

/-- Theorem: For any point P on the parabola y² = 4x, 
    the circle with diameter PF (where F is the focus) 
    is tangent to the y-axis -/
theorem parabola_circle_tangent_to_yaxis 
  (P : Point) (h : P ∈ Parabola) : 
  ∃ (c : Circle), c.center = ⟨(P.x + focus.x) / 2, P.y / 2⟩ ∧ 
                  c.radius = (P.x + focus.x) / 2 ∧
                  isTangentToYAxis c :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_to_yaxis_l963_96355


namespace NUMINAMATH_CALUDE_intersection_point_theorem_l963_96371

theorem intersection_point_theorem (α β : ℝ) :
  (∃ x y : ℝ, 
    x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1 ∧
    x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1 ∧
    y = -x) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_theorem_l963_96371


namespace NUMINAMATH_CALUDE_min_tosses_for_heads_l963_96381

theorem min_tosses_for_heads (p : ℝ) (h_p : p = 1/2) :
  ∃ n : ℕ, n ≥ 1 ∧
  (∀ k : ℕ, k ≥ n → 1 - p^k ≥ 15/16) ∧
  (∀ k : ℕ, k < n → 1 - p^k < 15/16) ∧
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_min_tosses_for_heads_l963_96381


namespace NUMINAMATH_CALUDE_system_sum_l963_96330

theorem system_sum (x y z : ℝ) 
  (eq1 : x + y = 4)
  (eq2 : y + z = 6)
  (eq3 : z + x = 8) :
  x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_system_sum_l963_96330


namespace NUMINAMATH_CALUDE_max_value_of_f_l963_96315

def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ 
  (∀ x, x ∈ Set.Icc 0 1 → f x ≤ f c) ∧
  f c = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l963_96315


namespace NUMINAMATH_CALUDE_triangle_angle_cosine_l963_96346

theorem triangle_angle_cosine (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  A + C = 2 * B ∧ -- Given condition
  1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B -- Given condition
  → Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 := by
    sorry

end NUMINAMATH_CALUDE_triangle_angle_cosine_l963_96346


namespace NUMINAMATH_CALUDE_rotate_d_180_degrees_l963_96336

/-- Rotation of a point by 180° about the origin -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotate_d_180_degrees :
  let d : ℝ × ℝ := (2, -3)
  rotate180 d = (-2, 3) := by
  sorry

end NUMINAMATH_CALUDE_rotate_d_180_degrees_l963_96336


namespace NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l963_96334

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- Define the derivative of f
def f_deriv (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- State the theorem
theorem derivative_even_implies_b_zero (a b c : ℝ) :
  (∀ x : ℝ, f_deriv a b c x = f_deriv a b c (-x)) →
  b = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l963_96334


namespace NUMINAMATH_CALUDE_pizza_cost_is_seven_l963_96387

def pizza_problem (box_cost : ℚ) : Prop :=
  let num_boxes : ℕ := 5
  let tip_ratio : ℚ := 1 / 7
  let total_paid : ℚ := 40
  let pizza_cost : ℚ := box_cost * num_boxes
  let tip : ℚ := pizza_cost * tip_ratio
  pizza_cost + tip = total_paid

theorem pizza_cost_is_seven :
  ∃ (box_cost : ℚ), pizza_problem box_cost ∧ box_cost = 7 :=
by sorry

end NUMINAMATH_CALUDE_pizza_cost_is_seven_l963_96387


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l963_96342

theorem arctan_equation_solution :
  ∃ x : ℝ, 2 * Real.arctan (1/5) + 2 * Real.arctan (1/10) + Real.arctan (1/x) = π/2 ∧ x = 120/119 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l963_96342


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_3_mod_17_l963_96357

theorem least_five_digit_congruent_to_3_mod_17 :
  ∀ n : ℕ, n ≥ 10000 → n ≡ 3 [ZMOD 17] → n ≥ 10004 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_3_mod_17_l963_96357


namespace NUMINAMATH_CALUDE_cricket_average_l963_96321

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 10 → 
  next_runs = 81 → 
  increase = 4 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase → 
  current_average = 37 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l963_96321


namespace NUMINAMATH_CALUDE_integral_sqrt_rational_equals_pi_sixth_l963_96322

theorem integral_sqrt_rational_equals_pi_sixth :
  ∫ x in (2 : ℝ)..3, Real.sqrt ((3 - 2*x) / (2*x - 7)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_rational_equals_pi_sixth_l963_96322


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l963_96319

def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  isGeometricSequence a →
  a 1 = 2 →
  (∀ n, a (n + 2)^2 + 4 * a n^2 = 4 * a (n + 1)^2) →
  ∀ n, a n = 2^((n + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l963_96319


namespace NUMINAMATH_CALUDE_average_value_equals_combination_l963_96320

def average_value (n : ℕ) : ℚ :=
  (n + 1) * n * (n - 1) / 6

theorem average_value_equals_combination (n : ℕ) (h : n > 0) :
  average_value n = Nat.choose (n + 1) 3 := by sorry

end NUMINAMATH_CALUDE_average_value_equals_combination_l963_96320


namespace NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l963_96360

theorem max_stamps_with_50_dollars (stamp_price : ℚ) (total_money : ℚ) :
  stamp_price = 25 / 100 →
  total_money = 50 →
  ⌊total_money / stamp_price⌋ = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_with_50_dollars_l963_96360


namespace NUMINAMATH_CALUDE_summit_conference_attendance_l963_96313

/-- The number of diplomats who attended the summit conference -/
def D : ℕ := 120

/-- The number of diplomats who spoke French -/
def french_speakers : ℕ := 20

/-- The number of diplomats who did not speak Hindi -/
def non_hindi_speakers : ℕ := 32

/-- The proportion of diplomats who spoke neither French nor Hindi -/
def neither_french_nor_hindi : ℚ := 1/5

/-- The proportion of diplomats who spoke both French and Hindi -/
def both_french_and_hindi : ℚ := 1/10

theorem summit_conference_attendance :
  D = 120 ∧
  french_speakers = 20 ∧
  non_hindi_speakers = 32 ∧
  neither_french_nor_hindi = 1/5 ∧
  both_french_and_hindi = 1/10 ∧
  (D : ℚ) * neither_french_nor_hindi + (D : ℚ) * both_french_and_hindi + french_speakers = D :=
sorry

end NUMINAMATH_CALUDE_summit_conference_attendance_l963_96313


namespace NUMINAMATH_CALUDE_range_equals_std_dev_l963_96349

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation
  symmetric : Bool
  within_range : ℝ → ℝ  -- function that gives the proportion within a range
  less_than : ℝ → ℝ  -- function that gives the proportion less than a value

/-- Theorem stating the relationship between the range and standard deviation -/
theorem range_equals_std_dev (D : SymmetricDistribution) (R : ℝ) :
  D.symmetric = true →
  D.within_range R = 0.68 →
  D.less_than (D.μ + R) = 0.84 →
  R = D.σ :=
by sorry

end NUMINAMATH_CALUDE_range_equals_std_dev_l963_96349


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l963_96389

/-- Triangle ABC with vertices A(-4,0), B(0,2), and C(2,-2) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The equation of a circle in the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure CircleEquation :=
  (D : ℝ)
  (E : ℝ)
  (F : ℝ)

/-- Function to check if a point satisfies a line equation -/
def satisfiesLineEquation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Function to check if a point satisfies a circle equation -/
def satisfiesCircleEquation (p : ℝ × ℝ) (eq : CircleEquation) : Prop :=
  p.1^2 + p.2^2 + eq.D * p.1 + eq.E * p.2 + eq.F = 0

/-- Theorem stating the properties of triangle ABC -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.A = (-4, 0))
  (h2 : t.B = (0, 2))
  (h3 : t.C = (2, -2)) :
  ∃ (medianAB : LineEquation) (circumcircle : CircleEquation),
    -- Median equation
    (medianAB = ⟨3, 4, -2⟩) ∧ 
    -- Circumcircle equation
    (circumcircle = ⟨2, 2, -8⟩) ∧
    -- Verify that C and the midpoint of AB satisfy the median equation
    (satisfiesLineEquation t.C medianAB) ∧
    (satisfiesLineEquation ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2) medianAB) ∧
    -- Verify that all vertices satisfy the circumcircle equation
    (satisfiesCircleEquation t.A circumcircle) ∧
    (satisfiesCircleEquation t.B circumcircle) ∧
    (satisfiesCircleEquation t.C circumcircle) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l963_96389


namespace NUMINAMATH_CALUDE_intersection_parallel_line_exists_specific_intersection_parallel_line_l963_96380

/-- Given two lines l₁ and l₂ in the plane, and a third line l₃,
    this theorem states that there exists a line l that passes through
    the intersection of l₁ and l₂ and is parallel to l₃. -/
theorem intersection_parallel_line_exists (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℝ) :
  ∃ (a b c : ℝ),
    -- l₁: a₁x + b₁y + c₁ = 0
    -- l₂: a₂x + b₂y + c₂ = 0
    -- l₃: a₃x + b₃y + c₃ = 0
    -- l: ax + by + c = 0
    -- l passes through the intersection of l₁ and l₂
    (∀ (x y : ℝ), a₁ * x + b₁ * y + c₁ = 0 ∧ a₂ * x + b₂ * y + c₂ = 0 → a * x + b * y + c = 0) ∧
    -- l is parallel to l₃
    (∃ (k : ℝ), k ≠ 0 ∧ a = k * a₃ ∧ b = k * b₃) :=
by
  sorry

/-- The specific instance of the theorem for the given problem -/
theorem specific_intersection_parallel_line :
  ∃ (a b c : ℝ),
    -- l₁: 2x + 3y - 5 = 0
    -- l₂: 3x - 2y - 3 = 0
    -- l₃: 2x + y - 3 = 0
    -- l: ax + by + c = 0
    (∀ (x y : ℝ), 2 * x + 3 * y - 5 = 0 ∧ 3 * x - 2 * y - 3 = 0 → a * x + b * y + c = 0) ∧
    (∃ (k : ℝ), k ≠ 0 ∧ a = k * 2 ∧ b = k * 1) ∧
    a = 26 ∧ b = -13 ∧ c = -29 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_exists_specific_intersection_parallel_line_l963_96380


namespace NUMINAMATH_CALUDE_ellipse_equation_l963_96385

-- Define the hyperbola E1
def E1 (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the ellipse E2
def E2 (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition that a > b > 0
def ellipse_condition (a b : ℝ) : Prop := a > b ∧ b > 0

-- Define the common focus condition
def common_focus (E1 E2 : ℝ → ℝ → Prop) : Prop := 
  ∃ (x y : ℝ), E1 x y ∧ E2 x y

-- Define the intersection condition
def intersect_in_quadrants (E1 E2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), E1 x1 y1 ∧ E2 x1 y1 ∧ E1 x2 y2 ∧ E2 x2 y2 ∧
    x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 < 0

-- Define the condition that chord MN passes through focus F2
def chord_through_focus (E1 E2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), E1 x1 y1 ∧ E2 x1 y1 ∧ E1 x2 y2 ∧ E2 x2 y2 ∧
    (y1 - y2) / (x1 - x2) = (y1 - 0) / (x1 - 3)

theorem ellipse_equation :
  ∀ (a b : ℝ),
    ellipse_condition a b →
    common_focus E1 (E2 · · a b) →
    intersect_in_quadrants E1 (E2 · · a b) →
    chord_through_focus E1 (E2 · · a b) →
    a^2 = 81/4 ∧ b^2 = 45/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l963_96385


namespace NUMINAMATH_CALUDE_solve_r_system_l963_96374

theorem solve_r_system (r s : ℚ) : 
  (r - 60) / 3 = (5 - 3 * r) / 4 → 
  s + 2 * r = 10 → 
  r = 255 / 13 := by
sorry

end NUMINAMATH_CALUDE_solve_r_system_l963_96374


namespace NUMINAMATH_CALUDE_original_number_proof_l963_96308

theorem original_number_proof (w : ℝ) : 
  (w + 0.125 * w) - (w - 0.25 * w) = 30 → w = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l963_96308


namespace NUMINAMATH_CALUDE_vieta_cubic_formulas_l963_96376

theorem vieta_cubic_formulas (a b c d x₁ x₂ x₃ : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = a * (x - x₁) * (x - x₂) * (x - x₃)) →
  (x₁ + x₂ + x₃ = -b / a) ∧ 
  (x₁ * x₂ + x₁ * x₃ + x₂ * x₃ = c / a) ∧ 
  (x₁ * x₂ * x₃ = -d / a) := by
  sorry

end NUMINAMATH_CALUDE_vieta_cubic_formulas_l963_96376


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l963_96309

/-- Represents an arrangement of numbers satisfying the given conditions -/
def ValidArrangement (n : ℕ) := List ℕ

/-- Checks if the arrangement is valid for a given n -/
def isValidArrangement (n : ℕ) (arr : ValidArrangement n) : Prop :=
  (arr.length = 2*n + 1) ∧
  (arr.count 0 = 1) ∧
  (∀ m : ℕ, m ≥ 1 → m ≤ n → arr.count m = 2) ∧
  (∀ m : ℕ, m ≥ 1 → m ≤ n → 
    ∃ i j : ℕ, i < j ∧ 
    (arr.get! i = m) ∧ 
    (arr.get! j = m) ∧ 
    (j - i - 1 = m))

/-- Theorem stating that a valid arrangement exists for any natural number n -/
theorem valid_arrangement_exists (n : ℕ) : ∃ arr : ValidArrangement n, isValidArrangement n arr :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l963_96309


namespace NUMINAMATH_CALUDE_prob_both_divisible_by_four_is_one_sixteenth_l963_96352

/-- Represents a fair 12-sided die -/
def TwelveSidedDie := Fin 12

/-- The probability of getting a number divisible by 4 on a 12-sided die -/
def prob_divisible_by_four (die : TwelveSidedDie) : ℚ :=
  3 / 12

/-- The probability of getting two numbers divisible by 4 when tossing two 12-sided dice -/
def prob_both_divisible_by_four (die1 die2 : TwelveSidedDie) : ℚ :=
  (prob_divisible_by_four die1) * (prob_divisible_by_four die2)

theorem prob_both_divisible_by_four_is_one_sixteenth :
  ∀ (die1 die2 : TwelveSidedDie), prob_both_divisible_by_four die1 die2 = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_divisible_by_four_is_one_sixteenth_l963_96352


namespace NUMINAMATH_CALUDE_unique_solution_condition_l963_96306

theorem unique_solution_condition (k : ℚ) : 
  (∃! x : ℝ, (x + 3) * (x + 2) = k + 3 * x) ↔ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l963_96306


namespace NUMINAMATH_CALUDE_price_adjustment_l963_96333

theorem price_adjustment (original_price : ℝ) (original_price_pos : original_price > 0) :
  let increased_price := original_price * (1 + 0.25)
  let decrease_percentage := (increased_price - original_price) / increased_price
  decrease_percentage = 0.20 := by
sorry

end NUMINAMATH_CALUDE_price_adjustment_l963_96333


namespace NUMINAMATH_CALUDE_sin_range_theorem_l963_96327

theorem sin_range_theorem (x : ℝ) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  Real.sin x ≥ Real.sqrt 2 / 2 → 
  x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_sin_range_theorem_l963_96327


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l963_96370

theorem quadratic_equations_solutions :
  (∃ x : ℝ, 2*x^2 - 4*x - 1 = 0) ∧
  (∃ x : ℝ, 4*(x+2)^2 - 9*(x-3)^2 = 0) ∧
  (∀ x : ℝ, 2*x^2 - 4*x - 1 = 0 → x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) ∧
  (∀ x : ℝ, 4*(x+2)^2 - 9*(x-3)^2 = 0 → x = 1 ∨ x = 13) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l963_96370


namespace NUMINAMATH_CALUDE_max_operation_value_l963_96340

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The operation to be maximized -/
def operation (X Y Z : Digit) : ℕ := 99 * X.val + 9 * Y.val - 9 * Z.val

/-- The theorem statement -/
theorem max_operation_value :
  ∃ (X Y Z : Digit), X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧
    operation X Y Z = 900 ∧
    ∀ (A B C : Digit), A ≠ B → B ≠ C → A ≠ C →
      operation A B C ≤ 900 :=
sorry

end NUMINAMATH_CALUDE_max_operation_value_l963_96340


namespace NUMINAMATH_CALUDE_solution_relationship_l963_96397

theorem solution_relationship (x y : ℝ) : 
  (2 * x + y = 7) → (x - y = 5) → (x + 2 * y = 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_relationship_l963_96397


namespace NUMINAMATH_CALUDE_range_of_a_l963_96338

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < -3 ∨ x > 1
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (h : ∀ x, ¬(p x) → ¬(q x a)) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l963_96338


namespace NUMINAMATH_CALUDE_trig_identity_l963_96344

theorem trig_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l963_96344


namespace NUMINAMATH_CALUDE_x_intercept_distance_l963_96369

/-- Given two lines intersecting at (8, 20), one with slope 4 and the other with slope -3,
    the distance between their x-intercepts is 35/3. -/
theorem x_intercept_distance (line1 line2 : (ℝ → ℝ)) : 
  (∀ x, line1 x = 4 * x - 12) →
  (∀ x, line2 x = -3 * x + 44) →
  line1 8 = 20 →
  line2 8 = 20 →
  |((0 - (-12)) / 4) - ((0 - 44) / (-3))| = 35/3 := by
sorry

end NUMINAMATH_CALUDE_x_intercept_distance_l963_96369


namespace NUMINAMATH_CALUDE_square_area_equals_side_perimeter_l963_96393

/-- A square with area numerically equal to its side length has a perimeter of 4 units. -/
theorem square_area_equals_side_perimeter :
  ∀ s : ℝ, s > 0 → s^2 = s → 4 * s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_side_perimeter_l963_96393


namespace NUMINAMATH_CALUDE_identify_counterfeit_l963_96362

/-- Represents a coin with its denomination and weight -/
structure Coin where
  denomination : Nat
  weight : Nat

/-- Represents the state of a balance scale -/
inductive Balance
  | Left
  | Right
  | Equal

/-- Represents a weighing operation on the balance scale -/
def weigh (left right : List Coin) : Balance :=
  sorry

/-- Represents the set of coins -/
def coins : List Coin :=
  [⟨1, 1⟩, ⟨2, 2⟩, ⟨3, 3⟩, ⟨5, 5⟩]

/-- Represents the counterfeit coin -/
def counterfeit : Coin :=
  sorry

/-- The main theorem stating that the counterfeit coin can be identified in two weighings -/
theorem identify_counterfeit :
  ∃ (weighing1 weighing2 : List Coin × List Coin),
    let result1 := weigh weighing1.1 weighing1.2
    let result2 := weigh weighing2.1 weighing2.2
    ∃ (identified : Coin), identified = counterfeit :=
  sorry

end NUMINAMATH_CALUDE_identify_counterfeit_l963_96362


namespace NUMINAMATH_CALUDE_blouse_price_proof_l963_96300

/-- The original price of a blouse before discount -/
def original_price : ℝ := 180

/-- The discount percentage applied to the blouse -/
def discount_percentage : ℝ := 18

/-- The price paid after applying the discount -/
def discounted_price : ℝ := 147.60

/-- Theorem stating that the original price is correct given the discount and discounted price -/
theorem blouse_price_proof : 
  original_price * (1 - discount_percentage / 100) = discounted_price := by
  sorry

end NUMINAMATH_CALUDE_blouse_price_proof_l963_96300


namespace NUMINAMATH_CALUDE_expression_value_l963_96325

theorem expression_value (y d : ℝ) (h1 : y > 0) 
  (h2 : (8 * y) / 20 + (3 * y) / d = 0.7 * y) : d = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l963_96325


namespace NUMINAMATH_CALUDE_circle_equation_l963_96332

/-- A circle C in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line 2x - y + 3 = 0 -/
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

/-- Circle C satisfies the given conditions -/
def satisfies_conditions (C : Circle) : Prop :=
  let (a, b) := C.center
  line a b ∧
  (1 - a)^2 + (3 - b)^2 = C.radius^2 ∧
  (3 - a)^2 + (5 - b)^2 = C.radius^2

/-- The standard equation of circle C -/
def standard_equation (C : Circle) (x y : ℝ) : Prop :=
  let (a, b) := C.center
  (x - a)^2 + (y - b)^2 = C.radius^2

theorem circle_equation :
  ∃ C : Circle, satisfies_conditions C ∧
    ∀ x y : ℝ, standard_equation C x y ↔ (x - 1)^2 + (y - 5)^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l963_96332


namespace NUMINAMATH_CALUDE_rotation_theorem_l963_96391

/-- Represents a square board with side length 2^n -/
structure Board (n : Nat) where
  size : Nat := 2^n
  elements : Fin (size * size) → Nat

/-- Represents the state of the board after rotations -/
def rotatedBoard (n : Nat) : Board n → Board n :=
  sorry

/-- The main diagonal of a board -/
def mainDiagonal (n : Nat) (b : Board n) : List Nat :=
  sorry

/-- The other main diagonal (bottom-left to top-right) of a board -/
def otherMainDiagonal (n : Nat) (b : Board n) : List Nat :=
  sorry

/-- Initial board setup -/
def initialBoard : Board 5 :=
  { elements := λ i => i.val + 1 }

theorem rotation_theorem :
  mainDiagonal 5 (rotatedBoard 5 initialBoard) =
    (otherMainDiagonal 5 initialBoard).reverse := by
  sorry

end NUMINAMATH_CALUDE_rotation_theorem_l963_96391


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l963_96348

/-- Given two parallel vectors a and b in R², if a = (4, 2) and b = (x, 3), then x = 6 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (4, 2)) 
  (h2 : b = (x, 3)) 
  (h3 : ∃ (k : ℝ), b = k • a) : 
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l963_96348


namespace NUMINAMATH_CALUDE_second_set_is_twenty_feet_l963_96354

/-- The length of the first set of wood in feet -/
def first_set_length : ℝ := 4

/-- The factor by which the second set is longer than the first set -/
def length_factor : ℝ := 5

/-- The length of the second set of wood in feet -/
def second_set_length : ℝ := first_set_length * length_factor

/-- Theorem stating that the second set of wood is 20 feet long -/
theorem second_set_is_twenty_feet : second_set_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_set_is_twenty_feet_l963_96354


namespace NUMINAMATH_CALUDE_total_crayons_l963_96398

/-- Given that each child has 8 crayons and there are 7 children, prove that the total number of crayons is 56. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
  (h1 : crayons_per_child = 8) (h2 : num_children = 7) : 
  crayons_per_child * num_children = 56 := by
  sorry


end NUMINAMATH_CALUDE_total_crayons_l963_96398


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_linear_l963_96335

theorem power_of_two_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_linear_l963_96335


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l963_96375

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 45 and 500 -/
def product : ℕ := 45 * 500

theorem product_trailing_zeros :
  trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l963_96375


namespace NUMINAMATH_CALUDE_number_division_problem_l963_96396

theorem number_division_problem (N : ℕ) (D : ℕ) (h1 : N % D = 0) (h2 : N / D = 2) (h3 : N % 4 = 2) : D = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l963_96396


namespace NUMINAMATH_CALUDE_integral_tan_ln_cos_l963_96326

theorem integral_tan_ln_cos (x : ℝ) :
  HasDerivAt (fun x => -1/2 * (Real.log (Real.cos x))^2) (Real.tan x * Real.log (Real.cos x)) x :=
by sorry

end NUMINAMATH_CALUDE_integral_tan_ln_cos_l963_96326


namespace NUMINAMATH_CALUDE_harvest_difference_l963_96399

theorem harvest_difference (apples peaches pears : ℕ) : 
  apples = 60 →
  peaches = 3 * apples →
  pears = apples / 2 →
  (apples + peaches) - pears = 210 := by
  sorry

end NUMINAMATH_CALUDE_harvest_difference_l963_96399


namespace NUMINAMATH_CALUDE_triangle_tangent_equality_l963_96368

theorem triangle_tangent_equality (A B : ℝ) (a b : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : A + B < π) :
  a * Real.tan A + b * Real.tan B = (a + b) * Real.tan ((A + B) / 2) ↔ a = b :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_equality_l963_96368


namespace NUMINAMATH_CALUDE_tshirt_price_l963_96302

/-- The original price of a t-shirt satisfies the given conditions -/
theorem tshirt_price (discount : ℚ) (quantity : ℕ) (revenue : ℚ) 
  (h1 : discount = 8)
  (h2 : quantity = 130)
  (h3 : revenue = 5590) :
  ∃ (original_price : ℚ), 
    quantity * (original_price - discount) = revenue ∧ 
    original_price = 51 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_price_l963_96302


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l963_96388

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (h_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y₁ ≠ 0 ∧ y₂ ≠ 0) 
  (h_inverse : ∃ c : ℝ, c ≠ 0 ∧ x₁ * y₁ = c ∧ x₂ * y₂ = c) 
  (h_ratio : x₁ / x₂ = 3 / 5) : 
  y₁ / y₂ = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l963_96388


namespace NUMINAMATH_CALUDE_john_earnings_increase_l963_96383

/-- Calculates the percentage increase in earnings -/
def percentage_increase (initial_earnings final_earnings : ℚ) : ℚ :=
  (final_earnings - initial_earnings) / initial_earnings * 100

/-- Represents John's weekly earnings from two jobs -/
structure WeeklyEarnings where
  job_a_initial : ℚ
  job_a_final : ℚ
  job_b_initial : ℚ
  job_b_final : ℚ

theorem john_earnings_increase (john : WeeklyEarnings)
  (h1 : john.job_a_initial = 60)
  (h2 : john.job_a_final = 78)
  (h3 : john.job_b_initial = 100)
  (h4 : john.job_b_final = 120) :
  percentage_increase (john.job_a_initial + john.job_b_initial)
                      (john.job_a_final + john.job_b_final) = 23.75 := by
  sorry

end NUMINAMATH_CALUDE_john_earnings_increase_l963_96383


namespace NUMINAMATH_CALUDE_acid_solution_concentration_l963_96323

theorem acid_solution_concentration 
  (x : ℝ) -- original concentration
  (h1 : 0.5 * x + 0.5 * 30 = 40) -- mixing equation
  : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_acid_solution_concentration_l963_96323


namespace NUMINAMATH_CALUDE_thirty_six_has_nine_divisors_l963_96395

/-- The number of positive divisors of 36 -/
def num_divisors_36 : ℕ := sorry

/-- 36 has exactly 9 positive divisors -/
theorem thirty_six_has_nine_divisors : num_divisors_36 = 9 := by sorry

end NUMINAMATH_CALUDE_thirty_six_has_nine_divisors_l963_96395


namespace NUMINAMATH_CALUDE_coin_combination_reaches_target_l963_96317

-- Define coin types and their values
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50
def dollar_coin_value : ℕ := 100

-- Define Grace's coin quantities
def pennies : ℕ := 25
def nickels : ℕ := 15
def dimes : ℕ := 20
def quarters : ℕ := 10
def half_dollars : ℕ := 5
def dollar_coins : ℕ := 3

-- Define the target amount
def target_amount : ℕ := 385

-- Theorem to prove
theorem coin_combination_reaches_target :
  ∃ (p d h : ℕ),
    p ≤ pennies ∧
    d ≤ dimes ∧
    h ≤ half_dollars ∧
    p * penny_value + d * dime_value + h * half_dollar_value = target_amount :=
by
  sorry

end NUMINAMATH_CALUDE_coin_combination_reaches_target_l963_96317


namespace NUMINAMATH_CALUDE_power_of_power_l963_96343

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l963_96343
