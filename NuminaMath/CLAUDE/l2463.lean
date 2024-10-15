import Mathlib

namespace NUMINAMATH_CALUDE_point_coordinates_in_third_quadrant_l2463_246335

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- The distance between a point and the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- Predicate for a point being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y < 0

theorem point_coordinates_in_third_quadrant 
  (M : Point) 
  (h1 : isInThirdQuadrant M) 
  (h2 : distToXAxis M = 1) 
  (h3 : distToYAxis M = 2) : 
  M.x = -2 ∧ M.y = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_in_third_quadrant_l2463_246335


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l2463_246331

/-- Proves that given the spending percentages and savings amount, the monthly income is 40000 --/
theorem monthly_income_calculation (income : ℝ) 
  (household_percent : income * (45 / 100) = income * 0.45)
  (clothes_percent : income * (25 / 100) = income * 0.25)
  (medicines_percent : income * (7.5 / 100) = income * 0.075)
  (savings : income * (1 - 0.45 - 0.25 - 0.075) = 9000) :
  income = 40000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l2463_246331


namespace NUMINAMATH_CALUDE_cook_selection_ways_l2463_246368

theorem cook_selection_ways (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 2) :
  Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_cook_selection_ways_l2463_246368


namespace NUMINAMATH_CALUDE_ladder_problem_l2463_246360

-- Define the ladder setup
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the theorem
theorem ladder_problem :
  -- Part 1: Horizontal distance
  ∃ (horizontal_distance : ℝ),
    horizontal_distance^2 + wall_height^2 = ladder_length^2 ∧
    horizontal_distance = 5 ∧
  -- Part 2: Height reached by 8-meter ladder
  ∃ (height_8m : ℝ),
    height_8m = (wall_height * 8) / ladder_length ∧
    height_8m = 96 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l2463_246360


namespace NUMINAMATH_CALUDE_factorization_equality_l2463_246350

theorem factorization_equality (x : ℝ) : -x^3 - 2*x^2 - x = -x*(x+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2463_246350


namespace NUMINAMATH_CALUDE_equation_solution_l2463_246372

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 1 →
  ((3 * x + 6) / (x^2 + 6 * x - 7) = (3 - x) / (x - 1)) ↔ (x = -5 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2463_246372


namespace NUMINAMATH_CALUDE_mean_temperature_is_87_5_l2463_246308

def temperatures : List ℝ := [82, 80, 83, 88, 90, 92, 90, 95]

theorem mean_temperature_is_87_5 :
  (temperatures.sum / temperatures.length : ℝ) = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_87_5_l2463_246308


namespace NUMINAMATH_CALUDE_select_two_from_four_l2463_246357

theorem select_two_from_four (n : ℕ) (k : ℕ) : n = 4 → k = 2 → Nat.choose n k = 6 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_four_l2463_246357


namespace NUMINAMATH_CALUDE_size_comparison_l2463_246389

-- Define a rectangular parallelepiped
structure RectParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

-- Define the size of a rectangular parallelepiped
def size (p : RectParallelepiped) : ℝ :=
  p.length + p.width + p.height

-- Define the "fits inside" relation
def fits_inside (p' p : RectParallelepiped) : Prop :=
  p'.length ≤ p.length ∧ p'.width ≤ p.width ∧ p'.height ≤ p.height

-- Theorem statement
theorem size_comparison (p p' : RectParallelepiped) (h : fits_inside p' p) :
  size p' ≤ size p := by
  sorry

end NUMINAMATH_CALUDE_size_comparison_l2463_246389


namespace NUMINAMATH_CALUDE_infinite_solutions_l2463_246395

theorem infinite_solutions (p : Nat) (hp : p.Prime) (hp_gt_7 : p > 7) :
  ∃ f : Nat → Nat,
    Function.Injective f ∧
    ∀ k : Nat, 
      (f k ≡ 1 [MOD 2016]) ∧ 
      (p ∣ (2^(f k) + f k)) :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l2463_246395


namespace NUMINAMATH_CALUDE_x_power_ln_ln_minus_ln_x_power_ln_l2463_246374

theorem x_power_ln_ln_minus_ln_x_power_ln (x : ℝ) (h : x > 1) :
  x^(Real.log (Real.log x)) - (Real.log x)^(Real.log x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_power_ln_ln_minus_ln_x_power_ln_l2463_246374


namespace NUMINAMATH_CALUDE_girls_in_class_l2463_246353

/-- The number of boys in the class -/
def num_boys : ℕ := 13

/-- The number of ways to select 1 girl and 2 boys -/
def num_selections : ℕ := 780

/-- The number of girls in the class -/
def num_girls : ℕ := 10

theorem girls_in_class : 
  num_girls * (num_boys.choose 2) = num_selections :=
sorry

end NUMINAMATH_CALUDE_girls_in_class_l2463_246353


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l2463_246311

-- Define the ⊗ operation
noncomputable def bowtie (a b : ℝ) : ℝ := a + 3 * Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem bowtie_equation_solution (h : ℝ) :
  bowtie 4 h = 10 → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l2463_246311


namespace NUMINAMATH_CALUDE_reading_time_difference_l2463_246344

/-- Proves that the difference in reading time between two readers is 144 minutes given their reading rates and book length. -/
theorem reading_time_difference 
  (xanthia_rate : ℝ) 
  (molly_rate : ℝ) 
  (book_pages : ℝ) 
  (h1 : xanthia_rate = 75) 
  (h2 : molly_rate = 45) 
  (h3 : book_pages = 270) : 
  (book_pages / molly_rate - book_pages / xanthia_rate) * 60 = 144 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l2463_246344


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l2463_246330

theorem right_triangle_ratio (x d : ℝ) (h1 : x > d) (h2 : d > 0) : 
  (x^2)^2 + (x^2 - d)^2 = (x^2 + d)^2 → x / d = 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l2463_246330


namespace NUMINAMATH_CALUDE_star_emilio_sum_difference_l2463_246380

def star_list : List Nat := List.range 50 |>.map (· + 1)

def emilio_transform (n : Nat) : Nat :=
  let s := toString n
  let s' := s.replace "2" "1" |>.replace "3" "2"
  s'.toNat!

def emilio_list : List Nat := star_list.map emilio_transform

theorem star_emilio_sum_difference : 
  star_list.sum - emilio_list.sum = 210 := by
  sorry

end NUMINAMATH_CALUDE_star_emilio_sum_difference_l2463_246380


namespace NUMINAMATH_CALUDE_nancy_hula_hoop_time_l2463_246356

theorem nancy_hula_hoop_time (morgan_time casey_time nancy_time : ℕ) : 
  morgan_time = 21 →
  morgan_time = 3 * casey_time →
  nancy_time = casey_time + 3 →
  nancy_time = 10 := by
sorry

end NUMINAMATH_CALUDE_nancy_hula_hoop_time_l2463_246356


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2463_246366

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, y < 3*y - 10 → y ≥ 6 ∧ 6 < 3*6 - 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2463_246366


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inv_sqrt_x_l2463_246386

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inv_sqrt_x_l2463_246386


namespace NUMINAMATH_CALUDE_charity_event_probability_l2463_246375

/-- The probability of selecting a boy for Saturday and a girl for Sunday
    from a group of 2 boys and 2 girls for a two-day event. -/
theorem charity_event_probability :
  let total_people : ℕ := 2 + 2  -- 2 boys + 2 girls
  let total_combinations : ℕ := total_people * (total_people - 1)
  let favorable_outcomes : ℕ := 2 * 2  -- 2 boys for Saturday * 2 girls for Sunday
  (favorable_outcomes : ℚ) / total_combinations = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_charity_event_probability_l2463_246375


namespace NUMINAMATH_CALUDE_semicircles_area_ratio_l2463_246318

theorem semicircles_area_ratio (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let semicircle1_area := (π * r^2) / 2
  let semicircle2_area := (π * (r/2)^2) / 2
  (semicircle1_area + semicircle2_area) / circle_area = 5/8 := by
sorry

end NUMINAMATH_CALUDE_semicircles_area_ratio_l2463_246318


namespace NUMINAMATH_CALUDE_existence_of_x0_l2463_246316

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem existence_of_x0 
  (hcont : ContinuousOn f (Set.Icc 0 1))
  (hdiff : DifferentiableOn ℝ f (Set.Ioo 0 1))
  (hf0 : f 0 = 1)
  (hf1 : f 1 = 0) :
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 ∧ 
    |deriv f x0| ≥ 2018 * (f x0)^2018 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x0_l2463_246316


namespace NUMINAMATH_CALUDE_pechkin_calculation_error_l2463_246369

/-- Represents Pechkin's journey --/
structure PechkinJourney where
  totalDistance : ℝ
  walkingSpeed : ℝ
  cyclingSpeed : ℝ
  walkingDistance : ℝ
  cyclingTime : ℝ
  totalTime : ℝ

/-- Conditions of Pechkin's journey --/
def journeyConditions (j : PechkinJourney) : Prop :=
  j.walkingSpeed = 5 ∧
  j.cyclingSpeed = 12 ∧
  j.walkingDistance = j.totalDistance / 2 ∧
  j.cyclingTime = j.totalTime / 3

/-- Theorem stating that Pechkin's calculations are inconsistent --/
theorem pechkin_calculation_error (j : PechkinJourney) 
  (h : journeyConditions j) : 
  j.cyclingSpeed * j.cyclingTime ≠ j.totalDistance - j.walkingDistance :=
sorry

end NUMINAMATH_CALUDE_pechkin_calculation_error_l2463_246369


namespace NUMINAMATH_CALUDE_expression_evaluation_l2463_246314

theorem expression_evaluation : 
  |((4:ℝ)^2 - 8*((3:ℝ)^2 - 12))^2| - |Real.sin (5*π/6) - Real.cos (11*π/3)| = 1600 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2463_246314


namespace NUMINAMATH_CALUDE_negative_inequality_l2463_246365

theorem negative_inequality (a b : ℝ) (h : a > b) : -b > -a := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l2463_246365


namespace NUMINAMATH_CALUDE_expression_not_33_l2463_246379

theorem expression_not_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_not_33_l2463_246379


namespace NUMINAMATH_CALUDE_goose_eggs_count_goose_eggs_solution_l2463_246339

theorem goose_eggs_count : ℕ → Prop :=
  fun total_eggs =>
    let hatched := (1 : ℚ) / 4 * total_eggs
    let survived_first_month := (4 : ℚ) / 5 * hatched
    let survived_six_months := (2 : ℚ) / 5 * survived_first_month
    let survived_first_year := (4 : ℚ) / 7 * survived_six_months
    survived_first_year = 120 ∧ total_eggs = 2625

/-- The number of goose eggs laid at the pond is 2625. -/
theorem goose_eggs_solution : goose_eggs_count 2625 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_goose_eggs_solution_l2463_246339


namespace NUMINAMATH_CALUDE_number_of_children_l2463_246364

theorem number_of_children : ∃ n : ℕ, 
  (∃ b : ℕ, b = 3 * n + 4) ∧ 
  (∃ b : ℕ, b = 4 * n - 3) ∧ 
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l2463_246364


namespace NUMINAMATH_CALUDE_parallelogram_area_is_41_l2463_246388

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : ℝ × ℝ) : ℝ :=
  |v.1 * w.2 - v.2 * w.1|

/-- Given vectors and their components -/
def v : ℝ × ℝ := (8, -5)
def w : ℝ × ℝ := (13, -3)

/-- Theorem: The area of the parallelogram formed by v and w is 41 -/
theorem parallelogram_area_is_41 : parallelogramArea v w = 41 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_is_41_l2463_246388


namespace NUMINAMATH_CALUDE_students_just_passed_l2463_246317

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) :
  total = 300 →
  first_div_percent = 29 / 100 →
  second_div_percent = 54 / 100 →
  (total : ℚ) * (1 - first_div_percent - second_div_percent) = 51 := by
sorry

end NUMINAMATH_CALUDE_students_just_passed_l2463_246317


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2463_246383

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) (h4 : w = 5) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (rectangle_length + w) = 22 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2463_246383


namespace NUMINAMATH_CALUDE_cos_equation_solution_l2463_246310

theorem cos_equation_solution (x : ℝ) : 
  (Real.cos (2 * x) - 2 * Real.cos (4 * x))^2 = 9 + (Real.cos (5 * x))^2 ↔ 
  ∃ k : ℤ, x = π / 2 + k * π :=
sorry

end NUMINAMATH_CALUDE_cos_equation_solution_l2463_246310


namespace NUMINAMATH_CALUDE_negation_equivalence_l2463_246371

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≥ 1 ∨ x > 2) ↔ (∀ x : ℝ, x < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2463_246371


namespace NUMINAMATH_CALUDE_x_lt_one_necessary_not_sufficient_l2463_246361

theorem x_lt_one_necessary_not_sufficient :
  ∀ x : ℝ,
  (∀ x, (1 / x > 1 → x < 1)) ∧
  (∃ x, x < 1 ∧ ¬(1 / x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_one_necessary_not_sufficient_l2463_246361


namespace NUMINAMATH_CALUDE_water_needed_proof_l2463_246396

/-- The ratio of water to lemon juice in the lemonade recipe -/
def water_ratio : ℚ := 8 / 10

/-- The number of gallons of lemonade to make -/
def gallons_to_make : ℚ := 2

/-- The number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- The number of liters in a quart -/
def liters_per_quart : ℚ := 95 / 100

/-- The amount of water needed in liters -/
def water_needed : ℚ := 
  water_ratio * gallons_to_make * quarts_per_gallon * liters_per_quart

theorem water_needed_proof : water_needed = 608 / 100 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_proof_l2463_246396


namespace NUMINAMATH_CALUDE_lucas_change_l2463_246343

def banana_cost : ℚ := 70 / 100
def orange_cost : ℚ := 80 / 100
def banana_quantity : ℕ := 5
def orange_quantity : ℕ := 2
def paid_amount : ℚ := 10

def total_cost : ℚ := banana_cost * banana_quantity + orange_cost * orange_quantity

theorem lucas_change :
  paid_amount - total_cost = 490 / 100 := by sorry

end NUMINAMATH_CALUDE_lucas_change_l2463_246343


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2463_246358

theorem fixed_point_on_line (a : ℝ) : (a + 1) * (-4) - (2 * a + 5) * (-2) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2463_246358


namespace NUMINAMATH_CALUDE_halloween_candy_l2463_246303

theorem halloween_candy (debby_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) :
  debby_candy = 32 →
  sister_candy = 42 →
  remaining_candy = 39 →
  debby_candy + sister_candy - remaining_candy = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_l2463_246303


namespace NUMINAMATH_CALUDE_train_speed_l2463_246362

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 3000) (h2 : time = 120) :
  length / time * (3600 / 1000) = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2463_246362


namespace NUMINAMATH_CALUDE_equation_proof_l2463_246378

theorem equation_proof : Real.sqrt (5 + Real.sqrt (3 + Real.sqrt 14)) = (2 + Real.sqrt 14) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2463_246378


namespace NUMINAMATH_CALUDE_min_rows_for_hockey_arena_l2463_246394

/-- Represents a seating arrangement in a hockey arena --/
structure ArenaSeating where
  total_students : ℕ
  seats_per_row : ℕ
  max_students_per_school : ℕ
  same_row_constraint : Bool

/-- Calculates the minimum number of rows required for the given seating arrangement --/
def min_rows_required (seating : ArenaSeating) : ℕ :=
  sorry

/-- Theorem stating the minimum number of rows required for the given problem --/
theorem min_rows_for_hockey_arena :
  let seating := ArenaSeating.mk 2016 168 40 true
  min_rows_required seating = 15 :=
sorry

end NUMINAMATH_CALUDE_min_rows_for_hockey_arena_l2463_246394


namespace NUMINAMATH_CALUDE_lcm_of_20_45_36_l2463_246376

theorem lcm_of_20_45_36 : Nat.lcm (Nat.lcm 20 45) 36 = 180 := by sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_36_l2463_246376


namespace NUMINAMATH_CALUDE_smaller_std_dev_more_stable_smaller_variance_more_stable_smaller_mean_not_necessarily_more_stable_l2463_246338

-- Define a dataset as a list of real numbers
def Dataset := List ℝ

-- Define standard deviation
def standardDeviation (data : Dataset) : ℝ :=
  sorry

-- Define variance
def variance (data : Dataset) : ℝ :=
  sorry

-- Define mean
def mean (data : Dataset) : ℝ :=
  sorry

-- Define a measure of concentration and stability
def isConcentratedAndStable (data : Dataset) : Prop :=
  sorry

-- Theorem stating that smaller standard deviation implies more concentrated and stable distribution
theorem smaller_std_dev_more_stable (data1 data2 : Dataset) :
  standardDeviation data1 < standardDeviation data2 →
  isConcentratedAndStable data1 → isConcentratedAndStable data2 :=
sorry

-- Theorem stating that smaller variance implies more concentrated and stable distribution
theorem smaller_variance_more_stable (data1 data2 : Dataset) :
  variance data1 < variance data2 →
  isConcentratedAndStable data1 → isConcentratedAndStable data2 :=
sorry

-- Theorem stating that smaller mean does not necessarily imply more concentrated and stable distribution
theorem smaller_mean_not_necessarily_more_stable :
  ∃ (data1 data2 : Dataset), mean data1 < mean data2 ∧
  isConcentratedAndStable data2 ∧ ¬isConcentratedAndStable data1 :=
sorry

end NUMINAMATH_CALUDE_smaller_std_dev_more_stable_smaller_variance_more_stable_smaller_mean_not_necessarily_more_stable_l2463_246338


namespace NUMINAMATH_CALUDE_rectangle_area_l2463_246393

/-- Given a rectangle with width 4 inches and perimeter 30 inches, prove its area is 44 square inches -/
theorem rectangle_area (width : ℝ) (perimeter : ℝ) : 
  width = 4 → perimeter = 30 → width * ((perimeter / 2) - width) = 44 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2463_246393


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_11_squared_l2463_246349

theorem sum_of_divisors_of_11_squared (a b c : ℕ+) : 
  a * b * c = 11^2 →
  a ∣ 11^2 ∧ b ∣ 11^2 ∧ c ∣ 11^2 →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 23 := by
sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_11_squared_l2463_246349


namespace NUMINAMATH_CALUDE_derivative_not_in_second_quadrant_l2463_246391

-- Define the quadratic function
def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the derivative of f
def f' (x : ℝ) (b : ℝ) : ℝ := 2*x + b

-- Theorem statement
theorem derivative_not_in_second_quadrant (b c : ℝ) :
  (∀ x, f x b c = f (-x + 4) b c) →  -- axis of symmetry is x = 2
  ∀ x y, f' x b = y → ¬(x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_derivative_not_in_second_quadrant_l2463_246391


namespace NUMINAMATH_CALUDE_inverse_proportion_l2463_246321

/-- Given that x is inversely proportional to y, prove that if x = 4 when y = 2, then x = -8/5 when y = -5 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 4 * 2 = k) :
  -5 * (-8/5 : ℝ) = k := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_l2463_246321


namespace NUMINAMATH_CALUDE_triangle_max_area_l2463_246346

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  (2 * a + b) * Real.cos C + c * Real.cos B = 0 →
  c = 6 →
  ∃ (S : ℝ), S ≤ 3 * Real.sqrt 3 ∧
    ∀ (S' : ℝ), S' = 1/2 * a * b * Real.sin C → S' ≤ S :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2463_246346


namespace NUMINAMATH_CALUDE_restaurant_students_l2463_246392

theorem restaurant_students (burger_orders : ℕ) (hotdog_orders : ℕ) : 
  burger_orders = 30 →
  burger_orders = 2 * hotdog_orders →
  burger_orders + hotdog_orders = 45 := by
sorry

end NUMINAMATH_CALUDE_restaurant_students_l2463_246392


namespace NUMINAMATH_CALUDE_z_value_when_x_is_4_l2463_246390

/-- The constant k in the inverse relationship -/
def k : ℚ := 392

/-- The inverse relationship between z and x -/
def inverse_relation (z x : ℚ) : Prop :=
  7 * z = k / (x^3)

theorem z_value_when_x_is_4 :
  ∀ z : ℚ, inverse_relation 7 2 → inverse_relation z 4 → z = 7/8 :=
by sorry

end NUMINAMATH_CALUDE_z_value_when_x_is_4_l2463_246390


namespace NUMINAMATH_CALUDE_equation_solution_l2463_246334

theorem equation_solution : 
  ∃! x : ℚ, (x - 30) / 3 = (5 - 3 * x) / 4 + 2 :=
by
  use 15
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2463_246334


namespace NUMINAMATH_CALUDE_scarlett_oil_addition_l2463_246328

/-- The amount of oil Scarlett needs to add to her measuring cup -/
def oil_to_add (current : ℚ) (desired : ℚ) : ℚ :=
  desired - current

/-- Theorem: Given the current amount of oil and the desired amount, 
    prove that Scarlett needs to add 0.67 cup of oil -/
theorem scarlett_oil_addition (current : ℚ) (desired : ℚ)
  (h1 : current = 17/100)
  (h2 : desired = 84/100) :
  oil_to_add current desired = 67/100 := by
  sorry

end NUMINAMATH_CALUDE_scarlett_oil_addition_l2463_246328


namespace NUMINAMATH_CALUDE_book_sale_price_l2463_246340

theorem book_sale_price (total_books : ℕ) (sold_books : ℕ) (unsold_books : ℕ) (total_amount : ℚ) : 
  sold_books = (2 : ℕ) * total_books / 3 →
  unsold_books = 30 →
  sold_books + unsold_books = total_books →
  total_amount = 255 →
  total_amount / sold_books = 17/4 := by
  sorry

#eval (17 : ℚ) / 4  -- This should evaluate to 4.25

end NUMINAMATH_CALUDE_book_sale_price_l2463_246340


namespace NUMINAMATH_CALUDE_complex_number_existence_l2463_246377

theorem complex_number_existence : ∃ (z : ℂ), 
  Complex.abs z = Real.sqrt 7 ∧ 
  z.re < 0 ∧ 
  z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_existence_l2463_246377


namespace NUMINAMATH_CALUDE_book_ratio_is_four_to_one_l2463_246370

/-- The number of books Zig wrote -/
def zig_books : ℕ := 60

/-- The total number of books Zig and Flo wrote together -/
def total_books : ℕ := 75

/-- The number of books Flo wrote -/
def flo_books : ℕ := total_books - zig_books

/-- The ratio of books written by Zig to books written by Flo -/
def book_ratio : ℚ := zig_books / flo_books

theorem book_ratio_is_four_to_one :
  book_ratio = 4 / 1 := by sorry

end NUMINAMATH_CALUDE_book_ratio_is_four_to_one_l2463_246370


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2463_246326

theorem two_numbers_problem (x y : ℝ) :
  (2 * (x + y) = x^2 - y^2) ∧ (2 * (x + y) = x * y / 4 - 56) →
  ((x = 26 ∧ y = 24) ∨ (x = -8 ∧ y = -10)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2463_246326


namespace NUMINAMATH_CALUDE_problem_statement_l2463_246309

theorem problem_statement (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2463_246309


namespace NUMINAMATH_CALUDE_product_sign_l2463_246363

theorem product_sign (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^4 - y^4 > x) (h2 : y^4 - x^4 > y) : x * y > 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sign_l2463_246363


namespace NUMINAMATH_CALUDE_find_other_number_l2463_246319

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 4 * b = 161) 
  (h2 : a = 17 ∨ b = 17) : (a = 31 ∧ b = 17) ∨ (a = 17 ∧ b = 31) :=
sorry

end NUMINAMATH_CALUDE_find_other_number_l2463_246319


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2463_246355

theorem opposite_of_negative_2023 : -((-2023) : ℤ) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2463_246355


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_l2463_246304

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem range_of_f_on_interval :
  ∃ (y : ℝ), y ∈ Set.Icc 0 (Real.exp (Real.pi / 2)) ↔
  ∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧ f x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_l2463_246304


namespace NUMINAMATH_CALUDE_four_integer_sum_l2463_246367

theorem four_integer_sum (a b c d : ℕ) : 
  a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 →
  a * b * c * d = 14400 →
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧
  Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1 →
  a + b + c + d = 98 := by
sorry

end NUMINAMATH_CALUDE_four_integer_sum_l2463_246367


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2463_246347

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (5 + n) = 7 → n = 44 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2463_246347


namespace NUMINAMATH_CALUDE_four_weavers_four_days_l2463_246384

/-- The number of mats woven by a group of weavers over a period of days. -/
def mats_woven (weavers : ℕ) (days : ℕ) : ℚ :=
  (25 : ℚ) * weavers * days / (10 * 10)

/-- Theorem stating that 4 mat-weavers will weave 4 mats in 4 days given the rate
    at which 10 mat-weavers can weave 25 mats in 10 days. -/
theorem four_weavers_four_days :
  mats_woven 4 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_weavers_four_days_l2463_246384


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2463_246381

theorem imaginary_part_of_z (z : ℂ) : z = (1 - I) / (1 + 3*I) → z.im = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2463_246381


namespace NUMINAMATH_CALUDE_divisible_by_forty_l2463_246300

theorem divisible_by_forty (n : ℕ) 
  (h1 : ∃ k : ℕ, 2 * n + 1 = k ^ 2) 
  (h2 : ∃ m : ℕ, 3 * n + 1 = m ^ 2) : 
  40 ∣ n := by
sorry

end NUMINAMATH_CALUDE_divisible_by_forty_l2463_246300


namespace NUMINAMATH_CALUDE_third_number_in_expression_l2463_246325

theorem third_number_in_expression (x : ℝ) : 
  (26.3 * 12 * x) / 3 + 125 = 2229 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_expression_l2463_246325


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l2463_246315

/-- Calculates the length of the second train given the speeds of two trains, 
    the time they take to clear each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (clear_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_ms := relative_speed * 1000 / 3600
  let total_distance := relative_speed_ms * clear_time
  total_distance - length1

/-- The length of the second train is approximately 165.12 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, abs (second_train_length 80 65 7.0752960452818945 120 - 165.12) < ε :=
by sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_solution_l2463_246315


namespace NUMINAMATH_CALUDE_sum_of_sequence_equals_63_over_19_l2463_246373

def A : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => 2 * A (n + 1) + A n

theorem sum_of_sequence_equals_63_over_19 :
  ∑' n, A n / 5^n = 63 / 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_sequence_equals_63_over_19_l2463_246373


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2463_246385

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 4*a^2 + a + 1) * (b^3 + 4*b^2 + b + 1) * (c^3 + 4*c^2 + c + 1) / (a * b * c) ≥ 512 ∧
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    (x^3 + 4*x^2 + x + 1) * (y^3 + 4*y^2 + y + 1) * (z^3 + 4*z^2 + z + 1) / (x * y * z) = 512) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2463_246385


namespace NUMINAMATH_CALUDE_parallelogram_solution_l2463_246301

-- Define the parallelogram EFGH
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ

-- Define the specific parallelogram from the problem
def specificParallelogram : Parallelogram where
  EF := 45
  FG := fun y ↦ 4 * y^2
  GH := fun x ↦ 3 * x + 6
  HE := 32

-- Theorem statement
theorem parallelogram_solution (p : Parallelogram) 
  (h1 : p = specificParallelogram) : 
  ∃ (x y : ℝ), p.GH x = p.EF ∧ p.FG y = p.HE ∧ x = 13 ∧ y = 2 * Real.sqrt 2 := by
  sorry

#check parallelogram_solution

end NUMINAMATH_CALUDE_parallelogram_solution_l2463_246301


namespace NUMINAMATH_CALUDE_total_pencils_count_l2463_246302

/-- The number of people in the group -/
def num_people : ℕ := 5

/-- The number of pencils each person has -/
def pencils_per_person : ℕ := 15

/-- The total number of pencils for the group -/
def total_pencils : ℕ := num_people * pencils_per_person

theorem total_pencils_count : total_pencils = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_count_l2463_246302


namespace NUMINAMATH_CALUDE_fraction_product_minus_one_l2463_246397

theorem fraction_product_minus_one : 
  (2/3) * (3/4) * (4/5) * (5/6) * (6/7) * (7/8) * (8/9) - 1 = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_minus_one_l2463_246397


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l2463_246345

theorem square_perimeter_sum (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 45) :
  4*x + 4*y = 4*Real.sqrt 65 + 8*Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l2463_246345


namespace NUMINAMATH_CALUDE_library_book_count_l2463_246348

theorem library_book_count (initial_books : ℕ) (loaned_books : ℕ) (return_rate : ℚ) : 
  initial_books = 75 →
  loaned_books = 40 →
  return_rate = 4/5 →
  initial_books - loaned_books + (return_rate * loaned_books).floor = 67 := by
sorry

end NUMINAMATH_CALUDE_library_book_count_l2463_246348


namespace NUMINAMATH_CALUDE_vector_subtraction_l2463_246327

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2463_246327


namespace NUMINAMATH_CALUDE_limit_special_function_l2463_246359

/-- The limit of (2 - e^(x^2))^(1 / (1 - cos(π * x))) as x approaches 0 is e^(-2 / π^2) -/
theorem limit_special_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |((2 - Real.exp (x^2))^(1 / (1 - Real.cos (π * x)))) - Real.exp (-2 / π^2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_special_function_l2463_246359


namespace NUMINAMATH_CALUDE_process_never_stops_l2463_246324

/-- Represents a large number as a list of digits -/
def LargeNumber := List Nat

/-- The initial number with 900 digits, all 1s -/
def initial_number : LargeNumber := List.replicate 900 1

/-- Extracts the last two digits of a LargeNumber -/
def last_two_digits (n : LargeNumber) : Nat :=
  match n.reverse with
  | d1 :: d2 :: _ => d1 + 10 * d2
  | _ => 0

/-- Applies the transformation rule to a LargeNumber -/
def transform (n : LargeNumber) : Nat :=
  let a := n.foldl (fun acc d => acc * 10 + d) 0 / 100
  let b := last_two_digits n
  2 * a + 8 * b

/-- Predicate to check if a number is less than 100 -/
def is_less_than_100 (n : Nat) : Prop := n < 100

/-- Main theorem: The process will never stop -/
theorem process_never_stops :
  ∀ n : Nat, ∃ m : Nat, m > n ∧ ¬(is_less_than_100 (transform (List.replicate m 1))) :=
  sorry


end NUMINAMATH_CALUDE_process_never_stops_l2463_246324


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l2463_246342

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = π) :
  -2 < Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ∧
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≤ 3 * Real.sqrt 3 / 2 ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 3 * Real.sqrt 3 / 2 ↔
   A = 7 * π / 9 ∧ B = π / 9 ∧ C = π / 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l2463_246342


namespace NUMINAMATH_CALUDE_certain_number_proof_l2463_246306

theorem certain_number_proof (z : ℤ) (h1 : z % 9 = 6) 
  (h2 : ∃ x : ℤ, ∃ m : ℤ, (z + x) / 9 = m) : 
  ∃ x : ℤ, x = 3 ∧ ∃ m : ℤ, (z + x) / 9 = m :=
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2463_246306


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2463_246399

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) : 
  let f1 := fun x : ℝ => x^2 + a*x + b
  let f2 := fun x : ℝ => x^2 + c*x + d
  let vertex1 := (-a/2, f1 (-a/2))
  let vertex2 := (-c/2, f2 (-c/2))
  (a = -4 ∧ b = 5 ∧ c = 6 ∧ d = 13) →
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2463_246399


namespace NUMINAMATH_CALUDE_clara_owes_mandy_l2463_246322

/-- The amount Clara owes Mandy for cleaning rooms -/
def amount_owed (rate : ℚ) (rooms : ℚ) (discount_threshold : ℚ) (discount_rate : ℚ) : ℚ :=
  let base_amount := rate * rooms
  if rooms > discount_threshold then
    base_amount * (1 - discount_rate)
  else
    base_amount

/-- Theorem stating the amount Clara owes Mandy -/
theorem clara_owes_mandy :
  let rate : ℚ := 15 / 4
  let rooms : ℚ := 12 / 5
  let discount_threshold : ℚ := 2
  let discount_rate : ℚ := 1 / 10
  amount_owed rate rooms discount_threshold discount_rate = 81 / 10 := by
  sorry

end NUMINAMATH_CALUDE_clara_owes_mandy_l2463_246322


namespace NUMINAMATH_CALUDE_card_distribution_proof_l2463_246333

/-- Represents the number of cards each player has -/
structure CardDistribution :=
  (alfred : ℕ)
  (bruno : ℕ)
  (christophe : ℕ)
  (damien : ℕ)

/-- The total number of cards in the deck -/
def totalCards : ℕ := 32

/-- Redistribution function for Alfred -/
def redistributeAlfred (d : CardDistribution) : CardDistribution :=
  { alfred := d.alfred - d.alfred / 2,
    bruno := d.bruno + d.alfred / 4,
    christophe := d.christophe + d.alfred / 4,
    damien := d.damien }

/-- Redistribution function for Bruno -/
def redistributeBruno (d : CardDistribution) : CardDistribution :=
  { alfred := d.alfred + d.bruno / 4,
    bruno := d.bruno - d.bruno / 2,
    christophe := d.christophe + d.bruno / 4,
    damien := d.damien }

/-- Redistribution function for Christophe -/
def redistributeChristophe (d : CardDistribution) : CardDistribution :=
  { alfred := d.alfred + d.christophe / 4,
    bruno := d.bruno + d.christophe / 4,
    christophe := d.christophe - d.christophe / 2,
    damien := d.damien }

/-- The initial distribution of cards -/
def initialDistribution : CardDistribution :=
  { alfred := 4, bruno := 7, christophe := 13, damien := 8 }

theorem card_distribution_proof :
  let finalDist := redistributeChristophe (redistributeBruno (redistributeAlfred initialDistribution))
  (finalDist.alfred = finalDist.bruno) ∧
  (finalDist.bruno = finalDist.christophe) ∧
  (finalDist.christophe = finalDist.damien) ∧
  (finalDist.alfred + finalDist.bruno + finalDist.christophe + finalDist.damien = totalCards) :=
by sorry

end NUMINAMATH_CALUDE_card_distribution_proof_l2463_246333


namespace NUMINAMATH_CALUDE_peaches_in_basket_c_l2463_246337

theorem peaches_in_basket_c (total_baskets : ℕ) (avg_fruits : ℕ) 
  (fruits_a : ℕ) (fruits_b : ℕ) (fruits_d : ℕ) (fruits_e : ℕ) :
  total_baskets = 5 →
  avg_fruits = 25 →
  fruits_a = 15 →
  fruits_b = 30 →
  fruits_d = 25 →
  fruits_e = 35 →
  (total_baskets * avg_fruits) - (fruits_a + fruits_b + fruits_d + fruits_e) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_peaches_in_basket_c_l2463_246337


namespace NUMINAMATH_CALUDE_minimum_value_and_range_of_a_l2463_246332

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1/2 * x^2 - x

theorem minimum_value_and_range_of_a :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = 2) →
  (∃ x_min : ℝ, x_min > 0 ∧ ∀ x : ℝ, x > 0 → f a x ≥ f a x_min) →
  (f a 2 = -2 * Real.log 2) ∧
  (∀ x : ℝ, x > Real.exp 1 → f a x - a * x > 0) →
  a ≤ (Real.exp 2 - 2 * Real.exp 1) / (2 * (Real.exp 1 - 1)) :=
sorry

end NUMINAMATH_CALUDE_minimum_value_and_range_of_a_l2463_246332


namespace NUMINAMATH_CALUDE_empty_fixed_implies_empty_stable_l2463_246307

/-- Quadratic function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Set of fixed points -/
def A (a b c : ℝ) : Set ℝ := {x | f a b c x = x}

/-- Set of stable points -/
def B (a b c : ℝ) : Set ℝ := {x | f a b c (f a b c x) = x}

/-- Theorem: If A is empty, then B is empty for quadratic functions -/
theorem empty_fixed_implies_empty_stable (a b c : ℝ) (ha : a ≠ 0) :
  A a b c = ∅ → B a b c = ∅ := by sorry

end NUMINAMATH_CALUDE_empty_fixed_implies_empty_stable_l2463_246307


namespace NUMINAMATH_CALUDE_max_segments_theorem_l2463_246382

/-- A configuration of points on a plane. -/
structure PointConfiguration where
  n : ℕ  -- number of points
  m : ℕ  -- number of points on the convex hull
  no_collinear_triple : Bool  -- no three points are collinear
  m_le_n : m ≤ n  -- number of points on convex hull cannot exceed total points

/-- The maximum number of non-intersecting line segments for a given point configuration. -/
def max_segments (config : PointConfiguration) : ℕ :=
  3 * config.n - config.m - 3

/-- Theorem stating the maximum number of non-intersecting line segments. -/
theorem max_segments_theorem (config : PointConfiguration) :
  config.no_collinear_triple →
  max_segments config = 3 * config.n - config.m - 3 :=
sorry

end NUMINAMATH_CALUDE_max_segments_theorem_l2463_246382


namespace NUMINAMATH_CALUDE_porter_buns_problem_l2463_246313

/-- Calculates the maximum number of buns that can be transported to a construction site. -/
def max_buns_transported (total_buns : ℕ) (buns_per_trip : ℕ) (buns_eaten_per_way : ℕ) : ℕ :=
  let num_trips : ℕ := total_buns / buns_per_trip
  let buns_eaten : ℕ := 2 * (num_trips - 1) * buns_eaten_per_way + buns_eaten_per_way
  total_buns - buns_eaten

/-- Theorem stating that given 200 total buns, 40 buns carried per trip, and 1 bun eaten per one-way trip,
    the maximum number of buns that can be transported to the construction site is 191. -/
theorem porter_buns_problem :
  max_buns_transported 200 40 1 = 191 := by
  sorry

end NUMINAMATH_CALUDE_porter_buns_problem_l2463_246313


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l2463_246336

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 2 = 3*x

-- Define the standard form of the quadratic equation
def standard_form (a b c x : ℝ) : Prop := a*x^2 + b*x + c = 0

-- Theorem statement
theorem quadratic_coefficients :
  ∃ (c : ℝ), ∀ (x : ℝ),
    quadratic_equation x ↔ standard_form 1 (-3) c x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l2463_246336


namespace NUMINAMATH_CALUDE_parallelogram_base_l2463_246398

/-- The base of a parallelogram given its area and height -/
theorem parallelogram_base (area height base : ℝ) 
  (h_area : area = 384) 
  (h_height : height = 16) 
  (h_formula : area = base * height) : 
  base = 24 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_l2463_246398


namespace NUMINAMATH_CALUDE_f_range_l2463_246312

def f (x : ℝ) : ℝ := -x^2

theorem f_range :
  ∀ y ∈ Set.range (f ∘ (Set.Icc (-3) 1).restrict f), -9 ≤ y ∧ y ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_f_range_l2463_246312


namespace NUMINAMATH_CALUDE_bananas_bought_l2463_246305

theorem bananas_bought (initial : ℕ) (eaten : ℕ) (remaining : ℕ) : 
  eaten = 1 → remaining = 11 → initial = eaten + remaining := by sorry

end NUMINAMATH_CALUDE_bananas_bought_l2463_246305


namespace NUMINAMATH_CALUDE_fence_length_proof_l2463_246354

theorem fence_length_proof (darren_length : ℝ) (doug_length : ℝ) : 
  darren_length = 1.2 * doug_length →
  darren_length = 360 →
  darren_length + doug_length = 660 :=
by
  sorry

end NUMINAMATH_CALUDE_fence_length_proof_l2463_246354


namespace NUMINAMATH_CALUDE_probability_theorem_l2463_246341

/-- Represents the number of students in each language class and their combinations --/
structure LanguageEnrollment where
  total : ℕ
  french : ℕ
  spanish : ℕ
  german : ℕ
  french_spanish : ℕ
  french_german : ℕ
  spanish_german : ℕ
  all_three : ℕ

/-- Calculates the probability of selecting at least one student from each language class --/
def probability_all_languages (e : LanguageEnrollment) : ℚ :=
  let total_combinations := (e.total.choose 3)
  let favorable_outcomes := 
    (e.french - e.french_spanish - e.french_german + e.all_three) * 
    (e.spanish - e.french_spanish - e.spanish_german + e.all_three) * 
    (e.german - e.french_german - e.spanish_german + e.all_three) +
    e.french_spanish * (e.german - e.french_german - e.spanish_german + e.all_three) +
    e.french_german * (e.spanish - e.french_spanish - e.spanish_german + e.all_three) +
    e.spanish_german * (e.french - e.french_spanish - e.french_german + e.all_three)
  favorable_outcomes / total_combinations

/-- The main theorem to prove --/
theorem probability_theorem (e : LanguageEnrollment) 
  (h1 : e.total = 40)
  (h2 : e.french = 26)
  (h3 : e.spanish = 29)
  (h4 : e.german = 12)
  (h5 : e.french_spanish = 9)
  (h6 : e.french_german = 9)
  (h7 : e.spanish_german = 9)
  (h8 : e.all_three = 2) :
  probability_all_languages e = 76 / 4940 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2463_246341


namespace NUMINAMATH_CALUDE_detergent_per_pound_l2463_246329

/-- Given that Mrs. Hilt used 18 ounces of detergent to wash 9 pounds of clothes,
    prove that she uses 2 ounces of detergent per pound of clothes. -/
theorem detergent_per_pound (total_detergent : ℝ) (total_clothes : ℝ) 
  (h1 : total_detergent = 18) 
  (h2 : total_clothes = 9) : 
  total_detergent / total_clothes = 2 := by
sorry

end NUMINAMATH_CALUDE_detergent_per_pound_l2463_246329


namespace NUMINAMATH_CALUDE_soda_cost_lucille_soda_cost_l2463_246351

/-- The cost of Lucille's soda given her weeding earnings and remaining money -/
theorem soda_cost (cents_per_weed : ℕ) (flower_bed_weeds : ℕ) (vegetable_patch_weeds : ℕ) 
  (grass_weeds : ℕ) (remaining_cents : ℕ) : ℕ :=
  let total_weeds := flower_bed_weeds + vegetable_patch_weeds + grass_weeds / 2
  let total_earnings := total_weeds * cents_per_weed
  total_earnings - remaining_cents

/-- Proof that Lucille's soda cost 99 cents -/
theorem lucille_soda_cost : soda_cost 6 11 14 32 147 = 99 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_lucille_soda_cost_l2463_246351


namespace NUMINAMATH_CALUDE_parallelepiped_vector_sum_l2463_246352

/-- In a parallelepiped ABCD-A₁B₁C₁D₁, if AC₁ = x⋅AB + 2y⋅BC + 3z⋅CC₁, then x + y + z = 11/6 -/
theorem parallelepiped_vector_sum (ABCD_A₁B₁C₁D₁ : Set (EuclideanSpace ℝ (Fin 3)))
  (AB BC CC₁ AC₁ : EuclideanSpace ℝ (Fin 3)) (x y z : ℝ) :
  AC₁ = x • AB + (2 * y) • BC + (3 * z) • CC₁ →
  AC₁ = AB + BC + CC₁ →
  x + y + z = 11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_vector_sum_l2463_246352


namespace NUMINAMATH_CALUDE_unique_face_reconstruction_l2463_246320

/-- Represents the numbers on the faces of a cube -/
structure CubeFaces where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Represents the sums on the edges of a cube -/
structure CubeEdges where
  ab : ℝ
  ac : ℝ
  ad : ℝ
  ae : ℝ
  bc : ℝ
  bf : ℝ
  cf : ℝ
  df : ℝ
  de : ℝ
  ef : ℝ
  bd : ℝ
  ce : ℝ

/-- Function to calculate edge sums from face numbers -/
def edgeSumsFromFaces (faces : CubeFaces) : CubeEdges :=
  { ab := faces.a + faces.b
  , ac := faces.a + faces.c
  , ad := faces.a + faces.d
  , ae := faces.a + faces.e
  , bc := faces.b + faces.c
  , bf := faces.b + faces.f
  , cf := faces.c + faces.f
  , df := faces.d + faces.f
  , de := faces.d + faces.e
  , ef := faces.e + faces.f
  , bd := faces.b + faces.d
  , ce := faces.c + faces.e }

/-- Theorem stating that face numbers can be uniquely reconstructed from edge sums -/
theorem unique_face_reconstruction (edges : CubeEdges) : 
  ∃! faces : CubeFaces, edgeSumsFromFaces faces = edges := by
  sorry


end NUMINAMATH_CALUDE_unique_face_reconstruction_l2463_246320


namespace NUMINAMATH_CALUDE_negation_of_square_nonnegative_l2463_246323

theorem negation_of_square_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_square_nonnegative_l2463_246323


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2463_246387

theorem average_speed_calculation (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) 
  (second_part_speed : ℝ) (h1 : total_distance = 400) (h2 : first_part_distance = 100) 
  (h3 : first_part_speed = 20) (h4 : second_part_speed = 15) : 
  (total_distance / (first_part_distance / first_part_speed + 
  (total_distance - first_part_distance) / second_part_speed)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2463_246387
