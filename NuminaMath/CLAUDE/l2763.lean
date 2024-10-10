import Mathlib

namespace product_expansion_l2763_276310

theorem product_expansion (x : ℝ) (h : x ≠ 0) :
  (3 / 7) * ((7 / x^3) - 14 * x^4) = 3 / x^3 - 6 * x^4 := by
  sorry

end product_expansion_l2763_276310


namespace integers_between_cubes_l2763_276342

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(10.6 : ℝ)^3⌋ - ⌈(10.5 : ℝ)^3⌉ + 1) ∧ n = 33 := by
  sorry

end integers_between_cubes_l2763_276342


namespace sector_area_l2763_276345

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 3) (h2 : θ = 120 * π / 180) :
  (θ / (2 * π)) * π * r^2 = 3 * π := by
  sorry

end sector_area_l2763_276345


namespace series_general_term_l2763_276355

theorem series_general_term (n : ℕ) (a : ℕ → ℚ) :
  (∀ k, a k = 1 / (k^2 : ℚ)) →
  a n = 1 / (n^2 : ℚ) := by sorry

end series_general_term_l2763_276355


namespace perimeter_of_equilateral_triangle_with_base_8_l2763_276393

-- Define an equilateral triangle
structure EquilateralTriangle where
  base : ℝ
  is_positive : base > 0

-- Define the perimeter of an equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.base

-- Theorem statement
theorem perimeter_of_equilateral_triangle_with_base_8 :
  ∀ t : EquilateralTriangle, t.base = 8 → perimeter t = 24 := by
  sorry

end perimeter_of_equilateral_triangle_with_base_8_l2763_276393


namespace choose_seven_two_l2763_276362

theorem choose_seven_two : Nat.choose 7 2 = 21 := by sorry

end choose_seven_two_l2763_276362


namespace triangle_area_l2763_276346

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) : 
  (1/2) * a * b = 180 := by
  sorry

end triangle_area_l2763_276346


namespace range_of_a_l2763_276352

theorem range_of_a (x y a : ℝ) : 
  x - y = 2 → 
  x + y = a → 
  x > -1 → 
  y < 0 → 
  -4 < a ∧ a < 2 :=
by sorry

end range_of_a_l2763_276352


namespace A_power_50_l2763_276380

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 2; -8, -5]

theorem A_power_50 : A ^ 50 = !![(-199 : ℤ), -100; 400, 201] := by sorry

end A_power_50_l2763_276380


namespace complex_in_second_quadrant_l2763_276322

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_in_second_quadrant :
  let z : ℂ := -1 + 3*Complex.I
  second_quadrant z :=
by
  sorry

end complex_in_second_quadrant_l2763_276322


namespace age_sum_proof_l2763_276347

theorem age_sum_proof (patrick michael monica : ℕ) : 
  3 * michael = 5 * patrick →
  3 * monica = 5 * michael →
  monica - patrick = 80 →
  patrick + michael + monica = 245 := by
sorry

end age_sum_proof_l2763_276347


namespace consecutive_odd_numbers_sum_l2763_276351

theorem consecutive_odd_numbers_sum (a b c d e : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1 ∧ 
            b = 2*k + 3 ∧ 
            c = 2*k + 5 ∧ 
            d = 2*k + 7 ∧ 
            e = 2*k + 9) →
  a + b + c + d + e = 130 →
  c = 26 := by
sorry

end consecutive_odd_numbers_sum_l2763_276351


namespace smallest_representable_difference_l2763_276383

theorem smallest_representable_difference : ∃ (m n : ℕ+), 
  14 = 19^(n : ℕ) - 5^(m : ℕ) ∧ 
  ∀ (k : ℕ+) (m' n' : ℕ+), k < 14 → k ≠ 19^(n' : ℕ) - 5^(m' : ℕ) :=
by sorry

end smallest_representable_difference_l2763_276383


namespace certain_number_problem_l2763_276360

theorem certain_number_problem (x : ℝ) : ((2 * (x + 5)) / 5) - 5 = 22 → x = 62.5 := by
  sorry

end certain_number_problem_l2763_276360


namespace function_value_proof_l2763_276324

theorem function_value_proof (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f ((1/2) * x - 1) = 2 * x - 5) →
  f a = 6 →
  a = 7/4 := by
sorry

end function_value_proof_l2763_276324


namespace quadratic_real_roots_l2763_276302

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end quadratic_real_roots_l2763_276302


namespace biology_enrollment_percentage_l2763_276348

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) 
  (h1 : total_students = 880) (h2 : not_enrolled = 616) :
  (((total_students - not_enrolled : ℝ) / total_students) * 100 : ℝ) = 30 := by
  sorry

end biology_enrollment_percentage_l2763_276348


namespace rhombus_side_length_l2763_276381

/-- Given a rhombus with diagonal sum L and area S, its side length is (√(L² - 4S)) / 2 -/
theorem rhombus_side_length (L S : ℝ) (h1 : L > 0) (h2 : S > 0) (h3 : L^2 ≥ 4*S) :
  ∃ (side_length : ℝ), side_length = (Real.sqrt (L^2 - 4*S)) / 2 :=
by sorry

end rhombus_side_length_l2763_276381


namespace x_range_for_sqrt_equality_l2763_276320

theorem x_range_for_sqrt_equality (x : ℝ) : 
  (Real.sqrt (x / (1 - x)) = Real.sqrt x / Real.sqrt (1 - x)) → 
  (0 ≤ x ∧ x < 1) :=
by sorry

end x_range_for_sqrt_equality_l2763_276320


namespace arithmetic_calculation_l2763_276305

theorem arithmetic_calculation : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end arithmetic_calculation_l2763_276305


namespace haydens_earnings_l2763_276375

/-- Represents Hayden's work day --/
structure WorkDay where
  totalHours : ℕ
  peakHours : ℕ
  totalRides : ℕ
  longDistanceRides : ℕ
  shortDistanceGallons : ℕ
  longDistanceGallons : ℕ
  maintenanceCost : ℕ
  tollCount : ℕ
  parkingExpense : ℕ
  positiveReviews : ℕ
  excellentReviews : ℕ

/-- Calculate Hayden's earnings for a given work day --/
def calculateEarnings (day : WorkDay) : ℚ :=
  sorry

/-- Theorem stating that Hayden's earnings for the given day equal $411.75 --/
theorem haydens_earnings : 
  let day : WorkDay := {
    totalHours := 12,
    peakHours := 3,
    totalRides := 6,
    longDistanceRides := 3,
    shortDistanceGallons := 10,
    longDistanceGallons := 20,
    maintenanceCost := 30,
    tollCount := 2,
    parkingExpense := 10,
    positiveReviews := 2,
    excellentReviews := 1
  }
  calculateEarnings day = 411.75 := by sorry

end haydens_earnings_l2763_276375


namespace no_function_satisfies_composite_condition_l2763_276350

theorem no_function_satisfies_composite_condition :
  ∀ f : ℝ → ℝ, ∃ x : ℝ, f (f x) ≠ x^2 - 1996 := by
  sorry

end no_function_satisfies_composite_condition_l2763_276350


namespace correct_calculation_l2763_276371

theorem correct_calculation : 3 * Real.sqrt 2 - (Real.sqrt 2) / 2 = (5 / 2) * Real.sqrt 2 := by
  sorry

end correct_calculation_l2763_276371


namespace sum_of_divisors_theorem_l2763_276370

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j * 5^k is 3780, then i + j + k = 8 -/
theorem sum_of_divisors_theorem (i j k : ℕ) : 
  sum_of_divisors (2^i * 3^j * 5^k) = 3780 → i + j + k = 8 := by
  sorry

end sum_of_divisors_theorem_l2763_276370


namespace henry_collection_cost_l2763_276341

/-- The amount of money Henry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem stating that Henry needs $30 to finish his collection -/
theorem henry_collection_cost :
  let current := 3  -- Henry's current number of action figures
  let total := 8    -- Total number of action figures needed for a complete collection
  let cost := 6     -- Cost of each action figure in dollars
  money_needed current total cost = 30 := by
sorry

end henry_collection_cost_l2763_276341


namespace circle_diameter_theorem_l2763_276353

/-- A circle with two intersecting perpendicular chords -/
structure CircleWithChords where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The lengths of the segments of the first chord -/
  chord1_seg1 : ℝ
  chord1_seg2 : ℝ
  /-- The lengths of the segments of the second chord -/
  chord2_seg1 : ℝ
  chord2_seg2 : ℝ
  /-- The chords are perpendicular -/
  chords_perpendicular : True
  /-- The product of segments of each chord equals the square of the radius -/
  chord1_property : chord1_seg1 * chord1_seg2 = radius ^ 2
  chord2_property : chord2_seg1 * chord2_seg2 = radius ^ 2

/-- The theorem to be proved -/
theorem circle_diameter_theorem (c : CircleWithChords) 
  (h1 : c.chord1_seg1 = 3 ∧ c.chord1_seg2 = 4) 
  (h2 : c.chord2_seg1 = 6 ∧ c.chord2_seg2 = 2) : 
  2 * c.radius = 4 * Real.sqrt 14 := by
  sorry

end circle_diameter_theorem_l2763_276353


namespace camdens_dogs_legs_l2763_276334

def number_of_dogs (name : String) : ℕ :=
  match name with
  | "Justin" => 14
  | "Rico" => 24
  | "Camden" => 18
  | _ => 0

theorem camdens_dogs_legs : 
  (∀ (name : String), number_of_dogs name ≥ 0) →
  number_of_dogs "Rico" = number_of_dogs "Justin" + 10 →
  number_of_dogs "Camden" = (3 * number_of_dogs "Rico") / 4 →
  number_of_dogs "Camden" * 4 = 72 :=
by
  sorry

end camdens_dogs_legs_l2763_276334


namespace base7_product_sum_theorem_l2763_276335

/-- Converts a base-7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := sorry

theorem base7_product_sum_theorem :
  let a := 35
  let b := 21
  let product := multiplyBase7 a b
  let digitSum := sumDigitsBase7 product
  multiplyBase7 digitSum 3 = 63
  := by sorry

end base7_product_sum_theorem_l2763_276335


namespace unique_solution_l2763_276396

-- Define the machine's rule
def machineRule (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 5 * n + 2

-- Define a function that applies the rule n times
def applyNTimes (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => applyNTimes n (machineRule x)

-- Theorem statement
theorem unique_solution : ∀ n : ℕ, n > 0 → (applyNTimes 6 n = 4 ↔ n = 256) :=
sorry

end unique_solution_l2763_276396


namespace floor_division_equality_l2763_276384

theorem floor_division_equality (α : ℝ) (d : ℕ) (h_α : α > 0) :
  ⌊α / d⌋ = ⌊⌊α⌋ / d⌋ := by
  sorry

end floor_division_equality_l2763_276384


namespace gcd_324_243_135_l2763_276354

theorem gcd_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by sorry

end gcd_324_243_135_l2763_276354


namespace average_of_quadratic_solutions_l2763_276398

theorem average_of_quadratic_solutions (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 3 * a * x + b
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (x₁ + x₂) / 2 = -3 / 2 := by
sorry

end average_of_quadratic_solutions_l2763_276398


namespace cube_painting_cost_l2763_276316

/-- The cost of painting a cube's surface area given its volume and paint cost per area unit -/
theorem cube_painting_cost (volume : ℝ) (cost_per_area : ℝ) : 
  volume = 9261 → 
  cost_per_area = 13 / 100 →
  6 * (volume ^ (1/3))^2 * cost_per_area = 344.98 := by
sorry

end cube_painting_cost_l2763_276316


namespace banana_permutations_count_l2763_276315

/-- The number of unique permutations of a multiset with 6 elements,
    where one element appears 3 times, another appears 2 times,
    and the third appears once. -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)

/-- Theorem stating that the number of unique permutations of "BANANA" is 60. -/
theorem banana_permutations_count : banana_permutations = 60 := by
  sorry

end banana_permutations_count_l2763_276315


namespace work_completion_time_l2763_276337

-- Define the work rates and time
def work_rate_B : ℚ := 1 / 18
def work_rate_A : ℚ := 2 * work_rate_B
def time_together : ℚ := 6

-- State the theorem
theorem work_completion_time :
  (work_rate_A = 2 * work_rate_B) →
  (work_rate_B = 1 / 18) →
  (time_together * (work_rate_A + work_rate_B) = 1) :=
by
  sorry

end work_completion_time_l2763_276337


namespace max_square_plots_l2763_276328

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℕ
  width : ℕ

/-- Represents the available fencing and field dimensions -/
structure FencingProblem where
  field : FieldDimensions
  available_fencing : ℕ

/-- Calculates the number of square plots given the side length -/
def num_plots (f : FieldDimensions) (side : ℕ) : ℕ :=
  (f.length / side) * (f.width / side)

/-- Calculates the required internal fencing given the side length -/
def required_fencing (f : FieldDimensions) (side : ℕ) : ℕ :=
  f.length * ((f.width / side) - 1) + f.width * ((f.length / side) - 1)

/-- Theorem: The maximum number of square plots is 18 -/
theorem max_square_plots (p : FencingProblem) 
  (h1 : p.field.length = 30)
  (h2 : p.field.width = 60)
  (h3 : p.available_fencing = 2500) :
  ∃ (side : ℕ), 
    side > 0 ∧ 
    side ∣ p.field.length ∧ 
    side ∣ p.field.width ∧
    required_fencing p.field side ≤ p.available_fencing ∧
    num_plots p.field side = 18 ∧
    ∀ (other_side : ℕ), other_side > side → 
      ¬(other_side ∣ p.field.length ∧ 
        other_side ∣ p.field.width ∧
        required_fencing p.field other_side ≤ p.available_fencing) :=
  sorry

end max_square_plots_l2763_276328


namespace rectangle_perimeter_l2763_276365

/-- Given a rectangle with width 10 m and area 150 square meters, 
    if its length is increased such that the new area is 1 (1/3) times the original area, 
    then the new perimeter of the rectangle is 60 meters. -/
theorem rectangle_perimeter (width : ℝ) (original_area : ℝ) (new_area : ℝ) : 
  width = 10 →
  original_area = 150 →
  new_area = original_area * (4/3) →
  let original_length := original_area / width
  let new_length := new_area / width
  2 * (new_length + width) = 60 := by
  sorry

end rectangle_perimeter_l2763_276365


namespace cookie_jar_problem_l2763_276379

theorem cookie_jar_problem (initial_cookies : ℕ) 
  (cookies_removed : ℕ) (cookies_added : ℕ) : 
  initial_cookies = 7 → 
  cookies_removed = 1 → 
  cookies_added = 5 → 
  initial_cookies - cookies_removed = (initial_cookies + cookies_added) / 2 := by
  sorry

end cookie_jar_problem_l2763_276379


namespace largest_angle_in_hexagon_l2763_276399

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Sum of angles in a hexagon is 720°
  A + B + C + D + E + F = 720 ∧
  -- Given conditions
  A = 90 ∧
  B = 120 ∧
  C = 95 ∧
  D = E ∧
  F = 2 * D + 25

-- Theorem statement
theorem largest_angle_in_hexagon (A B C D E F : ℝ) 
  (h : Hexagon A B C D E F) : 
  max A (max B (max C (max D (max E F)))) = 220 := by
  sorry


end largest_angle_in_hexagon_l2763_276399


namespace xyz_product_l2763_276343

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 100 := by
sorry

end xyz_product_l2763_276343


namespace quadratic_intercept_problem_l2763_276391

/-- Given two quadratic functions with specific y-intercepts and rational x-intercepts, prove that h = 1 -/
theorem quadratic_intercept_problem (j k : ℚ) : 
  (∃ x₁ x₂ : ℚ, 4 * (x₁ - 1)^2 + j = 0 ∧ 4 * (x₂ - 1)^2 + j = 0 ∧ x₁ ≠ x₂) →
  (∃ y₁ y₂ : ℚ, 3 * (y₁ - 1)^2 + k = 0 ∧ 3 * (y₂ - 1)^2 + k = 0 ∧ y₁ ≠ y₂) →
  4 * 1^2 + j = 2021 →
  3 * 1^2 + k = 2022 →
  (∀ h : ℚ, (∃ x₁ x₂ : ℚ, 4 * (x₁ - h)^2 + j = 0 ∧ 4 * (x₂ - h)^2 + j = 0 ∧ x₁ ≠ x₂) →
             (∃ y₁ y₂ : ℚ, 3 * (y₁ - h)^2 + k = 0 ∧ 3 * (y₂ - h)^2 + k = 0 ∧ y₁ ≠ y₂) →
             4 * h^2 + j = 2021 →
             3 * h^2 + k = 2022 →
             h = 1) :=
by sorry

end quadratic_intercept_problem_l2763_276391


namespace marble_weight_problem_l2763_276364

theorem marble_weight_problem (piece1 piece2 total : ℝ) 
  (h1 : piece1 = 0.3333333333333333)
  (h2 : piece2 = 0.3333333333333333)
  (h3 : total = 0.75) :
  total - (piece1 + piece2) = 0.08333333333333337 := by
  sorry

end marble_weight_problem_l2763_276364


namespace set_of_positive_rationals_l2763_276374

def is_closed_under_addition_and_multiplication (S : Set ℚ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a * b) ∈ S

def has_trichotomy_property (S : Set ℚ) : Prop :=
  ∀ r : ℚ, (r ∈ S ∧ -r ∉ S ∧ r ≠ 0) ∨ (-r ∈ S ∧ r ∉ S ∧ r ≠ 0) ∨ (r = 0 ∧ r ∉ S ∧ -r ∉ S)

theorem set_of_positive_rationals (S : Set ℚ) 
  (h1 : is_closed_under_addition_and_multiplication S)
  (h2 : has_trichotomy_property S) :
  S = {r : ℚ | 0 < r} :=
sorry

end set_of_positive_rationals_l2763_276374


namespace hyperbola_asymptote_slope_l2763_276339

/-- Given a hyperbola with equation x²/121 - y²/81 = 1, 
    prove that the positive value n in its asymptote equations y = ±nx is 9/11 -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  (x^2 / 121 - y^2 / 81 = 1) →
  (∃ (n : ℝ), n > 0 ∧ (y = n*x ∨ y = -n*x) ∧ n = 9/11) :=
by sorry

end hyperbola_asymptote_slope_l2763_276339


namespace all_turbans_zero_price_l2763_276387

/-- Represents a servant's employment details -/
structure Servant where
  fullYearSalary : ℚ
  monthsWorked : ℚ
  actualPayment : ℚ

/-- Calculates the price of a turban given a servant's details -/
def turbanPrice (s : Servant) : ℚ :=
  s.actualPayment - (s.monthsWorked / 12) * s.fullYearSalary

/-- The main theorem proving that all turbans have zero price -/
theorem all_turbans_zero_price (servantA servantB servantC : Servant)
  (hA : servantA = { fullYearSalary := 120, monthsWorked := 8, actualPayment := 80 })
  (hB : servantB = { fullYearSalary := 150, monthsWorked := 7, actualPayment := 87.5 })
  (hC : servantC = { fullYearSalary := 180, monthsWorked := 10, actualPayment := 150 }) :
  turbanPrice servantA = 0 ∧ turbanPrice servantB = 0 ∧ turbanPrice servantC = 0 := by
  sorry


end all_turbans_zero_price_l2763_276387


namespace fermat_little_theorem_l2763_276395

theorem fermat_little_theorem (p : ℕ) (a : ℤ) (h : Nat.Prime p) :
  ∃ k : ℤ, a^p - a = k * p :=
sorry

end fermat_little_theorem_l2763_276395


namespace prob_two_nondefective_pens_l2763_276366

/-- Given a box of 8 pens with 2 defective pens, the probability of selecting 2 non-defective pens at random is 15/28. -/
theorem prob_two_nondefective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 8) 
  (h2 : defective_pens = 2) : 
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 15 / 28 := by
  sorry

#check prob_two_nondefective_pens

end prob_two_nondefective_pens_l2763_276366


namespace inequality_system_solution_l2763_276304

theorem inequality_system_solution (x : ℝ) : 
  (abs (x - 1) > 1 ∧ 1 / (4 - x) ≤ 1) ↔ (x < 0 ∨ (2 < x ∧ x ≤ 3) ∨ x > 4) :=
by sorry

end inequality_system_solution_l2763_276304


namespace coupon_discount_percentage_l2763_276309

theorem coupon_discount_percentage (original_price increased_price final_price : ℝ) 
  (h1 : original_price = 200)
  (h2 : increased_price = original_price * 1.3)
  (h3 : final_price = 182) : 
  (increased_price - final_price) / increased_price = 0.3 := by
sorry

end coupon_discount_percentage_l2763_276309


namespace power_function_property_l2763_276378

/-- A power function with a specific property -/
def PowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x > 0, f x = x ^ α

theorem power_function_property (f : ℝ → ℝ) (h1 : PowerFunction f) (h2 : f 4 / f 2 = 3) :
  f (1/2) = 1/3 := by
  sorry

end power_function_property_l2763_276378


namespace quarter_circle_sum_approaches_semi_circumference_l2763_276394

/-- The sum of quarter-circle arc lengths approaches the semi-circumference as n approaches infinity --/
theorem quarter_circle_sum_approaches_semi_circumference (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |2 * n * (π * D / (4 * n)) - π * D / 2| < ε :=
sorry

end quarter_circle_sum_approaches_semi_circumference_l2763_276394


namespace average_of_first_25_odd_primes_l2763_276369

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def first_25_odd_primes : List ℕ := 
  [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]

theorem average_of_first_25_odd_primes : 
  (∀ p ∈ first_25_odd_primes, is_prime p ∧ is_odd p) → 
  (List.sum first_25_odd_primes).toFloat / 25 = 47.48 := by
  sorry

end average_of_first_25_odd_primes_l2763_276369


namespace candy_store_food_colouring_l2763_276329

/-- The amount of food colouring used by a candy store in one day -/
def total_food_colouring (lollipop_count : ℕ) (hard_candy_count : ℕ) 
  (lollipop_colouring : ℕ) (hard_candy_colouring : ℕ) : ℕ :=
  lollipop_count * lollipop_colouring + hard_candy_count * hard_candy_colouring

/-- Theorem stating the total amount of food colouring used by the candy store -/
theorem candy_store_food_colouring : 
  total_food_colouring 100 5 5 20 = 600 := by
  sorry

end candy_store_food_colouring_l2763_276329


namespace inequality_proof_l2763_276319

theorem inequality_proof (a b : ℝ) (h : a * b > 0) : b / a + a / b ≥ 2 := by
  sorry

end inequality_proof_l2763_276319


namespace alfonso_savings_l2763_276333

theorem alfonso_savings (daily_rate : ℕ) (days_per_week : ℕ) (total_weeks : ℕ) (helmet_cost : ℕ) :
  let total_earned := daily_rate * days_per_week * total_weeks
  helmet_cost - total_earned = 40 :=
by
  sorry

end alfonso_savings_l2763_276333


namespace sqrt_equation_solution_l2763_276323

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (3 + Real.sqrt (4 * y - 5)) = Real.sqrt 8 → y = 7.5 := by
  sorry

end sqrt_equation_solution_l2763_276323


namespace function_has_infinitely_many_extreme_points_l2763_276306

/-- The function f(x) = x^2 - 2x cos(x) has infinitely many extreme points -/
theorem function_has_infinitely_many_extreme_points :
  ∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - 2*x*(Real.cos x)) ∧
  (∀ n : ℕ, ∃ (S : Finset ℝ), S.card ≥ n ∧ 
    (∀ x ∈ S, ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x)) :=
by sorry


end function_has_infinitely_many_extreme_points_l2763_276306


namespace division_problem_l2763_276308

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by sorry

end division_problem_l2763_276308


namespace lesser_number_problem_l2763_276301

theorem lesser_number_problem (x y : ℝ) 
  (sum_eq : x + y = 50) 
  (diff_eq : x - y = 7) : 
  y = 21.5 := by
sorry

end lesser_number_problem_l2763_276301


namespace jorge_corn_yield_l2763_276326

/-- Represents the yield calculation for Jorge's corn plantation --/
def jorge_yield (total_acres : ℝ) (clay_rich_fraction : ℝ) (total_yield : ℝ) (other_soil_yield : ℝ) : Prop :=
  let clay_rich_acres := clay_rich_fraction * total_acres
  let other_soil_acres := (1 - clay_rich_fraction) * total_acres
  let clay_rich_yield := (other_soil_yield / 2) * clay_rich_acres
  let other_soil_total_yield := other_soil_yield * other_soil_acres
  clay_rich_yield + other_soil_total_yield = total_yield

theorem jorge_corn_yield :
  jorge_yield 60 (1/3) 20000 400 := by
  sorry

end jorge_corn_yield_l2763_276326


namespace log_equality_l2763_276382

-- Define the logarithm base 2 (lg)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_equality : lg (5/2) + 2 * lg 2 + 2^(Real.log 3 / Real.log 4) = 1 + Real.sqrt 3 := by
  sorry

end log_equality_l2763_276382


namespace honey_production_l2763_276390

theorem honey_production (num_hives : ℕ) (jar_capacity : ℚ) (jars_for_half : ℕ) 
  (h1 : num_hives = 5)
  (h2 : jar_capacity = 1/2)
  (h3 : jars_for_half = 100) :
  (2 * jars_for_half : ℚ) * jar_capacity / num_hives = 20 := by
  sorry

end honey_production_l2763_276390


namespace quadratic_factorization_l2763_276303

theorem quadratic_factorization (x : ℝ) : -2 * x^2 + 2 * x - (1/2) = -2 * (x - 1/2)^2 := by
  sorry

end quadratic_factorization_l2763_276303


namespace total_snakes_l2763_276336

/-- Given information about pet ownership, prove the total number of snakes. -/
theorem total_snakes (total_people : ℕ) (only_dogs : ℕ) (only_cats : ℕ) (only_snakes : ℕ)
  (dogs_and_cats : ℕ) (cats_and_snakes : ℕ) (dogs_and_snakes : ℕ) (all_three : ℕ)
  (h1 : total_people = 120)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_snakes = 12)
  (h5 : dogs_and_cats = 15)
  (h6 : cats_and_snakes = 10)
  (h7 : dogs_and_snakes = 8)
  (h8 : all_three = 5) :
  only_snakes + cats_and_snakes + dogs_and_snakes + all_three = 35 := by
  sorry

end total_snakes_l2763_276336


namespace william_tickets_l2763_276317

/-- William's ticket problem -/
theorem william_tickets : ∀ (initial additional : ℕ), 
  initial = 15 → additional = 3 → initial + additional = 18 := by
  sorry

end william_tickets_l2763_276317


namespace quadratic_factorization_l2763_276300

theorem quadratic_factorization (a b c : ℤ) :
  (∀ x : ℚ, x^2 + 16*x + 63 = (x + a) * (x + b)) →
  (∀ x : ℚ, x^2 + 6*x - 72 = (x + b) * (x - c)) →
  a + b + c = 25 := by
  sorry

end quadratic_factorization_l2763_276300


namespace income_calculation_l2763_276357

/-- Calculates a person's income given the income to expenditure ratio and savings amount. -/
def calculate_income (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : ℕ :=
  (income_ratio * savings) / (income_ratio - expenditure_ratio)

/-- Proves that given a person's income to expenditure ratio of 5:4 and savings of Rs. 3000, 
    the person's income is Rs. 15000. -/
theorem income_calculation (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) 
  (h1 : income_ratio = 5) 
  (h2 : expenditure_ratio = 4) 
  (h3 : savings = 3000) : 
  calculate_income income_ratio expenditure_ratio savings = 15000 := by
  sorry

end income_calculation_l2763_276357


namespace complex_modulus_problem_l2763_276385

theorem complex_modulus_problem (z : ℂ) : (1 - Complex.I) * z = Complex.I → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l2763_276385


namespace circle_op_not_commutative_l2763_276372

/-- Defines the "☉" operation for plane vectors -/
def circle_op (a b : ℝ × ℝ) : ℝ :=
  let (m, n) := a
  let (p, q) := b
  m * q - n * p

/-- Theorem stating that the "☉" operation is not commutative -/
theorem circle_op_not_commutative :
  ∃ (a b : ℝ × ℝ), circle_op a b ≠ circle_op b a :=
sorry

end circle_op_not_commutative_l2763_276372


namespace garden_area_difference_l2763_276318

def alice_length : ℝ := 15
def alice_width : ℝ := 30
def bob_length : ℝ := 18
def bob_width : ℝ := 28

theorem garden_area_difference :
  bob_length * bob_width - alice_length * alice_width = 54 := by
  sorry

end garden_area_difference_l2763_276318


namespace subscription_cost_l2763_276361

theorem subscription_cost (reduction_percentage : ℝ) (reduction_amount : ℝ) (original_cost : ℝ) : 
  reduction_percentage = 0.30 →
  reduction_amount = 658 →
  reduction_percentage * original_cost = reduction_amount →
  original_cost = 2193 := by
  sorry

end subscription_cost_l2763_276361


namespace monopolist_optimal_quantity_l2763_276321

/-- Represents the demand function for a monopolist's product -/
def demand (P : ℝ) : ℝ := 10 - P

/-- Represents the revenue function for the monopolist -/
def revenue (Q : ℝ) : ℝ := Q * (10 - Q)

/-- Represents the profit function for the monopolist -/
def profit (Q : ℝ) : ℝ := revenue Q

/-- The maximum quantity of goods the monopolist can sell -/
def max_quantity : ℝ := 10

/-- Theorem: The monopolist maximizes profit by selling 5 units -/
theorem monopolist_optimal_quantity :
  ∃ (Q : ℝ), Q = 5 ∧ 
  Q ≤ max_quantity ∧
  ∀ (Q' : ℝ), Q' ≤ max_quantity → profit Q' ≤ profit Q :=
sorry

end monopolist_optimal_quantity_l2763_276321


namespace divisibility_property_l2763_276340

theorem divisibility_property (a b c : ℕ) : 
  a ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) → 
  b ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) → 
  c ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) → 
  a ≠ b → b ≠ c → a ≠ c →
  ∃ k : ℤ, a * b * c + (7 - a) * (7 - b) * (7 - c) = 7 * k := by
sorry

end divisibility_property_l2763_276340


namespace equation_solution_l2763_276311

theorem equation_solution : ∃ x : ℚ, (x^2 + 4*x + 7) / (x + 5) = x + 6 ∧ x = -23/7 := by
  sorry

end equation_solution_l2763_276311


namespace correct_average_proof_l2763_276307

/-- The number of students in the class -/
def num_students : ℕ := 60

/-- The incorrect average marks -/
def incorrect_average : ℚ := 82

/-- Reema's correct mark -/
def reema_correct : ℕ := 78

/-- Reema's incorrect mark -/
def reema_incorrect : ℕ := 68

/-- Mark's correct mark -/
def mark_correct : ℕ := 95

/-- Mark's incorrect mark -/
def mark_incorrect : ℕ := 91

/-- Jenny's correct mark -/
def jenny_correct : ℕ := 84

/-- Jenny's incorrect mark -/
def jenny_incorrect : ℕ := 74

/-- The correct average marks -/
def correct_average : ℚ := 82.40

theorem correct_average_proof :
  let incorrect_total := (incorrect_average * num_students : ℚ)
  let mark_difference := (reema_correct - reema_incorrect) + (mark_correct - mark_incorrect) + (jenny_correct - jenny_incorrect)
  let correct_total := incorrect_total + mark_difference
  (correct_total / num_students : ℚ) = correct_average := by sorry

end correct_average_proof_l2763_276307


namespace cubic_root_sum_l2763_276344

theorem cubic_root_sum (u v w : ℝ) : 
  u^3 - 6*u^2 + 11*u - 6 = 0 →
  v^3 - 6*v^2 + 11*v - 6 = 0 →
  w^3 - 6*w^2 + 11*w - 6 = 0 →
  u * v / w + v * w / u + w * u / v = 49 / 6 := by
sorry

end cubic_root_sum_l2763_276344


namespace max_visible_blue_cubes_l2763_276397

/-- Represents a column of cubes with red and blue colors -/
structure CubeColumn :=
  (total : Nat)
  (blue : Nat)
  (red : Nat)
  (h_sum : blue + red = total)

/-- Represents a row of three columns on the board -/
structure BoardRow :=
  (left : CubeColumn)
  (middle : CubeColumn)
  (right : CubeColumn)

/-- The entire 3x3 board configuration -/
structure Board :=
  (front : BoardRow)
  (middle : BoardRow)
  (back : BoardRow)

/-- Calculates the maximum number of visible blue cubes in a row -/
def maxVisibleBlueInRow (row : BoardRow) : Nat :=
  row.left.blue + max 0 (row.middle.total - row.left.total) + max 0 (row.right.total - max row.left.total row.middle.total)

/-- The main theorem stating the maximum number of visible blue cubes -/
theorem max_visible_blue_cubes (board : Board) : 
  maxVisibleBlueInRow board.front + maxVisibleBlueInRow board.middle + maxVisibleBlueInRow board.back ≤ 12 :=
sorry

end max_visible_blue_cubes_l2763_276397


namespace masters_sample_size_l2763_276377

/-- Calculates the sample size for a specific stratum in stratified sampling -/
def stratifiedSampleSize (totalRatio : ℕ) (stratumRatio : ℕ) (totalSample : ℕ) : ℕ :=
  (stratumRatio * totalSample) / totalRatio

/-- Proves that the sample size for master's students is 36 given the conditions -/
theorem masters_sample_size :
  let totalRatio : ℕ := 5 + 15 + 9 + 1
  let mastersRatio : ℕ := 9
  let totalSample : ℕ := 120
  stratifiedSampleSize totalRatio mastersRatio totalSample = 36 := by
  sorry

#eval stratifiedSampleSize 30 9 120

end masters_sample_size_l2763_276377


namespace quadratic_roots_condition_l2763_276349

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : Prop :=
  k * x^2 + 2 * x + 1 = 0

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y

-- Theorem statement
theorem quadratic_roots_condition (k : ℝ) :
  has_two_distinct_real_roots k ↔ k < 1 ∧ k ≠ 0 :=
sorry

end quadratic_roots_condition_l2763_276349


namespace monomial_properties_l2763_276358

-- Define the monomial structure
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define the monomial -2x^2y
def monomial : Monomial ℤ :=
  { coeff := -2,
    vars := [(1, 2), (2, 1)] }  -- Representing x^2 and y^1

-- Theorem statement
theorem monomial_properties :
  (monomial.coeff = -2) ∧
  (List.sum (monomial.vars.map (λ (_, exp) => exp)) = 3) :=
by sorry

end monomial_properties_l2763_276358


namespace power_multiplication_l2763_276331

theorem power_multiplication (a : ℝ) : (-a^2)^3 * a^3 = -a^9 := by sorry

end power_multiplication_l2763_276331


namespace finite_good_numbers_not_divisible_by_l2763_276313

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- n is a good number if τ(m) < τ(n) for all m < n -/
def is_good (n : ℕ+) : Prop :=
  ∀ m : ℕ+, m < n → tau m < tau n

/-- The set of good numbers not divisible by k is finite -/
theorem finite_good_numbers_not_divisible_by (k : ℕ+) :
  {n : ℕ+ | is_good n ∧ ¬k ∣ n}.Finite := by sorry

end finite_good_numbers_not_divisible_by_l2763_276313


namespace percentage_less_l2763_276338

theorem percentage_less (x y : ℝ) (h : x = 3 * y) : 
  (x - y) / x * 100 = 200 / 3 := by
  sorry

end percentage_less_l2763_276338


namespace half_times_two_thirds_times_three_fourths_l2763_276386

theorem half_times_two_thirds_times_three_fourths :
  (1 / 2 : ℚ) * (2 / 3 : ℚ) * (3 / 4 : ℚ) = 1 / 4 := by
  sorry

end half_times_two_thirds_times_three_fourths_l2763_276386


namespace paint_fraction_proof_l2763_276368

def paint_problem (initial_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) : Prop :=
  let remaining_after_first_week : ℚ := initial_paint - (first_week_fraction * initial_paint)
  let used_second_week : ℚ := total_used - (first_week_fraction * initial_paint)
  (used_second_week / remaining_after_first_week) = 1 / 6

theorem paint_fraction_proof :
  paint_problem 360 (1 / 4) 135 := by
  sorry

end paint_fraction_proof_l2763_276368


namespace product_inequality_l2763_276367

theorem product_inequality (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
  (hab : a + b = 2) (hcd : c + d = 2) : 
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := by
  sorry

end product_inequality_l2763_276367


namespace family_average_age_l2763_276327

theorem family_average_age 
  (n : ℕ) 
  (youngest_age : ℕ) 
  (past_average : ℚ) : 
  n = 7 → 
  youngest_age = 5 → 
  past_average = 28 → 
  (((n - 1) * past_average + (n - 1) * youngest_age + youngest_age) / n : ℚ) = 209/7 := by
  sorry

end family_average_age_l2763_276327


namespace scrabble_middle_letter_value_l2763_276356

/-- Given a three-letter word in Scrabble with known conditions, 
    prove the value of the middle letter. -/
theorem scrabble_middle_letter_value 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ) 
  (total_score : ℕ) 
  (h1 : first_letter_value = 1)
  (h2 : third_letter_value = 1)
  (h3 : total_score = 30)
  (h4 : ∃ (middle_letter_value : ℕ), 
    3 * (first_letter_value + middle_letter_value + third_letter_value) = total_score) :
  ∃ (middle_letter_value : ℕ), middle_letter_value = 8 := by
  sorry

end scrabble_middle_letter_value_l2763_276356


namespace jason_grass_cutting_time_l2763_276312

/-- The time Jason spends cutting grass over a weekend -/
def time_cutting_grass (time_per_lawn : ℕ) (lawns_per_day : ℕ) (days : ℕ) : ℕ :=
  time_per_lawn * lawns_per_day * days

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem jason_grass_cutting_time :
  let time_per_lawn := 30
  let lawns_per_day := 8
  let days := 2
  minutes_to_hours (time_cutting_grass time_per_lawn lawns_per_day days) = 8 := by
  sorry

end jason_grass_cutting_time_l2763_276312


namespace inequality_proof_l2763_276330

theorem inequality_proof (x a : ℝ) (h1 : x < a) (h2 : a < -1) (h3 : x < 0) (h4 : a < 0) :
  x^2 > a*x ∧ a*x > a^2 + 1 := by
  sorry

end inequality_proof_l2763_276330


namespace green_pieces_count_l2763_276363

theorem green_pieces_count (amber : ℕ) (clear : ℕ) (green : ℕ) :
  amber = 20 →
  clear = 85 →
  green = (25 : ℚ) / 100 * (amber + green + clear) →
  green = 35 := by
sorry

end green_pieces_count_l2763_276363


namespace shekar_average_marks_l2763_276388

def shekar_scores : List Nat := [76, 65, 82, 67, 55]

theorem shekar_average_marks :
  let total_marks := shekar_scores.sum
  let num_subjects := shekar_scores.length
  (total_marks / num_subjects : ℚ) = 69 := by
  sorry

end shekar_average_marks_l2763_276388


namespace chemistry_class_size_l2763_276359

theorem chemistry_class_size 
  (total_students : ℕ) 
  (chem_only : ℕ) 
  (bio_only : ℕ) 
  (both : ℕ) 
  (h1 : total_students = 70)
  (h2 : total_students = chem_only + bio_only + both)
  (h3 : chem_only + both = 2 * (bio_only + both))
  (h4 : both = 8) : 
  chem_only + both = 52 := by
  sorry

end chemistry_class_size_l2763_276359


namespace inverse_relationship_R_squared_residuals_l2763_276325

/-- Represents the coefficient of determination in regression analysis -/
def R_squared : ℝ := sorry

/-- Represents the sum of squares of residuals in regression analysis -/
def sum_of_squares_residuals : ℝ := sorry

/-- States that there is an inverse relationship between R² and the sum of squares of residuals -/
theorem inverse_relationship_R_squared_residuals :
  ∀ (R₁ R₂ : ℝ) (SSR₁ SSR₂ : ℝ),
    R₁ < R₂ → SSR₁ > SSR₂ :=
by sorry

end inverse_relationship_R_squared_residuals_l2763_276325


namespace simple_interest_principal_l2763_276392

/-- Simple interest calculation -/
theorem simple_interest_principal (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 4.8)
  (h2 : T = 12)
  (h3 : R = 0.05)
  (h4 : SI = P * R * T) :
  P = 8 := by
  sorry

end simple_interest_principal_l2763_276392


namespace rectangle_arrangement_exists_l2763_276376

theorem rectangle_arrangement_exists : ∃ (a b c d : ℕ+), 
  (a * b + c * d = 49) ∧ 
  ((2 * (a + b) = 4 * (c + d)) ∨ (2 * (c + d) = 4 * (a + b))) := by
  sorry

end rectangle_arrangement_exists_l2763_276376


namespace juice_cost_calculation_l2763_276373

theorem juice_cost_calculation (orange_cost apple_cost total_bottles orange_bottles : ℕ) 
  (h1 : orange_cost = 70)
  (h2 : apple_cost = 60)
  (h3 : total_bottles = 70)
  (h4 : orange_bottles = 42) :
  orange_cost * orange_bottles + apple_cost * (total_bottles - orange_bottles) = 4620 := by
  sorry

#check juice_cost_calculation

end juice_cost_calculation_l2763_276373


namespace simplest_square_root_l2763_276389

theorem simplest_square_root :
  let options := [Real.sqrt 8, (Real.sqrt 2)⁻¹, Real.sqrt 2, Real.sqrt (1/2)]
  ∃ (x : ℝ), x ∈ options ∧ 
    (∀ y ∈ options, x ≠ y → (∃ z : ℝ, z ≠ 1 ∧ y = z * x ∨ y = x / z ∨ y = Real.sqrt (z * x^2))) ∧
    x = Real.sqrt 2 := by
  sorry

end simplest_square_root_l2763_276389


namespace quiz_goal_achievement_l2763_276314

theorem quiz_goal_achievement (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (current_as : ℕ) : 
  total_quizzes = 40 →
  goal_percentage = 85 / 100 →
  completed_quizzes = 25 →
  current_as = 20 →
  ∃ (max_non_as : ℕ), 
    max_non_as = 1 ∧ 
    (current_as + (total_quizzes - completed_quizzes - max_non_as)) / total_quizzes ≥ goal_percentage ∧
    ∀ (x : ℕ), x > max_non_as → 
      (current_as + (total_quizzes - completed_quizzes - x)) / total_quizzes < goal_percentage :=
by sorry

end quiz_goal_achievement_l2763_276314


namespace impossible_three_coin_piles_l2763_276332

/-- Represents the coin removal and division process -/
def coin_process (initial_coins : ℕ) (steps : ℕ) : Prop :=
  ∃ (final_piles : ℕ),
    (initial_coins - steps = 3 * final_piles) ∧
    (final_piles = steps + 1)

/-- Theorem stating the impossibility of ending with only piles of three coins -/
theorem impossible_three_coin_piles : ¬∃ (steps : ℕ), coin_process 2013 steps :=
  sorry

end impossible_three_coin_piles_l2763_276332
