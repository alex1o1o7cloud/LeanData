import Mathlib

namespace NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_for_increasing_l3123_312330

-- Define a geometric sequence
def geometric_sequence (a₀ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₀ * q^n

-- Define monotonically increasing sequence
def monotonically_increasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s n ≤ s (n + 1)

-- Theorem statement
theorem q_gt_one_neither_sufficient_nor_necessary_for_increasing
  (a₀ : ℝ) (q : ℝ) :
  ¬(((q > 1) → monotonically_increasing (geometric_sequence a₀ q)) ∧
    (monotonically_increasing (geometric_sequence a₀ q) → (q > 1))) :=
by sorry

end NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_for_increasing_l3123_312330


namespace NUMINAMATH_CALUDE_final_temperature_of_mixed_gases_l3123_312335

/-- The final temperature of mixed gases in thermally insulated vessels -/
theorem final_temperature_of_mixed_gases
  (V₁ V₂ : ℝ) (p₁ p₂ : ℝ) (T₁ T₂ : ℝ) (R : ℝ) :
  V₁ = 1 →
  V₂ = 2 →
  p₁ = 2 →
  p₂ = 3 →
  T₁ = 300 →
  T₂ = 400 →
  R > 0 →
  let n₁ := p₁ * V₁ / (R * T₁)
  let n₂ := p₂ * V₂ / (R * T₂)
  let T := (n₁ * T₁ + n₂ * T₂) / (n₁ + n₂)
  ∃ ε > 0, |T - 369| < ε :=
sorry

end NUMINAMATH_CALUDE_final_temperature_of_mixed_gases_l3123_312335


namespace NUMINAMATH_CALUDE_inner_set_area_of_specific_triangle_l3123_312345

/-- Triangle with side lengths a, b, c -/
structure Triangle (a b c : ℝ) where
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Set of points inside a triangle not within distance d of any side -/
def InnerSet (T : Triangle a b c) (d : ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Main theorem -/
theorem inner_set_area_of_specific_triangle :
  let T : Triangle 26 51 73 := ⟨sorry, sorry, sorry, sorry⟩
  let S := InnerSet T 5
  area S = 135 / 28 := by
  sorry

end NUMINAMATH_CALUDE_inner_set_area_of_specific_triangle_l3123_312345


namespace NUMINAMATH_CALUDE_work_ratio_l3123_312314

/-- Given that A can finish a work in 18 days and A and B working together can finish 1/6 of the work in a day, 
    prove that the ratio of time taken by B to A to finish the work is 1:2 -/
theorem work_ratio (a_time : ℝ) (combined_rate : ℝ) 
  (ha : a_time = 18)
  (hc : combined_rate = 1/6) : 
  (a_time / 2) / a_time = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_work_ratio_l3123_312314


namespace NUMINAMATH_CALUDE_valid_probability_is_one_fourteenth_l3123_312354

/-- Represents a bead color -/
inductive BeadColor
| Red
| White
| Blue

/-- Represents a configuration of beads -/
def BeadConfiguration := List BeadColor

/-- Checks if a configuration has no adjacent beads of the same color -/
def noAdjacentSameColor (config : BeadConfiguration) : Bool :=
  sorry

/-- Generates all possible bead configurations -/
def allConfigurations : List BeadConfiguration :=
  sorry

/-- Counts the number of valid configurations -/
def countValidConfigurations : Nat :=
  sorry

/-- The total number of possible configurations -/
def totalConfigurations : Nat := 420

/-- The probability of a valid configuration -/
def validProbability : ℚ :=
  sorry

theorem valid_probability_is_one_fourteenth :
  validProbability = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_valid_probability_is_one_fourteenth_l3123_312354


namespace NUMINAMATH_CALUDE_kekai_remaining_money_l3123_312351

def shirt_price : ℝ := 1
def shirt_discount : ℝ := 0.2
def pants_price : ℝ := 3
def pants_discount : ℝ := 0.1
def hat_price : ℝ := 2
def hat_discount : ℝ := 0
def shoes_price : ℝ := 10
def shoes_discount : ℝ := 0.15
def parent_contribution : ℝ := 0.35

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def hats_sold : ℕ := 3
def shoes_sold : ℕ := 2

def total_sales (shirt_price pants_price hat_price shoes_price : ℝ)
                (shirt_discount pants_discount hat_discount shoes_discount : ℝ)
                (shirts_sold pants_sold hats_sold shoes_sold : ℕ) : ℝ :=
  (shirt_price * (1 - shirt_discount) * shirts_sold) +
  (pants_price * (1 - pants_discount) * pants_sold) +
  (hat_price * (1 - hat_discount) * hats_sold) +
  (shoes_price * (1 - shoes_discount) * shoes_sold)

def remaining_money (total : ℝ) (contribution : ℝ) : ℝ :=
  total * (1 - contribution)

theorem kekai_remaining_money :
  remaining_money (total_sales shirt_price pants_price hat_price shoes_price
                                shirt_discount pants_discount hat_discount shoes_discount
                                shirts_sold pants_sold hats_sold shoes_sold)
                  parent_contribution = 26.32 := by
  sorry

end NUMINAMATH_CALUDE_kekai_remaining_money_l3123_312351


namespace NUMINAMATH_CALUDE_box_interior_area_l3123_312317

/-- Calculates the surface area of the interior of a box formed from a rectangular sheet of cardboard
    with square corners cut out and edges folded upwards. -/
def interior_surface_area (sheet_length : ℕ) (sheet_width : ℕ) (corner_size : ℕ) : ℕ :=
  (sheet_length - 2 * corner_size) * (sheet_width - 2 * corner_size)

/-- Theorem stating that the interior surface area of the box formed from a 35x50 sheet
    with 7-unit corners cut out is 756 square units. -/
theorem box_interior_area :
  interior_surface_area 35 50 7 = 756 := by
  sorry

end NUMINAMATH_CALUDE_box_interior_area_l3123_312317


namespace NUMINAMATH_CALUDE_expression_evaluation_l3123_312315

theorem expression_evaluation (d : ℕ) (h : d = 4) : 
  (d^d - d*(d-2)^d + Nat.factorial (d-1))^2 = 39204 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3123_312315


namespace NUMINAMATH_CALUDE_toms_age_ratio_l3123_312391

theorem toms_age_ratio (T N : ℝ) : T > 0 → N > 0 → T - N = 3 * (T - 3 * N) → T / N = 4 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l3123_312391


namespace NUMINAMATH_CALUDE_square_difference_product_l3123_312331

theorem square_difference_product : (476 + 424)^2 - 4 * 476 * 424 = 4624 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_product_l3123_312331


namespace NUMINAMATH_CALUDE_silverware_probability_l3123_312396

def total_silverware : ℕ := 24
def forks : ℕ := 8
def spoons : ℕ := 10
def knives : ℕ := 6
def pieces_removed : ℕ := 4

theorem silverware_probability :
  let total_ways := Nat.choose total_silverware pieces_removed
  let favorable_ways := Nat.choose forks 2 * Nat.choose spoons 2
  (favorable_ways : ℚ) / total_ways = 18 / 91 := by sorry

end NUMINAMATH_CALUDE_silverware_probability_l3123_312396


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_prism_l3123_312361

/-- Given a right square prism with height 4 and volume 16, 
    where all vertices are on the surface of a sphere,
    prove that the surface area of the sphere is 24π -/
theorem sphere_surface_area_from_prism (h : ℝ) (v : ℝ) (r : ℝ) : 
  h = 4 →
  v = 16 →
  v = h * r^2 →
  r^2 + h^2 / 4 + r^2 = (2 * r)^2 →
  4 * π * ((r^2 + h^2 / 4 + r^2) / 4) = 24 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_prism_l3123_312361


namespace NUMINAMATH_CALUDE_sixth_salary_proof_l3123_312384

theorem sixth_salary_proof (known_salaries : List ℝ) 
  (h1 : known_salaries = [1000, 2500, 3100, 3650, 2000])
  (h2 : (known_salaries.sum + x) / 6 = 2291.67) : x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sixth_salary_proof_l3123_312384


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_minus_five_l3123_312346

theorem sqrt_two_times_sqrt_three_minus_five (x : ℝ) :
  x = Real.sqrt 2 * Real.sqrt 3 - 5 → x = Real.sqrt 6 - 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_minus_five_l3123_312346


namespace NUMINAMATH_CALUDE_union_of_sets_l3123_312336

theorem union_of_sets : 
  let A : Set ℕ := {1, 3}
  let B : Set ℕ := {1, 2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3123_312336


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3123_312379

/-- The equation (x-3)^2 = 9(y+2)^2 - 81 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b h k : ℝ) (A B : ℝ → ℝ → Prop),
    (∀ x y, A x y ↔ (x - 3)^2 = 9*(y + 2)^2 - 81) ∧
    (∀ x y, B x y ↔ ((x - h) / a)^2 - ((y - k) / b)^2 = 1) ∧
    (∀ x y, A x y ↔ B x y) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3123_312379


namespace NUMINAMATH_CALUDE_ratio_w_y_l3123_312352

-- Define the ratios
def ratio_w_x : ℚ := 5 / 4
def ratio_y_z : ℚ := 7 / 5
def ratio_z_x : ℚ := 1 / 8

-- Theorem statement
theorem ratio_w_y (w x y z : ℚ) 
  (hw : w / x = ratio_w_x)
  (hy : y / z = ratio_y_z)
  (hz : z / x = ratio_z_x) : 
  w / y = 25 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_y_l3123_312352


namespace NUMINAMATH_CALUDE_mary_travel_time_l3123_312307

/-- The total time Mary spends from calling the Uber to the plane being ready for takeoff -/
def total_time (uber_to_house : ℕ) (bag_check : ℕ) (wait_for_boarding : ℕ) : ℕ :=
  let uber_to_airport := 5 * uber_to_house
  let security := 3 * bag_check
  let wait_for_takeoff := 2 * wait_for_boarding
  uber_to_house + uber_to_airport + bag_check + security + wait_for_boarding + wait_for_takeoff

/-- The theorem stating that Mary's total travel preparation time is 3 hours -/
theorem mary_travel_time :
  total_time 10 15 20 = 180 :=
sorry

end NUMINAMATH_CALUDE_mary_travel_time_l3123_312307


namespace NUMINAMATH_CALUDE_minimum_guests_l3123_312395

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 323) (h2 : max_per_guest = 2) :
  ∃ (min_guests : ℕ), min_guests = 162 ∧ min_guests * max_per_guest ≥ total_food ∧
  ∀ (n : ℕ), n * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l3123_312395


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3123_312323

theorem polynomial_expansion (x : ℝ) : 
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(2*x^2 + 5*x - 72) + (4*x - 21)*(x - 2)*(x - 3) = 
  5*x^3 - 23*x^2 + 43*x - 34 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3123_312323


namespace NUMINAMATH_CALUDE_minimize_sum_distances_l3123_312399

/-- A type representing points on a line -/
structure Point where
  x : ℝ

/-- The distance between two points on a line -/
def distance (p q : Point) : ℝ := |p.x - q.x|

/-- The sum of distances from a point to a list of points -/
def sum_distances (q : Point) (points : List Point) : ℝ :=
  points.foldl (fun sum p => sum + distance p q) 0

theorem minimize_sum_distances 
  (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ : Point)
  (h : p₁.x < p₂.x ∧ p₂.x < p₃.x ∧ p₃.x < p₄.x ∧ p₄.x < p₅.x ∧ p₅.x < p₆.x ∧ p₆.x < p₇.x ∧ p₇.x < p₈.x) :
  ∃ (q : Point), 
    (∀ (r : Point), sum_distances q [p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈] ≤ sum_distances r [p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈]) ∧
    q.x = (p₄.x + p₅.x) / 2 := by
  sorry


end NUMINAMATH_CALUDE_minimize_sum_distances_l3123_312399


namespace NUMINAMATH_CALUDE_hash_difference_l3123_312363

/-- Custom operation # -/
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

/-- Theorem stating the result of (5 # 3) - (3 # 5) -/
theorem hash_difference : hash 5 3 - hash 3 5 = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l3123_312363


namespace NUMINAMATH_CALUDE_job_completion_time_l3123_312373

-- Define the problem parameters
def initial_workers : ℕ := 6
def initial_days : ℕ := 8
def days_before_joining : ℕ := 3
def additional_workers : ℕ := 4

-- Define the total work as a fraction
def total_work : ℚ := 1

-- Define the work rate of one worker per day
def work_rate_per_worker : ℚ := 1 / (initial_workers * initial_days)

-- Define the work completed in the first 3 days
def work_completed_first_phase : ℚ := initial_workers * work_rate_per_worker * days_before_joining

-- Define the remaining work
def remaining_work : ℚ := total_work - work_completed_first_phase

-- Define the total number of workers after joining
def total_workers : ℕ := initial_workers + additional_workers

-- Define the work rate of all workers after joining
def work_rate_after_joining : ℚ := total_workers * work_rate_per_worker

-- State the theorem
theorem job_completion_time :
  ∃ (remaining_days : ℕ), 
    (days_before_joining : ℚ) + remaining_days = 6 ∧
    remaining_work = work_rate_after_joining * remaining_days :=
sorry

end NUMINAMATH_CALUDE_job_completion_time_l3123_312373


namespace NUMINAMATH_CALUDE_product_97_103_l3123_312388

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_97_103_l3123_312388


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3123_312369

theorem fraction_to_decimal : (59 : ℚ) / 160 = (36875 : ℚ) / 100000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3123_312369


namespace NUMINAMATH_CALUDE_cost_difference_is_six_l3123_312316

/-- Represents the cost and consumption of a pizza -/
structure PizzaCost where
  totalSlices : ℕ
  plainCost : ℚ
  toppingCost : ℚ
  daveToppedSlices : ℕ
  davePlainSlices : ℕ

/-- Calculates the difference in cost between Dave's and Doug's portions -/
def costDifference (p : PizzaCost) : ℚ :=
  let totalCost := p.plainCost + p.toppingCost
  let costPerSlice := totalCost / p.totalSlices
  let daveCost := costPerSlice * (p.daveToppedSlices + p.davePlainSlices)
  let dougSlices := p.totalSlices - p.daveToppedSlices - p.davePlainSlices
  let dougCost := (p.plainCost / p.totalSlices) * dougSlices
  daveCost - dougCost

/-- Theorem stating that the cost difference is $6 -/
theorem cost_difference_is_six (p : PizzaCost) 
  (h1 : p.totalSlices = 12)
  (h2 : p.plainCost = 12)
  (h3 : p.toppingCost = 3)
  (h4 : p.daveToppedSlices = 6)
  (h5 : p.davePlainSlices = 2) :
  costDifference p = 6 := by
  sorry

#eval costDifference { totalSlices := 12, plainCost := 12, toppingCost := 3, daveToppedSlices := 6, davePlainSlices := 2 }

end NUMINAMATH_CALUDE_cost_difference_is_six_l3123_312316


namespace NUMINAMATH_CALUDE_new_arithmetic_mean_l3123_312309

def original_set_size : ℕ := 60
def original_mean : ℚ := 42
def removed_numbers : List ℚ := [50, 60, 70]

theorem new_arithmetic_mean :
  let original_sum : ℚ := original_mean * original_set_size
  let removed_sum : ℚ := removed_numbers.sum
  let new_sum : ℚ := original_sum - removed_sum
  let new_set_size : ℕ := original_set_size - removed_numbers.length
  (new_sum / new_set_size : ℚ) = 41 := by sorry

end NUMINAMATH_CALUDE_new_arithmetic_mean_l3123_312309


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3123_312302

/-- The equation from the original problem -/
def equation (x : ℝ) : Prop :=
  1/x + 1/(x + 3) - 1/(x + 6) - 1/(x + 9) - 1/(x + 12) - 1/(x + 15) + 1/(x + 18) + 1/(x + 21) = 0

/-- Definition of the root form -/
def root_form (a b c d : ℝ) (x : ℝ) : Prop :=
  (x = -a + Real.sqrt (b + c * Real.sqrt d)) ∨ (x = -a + Real.sqrt (b - c * Real.sqrt d)) ∨
  (x = -a - Real.sqrt (b + c * Real.sqrt d)) ∨ (x = -a - Real.sqrt (b - c * Real.sqrt d))

/-- d is not divisible by the square of a prime -/
def not_divisible_by_prime_square (d : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ d)

theorem root_sum_theorem (a b c : ℝ) (d : ℕ) :
  (∃ x : ℝ, equation x ∧ root_form a b c (d : ℝ) x) →
  not_divisible_by_prime_square d →
  a + b + c + d = 57.5 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3123_312302


namespace NUMINAMATH_CALUDE_gauss_polynomial_reciprocal_l3123_312347

/-- Definition of a Gauss polynomial -/
def is_gauss_polynomial (g : ℤ → ℤ → (ℝ → ℝ)) : Prop :=
  ∀ (k l : ℤ) (x : ℝ), x ≠ 0 → x^(k*l) * g k l (1/x) = g k l x

/-- Theorem: Gauss polynomials are reciprocal -/
theorem gauss_polynomial_reciprocal (g : ℤ → ℤ → (ℝ → ℝ)) (h : is_gauss_polynomial g) :
  ∀ (k l : ℤ) (x : ℝ), x ≠ 0 → x^(k*l) * g k l (1/x) = g k l x :=
sorry

end NUMINAMATH_CALUDE_gauss_polynomial_reciprocal_l3123_312347


namespace NUMINAMATH_CALUDE_jia_candies_theorem_l3123_312377

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_parallel_pairs : ℕ

/-- Calculates the number of intersections for a given number of lines -/
def num_intersections (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Calculates the total number of candies for a given line configuration -/
def total_candies (config : LineConfiguration) : ℕ :=
  num_intersections config.num_lines + config.num_parallel_pairs

/-- Theorem: Given 5 lines with one parallel pair, Jia receives 11 candies -/
theorem jia_candies_theorem (config : LineConfiguration) 
  (h1 : config.num_lines = 5)
  (h2 : config.num_parallel_pairs = 1) :
  total_candies config = 11 := by
  sorry

end NUMINAMATH_CALUDE_jia_candies_theorem_l3123_312377


namespace NUMINAMATH_CALUDE_girl_scout_cookies_l3123_312300

theorem girl_scout_cookies (boxes_per_case : ℕ) (boxes_sold : ℕ) (unpacked_boxes : ℕ) :
  boxes_per_case = 12 →
  boxes_sold > 0 →
  unpacked_boxes = 7 →
  ∃ n : ℕ, boxes_sold = 12 * n + 7 :=
by sorry

end NUMINAMATH_CALUDE_girl_scout_cookies_l3123_312300


namespace NUMINAMATH_CALUDE_rectangle_area_l3123_312310

/-- The area of a rectangle formed by three identical smaller rectangles -/
theorem rectangle_area (shorter_side : ℝ) (h : shorter_side = 7) : 
  let longer_side := 3 * shorter_side
  let large_rectangle_length := 3 * shorter_side
  let large_rectangle_width := longer_side
  large_rectangle_length * large_rectangle_width = 441 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3123_312310


namespace NUMINAMATH_CALUDE_translation_theorem_l3123_312357

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point left by a given amount -/
def translateLeft (p : Point) (units : ℝ) : Point :=
  { x := p.x - units, y := p.y }

/-- Translates a point down by a given amount -/
def translateDown (p : Point) (units : ℝ) : Point :=
  { x := p.x, y := p.y - units }

theorem translation_theorem :
  let M : Point := { x := 5, y := 2 }
  let M' : Point := translateDown (translateLeft M 3) 2
  M'.x = 2 ∧ M'.y = 0 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l3123_312357


namespace NUMINAMATH_CALUDE_officer_average_salary_l3123_312301

/-- Proves that the average salary of officers is 440 Rs/month -/
theorem officer_average_salary
  (total_avg : ℝ)
  (non_officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h_total_avg : total_avg = 120)
  (h_non_officer_avg : non_officer_avg = 110)
  (h_officer_count : officer_count = 15)
  (h_non_officer_count : non_officer_count = 480) :
  (total_avg * (officer_count + non_officer_count) - non_officer_avg * non_officer_count) / officer_count = 440 :=
by sorry

end NUMINAMATH_CALUDE_officer_average_salary_l3123_312301


namespace NUMINAMATH_CALUDE_root_implies_range_m_l3123_312343

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem root_implies_range_m :
  ∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x = m) → m ∈ Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_root_implies_range_m_l3123_312343


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3123_312375

theorem polynomial_factorization (m n : ℝ) : 
  (∀ x, x^2 + m*x + n = (x+1)*(x+3)) → m - n = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3123_312375


namespace NUMINAMATH_CALUDE_taco_truck_problem_l3123_312381

/-- The price of a hard shell taco, given the conditions of the taco truck problem -/
def hard_shell_taco_price : ℝ := 5

theorem taco_truck_problem :
  let soft_taco_price : ℝ := 2
  let family_hard_tacos : ℕ := 4
  let family_soft_tacos : ℕ := 3
  let other_customers : ℕ := 10
  let other_customer_soft_tacos : ℕ := 2
  let total_earnings : ℝ := 66

  family_hard_tacos * hard_shell_taco_price +
  family_soft_tacos * soft_taco_price +
  other_customers * other_customer_soft_tacos * soft_taco_price = total_earnings :=
by
  sorry

#eval hard_shell_taco_price

end NUMINAMATH_CALUDE_taco_truck_problem_l3123_312381


namespace NUMINAMATH_CALUDE_f_monotone_iff_f_greater_than_2x_l3123_312392

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 + Real.log ((x / a) + 1)

-- Theorem 1: Monotonicity condition
theorem f_monotone_iff (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 0, Monotone (f a)) ↔ a ∈ Set.Iic (1 - Real.exp 1) ∪ Set.Ici 1 :=
sorry

-- Theorem 2: Inequality for specific a and x
theorem f_greater_than_2x (a : ℝ) (x : ℝ) (ha : a ∈ Set.Ioo 0 1) (hx : x > 0) :
  f a x > 2 * x :=
sorry

end NUMINAMATH_CALUDE_f_monotone_iff_f_greater_than_2x_l3123_312392


namespace NUMINAMATH_CALUDE_homework_problems_left_l3123_312356

theorem homework_problems_left (math_problems science_problems finished_problems : ℕ) 
  (h1 : math_problems = 46)
  (h2 : science_problems = 9)
  (h3 : finished_problems = 40) :
  math_problems + science_problems - finished_problems = 15 :=
by sorry

end NUMINAMATH_CALUDE_homework_problems_left_l3123_312356


namespace NUMINAMATH_CALUDE_jack_marathon_time_l3123_312376

/-- Proves that Jack's marathon time is 5 hours given the specified conditions -/
theorem jack_marathon_time
  (marathon_distance : ℝ)
  (jill_time : ℝ)
  (speed_ratio : ℝ)
  (h1 : marathon_distance = 42)
  (h2 : jill_time = 4.2)
  (h3 : speed_ratio = 0.8400000000000001)
  : ℝ :=
by
  sorry

#check jack_marathon_time

end NUMINAMATH_CALUDE_jack_marathon_time_l3123_312376


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l3123_312326

/-- Given a cylinder whose radius is increased by a factor x and height is doubled,
    resulting in a volume 18 times the original, prove that x = 3 -/
theorem cylinder_volume_increase (r h x : ℝ) (hr : r > 0) (hh : h > 0) (hx : x > 0) :
  2 * x^2 * (π * r^2 * h) = 18 * (π * r^2 * h) → x = 3 := by
  sorry

#check cylinder_volume_increase

end NUMINAMATH_CALUDE_cylinder_volume_increase_l3123_312326


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_six_squared_l3123_312319

theorem gcd_factorial_eight_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_six_squared_l3123_312319


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_of_100_factorial_l3123_312305

-- Define 100!
def factorial_100 : ℕ := Nat.factorial 100

-- Define the function to get the last two nonzero digits
def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

-- Theorem statement
theorem last_two_nonzero_digits_of_100_factorial :
  last_two_nonzero_digits (factorial_100 / (10^24)) = 76 := by
  sorry

#eval last_two_nonzero_digits (factorial_100 / (10^24))

end NUMINAMATH_CALUDE_last_two_nonzero_digits_of_100_factorial_l3123_312305


namespace NUMINAMATH_CALUDE_arithmetic_square_root_l3123_312358

theorem arithmetic_square_root (n : ℝ) (h1 : n > 0) 
  (h2 : ∃ x : ℝ, (x + 1)^2 = n ∧ (2*x - 4)^2 = n) : 
  Real.sqrt n = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_l3123_312358


namespace NUMINAMATH_CALUDE_f_plus_three_odd_l3123_312350

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be odd
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_three_odd 
  (h1 : IsOdd (fun x ↦ f (x + 1)))
  (h2 : IsOdd (fun x ↦ f (x - 1))) :
  IsOdd (fun x ↦ f (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_f_plus_three_odd_l3123_312350


namespace NUMINAMATH_CALUDE_packaging_methods_different_boxes_l3123_312393

theorem packaging_methods_different_boxes (n : ℕ) (m : ℕ) :
  n > 0 → m > 0 → (number_of_packaging_methods : ℕ) = m^n :=
by sorry

end NUMINAMATH_CALUDE_packaging_methods_different_boxes_l3123_312393


namespace NUMINAMATH_CALUDE_polygon_properties_l3123_312332

-- Define the polygon
structure Polygon where
  n : ℕ  -- number of sides
  h : n > 2  -- a polygon must have at least 3 sides

-- Define the ratio of interior to exterior angles
def interiorToExteriorRatio (p : Polygon) : ℚ :=
  (p.n - 2) / 2

-- Theorem statement
theorem polygon_properties (p : Polygon) 
  (h : interiorToExteriorRatio p = 13 / 2) : 
  p.n = 15 ∧ (p.n * (p.n - 3)) / 2 = 90 := by
  sorry


end NUMINAMATH_CALUDE_polygon_properties_l3123_312332


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3123_312371

theorem inequality_and_equality_condition (a b : ℝ) : 
  (a^2 + 4*b^2 + 4*b - 4*a + 5 ≥ 0) ∧ 
  (a^2 + 4*b^2 + 4*b - 4*a + 5 = 0 ↔ a = 2 ∧ b = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3123_312371


namespace NUMINAMATH_CALUDE_more_apples_than_pears_l3123_312339

theorem more_apples_than_pears :
  let total_fruits : ℕ := 85
  let num_apples : ℕ := 48
  let num_pears : ℕ := total_fruits - num_apples
  num_apples - num_pears = 11 :=
by sorry

end NUMINAMATH_CALUDE_more_apples_than_pears_l3123_312339


namespace NUMINAMATH_CALUDE_students_not_enrolled_l3123_312394

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : french = 65)
  (h3 : german = 50)
  (h4 : both = 25) :
  total - (french + german - both) = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l3123_312394


namespace NUMINAMATH_CALUDE_sin_sum_product_l3123_312355

theorem sin_sum_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (9 * x) = 2 * Real.sin (6 * x) * Real.cos (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_l3123_312355


namespace NUMINAMATH_CALUDE_plane_from_three_points_l3123_312311

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Three points are non-collinear if they do not lie on the same line -/
def NonCollinear (p1 p2 p3 : Point3D) : Prop :=
  ¬∃ (t : ℝ), p3 = Point3D.mk (p1.x + t * (p2.x - p1.x)) (p1.y + t * (p2.y - p1.y)) (p1.z + t * (p2.z - p1.z))

/-- A plane is uniquely determined by three non-collinear points -/
theorem plane_from_three_points (p1 p2 p3 : Point3D) (h : NonCollinear p1 p2 p3) :
  ∃! (plane : Plane3D), (plane.a * p1.x + plane.b * p1.y + plane.c * p1.z + plane.d = 0) ∧
                        (plane.a * p2.x + plane.b * p2.y + plane.c * p2.z + plane.d = 0) ∧
                        (plane.a * p3.x + plane.b * p3.y + plane.c * p3.z + plane.d = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_from_three_points_l3123_312311


namespace NUMINAMATH_CALUDE_smallest_integer_y_l3123_312368

theorem smallest_integer_y : ∃ y : ℤ, (1 : ℚ) / 4 < (y : ℚ) / 7 ∧ (y : ℚ) / 7 < 2 / 3 ∧ ∀ z : ℤ, (1 : ℚ) / 4 < (z : ℚ) / 7 ∧ (z : ℚ) / 7 < 2 / 3 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l3123_312368


namespace NUMINAMATH_CALUDE_paintings_on_last_page_paintings_on_last_page_zero_l3123_312341

theorem paintings_on_last_page (initial_albums : Nat) (pages_per_album : Nat) 
  (initial_paintings_per_page : Nat) (new_paintings_per_page : Nat) 
  (filled_albums : Nat) (filled_pages_last_album : Nat) : Nat :=
  let total_paintings := initial_albums * pages_per_album * initial_paintings_per_page
  let total_pages_filled := filled_albums * pages_per_album + filled_pages_last_album
  total_paintings - (total_pages_filled * new_paintings_per_page)

theorem paintings_on_last_page_zero : 
  paintings_on_last_page 10 36 8 9 6 28 = 0 := by
  sorry

end NUMINAMATH_CALUDE_paintings_on_last_page_paintings_on_last_page_zero_l3123_312341


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l3123_312337

theorem consecutive_numbers_sum (n : ℕ) : 
  n + (n + 1) + (n + 2) = 60 → (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l3123_312337


namespace NUMINAMATH_CALUDE_quadratic_roots_and_equation_l3123_312313

theorem quadratic_roots_and_equation (x₁ x₂ a : ℝ) : 
  (x₁^2 + 4*x₁ - 3 = 0) →
  (x₂^2 + 4*x₂ - 3 = 0) →
  (2*x₁*(x₂^2 + 3*x₂ - 3) + a = 2) →
  (a = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_equation_l3123_312313


namespace NUMINAMATH_CALUDE_max_value_theorem_l3123_312340

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 81/4 ∧
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧
    (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 81/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3123_312340


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3123_312344

theorem matrix_equation_solution : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N^3 - 3 * N^2 + 4 * N = !![8, 16; 4, 8] :=
by
  -- Define the matrix N
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  
  -- Assert that N satisfies the equation
  have h : N^3 - 3 * N^2 + 4 * N = !![8, 16; 4, 8] := by sorry
  
  -- Prove existence
  exact ⟨N, h⟩

#check matrix_equation_solution

end NUMINAMATH_CALUDE_matrix_equation_solution_l3123_312344


namespace NUMINAMATH_CALUDE_third_month_sale_l3123_312333

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def sixth_month_sale : ℕ := 4791
def first_month_sale : ℕ := 6635
def second_month_sale : ℕ := 6927
def fourth_month_sale : ℕ := 7230
def fifth_month_sale : ℕ := 6562

theorem third_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := first_month_sale + second_month_sale + fourth_month_sale + fifth_month_sale + sixth_month_sale
  total_sales - known_sales = 14085 := by
sorry

end NUMINAMATH_CALUDE_third_month_sale_l3123_312333


namespace NUMINAMATH_CALUDE_two_year_increase_l3123_312349

def yearly_increase (amount : ℚ) : ℚ := amount * (1 + 1/8)

theorem two_year_increase (P : ℚ) (h : P = 2880) : 
  yearly_increase (yearly_increase P) = 3645 := by
  sorry

end NUMINAMATH_CALUDE_two_year_increase_l3123_312349


namespace NUMINAMATH_CALUDE_evaluate_expression_l3123_312329

theorem evaluate_expression (a : ℝ) (h : a = 3) : (5 * a^2 - 11 * a + 6) * (2 * a - 4) = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3123_312329


namespace NUMINAMATH_CALUDE_combined_earnings_proof_l3123_312304

/-- Given Dwayne's annual earnings and the difference between Brady's and Dwayne's earnings,
    calculate their combined annual earnings. -/
def combinedEarnings (dwayneEarnings : ℕ) (earningsDifference : ℕ) : ℕ :=
  dwayneEarnings + (dwayneEarnings + earningsDifference)

/-- Theorem stating that given the specific values from the problem,
    the combined earnings of Brady and Dwayne are $3450. -/
theorem combined_earnings_proof :
  combinedEarnings 1500 450 = 3450 := by
  sorry

end NUMINAMATH_CALUDE_combined_earnings_proof_l3123_312304


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3123_312374

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : ways_to_put_balls_in_boxes 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3123_312374


namespace NUMINAMATH_CALUDE_jason_additional_manager_months_l3123_312324

/-- Calculates the additional months Jason worked as a manager -/
def additional_manager_months (bartender_years : ℕ) (manager_years : ℕ) (total_months : ℕ) : ℕ :=
  total_months - (bartender_years * 12 + manager_years * 12)

/-- Proves that Jason worked 6 additional months as a manager -/
theorem jason_additional_manager_months :
  additional_manager_months 9 3 150 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jason_additional_manager_months_l3123_312324


namespace NUMINAMATH_CALUDE_horner_v₁_is_8_l3123_312397

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 4x^5 - 12x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 4 * x^5 - 12 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- The coefficients of the polynomial f in descending order of degree -/
def f_coeffs : List ℝ := [4, -12, 3.5, -2.6, 1.7, -0.8]

/-- The value of x at which we evaluate the polynomial -/
def x : ℝ := 5

/-- v₁ in Horner's method for f(x) when x = 5 -/
def v₁ : ℝ := 4 * x - 12

theorem horner_v₁_is_8 : v₁ = 8 := by sorry

end NUMINAMATH_CALUDE_horner_v₁_is_8_l3123_312397


namespace NUMINAMATH_CALUDE_odot_inequality_range_l3123_312389

-- Define the ⊙ operation
def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem odot_inequality_range :
  ∀ x : ℝ, odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_odot_inequality_range_l3123_312389


namespace NUMINAMATH_CALUDE_repeating_decimal_85_l3123_312359

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_85 :
  RepeatingDecimal 8 5 = 85 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_85_l3123_312359


namespace NUMINAMATH_CALUDE_caravan_camels_l3123_312342

theorem caravan_camels (hens goats keepers : ℕ) (camel_feet : ℕ) : 
  hens = 50 → 
  goats = 45 → 
  keepers = 15 → 
  camel_feet = (hens + goats + keepers + 224) * 2 - (hens * 2 + goats * 4 + keepers * 2) → 
  camel_feet / 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_caravan_camels_l3123_312342


namespace NUMINAMATH_CALUDE_smallest_number_l3123_312325

def binary_to_decimal (n : ℕ) : ℕ := n

def base_6_to_decimal (n : ℕ) : ℕ := n

def base_4_to_decimal (n : ℕ) : ℕ := n

def base_9_to_decimal (n : ℕ) : ℕ := n

theorem smallest_number :
  let a := binary_to_decimal 111111
  let b := base_6_to_decimal 210
  let c := base_4_to_decimal 1000
  let d := base_9_to_decimal 81
  a < b ∧ a < c ∧ a < d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3123_312325


namespace NUMINAMATH_CALUDE_cube_volume_l3123_312370

/-- The volume of a cube with total edge length of 60 cm is 125 cubic centimeters. -/
theorem cube_volume (total_edge_length : ℝ) (h : total_edge_length = 60) : 
  (total_edge_length / 12)^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l3123_312370


namespace NUMINAMATH_CALUDE_ellipse_constant_slope_l3123_312364

/-- An ellipse with the given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_point : 4/a^2 + 1/b^2 = 1

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2/E.a^2 + y^2/E.b^2 = 1

/-- The theorem statement -/
theorem ellipse_constant_slope (E : Ellipse) 
  (h_bisector : ∀ (P Q : PointOnEllipse E), 
    (∃ (k : ℝ), k * (P.x - 2) = P.y - 1 ∧ k * (Q.x - 2) = -(Q.y - 1))) :
  ∀ (P Q : PointOnEllipse E), (Q.y - P.y) / (Q.x - P.x) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_constant_slope_l3123_312364


namespace NUMINAMATH_CALUDE_expo_assignment_count_l3123_312308

/-- Represents the four pavilions at the Shanghai World Expo -/
inductive Pavilion
  | China
  | UK
  | Australia
  | Russia

/-- The total number of volunteers -/
def total_volunteers : Nat := 5

/-- The number of pavilions -/
def num_pavilions : Nat := 4

/-- A function that represents a valid assignment of volunteers to pavilions -/
def is_valid_assignment (assignment : Pavilion → Nat) : Prop :=
  (∀ p : Pavilion, assignment p > 0) ∧
  (assignment Pavilion.China + assignment Pavilion.UK + 
   assignment Pavilion.Australia + assignment Pavilion.Russia = total_volunteers)

/-- The number of ways for A and B to be assigned to pavilions -/
def num_ways_AB : Nat := num_pavilions * num_pavilions

/-- The theorem to be proved -/
theorem expo_assignment_count :
  (∃ (ways : Nat), ways = num_ways_AB ∧
    ∃ (remaining_assignments : Nat),
      ways * remaining_assignments = 72 ∧
      ∀ (assignment : Pavilion → Nat),
        is_valid_assignment assignment →
        remaining_assignments > 0) := by
  sorry

end NUMINAMATH_CALUDE_expo_assignment_count_l3123_312308


namespace NUMINAMATH_CALUDE_set_intersection_problem_l3123_312303

theorem set_intersection_problem (m : ℝ) : 
  let A : Set ℝ := {-1, 3, m}
  let B : Set ℝ := {3, 4}
  B ∩ A = B → m = 4 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l3123_312303


namespace NUMINAMATH_CALUDE_proportional_sum_ratio_l3123_312372

theorem proportional_sum_ratio (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4 ∧ z / 4 ≠ 0) : 
  (2 * x + 3 * y) / z = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_proportional_sum_ratio_l3123_312372


namespace NUMINAMATH_CALUDE_latoya_card_credit_l3123_312306

/-- Calculates the remaining credit on a prepaid phone card after a call -/
def remaining_credit (initial_value : ℚ) (cost_per_minute : ℚ) (call_duration : ℕ) : ℚ :=
  initial_value - (cost_per_minute * call_duration)

/-- Theorem stating the remaining credit on Latoya's prepaid phone card -/
theorem latoya_card_credit :
  let initial_value : ℚ := 30
  let cost_per_minute : ℚ := 16 / 100
  let call_duration : ℕ := 22
  remaining_credit initial_value cost_per_minute call_duration = 2648 / 100 := by
sorry

end NUMINAMATH_CALUDE_latoya_card_credit_l3123_312306


namespace NUMINAMATH_CALUDE_equation_equivalent_to_lines_l3123_312386

/-- The set of points satisfying the given equation is equivalent to the union of two lines -/
theorem equation_equivalent_to_lines :
  ∀ x y : ℝ, 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_lines_l3123_312386


namespace NUMINAMATH_CALUDE_football_season_games_l3123_312367

/-- Calculates the total number of football games in a season -/
def total_games (months : ℕ) (games_per_month : ℕ) : ℕ :=
  months * games_per_month

theorem football_season_games :
  let season_length : ℕ := 17
  let games_per_month : ℕ := 19
  total_games season_length games_per_month = 323 := by
sorry

end NUMINAMATH_CALUDE_football_season_games_l3123_312367


namespace NUMINAMATH_CALUDE_largest_three_digit_base6_l3123_312380

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (d1 d2 d3 : Nat) : Nat :=
  d1 * 6^2 + d2 * 6^1 + d3 * 6^0

/-- The largest digit in base-6 --/
def maxBase6Digit : Nat := 5

theorem largest_three_digit_base6 :
  base6ToBase10 maxBase6Digit maxBase6Digit maxBase6Digit = 215 := by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_base6_l3123_312380


namespace NUMINAMATH_CALUDE_tic_tac_toe_ties_l3123_312378

theorem tic_tac_toe_ties (james_win_rate mary_win_rate : ℚ)
  (h1 : james_win_rate = 4 / 9)
  (h2 : mary_win_rate = 5 / 18) :
  1 - (james_win_rate + mary_win_rate) = 5 / 18 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_ties_l3123_312378


namespace NUMINAMATH_CALUDE_s_range_l3123_312360

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem s_range : Set.range s = {y : ℝ | y < 0 ∨ y > 0} := by sorry

end NUMINAMATH_CALUDE_s_range_l3123_312360


namespace NUMINAMATH_CALUDE_positive_solution_square_root_form_l3123_312348

theorem positive_solution_square_root_form :
  ∃ (a' b' : ℕ+), 
    (∃ (x : ℝ), x^2 + 14*x = 96 ∧ x > 0 ∧ x = Real.sqrt a' - b') ∧
    a' = 145 ∧ 
    b' = 7 ∧
    (a' : ℕ) + (b' : ℕ) = 152 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_square_root_form_l3123_312348


namespace NUMINAMATH_CALUDE_max_r_value_l3123_312320

theorem max_r_value (p q r : ℝ) (sum_eq : p + q + r = 6) (prod_sum_eq : p * q + p * r + q * r = 8) :
  r ≤ 2 + Real.sqrt (20 / 3) := by
  sorry

end NUMINAMATH_CALUDE_max_r_value_l3123_312320


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_not_sufficient_l3123_312365

theorem quadratic_inequality_necessary_not_sufficient :
  (∃ x : ℝ, (|x - 2| < 1 ∧ ¬(x^2 - 5*x + 4 < 0))) ∧
  (∀ x : ℝ, (x^2 - 5*x + 4 < 0 → |x - 2| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_not_sufficient_l3123_312365


namespace NUMINAMATH_CALUDE_y_derivative_l3123_312328

open Real

noncomputable def y (x : ℝ) : ℝ :=
  (6^x * (sin (4*x) * log 6 - 4 * cos (4*x))) / (16 + (log 6)^2)

theorem y_derivative (x : ℝ) : 
  deriv y x = 6^x * sin (4*x) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l3123_312328


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3123_312387

theorem polynomial_multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3123_312387


namespace NUMINAMATH_CALUDE_decimal_to_binary_15_l3123_312382

theorem decimal_to_binary_15 : (15 : ℕ) = 0b1111 := by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_15_l3123_312382


namespace NUMINAMATH_CALUDE_remaining_digits_average_l3123_312385

theorem remaining_digits_average (digits : Finset ℕ) (subset : Finset ℕ) :
  Finset.card digits = 9 →
  (Finset.sum digits id) / 9 = 18 →
  Finset.card subset = 4 →
  subset ⊆ digits →
  (Finset.sum subset id) / 4 = 8 →
  let remaining := digits \ subset
  ((Finset.sum remaining id) / (Finset.card remaining) : ℚ) = 26 := by
sorry

end NUMINAMATH_CALUDE_remaining_digits_average_l3123_312385


namespace NUMINAMATH_CALUDE_relay_race_time_difference_l3123_312383

def apple_distance : ℝ := 24
def apple_speed : ℝ := 3
def mac_distance : ℝ := 28
def mac_speed : ℝ := 4
def orange_distance : ℝ := 32
def orange_speed : ℝ := 5

def minutes_per_hour : ℝ := 60

theorem relay_race_time_difference :
  (apple_distance / apple_speed + mac_distance / mac_speed) * minutes_per_hour -
  (orange_distance / orange_speed * minutes_per_hour) = 516 := by
sorry

end NUMINAMATH_CALUDE_relay_race_time_difference_l3123_312383


namespace NUMINAMATH_CALUDE_platform_length_calculation_l3123_312327

/-- Calculates the length of a platform given train parameters --/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : 
  train_length = 180 ∧ 
  train_speed_kmph = 72 ∧ 
  crossing_time = 20 →
  (train_speed_kmph * 1000 / 3600) * crossing_time - train_length = 220 := by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l3123_312327


namespace NUMINAMATH_CALUDE_gcf_72_108_l3123_312398

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_108_l3123_312398


namespace NUMINAMATH_CALUDE_sixteen_students_not_liking_sports_l3123_312390

/-- The number of students who do not like basketball, cricket, or football -/
def students_not_liking_sports (total : ℕ) (basketball cricket football : ℕ) 
  (basketball_cricket cricket_football basketball_football : ℕ) (all_three : ℕ) : ℕ :=
  total - (basketball + cricket + football - basketball_cricket - cricket_football - basketball_football + all_three)

/-- Theorem stating that 16 students do not like any of the three sports -/
theorem sixteen_students_not_liking_sports : 
  students_not_liking_sports 50 20 18 12 8 6 5 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_students_not_liking_sports_l3123_312390


namespace NUMINAMATH_CALUDE_simplify_powers_l3123_312353

theorem simplify_powers (a : ℝ) : (a^5 * a^3) * (a^2)^4 = a^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_powers_l3123_312353


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3123_312338

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3123_312338


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3123_312362

/-- Condition p: 0 < x < 2 -/
def p (x : ℝ) : Prop := 0 < x ∧ x < 2

/-- Condition q: -1 < x < 3 -/
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

/-- p is sufficient but not necessary for q -/
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3123_312362


namespace NUMINAMATH_CALUDE_hoseok_multiplication_l3123_312318

theorem hoseok_multiplication (x : ℚ) : x / 11 = 2 → 6 * x = 132 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_multiplication_l3123_312318


namespace NUMINAMATH_CALUDE_mary_chestnut_pick_l3123_312321

/-- Given three people picking chestnuts with specific relationships between their picks,
    prove that one person picked a certain amount. -/
theorem mary_chestnut_pick (peter lucy mary : ℝ) 
  (h1 : mary = 2 * peter)
  (h2 : lucy = peter + 2)
  (h3 : peter + mary + lucy = 26) :
  mary = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_chestnut_pick_l3123_312321


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3123_312322

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def nonagon_sides : ℕ := 9

theorem nonagon_diagonals : num_diagonals nonagon_sides = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3123_312322


namespace NUMINAMATH_CALUDE_log_sqrt12_1728sqrt12_l3123_312334

theorem log_sqrt12_1728sqrt12 : Real.log (1728 * Real.sqrt 12) / Real.log (Real.sqrt 12) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt12_1728sqrt12_l3123_312334


namespace NUMINAMATH_CALUDE_train_crossing_time_l3123_312312

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 350 →
  train_speed_kmh = 60 →
  crossing_time = 21 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3123_312312


namespace NUMINAMATH_CALUDE_ellipse_and_line_equations_l3123_312366

noncomputable section

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal distance
def c : ℝ := Real.sqrt 3

-- Define the perimeter of triangle MF₁F₂
def triangle_perimeter : ℝ := 4 + 2 * Real.sqrt 3

-- Define point P
def P : ℝ × ℝ := (0, 2)

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem ellipse_and_line_equations :
  ∃ (a b : ℝ),
    -- Conditions
    ellipse_C a b (-c) 0 ∧
    (∀ x y, ellipse_C a b x y → 
      ∃ (m : ℝ × ℝ), Real.sqrt ((x - (-c))^2 + y^2) + Real.sqrt ((x - c)^2 + y^2) = triangle_perimeter) ∧
    -- Conclusions
    (a = 2 ∧ b = 1) ∧
    (∃ (k : ℝ), k = 2 ∨ k = -2) ∧
    (∀ k, k = 2 ∨ k = -2 →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        ellipse_C 2 1 x₁ y₁ ∧
        ellipse_C 2 1 x₂ y₂ ∧
        y₁ = k * x₁ - 2 ∧
        y₂ = k * x₂ - 2 ∧
        perpendicular x₁ y₁ x₂ y₂) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_equations_l3123_312366
