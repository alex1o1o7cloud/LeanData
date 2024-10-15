import Mathlib

namespace NUMINAMATH_CALUDE_gcd_sum_fraction_eq_half_iff_special_triples_l1582_158222

/-- Given positive integers a, b, c satisfying a < b < c, prove that
    (a.gcd b + b.gcd c + c.gcd a) / (a + b + c) = 1/2
    if and only if there exists a positive integer d such that
    (a, b, c) = (d, 2*d, 3*d) or (a, b, c) = (d, 3*d, 6*d) -/
theorem gcd_sum_fraction_eq_half_iff_special_triples
  (a b c : ℕ+) (h1 : a < b) (h2 : b < c) :
  (a.gcd b + b.gcd c + c.gcd a : ℚ) / (a + b + c) = 1/2 ↔
  (∃ d : ℕ+, (a, b, c) = (d, 2*d, 3*d) ∨ (a, b, c) = (d, 3*d, 6*d)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_fraction_eq_half_iff_special_triples_l1582_158222


namespace NUMINAMATH_CALUDE_cubic_increasing_iff_a_positive_l1582_158277

/-- A cubic function f(x) = ax³ + x is increasing on ℝ if and only if a > 0 -/
theorem cubic_increasing_iff_a_positive (a : ℝ) :
  (∀ x : ℝ, StrictMono (fun x => a * x^3 + x)) ↔ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_increasing_iff_a_positive_l1582_158277


namespace NUMINAMATH_CALUDE_g_at_3_equals_20_l1582_158233

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 5)

noncomputable def g (x : ℝ) : ℝ := 4 * (f⁻¹ x)

theorem g_at_3_equals_20 : g 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_g_at_3_equals_20_l1582_158233


namespace NUMINAMATH_CALUDE_cart_distance_theorem_l1582_158243

def cart_distance (initial_distance : ℕ) (first_increment : ℕ) (second_increment : ℕ) (total_time : ℕ) : ℕ :=
  let first_section := (total_time / 2) * (2 * initial_distance + (total_time / 2 - 1) * first_increment) / 2
  let final_first_speed := initial_distance + (total_time / 2 - 1) * first_increment
  let second_section := (total_time / 2) * (2 * final_first_speed + (total_time / 2 - 1) * second_increment) / 2
  first_section + second_section

theorem cart_distance_theorem :
  cart_distance 8 10 6 30 = 4020 := by
  sorry

end NUMINAMATH_CALUDE_cart_distance_theorem_l1582_158243


namespace NUMINAMATH_CALUDE_cab_driver_income_l1582_158228

theorem cab_driver_income (day1 day3 day4 day5 average : ℕ) 
  (h1 : day1 = 300)
  (h2 : day3 = 750)
  (h3 : day4 = 200)
  (h4 : day5 = 600)
  (h5 : average = 400)
  (h6 : (day1 + day3 + day4 + day5 + (5 * average - (day1 + day3 + day4 + day5))) / 5 = average) :
  5 * average - (day1 + day3 + day4 + day5) = 150 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1582_158228


namespace NUMINAMATH_CALUDE_min_value_expression_l1582_158229

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (y/x) + (1/y) ≥ 4 ∧ ((y/x) + (1/y) = 4 ↔ x = 1/3 ∧ y = 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1582_158229


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l1582_158202

noncomputable def f (x : ℝ) := x^2 - Real.log x

def line (x : ℝ) := x - 2

theorem min_distance_curve_to_line :
  ∀ x > 0, ∃ d : ℝ,
    d = Real.sqrt 2 ∧
    ∀ y > 0, 
      let p₁ := (x, f x)
      let p₂ := (y, line y)
      d ≤ Real.sqrt ((x - y)^2 + (f x - line y)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l1582_158202


namespace NUMINAMATH_CALUDE_arnel_pencil_boxes_l1582_158263

/-- The number of boxes of pencils Arnel had -/
def number_of_boxes : ℕ := sorry

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 5

/-- The number of pencils Arnel kept for himself -/
def pencils_kept : ℕ := 10

/-- The number of Arnel's friends -/
def number_of_friends : ℕ := 5

/-- The number of pencils each friend received -/
def pencils_per_friend : ℕ := 8

theorem arnel_pencil_boxes :
  number_of_boxes = 10 ∧
  number_of_boxes * pencils_per_box = 
    pencils_kept + number_of_friends * pencils_per_friend :=
by sorry

end NUMINAMATH_CALUDE_arnel_pencil_boxes_l1582_158263


namespace NUMINAMATH_CALUDE_lastDigitOf2Power2023_l1582_158287

-- Define the pattern of last digits for powers of 2
def lastDigitPattern : Fin 4 → Nat
  | 0 => 2
  | 1 => 4
  | 2 => 8
  | 3 => 6

-- Define the function to get the last digit of 2^n
def lastDigitOfPowerOf2 (n : Nat) : Nat :=
  lastDigitPattern ((n - 1) % 4)

-- Theorem statement
theorem lastDigitOf2Power2023 : lastDigitOfPowerOf2 2023 = 8 := by
  sorry

end NUMINAMATH_CALUDE_lastDigitOf2Power2023_l1582_158287


namespace NUMINAMATH_CALUDE_zoo_field_trip_zoo_field_trip_result_l1582_158291

/-- Calculates the number of individuals left at the zoo after a field trip -/
theorem zoo_field_trip (students_per_class : ℕ) (num_classes : ℕ) (parent_chaperones : ℕ) 
  (teachers : ℕ) (students_left : ℕ) (chaperones_left : ℕ) : ℕ :=
  let initial_total := students_per_class * num_classes + parent_chaperones + teachers
  let left_total := students_left + chaperones_left
  initial_total - left_total

/-- Proves that the number of individuals left at the zoo is 15 -/
theorem zoo_field_trip_result : 
  zoo_field_trip 10 2 5 2 10 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_zoo_field_trip_zoo_field_trip_result_l1582_158291


namespace NUMINAMATH_CALUDE_enclosing_polygon_sides_l1582_158257

/-- The number of sides of the regular polygon that exactly encloses a regular decagon --/
def n : ℕ := 5

/-- The number of sides of the regular polygon being enclosed (decagon) --/
def m : ℕ := 10

theorem enclosing_polygon_sides :
  (∀ (k : ℕ), k > 2 → (360 : ℝ) / k = (720 : ℝ) / m) → n = 5 := by sorry

end NUMINAMATH_CALUDE_enclosing_polygon_sides_l1582_158257


namespace NUMINAMATH_CALUDE_f_has_unique_minimum_l1582_158204

open Real

-- Define the function f(x) = 2x - ln x
noncomputable def f (x : ℝ) : ℝ := 2 * x - log x

-- Theorem statement
theorem f_has_unique_minimum :
  ∃! (x : ℝ), x > 0 ∧ IsLocalMin f x ∧ f x = 1 + log 2 := by
  sorry

end NUMINAMATH_CALUDE_f_has_unique_minimum_l1582_158204


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_k_range_l1582_158285

theorem quadratic_always_positive_implies_k_range (k : ℝ) :
  (∀ x : ℝ, x^2 + 2*k*x - (k - 2) > 0) → k ∈ Set.Ioo (-2 : ℝ) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_k_range_l1582_158285


namespace NUMINAMATH_CALUDE_fraction_transformation_l1582_158294

theorem fraction_transformation (a b c d x : ℤ) 
  (hb : b ≠ 0) 
  (hcd : c - d ≠ 0) 
  (h_simplest : ∀ k : ℤ, k ∣ c ∧ k ∣ d → k = 1 ∨ k = -1) 
  (h_eq : (2 * a + x) * d = (b - x) * c) : 
  x = (b * c - 2 * a * d) / (d + c) := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1582_158294


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_eq_5_subset_condition_l1582_158240

-- Define the sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2*m - 3}

-- Theorem for part (1)
theorem intersection_and_union_when_m_eq_5 :
  (A ∩ B 5 = A) ∧ (Aᶜ ∪ B 5 = Set.univ) := by sorry

-- Theorem for part (2)
theorem subset_condition :
  ∀ m, A ⊆ B m ↔ m > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_eq_5_subset_condition_l1582_158240


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1582_158231

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |3 * x - 1| + a * x + 3

-- Part 1: Prove the solution set for f(x) ≤ 5 when a = 1
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} := by sorry

-- Part 2: Prove the range of a for which f(x) has a minimum value
theorem range_of_a_part2 :
  ∀ a : ℝ, (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) → -3 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1582_158231


namespace NUMINAMATH_CALUDE_connie_marbles_proof_l1582_158209

/-- The number of marbles Connie had initially -/
def initial_marbles : ℕ := 2856

/-- The number of marbles Connie had after losing half -/
def marbles_after_loss : ℕ := initial_marbles / 2

/-- The number of marbles Connie had after giving away 2/3 of the remaining marbles -/
def final_marbles : ℕ := 476

theorem connie_marbles_proof : 
  initial_marbles = 2856 ∧ 
  marbles_after_loss = initial_marbles / 2 ∧
  final_marbles = marbles_after_loss / 3 ∧
  final_marbles = 476 := by sorry

end NUMINAMATH_CALUDE_connie_marbles_proof_l1582_158209


namespace NUMINAMATH_CALUDE_mikes_file_space_l1582_158258

/-- The amount of space Mike's files take up on his disk drive. -/
def space_taken_by_files (total_space : ℕ) (space_left : ℕ) : ℕ :=
  total_space - space_left

/-- Proof that Mike's files take up 26 GB of space. -/
theorem mikes_file_space :
  space_taken_by_files 28 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_mikes_file_space_l1582_158258


namespace NUMINAMATH_CALUDE_only_two_random_events_l1582_158296

-- Define the events
inductive Event
| SameChargesRepel
| SunnyTomorrow
| FreeFallStraightLine
| ExponentialIncreasing

-- Define a predicate for random events
def IsRandomEvent : Event → Prop :=
  fun e => match e with
  | Event.SunnyTomorrow => True
  | Event.ExponentialIncreasing => True
  | _ => False

-- Theorem statement
theorem only_two_random_events :
  (∀ e : Event, IsRandomEvent e ↔ (e = Event.SunnyTomorrow ∨ e = Event.ExponentialIncreasing)) :=
by sorry

end NUMINAMATH_CALUDE_only_two_random_events_l1582_158296


namespace NUMINAMATH_CALUDE_farm_ratio_l1582_158234

/-- Given a farm with horses and cows, prove that the initial ratio of horses to cows is 4:1 --/
theorem farm_ratio (initial_horses initial_cows : ℕ) : 
  (initial_horses - 15 : ℚ) / (initial_cows + 15 : ℚ) = 7 / 3 →
  (initial_horses - 15) = (initial_cows + 15 + 60) →
  (initial_horses : ℚ) / initial_cows = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_farm_ratio_l1582_158234


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l1582_158280

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = ![![3, -2], ![0, 1]] →
  (B^2)⁻¹ = ![![9, -6], ![0, 1]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l1582_158280


namespace NUMINAMATH_CALUDE_job_completion_time_l1582_158249

/-- 
Given two people who can complete a job independently in 10 and 15 days respectively,
this theorem proves that they can complete the job together in 6 days.
-/
theorem job_completion_time 
  (ram_time : ℝ) 
  (gohul_time : ℝ) 
  (h1 : ram_time = 10) 
  (h2 : gohul_time = 15) : 
  (ram_time * gohul_time) / (ram_time + gohul_time) = 6 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1582_158249


namespace NUMINAMATH_CALUDE_sequence_inequality_l1582_158203

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : a 1 = π / 3)
  (h2 : ∀ n, 0 < a n ∧ a n < π / 3)
  (h3 : ∀ n ≥ 2, Real.sin (a (n + 1)) ≤ (1 / 3) * Real.sin (3 * a n)) :
  ∀ n, Real.sin (a n) < 1 / Real.sqrt n := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1582_158203


namespace NUMINAMATH_CALUDE_vector_on_line_iff_k_eq_half_l1582_158235

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

/-- A line passing through points represented by vectors p and q -/
def line (p q : n) : Set n :=
  {x | ∃ t : ℝ, x = p + t • (q - p)}

/-- The vector that should lie on the line -/
def vector_on_line (p q : n) (k : ℝ) : n :=
  k • p + (1/2) • q

/-- Theorem stating that the vector lies on the line if and only if k = 1/2 -/
theorem vector_on_line_iff_k_eq_half (p q : n) :
  ∀ k : ℝ, vector_on_line p q k ∈ line p q ↔ k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_iff_k_eq_half_l1582_158235


namespace NUMINAMATH_CALUDE_total_cost_train_and_bus_l1582_158251

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := 8.35

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

/-- The difference in cost between a train ride and a bus ride -/
def cost_difference : ℝ := 6.85

theorem total_cost_train_and_bus :
  train_cost + bus_cost = 9.85 ∧
  train_cost = bus_cost + cost_difference :=
sorry

end NUMINAMATH_CALUDE_total_cost_train_and_bus_l1582_158251


namespace NUMINAMATH_CALUDE_intersection_not_empty_l1582_158220

theorem intersection_not_empty : ∃ (n : ℕ) (k : ℕ), n > 1 ∧ 2^n - n = k^2 := by sorry

end NUMINAMATH_CALUDE_intersection_not_empty_l1582_158220


namespace NUMINAMATH_CALUDE_g_at_5_l1582_158200

/-- A function g satisfying the given equation for all real x -/
def g : ℝ → ℝ := sorry

/-- The main property of function g -/
axiom g_property : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 1

/-- The theorem to be proved -/
theorem g_at_5 : g 5 = 3/4 := by sorry

end NUMINAMATH_CALUDE_g_at_5_l1582_158200


namespace NUMINAMATH_CALUDE_employed_females_percentage_l1582_158246

theorem employed_females_percentage
  (total_employed : ℝ)
  (employable_population : ℝ)
  (h1 : total_employed = 1.2 * employable_population)
  (h2 : 0.8 * employable_population = total_employed * (80 / 100)) :
  (total_employed - 0.8 * employable_population) / total_employed = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l1582_158246


namespace NUMINAMATH_CALUDE_smallest_sum_four_consecutive_composites_l1582_158224

def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

def fourConsecutiveComposites (n : ℕ) : Prop :=
  isComposite n ∧ isComposite (n + 1) ∧ isComposite (n + 2) ∧ isComposite (n + 3)

theorem smallest_sum_four_consecutive_composites :
  ∃ n : ℕ, fourConsecutiveComposites n ∧
    (∀ m : ℕ, fourConsecutiveComposites m → n ≤ m) ∧
    n + (n + 1) + (n + 2) + (n + 3) = 102 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_four_consecutive_composites_l1582_158224


namespace NUMINAMATH_CALUDE_function_inequality_l1582_158218

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, x * (deriv (deriv f) x) + f x > 0

theorem function_inequality {f : ℝ → ℝ} (hf : Differentiable ℝ f) 
    (hf' : Differentiable ℝ (deriv f)) (hcond : SatisfiesCondition f) 
    {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) : 
    a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1582_158218


namespace NUMINAMATH_CALUDE_math_pass_count_l1582_158237

/-- Represents the number of students in various categories -/
structure StudentCounts where
  english : ℕ
  math : ℕ
  bothSubjects : ℕ
  onlyEnglish : ℕ
  onlyMath : ℕ

/-- Theorem stating the number of students who pass in Math -/
theorem math_pass_count (s : StudentCounts) 
  (h1 : s.english = 30)
  (h2 : s.english = s.onlyEnglish + s.bothSubjects)
  (h3 : s.onlyEnglish = s.onlyMath + 10)
  (h4 : s.math = s.onlyMath + s.bothSubjects) :
  s.math = 20 := by
  sorry

end NUMINAMATH_CALUDE_math_pass_count_l1582_158237


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_one_l1582_158211

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem f_neg_two_eq_neg_one : f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_one_l1582_158211


namespace NUMINAMATH_CALUDE_factor_polynomial_l1582_158252

theorem factor_polynomial (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1582_158252


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l1582_158279

/-- The maximum distance from a point on the circle ρ = 8sinθ to the line θ = π/3 is 6 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 4)^2 = 16}
  let line := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1}
  ∃ (max_dist : ℝ), max_dist = 6 ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      ∀ (q : ℝ × ℝ), q ∈ line →
        Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l1582_158279


namespace NUMINAMATH_CALUDE_odd_periodic_symmetry_ln_quotient_odd_main_theorem_l1582_158223

-- Definition of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Definition of a periodic function
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

-- Theorem 1: Odd function with period 4 is symmetric about (2,0)
theorem odd_periodic_symmetry (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : IsPeriodic f 4) :
  ∀ x, f (2 + x) = f (2 - x) :=
sorry

-- Theorem 2: ln((1+x)/(1-x)) is an odd function on (-1,1)
theorem ln_quotient_odd :
  IsOdd (fun x => Real.log ((1 + x) / (1 - x))) :=
sorry

-- Main theorem combining both results
theorem main_theorem :
  (∃ f : ℝ → ℝ, IsOdd f ∧ IsPeriodic f 4 ∧ (∀ x, f (2 + x) = f (2 - x))) ∧
  IsOdd (fun x => Real.log ((1 + x) / (1 - x))) :=
sorry

end NUMINAMATH_CALUDE_odd_periodic_symmetry_ln_quotient_odd_main_theorem_l1582_158223


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1582_158232

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : x^2 + y^2 + x*y = 1)
  (h2 : y^2 + z^2 + y*z = 2)
  (h3 : x^2 + z^2 + x*z = 3) :
  x + y + z = Real.sqrt (3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1582_158232


namespace NUMINAMATH_CALUDE_ellipse_equation_l1582_158236

theorem ellipse_equation (a b c : ℝ) (h1 : 2 * a = 10) (h2 : c / a = 3 / 5) (h3 : b^2 = a^2 - c^2) :
  ∀ (x y : ℝ), (x^2 / 16 + y^2 / 25 = 1) ↔ (x^2 / b^2 + y^2 / a^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1582_158236


namespace NUMINAMATH_CALUDE_no_valid_n_l1582_158213

theorem no_valid_n : ¬ ∃ (n : ℕ), n > 0 ∧ 
  (3*n - 3 + 2*n + 7 > 4*n + 6) ∧
  (3*n - 3 + 4*n + 6 > 2*n + 7) ∧
  (2*n + 7 + 4*n + 6 > 3*n - 3) ∧
  (2*n + 7 > 4*n + 6) ∧
  (4*n + 6 > 3*n - 3) :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_l1582_158213


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l1582_158270

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The kite formed by the intersections of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- The area of a kite -/
def kite_area (k : Kite) : ℝ := sorry

/-- The theorem to be proved -/
theorem parabola_kite_sum (c d : ℝ) :
  let p1 : Parabola := ⟨c, 3⟩
  let p2 : Parabola := ⟨-d, 7⟩
  let k : Kite := ⟨p1, p2⟩
  kite_area k = 20 → c + d = 18/25 := by sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l1582_158270


namespace NUMINAMATH_CALUDE_faye_team_size_l1582_158250

def team_size (total_points : ℕ) (faye_points : ℕ) (others_points : ℕ) : ℕ :=
  (total_points - faye_points) / others_points + 1

theorem faye_team_size :
  team_size 68 28 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_faye_team_size_l1582_158250


namespace NUMINAMATH_CALUDE_lines_intersection_l1582_158219

theorem lines_intersection (k : ℝ) : 
  ∃ (x y : ℝ), ∀ (k : ℝ), k * x + y + 3 * k + 1 = 0 ∧ x = -3 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_intersection_l1582_158219


namespace NUMINAMATH_CALUDE_tangent_line_circle_range_l1582_158253

theorem tangent_line_circle_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∀ x y : ℝ, x + y + a = 0 → (x - b)^2 + (y - 1)^2 ≥ 2) 
  (h_touch : ∃ x y : ℝ, x + y + a = 0 ∧ (x - b)^2 + (y - 1)^2 = 2) :
  ∀ z : ℝ, z > 0 → ∃ c : ℝ, c > 0 ∧ c < 1 ∧ z = c^2 / (1 - c) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_range_l1582_158253


namespace NUMINAMATH_CALUDE_triangle_problem_l1582_158216

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = 2 →
  c = Real.sqrt 2 →
  Real.cos A = Real.sqrt 2 / 4 →
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c →
  a < b + c ∧ b < a + c ∧ c < a + b →
  -- Conclusions
  Real.sin C = Real.sqrt 7 / 4 ∧
  b = 1 ∧
  Real.cos (2 * A + π / 3) = (-3 + Real.sqrt 21) / 8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1582_158216


namespace NUMINAMATH_CALUDE_exists_valid_square_forming_strategy_l1582_158272

/-- Represents a geometric shape on a graph paper --/
structure Shape :=
  (area : ℝ)
  (is_square : Bool)

/-- Represents a cutting strategy for a shape --/
structure CuttingStrategy :=
  (num_parts : ℕ)
  (all_triangles : Bool)

/-- The original figure given in the problem --/
def original_figure : Shape :=
  { area := 1, is_square := false }

/-- Checks if a cutting strategy is valid for the given conditions --/
def is_valid_strategy (s : CuttingStrategy) : Bool :=
  (s.num_parts ≤ 4) ∨ (s.num_parts ≤ 5 ∧ s.all_triangles)

/-- Theorem stating that there exists a valid cutting strategy to form a square --/
theorem exists_valid_square_forming_strategy :
  ∃ (s : CuttingStrategy) (result : Shape),
    is_valid_strategy s ∧
    result.is_square ∧
    result.area = original_figure.area :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_square_forming_strategy_l1582_158272


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1582_158245

theorem unique_three_digit_number : 
  ∃! (m g u : ℕ), 
    m ≠ g ∧ m ≠ u ∧ g ≠ u ∧
    m ∈ Finset.range 10 ∧ g ∈ Finset.range 10 ∧ u ∈ Finset.range 10 ∧
    100 * m + 10 * g + u ≥ 100 ∧ 100 * m + 10 * g + u < 1000 ∧
    100 * m + 10 * g + u = (m + g + u) * (m + g + u - 2) ∧
    100 * m + 10 * g + u = 195 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1582_158245


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1582_158276

theorem imaginary_part_of_one_minus_i_squared (i : ℂ) : Complex.im ((1 - i)^2) = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1582_158276


namespace NUMINAMATH_CALUDE_journey_distance_l1582_158265

/-- Prove that the total distance traveled is 300 km given the specified conditions -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h_total_time : total_time = 11)
  (h_speed1 : speed1 = 30)
  (h_speed2 : speed2 = 25)
  (h_half_distance : ∀ d : ℝ, d / speed1 + d / speed2 = total_time → d = 300) :
  ∃ d : ℝ, d = 300 ∧ d / (2 * speed1) + d / (2 * speed2) = total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l1582_158265


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1582_158261

theorem sum_of_numbers (a b : ℝ) (h1 : a - b = 5) (h2 : max a b = 25) : a + b = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1582_158261


namespace NUMINAMATH_CALUDE_payment_divisible_by_25_l1582_158238

theorem payment_divisible_by_25 (B : ℕ) (h : B ≤ 9) : 
  ∃ k : ℕ, 2000 + 100 * B + 5 = 25 * k := by
  sorry

end NUMINAMATH_CALUDE_payment_divisible_by_25_l1582_158238


namespace NUMINAMATH_CALUDE_smallest_numbers_with_percentage_property_l1582_158264

theorem smallest_numbers_with_percentage_property :
  ∃ (a b : ℕ), a = 21 ∧ b = 19 ∧
  (∀ (x y : ℕ), (95 * x = 105 * y) → (x ≥ a ∨ y ≥ b)) ∧
  (95 * a = 105 * b) := by
  sorry

end NUMINAMATH_CALUDE_smallest_numbers_with_percentage_property_l1582_158264


namespace NUMINAMATH_CALUDE_largest_multiple_18_with_9_0_m_div_18_eq_555_l1582_158247

/-- A function that checks if a natural number consists only of digits 9 and 0 -/
def onlyNineAndZero (n : ℕ) : Prop := sorry

/-- The largest positive multiple of 18 consisting only of digits 9 and 0 -/
def m : ℕ := sorry

theorem largest_multiple_18_with_9_0 :
  m > 0 ∧
  m % 18 = 0 ∧
  onlyNineAndZero m ∧
  ∀ k : ℕ, k > m → (k % 18 = 0 → ¬onlyNineAndZero k) :=
sorry

theorem m_div_18_eq_555 : m / 18 = 555 := sorry

end NUMINAMATH_CALUDE_largest_multiple_18_with_9_0_m_div_18_eq_555_l1582_158247


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_min_value_of_expression_l1582_158207

/-- Given a > 0, b > 0, and the minimum value of |x+a| + |x-b| is 4, then a + b = 4 -/
theorem sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_min : ∀ x, |x + a| + |x - b| ≥ 4) : a + b = 4 := by sorry

/-- Given a + b = 4, the minimum value of (1/4)a² + (1/9)b² is 16/13 -/
theorem min_value_of_expression (a b : ℝ) (h : a + b = 4) :
  ∀ x y, x > 0 → y > 0 → x + y = 4 → (1/4) * a^2 + (1/9) * b^2 ≤ (1/4) * x^2 + (1/9) * y^2 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_min_value_of_expression_l1582_158207


namespace NUMINAMATH_CALUDE_R_value_at_7_l1582_158298

/-- The function that defines R in terms of S and h -/
def R (S h : ℝ) : ℝ := h * S + 2 * S - 6

/-- The theorem stating that if R = 28 when S = 5, then R = 41 when S = 7 -/
theorem R_value_at_7 (h : ℝ) (h_condition : R 5 h = 28) : R 7 h = 41 := by
  sorry

end NUMINAMATH_CALUDE_R_value_at_7_l1582_158298


namespace NUMINAMATH_CALUDE_quadratic_intercept_distance_l1582_158295

/-- A quadratic function -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex (qf : QuadraticFunction) : ℝ := sorry

theorem quadratic_intercept_distance 
  (f g : QuadraticFunction)
  (h1 : ∀ x, g.f x = -f.f (120 - x))
  (h2 : ∃ v, g.f (vertex f) = v)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h3 : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h4 : f.f x₁ = 0 ∨ g.f x₁ = 0)
  (h5 : f.f x₂ = 0 ∨ g.f x₂ = 0)
  (h6 : f.f x₃ = 0 ∨ g.f x₃ = 0)
  (h7 : f.f x₄ = 0 ∨ g.f x₄ = 0)
  (h8 : x₃ - x₂ = 120) :
  x₄ - x₁ = 360 + 240 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_intercept_distance_l1582_158295


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1582_158215

theorem trigonometric_identity (θ : Real) (h : Real.tan θ = 3) :
  Real.sin θ^2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1582_158215


namespace NUMINAMATH_CALUDE_determinant_inequality_l1582_158248

def det2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (a : ℝ) : 
  det2 (a^2) 1 3 2 < det2 a 0 4 1 ↔ -1 < a ∧ a < 3/2 := by sorry

end NUMINAMATH_CALUDE_determinant_inequality_l1582_158248


namespace NUMINAMATH_CALUDE_joan_gemstones_l1582_158208

/-- Represents Joan's rock collection --/
structure RockCollection where
  minerals_yesterday : ℕ
  gemstones : ℕ
  minerals_today : ℕ

/-- Theorem about Joan's rock collection --/
theorem joan_gemstones (collection : RockCollection) 
  (h1 : collection.gemstones = collection.minerals_yesterday / 2)
  (h2 : collection.minerals_today = collection.minerals_yesterday + 6)
  (h3 : collection.minerals_today = 48) : 
  collection.gemstones = 21 := by
sorry

end NUMINAMATH_CALUDE_joan_gemstones_l1582_158208


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l1582_158221

theorem negation_of_existence_proposition :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l1582_158221


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1582_158283

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * (10 * c + d : ℚ) / 99 → c + d = 11 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1582_158283


namespace NUMINAMATH_CALUDE_angela_unfinished_problems_l1582_158210

theorem angela_unfinished_problems (total : Nat) (martha : Nat) (jenna : Nat) (mark : Nat)
  (h1 : total = 20)
  (h2 : martha = 2)
  (h3 : jenna = 4 * martha - 2)
  (h4 : mark = jenna / 2)
  (h5 : martha + jenna + mark ≤ total) :
  total - (martha + jenna + mark) = 9 := by
sorry

end NUMINAMATH_CALUDE_angela_unfinished_problems_l1582_158210


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1582_158290

/-- Represents a hyperbola with center (h, k) and parameters a and b -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (hyp : Hyperbola) (x y : ℝ) : Prop :=
  (x - hyp.h)^2 / hyp.a^2 - (y - hyp.k)^2 / hyp.b^2 = 1

theorem hyperbola_sum (hyp : Hyperbola) 
  (center : hyp.h = -3 ∧ hyp.k = 1)
  (vertex_distance : 2 * hyp.a = 8)
  (foci_distance : Real.sqrt (hyp.a^2 + hyp.b^2) = 5) :
  hyp.h + hyp.k + hyp.a + hyp.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1582_158290


namespace NUMINAMATH_CALUDE_complement_of_35_degree_angle_l1582_158275

theorem complement_of_35_degree_angle (A : Real) : 
  A = 35 → 90 - A = 55 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_35_degree_angle_l1582_158275


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1582_158242

theorem complex_fraction_simplification :
  (3 - 2 * Complex.I) / (1 + 4 * Complex.I) = -5/17 - 14/17 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1582_158242


namespace NUMINAMATH_CALUDE_fraction_equality_l1582_158241

def fraction_pairs : Set (ℤ × ℤ) :=
  {(0, 6), (1, -1), (6, -6), (13, -7), (-2, -22), (-3, -15), (-8, -10), (-15, -9)}

theorem fraction_equality (k l : ℤ) :
  (7 * k - 5) / (5 * k - 3) = (6 * l - 1) / (4 * l - 3) ↔ (k, l) ∈ fraction_pairs := by
  sorry

#check fraction_equality

end NUMINAMATH_CALUDE_fraction_equality_l1582_158241


namespace NUMINAMATH_CALUDE_arrange_balls_theorem_l1582_158205

/-- The number of ways to arrange balls of different types in a row -/
def arrangeMultisetBalls (n₁ n₂ n₃ : ℕ) : ℕ :=
  Nat.factorial (n₁ + n₂ + n₃) / (Nat.factorial n₁ * Nat.factorial n₂ * Nat.factorial n₃)

/-- Theorem stating that arranging 5 basketballs, 3 volleyballs, and 2 footballs yields 2520 ways -/
theorem arrange_balls_theorem : arrangeMultisetBalls 5 3 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_arrange_balls_theorem_l1582_158205


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1582_158217

theorem divisibility_by_five (x y : ℕ) (h1 : 2 * x ^ 2 - 1 = y ^ 15) (h2 : x > 1) :
  5 ∣ x := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1582_158217


namespace NUMINAMATH_CALUDE_square_diff_equals_four_l1582_158267

theorem square_diff_equals_four (a b : ℝ) (h : a = b + 2) : a^2 - 2*a*b + b^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_equals_four_l1582_158267


namespace NUMINAMATH_CALUDE_opposite_of_2024_l1582_158239

theorem opposite_of_2024 : -(2024 : ℤ) = -2024 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2024_l1582_158239


namespace NUMINAMATH_CALUDE_four_digit_addition_l1582_158256

theorem four_digit_addition (A B C D : ℕ) : 
  4000 * A + 500 * B + 100 * C + 20 * D + 7 = 8070 → C = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_addition_l1582_158256


namespace NUMINAMATH_CALUDE_snake_eating_time_l1582_158260

/-- Represents the number of weeks it takes for a snake to eat one mouse. -/
def weeks_per_mouse (mice_per_decade : ℕ) : ℚ :=
  (10 * 52) / mice_per_decade

/-- Proves that a snake eating 130 mice in a decade takes 4 weeks to eat one mouse. -/
theorem snake_eating_time : weeks_per_mouse 130 = 4 := by
  sorry

end NUMINAMATH_CALUDE_snake_eating_time_l1582_158260


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l1582_158274

theorem sin_two_alpha_value (α : Real) (h : Real.sin α - Real.cos α = 4/3) :
  Real.sin (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l1582_158274


namespace NUMINAMATH_CALUDE_bran_remaining_payment_l1582_158292

def tuition_fee : ℝ := 90
def monthly_earnings : ℝ := 15
def scholarship_percentage : ℝ := 0.30
def payment_period : ℕ := 3

theorem bran_remaining_payment :
  tuition_fee * (1 - scholarship_percentage) - monthly_earnings * payment_period = 18 := by
  sorry

end NUMINAMATH_CALUDE_bran_remaining_payment_l1582_158292


namespace NUMINAMATH_CALUDE_smallest_sector_angle_l1582_158262

def circle_sectors (n : ℕ) (a : ℕ → ℕ) : Prop :=
  (∀ i, i ∈ Finset.range n → a i > 0) ∧
  (∀ i j k, i < j ∧ j < k → a j - a i = a k - a j) ∧
  (Finset.sum (Finset.range n) a = 360)

theorem smallest_sector_angle :
  ∀ a : ℕ → ℕ, circle_sectors 16 a → ∃ i, a i = 15 ∧ ∀ j, a j ≥ a i := by
  sorry

end NUMINAMATH_CALUDE_smallest_sector_angle_l1582_158262


namespace NUMINAMATH_CALUDE_exists_arithmetic_not_m_sequence_l1582_158206

/-- Definition of "M sequence" -/
def is_m_sequence (b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  (∀ n, b n < b (n + 1)) ∧ 
  (∀ n, c n < c (n + 1)) ∧
  (∀ n, ∃ m, c n ≤ b m ∧ b m ≤ c (n + 1))

/-- Arithmetic sequence -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) - a n = d

/-- Partial sum sequence -/
def partial_sum (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => partial_sum a n + a (n + 1)

/-- Main theorem -/
theorem exists_arithmetic_not_m_sequence :
  ∃ a : ℕ → ℝ, is_arithmetic a ∧ ¬(is_m_sequence a (partial_sum a)) := by
  sorry

end NUMINAMATH_CALUDE_exists_arithmetic_not_m_sequence_l1582_158206


namespace NUMINAMATH_CALUDE_furniture_fraction_l1582_158259

/-- Prove that the fraction of savings spent on furniture is 3/4, given that
the original savings were $800, the TV cost $200, and the rest was spent on furniture. -/
theorem furniture_fraction (savings : ℚ) (tv_cost : ℚ) (furniture_cost : ℚ) 
  (h1 : savings = 800)
  (h2 : tv_cost = 200)
  (h3 : furniture_cost + tv_cost = savings) :
  furniture_cost / savings = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_furniture_fraction_l1582_158259


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l1582_158212

theorem factor_difference_of_squares (x : ℝ) : 
  81 - 16 * (x - 1)^2 = (13 - 4*x) * (5 + 4*x) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l1582_158212


namespace NUMINAMATH_CALUDE_cos_a3_plus_a5_l1582_158273

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem cos_a3_plus_a5 (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h1 : a 1 + a 4 + a 7 = 5 * Real.pi / 4) : 
  Real.cos (a 3 + a 5) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_a3_plus_a5_l1582_158273


namespace NUMINAMATH_CALUDE_february_greatest_difference_l1582_158255

-- Define the sales data for drummers and bugle players
def drummer_sales : Fin 5 → ℕ
  | 0 => 4  -- January
  | 1 => 5  -- February
  | 2 => 4  -- March
  | 3 => 3  -- April
  | 4 => 2  -- May

def bugle_sales : Fin 5 → ℕ
  | 0 => 3  -- January
  | 1 => 3  -- February
  | 2 => 4  -- March
  | 3 => 4  -- April
  | 4 => 3  -- May

-- Define the percentage difference function
def percentage_difference (a b : ℕ) : ℚ :=
  (max a b - min a b : ℚ) / (min a b : ℚ) * 100

-- Define a function to calculate the percentage difference for each month
def month_percentage_difference (i : Fin 5) : ℚ :=
  percentage_difference (drummer_sales i) (bugle_sales i)

-- Theorem: February has the greatest percentage difference
theorem february_greatest_difference :
  ∀ i : Fin 5, i ≠ 1 → month_percentage_difference 1 ≥ month_percentage_difference i :=
by sorry

end NUMINAMATH_CALUDE_february_greatest_difference_l1582_158255


namespace NUMINAMATH_CALUDE_worksheets_graded_l1582_158244

/-- Represents the problem of determining the number of worksheets graded before new ones were turned in. -/
theorem worksheets_graded (initial : ℕ) (new_turned_in : ℕ) (final : ℕ) : 
  initial = 34 → new_turned_in = 36 → final = 63 → 
  ∃ (graded : ℕ), graded = 7 ∧ initial - graded + new_turned_in = final := by
  sorry

end NUMINAMATH_CALUDE_worksheets_graded_l1582_158244


namespace NUMINAMATH_CALUDE_blackboard_division_l1582_158225

theorem blackboard_division : (96 : ℕ) / 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_blackboard_division_l1582_158225


namespace NUMINAMATH_CALUDE_count_non_negative_l1582_158289

theorem count_non_negative : 
  let numbers := [-(-4), |-1|, -|0|, (-2)^3]
  (numbers.filter (λ x => x ≥ 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_non_negative_l1582_158289


namespace NUMINAMATH_CALUDE_equation_solutions_l1582_158297

theorem equation_solutions :
  ∀ x y : ℕ, 1 + 3^x = 2^y ↔ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1582_158297


namespace NUMINAMATH_CALUDE_cos_135_degrees_l1582_158254

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l1582_158254


namespace NUMINAMATH_CALUDE_solution_set_x_one_minus_x_l1582_158230

theorem solution_set_x_one_minus_x (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_one_minus_x_l1582_158230


namespace NUMINAMATH_CALUDE_basketball_score_proof_l1582_158299

theorem basketball_score_proof (total : ℕ) 
  (hA : total / 4 = total / 4)  -- Player A scored 1/4 of total
  (hB : (total * 2) / 7 = (total * 2) / 7)  -- Player B scored 2/7 of total
  (hC : 15 ≤ total)  -- Player C scored 15 points
  (hRemaining : ∀ i : Fin 7, (total - (total / 4 + (total * 2) / 7 + 15)) / 7 ≤ 2)  -- Remaining players scored no more than 2 points each
  : total - (total / 4 + (total * 2) / 7 + 15) = 13 :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l1582_158299


namespace NUMINAMATH_CALUDE_fraction_relation_l1582_158278

theorem fraction_relation (n d : ℚ) (k : ℚ) : 
  d = k * (2 * n) →
  (n + 1) / (d + 1) = 3 / 5 →
  n / d = 5 / 9 →
  k = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l1582_158278


namespace NUMINAMATH_CALUDE_postcard_perimeter_l1582_158268

/-- The perimeter of a rectangle with width 6 inches and height 4 inches is 20 inches. -/
theorem postcard_perimeter : 
  let width : ℝ := 6
  let height : ℝ := 4
  let perimeter := 2 * (width + height)
  perimeter = 20 :=
by sorry

end NUMINAMATH_CALUDE_postcard_perimeter_l1582_158268


namespace NUMINAMATH_CALUDE_product_equality_l1582_158293

theorem product_equality : 469111111 * 99999999 = 46911111053088889 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1582_158293


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l1582_158201

/-- Represents the possible states of a cell in the game grid -/
inductive Cell
| Empty : Cell
| S : Cell
| O : Cell

/-- Represents the game state -/
structure GameState where
  grid : Vector Cell 2000
  currentPlayer : Nat

/-- Checks if a player has won by forming SOS pattern -/
def hasWon (state : GameState) : Bool :=
  sorry

/-- Checks if the game is a draw -/
def isDraw (state : GameState) : Bool :=
  sorry

/-- Represents a player's strategy -/
def Strategy := GameState → Nat

/-- Checks if a strategy is winning for the given player -/
def isWinningStrategy (player : Nat) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy 2 strategy :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l1582_158201


namespace NUMINAMATH_CALUDE_sqrt_of_sum_of_cubes_l1582_158266

theorem sqrt_of_sum_of_cubes : Real.sqrt (5 * (4^3 + 4^3 + 4^3 + 4^3)) = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sum_of_cubes_l1582_158266


namespace NUMINAMATH_CALUDE_pyramid_total_area_l1582_158214

-- Define the square side length and pyramid height
def squareSide : ℝ := 6
def pyramidHeight : ℝ := 4

-- Define the structure of our pyramid
structure Pyramid where
  base : ℝ
  height : ℝ

-- Define our specific pyramid
def ourPyramid : Pyramid :=
  { base := squareSide,
    height := pyramidHeight }

-- Theorem statement
theorem pyramid_total_area (p : Pyramid) (h : p = ourPyramid) :
  let diagonal := p.base * Real.sqrt 2
  let slantHeight := Real.sqrt (p.height^2 + (diagonal/2)^2)
  let triangleHeight := Real.sqrt (slantHeight^2 - (p.base/2)^2)
  let squareArea := p.base^2
  let triangleArea := 4 * (p.base * triangleHeight / 2)
  squareArea + triangleArea = 96 := by sorry

end NUMINAMATH_CALUDE_pyramid_total_area_l1582_158214


namespace NUMINAMATH_CALUDE_clothing_discount_l1582_158271

theorem clothing_discount (P : ℝ) (f : ℝ) (h1 : P > 0) (h2 : f > 0) (h3 : f < 1) :
  (f * P - (1/2) * P = 0.4 * (f * P)) → f = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_clothing_discount_l1582_158271


namespace NUMINAMATH_CALUDE_edwards_earnings_l1582_158226

/-- Edward's lawn mowing business earnings --/
theorem edwards_earnings (summer_earnings : ℕ) (supplies_cost : ℕ) (total_earnings : ℕ)
  (h1 : summer_earnings = 27)
  (h2 : supplies_cost = 5)
  (h3 : total_earnings = 24)
  : ∃ spring_earnings : ℕ,
    spring_earnings + (summer_earnings - supplies_cost) = total_earnings ∧
    spring_earnings = 2 := by
  sorry

end NUMINAMATH_CALUDE_edwards_earnings_l1582_158226


namespace NUMINAMATH_CALUDE_rolling_cube_path_length_l1582_158227

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Represents the path of a point on a rolling cube -/
def RollingCubePath (c : Cube) : ℝ := sorry

/-- Theorem stating the length of the path followed by the center point on the top face of a rolling cube -/
theorem rolling_cube_path_length (c : Cube) (h : c.sideLength = 2) :
  RollingCubePath c = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rolling_cube_path_length_l1582_158227


namespace NUMINAMATH_CALUDE_unique_solution_equation_l1582_158282

theorem unique_solution_equation :
  ∃! x : ℝ, x ≠ 2 ∧ x > 0 ∧ (3 * x^2 - 12 * x) / (x^2 - 4 * x) = x - 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l1582_158282


namespace NUMINAMATH_CALUDE_sin_390_degrees_l1582_158286

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l1582_158286


namespace NUMINAMATH_CALUDE_twentieth_term_da_yan_l1582_158269

def da_yan_sequence (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 2
  else
    (n^2 - 1) / 2

theorem twentieth_term_da_yan (n : ℕ) : da_yan_sequence 20 = 200 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_da_yan_l1582_158269


namespace NUMINAMATH_CALUDE_bouncy_balls_per_package_l1582_158284

theorem bouncy_balls_per_package (total_packages : Nat) (total_balls : Nat) :
  total_packages = 16 →
  total_balls = 160 →
  ∃ (balls_per_package : Nat), balls_per_package * total_packages = total_balls ∧ balls_per_package = 10 := by
  sorry

end NUMINAMATH_CALUDE_bouncy_balls_per_package_l1582_158284


namespace NUMINAMATH_CALUDE_negative_five_plus_eight_equals_three_l1582_158281

theorem negative_five_plus_eight_equals_three : -5 + 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_plus_eight_equals_three_l1582_158281


namespace NUMINAMATH_CALUDE_fractional_equation_one_l1582_158288

theorem fractional_equation_one (x : ℝ) : 
  x ≠ 0 ∧ x ≠ -1 → (2 / x = 3 / (x + 1) ↔ x = 2) := by sorry

end NUMINAMATH_CALUDE_fractional_equation_one_l1582_158288
