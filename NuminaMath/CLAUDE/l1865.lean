import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_comparison_l1865_186597

theorem reciprocal_comparison : 
  ((-1/3 : ℚ) < (-3 : ℚ) → False) ∧
  ((-3/2 : ℚ) < (-2/3 : ℚ)) ∧
  ((1/4 : ℚ) < (4 : ℚ)) ∧
  ((3/4 : ℚ) < (4/3 : ℚ) → False) ∧
  ((4/3 : ℚ) < (3/4 : ℚ) → False) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_l1865_186597


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l1865_186555

theorem maintenance_check_increase (original_days new_days : ℝ) 
  (h1 : original_days = 20)
  (h2 : new_days = 25) :
  ((new_days - original_days) / original_days) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l1865_186555


namespace NUMINAMATH_CALUDE_article_purchase_price_l1865_186561

/-- The purchase price of an article given specific markup conditions -/
theorem article_purchase_price : 
  ∀ (markup overhead_percentage net_profit purchase_price : ℝ),
  markup = 40 →
  overhead_percentage = 0.15 →
  net_profit = 12 →
  markup = overhead_percentage * purchase_price + net_profit →
  purchase_price = 186.67 := by
sorry

end NUMINAMATH_CALUDE_article_purchase_price_l1865_186561


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l1865_186554

theorem scientific_notation_proof : 
  (192000000 : ℝ) = 1.92 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l1865_186554


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l1865_186548

theorem baker_cakes_sold (bought : ℕ) (difference : ℕ) (sold : ℕ) : 
  bought = 154 → difference = 63 → bought = sold + difference → sold = 91 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l1865_186548


namespace NUMINAMATH_CALUDE_equidistant_is_circumcenter_l1865_186577

/-- Triangle represented by complex coordinates of its vertices -/
structure ComplexTriangle where
  z₁ : ℂ
  z₂ : ℂ
  z₃ : ℂ

/-- A point is equidistant from all vertices of the triangle -/
def isEquidistant (z : ℂ) (t : ComplexTriangle) : Prop :=
  Complex.abs (z - t.z₁) = Complex.abs (z - t.z₂) ∧
  Complex.abs (z - t.z₂) = Complex.abs (z - t.z₃)

/-- The circumcenter of a triangle -/
def isCircumcenter (z : ℂ) (t : ComplexTriangle) : Prop :=
  -- Definition of circumcenter (placeholder)
  True

theorem equidistant_is_circumcenter (t : ComplexTriangle) (z : ℂ) :
  isEquidistant z t → isCircumcenter z t := by
  sorry

end NUMINAMATH_CALUDE_equidistant_is_circumcenter_l1865_186577


namespace NUMINAMATH_CALUDE_faster_train_speed_l1865_186512

theorem faster_train_speed 
  (train_length : ℝ) 
  (slower_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 80) 
  (h2 : slower_speed = 36) 
  (h3 : passing_time = 36 / 3600) : 
  ∃ (faster_speed : ℝ), faster_speed = 52 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l1865_186512


namespace NUMINAMATH_CALUDE_unique_fraction_sum_l1865_186552

theorem unique_fraction_sum (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (n m : ℕ), n ≠ m ∧ 2/p = 1/n + 1/m ∧ n = (p + 1)/2 ∧ m = p * (p + 1)/2 :=
by sorry

end NUMINAMATH_CALUDE_unique_fraction_sum_l1865_186552


namespace NUMINAMATH_CALUDE_club_truncator_probability_l1865_186571

/-- The number of matches Club Truncator plays -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def single_match_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 2741/6561

theorem club_truncator_probability :
  let total_outcomes := 3^num_matches
  let same_wins_losses := 1079
  (total_outcomes - same_wins_losses) / (2 * total_outcomes) = more_wins_prob :=
sorry

end NUMINAMATH_CALUDE_club_truncator_probability_l1865_186571


namespace NUMINAMATH_CALUDE_store_discount_difference_l1865_186511

theorem store_discount_difference :
  let initial_discount : ℝ := 0.25
  let additional_discount : ℝ := 0.10
  let claimed_discount : ℝ := 0.35
  let price_after_initial := 1 - initial_discount
  let price_after_both := price_after_initial * (1 - additional_discount)
  let true_discount := 1 - price_after_both
  claimed_discount - true_discount = 0.025 := by
sorry

end NUMINAMATH_CALUDE_store_discount_difference_l1865_186511


namespace NUMINAMATH_CALUDE_game_score_invariant_final_score_difference_l1865_186547

def game_score (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem game_score_invariant (n : ℕ) (h : n ≥ 2) :
  ∀ (moves : List (ℕ × ℕ × ℕ)),
    moves.all (λ (a, b, c) => a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 1 ∧ b + c = a) →
    moves.foldl (λ acc (a, b, c) => acc + b * c) 0 = game_score n :=
  sorry

theorem final_score_difference (n : ℕ) (h : n ≥ 2) :
  let M := game_score n
  let m := game_score n
  M - m = 0 :=
  sorry

end NUMINAMATH_CALUDE_game_score_invariant_final_score_difference_l1865_186547


namespace NUMINAMATH_CALUDE_cory_fruit_arrangements_l1865_186569

/-- The number of ways to arrange fruits over a week -/
def arrangeWeekFruits (apples oranges : ℕ) : ℕ :=
  Nat.factorial (apples + oranges + 1) / (Nat.factorial apples * Nat.factorial oranges)

/-- The number of ways to arrange fruits over a week, excluding banana on first day -/
def arrangeWeekFruitsNoBananaFirst (apples oranges : ℕ) : ℕ :=
  (apples + oranges) * arrangeWeekFruits apples oranges

theorem cory_fruit_arrangements :
  arrangeWeekFruitsNoBananaFirst 4 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_cory_fruit_arrangements_l1865_186569


namespace NUMINAMATH_CALUDE_trig_identity_l1865_186598

theorem trig_identity (x : ℝ) (h : Real.sin (π / 6 - x) = 1 / 2) :
  Real.sin (19 * π / 6 - x) + (Real.sin (-2 * π / 3 + x))^2 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1865_186598


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_plane_l1865_186532

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem statement
theorem line_parallel_perpendicular_plane 
  (m n : Line) (α : Plane) :
  parallel_line_plane m α → 
  perpendicular_line_plane n α → 
  perpendicular_line_line m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_plane_l1865_186532


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1865_186513

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y / z + y * z / x + z * x / y) > 2 * (x^3 + y^3 + z^3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1865_186513


namespace NUMINAMATH_CALUDE_student_weight_loss_l1865_186527

/-- The amount of weight a student needs to lose to weigh twice as much as his sister. -/
def weight_to_lose (total_weight sister_weight : ℝ) : ℝ :=
  total_weight - sister_weight - 2 * sister_weight

theorem student_weight_loss (total_weight student_weight : ℝ) 
  (h1 : total_weight = 104)
  (h2 : student_weight = 71) :
  weight_to_lose total_weight (total_weight - student_weight) = 5 := by
  sorry

#eval weight_to_lose 104 33

end NUMINAMATH_CALUDE_student_weight_loss_l1865_186527


namespace NUMINAMATH_CALUDE_train_length_proof_l1865_186546

theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 ∧ crossing_time = 36 ∧ train_speed = 40 →
  train_speed * crossing_time - bridge_length = 1140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l1865_186546


namespace NUMINAMATH_CALUDE_building_height_from_shadows_l1865_186514

/-- Given a flagstaff and a building casting shadows under similar conditions,
    calculate the height of the building using the concept of similar triangles. -/
theorem building_height_from_shadows
  (flagstaff_height : ℝ)
  (flagstaff_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagstaff_height : flagstaff_height = 17.5)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75) :
  ∃ (building_height : ℝ),
    (building_height / building_shadow = flagstaff_height / flagstaff_shadow) ∧
    (abs (building_height - 12.44) < 0.01) :=
sorry

end NUMINAMATH_CALUDE_building_height_from_shadows_l1865_186514


namespace NUMINAMATH_CALUDE_simplify_fraction_calculate_logarithmic_expression_l1865_186560

-- Part 1
theorem simplify_fraction (a : ℝ) (ha : a > 0) :
  a^2 / (Real.sqrt a * 3 * a^2) = a^(5/6) := by sorry

-- Part 2
theorem calculate_logarithmic_expression :
  (2 * Real.log 2 + Real.log 3) / (1 + 1/2 * Real.log 0.36 + 1/3 * Real.log 8) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_calculate_logarithmic_expression_l1865_186560


namespace NUMINAMATH_CALUDE_six_rounds_maximize_configurations_optimal_rounds_is_six_l1865_186543

/-- The number of cities and days in the championship --/
def n : ℕ := 8

/-- The number of possible configurations for k rounds --/
def N (k : ℕ) : ℚ :=
  (Nat.factorial n * Nat.factorial n) / (Nat.factorial k * (Nat.factorial (n - k))^2)

/-- The theorem stating that 6 rounds maximizes the number of configurations --/
theorem six_rounds_maximize_configurations :
  ∀ k : ℕ, k ≠ 6 → k ≤ n → N k ≤ N 6 := by
  sorry

/-- The main theorem proving that 6 is the optimal number of rounds --/
theorem optimal_rounds_is_six :
  ∃ k : ℕ, k ≤ n ∧ (∀ j : ℕ, j ≤ n → N j ≤ N k) ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_rounds_maximize_configurations_optimal_rounds_is_six_l1865_186543


namespace NUMINAMATH_CALUDE_problem_solution_problem_solution_2_l1865_186535

def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∀ x, p x → q x m) → m ∈ Set.Ici 4 :=
sorry

theorem problem_solution_2 :
  ∃ S : Set ℝ, S = Set.Icc (-4) (-1) ∪ Set.Ioc 5 6 ∧
  ∀ x, x ∈ S ↔ (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_problem_solution_2_l1865_186535


namespace NUMINAMATH_CALUDE_red_bellies_percentage_l1865_186506

/-- Represents the total number of minnows in the pond -/
def total_minnows : ℕ := 50

/-- Represents the number of minnows with red bellies -/
def red_bellies : ℕ := 20

/-- Represents the number of minnows with white bellies -/
def white_bellies : ℕ := 15

/-- Represents the percentage of minnows with green bellies -/
def green_bellies_percent : ℚ := 30 / 100

/-- Theorem stating that the percentage of minnows with red bellies is 40% -/
theorem red_bellies_percentage :
  (red_bellies : ℚ) / total_minnows * 100 = 40 := by
  sorry

/-- Lemma verifying that the total number of minnows is correct -/
lemma total_minnows_check :
  total_minnows = red_bellies + white_bellies + (green_bellies_percent * total_minnows) := by
  sorry

end NUMINAMATH_CALUDE_red_bellies_percentage_l1865_186506


namespace NUMINAMATH_CALUDE_percentage_problem_l1865_186541

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x * (x / 100) = 4) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1865_186541


namespace NUMINAMATH_CALUDE_employment_percentage_l1865_186588

theorem employment_percentage (population : ℝ) 
  (h1 : population > 0)
  (h2 : (80 : ℝ) / 100 * population = employed_males)
  (h3 : (1 : ℝ) / 3 * total_employed = employed_females)
  (h4 : employed_males + employed_females = total_employed) :
  total_employed / population = (60 : ℝ) / 100 := by
sorry

end NUMINAMATH_CALUDE_employment_percentage_l1865_186588


namespace NUMINAMATH_CALUDE_inequality_proofs_l1865_186509

theorem inequality_proofs (x : ℝ) : 
  (6 + 3 * x > 30 → x > 8) ∧ 
  (1 - x < 3 - (x - 5) / 2 → x > -9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l1865_186509


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l1865_186564

theorem students_taking_neither_music_nor_art 
  (total_students : ℕ) 
  (music_students : ℕ) 
  (art_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 500) 
  (h2 : music_students = 30) 
  (h3 : art_students = 10) 
  (h4 : both_students = 10) : 
  total_students - (music_students + art_students - both_students) = 460 := by
sorry

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l1865_186564


namespace NUMINAMATH_CALUDE_five_fridays_in_august_l1865_186524

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to count occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem stating that if July has five Tuesdays, August will have five Fridays -/
theorem five_fridays_in_august 
  (july : Month) 
  (august : Month) 
  (h1 : july.days = 31) 
  (h2 : august.days = 31) 
  (h3 : countDayOccurrences july DayOfWeek.Tuesday = 5) :
  countDayOccurrences august DayOfWeek.Friday = 5 :=
sorry

end NUMINAMATH_CALUDE_five_fridays_in_august_l1865_186524


namespace NUMINAMATH_CALUDE_utopia_national_park_elephant_rate_l1865_186525

/-- Proves that the rate of new elephants entering Utopia National Park is 1500 per hour --/
theorem utopia_national_park_elephant_rate : 
  let initial_elephants : ℕ := 30000
  let exodus_duration : ℕ := 4
  let exodus_rate : ℕ := 2880
  let new_elephants_duration : ℕ := 7
  let final_elephants : ℕ := 28980
  
  let elephants_after_exodus := initial_elephants - exodus_duration * exodus_rate
  let new_elephants := final_elephants - elephants_after_exodus
  let new_elephants_rate := new_elephants / new_elephants_duration
  
  new_elephants_rate = 1500 := by
  sorry

end NUMINAMATH_CALUDE_utopia_national_park_elephant_rate_l1865_186525


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l1865_186539

theorem arctan_tan_difference (θ : Real) :
  0 ≤ θ ∧ θ ≤ π →
  Real.arctan (Real.tan (5 * π / 12) - 3 * Real.tan (π / 12)) = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l1865_186539


namespace NUMINAMATH_CALUDE_original_plan_calculation_l1865_186518

def thursday_sales : ℕ := 210
def friday_sales : ℕ := 2 * thursday_sales
def saturday_sales : ℕ := 130
def sunday_sales : ℕ := saturday_sales / 2
def excess_sales : ℕ := 325

def total_sales : ℕ := thursday_sales + friday_sales + saturday_sales + sunday_sales

theorem original_plan_calculation :
  total_sales - excess_sales = 500 := by sorry

end NUMINAMATH_CALUDE_original_plan_calculation_l1865_186518


namespace NUMINAMATH_CALUDE_x_squared_gt_16_necessary_not_sufficient_for_x_gt_4_l1865_186592

theorem x_squared_gt_16_necessary_not_sufficient_for_x_gt_4 :
  (∃ x : ℝ, x^2 > 16 ∧ x ≤ 4) ∧
  (∀ x : ℝ, x > 4 → x^2 > 16) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_gt_16_necessary_not_sufficient_for_x_gt_4_l1865_186592


namespace NUMINAMATH_CALUDE_fraction_equation_transformation_l1865_186593

/-- Given the fractional equation (x / (x - 1)) - (2 / x) = 1,
    prove that eliminating the denominators results in x^2 - 2(x-1) = x(x-1) -/
theorem fraction_equation_transformation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1)) - (2 / x) = 1 ↔ x^2 - 2*(x-1) = x*(x-1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_transformation_l1865_186593


namespace NUMINAMATH_CALUDE_hotel_flat_fee_l1865_186545

/-- Given a hotel charging a flat fee for the first night and a fixed amount for additional nights,
    prove that the flat fee is $60 if a 4-night stay costs $205 and a 7-night stay costs $350. -/
theorem hotel_flat_fee (flat_fee nightly_fee : ℚ) : 
  (flat_fee + 3 * nightly_fee = 205) →
  (flat_fee + 6 * nightly_fee = 350) →
  flat_fee = 60 := by sorry

end NUMINAMATH_CALUDE_hotel_flat_fee_l1865_186545


namespace NUMINAMATH_CALUDE_sine_symmetry_axis_l1865_186508

/-- The symmetry axis of the graph of y = sin(x - π/3) is x = -π/6 -/
theorem sine_symmetry_axis :
  ∀ x : ℝ, (∀ y : ℝ, y = Real.sin (x - π/3)) →
  (∃ k : ℤ, x = -π/6 + k * π) :=
sorry

end NUMINAMATH_CALUDE_sine_symmetry_axis_l1865_186508


namespace NUMINAMATH_CALUDE_solve_equation_l1865_186574

theorem solve_equation (x : ℝ) (h : Real.sqrt ((2 / x) + 3) = 2) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1865_186574


namespace NUMINAMATH_CALUDE_davids_physics_marks_l1865_186540

/-- Given David's marks in various subjects and his average, prove his marks in Physics --/
theorem davids_physics_marks
  (english : ℕ)
  (mathematics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (total_subjects : ℕ)
  (h_english : english = 86)
  (h_mathematics : mathematics = 85)
  (h_chemistry : chemistry = 87)
  (h_biology : biology = 95)
  (h_average : average = 89)
  (h_subjects : total_subjects = 5) :
  ∃ (physics : ℕ), physics = 92 ∧
    average * total_subjects = english + mathematics + physics + chemistry + biology :=
by sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l1865_186540


namespace NUMINAMATH_CALUDE_angle_OA_OC_l1865_186558

def angle_between_vectors (OA OB OC : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem angle_OA_OC (OA OB OC : ℝ × ℝ × ℝ) 
  (h1 : ‖OA‖ = 1)
  (h2 : ‖OB‖ = 2)
  (h3 : Real.cos (angle_between_vectors OA OB OC) = -1/2)
  (h4 : OC = (1/2 : ℝ) • OA + (1/4 : ℝ) • OB) :
  angle_between_vectors OA OC OC = π/3 :=
sorry

end NUMINAMATH_CALUDE_angle_OA_OC_l1865_186558


namespace NUMINAMATH_CALUDE_prob_odd_sum_is_half_l1865_186595

/-- Represents a wheel with numbers from 1 to n -/
def Wheel (n : ℕ) := Finset (Fin n)

/-- The probability of selecting an odd number from a wheel -/
def prob_odd (w : Wheel n) : ℚ :=
  (w.filter (λ x => x.val % 2 = 1)).card / w.card

/-- The probability of selecting an even number from a wheel -/
def prob_even (w : Wheel n) : ℚ :=
  (w.filter (λ x => x.val % 2 = 0)).card / w.card

/-- The first wheel with numbers 1 to 5 -/
def wheel1 : Wheel 5 := Finset.univ

/-- The second wheel with numbers 1 to 4 -/
def wheel2 : Wheel 4 := Finset.univ

theorem prob_odd_sum_is_half :
  prob_odd wheel1 * prob_even wheel2 + prob_even wheel1 * prob_odd wheel2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_sum_is_half_l1865_186595


namespace NUMINAMATH_CALUDE_square_roots_theorem_l1865_186589

theorem square_roots_theorem (n : ℝ) (h : n > 0) :
  (∃ a : ℝ, (2*a + 1)^2 = n ∧ (a + 5)^2 = n) → 
  (∃ a : ℝ, 2*a + 1 + a + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l1865_186589


namespace NUMINAMATH_CALUDE_sum_of_differences_l1865_186534

def S : Finset ℕ := Finset.range 11

def pairDifference (i j : ℕ) : ℕ := 
  if i < j then 2^j - 2^i else 2^i - 2^j

def N : ℕ := Finset.sum (S.product S) (fun (p : ℕ × ℕ) => pairDifference p.1 p.2)

theorem sum_of_differences : N = 16398 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_differences_l1865_186534


namespace NUMINAMATH_CALUDE_dans_remaining_money_is_14_02_l1865_186557

/-- Calculates the remaining money after Dan's shopping trip -/
def dans_remaining_money (initial_money : ℚ) (candy_price : ℚ) (candy_count : ℕ) 
  (toy_price : ℚ) (toy_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let candy_total := candy_price * candy_count
  let discounted_toy := toy_price * (1 - toy_discount)
  let subtotal := candy_total + discounted_toy
  let total_with_tax := subtotal * (1 + sales_tax)
  initial_money - total_with_tax

/-- Theorem stating that Dan's remaining money after shopping is $14.02 -/
theorem dans_remaining_money_is_14_02 :
  dans_remaining_money 45 4 4 15 0.1 0.05 = 14.02 := by
  sorry

#eval dans_remaining_money 45 4 4 15 0.1 0.05

end NUMINAMATH_CALUDE_dans_remaining_money_is_14_02_l1865_186557


namespace NUMINAMATH_CALUDE_eunji_exam_result_l1865_186504

def exam_problem (exam_a_total exam_b_total exam_a_wrong exam_b_extra_wrong : ℕ) : Prop :=
  let exam_a_right := exam_a_total - exam_a_wrong
  let exam_b_wrong := exam_a_wrong + exam_b_extra_wrong
  let exam_b_right := exam_b_total - exam_b_wrong
  exam_a_right + exam_b_right = 9

theorem eunji_exam_result :
  exam_problem 12 15 8 2 := by
  sorry

end NUMINAMATH_CALUDE_eunji_exam_result_l1865_186504


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1865_186553

/-- Given an arithmetic sequence, if the sum of the first n terms is P 
    and the sum of the first 2n terms is q, then the sum of the first 3n terms is 3(2P - q). -/
theorem arithmetic_sequence_sum (n : ℕ) (P q : ℝ) :
  (∃ (a d : ℝ), P = n / 2 * (2 * a + (n - 1) * d) ∧ q = n * (2 * a + (2 * n - 1) * d)) →
  (∃ (S_3n : ℝ), S_3n = 3 * (2 * P - q)) :=
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1865_186553


namespace NUMINAMATH_CALUDE_age_difference_l1865_186533

theorem age_difference (A B : ℕ) : 
  B = 39 → 
  A + 10 = 2 * (B - 10) → 
  A - B = 9 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1865_186533


namespace NUMINAMATH_CALUDE_total_cost_is_1046_l1865_186502

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 349 / 100

/-- The number of sandwiches -/
def num_sandwiches : ℕ := 2

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 87 / 100

/-- The number of sodas -/
def num_sodas : ℕ := 4

/-- The total cost of the order -/
def total_cost : ℚ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem total_cost_is_1046 : total_cost = 1046 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_1046_l1865_186502


namespace NUMINAMATH_CALUDE_cone_base_radius_l1865_186523

/-- Proves that a cone with a lateral surface made from a sector of a circle
    with radius 9 cm and central angle 240° has a circular base with radius 6 cm. -/
theorem cone_base_radius (r : ℝ) (θ : ℝ) (h1 : r = 9) (h2 : θ = 240 * π / 180) :
  r * θ / (2 * π) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1865_186523


namespace NUMINAMATH_CALUDE_solution_Y_initial_weight_l1865_186550

/-- Represents the composition and transformation of a solution --/
structure Solution where
  initialWeight : ℝ
  liquidXPercentage : ℝ
  waterPercentage : ℝ
  evaporatedWater : ℝ
  addedSolutionWeight : ℝ
  newLiquidXPercentage : ℝ

/-- Theorem stating the initial weight of solution Y given the conditions --/
theorem solution_Y_initial_weight (s : Solution) 
  (h1 : s.liquidXPercentage = 0.30)
  (h2 : s.waterPercentage = 0.70)
  (h3 : s.liquidXPercentage + s.waterPercentage = 1)
  (h4 : s.evaporatedWater = 2)
  (h5 : s.addedSolutionWeight = 2)
  (h6 : s.newLiquidXPercentage = 0.36)
  (h7 : s.newLiquidXPercentage * s.initialWeight = 
        s.liquidXPercentage * s.initialWeight + 
        s.liquidXPercentage * s.addedSolutionWeight) :
  s.initialWeight = 10 := by
  sorry


end NUMINAMATH_CALUDE_solution_Y_initial_weight_l1865_186550


namespace NUMINAMATH_CALUDE_siamese_twins_case_l1865_186521

/-- Represents a person on trial --/
structure Defendant where
  guilty : Bool

/-- Represents a pair of defendants --/
structure DefendantPair where
  defendant1 : Defendant
  defendant2 : Defendant
  areConjoined : Bool

/-- Represents the judge's decision --/
def judgeDecision (pair : DefendantPair) : Bool :=
  pair.defendant1.guilty ≠ pair.defendant2.guilty → 
  (pair.defendant1.guilty ∨ pair.defendant2.guilty) → 
  pair.areConjoined

theorem siamese_twins_case (pair : DefendantPair) :
  pair.defendant1.guilty ≠ pair.defendant2.guilty →
  (pair.defendant1.guilty ∨ pair.defendant2.guilty) →
  judgeDecision pair →
  pair.areConjoined := by
  sorry


end NUMINAMATH_CALUDE_siamese_twins_case_l1865_186521


namespace NUMINAMATH_CALUDE_range_of_m_l1865_186565

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, f m x < -m + 4) → m < 5/7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1865_186565


namespace NUMINAMATH_CALUDE_sentence_A_most_appropriate_l1865_186549

/-- Represents a sentence to be evaluated for appropriateness --/
inductive Sentence
| A
| B
| C
| D

/-- Criteria for evaluating the appropriateness of a sentence --/
structure EvaluationCriteria :=
  (identity : Bool)
  (status : Bool)
  (occasion : Bool)
  (audience : Bool)
  (purpose : Bool)
  (respectfulLanguage : Bool)
  (toneOfDiscourse : Bool)

/-- Evaluates a sentence based on the given criteria --/
def evaluateSentence (s : Sentence) (c : EvaluationCriteria) : Bool :=
  match s with
  | Sentence.A => c.identity ∧ c.status ∧ c.occasion ∧ c.audience ∧ c.purpose ∧ c.respectfulLanguage ∧ c.toneOfDiscourse
  | Sentence.B => false
  | Sentence.C => false
  | Sentence.D => false

/-- The criteria used for evaluation --/
def criteria : EvaluationCriteria :=
  { identity := true
  , status := true
  , occasion := true
  , audience := true
  , purpose := true
  , respectfulLanguage := true
  , toneOfDiscourse := true }

/-- Theorem stating that sentence A is the most appropriate --/
theorem sentence_A_most_appropriate :
  ∀ s : Sentence, s ≠ Sentence.A → ¬(evaluateSentence s criteria) ∧ evaluateSentence Sentence.A criteria :=
sorry

end NUMINAMATH_CALUDE_sentence_A_most_appropriate_l1865_186549


namespace NUMINAMATH_CALUDE_existence_of_common_source_l1865_186599

/-- Represents the process of obtaining one number from another through digit manipulation -/
def Obtainable (m n : ℕ) : Prop := sorry

/-- Checks if a natural number contains the digit 5 in its decimal representation -/
def ContainsDigitFive (n : ℕ) : Prop := sorry

theorem existence_of_common_source (S : Finset ℕ) 
  (h1 : S.Nonempty) 
  (h2 : ∀ s ∈ S, ¬ContainsDigitFive s) : 
  ∃ N : ℕ, ∀ s ∈ S, Obtainable s N := by sorry

end NUMINAMATH_CALUDE_existence_of_common_source_l1865_186599


namespace NUMINAMATH_CALUDE_min_specialists_needed_l1865_186510

/-- Represents the number of specialists in energy efficiency -/
def energy_efficiency : ℕ := 95

/-- Represents the number of specialists in waste management -/
def waste_management : ℕ := 80

/-- Represents the number of specialists in water conservation -/
def water_conservation : ℕ := 110

/-- Represents the number of specialists in both energy efficiency and waste management -/
def energy_waste : ℕ := 30

/-- Represents the number of specialists in both waste management and water conservation -/
def waste_water : ℕ := 35

/-- Represents the number of specialists in both energy efficiency and water conservation -/
def energy_water : ℕ := 25

/-- Represents the number of specialists in all three areas -/
def all_three : ℕ := 15

/-- Theorem stating the minimum number of specialists needed -/
theorem min_specialists_needed : 
  energy_efficiency + waste_management + water_conservation - 
  energy_waste - waste_water - energy_water + all_three = 210 := by
  sorry

end NUMINAMATH_CALUDE_min_specialists_needed_l1865_186510


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1865_186507

theorem quadratic_equation_roots (m : ℤ) :
  (∃ a b : ℕ+, a ≠ b ∧ 
    (a : ℝ)^2 + m * (a : ℝ) - m + 1 = 0 ∧
    (b : ℝ)^2 + m * (b : ℝ) - m + 1 = 0) →
  m = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1865_186507


namespace NUMINAMATH_CALUDE_orange_selling_gain_percentage_orange_selling_specific_gain_l1865_186591

/-- Calculates the gain percentage when changing selling rates of oranges -/
theorem orange_selling_gain_percentage 
  (initial_rate : ℝ) 
  (initial_loss_percentage : ℝ)
  (new_rate : ℝ) : ℝ :=
  let cost_price := 1 / (initial_rate * (1 - initial_loss_percentage / 100))
  let new_selling_price := 1 / new_rate
  let gain_percentage := (new_selling_price / cost_price - 1) * 100
  gain_percentage

/-- Proves that the specific change in orange selling rates results in a 44% gain -/
theorem orange_selling_specific_gain : 
  orange_selling_gain_percentage 12 10 7.5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_orange_selling_gain_percentage_orange_selling_specific_gain_l1865_186591


namespace NUMINAMATH_CALUDE_secret_society_friendships_l1865_186503

/-- Represents a member of the secret society -/
structure Member where
  balance : Int

/-- Represents the secret society -/
structure SecretSociety where
  members : Finset Member
  friendships : Finset (Member × Member)
  
/-- A function that represents giving one dollar to all friends -/
def giveDollarToFriends (s : SecretSociety) (m : Member) : SecretSociety :=
  sorry

/-- A predicate that checks if money can be arbitrarily redistributed -/
def canRedistributeArbitrarily (s : SecretSociety) : Prop :=
  sorry

theorem secret_society_friendships 
  (s : SecretSociety) 
  (h1 : s.members.card = 2011) 
  (h2 : canRedistributeArbitrarily s) : 
  s.friendships.card = 2010 :=
sorry

end NUMINAMATH_CALUDE_secret_society_friendships_l1865_186503


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_l1865_186501

/-- An ellipse with foci on the y-axis represented by the equation x²/a - y²/b = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  is_ellipse : a > 0 ∧ b < 0 ∧ -b > a

theorem ellipse_foci_y_axis (e : Ellipse) : Real.sqrt (-e.b) > Real.sqrt e.a := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_l1865_186501


namespace NUMINAMATH_CALUDE_propositions_are_true_l1865_186556

-- Define the propositions
def similar_triangles_equal_perimeters : Prop := sorry
def similar_triangles_equal_angles : Prop := sorry
def sqrt_9_not_negative_3 : Prop := sorry
def diameter_bisects_chord : Prop := sorry
def diameter_bisects_arcs : Prop := sorry

-- Theorem to prove
theorem propositions_are_true :
  (similar_triangles_equal_perimeters ∨ similar_triangles_equal_angles) ∧
  sqrt_9_not_negative_3 ∧
  (diameter_bisects_chord ∧ diameter_bisects_arcs) :=
by
  sorry

end NUMINAMATH_CALUDE_propositions_are_true_l1865_186556


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1865_186505

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) →
  l * w = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1865_186505


namespace NUMINAMATH_CALUDE_daniels_age_l1865_186530

theorem daniels_age (emily_age : ℕ) (brianna_age : ℕ) (daniel_age : ℕ) : 
  emily_age = 48 →
  brianna_age = emily_age / 3 →
  daniel_age = brianna_age - 3 →
  daniel_age * 2 = brianna_age →
  daniel_age = 13 := by
sorry

end NUMINAMATH_CALUDE_daniels_age_l1865_186530


namespace NUMINAMATH_CALUDE_sum_of_z_values_l1865_186515

theorem sum_of_z_values (f : ℝ → ℝ) (h : ∀ x, f (x / 3) = x^2 + x + 1) :
  let z₁ := (2 : ℝ) / 9
  let z₂ := -(1 : ℝ) / 3
  (f (3 * z₁) = 7 ∧ f (3 * z₂) = 7) ∧ z₁ + z₂ = -(1 : ℝ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_z_values_l1865_186515


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1865_186520

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 2| + |x + 1| ≤ 5} = Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1865_186520


namespace NUMINAMATH_CALUDE_problem_solution_l1865_186536

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x + f a (2 * x)

noncomputable def g (x : ℝ) : ℝ := f 2 x - f 2 (-x)

theorem problem_solution :
  (∀ a : ℝ, a > 0 → (∃ x : ℝ, F a x = 3) → (∀ y : ℝ, F a y ≥ 3) → a = 6) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → 2 * m + 3 * n = (⨆ x, g x) →
    (∀ p q : ℝ, p > 0 → q > 0 → 1 / p + 2 / (3 * q) ≥ 2) ∧
    (∃ r s : ℝ, r > 0 ∧ s > 0 ∧ 1 / r + 2 / (3 * s) = 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1865_186536


namespace NUMINAMATH_CALUDE_apple_pairing_l1865_186566

theorem apple_pairing (weights : Fin 300 → ℝ) 
  (h_positive : ∀ i, weights i > 0)
  (h_ratio : ∀ i j, weights i ≤ 3 * weights j) :
  ∃ (pairs : Fin 150 → Fin 300 × Fin 300),
    (∀ i, (pairs i).1 ≠ (pairs i).2) ∧
    (∀ i, i ≠ j → (pairs i).1 ≠ (pairs j).1 ∧ (pairs i).1 ≠ (pairs j).2 ∧
                  (pairs i).2 ≠ (pairs j).1 ∧ (pairs i).2 ≠ (pairs j).2) ∧
    (∀ i j, weights (pairs i).1 + weights (pairs i).2 ≤ 
            2 * (weights (pairs j).1 + weights (pairs j).2)) :=
sorry

end NUMINAMATH_CALUDE_apple_pairing_l1865_186566


namespace NUMINAMATH_CALUDE_expand_expression_l1865_186570

theorem expand_expression (x : ℝ) : 5 * (9 * x^3 - 4 * x^2 + 3 * x - 7) = 45 * x^3 - 20 * x^2 + 15 * x - 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1865_186570


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_is_lower_bound_l1865_186568

theorem min_value_of_function (x : ℝ) (h : x > 1) : 
  2 * x + 2 / (x - 1) ≥ 6 := by
  sorry

theorem min_value_is_lower_bound (ε : ℝ) (hε : ε > 0) : 
  ∃ x : ℝ, x > 1 ∧ 2 * x + 2 / (x - 1) < 6 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_is_lower_bound_l1865_186568


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l1865_186582

/-- Calculates the average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 80) :
  (speed1 + speed2) / 2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l1865_186582


namespace NUMINAMATH_CALUDE_revenue_difference_l1865_186584

/-- The revenue generated by a single jersey -/
def jersey_revenue : ℕ := 210

/-- The revenue generated by a single t-shirt -/
def tshirt_revenue : ℕ := 240

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 177

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 23

/-- The difference in revenue between t-shirts and jerseys -/
theorem revenue_difference : 
  tshirt_revenue * tshirts_sold - jersey_revenue * jerseys_sold = 37650 := by
  sorry

end NUMINAMATH_CALUDE_revenue_difference_l1865_186584


namespace NUMINAMATH_CALUDE_odd_function_property_l1865_186572

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h_odd : IsOdd f)
    (h_slope : ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 1)
    (m : ℝ) (h_m : f m > m) : m > 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1865_186572


namespace NUMINAMATH_CALUDE_factor_implies_a_value_l1865_186519

theorem factor_implies_a_value (a b : ℤ) :
  (∀ x : ℝ, (x^2 - x - 1 = 0) → (a*x^19 + b*x^18 + 1 = 0)) →
  a = 1597 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_a_value_l1865_186519


namespace NUMINAMATH_CALUDE_intersection_points_l1865_186500

-- Define the equations
def eq1 (x y : ℝ) : Prop := 4 + (x + 2) * y = x^2
def eq2 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 16

-- Theorem stating the intersection points
theorem intersection_points :
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1 = -2 ∧ y1 = -4) ∧
    (x2 = -2 ∧ y2 = 4) ∧
    (x3 = 2 ∧ y3 = 0) ∧
    eq1 x1 y1 ∧ eq2 x1 y1 ∧
    eq1 x2 y2 ∧ eq2 x2 y2 ∧
    eq1 x3 y3 ∧ eq2 x3 y3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_l1865_186500


namespace NUMINAMATH_CALUDE_rhombus_area_triple_diagonals_l1865_186596

/-- The area of a rhombus with diagonals that are 3 times longer than a rhombus
    with diagonals 6 cm and 4 cm is 108 cm². -/
theorem rhombus_area_triple_diagonals (d1 d2 : ℝ) : 
  d1 = 6 → d2 = 4 → (3 * d1 * 3 * d2) / 2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_triple_diagonals_l1865_186596


namespace NUMINAMATH_CALUDE_g_at_negative_two_l1865_186580

/-- The function g is defined as g(x) = 2x^2 - 3x + 1 for all real x. -/
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

/-- Theorem: The value of g(-2) is 15. -/
theorem g_at_negative_two : g (-2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l1865_186580


namespace NUMINAMATH_CALUDE_wade_tips_theorem_l1865_186585

def tips_per_customer : ℕ := 2
def friday_customers : ℕ := 28
def sunday_customers : ℕ := 36

def saturday_customers : ℕ := 3 * friday_customers

def total_tips : ℕ := tips_per_customer * (friday_customers + saturday_customers + sunday_customers)

theorem wade_tips_theorem : total_tips = 296 := by
  sorry

end NUMINAMATH_CALUDE_wade_tips_theorem_l1865_186585


namespace NUMINAMATH_CALUDE_difference_of_equal_distinct_prime_factors_l1865_186516

def distinctPrimeFactors (n : ℕ) : Finset ℕ :=
  sorry

theorem difference_of_equal_distinct_prime_factors :
  ∀ n : ℕ, ∃ a b : ℕ, n = a - b ∧ (distinctPrimeFactors a).card = (distinctPrimeFactors b).card :=
sorry

end NUMINAMATH_CALUDE_difference_of_equal_distinct_prime_factors_l1865_186516


namespace NUMINAMATH_CALUDE_partnership_profit_l1865_186573

/-- Represents a partnership between two individuals -/
structure Partnership where
  investmentA : ℕ
  investmentB : ℕ
  periodA : ℕ
  periodB : ℕ

/-- Calculates the total profit of a partnership -/
def totalProfit (p : Partnership) (profitB : ℕ) : ℕ :=
  7 * profitB

theorem partnership_profit (p : Partnership) (h1 : p.investmentA = 3 * p.investmentB)
    (h2 : p.periodA = 2 * p.periodB) (h3 : 4000 = p.investmentB * p.periodB) :
  totalProfit p 4000 = 28000 := by
  sorry

#eval totalProfit ⟨30000, 10000, 10, 5⟩ 4000

end NUMINAMATH_CALUDE_partnership_profit_l1865_186573


namespace NUMINAMATH_CALUDE_altitude_angle_bisector_median_concurrent_l1865_186590

/-- Triangle ABC with sides a, b, c -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (a b c : ℝ)
  (side_a : dist B C = a)
  (side_b : dist C A = b)
  (side_c : dist A B = c)

/-- Altitude from A to BC -/
def altitude (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

/-- Angle bisector from B -/
def angle_bisector (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

/-- Median from C to AB -/
def median (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

/-- Three lines are concurrent -/
def concurrent (l₁ l₂ l₃ : ℝ × ℝ → ℝ × ℝ) : Prop := sorry

theorem altitude_angle_bisector_median_concurrent (t : Triangle) :
  concurrent (altitude t) (angle_bisector t) (median t) ↔
  t.a^2 * (t.a - t.c) = (t.b^2 - t.c^2) * (t.a + t.c) :=
sorry

end NUMINAMATH_CALUDE_altitude_angle_bisector_median_concurrent_l1865_186590


namespace NUMINAMATH_CALUDE_lukes_coin_piles_l1865_186526

/-- Given that Luke has an equal number of piles of quarters and dimes,
    each pile contains 3 coins, and the total number of coins is 30,
    prove that the number of piles of quarters is 5. -/
theorem lukes_coin_piles (num_quarter_piles num_dime_piles : ℕ)
  (h1 : num_quarter_piles = num_dime_piles)
  (h2 : ∀ pile, pile = num_quarter_piles ∨ pile = num_dime_piles → 3 * pile = num_quarter_piles * 3 + num_dime_piles * 3)
  (h3 : num_quarter_piles * 3 + num_dime_piles * 3 = 30) :
  num_quarter_piles = 5 := by
  sorry

end NUMINAMATH_CALUDE_lukes_coin_piles_l1865_186526


namespace NUMINAMATH_CALUDE_brown_mice_count_l1865_186542

theorem brown_mice_count (total : ℕ) (white : ℕ) : 
  (2 : ℚ) / 3 * total = white → white = 14 → total - white = 7 := by
  sorry

end NUMINAMATH_CALUDE_brown_mice_count_l1865_186542


namespace NUMINAMATH_CALUDE_point_reflection_origin_l1865_186567

/-- Given a point P(4, -3) in the Cartesian coordinate system,
    its coordinates with respect to the origin are (-4, 3). -/
theorem point_reflection_origin : 
  let P : ℝ × ℝ := (4, -3)
  let P_reflected : ℝ × ℝ := (-4, 3)
  P_reflected = (-(P.1), -(P.2)) :=
by sorry

end NUMINAMATH_CALUDE_point_reflection_origin_l1865_186567


namespace NUMINAMATH_CALUDE_problem_statement_l1865_186576

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -8)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 12)
  (h3 : a * b * c = 1) :
  b / (a + b) + c / (b + c) + a / (c + a) = -8.5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1865_186576


namespace NUMINAMATH_CALUDE_blocks_used_for_tower_l1865_186544

theorem blocks_used_for_tower (initial_blocks : ℕ) (remaining_blocks : ℕ) : 
  initial_blocks = 97 → remaining_blocks = 72 → initial_blocks - remaining_blocks = 25 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_for_tower_l1865_186544


namespace NUMINAMATH_CALUDE_mirror_position_l1865_186538

theorem mirror_position (wall_width mirror_width : ℝ) (h1 : wall_width = 26) (h2 : mirror_width = 4) :
  let distance := (wall_width - mirror_width) / 2
  distance = 11 := by sorry

end NUMINAMATH_CALUDE_mirror_position_l1865_186538


namespace NUMINAMATH_CALUDE_positive_numbers_l1865_186559

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (pairwise_sum_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l1865_186559


namespace NUMINAMATH_CALUDE_percentage_increase_l1865_186551

theorem percentage_increase (original : ℝ) (new : ℝ) :
  original = 30 ∧ new = 40 →
  (new - original) / original * 100 = 100 / 3 :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_l1865_186551


namespace NUMINAMATH_CALUDE_quadratic_properties_l1865_186583

-- Define the quadratic function
def quadratic (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the points
def point_A : ℝ × ℝ := (2, 0)
def point_B (n y1 : ℝ) : ℝ × ℝ := (3*n - 4, y1)
def point_C (n y2 : ℝ) : ℝ × ℝ := (5*n + 6, y2)

theorem quadratic_properties (b c n y1 y2 : ℝ) 
  (h1 : quadratic 2 b c = 0)  -- A(2,0) is on the curve
  (h2 : quadratic (3*n - 4) b c = y1)  -- B is on the curve
  (h3 : quadratic (5*n + 6) b c = y2)  -- C is on the curve
  (h4 : ∀ x, quadratic x b c ≥ quadratic 2 b c)  -- A is the vertex
  (h5 : n < -5) :  -- Given condition
  -- 1) The function can be expressed as y = x^2 - 4x + 4
  (∀ x, quadratic x b c = x^2 - 4*x + 4) ∧
  -- 2) If y1 = y2, then b+c < -38
  (y1 = y2 → b + c < -38) ∧
  -- 3) If c > 0, then y1 < y2
  (c > 0 → y1 < y2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1865_186583


namespace NUMINAMATH_CALUDE_runners_meet_time_l1865_186517

/-- Represents a runner with a constant speed -/
structure Runner where
  speed : ℝ

/-- Represents the circular track -/
structure Track where
  length : ℝ

/-- Calculates the time when all runners meet again -/
def meeting_time (track : Track) (runners : List Runner) : ℝ :=
  sorry

theorem runners_meet_time (track : Track) (runners : List Runner) :
  track.length = 600 ∧
  runners = [
    Runner.mk 4.5,
    Runner.mk 4.9,
    Runner.mk 5.1
  ] →
  meeting_time track runners = 3000 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_time_l1865_186517


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1865_186581

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors a and b
variable (a b : V)

-- State the theorem
theorem angle_between_vectors 
  (h1 : ‖a‖ = Real.sqrt 3)
  (h2 : ‖b‖ = 1)
  (h3 : ‖a - 2 • b‖ = 1) :
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1865_186581


namespace NUMINAMATH_CALUDE_zeros_after_one_in_factorial_power_is_2400_l1865_186528

/-- The number of zeros following the digit '1' in the decimal expansion of (100!)^100 -/
def zeros_after_one_in_factorial_power : ℕ :=
  let factors_of_five : ℕ := (100 / 5) + (100 / 25)
  let zeros_in_factorial : ℕ := factors_of_five
  zeros_in_factorial * 100

/-- Theorem stating that the number of zeros after '1' in (100!)^100 is 2400 -/
theorem zeros_after_one_in_factorial_power_is_2400 :
  zeros_after_one_in_factorial_power = 2400 := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_factorial_power_is_2400_l1865_186528


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1865_186586

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (k : ℚ), k * (a.val * Real.sqrt 6 + b.val * Real.sqrt 8) / c.val = 
    Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) →
  (∀ (x y z : ℕ+), (∃ (l : ℚ), l * (x.val * Real.sqrt 6 + y.val * Real.sqrt 8) / z.val = 
    Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) → 
    c.val ≤ z.val) →
  a.val + b.val + c.val = 192 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1865_186586


namespace NUMINAMATH_CALUDE_triangle_properties_l1865_186587

theorem triangle_properties (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ 
  b = 5 ∧ 
  c = 4 * Real.sqrt 2 ∧
  a^2 + b^2 = c^2 ∧
  a + b > c ∧
  b + c > a ∧
  c + a > b := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1865_186587


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1865_186522

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 1 → x > 0) ∧ (∃ x, x > 0 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1865_186522


namespace NUMINAMATH_CALUDE_fraction_reducibility_l1865_186575

def is_reducible (a : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d ∣ (2 * a + 5) ∧ d ∣ (3 * a + 4)

theorem fraction_reducibility (a : ℕ) :
  is_reducible a ↔ ∃ k : ℕ, a = 7 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_reducibility_l1865_186575


namespace NUMINAMATH_CALUDE_data_analysis_l1865_186578

def data : List ℝ := [7, 11, 10, 11, 6, 14, 11, 10, 11, 9]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_analysis (d : List ℝ) (h : d = data) : 
  mode d = 11 ∧ 
  median d ≠ 10 ∧ 
  mean d = 10 ∧ 
  variance d = 4.6 := by sorry

end NUMINAMATH_CALUDE_data_analysis_l1865_186578


namespace NUMINAMATH_CALUDE_triangle_area_is_63_l1865_186529

/-- The area of a triangle formed by three lines -/
def triangleArea (m1 m2 : ℚ) : ℚ :=
  let x1 : ℚ := 1
  let y1 : ℚ := 1
  let x2 : ℚ := (14/5)
  let y2 : ℚ := (23/5)
  let x3 : ℚ := (11/2)
  let y3 : ℚ := (5/2)
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The theorem stating that the area of the triangle is 6.3 -/
theorem triangle_area_is_63 :
  triangleArea (3/2) (1/3) = 63/10 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_63_l1865_186529


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1865_186579

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 4*x} = {0, 4} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1865_186579


namespace NUMINAMATH_CALUDE_stickers_needed_for_prizes_l1865_186563

def christine_stickers : ℕ := 2500
def robert_stickers : ℕ := 1750
def small_prize_requirement : ℕ := 4000
def medium_prize_requirement : ℕ := 7000
def large_prize_requirement : ℕ := 10000

def total_stickers : ℕ := christine_stickers + robert_stickers

theorem stickers_needed_for_prizes :
  (max 0 (small_prize_requirement - total_stickers) = 0) ∧
  (max 0 (medium_prize_requirement - total_stickers) = 2750) ∧
  (max 0 (large_prize_requirement - total_stickers) = 5750) := by
  sorry

end NUMINAMATH_CALUDE_stickers_needed_for_prizes_l1865_186563


namespace NUMINAMATH_CALUDE_unique_sum_of_squares_l1865_186531

theorem unique_sum_of_squares (p q r : ℕ+) : 
  p + q + r = 30 →
  Nat.gcd p.val q.val + Nat.gcd q.val r.val + Nat.gcd r.val p.val = 10 →
  p ^ 2 + q ^ 2 + r ^ 2 = 584 := by
  sorry

end NUMINAMATH_CALUDE_unique_sum_of_squares_l1865_186531


namespace NUMINAMATH_CALUDE_messenger_catches_up_l1865_186594

/-- Represents the scenario of a messenger catching up to Ilya Muromets --/
def catchUpScenario (ilyaSpeed : ℝ) : Prop :=
  let messengerSpeed := 2 * ilyaSpeed
  let horseSpeed := 5 * messengerSpeed
  let initialDelay := 10 -- seconds
  let ilyaDistance := ilyaSpeed * initialDelay
  let horseDistance := horseSpeed * initialDelay
  let totalDistance := ilyaDistance + horseDistance
  let relativeSpeed := messengerSpeed - ilyaSpeed
  let catchUpTime := totalDistance / relativeSpeed
  catchUpTime = 110

/-- Theorem stating that under the given conditions, 
    the messenger catches up to Ilya Muromets in 110 seconds --/
theorem messenger_catches_up (ilyaSpeed : ℝ) (ilyaSpeed_pos : 0 < ilyaSpeed) :
  catchUpScenario ilyaSpeed := by
  sorry

#check messenger_catches_up

end NUMINAMATH_CALUDE_messenger_catches_up_l1865_186594


namespace NUMINAMATH_CALUDE_correct_calculation_l1865_186562

theorem correct_calculation (x : ℚ) (h : 6 * x = 42) : 3 * x = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1865_186562


namespace NUMINAMATH_CALUDE_mini_crossword_probability_l1865_186537

/-- Represents a crossword puzzle -/
structure Crossword :=
  (size : Nat)
  (num_clues : Nat)
  (prob_know_clue : ℚ)

/-- Calculates the probability of filling in all unshaded squares in a crossword -/
def probability_fill_crossword (c : Crossword) : ℚ :=
  sorry

/-- The specific crossword from the problem -/
def mini_crossword : Crossword :=
  { size := 5
  , num_clues := 10
  , prob_know_clue := 1/2
  }

/-- Theorem stating the probability of filling in all unshaded squares in the mini crossword -/
theorem mini_crossword_probability :
  probability_fill_crossword mini_crossword = 11/128 :=
sorry

end NUMINAMATH_CALUDE_mini_crossword_probability_l1865_186537
