import Mathlib

namespace NUMINAMATH_CALUDE_missile_interception_time_l1313_131300

/-- The time taken for a missile to intercept a plane -/
theorem missile_interception_time 
  (r : ℝ) -- radius of the circular path
  (v : ℝ) -- speed of both the plane and the missile
  (h : r = 10 ∧ v = 1000) -- specific values given in the problem
  : (r * π) / (2 * v) = π / 200 := by
  sorry

#check missile_interception_time

end NUMINAMATH_CALUDE_missile_interception_time_l1313_131300


namespace NUMINAMATH_CALUDE_sum_45_52_base4_l1313_131326

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number in base 10 -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_45_52_base4 : 
  toBase4 (45 + 52) = [1, 2, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_45_52_base4_l1313_131326


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l1313_131388

/-- Represents the speed of a rower in different conditions -/
structure RowerSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the downstream speed of a rower given upstream and still water speeds -/
def calculateDownstreamSpeed (upstream stillWater : ℝ) : ℝ :=
  2 * stillWater - upstream

/-- Theorem stating that given the upstream and still water speeds, 
    the calculated downstream speed is correct -/
theorem downstream_speed_calculation 
  (speed : RowerSpeed) 
  (h1 : speed.upstream = 25)
  (h2 : speed.stillWater = 32) :
  speed.downstream = calculateDownstreamSpeed speed.upstream speed.stillWater ∧ 
  speed.downstream = 39 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l1313_131388


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1313_131362

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 ≥ 1 → (x ≥ 0 ∨ x ≤ -1))) ↔
  (∀ x : ℝ, (-1 < x ∧ x < 0 → x^2 < 1)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1313_131362


namespace NUMINAMATH_CALUDE_sum_ages_after_ten_years_l1313_131347

/-- Given Ann's age and Tom's age relative to Ann's, calculate the sum of their ages after a certain number of years. -/
def sum_ages_after_years (ann_age : ℕ) (tom_age_multiplier : ℕ) (years_later : ℕ) : ℕ :=
  (ann_age + years_later) + (ann_age * tom_age_multiplier + years_later)

/-- Prove that given Ann is 6 years old and Tom is twice her age, the sum of their ages 10 years later will be 38 years. -/
theorem sum_ages_after_ten_years :
  sum_ages_after_years 6 2 10 = 38 := by
  sorry

end NUMINAMATH_CALUDE_sum_ages_after_ten_years_l1313_131347


namespace NUMINAMATH_CALUDE_number_of_factors_of_48_l1313_131321

theorem number_of_factors_of_48 : Nat.card (Nat.divisors 48) = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_48_l1313_131321


namespace NUMINAMATH_CALUDE_workshop_attendance_workshop_attendance_proof_l1313_131366

theorem workshop_attendance : ℕ → Prop :=
  fun total =>
    ∃ (wolf nobel wolf_nobel non_wolf_nobel non_wolf_non_nobel : ℕ),
      -- Total Wolf Prize laureates
      wolf = 31 ∧
      -- Wolf Prize laureates who are also Nobel Prize laureates
      wolf_nobel = 18 ∧
      -- Total Nobel Prize laureates
      nobel = 29 ∧
      -- Difference between Nobel (non-Wolf) and non-Nobel (non-Wolf)
      non_wolf_nobel = non_wolf_non_nobel + 3 ∧
      -- Total scientists is sum of all categories
      total = wolf + non_wolf_nobel + non_wolf_non_nobel ∧
      -- Consistency check for Nobel laureates
      nobel = wolf_nobel + non_wolf_nobel ∧
      -- The total number of scientists is 50
      total = 50

theorem workshop_attendance_proof : workshop_attendance 50 := by
  sorry

end NUMINAMATH_CALUDE_workshop_attendance_workshop_attendance_proof_l1313_131366


namespace NUMINAMATH_CALUDE_baker_cakes_problem_l1313_131307

theorem baker_cakes_problem (initial_cakes : ℕ) 
  (h1 : initial_cakes - 78 + 31 = initial_cakes) 
  (h2 : 78 = 31 + 47) : 
  initial_cakes = 109 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l1313_131307


namespace NUMINAMATH_CALUDE_pickle_ratio_l1313_131322

/-- Prove the ratio of pickle slices Tammy can eat to Sammy can eat -/
theorem pickle_ratio (sammy tammy ron : ℕ) : 
  sammy = 15 → 
  ron = 24 → 
  ron = (80 * tammy) / 100 → 
  tammy / sammy = 2 := by
  sorry

end NUMINAMATH_CALUDE_pickle_ratio_l1313_131322


namespace NUMINAMATH_CALUDE_binomial_coefficient_geometric_mean_l1313_131340

theorem binomial_coefficient_geometric_mean (a : ℚ) : 
  (∃ (k : ℕ), k = 7 ∧ 
    (Nat.choose k 4 * a^3)^2 = (Nat.choose k 5 * a^2) * (Nat.choose k 2 * a^5)) ↔ 
  a = 25 / 9 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_geometric_mean_l1313_131340


namespace NUMINAMATH_CALUDE_baseball_cost_l1313_131359

def marbles_cost : ℚ := 9.05
def football_cost : ℚ := 4.95
def total_cost : ℚ := 20.52

theorem baseball_cost : total_cost - (marbles_cost + football_cost) = 6.52 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cost_l1313_131359


namespace NUMINAMATH_CALUDE_cos_600_degrees_l1313_131387

theorem cos_600_degrees : Real.cos (600 * π / 180) = -(1/2) := by
  sorry

end NUMINAMATH_CALUDE_cos_600_degrees_l1313_131387


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1313_131386

theorem smallest_number_of_eggs (total_eggs : ℕ) (num_containers : ℕ) : 
  total_eggs > 150 →
  total_eggs = 12 * num_containers - 3 →
  (∀ n : ℕ, n < num_containers → 12 * n - 3 ≤ 150) →
  total_eggs = 153 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1313_131386


namespace NUMINAMATH_CALUDE_equation_solution_l1313_131335

theorem equation_solution :
  ∃! x : ℝ, (3 : ℝ)^x * (9 : ℝ)^x = (27 : ℝ)^(x - 4) :=
by
  use -6
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1313_131335


namespace NUMINAMATH_CALUDE_largest_s_proof_l1313_131317

/-- The largest possible value of s for which there exist regular polygons P1 (r-gon) and P2 (s-gon)
    satisfying the given conditions -/
def largest_s : ℕ := 117

theorem largest_s_proof (r s : ℕ) : 
  r ≥ s → 
  s ≥ 3 → 
  (r - 2) * s * 60 = (s - 2) * r * 59 → 
  s ≤ largest_s := by
  sorry

#check largest_s_proof

end NUMINAMATH_CALUDE_largest_s_proof_l1313_131317


namespace NUMINAMATH_CALUDE_lottery_expected_wins_l1313_131312

/-- A lottery with a winning probability of 1/4 -/
structure Lottery where
  win_prob : ℝ
  win_prob_eq : win_prob = 1/4

/-- The expected number of winning tickets when drawing n tickets -/
def expected_wins (L : Lottery) (n : ℕ) : ℝ := n * L.win_prob

theorem lottery_expected_wins (L : Lottery) : expected_wins L 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lottery_expected_wins_l1313_131312


namespace NUMINAMATH_CALUDE_shopping_cost_after_discount_l1313_131396

/-- Calculate the total cost after discount for a shopping trip --/
theorem shopping_cost_after_discount :
  let tshirt_cost : ℕ := 20
  let pants_cost : ℕ := 80
  let shoes_cost : ℕ := 150
  let discount_rate : ℚ := 1 / 10
  let tshirt_quantity : ℕ := 4
  let pants_quantity : ℕ := 3
  let shoes_quantity : ℕ := 2
  let total_cost_before_discount : ℕ := 
    tshirt_cost * tshirt_quantity + 
    pants_cost * pants_quantity + 
    shoes_cost * shoes_quantity
  let discount_amount : ℚ := discount_rate * total_cost_before_discount
  let total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount
  total_cost_after_discount = 558 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_after_discount_l1313_131396


namespace NUMINAMATH_CALUDE_inequality_range_l1313_131399

theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, (x^2 + a*x > 4*x + a - 3) ↔ (x > 3 ∨ x < -1) := by sorry

end NUMINAMATH_CALUDE_inequality_range_l1313_131399


namespace NUMINAMATH_CALUDE_meat_for_hamburgers_l1313_131356

/-- Given that 5 pounds of meat can make 12 hamburgers, 
    prove that 15 pounds of meat are needed to make 36 hamburgers. -/
theorem meat_for_hamburgers : 
  ∀ (meat_per_batch : ℝ) (hamburgers_per_batch : ℝ) (total_hamburgers : ℝ),
    meat_per_batch = 5 →
    hamburgers_per_batch = 12 →
    total_hamburgers = 36 →
    (meat_per_batch / hamburgers_per_batch) * total_hamburgers = 15 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_hamburgers_l1313_131356


namespace NUMINAMATH_CALUDE_disneyland_attractions_permutations_l1313_131304

theorem disneyland_attractions_permutations : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_disneyland_attractions_permutations_l1313_131304


namespace NUMINAMATH_CALUDE_citizenship_test_study_time_l1313_131331

/-- Calculates the total study time in hours for a citizenship test -/
theorem citizenship_test_study_time
  (total_questions : ℕ)
  (multiple_choice_questions : ℕ)
  (fill_in_blank_questions : ℕ)
  (time_per_multiple_choice : ℕ)
  (time_per_fill_in_blank : ℕ)
  (h1 : total_questions = 60)
  (h2 : multiple_choice_questions = 30)
  (h3 : fill_in_blank_questions = 30)
  (h4 : time_per_multiple_choice = 15)
  (h5 : time_per_fill_in_blank = 25)
  (h6 : total_questions = multiple_choice_questions + fill_in_blank_questions) :
  (multiple_choice_questions * time_per_multiple_choice +
   fill_in_blank_questions * time_per_fill_in_blank) / 60 = 20 := by
sorry

end NUMINAMATH_CALUDE_citizenship_test_study_time_l1313_131331


namespace NUMINAMATH_CALUDE_courtyard_length_l1313_131369

theorem courtyard_length (width : ℝ) (tiles_per_sqft : ℝ) 
  (green_ratio : ℝ) (red_ratio : ℝ) (green_cost : ℝ) (red_cost : ℝ) 
  (total_cost : ℝ) (L : ℝ) : 
  width = 25 ∧ 
  tiles_per_sqft = 4 ∧ 
  green_ratio = 0.4 ∧ 
  red_ratio = 0.6 ∧ 
  green_cost = 3 ∧ 
  red_cost = 1.5 ∧ 
  total_cost = 2100 ∧ 
  total_cost = (green_ratio * tiles_per_sqft * L * width * green_cost) + 
               (red_ratio * tiles_per_sqft * L * width * red_cost) → 
  L = 10 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l1313_131369


namespace NUMINAMATH_CALUDE_factorization_theorem_l1313_131330

theorem factorization_theorem (x y z : ℝ) :
  x^3 * (y^2 - z^2) - y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x*y + z^2 - z*x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l1313_131330


namespace NUMINAMATH_CALUDE_poker_loss_l1313_131377

theorem poker_loss (initial_amount winnings debt : ℤ) : 
  initial_amount = 100 → winnings = 65 → debt = 50 → 
  (initial_amount + winnings + debt) = 215 := by
sorry

end NUMINAMATH_CALUDE_poker_loss_l1313_131377


namespace NUMINAMATH_CALUDE_max_value_inequality_l1313_131315

theorem max_value_inequality (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1/2)
  (b_ge : b ≥ -3/2)
  (c_ge : c ≥ -2) :
  Real.sqrt (4*a + 2) + Real.sqrt (4*b + 6) + Real.sqrt (4*c + 8) ≤ 2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1313_131315


namespace NUMINAMATH_CALUDE_fraction_equality_l1313_131345

theorem fraction_equality {a b c : ℝ} (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1313_131345


namespace NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1313_131365

/-- A geometric progression where each term is positive and any given term 
    is equal to the sum of the next three following terms. -/
structure GeometricProgression where
  a : ℝ  -- First term
  r : ℝ  -- Common ratio
  a_pos : 0 < a  -- Each term is positive
  r_pos : 0 < r  -- Common ratio is positive (to ensure all terms are positive)
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

theorem geometric_progression_common_ratio 
  (gp : GeometricProgression) : 
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 ∧ 
  abs (gp.r - 0.5437) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1313_131365


namespace NUMINAMATH_CALUDE_negative_one_half_less_than_negative_one_third_l1313_131325

theorem negative_one_half_less_than_negative_one_third :
  -1/2 < -1/3 := by sorry

end NUMINAMATH_CALUDE_negative_one_half_less_than_negative_one_third_l1313_131325


namespace NUMINAMATH_CALUDE_range_of_a_given_false_proposition_l1313_131392

theorem range_of_a_given_false_proposition : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 ≤ 0) → 
  (∀ a : ℝ, -1 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_false_proposition_l1313_131392


namespace NUMINAMATH_CALUDE_triangle_is_right_angle_l1313_131324

/-- 
Given a triangle ABC where a, b, and c are the lengths of the sides opposite to angles A, B, and C respectively,
if 1 + cos A = (b + c) / c, then the triangle is a right triangle.
-/
theorem triangle_is_right_angle 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_cos : 1 + Real.cos A = (b + c) / c)
  : a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angle_l1313_131324


namespace NUMINAMATH_CALUDE_range_of_a_l1313_131397

-- Define the propositions
def proposition_p (a : ℝ) : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1) 1 → a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a*x + 2 < 0

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, proposition_p a ∧ ¬proposition_q a ↔ a ∈ Set.Icc (-2 * Real.sqrt 2) (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1313_131397


namespace NUMINAMATH_CALUDE_box_surface_area_l1313_131306

/-- Represents the dimensions of a rectangular sheet. -/
structure SheetDimensions where
  length : ℕ
  width : ℕ

/-- Represents the size of the square cut from each corner. -/
def CornerCutSize : ℕ := 4

/-- Calculates the surface area of the interior of the box formed by folding a rectangular sheet
    with squares cut from each corner. -/
def interiorSurfaceArea (sheet : SheetDimensions) : ℕ :=
  sheet.length * sheet.width - 4 * (CornerCutSize * CornerCutSize)

/-- Theorem stating that the surface area of the interior of the box is 936 square units. -/
theorem box_surface_area :
  interiorSurfaceArea ⟨25, 40⟩ = 936 := by sorry

end NUMINAMATH_CALUDE_box_surface_area_l1313_131306


namespace NUMINAMATH_CALUDE_sequence_non_positive_l1313_131376

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h : ∀ k : ℕ, k ∈ Finset.range (n - 1) → a k - 2 * a (k + 1) + a (k + 2) ≥ 0) : 
  ∀ k : ℕ, k ≤ n → a k ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l1313_131376


namespace NUMINAMATH_CALUDE_decimal_difference_l1313_131314

-- Define the repeating decimal 0.727272...
def repeating_decimal : ℚ := 72 / 99

-- Define the terminating decimal 0.72
def terminating_decimal : ℚ := 72 / 100

-- Theorem statement
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 2 / 275 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l1313_131314


namespace NUMINAMATH_CALUDE_productivity_increase_l1313_131358

/-- Represents the productivity level during a work shift -/
structure Productivity where
  planned : ℝ
  reduced : ℝ

/-- Represents a work shift with its duration and productivity levels -/
structure Shift where
  duration : ℝ
  plannedHours : ℝ
  productivity : Productivity

/-- Calculates the total work done during a shift -/
def totalWork (s : Shift) : ℝ :=
  s.plannedHours * s.productivity.planned +
  (s.duration - s.plannedHours) * s.productivity.reduced

/-- Theorem stating the productivity increase when extending the workday -/
theorem productivity_increase
  (initialShift : Shift)
  (extendedShift : Shift)
  (h1 : initialShift.duration = 8)
  (h2 : initialShift.plannedHours = 6)
  (h3 : initialShift.productivity.planned = 1)
  (h4 : initialShift.productivity.reduced = 0.75)
  (h5 : extendedShift.duration = 9)
  (h6 : extendedShift.plannedHours = 6)
  (h7 : extendedShift.productivity.planned = 1)
  (h8 : extendedShift.productivity.reduced = 0.7) :
  (totalWork extendedShift - totalWork initialShift) / totalWork initialShift = 0.08 := by
  sorry


end NUMINAMATH_CALUDE_productivity_increase_l1313_131358


namespace NUMINAMATH_CALUDE_chosen_number_proof_l1313_131305

theorem chosen_number_proof (x : ℝ) : (x / 12) - 240 = 8 ↔ x = 2976 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l1313_131305


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1313_131381

-- Define a plane
class Plane where
  -- Add any necessary properties for a plane

-- Define a line
class Line where
  -- Add any necessary properties for a line

-- Define perpendicularity between a line and a plane
def perpendicular_to_plane (l : Line) (p : Plane) : Prop :=
  sorry -- Definition of perpendicularity between a line and a plane

-- Define parallel lines
def parallel_lines (l1 l2 : Line) : Prop :=
  sorry -- Definition of parallel lines

-- Theorem statement
theorem perpendicular_lines_parallel (p : Plane) (l1 l2 : Line) :
  perpendicular_to_plane l1 p → perpendicular_to_plane l2 p → parallel_lines l1 l2 :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1313_131381


namespace NUMINAMATH_CALUDE_expression_evaluation_l1313_131390

theorem expression_evaluation : (32 * 2 - 16) / (8 - (2 * 3)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1313_131390


namespace NUMINAMATH_CALUDE_right_triangle_height_l1313_131373

theorem right_triangle_height (base height hypotenuse : ℝ) : 
  base = 4 →
  base + height + hypotenuse = 12 →
  base^2 + height^2 = hypotenuse^2 →
  height = 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_height_l1313_131373


namespace NUMINAMATH_CALUDE_correct_formula_l1313_131363

def f (x : ℝ) : ℝ := 200 - 10*x - 10*x^2

theorem correct_formula : 
  f 0 = 200 ∧ f 1 = 170 ∧ f 2 = 120 ∧ f 3 = 50 ∧ f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_formula_l1313_131363


namespace NUMINAMATH_CALUDE_parabola_shift_l1313_131395

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift :
  let original := Parabola.mk 5 0 0
  let shifted := shift original 2 3
  shifted = Parabola.mk 5 (-20) 23 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l1313_131395


namespace NUMINAMATH_CALUDE_food_cost_calculation_l1313_131398

def hospital_bill_breakdown (total : ℝ) (medication_percent : ℝ) (overnight_percent : ℝ) (ambulance : ℝ) : ℝ := 
  let medication := medication_percent * total
  let remaining_after_medication := total - medication
  let overnight := overnight_percent * remaining_after_medication
  let food := total - medication - overnight - ambulance
  food

theorem food_cost_calculation :
  hospital_bill_breakdown 5000 0.5 0.25 1700 = 175 := by
  sorry

end NUMINAMATH_CALUDE_food_cost_calculation_l1313_131398


namespace NUMINAMATH_CALUDE_number_division_problem_l1313_131320

theorem number_division_problem (x y : ℚ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - 24) / y = 3) : 
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l1313_131320


namespace NUMINAMATH_CALUDE_group_size_correct_l1313_131329

/-- The number of people in a group where:
  1. The average weight increase is 3 kg when a new person replaces one person.
  2. The replaced person weighs 70 kg.
  3. The new person weighs 94 kg.
-/
def groupSize : ℕ := 8

/-- The average weight increase when the new person joins -/
def averageIncrease : ℝ := 3

/-- The weight of the person being replaced -/
def replacedWeight : ℝ := 70

/-- The weight of the new person joining the group -/
def newWeight : ℝ := 94

/-- Theorem stating that the group size is correct given the conditions -/
theorem group_size_correct :
  groupSize * averageIncrease = newWeight - replacedWeight :=
by sorry

end NUMINAMATH_CALUDE_group_size_correct_l1313_131329


namespace NUMINAMATH_CALUDE_ab_equals_six_l1313_131383

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l1313_131383


namespace NUMINAMATH_CALUDE_julia_grocery_purchase_l1313_131394

/-- Represents the cost of items and the total bill for Julia's grocery purchase. -/
def grocery_bill (snickers_cost : ℚ) : ℚ :=
  let mms_cost := 2 * snickers_cost
  let pepsi_cost := 2 * mms_cost
  let bread_cost := 3 * pepsi_cost
  2 * snickers_cost + 3 * mms_cost + 4 * pepsi_cost + 5 * bread_cost

/-- Theorem stating the total cost of Julia's purchase and the additional amount she needs to pay. -/
theorem julia_grocery_purchase (snickers_cost : ℚ) (h : snickers_cost = 3/2) :
  grocery_bill snickers_cost = 126 ∧ grocery_bill snickers_cost - 100 = 26 := by
  sorry

#eval grocery_bill (3/2)

end NUMINAMATH_CALUDE_julia_grocery_purchase_l1313_131394


namespace NUMINAMATH_CALUDE_base5_arithmetic_sequence_implies_xyz_decimal_l1313_131360

/-- Converts a base-5 number to decimal -/
def toDecimal (a b c : Nat) : Nat :=
  a * 25 + b * 5 + c

/-- Checks if a number is a valid base-5 digit -/
def isBase5Digit (n : Nat) : Prop :=
  n ≥ 0 ∧ n < 5

theorem base5_arithmetic_sequence_implies_xyz_decimal (V W X Y Z : Nat) :
  isBase5Digit V ∧ isBase5Digit W ∧ isBase5Digit X ∧ isBase5Digit Y ∧ isBase5Digit Z →
  toDecimal V Y X = toDecimal V Y Z + 1 →
  toDecimal V V W = toDecimal V Y X + 1 →
  toDecimal X Y Z = 108 := by
  sorry

end NUMINAMATH_CALUDE_base5_arithmetic_sequence_implies_xyz_decimal_l1313_131360


namespace NUMINAMATH_CALUDE_fewer_men_than_women_l1313_131339

theorem fewer_men_than_women (total : ℕ) (men : ℕ) (h1 : total = 180) (h2 : men = 80) (h3 : men < total - men) :
  total - men - men = 20 := by
  sorry

end NUMINAMATH_CALUDE_fewer_men_than_women_l1313_131339


namespace NUMINAMATH_CALUDE_roots_expression_simplification_l1313_131348

theorem roots_expression_simplification (p q : ℝ) (α β γ δ : ℝ) 
  (h1 : α^2 + p*α + 2 = 0) 
  (h2 : β^2 + p*β + 2 = 0) 
  (h3 : γ^2 + q*γ + 2 = 0) 
  (h4 : δ^2 + q*δ + 2 = 0) : 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 2*(p^2 - q^2) := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_simplification_l1313_131348


namespace NUMINAMATH_CALUDE_arithmetic_not_geometric_l1313_131367

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r, ∀ n, a (n + 1) = r * a n

theorem arithmetic_not_geometric (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d ∧ a 1 = 2 →
  ¬(d = 4 ↔ geometric_sequence (λ n => a n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_not_geometric_l1313_131367


namespace NUMINAMATH_CALUDE_infiniteContinuedFraction_eq_infiniteContinuedFraction_value_l1313_131351

/-- The value of the infinite continued fraction 1 / (1 + 1 / (1 + ...)) -/
noncomputable def infiniteContinuedFraction : ℝ :=
  Real.sqrt 5 / 2 + 1 / 2

/-- The infinite continued fraction satisfies the equation x = 1 + 1/x -/
theorem infiniteContinuedFraction_eq : 
  infiniteContinuedFraction = 1 + 1 / infiniteContinuedFraction := by
sorry

/-- The infinite continued fraction 1 / (1 + 1 / (1 + ...)) is equal to (√5 + 1) / 2 -/
theorem infiniteContinuedFraction_value : 
  infiniteContinuedFraction = Real.sqrt 5 / 2 + 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_infiniteContinuedFraction_eq_infiniteContinuedFraction_value_l1313_131351


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1313_131368

/-- The equation of a hyperbola with given conditions -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ c : ℝ, c / a = Real.sqrt 3) →
  (∃ k : ℝ, ∀ x : ℝ, k = -1 ∧ k = a^2 / (a * Real.sqrt 3)) →
  (x^2 / 3 - y^2 / 6 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1313_131368


namespace NUMINAMATH_CALUDE_problem_statement_l1313_131379

theorem problem_statement : ∃ y : ℝ, (8000 * 6000 : ℝ) = 480 * (10 ^ y) → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1313_131379


namespace NUMINAMATH_CALUDE_simplify_expression_l1313_131349

theorem simplify_expression (x : ℝ) : (3*x - 6)*(2*x + 8) - (x + 6)*(3*x + 1) = 3*x^2 - 7*x - 54 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1313_131349


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l1313_131354

-- Define base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  lg 2 ^ 2 + lg 2 * lg 5 + lg 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l1313_131354


namespace NUMINAMATH_CALUDE_sum_of_squares_l1313_131313

theorem sum_of_squares (m n : ℝ) (h1 : m + n = 7) (h2 : m * n = 3) : m^2 + n^2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1313_131313


namespace NUMINAMATH_CALUDE_ellipse_chord_ratio_range_l1313_131338

/-- Define an ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Define a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define a line with slope k passing through a point -/
structure Line where
  k : ℝ
  p : Point
  h_k : k ≠ 0

/-- Theorem statement -/
theorem ellipse_chord_ratio_range (C : Ellipse) (F : Point) (l : Line) :
  F.x = 1 ∧ F.y = 0 ∧ l.p = F →
  ∃ (B₁ B₂ M N D P : Point),
    -- B₁ and B₂ are endpoints of minor axis
    B₁.x = 0 ∧ B₂.x = 0 ∧ B₁.y = -C.b ∧ B₂.y = C.b ∧
    -- Condition on FB₁ · FB₂
    (F.x - B₁.x) * (F.x - B₂.x) + (F.y - B₁.y) * (F.y - B₂.y) = -C.a ∧
    -- M and N are intersections of l and C
    (M.y - F.y = l.k * (M.x - F.x) ∧ M.x^2 / C.a^2 + M.y^2 / C.b^2 = 1) ∧
    (N.y - F.y = l.k * (N.x - F.x) ∧ N.x^2 / C.a^2 + N.y^2 / C.b^2 = 1) ∧
    -- P is midpoint of MN
    P.x = (M.x + N.x) / 2 ∧ P.y = (M.y + N.y) / 2 ∧
    -- D is on x-axis and PD is perpendicular to MN
    D.y = 0 ∧ (P.y - D.y) * (N.x - M.x) = -(P.x - D.x) * (N.y - M.y) →
    -- Conclusion: range of DP/MN
    ∀ r : ℝ, (r = (Real.sqrt ((P.x - D.x)^2 + (P.y - D.y)^2)) /
               (Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2))) →
      0 < r ∧ r < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_ratio_range_l1313_131338


namespace NUMINAMATH_CALUDE_range_of_negative_values_l1313_131371

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_decreasing : ∀ x y, x < y → y < 0 → f x > f y)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l1313_131371


namespace NUMINAMATH_CALUDE_range_of_m_l1313_131319

theorem range_of_m (x m : ℝ) : 
  (∀ x, (2 ≤ x ∧ x ≤ 3) → |x - m| < 2) →
  (∃ a b : ℝ, a < b ∧ (∀ m, a < m ∧ m < b ↔ (∀ x, (2 ≤ x ∧ x ≤ 3) → |x - m| < 2))) ∧
  (∀ a b : ℝ, (∀ m, a < m ∧ m < b ↔ (∀ x, (2 ≤ x ∧ x ≤ 3) → |x - m| < 2)) → a = 1 ∧ b = 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1313_131319


namespace NUMINAMATH_CALUDE_choose_four_from_thirty_l1313_131375

theorem choose_four_from_thirty : Nat.choose 30 4 = 27405 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_thirty_l1313_131375


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1313_131318

theorem complex_fraction_equality : (1 + 2*Complex.I) / (1 - Complex.I)^2 = 1 - (1/2)*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1313_131318


namespace NUMINAMATH_CALUDE_polynomial_roots_l1313_131374

def P (x : ℂ) : ℂ := x^5 - 5*x^4 + 11*x^3 - 13*x^2 + 9*x - 3

theorem polynomial_roots :
  let roots : List ℂ := [1, (3 + Complex.I * Real.sqrt 3) / 2, (1 - Complex.I * Real.sqrt 3) / 2,
                         (3 - Complex.I * Real.sqrt 3) / 2, (1 + Complex.I * Real.sqrt 3) / 2]
  ∀ x : ℂ, (P x = 0) ↔ (x ∈ roots) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1313_131374


namespace NUMINAMATH_CALUDE_hyperbola_b_plus_k_l1313_131311

/-- Given a hyperbola with asymptotes y = 3x + 6 and y = -3x + 2, passing through (2, 12),
    prove that b + k = (16√2 + 36) / 9, where (y-k)²/a² - (x-h)²/b² = 1 is the standard form. -/
theorem hyperbola_b_plus_k (a b h k : ℝ) : a > 0 → b > 0 →
  (∀ x y, y = 3*x + 6 ∨ y = -3*x + 2) →  -- Asymptotes
  ((12 - k)^2 / a^2) - ((2 - h)^2 / b^2) = 1 →  -- Point (2, 12) satisfies the equation
  (∀ x y, (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1) →  -- Standard form
  b + k = (16 * Real.sqrt 2 + 36) / 9 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_b_plus_k_l1313_131311


namespace NUMINAMATH_CALUDE_strawberries_in_buckets_l1313_131370

theorem strawberries_in_buckets
  (total_strawberries : ℕ)
  (num_buckets : ℕ)
  (removed_per_bucket : ℕ)
  (h1 : total_strawberries = 300)
  (h2 : num_buckets = 5)
  (h3 : removed_per_bucket = 20)
  : (total_strawberries / num_buckets) - removed_per_bucket = 40 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_in_buckets_l1313_131370


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1313_131355

theorem rectangle_perimeter (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * a + 2 * b > 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1313_131355


namespace NUMINAMATH_CALUDE_john_started_five_days_ago_l1313_131361

/-- Represents the number of days John has worked -/
def days_worked : ℕ := sorry

/-- Represents the daily wage John earns -/
def daily_wage : ℚ := sorry

/-- The total amount John has earned so far -/
def current_earnings : ℚ := 250

/-- The number of additional days John needs to work -/
def additional_days : ℕ := 10

theorem john_started_five_days_ago :
  days_worked = 5 ∧
  daily_wage * days_worked = current_earnings ∧
  daily_wage * (days_worked + additional_days) = 2 * current_earnings :=
sorry

end NUMINAMATH_CALUDE_john_started_five_days_ago_l1313_131361


namespace NUMINAMATH_CALUDE_equation_equivalence_l1313_131389

theorem equation_equivalence (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ (y = 6 * x / (x - 9)) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1313_131389


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1313_131302

theorem complex_equation_solution (z : ℂ) (p q : ℝ) : 
  (∃ b : ℝ, z = Complex.I * b) →  -- z is purely imaginary
  (∃ c : ℝ, (z + 2)^2 + Complex.I * 8 = Complex.I * c) →  -- (z+2)^2 + 8i is purely imaginary
  2 * (z - 1)^2 + p * (z - 1) + q = 0 →  -- z-1 is a root of 2x^2 + px + q = 0
  z = Complex.I * 2 ∧ p = 4 ∧ q = 10 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1313_131302


namespace NUMINAMATH_CALUDE_expression_evaluation_l1313_131393

theorem expression_evaluation : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1313_131393


namespace NUMINAMATH_CALUDE_samantha_laundry_loads_l1313_131391

/-- The number of loads of laundry Samantha did in the wash -/
def laundry_loads : ℕ :=
  -- We'll define this later in the theorem
  sorry

/-- The cost of using a washer for one load -/
def washer_cost : ℚ := 4

/-- The cost of using a dryer for 10 minutes -/
def dryer_cost_per_10min : ℚ := (1 : ℚ) / 4

/-- The number of dryers Samantha uses -/
def num_dryers : ℕ := 3

/-- The number of minutes Samantha uses each dryer -/
def dryer_minutes : ℕ := 40

/-- The total amount Samantha spends -/
def total_spent : ℚ := 11

theorem samantha_laundry_loads :
  laundry_loads = 2 ∧
  laundry_loads * washer_cost +
    (num_dryers * (dryer_minutes / 10) * dryer_cost_per_10min) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_samantha_laundry_loads_l1313_131391


namespace NUMINAMATH_CALUDE_find_b_value_l1313_131378

theorem find_b_value (a b : ℚ) (eq1 : 3 * a + 3 = 0) (eq2 : 2 * b - a = 4) : b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l1313_131378


namespace NUMINAMATH_CALUDE_car_expenses_sum_l1313_131357

theorem car_expenses_sum : 
  let speakers_cost : ℚ := 118.54
  let tires_cost : ℚ := 106.33
  let tints_cost : ℚ := 85.27
  let maintenance_cost : ℚ := 199.75
  let cover_cost : ℚ := 15.63
  speakers_cost + tires_cost + tints_cost + maintenance_cost + cover_cost = 525.52 := by
  sorry

end NUMINAMATH_CALUDE_car_expenses_sum_l1313_131357


namespace NUMINAMATH_CALUDE_linear_function_composition_l1313_131384

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k b : ℝ), k ≠ 0 ∧ ∀ x, f x = k * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 3) →
  (∀ x, f x = 2 * x + 1) ∨ (∀ x, f x = -2 * x - 3) :=
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l1313_131384


namespace NUMINAMATH_CALUDE_jellybean_count_l1313_131301

def jellybean_problem (initial : ℕ) (first_removal : ℕ) (added_back : ℕ) (second_removal : ℕ) : ℕ :=
  initial - first_removal + added_back - second_removal

theorem jellybean_count : jellybean_problem 37 15 5 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l1313_131301


namespace NUMINAMATH_CALUDE_parabola_and_intersection_l1313_131333

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/2 - y^2/2 = 1

-- Define the line passing through P(3,1) with slope 1
def line (x y : ℝ) : Prop := y = x - 2

-- Theorem statement
theorem parabola_and_intersection :
  -- Conditions
  (∀ x y, parabola_C x y → (x = 0 ∧ y = 0) → True) → -- Vertex at origin
  (∀ x, parabola_C x 0 → True) → -- Axis of symmetry is coordinate axis
  (∃ x₀, x₀ = -2 ∧ (∀ x y, parabola_C x y → |x - x₀| = y^2/(4*x₀))) → -- Directrix passes through left focus of hyperbola
  -- Conclusions
  (∀ x y, parabola_C x y ↔ y^2 = 8*x) ∧ -- Equation of parabola C
  (∃ x₁ y₁ x₂ y₂, 
    parabola_C x₁ y₁ ∧ parabola_C x₂ y₂ ∧ 
    line x₁ y₁ ∧ line x₂ y₂ ∧ 
    ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) = 16) -- Length of MN is 16
  := by sorry

end NUMINAMATH_CALUDE_parabola_and_intersection_l1313_131333


namespace NUMINAMATH_CALUDE_gcd_180_270_l1313_131364

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l1313_131364


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l1313_131328

theorem difference_of_squares_factorization (x : ℝ) : 9 - 4 * x^2 = (3 - 2*x) * (3 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l1313_131328


namespace NUMINAMATH_CALUDE_strip_length_is_one_million_l1313_131309

/-- The number of meters in a kilometer -/
def meters_per_km : ℕ := 1000

/-- The number of cubic meters in a cubic kilometer -/
def cubic_meters_in_cubic_km : ℕ := meters_per_km ^ 3

/-- The length of the strip in kilometers -/
def strip_length_km : ℕ := cubic_meters_in_cubic_km / meters_per_km

theorem strip_length_is_one_million :
  strip_length_km = 1000000 := by
  sorry


end NUMINAMATH_CALUDE_strip_length_is_one_million_l1313_131309


namespace NUMINAMATH_CALUDE_unique_multiple_of_72_l1313_131334

def is_multiple_of_72 (n : ℕ) : Prop := ∃ k : ℕ, n = 72 * k

def is_form_a679b (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b

theorem unique_multiple_of_72 :
  ∀ n : ℕ, is_form_a679b n ∧ is_multiple_of_72 n ↔ n = 36792 :=
by sorry

end NUMINAMATH_CALUDE_unique_multiple_of_72_l1313_131334


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1313_131337

/-- Given a hyperbola with the following properties:
  - Point P is on the right branch of the hyperbola (x²/a² - y²/b² = 1), where a > 0 and b > 0
  - F₁ and F₂ are the left and right foci of the hyperbola
  - (OP + OF₂) · F₂P = 0, where O is the origin
  - |PF₁| = √3|PF₂|
  Its eccentricity is √3 + 1 -/
theorem hyperbola_eccentricity (a b : ℝ) (P F₁ F₂ O : ℝ × ℝ) 
  (h_a : a > 0) (h_b : b > 0)
  (h_P : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (h_foci : F₁.1 < 0 ∧ F₂.1 > 0)
  (h_origin : O = (0, 0))
  (h_perpendicular : (P - O + (F₂ - O)) • (P - F₂) = 0)
  (h_distance_ratio : ‖P - F₁‖ = Real.sqrt 3 * ‖P - F₂‖) :
  let c := ‖F₂ - O‖
  c / a = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1313_131337


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1313_131343

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := n

/-- Irreducibility of a polynomial -/
def irreducible (p : IntPolynomial n) : Prop := sorry

/-- The modulus of a complex number is not greater than 1 -/
def modulusNotGreaterThanOne (z : ℂ) : Prop := Complex.abs z ≤ 1

/-- The roots of a polynomial -/
def roots (p : IntPolynomial n) : Set ℂ := sorry

/-- The statement of the theorem -/
theorem polynomial_factorization 
  (n : ℕ+) 
  (f : IntPolynomial n.val) 
  (h_irred : irreducible f) 
  (h_an : f (Fin.last n.val) ≠ 0)
  (h_roots : ∀ z ∈ roots f, modulusNotGreaterThanOne z) :
  ∃ (m : ℕ+) (g : IntPolynomial m.val), 
    ∃ (h : IntPolynomial (n.val + m.val)), 
      h = sorry ∧ 
      (∀ i, h i = if i.val < n.val then f i else if i.val < n.val + m.val then g (i - n.val) else 0) ∧
      h = λ i => if i.val = n.val + m.val - 1 then 1 else if i.val = n.val + m.val then -1 else 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1313_131343


namespace NUMINAMATH_CALUDE_edwards_lawn_mowing_earnings_l1313_131372

/-- Edward's lawn mowing business earnings and expenses --/
theorem edwards_lawn_mowing_earnings 
  (spring_earnings : ℕ) 
  (summer_earnings : ℕ) 
  (supplies_cost : ℕ) 
  (h1 : spring_earnings = 2)
  (h2 : summer_earnings = 27)
  (h3 : supplies_cost = 5) :
  spring_earnings + summer_earnings - supplies_cost = 24 :=
by sorry

end NUMINAMATH_CALUDE_edwards_lawn_mowing_earnings_l1313_131372


namespace NUMINAMATH_CALUDE_factors_of_N_squared_not_dividing_N_l1313_131352

theorem factors_of_N_squared_not_dividing_N : ∃ (S : Finset ℕ), 
  (∀ d ∈ S, d ∣ (2019^2 - 1)^2 ∧ ¬(d ∣ (2019^2 - 1))) ∧ 
  (∀ d : ℕ, d ∣ (2019^2 - 1)^2 ∧ ¬(d ∣ (2019^2 - 1)) → d ∈ S) ∧ 
  S.card = 157 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_N_squared_not_dividing_N_l1313_131352


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_seven_l1313_131310

theorem power_of_three_plus_five_mod_seven : (3^90 + 5) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_seven_l1313_131310


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l1313_131342

theorem arithmetic_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  ∃ q : ℚ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n + q := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l1313_131342


namespace NUMINAMATH_CALUDE_sum_lent_is_2000_l1313_131316

/-- Prove that the sum lent is 2000, given the conditions of the loan --/
theorem sum_lent_is_2000 
  (interest_rate : ℝ) 
  (loan_duration : ℝ) 
  (interest_difference : ℝ) 
  (h1 : interest_rate = 0.03) 
  (h2 : loan_duration = 3) 
  (h3 : ∀ sum_lent : ℝ, sum_lent * interest_rate * loan_duration = sum_lent - interest_difference) 
  (h4 : interest_difference = 1820) : 
  ∃ sum_lent : ℝ, sum_lent = 2000 := by
  sorry


end NUMINAMATH_CALUDE_sum_lent_is_2000_l1313_131316


namespace NUMINAMATH_CALUDE_line_graph_shows_trends_l1313_131385

-- Define the types of statistical graphs
inductive StatGraph
  | BarGraph
  | LineGraph
  | PieChart
  | Histogram

-- Define the properties of statistical graphs
def comparesQuantities (g : StatGraph) : Prop :=
  g = StatGraph.BarGraph

def showsTrends (g : StatGraph) : Prop :=
  g = StatGraph.LineGraph

def displaysParts (g : StatGraph) : Prop :=
  g = StatGraph.PieChart

def showsDistribution (g : StatGraph) : Prop :=
  g = StatGraph.Histogram

-- Define the set of common statistical graphs
def commonGraphs : Set StatGraph :=
  {StatGraph.BarGraph, StatGraph.LineGraph, StatGraph.PieChart, StatGraph.Histogram}

-- Theorem: The line graph is the type that can display the trend of data
theorem line_graph_shows_trends :
  ∃ (g : StatGraph), g ∈ commonGraphs ∧ showsTrends g ∧
    ∀ (h : StatGraph), h ∈ commonGraphs → showsTrends h → h = g :=
  sorry

end NUMINAMATH_CALUDE_line_graph_shows_trends_l1313_131385


namespace NUMINAMATH_CALUDE_tangent_slope_implies_n_value_l1313_131380

/-- The function f(x) defined as x^n + 3^x --/
noncomputable def f (n : ℝ) (x : ℝ) : ℝ := x^n + 3^x

/-- The derivative of f(x) --/
noncomputable def f_derivative (n : ℝ) (x : ℝ) : ℝ := n * x^(n-1) + 3^x * Real.log 3

theorem tangent_slope_implies_n_value (n : ℝ) :
  f n 1 = 4 →
  f_derivative n 1 = 3 + 3 * Real.log 3 →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_n_value_l1313_131380


namespace NUMINAMATH_CALUDE_line_through_point_l1313_131350

/-- Given a line with equation 5x + by + 2 = d passing through the point (40, 5),
    prove that d = 202 + 5b -/
theorem line_through_point (b : ℝ) : 
  ∃ (d : ℝ), 5 * 40 + b * 5 + 2 = d ∧ d = 202 + 5 * b := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1313_131350


namespace NUMINAMATH_CALUDE_g_of_nine_l1313_131332

/-- Given a function g(x) = ax^7 - bx^3 + cx - 7 where g(-9) = 9, prove that g(9) = -23 -/
theorem g_of_nine (a b c : ℝ) (g : ℝ → ℝ) 
  (h1 : ∀ x, g x = a * x^7 - b * x^3 + c * x - 7)
  (h2 : g (-9) = 9) : 
  g 9 = -23 := by sorry

end NUMINAMATH_CALUDE_g_of_nine_l1313_131332


namespace NUMINAMATH_CALUDE_hexagon_area_difference_l1313_131382

/-- The area between a regular hexagon with side length 8 and a smaller hexagon
    formed by joining the midpoints of its sides is 72√3. -/
theorem hexagon_area_difference : 
  let s : ℝ := 8
  let area_large := (3 * Real.sqrt 3 / 2) * s^2
  let area_small := (3 * Real.sqrt 3 / 2) * (s/2)^2
  area_large - area_small = 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_difference_l1313_131382


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1313_131341

theorem divisibility_equivalence (a b c : ℕ) (h : c ≥ 1) :
  a ∣ b ↔ a^c ∣ b^c := by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1313_131341


namespace NUMINAMATH_CALUDE_proportion_problem_l1313_131336

theorem proportion_problem (x y : ℝ) : 
  (0.60 : ℝ) / x = y / 2 → 
  x = 0.19999999999999998 → 
  y = 6 := by sorry

end NUMINAMATH_CALUDE_proportion_problem_l1313_131336


namespace NUMINAMATH_CALUDE_solve_equation_l1313_131323

theorem solve_equation (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 26 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1313_131323


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l1313_131353

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l1313_131353


namespace NUMINAMATH_CALUDE_katie_marbles_l1313_131346

theorem katie_marbles (pink : ℕ) (orange : ℕ) (purple : ℕ) 
  (h1 : pink = 13)
  (h2 : orange = pink - 9)
  (h3 : purple = 4 * orange) :
  pink + orange + purple = 33 := by
sorry

end NUMINAMATH_CALUDE_katie_marbles_l1313_131346


namespace NUMINAMATH_CALUDE_base_k_conversion_l1313_131327

theorem base_k_conversion (k : ℕ) (h : k > 0) : 
  (1 * k^2 + 3 * k + 2 = 30) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_k_conversion_l1313_131327


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l1313_131308

/-- The polar equation r = 2 / (2sin θ - cos θ) represents a line in Cartesian coordinates. -/
theorem polar_to_cartesian_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (x y : ℝ), (∃ (r θ : ℝ), r > 0 ∧
    r = 2 / (2 * Real.sin θ - Real.cos θ) ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ) →
  a * x + b * y = c :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l1313_131308


namespace NUMINAMATH_CALUDE_congruence_problem_l1313_131303

theorem congruence_problem (x : ℤ) 
  (h1 : (2 + x) % (2^3) = 3^2 % (2^3))
  (h2 : (4 + x) % (3^3) = 2^2 % (3^3))
  (h3 : (6 + x) % (5^3) = 7^2 % (5^3)) :
  x % 120 = 103 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l1313_131303


namespace NUMINAMATH_CALUDE_square_vertices_not_on_arithmetic_circles_l1313_131344

theorem square_vertices_not_on_arithmetic_circles : ¬∃ (a d : ℝ), a > 0 ∧ d > 0 ∧
  ((a ^ 2 + (a + d) ^ 2 = (a + 2*d) ^ 2 + (a + 3*d) ^ 2) ∨
   (a ^ 2 + (a + 2*d) ^ 2 = (a + d) ^ 2 + (a + 3*d) ^ 2) ∨
   ((a + d) ^ 2 + (a + 2*d) ^ 2 = a ^ 2 + (a + 3*d) ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_square_vertices_not_on_arithmetic_circles_l1313_131344
