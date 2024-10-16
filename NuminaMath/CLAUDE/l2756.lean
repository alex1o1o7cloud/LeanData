import Mathlib

namespace NUMINAMATH_CALUDE_pond_length_l2756_275643

/-- Given a rectangular pond with width 10 m, depth 5 m, and volume of extracted soil 1000 cubic meters, the length of the pond is 20 m. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) (length : ℝ) : 
  width = 10 → depth = 5 → volume = 1000 → volume = length * width * depth → length = 20 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l2756_275643


namespace NUMINAMATH_CALUDE_not_all_parallel_lines_in_plane_l2756_275662

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contained_in_plane : Line → Plane → Prop)

-- State the theorem
theorem not_all_parallel_lines_in_plane 
  (b : Line) (a : Line) (α : Plane)
  (h1 : parallel_line_plane b α)
  (h2 : contained_in_plane a α) :
  ¬ (∀ (l : Line), parallel_line_plane l α → ∀ (m : Line), contained_in_plane m α → parallel_lines l m) :=
sorry

end NUMINAMATH_CALUDE_not_all_parallel_lines_in_plane_l2756_275662


namespace NUMINAMATH_CALUDE_pill_cost_calculation_l2756_275640

/-- The cost of one pill in dollars -/
def pill_cost : ℝ := 1.50

/-- The number of pills John takes per day -/
def pills_per_day : ℕ := 2

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The percentage of the cost that John pays (insurance covers the rest) -/
def john_payment_percentage : ℝ := 0.60

/-- The amount John pays for pills in a month in dollars -/
def john_monthly_payment : ℝ := 54

theorem pill_cost_calculation :
  pill_cost = john_monthly_payment / (pills_per_day * days_in_month * john_payment_percentage) :=
sorry

end NUMINAMATH_CALUDE_pill_cost_calculation_l2756_275640


namespace NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l2756_275633

-- Define a type for points in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a function to check if four points are non-coplanar
def nonCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a function to count the number of planes determined by four points
def countPlanes (p1 p2 p3 p4 : Point3D) : ℕ := sorry

-- Theorem statement
theorem four_noncoplanar_points_determine_four_planes 
  (p1 p2 p3 p4 : Point3D) 
  (h : nonCoplanar p1 p2 p3 p4) : 
  countPlanes p1 p2 p3 p4 = 4 := by sorry

end NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l2756_275633


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2756_275614

def set_A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def set_B : Set ℝ := {x | x - 1 < 0}

theorem union_of_A_and_B : set_A ∪ set_B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2756_275614


namespace NUMINAMATH_CALUDE_cone_volume_over_pi_l2756_275671

/-- The volume of a cone formed from a 300-degree sector of a circle with radius 18, divided by π, is equal to 225√11 -/
theorem cone_volume_over_pi (r : ℝ) (sector_angle : ℝ) :
  r = 18 →
  sector_angle = 300 →
  let base_radius := sector_angle / 360 * r
  let height := Real.sqrt (r^2 - base_radius^2)
  let volume := (1/3) * π * base_radius^2 * height
  volume / π = 225 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_over_pi_l2756_275671


namespace NUMINAMATH_CALUDE_min_value_and_range_l2756_275692

theorem min_value_and_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b - a*b = 0 → x + 2*y ≤ a + 2*b) ∧ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_range_l2756_275692


namespace NUMINAMATH_CALUDE_tv_discount_percentage_l2756_275676

def original_price : ℚ := 480
def first_installment : ℚ := 150
def num_monthly_installments : ℕ := 3
def monthly_installment : ℚ := 102

def total_payment : ℚ := first_installment + (monthly_installment * num_monthly_installments)
def discount : ℚ := original_price - total_payment
def discount_percentage : ℚ := (discount / original_price) * 100

theorem tv_discount_percentage :
  discount_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_tv_discount_percentage_l2756_275676


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_l2756_275649

theorem negation_of_existence_is_forall :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_l2756_275649


namespace NUMINAMATH_CALUDE_stadium_area_calculation_l2756_275615

/-- Calculates the total surface area of a rectangular stadium in square feet,
    given its dimensions in yards. -/
def stadium_surface_area (length_yd width_yd height_yd : ℕ) : ℕ :=
  let length := length_yd * 3
  let width := width_yd * 3
  let height := height_yd * 3
  2 * (length * width + length * height + width * height)

/-- Theorem stating that the surface area of a stadium with given dimensions is 110,968 sq ft. -/
theorem stadium_area_calculation :
  stadium_surface_area 62 48 30 = 110968 := by
  sorry

#eval stadium_surface_area 62 48 30

end NUMINAMATH_CALUDE_stadium_area_calculation_l2756_275615


namespace NUMINAMATH_CALUDE_tom_shares_problem_l2756_275679

theorem tom_shares_problem (initial_cost : ℕ) (sold_shares : ℕ) (sold_price : ℕ) (total_profit : ℕ) :
  initial_cost = 3 →
  sold_shares = 10 →
  sold_price = 4 →
  total_profit = 40 →
  ∃ (initial_shares : ℕ), 
    initial_shares = sold_shares ∧
    sold_shares * (sold_price - initial_cost) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_tom_shares_problem_l2756_275679


namespace NUMINAMATH_CALUDE_cristina_pace_race_scenario_l2756_275604

/-- Cristina's pace in a race with Nicky --/
theorem cristina_pace (head_start : ℝ) (nicky_pace : ℝ) (catch_up_time : ℝ) : ℝ :=
  let nicky_distance := head_start * nicky_pace + catch_up_time * nicky_pace
  nicky_distance / catch_up_time

/-- The race scenario --/
theorem race_scenario : cristina_pace 12 3 30 = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_cristina_pace_race_scenario_l2756_275604


namespace NUMINAMATH_CALUDE_remainder_theorem_l2756_275616

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 50 * k - 1) :
  (n^2 + 2*n + 3) % 50 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2756_275616


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l2756_275667

/-- Given an angle θ formed by the positive x-axis and a line passing through 
    the origin and the point (-3,1), prove that cos(2θ) = 4/5 -/
theorem cos_double_angle_special_case : 
  ∀ θ : Real, 
  (∃ (x y : Real), x = -3 ∧ y = 1 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ 
                    y = Real.sin θ * Real.sqrt (x^2 + y^2)) → 
  Real.cos (2 * θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l2756_275667


namespace NUMINAMATH_CALUDE_quadratic_completion_l2756_275691

theorem quadratic_completion (b : ℝ) (p : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 1 = (x + p)^2 - 7/4) → 
  b = Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_l2756_275691


namespace NUMINAMATH_CALUDE_actual_sampling_method_is_other_l2756_275644

/-- Represents the sampling method used in the survey --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | Other

/-- Represents the characteristics of the sampling process --/
structure SamplingProcess where
  location : String
  selection : String
  endCondition : String

/-- The actual sampling process used in the survey --/
def actualSamplingProcess : SamplingProcess :=
  { location := "shopping mall entrance",
    selection := "randomly selected individuals",
    endCondition := "until predetermined number of respondents reached" }

/-- Theorem stating that the actual sampling method is not one of the three standard methods --/
theorem actual_sampling_method_is_other (sm : SamplingMethod) 
  (h : sm = SamplingMethod.SimpleRandom ∨ 
       sm = SamplingMethod.Stratified ∨ 
       sm = SamplingMethod.Systematic) : 
  sm ≠ SamplingMethod.Other → False := by
  sorry

end NUMINAMATH_CALUDE_actual_sampling_method_is_other_l2756_275644


namespace NUMINAMATH_CALUDE_field_trip_seats_l2756_275625

theorem field_trip_seats (students : ℕ) (buses : ℕ) (seats_per_bus : ℕ) 
  (h1 : students = 28) 
  (h2 : buses = 4) 
  (h3 : students = buses * seats_per_bus) : 
  seats_per_bus = 7 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_seats_l2756_275625


namespace NUMINAMATH_CALUDE_range_of_f_l2756_275678

def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Icc (-8) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2756_275678


namespace NUMINAMATH_CALUDE_salt_mixture_percentage_l2756_275650

/-- The percentage of salt in the initial solution -/
def P : ℝ := sorry

/-- The amount of initial solution in ounces -/
def initial_amount : ℝ := 40

/-- The amount of 60% solution added in ounces -/
def added_amount : ℝ := 40

/-- The percentage of salt in the added solution -/
def added_percentage : ℝ := 60

/-- The total amount of the resulting mixture in ounces -/
def total_amount : ℝ := initial_amount + added_amount

/-- The percentage of salt in the resulting mixture -/
def result_percentage : ℝ := 40

theorem salt_mixture_percentage :
  P = 20 ∧
  (P / 100 * initial_amount + added_percentage / 100 * added_amount) / total_amount * 100 = result_percentage :=
sorry

end NUMINAMATH_CALUDE_salt_mixture_percentage_l2756_275650


namespace NUMINAMATH_CALUDE_original_number_proof_l2756_275655

theorem original_number_proof : ∃! n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (n / 1000 = 6) ∧
  (1000 * (n % 1000) + 6 = n - 1152) ∧
  n = 6538 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2756_275655


namespace NUMINAMATH_CALUDE_geography_history_difference_l2756_275603

/-- Represents the number of pages in each textbook --/
structure TextbookPages where
  history : ℕ
  geography : ℕ
  math : ℕ
  science : ℕ

/-- Conditions for Suzanna's textbooks --/
def suzanna_textbooks (t : TextbookPages) : Prop :=
  t.history = 160 ∧
  t.geography > t.history ∧
  t.math = (t.history + t.geography) / 2 ∧
  t.science = 2 * t.history ∧
  t.history + t.geography + t.math + t.science = 905

/-- Theorem stating the difference in pages between geography and history textbooks --/
theorem geography_history_difference (t : TextbookPages) 
  (h : suzanna_textbooks t) : t.geography - t.history = 70 := by
  sorry


end NUMINAMATH_CALUDE_geography_history_difference_l2756_275603


namespace NUMINAMATH_CALUDE_smallest_lychee_count_correct_l2756_275673

/-- The smallest number of lychees satisfying the distribution condition -/
def smallest_lychee_count : ℕ := 839

/-- Checks if a number satisfies the lychee distribution condition -/
def satisfies_condition (x : ℕ) : Prop :=
  ∀ n : ℕ, 3 ≤ n → n ≤ 8 → x % n = n - 1

theorem smallest_lychee_count_correct :
  satisfies_condition smallest_lychee_count ∧
  ∀ y : ℕ, y < smallest_lychee_count → ¬(satisfies_condition y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_lychee_count_correct_l2756_275673


namespace NUMINAMATH_CALUDE_correct_average_mark_l2756_275623

theorem correct_average_mark (n : ℕ) (initial_avg : ℚ) (wrong_mark correct_mark : ℚ) :
  n = 30 →
  initial_avg = 100 →
  wrong_mark = 70 →
  correct_mark = 10 →
  (n : ℚ) * initial_avg - wrong_mark + correct_mark = 98 * n :=
by sorry

end NUMINAMATH_CALUDE_correct_average_mark_l2756_275623


namespace NUMINAMATH_CALUDE_parabola_vertex_l2756_275672

/-- The parabola defined by y = (x-2)^2 + 4 has vertex at (2,4) -/
theorem parabola_vertex (x y : ℝ) :
  y = (x - 2)^2 + 4 → (2, 4) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2756_275672


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2756_275628

theorem triangle_perimeter (AB AC : ℝ) (h_right_angle : AB ^ 2 + AC ^ 2 = (AB + AC + Real.sqrt (AB ^ 2 + AC ^ 2)) ^ 2 - 2 * AB * AC) (h_AB : AB = 8) (h_AC : AC = 15) :
  AB + AC + Real.sqrt (AB ^ 2 + AC ^ 2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2756_275628


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2756_275627

theorem proposition_equivalence (m : ℝ) :
  (∃ x : ℝ, -x^2 - 2*m*x + 2*m - 3 ≥ 0) ↔ (m ≤ -3 ∨ m ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2756_275627


namespace NUMINAMATH_CALUDE_even_function_domain_l2756_275636

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the domain of the function
def Domain (a : ℝ) : Set ℝ := {x : ℝ | |x + 2 - a| < a}

-- Theorem statement
theorem even_function_domain (f : ℝ → ℝ) (a : ℝ) 
  (h_even : EvenFunction f) 
  (h_domain : Set.range f = Domain a) 
  (h_positive : a > 0) : 
  a = 2 := by sorry

end NUMINAMATH_CALUDE_even_function_domain_l2756_275636


namespace NUMINAMATH_CALUDE_boat_distance_upstream_l2756_275617

/-- Proves that the distance travelled upstream is 10 km given the conditions of the boat problem -/
theorem boat_distance_upstream 
  (boat_speed : ℝ) 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 25) 
  (h2 : upstream_time = 1) 
  (h3 : downstream_time = 0.25) : 
  (boat_speed - ((boat_speed * upstream_time - boat_speed * downstream_time) / (upstream_time + downstream_time))) * upstream_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_upstream_l2756_275617


namespace NUMINAMATH_CALUDE_total_spent_l2756_275647

def trick_deck_price : ℕ := 8
def victor_decks : ℕ := 6
def friend_decks : ℕ := 2

theorem total_spent : 
  trick_deck_price * victor_decks + trick_deck_price * friend_decks = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_l2756_275647


namespace NUMINAMATH_CALUDE_regression_equation_proof_l2756_275689

/-- Given an exponential model and a regression line equation, 
    prove the resulting regression equation. -/
theorem regression_equation_proof 
  (y : ℝ → ℝ) 
  (k a : ℝ) 
  (h1 : ∀ x, y x = Real.exp (k * x + a)) 
  (h2 : ∀ x, 0.25 * x - 2.58 = Real.log (y x)) : 
  ∀ x, y x = Real.exp (0.25 * x - 2.58) := by
sorry

end NUMINAMATH_CALUDE_regression_equation_proof_l2756_275689


namespace NUMINAMATH_CALUDE_fruit_cost_theorem_l2756_275660

/-- Given the prices of fruits satisfying certain conditions, prove the cost of a specific combination. -/
theorem fruit_cost_theorem (x y z : ℝ) 
  (h1 : 2 * x + y + 4 * z = 6) 
  (h2 : 4 * x + 2 * y + 2 * z = 4) : 
  4 * x + 2 * y + 5 * z = 8 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_theorem_l2756_275660


namespace NUMINAMATH_CALUDE_only_four_solutions_l2756_275680

/-- A digit is a natural number from 0 to 9. -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Convert a repeating decimal 0.aaaaa... to a fraction a/9. -/
def repeatingDecimalToFraction (a : Digit) : ℚ := a.val / 9

/-- The property that a pair of digits (a,b) satisfies √(0.aaaaa...) = 0.bbbbb... -/
def SatisfiesEquation (a b : Digit) : Prop :=
  (repeatingDecimalToFraction b) ^ 2 = repeatingDecimalToFraction a

/-- The theorem stating that only four specific digit pairs satisfy the equation. -/
theorem only_four_solutions :
  ∀ a b : Digit, SatisfiesEquation a b ↔ 
    ((a.val = 0 ∧ b.val = 0) ∨
     (a.val = 1 ∧ b.val = 3) ∨
     (a.val = 4 ∧ b.val = 6) ∨
     (a.val = 9 ∧ b.val = 9)) :=
by sorry

end NUMINAMATH_CALUDE_only_four_solutions_l2756_275680


namespace NUMINAMATH_CALUDE_kerosene_cost_l2756_275682

/-- The cost of a pound of rice in dollars -/
def rice_cost : ℚ := 33 / 100

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

theorem kerosene_cost :
  ∀ (egg_cost : ℚ) (kerosene_half_liter_cost : ℚ),
    egg_cost * dozen = rice_cost →  -- A dozen eggs cost as much as a pound of rice
    kerosene_half_liter_cost = egg_cost * 8 →  -- A half-liter of kerosene costs as much as 8 eggs
    (2 * kerosene_half_liter_cost * cents_per_dollar : ℚ) = 44 :=
by sorry

end NUMINAMATH_CALUDE_kerosene_cost_l2756_275682


namespace NUMINAMATH_CALUDE_square_diagonal_ratio_l2756_275629

theorem square_diagonal_ratio (a b : ℝ) (h : a > 0) (k : b > 0) :
  (4 * a) / (4 * b) = 3 / 2 → (a * Real.sqrt 2) / (b * Real.sqrt 2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_ratio_l2756_275629


namespace NUMINAMATH_CALUDE_triangle_DEF_circles_l2756_275698

/-- Triangle DEF with side lengths -/
structure Triangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ

/-- The inscribed circle of a triangle -/
def inscribedCircleDiameter (t : Triangle) : ℝ := sorry

/-- The circumscribed circle of a triangle -/
def circumscribedCircleRadius (t : Triangle) : ℝ := sorry

/-- Main theorem about triangle DEF -/
theorem triangle_DEF_circles :
  let t : Triangle := { DE := 13, DF := 8, EF := 9 }
  inscribedCircleDiameter t = 2 * Real.sqrt 14 ∧
  circumscribedCircleRadius t = (39 * Real.sqrt 14) / 35 := by sorry

end NUMINAMATH_CALUDE_triangle_DEF_circles_l2756_275698


namespace NUMINAMATH_CALUDE_cubic_sum_and_product_l2756_275606

theorem cubic_sum_and_product (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 ∧ a*b + b*c + c*a = -(a^3 + 12) / a := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_and_product_l2756_275606


namespace NUMINAMATH_CALUDE_integral_fractional_parts_sum_l2756_275695

theorem integral_fractional_parts_sum (x y : ℝ) : 
  (x = ⌊5 - 2 * Real.sqrt 3⌋) → 
  (y = (5 - 2 * Real.sqrt 3) - ⌊5 - 2 * Real.sqrt 3⌋) → 
  (x + y + 4 / y = 9) := by
  sorry

end NUMINAMATH_CALUDE_integral_fractional_parts_sum_l2756_275695


namespace NUMINAMATH_CALUDE_probability_is_correct_l2756_275641

/-- The set of numbers from which we're selecting -/
def number_set : Set Nat := {n | 60 ≤ n ∧ n ≤ 1000}

/-- Predicate for a number being two-digit and divisible by 3 -/
def is_two_digit_div_by_three (n : Nat) : Prop := 60 ≤ n ∧ n ≤ 99 ∧ n % 3 = 0

/-- The count of numbers in the set -/
def total_count : Nat := 941

/-- The count of two-digit numbers divisible by 3 in the set -/
def favorable_count : Nat := 14

/-- The probability of selecting a two-digit number divisible by 3 from the set -/
def probability : Rat := favorable_count / total_count

theorem probability_is_correct : probability = 14 / 941 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_correct_l2756_275641


namespace NUMINAMATH_CALUDE_range_of_f_when_a_is_1_a_values_when_f_min_is_3_l2756_275684

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

-- Theorem 1: Range of f when a = 1
theorem range_of_f_when_a_is_1 :
  ∀ y ∈ Set.Icc 0 9, ∃ x ∈ Set.Icc 0 2, f 1 x = y ∧
  ∀ x ∈ Set.Icc 0 2, 0 ≤ f 1 x ∧ f 1 x ≤ 9 :=
sorry

-- Theorem 2: Values of a when f has minimum value 3
theorem a_values_when_f_min_is_3 :
  (∃ x ∈ Set.Icc 0 2, f a x = 3 ∧ ∀ y ∈ Set.Icc 0 2, f a y ≥ 3) →
  (a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_when_a_is_1_a_values_when_f_min_is_3_l2756_275684


namespace NUMINAMATH_CALUDE_complex_multiplication_l2756_275668

/-- Given that i² = -1, prove that (4-5i)(-5+5i) = 5 + 45i --/
theorem complex_multiplication (i : ℂ) (hi : i^2 = -1) :
  (4 - 5*i) * (-5 + 5*i) = 5 + 45*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2756_275668


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l2756_275656

theorem soccer_penalty_kicks (total_players : ℕ) (goalies : ℕ) (h1 : total_players = 24) (h2 : goalies = 4) (h3 : goalies < total_players) :
  (total_players - 1) * goalies = 92 :=
by sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l2756_275656


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l2756_275663

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l2756_275663


namespace NUMINAMATH_CALUDE_emily_toys_left_l2756_275687

/-- The number of toys Emily started with -/
def initial_toys : ℕ := 7

/-- The number of toys Emily sold -/
def sold_toys : ℕ := 3

/-- The number of toys Emily has left -/
def remaining_toys : ℕ := initial_toys - sold_toys

/-- Theorem stating that Emily has 4 toys left -/
theorem emily_toys_left : remaining_toys = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_toys_left_l2756_275687


namespace NUMINAMATH_CALUDE_remainder_sum_l2756_275632

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 53) 
  (hd : d % 42 = 35) : 
  (c + d) % 21 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2756_275632


namespace NUMINAMATH_CALUDE_inscribe_smaller_circles_l2756_275600

-- Define a triangle type
structure Triangle where
  -- We don't need to specify the exact properties of a triangle here

-- Define a circle type
structure Circle where
  radius : ℝ

-- Define a function that checks if a circle can be inscribed in a triangle
def can_inscribe (t : Triangle) (c : Circle) : Prop :=
  sorry -- The exact definition is not important for this statement

-- Main theorem
theorem inscribe_smaller_circles 
  (t : Triangle) (r : ℝ) (n : ℕ) 
  (h : can_inscribe t (Circle.mk r)) :
  ∃ (circles : Finset Circle), 
    (circles.card = n^2) ∧ 
    (∀ c ∈ circles, c.radius = r / n) ∧
    (∀ c ∈ circles, can_inscribe t c) :=
sorry


end NUMINAMATH_CALUDE_inscribe_smaller_circles_l2756_275600


namespace NUMINAMATH_CALUDE_remainder_theorem_l2756_275688

theorem remainder_theorem (x y u v : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + 3 * u * y + 2) % y = (v + 2) % y := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2756_275688


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l2756_275659

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a given x -/
def QuadraticPolynomial.eval (f : QuadraticPolynomial) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The discriminant of a quadratic polynomial -/
def QuadraticPolynomial.discriminant (f : QuadraticPolynomial) : ℝ :=
  f.b^2 - 4 * f.a * f.c

/-- A function has exactly one solution when equal to a linear function -/
def has_exactly_one_solution (f : QuadraticPolynomial) (m : ℝ) (k : ℝ) : Prop :=
  ∃! x : ℝ, f.eval x = m * x + k

theorem quadratic_no_roots (f : QuadraticPolynomial) 
    (h1 : has_exactly_one_solution f 1 (-1))
    (h2 : has_exactly_one_solution f (-2) 2) :
    f.discriminant < 0 := by
  sorry

#check quadratic_no_roots

end NUMINAMATH_CALUDE_quadratic_no_roots_l2756_275659


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2756_275658

/-- A quadratic function with axis of symmetry at x = 2 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The axis of symmetry is at x = 2 -/
def axis_of_symmetry (b : ℝ) : ℝ := 2

theorem quadratic_inequality (b c : ℝ) : 
  f b c (axis_of_symmetry b) < f b c 1 ∧ f b c 1 < f b c 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2756_275658


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2756_275618

theorem sufficient_not_necessary (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∃ b : ℝ, b ≠ 2 ∧ (b - 1) * (b - 2) = 0) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2756_275618


namespace NUMINAMATH_CALUDE_rent_increase_problem_l2756_275607

theorem rent_increase_problem (num_friends : ℕ) (original_rent : ℝ) (increase_percentage : ℝ) (new_mean : ℝ) : 
  num_friends = 4 →
  original_rent = 1400 →
  increase_percentage = 0.20 →
  new_mean = 870 →
  (num_friends * new_mean - original_rent * increase_percentage) / num_friends = 800 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l2756_275607


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l2756_275670

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 6 →
  g = -2*a - c - e →
  (2*a + b*Complex.I) + (c + 2*d*Complex.I) + (e + f*Complex.I) + (g + 2*h*Complex.I) = 8*Complex.I →
  d + f + h = 3/2 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l2756_275670


namespace NUMINAMATH_CALUDE_officer_selection_with_past_officer_l2756_275638

/- Given conditions -/
def total_candidates : ℕ := 20
def past_officers : ℕ := 8
def positions_available : ℕ := 10

/- Theorem to prove -/
theorem officer_selection_with_past_officer :
  (Nat.choose total_candidates positions_available) - 
  (Nat.choose (total_candidates - past_officers) positions_available) = 184690 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_with_past_officer_l2756_275638


namespace NUMINAMATH_CALUDE_apple_grape_worth_l2756_275653

theorem apple_grape_worth (apple_value grape_value : ℚ) :
  (3/4 * 16) * apple_value = 10 * grape_value →
  (1/3 * 9) * apple_value = (5/2) * grape_value := by
  sorry

end NUMINAMATH_CALUDE_apple_grape_worth_l2756_275653


namespace NUMINAMATH_CALUDE_cosine_tangent_equality_l2756_275690

theorem cosine_tangent_equality : 4 * Real.cos (10 * π / 180) - Real.tan (80 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_tangent_equality_l2756_275690


namespace NUMINAMATH_CALUDE_hyperbola_circle_relation_l2756_275693

-- Define the hyperbola
def is_hyperbola (x y : ℝ) : Prop := y^2 - x^2/3 = 1

-- Define a focus of the hyperbola
def is_focus (x y : ℝ) : Prop := x = 0 ∧ (y = 2 ∨ y = -2)

-- Define the eccentricity of the hyperbola
def eccentricity : ℝ := 2

-- Define the circle
def is_circle (x y : ℝ) : Prop := x^2 + (y-2)^2 = 4

-- Theorem statement
theorem hyperbola_circle_relation :
  ∀ (x y cx cy : ℝ),
  is_hyperbola x y →
  is_focus cx cy →
  is_circle (x - cx) (y - cy) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_relation_l2756_275693


namespace NUMINAMATH_CALUDE_white_balls_count_l2756_275651

theorem white_balls_count 
  (total_balls : ℕ) 
  (total_draws : ℕ) 
  (white_draws : ℕ) 
  (h1 : total_balls = 20) 
  (h2 : total_draws = 404) 
  (h3 : white_draws = 101) : 
  (total_balls : ℚ) * (white_draws : ℚ) / (total_draws : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l2756_275651


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2756_275622

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2*y = 1) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧ ∀ z, z = 2^x + 4^y → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2756_275622


namespace NUMINAMATH_CALUDE_table_rotation_l2756_275610

theorem table_rotation (table_width : ℝ) (table_length : ℝ) : 
  table_width = 8 ∧ table_length = 12 →
  ∃ (S : ℕ), (S : ℝ) ≥ (table_width^2 + table_length^2).sqrt ∧
  ∀ (T : ℕ), (T : ℝ) ≥ (table_width^2 + table_length^2).sqrt → S ≤ T →
  S = 15 :=
by sorry

end NUMINAMATH_CALUDE_table_rotation_l2756_275610


namespace NUMINAMATH_CALUDE_x_plus_one_is_square_l2756_275634

def x : ℕ := (1 + 2) * (1 + 2^2) * (1 + 2^4) * (1 + 2^8) * (1 + 2^16) * (1 + 2^32) * (1 + 2^64) * (1 + 2^128) * (1 + 2^256)

theorem x_plus_one_is_square (x : ℕ := x) : x + 1 = 2^512 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_one_is_square_l2756_275634


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2756_275612

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  2 * ((1 / 2) * a 3) = 3 * a 1 + 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticCondition a →
  (a 20 + a 19) / (a 18 + a 17) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2756_275612


namespace NUMINAMATH_CALUDE_two_true_propositions_l2756_275661

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x = 3 → x^2 = 9

-- Define the converse proposition
def converse_prop (x : ℝ) : Prop := x^2 = 9 → x = 3

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := x ≠ 3 → x^2 ≠ 9

-- Define the contrapositive proposition
def contrapositive_prop (x : ℝ) : Prop := x^2 ≠ 9 → x ≠ 3

-- Define the negation proposition
def negation_prop (x : ℝ) : Prop := ¬(x = 3 → x^2 = 9)

-- Theorem statement
theorem two_true_propositions :
  ∃ (A B : (ℝ → Prop)), 
    (A = original_prop ∨ A = converse_prop ∨ A = inverse_prop ∨ A = contrapositive_prop ∨ A = negation_prop) ∧
    (B = original_prop ∨ B = converse_prop ∨ B = inverse_prop ∨ B = contrapositive_prop ∨ B = negation_prop) ∧
    A ≠ B ∧
    (∀ x, A x) ∧ 
    (∀ x, B x) ∧
    (∀ C, (C = original_prop ∨ C = converse_prop ∨ C = inverse_prop ∨ C = contrapositive_prop ∨ C = negation_prop) →
      C ≠ A → C ≠ B → ∃ x, ¬(C x)) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l2756_275661


namespace NUMINAMATH_CALUDE_max_candies_ben_l2756_275697

/-- The maximum number of candies Ben can eat -/
theorem max_candies_ben (total : ℕ) (h_total : total = 30) : ∃ (b : ℕ), b ≤ 6 ∧ 
  ∀ (k : ℕ+) (b' : ℕ), b' + 2 * b' + k * b' = total → b' ≤ b :=
sorry

end NUMINAMATH_CALUDE_max_candies_ben_l2756_275697


namespace NUMINAMATH_CALUDE_employee_payment_percentage_l2756_275694

theorem employee_payment_percentage (total_payment : ℝ) (b_payment : ℝ) :
  total_payment = 450 ∧ b_payment = 180 →
  (total_payment - b_payment) / b_payment * 100 = 150 := by
sorry

end NUMINAMATH_CALUDE_employee_payment_percentage_l2756_275694


namespace NUMINAMATH_CALUDE_pumpkin_price_theorem_l2756_275681

-- Define the prices of seeds
def tomato_price : ℚ := 1.5
def chili_price : ℚ := 0.9

-- Define the total spent and the number of packets bought
def total_spent : ℚ := 18
def pumpkin_packets : ℕ := 3
def tomato_packets : ℕ := 4
def chili_packets : ℕ := 5

-- Define the theorem
theorem pumpkin_price_theorem :
  ∃ (pumpkin_price : ℚ),
    pumpkin_price * pumpkin_packets +
    tomato_price * tomato_packets +
    chili_price * chili_packets = total_spent ∧
    pumpkin_price = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_price_theorem_l2756_275681


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l2756_275624

theorem closest_integer_to_cube_root_200 : 
  ∀ n : ℤ, |n - (200 : ℝ)^(1/3)| ≥ |6 - (200 : ℝ)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l2756_275624


namespace NUMINAMATH_CALUDE_f_g_f_1_equals_102_l2756_275696

def f (x : ℝ) : ℝ := 5 * x + 2

def g (x : ℝ) : ℝ := 3 * x - 1

theorem f_g_f_1_equals_102 : f (g (f 1)) = 102 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_1_equals_102_l2756_275696


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2756_275646

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2756_275646


namespace NUMINAMATH_CALUDE_power_of_one_fourth_l2756_275652

theorem power_of_one_fourth (n : ℤ) : 1024 * (1 / 4 : ℚ) ^ n = 64 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_one_fourth_l2756_275652


namespace NUMINAMATH_CALUDE_nobel_prize_laureates_l2756_275666

theorem nobel_prize_laureates (total_scientists : ℕ) 
                               (wolf_prize : ℕ) 
                               (wolf_and_nobel : ℕ) 
                               (non_wolf_nobel_diff : ℕ) : 
  total_scientists = 50 → 
  wolf_prize = 31 → 
  wolf_and_nobel = 12 → 
  non_wolf_nobel_diff = 3 → 
  (total_scientists - wolf_prize + wolf_and_nobel : ℕ) = 23 := by
  sorry

end NUMINAMATH_CALUDE_nobel_prize_laureates_l2756_275666


namespace NUMINAMATH_CALUDE_solve_equation1_solve_equation2_l2756_275654

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 - 2*x - 8 = 0
def equation2 (x : ℝ) : Prop := x^2 - 2*x - 5 = 0

-- Theorem for the first equation
theorem solve_equation1 : 
  ∃ x1 x2 : ℝ, x1 = 4 ∧ x2 = -2 ∧ equation1 x1 ∧ equation1 x2 ∧
  ∀ x : ℝ, equation1 x → x = x1 ∨ x = x2 :=
sorry

-- Theorem for the second equation
theorem solve_equation2 : 
  ∃ x1 x2 : ℝ, x1 = 1 + Real.sqrt 6 ∧ x2 = 1 - Real.sqrt 6 ∧ 
  equation2 x1 ∧ equation2 x2 ∧
  ∀ x : ℝ, equation2 x → x = x1 ∨ x = x2 :=
sorry

end NUMINAMATH_CALUDE_solve_equation1_solve_equation2_l2756_275654


namespace NUMINAMATH_CALUDE_close_numbers_properties_l2756_275635

/-- A set of close numbers -/
structure CloseNumbers where
  n : ℕ
  numbers : Fin n → ℝ
  sum : ℝ
  n_gt_one : n > 1
  close : ∀ i, numbers i < sum / (n - 1)

/-- Theorems about close numbers -/
theorem close_numbers_properties (cn : CloseNumbers) :
  (∀ i, cn.numbers i > 0) ∧
  (∀ i j k, cn.numbers i + cn.numbers j > cn.numbers k) ∧
  (∀ i j, cn.numbers i + cn.numbers j > cn.sum / (cn.n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_close_numbers_properties_l2756_275635


namespace NUMINAMATH_CALUDE_loot_box_average_loss_l2756_275648

/-- Represents the loot box problem with given parameters --/
structure LootBoxProblem where
  cost_per_box : ℝ
  standard_item_value : ℝ
  rare_item_value : ℝ
  rare_item_probability : ℝ
  total_spent : ℝ

/-- Calculates the average loss per loot box --/
def average_loss (p : LootBoxProblem) : ℝ :=
  let standard_prob := 1 - p.rare_item_probability
  let expected_value := standard_prob * p.standard_item_value + p.rare_item_probability * p.rare_item_value
  p.cost_per_box - expected_value

/-- Theorem stating the average loss per loot box --/
theorem loot_box_average_loss :
  let p : LootBoxProblem := {
    cost_per_box := 5,
    standard_item_value := 3.5,
    rare_item_value := 15,
    rare_item_probability := 0.1,
    total_spent := 40
  }
  average_loss p = 0.35 := by sorry

end NUMINAMATH_CALUDE_loot_box_average_loss_l2756_275648


namespace NUMINAMATH_CALUDE_cat_to_dog_probability_l2756_275621

-- Define the probabilities for each machine
def prob_A : ℚ := 1/3
def prob_B : ℚ := 2/5
def prob_C : ℚ := 1/4

-- Define the probability of a cat remaining a cat after all machines
def prob_cat_total : ℚ := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

-- The main theorem to prove
theorem cat_to_dog_probability :
  1 - prob_cat_total = 7/10 := by sorry

end NUMINAMATH_CALUDE_cat_to_dog_probability_l2756_275621


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2756_275699

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2756_275699


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2756_275683

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part (1)
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem for part (2)
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2756_275683


namespace NUMINAMATH_CALUDE_complex_number_sum_l2756_275609

theorem complex_number_sum (a b : ℝ) : 
  (Complex.I : ℂ)^5 * (Complex.I - 1) = Complex.mk a b → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_l2756_275609


namespace NUMINAMATH_CALUDE_factor_w4_minus_81_factors_are_monic_real_polynomials_l2756_275637

theorem factor_w4_minus_81 (w : ℝ) : 
  w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) := by sorry

theorem factors_are_monic_real_polynomials :
  ∀ w : ℝ, 
    (∃ a b c : ℝ, (w - 3) = w + a ∧ (w + 3) = w + b ∧ (w^2 + 9) = w^2 + c) := by sorry

end NUMINAMATH_CALUDE_factor_w4_minus_81_factors_are_monic_real_polynomials_l2756_275637


namespace NUMINAMATH_CALUDE_no_four_distinct_numbers_l2756_275602

theorem no_four_distinct_numbers : 
  ¬ ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
    (a^11 - a = b^11 - b) ∧ 
    (a^11 - a = c^11 - c) ∧ 
    (a^11 - a = d^11 - d) := by
  sorry

end NUMINAMATH_CALUDE_no_four_distinct_numbers_l2756_275602


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l2756_275686

/-- The line equation 5y - 6x = 15 intersects the x-axis at the point (-2.5, 0) -/
theorem line_x_axis_intersection :
  ∃! (x : ℝ), 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l2756_275686


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2756_275664

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2756_275664


namespace NUMINAMATH_CALUDE_alices_money_is_64_dollars_l2756_275674

/-- Represents the value of Alice's money after exchanging quarters for nickels -/
def alices_money_value (num_quarters : ℕ) (iron_nickel_percentage : ℚ) 
  (iron_nickel_value : ℚ) (regular_nickel_value : ℚ) : ℚ :=
  let total_cents := num_quarters * 25
  let total_nickels := total_cents / 5
  let iron_nickels := iron_nickel_percentage * total_nickels
  let regular_nickels := total_nickels - iron_nickels
  iron_nickels * iron_nickel_value + regular_nickels * regular_nickel_value

/-- Theorem stating that Alice's money value after exchange is $64 -/
theorem alices_money_is_64_dollars :
  alices_money_value 20 (1/5) 300 (5/100) = 6400/100 := by
  sorry

end NUMINAMATH_CALUDE_alices_money_is_64_dollars_l2756_275674


namespace NUMINAMATH_CALUDE_no_solution_equation_l2756_275630

theorem no_solution_equation :
  ¬∃ (x : ℝ), (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2756_275630


namespace NUMINAMATH_CALUDE_intersection_point_l2756_275605

/-- A parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric curve -/
def givenCurve : ParametricCurve where
  x := fun t => -2 + 5 * t
  y := fun t => 1 - 2 * t

/-- Theorem: The point (1/2, 0) is the intersection of the given curve with the x-axis -/
theorem intersection_point : 
  ∃ t : ℝ, givenCurve.x t = 1/2 ∧ givenCurve.y t = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l2756_275605


namespace NUMINAMATH_CALUDE_rabbit_speed_l2756_275665

/-- Proves that a rabbit catching up to a cat in 1 hour, given the cat's speed and head start, has a speed of 25 mph. -/
theorem rabbit_speed (cat_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  cat_speed = 20 →
  head_start = 0.25 →
  catch_up_time = 1 →
  let rabbit_speed := (cat_speed * (catch_up_time + head_start)) / catch_up_time
  rabbit_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l2756_275665


namespace NUMINAMATH_CALUDE_seventeen_integer_chords_l2756_275626

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceToP : ℝ

/-- Counts the number of integer-length chords containing P in the given circle -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem seventeen_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 13) 
  (h2 : c.distanceToP = 12) : 
  countIntegerChords c = 17 :=
sorry

end NUMINAMATH_CALUDE_seventeen_integer_chords_l2756_275626


namespace NUMINAMATH_CALUDE_equal_time_per_style_l2756_275685

-- Define the swimming styles
inductive SwimmingStyle
| FrontCrawl
| Breaststroke
| Backstroke
| Butterfly

-- Define the problem parameters
def totalDistance : ℝ := 600
def totalTime : ℝ := 15
def numStyles : ℕ := 4

-- Define the speed for each style (yards per minute)
def speed (style : SwimmingStyle) : ℝ :=
  match style with
  | SwimmingStyle.FrontCrawl => 45
  | SwimmingStyle.Breaststroke => 35
  | SwimmingStyle.Backstroke => 40
  | SwimmingStyle.Butterfly => 30

-- Theorem to prove
theorem equal_time_per_style :
  ∀ (style : SwimmingStyle),
  (totalTime / numStyles : ℝ) = 3.75 ∧
  (totalDistance / numStyles : ℝ) / speed style ≤ totalTime / numStyles :=
by sorry

end NUMINAMATH_CALUDE_equal_time_per_style_l2756_275685


namespace NUMINAMATH_CALUDE_roses_handed_out_l2756_275675

theorem roses_handed_out (total : ℕ) (left : ℕ) (handed_out : ℕ) : 
  total = 29 → left = 12 → handed_out = total - left → handed_out = 17 := by
  sorry

end NUMINAMATH_CALUDE_roses_handed_out_l2756_275675


namespace NUMINAMATH_CALUDE_modulo_congruence_l2756_275611

theorem modulo_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 100000 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_l2756_275611


namespace NUMINAMATH_CALUDE_compute_F_2_f_3_l2756_275677

-- Define function f
def f (a : ℝ) : ℝ := a^2 - 3*a + 2

-- Define function F
def F (a b : ℝ) : ℝ := b + a^3

-- Theorem to prove
theorem compute_F_2_f_3 : F 2 (f 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_compute_F_2_f_3_l2756_275677


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l2756_275620

theorem product_divisible_by_sum_implies_inequality (m n : ℕ) 
  (h : (m + n) ∣ (m * n)) : 
  m + n ≤ (Nat.gcd m n)^2 := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l2756_275620


namespace NUMINAMATH_CALUDE_set_equality_l2756_275669

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 3)}
def B : Set ℝ := {x | x ≤ -1}

-- Define the set we want to prove is equal to the complement of A ∪ B
def C : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- State the theorem
theorem set_equality : C = (Set.univ : Set ℝ) \ (A ∪ B) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2756_275669


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2756_275608

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2756_275608


namespace NUMINAMATH_CALUDE_converse_even_sum_l2756_275657

theorem converse_even_sum (a b : ℤ) : 
  (∀ a b : ℤ, (Even a ∧ Even b) → Even (a + b)) →
  (∀ a b : ℤ, Even (a + b) → (Even a ∧ Even b)) :=
sorry

end NUMINAMATH_CALUDE_converse_even_sum_l2756_275657


namespace NUMINAMATH_CALUDE_prob_two_math_books_l2756_275642

def total_books : ℕ := 5
def math_books : ℕ := 3

def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem prob_two_math_books : 
  (choose math_books 2 : ℚ) / (choose total_books 2) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_math_books_l2756_275642


namespace NUMINAMATH_CALUDE_inequality_proof_l2756_275631

theorem inequality_proof (x y : ℝ) :
  (x + y) / 2 * (x^2 + y^2) / 2 * (x^3 + y^3) / 2 ≤ (x^6 + y^6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2756_275631


namespace NUMINAMATH_CALUDE_f_property_l2756_275645

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2

-- State the theorem
theorem f_property (a b : ℝ) : f a b (-2) = -7 → f a b 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l2756_275645


namespace NUMINAMATH_CALUDE_homologous_functions_count_l2756_275601

def f (x : ℝ) : ℝ := x^2

def isValidDomain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x ∈ ({0, 1} : Set ℝ)) ∧
  (∀ y ∈ ({0, 1} : Set ℝ), ∃ x ∈ D, f x = y)

theorem homologous_functions_count :
  ∃! (domains : Finset (Set ℝ)), domains.card = 3 ∧
    ∀ D ∈ domains, isValidDomain D :=
sorry

end NUMINAMATH_CALUDE_homologous_functions_count_l2756_275601


namespace NUMINAMATH_CALUDE_speedboat_speed_l2756_275613

/-- Proves that the speed of a speedboat crossing a lake is 30 miles per hour,
    given specific conditions about the lake width, sailboat speed, and wait time. -/
theorem speedboat_speed
  (lake_width : ℝ)
  (sailboat_speed : ℝ)
  (wait_time : ℝ)
  (h_lake : lake_width = 60)
  (h_sail : sailboat_speed = 12)
  (h_wait : wait_time = 3)
  : (lake_width / (lake_width / sailboat_speed - wait_time)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_speedboat_speed_l2756_275613


namespace NUMINAMATH_CALUDE_exponent_of_five_in_forty_factorial_l2756_275639

theorem exponent_of_five_in_forty_factorial :
  ∃ k : ℕ, (40 : ℕ).factorial = 5^10 * k ∧ ¬(5 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_five_in_forty_factorial_l2756_275639


namespace NUMINAMATH_CALUDE_right_triangle_median_on_hypotenuse_l2756_275619

theorem right_triangle_median_on_hypotenuse (a b : ℝ) (h : a = 5 ∧ b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  (c / 2) = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_on_hypotenuse_l2756_275619
