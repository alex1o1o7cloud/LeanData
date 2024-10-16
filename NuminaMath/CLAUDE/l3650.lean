import Mathlib

namespace NUMINAMATH_CALUDE_cab_driver_income_day2_l3650_365051

def cab_driver_problem (day1 day2 day3 day4 day5 : ℕ) (average : ℚ) : Prop :=
  day1 = 250 ∧
  day3 = 750 ∧
  day4 = 400 ∧
  day5 = 500 ∧
  average = 460 ∧
  (day1 + day2 + day3 + day4 + day5) / 5 = average

theorem cab_driver_income_day2 :
  ∀ (day1 day2 day3 day4 day5 : ℕ) (average : ℚ),
    cab_driver_problem day1 day2 day3 day4 day5 average →
    day2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_day2_l3650_365051


namespace NUMINAMATH_CALUDE_ad_greater_than_bc_l3650_365071

theorem ad_greater_than_bc (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
sorry

end NUMINAMATH_CALUDE_ad_greater_than_bc_l3650_365071


namespace NUMINAMATH_CALUDE_equation_solutions_l3650_365002

def equation (n : ℕ) (x : ℝ) : Prop :=
  (((x + 1)^2)^(1/n : ℝ)) + (((x - 1)^2)^(1/n : ℝ)) = 4 * ((x^2 - 1)^(1/n : ℝ))

theorem equation_solutions :
  (∀ x : ℝ, equation 2 x ↔ x = 2 / Real.sqrt 3 ∨ x = -2 / Real.sqrt 3) ∧
  (∀ x : ℝ, equation 3 x ↔ x = 3 * Real.sqrt 3 / 5 ∨ x = -3 * Real.sqrt 3 / 5) ∧
  (∀ x : ℝ, equation 4 x ↔ x = 7 / (4 * Real.sqrt 3) ∨ x = -7 / (4 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3650_365002


namespace NUMINAMATH_CALUDE_remainder_proof_l3650_365067

theorem remainder_proof (x y : ℕ+) (r : ℕ) 
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 8 * (3 * y) + r)
  (h3 : 13 * y - x = 3) :
  r = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_proof_l3650_365067


namespace NUMINAMATH_CALUDE_system_solution_l3650_365042

theorem system_solution : 
  ∃ (j k : ℚ), (7 * j - 35 * k = -3) ∧ (3 * j - 2 * k = 5) ∧ (j = 547/273) ∧ (k = 44/91) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3650_365042


namespace NUMINAMATH_CALUDE_equation_solution_l3650_365040

theorem equation_solution (m : ℤ) : 
  (∃ x : ℕ+, 2 * m * x - 8 = (m + 2) * x) → 
  m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3650_365040


namespace NUMINAMATH_CALUDE_third_side_length_l3650_365083

theorem third_side_length (a b c : ℝ) : 
  a = 4 → b = 10 → c = 11 →
  a > 0 → b > 0 → c > 0 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  ∃ (x y z : ℝ), x = a ∧ y = b ∧ z = c ∧ 
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y > z ∧ y + z > x ∧ z + x > y :=
by sorry

end NUMINAMATH_CALUDE_third_side_length_l3650_365083


namespace NUMINAMATH_CALUDE_rhombus_area_l3650_365072

/-- The area of a rhombus with sides of length 4 and an acute angle of 45 degrees is 16 square units -/
theorem rhombus_area (side_length : ℝ) (acute_angle : ℝ) : 
  side_length = 4 → 
  acute_angle = 45 * π / 180 →
  side_length * side_length * Real.sin acute_angle = 16 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3650_365072


namespace NUMINAMATH_CALUDE_shark_teeth_relationship_hammerhead_shark_teeth_fraction_l3650_365009

/-- The number of teeth a tiger shark has -/
def tiger_shark_teeth : ℕ := 180

/-- The number of teeth a great white shark has -/
def great_white_shark_teeth : ℕ := 420

/-- The fraction of teeth a hammerhead shark has compared to a tiger shark -/
def hammerhead_fraction : ℚ := 1 / 6

/-- Theorem stating the relationship between shark teeth counts -/
theorem shark_teeth_relationship : 
  great_white_shark_teeth = 2 * (tiger_shark_teeth + hammerhead_fraction * tiger_shark_teeth) :=
by sorry

/-- Theorem proving the fraction of teeth a hammerhead shark has compared to a tiger shark -/
theorem hammerhead_shark_teeth_fraction : 
  hammerhead_fraction = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_shark_teeth_relationship_hammerhead_shark_teeth_fraction_l3650_365009


namespace NUMINAMATH_CALUDE_initial_blue_balls_l3650_365038

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) (initial_blue : ℕ) : 
  total = 18 →
  removed = 3 →
  prob = 1 / 5 →
  (initial_blue - removed : ℚ) / (total - removed) = prob →
  initial_blue = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_blue_balls_l3650_365038


namespace NUMINAMATH_CALUDE_probability_red_then_black_specific_l3650_365013

/-- Represents a deck of cards with red and black cards -/
structure Deck :=
  (total : ℕ)
  (red : ℕ)
  (black : ℕ)
  (h1 : red + black = total)

/-- Calculates the probability of drawing a red card first and a black card second -/
def probability_red_then_black (d : Deck) : ℚ :=
  (d.red : ℚ) / d.total * (d.black : ℚ) / (d.total - 1)

/-- Theorem: The probability of drawing a red card first and a black card second
    from a deck with 20 red cards and 32 black cards (total 52 cards) is 160/663 -/
theorem probability_red_then_black_specific :
  let d : Deck := ⟨52, 20, 32, by simp⟩
  probability_red_then_black d = 160 / 663 := by sorry

end NUMINAMATH_CALUDE_probability_red_then_black_specific_l3650_365013


namespace NUMINAMATH_CALUDE_frog_eyes_count_l3650_365081

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 6

/-- The number of eyes each frog has -/
def eyes_per_frog : ℕ := 2

/-- The total number of frog eyes in the pond -/
def total_frog_eyes : ℕ := num_frogs * eyes_per_frog

theorem frog_eyes_count : total_frog_eyes = 12 := by
  sorry

end NUMINAMATH_CALUDE_frog_eyes_count_l3650_365081


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3650_365073

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Proof that for an arithmetic sequence and distinct positive integers m, n, and p,
    the equation m(a_p - a_n) + n(a_m - a_p) + p(a_n - a_m) = 0 holds -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (m n p : ℕ) (h_arith : ArithmeticSequence a) (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p) :
  m * (a p - a n) + n * (a m - a p) + p * (a n - a m) = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3650_365073


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l3650_365043

theorem least_positive_integer_congruence :
  ∃! x : ℕ+, x.val + 7391 ≡ 167 [ZMOD 12] ∧
  ∀ y : ℕ+, y.val + 7391 ≡ 167 [ZMOD 12] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l3650_365043


namespace NUMINAMATH_CALUDE_percentage_passed_all_subjects_is_two_percent_l3650_365018

/-- Calculates the percentage of students who passed all subjects given failure rates -/
def percentage_passed_all_subjects (fail_hindi : ℝ) (fail_english : ℝ) (fail_math : ℝ)
  (fail_hindi_english : ℝ) (fail_english_math : ℝ) (fail_hindi_math : ℝ) (fail_all : ℝ) : ℝ :=
  100 - (fail_hindi + fail_english + fail_math - fail_hindi_english - fail_english_math - fail_hindi_math + fail_all)

/-- Theorem: The percentage of students who passed all subjects is 2% given the specified failure rates -/
theorem percentage_passed_all_subjects_is_two_percent :
  percentage_passed_all_subjects 46 54 32 18 12 10 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_all_subjects_is_two_percent_l3650_365018


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l3650_365014

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = x^2 -/
def original_parabola : Parabola := { a := 1, b := 0, c := 0 }

/-- Translates a parabola vertically by a given amount -/
def translate_vertical (p : Parabola) (amount : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + amount }

/-- Translates a parabola horizontally by a given amount -/
def translate_horizontal (p : Parabola) (amount : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * amount + p.b, c := p.a * amount^2 - p.b * amount + p.c }

/-- The resulting parabola after translations -/
def resulting_parabola : Parabola :=
  translate_horizontal (translate_vertical original_parabola 3) 5

theorem parabola_translation_theorem :
  resulting_parabola = { a := 1, b := -10, c := 28 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l3650_365014


namespace NUMINAMATH_CALUDE_milk_production_l3650_365089

theorem milk_production (y : ℝ) : 
  (y > 0) → 
  (y * (y + 1) * (y + 10)) / ((y + 2) * (y + 4)) = 
    (y + 10) / ((y + 4) * ((y + 2) / (y * (y + 1)))) := by
  sorry

end NUMINAMATH_CALUDE_milk_production_l3650_365089


namespace NUMINAMATH_CALUDE_base7_to_base10_ABC21_l3650_365048

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (a b c : Nat) : Nat :=
  a * 2401 + b * 343 + c * 49 + 15

/-- Theorem: The base 10 equivalent of ABC21₇ is A · 2401 + B · 343 + C · 49 + 15 --/
theorem base7_to_base10_ABC21 (A B C : Nat) 
  (hA : A ≤ 6) (hB : B ≤ 6) (hC : C ≤ 6) :
  base7ToBase10 A B C = A * 2401 + B * 343 + C * 49 + 15 := by
  sorry

#check base7_to_base10_ABC21

end NUMINAMATH_CALUDE_base7_to_base10_ABC21_l3650_365048


namespace NUMINAMATH_CALUDE_correct_statements_reflect_relationship_l3650_365068

-- Define the statements
inductive Statement
| WaitingForRabbit
| GoodThingsThroughHardship
| PreventMinorIssues
| Insignificant

-- Define the philosophical principles
structure PhilosophicalPrinciple where
  name : String
  description : String

-- Define the relationship between quantitative and qualitative change
def reflectsQuantQualRelationship (s : Statement) (p : PhilosophicalPrinciple) : Prop :=
  match s with
  | Statement.GoodThingsThroughHardship => p.name = "Accumulation"
  | Statement.PreventMinorIssues => p.name = "Moderation"
  | _ => False

-- Theorem statement
theorem correct_statements_reflect_relationship :
  ∃ (p1 p2 : PhilosophicalPrinciple),
    reflectsQuantQualRelationship Statement.GoodThingsThroughHardship p1 ∧
    reflectsQuantQualRelationship Statement.PreventMinorIssues p2 :=
  sorry

end NUMINAMATH_CALUDE_correct_statements_reflect_relationship_l3650_365068


namespace NUMINAMATH_CALUDE_min_value_of_f_l3650_365074

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2000

-- Theorem statement
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 1973 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3650_365074


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3650_365052

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallelPlanes : Plane → Plane → Prop)
variable (perpendicularLinePlane : Line → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (m n : Line) (α β : Plane)
  (h1 : parallelPlanes α β)
  (h2 : perpendicularLinePlane m α)
  (h3 : perpendicularLinePlane n β) :
  parallelLinePlane m β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3650_365052


namespace NUMINAMATH_CALUDE_scientific_notation_exponent_is_integer_l3650_365024

theorem scientific_notation_exponent_is_integer (x : ℝ) (A : ℝ) (N : ℝ) :
  x > 10 →
  x = A * 10^N →
  1 ≤ A →
  A < 10 →
  ∃ n : ℤ, N = n := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_exponent_is_integer_l3650_365024


namespace NUMINAMATH_CALUDE_star_property_l3650_365084

def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  λ (a, b) (c, d) ↦ (a - c, b + d)

theorem star_property :
  ∀ x y : ℤ, star (x, y) (2, 3) = star (5, 4) (1, 1) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_property_l3650_365084


namespace NUMINAMATH_CALUDE_chocolate_difference_l3650_365022

theorem chocolate_difference (t : ℚ) : 
  let sarah := (1 : ℚ) / 3 * t
  let andrew := (3 : ℚ) / 8 * t
  let cecily := t - (sarah + andrew)
  sarah - cecily = (1 : ℚ) / 24 * t := by sorry

end NUMINAMATH_CALUDE_chocolate_difference_l3650_365022


namespace NUMINAMATH_CALUDE_simplified_expression_l3650_365053

theorem simplified_expression (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h : c^3 + d^3 = 3*(c + d)) : c/d + d/c - 3/(c*d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l3650_365053


namespace NUMINAMATH_CALUDE_floor_expression_l3650_365055

theorem floor_expression (n : ℕ) (h : n = 2009) : 
  ⌊((n + 1)^3 / ((n - 1) * n : ℝ) - (n - 1)^3 / (n * (n + 1) : ℝ))⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_l3650_365055


namespace NUMINAMATH_CALUDE_smallest_divisor_square_plus_divisor_square_l3650_365093

theorem smallest_divisor_square_plus_divisor_square (n : ℕ) :
  n ≥ 2 →
  (∃ k d : ℕ,
    k > 1 ∧
    k ∣ n ∧
    (∀ m : ℕ, m > 1 → m ∣ n → m ≥ k) ∧
    d ∣ n ∧
    n = k^2 + d^2) ↔
  n = 8 ∨ n = 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_square_plus_divisor_square_l3650_365093


namespace NUMINAMATH_CALUDE_hiram_allyson_age_problem_l3650_365021

/-- The number added to Hiram's age -/
def x : ℕ := 12

theorem hiram_allyson_age_problem :
  let hiram_age : ℕ := 40
  let allyson_age : ℕ := 28
  hiram_age + x = 2 * allyson_age - 4 :=
by sorry

end NUMINAMATH_CALUDE_hiram_allyson_age_problem_l3650_365021


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3650_365030

theorem complex_equation_solution (x y : ℝ) : 
  (2 * x - y - 2 : ℂ) + (y - 2 : ℂ) * I = 0 → x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3650_365030


namespace NUMINAMATH_CALUDE_distinct_pattern_count_is_17_l3650_365039

/-- Represents a 3x3 grid pattern with exactly 3 shaded squares -/
def Pattern := Fin 9 → Bool

/-- Two patterns are rotationally equivalent if one can be obtained from the other by rotation -/
def RotationallyEquivalent (p1 p2 : Pattern) : Prop := sorry

/-- Count of distinct patterns under rotational equivalence -/
def DistinctPatternCount : ℕ := sorry

theorem distinct_pattern_count_is_17 : DistinctPatternCount = 17 := by sorry

end NUMINAMATH_CALUDE_distinct_pattern_count_is_17_l3650_365039


namespace NUMINAMATH_CALUDE_range_of_a_l3650_365070

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
  (¬ ∃ x : ℝ, x^2 - x + a = 0) ∧
  ((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∨ (∃ x : ℝ, x^2 - x + a = 0)) ∧
  ¬((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ (∃ x : ℝ, x^2 - x + a = 0)) →
  a > 1/4 ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3650_365070


namespace NUMINAMATH_CALUDE_train_travel_time_l3650_365032

/-- Represents the problem of calculating the travel time of two trains --/
theorem train_travel_time 
  (cattle_speed : ℝ) 
  (speed_difference : ℝ) 
  (head_start : ℝ) 
  (total_distance : ℝ) 
  (h1 : cattle_speed = 56) 
  (h2 : speed_difference = 33) 
  (h3 : head_start = 6) 
  (h4 : total_distance = 1284) :
  ∃ t : ℝ, 
    t > 0 ∧ 
    cattle_speed * (t + head_start) + (cattle_speed - speed_difference) * t = total_distance ∧ 
    t = 12 := by
  sorry


end NUMINAMATH_CALUDE_train_travel_time_l3650_365032


namespace NUMINAMATH_CALUDE_range_of_t_t_value_for_diameter_6_l3650_365056

-- Define the equation of the circle
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 + (Real.sqrt 3 * t + 1) * x + t * y + t^2 - 2 = 0

-- Theorem for the range of t
theorem range_of_t :
  ∀ t : ℝ, (∃ x y : ℝ, circle_equation x y t) → t > -(3 * Real.sqrt 3) / 2 :=
sorry

-- Theorem for the value of t when diameter is 6
theorem t_value_for_diameter_6 :
  ∃! t : ℝ, (∃ x y : ℝ, circle_equation x y t) ∧ 
  (∃ x₁ y₁ x₂ y₂ : ℝ, circle_equation x₁ y₁ t ∧ circle_equation x₂ y₂ t ∧ 
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6) ∧
  t = (9 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_t_value_for_diameter_6_l3650_365056


namespace NUMINAMATH_CALUDE_shoe_pairs_in_box_l3650_365010

theorem shoe_pairs_in_box (total_shoes : ℕ) (prob_matching : ℚ) : 
  total_shoes = 18 → prob_matching = 1 / 17 → ∃ n : ℕ, n * 2 = total_shoes ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_shoe_pairs_in_box_l3650_365010


namespace NUMINAMATH_CALUDE_wendy_furniture_time_l3650_365061

/-- Given the number of chairs, tables, and total time spent, 
    calculate the time spent on each piece of furniture. -/
def time_per_piece (chairs : ℕ) (tables : ℕ) (total_time : ℕ) : ℚ :=
  total_time / (chairs + tables)

/-- Theorem: For Wendy's furniture assembly, 
    the time spent on each piece is 6 minutes. -/
theorem wendy_furniture_time :
  let chairs := 4
  let tables := 4
  let total_time := 48
  time_per_piece chairs tables total_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_wendy_furniture_time_l3650_365061


namespace NUMINAMATH_CALUDE_equal_probabilities_l3650_365020

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  green : ℕ

/-- The initial state of the boxes -/
def initial_state : Box × Box :=
  ({red := 100, green := 0}, {red := 0, green := 100})

/-- The number of balls transferred between boxes -/
def transfer_count : ℕ := 8

/-- The final state after transferring balls -/
def final_state : Box × Box :=
  let (red_box, green_box) := initial_state
  let red_box' := {red := red_box.red - transfer_count, green := transfer_count}
  let green_box' := {red := transfer_count, green := green_box.green}
  (red_box', green_box')

/-- The probability of drawing a specific color from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => box.red / (box.red + box.green)
  | "green" => box.green / (box.red + box.green)
  | _ => 0

theorem equal_probabilities :
  let (final_red_box, final_green_box) := final_state
  prob_draw final_red_box "green" = prob_draw final_green_box "red" := by
  sorry


end NUMINAMATH_CALUDE_equal_probabilities_l3650_365020


namespace NUMINAMATH_CALUDE_total_miles_four_weeks_eq_272_l3650_365078

/-- Calculates the total miles Vins rides in a four-week period -/
def total_miles_four_weeks : ℕ :=
  let library_distance : ℕ := 6
  let school_distance : ℕ := 5
  let friend_distance : ℕ := 8
  let extra_return_distance : ℕ := 1
  let friend_shortcut : ℕ := 2
  let library_days_per_week : ℕ := 3
  let school_days_per_week : ℕ := 2
  let friend_visits_per_four_weeks : ℕ := 2
  let weeks : ℕ := 4

  let library_miles_per_week := (library_distance + library_distance + extra_return_distance) * library_days_per_week
  let school_miles_per_week := (school_distance + school_distance + extra_return_distance) * school_days_per_week
  let friend_miles_per_four_weeks := (friend_distance + friend_distance - friend_shortcut) * friend_visits_per_four_weeks

  (library_miles_per_week + school_miles_per_week) * weeks + friend_miles_per_four_weeks

theorem total_miles_four_weeks_eq_272 : total_miles_four_weeks = 272 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_four_weeks_eq_272_l3650_365078


namespace NUMINAMATH_CALUDE_items_sold_increase_after_discount_l3650_365041

/-- Theorem: Increase in items sold after discount
  If a store offers a 10% discount on all items and their gross income increases by 3.5%,
  then the number of items sold increases by 15%.
-/
theorem items_sold_increase_after_discount (P N : ℝ) (N' : ℝ) :
  P > 0 → N > 0 →
  (0.9 * P * N' = 1.035 * P * N) →
  (N' - N) / N * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_items_sold_increase_after_discount_l3650_365041


namespace NUMINAMATH_CALUDE_quadratic_common_root_relation_l3650_365016

/-- Given two quadratic equations with a common root and different other roots, 
    prove the relationship between their coefficients. -/
theorem quadratic_common_root_relation 
  (a b c A B C : ℝ) 
  (h1 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ A * x^2 + B * x + C = 0)
  (h2 : ∃ x y : ℝ, x ≠ y ∧ 
        (a * x^2 + b * x + c = 0 ∨ A * x^2 + B * x + C = 0) ∧
        (a * y^2 + b * y + c = 0 ∨ A * y^2 + B * y + C = 0)) :
  (A * c - C * a)^2 = (A * b - B * a) * (B * c - C * b) :=
sorry

end NUMINAMATH_CALUDE_quadratic_common_root_relation_l3650_365016


namespace NUMINAMATH_CALUDE_right_triangle_angles_l3650_365069

/-- Represents a right triangle with external angles on the hypotenuse in the ratio 9:11 -/
structure RightTriangle where
  -- First acute angle in degrees
  α : ℝ
  -- Second acute angle in degrees
  β : ℝ
  -- The triangle is right-angled
  right_angle : α + β = 90
  -- The external angles on the hypotenuse are in the ratio 9:11
  external_angle_ratio : (180 - α) / (90 + α) = 9 / 11

/-- Theorem stating the acute angles of the specified right triangle -/
theorem right_triangle_angles (t : RightTriangle) : t.α = 58.5 ∧ t.β = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l3650_365069


namespace NUMINAMATH_CALUDE_wheel_distance_covered_l3650_365059

/-- The distance covered by a wheel given its diameter and number of revolutions -/
theorem wheel_distance_covered (diameter : ℝ) (revolutions : ℝ) : 
  diameter = 14 → revolutions = 15.013648771610555 → 
  ∃ distance : ℝ, abs (distance - (π * diameter * revolutions)) < 0.001 ∧ abs (distance - 660.477) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_wheel_distance_covered_l3650_365059


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3650_365006

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 67 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 87 = 67 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l3650_365006


namespace NUMINAMATH_CALUDE_trailing_zeros_factorial_100_l3650_365085

-- Define a function to count trailing zeros in factorial
def trailingZerosInFactorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

-- Theorem statement
theorem trailing_zeros_factorial_100 : trailingZerosInFactorial 100 = 24 := by
  sorry


end NUMINAMATH_CALUDE_trailing_zeros_factorial_100_l3650_365085


namespace NUMINAMATH_CALUDE_jessies_weight_calculation_l3650_365019

/-- Calculates Jessie's current weight after changes due to jogging, diet, and strength training -/
def jessies_current_weight (initial_weight weight_lost_jogging weight_lost_diet weight_gained_training : ℕ) : ℕ :=
  initial_weight - weight_lost_jogging - weight_lost_diet + weight_gained_training

/-- Theorem stating that Jessie's current weight is 29 kilograms -/
theorem jessies_weight_calculation :
  jessies_current_weight 69 35 10 5 = 29 := by
  sorry

end NUMINAMATH_CALUDE_jessies_weight_calculation_l3650_365019


namespace NUMINAMATH_CALUDE_shipping_cost_per_pound_l3650_365001

/-- Shipping cost calculation -/
theorem shipping_cost_per_pound 
  (flat_fee : ℝ) 
  (weight : ℝ) 
  (total_cost : ℝ) 
  (h1 : flat_fee = 5)
  (h2 : weight = 5)
  (h3 : total_cost = 9)
  (h4 : total_cost = flat_fee + weight * (total_cost - flat_fee) / weight) :
  (total_cost - flat_fee) / weight = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_per_pound_l3650_365001


namespace NUMINAMATH_CALUDE_last_remaining_number_l3650_365091

/-- Represents the marking process on a list of numbers -/
def markingProcess (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else
  let m := markingProcess (n / 2)
  if m * 2 > n then 2 * m - 1 else 2 * m + 1

/-- The theorem stating that for 120 numbers, the last remaining number is 64 -/
theorem last_remaining_number :
  markingProcess 120 = 64 := by
  sorry

end NUMINAMATH_CALUDE_last_remaining_number_l3650_365091


namespace NUMINAMATH_CALUDE_english_only_enrollment_l3650_365023

/-- Represents the enrollment data for a class with English and German courses -/
structure ClassEnrollment where
  total : Nat
  both : Nat
  german : Nat

/-- Calculates the number of students enrolled only in English -/
def studentsOnlyEnglish (c : ClassEnrollment) : Nat :=
  c.total - c.german

/-- Theorem stating that 28 students are enrolled only in English -/
theorem english_only_enrollment (c : ClassEnrollment) 
  (h1 : c.total = 50)
  (h2 : c.both = 12)
  (h3 : c.german = 22)
  (h4 : c.total = studentsOnlyEnglish c + c.german) :
  studentsOnlyEnglish c = 28 := by
  sorry

#eval studentsOnlyEnglish { total := 50, both := 12, german := 22 }

end NUMINAMATH_CALUDE_english_only_enrollment_l3650_365023


namespace NUMINAMATH_CALUDE_chord_length_l3650_365026

/-- The length of the chord formed by the intersection of a circle and a line -/
theorem chord_length (x y : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    ((x - 1)^2 + y^2 = 1 ∧ x - 2*y + 1 = 0) → 
    (A.1 - 1)^2 + A.2^2 = 1 ∧ 
    A.1 - 2*A.2 + 1 = 0 ∧ 
    (B.1 - 1)^2 + B.2^2 = 1 ∧ 
    B.1 - 2*B.2 + 1 = 0 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = 2 * 5^(1/2) / 5) :=
sorry

end NUMINAMATH_CALUDE_chord_length_l3650_365026


namespace NUMINAMATH_CALUDE_pascal_triangle_prob_one_l3650_365065

/-- The number of rows in Pascal's Triangle we're considering -/
def n : ℕ := 20

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def total_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1s in the first n rows of Pascal's Triangle -/
def ones_count (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def prob_one (n : ℕ) : ℚ := (ones_count n : ℚ) / (total_elements n : ℚ)

theorem pascal_triangle_prob_one : 
  prob_one n = 39 / 210 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_prob_one_l3650_365065


namespace NUMINAMATH_CALUDE_root_equation_sum_l3650_365046

theorem root_equation_sum (a : ℝ) (h : a^2 + a - 1 = 0) : 
  (1 - a) / a + a / (1 + a) = 1 := by sorry

end NUMINAMATH_CALUDE_root_equation_sum_l3650_365046


namespace NUMINAMATH_CALUDE_james_socks_l3650_365090

theorem james_socks (red_pairs : ℕ) (black : ℕ) (white : ℕ) : 
  black = red_pairs -- number of black socks is equal to the number of pairs of red socks
  → white = 2 * (2 * red_pairs + black) -- number of white socks is twice the number of red and black socks combined
  → 2 * red_pairs + black + white = 90 -- total number of socks is 90
  → red_pairs = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_socks_l3650_365090


namespace NUMINAMATH_CALUDE_plain_lemonade_sales_l3650_365050

/-- The number of glasses of plain lemonade sold -/
def plain_lemonade_glasses : ℕ := 36

/-- The price of plain lemonade in dollars -/
def plain_lemonade_price : ℚ := 3/4

/-- The total revenue from strawberry lemonade in dollars -/
def strawberry_revenue : ℕ := 16

/-- The revenue difference between plain and strawberry lemonade in dollars -/
def revenue_difference : ℕ := 11

theorem plain_lemonade_sales :
  plain_lemonade_glasses * plain_lemonade_price = 
    (strawberry_revenue + revenue_difference : ℚ) := by sorry

end NUMINAMATH_CALUDE_plain_lemonade_sales_l3650_365050


namespace NUMINAMATH_CALUDE_prob_hit_135_prob_hit_exactly_3_l3650_365011

-- Define the probability of hitting the target
def hit_probability : ℚ := 3 / 5

-- Define the number of shots
def num_shots : ℕ := 5

-- Theorem for the first part
theorem prob_hit_135 : 
  (hit_probability * (1 - hit_probability) * hit_probability * (1 - hit_probability) * hit_probability) = 108 / 3125 := by
  sorry

-- Theorem for the second part
theorem prob_hit_exactly_3 :
  (Nat.choose num_shots 3 : ℚ) * hit_probability ^ 3 * (1 - hit_probability) ^ 2 = 216 / 625 := by
  sorry

end NUMINAMATH_CALUDE_prob_hit_135_prob_hit_exactly_3_l3650_365011


namespace NUMINAMATH_CALUDE_inequality_proof_l3650_365088

theorem inequality_proof (a b c d : ℝ) 
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c) (positive_d : 0 < d)
  (sum_condition : a + b + c + d = 3) :
  1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 ≤ 1/(a^2*b^2*c^2*d^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3650_365088


namespace NUMINAMATH_CALUDE_problem_solution_l3650_365058

theorem problem_solution (x : ℝ) : ((12 * x - 20) + (x / 2)) / 7 = 15 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3650_365058


namespace NUMINAMATH_CALUDE_cheryl_pesto_production_l3650_365036

/-- The number of cups of basil needed to make one cup of pesto -/
def basil_per_pesto : ℕ := 4

/-- The number of cups of basil Cheryl can harvest per week -/
def basil_per_week : ℕ := 16

/-- The number of weeks Cheryl can harvest basil -/
def harvest_weeks : ℕ := 8

/-- The total number of cups of pesto Cheryl can make -/
def total_pesto : ℕ := (basil_per_week * harvest_weeks) / basil_per_pesto

theorem cheryl_pesto_production :
  total_pesto = 32 := by sorry

end NUMINAMATH_CALUDE_cheryl_pesto_production_l3650_365036


namespace NUMINAMATH_CALUDE_triangular_pyramid_volume_l3650_365096

/-- Given a triangular pyramid with mutually perpendicular lateral faces of areas 6, 4, and 3, 
    its volume is 4. -/
theorem triangular_pyramid_volume (a b c : ℝ) 
  (h1 : a * b / 2 = 6) 
  (h2 : a * c / 2 = 4) 
  (h3 : b * c / 2 = 3) : 
  a * b * c / 6 = 4 := by
  sorry

#check triangular_pyramid_volume

end NUMINAMATH_CALUDE_triangular_pyramid_volume_l3650_365096


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l3650_365054

theorem sin_thirteen_pi_fourths : Real.sin (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l3650_365054


namespace NUMINAMATH_CALUDE_cricket_team_size_l3650_365035

/-- The number of players on a cricket team satisfying certain conditions -/
theorem cricket_team_size :
  ∀ (total_players throwers right_handed : ℕ),
    throwers = 37 →
    right_handed = 57 →
    3 * (right_handed - throwers) = 2 * (total_players - throwers) →
    total_players = 67 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l3650_365035


namespace NUMINAMATH_CALUDE_stating_count_line_segments_correct_l3650_365077

/-- Represents a regular n-sided convex polygon with n exterior points. -/
structure PolygonWithExteriorPoints (n : ℕ) where
  -- n ≥ 3 to ensure it's a valid polygon
  valid : n ≥ 3

/-- 
Calculates the number of line segments that can be drawn between all pairs 
of interior and exterior points of a regular n-sided convex polygon, 
excluding those connecting adjacent vertices.
-/
def countLineSegments (p : PolygonWithExteriorPoints n) : ℕ :=
  (n * (n - 3)) / 2 + n + n * (n - 3)

/-- 
Theorem stating that the number of line segments is correctly calculated 
by the formula (n(n-3)/2) + n + n(n-3).
-/
theorem count_line_segments_correct (p : PolygonWithExteriorPoints n) :
  countLineSegments p = (n * (n - 3)) / 2 + n + n * (n - 3) := by
  sorry

end NUMINAMATH_CALUDE_stating_count_line_segments_correct_l3650_365077


namespace NUMINAMATH_CALUDE_lomonosov_kvass_affordability_l3650_365075

theorem lomonosov_kvass_affordability 
  (x y : ℝ) 
  (initial_budget : x + y = 1) 
  (first_increase : 0.6 * x + 1.2 * y = 1) :
  1 ≥ 1.44 * y := by
  sorry

end NUMINAMATH_CALUDE_lomonosov_kvass_affordability_l3650_365075


namespace NUMINAMATH_CALUDE_paislee_calvin_ratio_l3650_365086

def calvin_points : ℕ := 500
def paislee_points : ℕ := 125

theorem paislee_calvin_ratio :
  (paislee_points : ℚ) / calvin_points = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_paislee_calvin_ratio_l3650_365086


namespace NUMINAMATH_CALUDE_pythagorean_triplets_l3650_365095

theorem pythagorean_triplets :
  ∀ (a b c : ℤ), a^2 + b^2 = c^2 ↔ 
    ∃ (d p q : ℤ), a = 2*d*p*q ∧ b = d*(q^2 - p^2) ∧ c = d*(p^2 + q^2) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triplets_l3650_365095


namespace NUMINAMATH_CALUDE_find_number_l3650_365076

theorem find_number : ∃ x : ℝ, 
  (0.8 : ℝ)^3 - (0.5 : ℝ)^3 / (0.8 : ℝ)^2 + x + (0.5 : ℝ)^2 = 0.3000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3650_365076


namespace NUMINAMATH_CALUDE_sum_f_negative_l3650_365063

-- Define the function f
variable (f : ℝ → ℝ)

-- State the properties of f
axiom f_symmetry (x : ℝ) : f (4 - x) = -f x
axiom f_monotone_increasing (x y : ℝ) : x > 2 → y > x → f y > f x

-- Define the theorem
theorem sum_f_negative (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ < 4) 
  (h2 : (x₁ - 2) * (x₂ - 2) < 0) : 
  f x₁ + f x₂ < 0 :=
sorry

end NUMINAMATH_CALUDE_sum_f_negative_l3650_365063


namespace NUMINAMATH_CALUDE_dice_sum_not_22_l3650_365079

theorem dice_sum_not_22 (a b c d e : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  1 ≤ e ∧ e ≤ 6 →
  a * b * c * d * e = 432 →
  a + b + c + d + e ≠ 22 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_not_22_l3650_365079


namespace NUMINAMATH_CALUDE_coordinates_of_C_l3650_365092

-- Define the points
def A : ℝ × ℝ := (11, 9)
def B : ℝ × ℝ := (2, -3)
def D : ℝ × ℝ := (-1, 3)

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  -- AB = AC
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  -- D is on BC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∧
  -- AD is perpendicular to BC
  (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0

-- Theorem statement
theorem coordinates_of_C :
  ∃ C : ℝ × ℝ, triangle_ABC C ∧ C = (-4, 9) := by sorry

end NUMINAMATH_CALUDE_coordinates_of_C_l3650_365092


namespace NUMINAMATH_CALUDE_router_time_calculation_l3650_365066

/-- Proves that the time spent turning the router off and on is 10 minutes -/
theorem router_time_calculation (total_time : ℕ) (router_time : ℕ) 
  (hold_time : ℕ) (yelling_time : ℕ) : 
  total_time = 100 ∧ 
  hold_time = 6 * router_time ∧ 
  yelling_time = hold_time / 2 ∧
  total_time = router_time + hold_time + yelling_time →
  router_time = 10 := by
sorry

end NUMINAMATH_CALUDE_router_time_calculation_l3650_365066


namespace NUMINAMATH_CALUDE_cos_is_semi_odd_tan_is_semi_odd_l3650_365000

-- Definition of a semi-odd function
def is_semi_odd (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = -f (2*a - x)

-- Statement for cos(x+1)
theorem cos_is_semi_odd :
  is_semi_odd (λ x => Real.cos (x + 1)) :=
sorry

-- Statement for tan(x)
theorem tan_is_semi_odd :
  is_semi_odd Real.tan :=
sorry

end NUMINAMATH_CALUDE_cos_is_semi_odd_tan_is_semi_odd_l3650_365000


namespace NUMINAMATH_CALUDE_systematic_sampling_sum_l3650_365049

/-- Systematic sampling function -/
def systematicSample (n : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => (start + i * (n / sampleSize)) % n + 1)

theorem systematic_sampling_sum (n : ℕ) (sampleSize : ℕ) (start : ℕ) :
  n = 50 →
  sampleSize = 5 →
  start ≤ n →
  systematicSample n sampleSize start = [4, a, 24, b, 44] →
  a + b = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_sum_l3650_365049


namespace NUMINAMATH_CALUDE_cafeteria_fruit_distribution_l3650_365008

/-- The number of students who wanted fruit in the school cafeteria -/
def students_wanting_fruit : ℕ := 21

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 6

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 15

/-- The number of extra apples left after distribution -/
def extra_apples : ℕ := 16

/-- Theorem stating that the number of students who wanted fruit is 21 -/
theorem cafeteria_fruit_distribution :
  students_wanting_fruit = red_apples + green_apples :=
by
  sorry

#check cafeteria_fruit_distribution

end NUMINAMATH_CALUDE_cafeteria_fruit_distribution_l3650_365008


namespace NUMINAMATH_CALUDE_robs_double_cards_fraction_l3650_365064

theorem robs_double_cards_fraction (total_cards : ℕ) (jess_doubles : ℕ) (jess_ratio : ℕ) :
  total_cards = 24 →
  jess_doubles = 40 →
  jess_ratio = 5 →
  (jess_doubles / jess_ratio : ℚ) / total_cards = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_robs_double_cards_fraction_l3650_365064


namespace NUMINAMATH_CALUDE_solve_system_l3650_365044

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  p = 52 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3650_365044


namespace NUMINAMATH_CALUDE_system_unique_solution_l3650_365060

theorem system_unique_solution (a b c : ℝ) : 
  (a^2 + 3*a + 1 = (b + c) / 2) ∧ 
  (b^2 + 3*b + 1 = (a + c) / 2) ∧ 
  (c^2 + 3*c + 1 = (a + b) / 2) → 
  (a = -1 ∧ b = -1 ∧ c = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_unique_solution_l3650_365060


namespace NUMINAMATH_CALUDE_profit_percent_l3650_365004

theorem profit_percent (P : ℝ) (C : ℝ) (h : P > 0) (h2 : C > 0) :
  (2/3 * P = 0.84 * C) → (P - C) / C * 100 = 26 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_l3650_365004


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3650_365094

theorem linear_equation_solution (k : ℝ) : 
  (-1 : ℝ) - k * 2 = 7 → k = -4 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3650_365094


namespace NUMINAMATH_CALUDE_pencil_count_l3650_365007

/-- The total number of pencils after adding more to an initial amount -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 41 initial pencils and 30 added pencils, the total is 71 -/
theorem pencil_count : total_pencils 41 30 = 71 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l3650_365007


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3650_365034

theorem trigonometric_identity (θ c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h : Real.sin θ ^ 6 / c + Real.cos θ ^ 6 / d = 1 / (c + d)) :
  Real.sin θ ^ 18 / c^5 + Real.cos θ ^ 18 / d^5 = (c^4 + d^4) / (c + d)^9 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3650_365034


namespace NUMINAMATH_CALUDE_rotated_P_coordinates_l3650_365005

/-- Square with side length 25 -/
def square_side_length : ℝ := 25

/-- Point Q coordinates -/
def Q : ℝ × ℝ := (0, 7)

/-- Point R is on x-axis -/
def R_on_x_axis (R : ℝ × ℝ) : Prop := R.2 = 0

/-- Line equation where S lies after rotation -/
def S_line_equation (x : ℝ) : Prop := x = 39

/-- Rotation of square about R -/
def rotated_square (P R S : ℝ × ℝ) : Prop :=
  R_on_x_axis R ∧ S_line_equation S.1 ∧ S.2 > 0

/-- Theorem: New coordinates of P after rotation -/
theorem rotated_P_coordinates (P R S : ℝ × ℝ) :
  square_side_length = 25 →
  Q = (0, 7) →
  rotated_square P R S →
  P = (19, 35) := by sorry

end NUMINAMATH_CALUDE_rotated_P_coordinates_l3650_365005


namespace NUMINAMATH_CALUDE_daniels_age_l3650_365057

theorem daniels_age (ishaan_age : ℕ) (years_until_4x : ℕ) (daniel_age : ℕ) : 
  ishaan_age = 6 →
  years_until_4x = 15 →
  daniel_age + years_until_4x = 4 * (ishaan_age + years_until_4x) →
  daniel_age = 69 := by
sorry

end NUMINAMATH_CALUDE_daniels_age_l3650_365057


namespace NUMINAMATH_CALUDE_percentage_difference_l3650_365028

theorem percentage_difference (w q y z : ℝ) 
  (hw : w = 0.6 * q) 
  (hq : q = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w = 0.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3650_365028


namespace NUMINAMATH_CALUDE_net_population_increase_in_one_day_l3650_365099

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℚ := 4

/-- Represents the death rate in people per two seconds -/
def death_rate : ℚ := 3

/-- Calculates the net population increase per second -/
def net_increase_per_second : ℚ := (birth_rate - death_rate) / 2

/-- Theorem stating the net population increase in one day -/
theorem net_population_increase_in_one_day :
  ⌊net_increase_per_second * seconds_per_day⌋ = 43200 := by
  sorry

#eval ⌊net_increase_per_second * seconds_per_day⌋

end NUMINAMATH_CALUDE_net_population_increase_in_one_day_l3650_365099


namespace NUMINAMATH_CALUDE_desired_average_sale_l3650_365062

def sales_first_five_months : List ℝ := [6435, 6927, 6855, 7230, 6562]
def sale_sixth_month : ℝ := 7991
def number_of_months : ℕ := 6

theorem desired_average_sale (sales : List ℝ) (sixth_sale : ℝ) (num_months : ℕ) :
  sales = sales_first_five_months →
  sixth_sale = sale_sixth_month →
  num_months = number_of_months →
  (sales.sum + sixth_sale) / num_months = 7000 := by
  sorry

end NUMINAMATH_CALUDE_desired_average_sale_l3650_365062


namespace NUMINAMATH_CALUDE_sum_of_imaginary_parts_l3650_365045

/-- Given three complex numbers with specific conditions, prove that s+u = 1 -/
theorem sum_of_imaginary_parts (p q r s t u : ℝ) : 
  q = 5 → 
  p = -r - 2*t → 
  Complex.mk (p + r + t) (q + s + u) = Complex.I * 6 → 
  s + u = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_imaginary_parts_l3650_365045


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3650_365082

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3650_365082


namespace NUMINAMATH_CALUDE_cory_candy_purchase_l3650_365087

/-- The amount of money Cory has initially -/
def cory_money : ℝ := 20

/-- The cost of one pack of candies -/
def candy_pack_cost : ℝ := 49

/-- The number of candy packs Cory wants to buy -/
def num_packs : ℕ := 2

/-- The additional amount of money Cory needs -/
def additional_money_needed : ℝ := num_packs * candy_pack_cost - cory_money

theorem cory_candy_purchase :
  additional_money_needed = 78 := by
  sorry

end NUMINAMATH_CALUDE_cory_candy_purchase_l3650_365087


namespace NUMINAMATH_CALUDE_independence_and_polynomial_value_l3650_365012

/-- The algebraic expression is independent of x -/
def is_independent_of_x (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (2 - 2*b) * x^2 + (a + 3) * x - 6*y + 7 = -6*y + 7

/-- The value of the polynomial given a and b -/
def polynomial_value (a b : ℝ) : ℝ :=
  3*(a^2 - 2*a*b - b^2) - (4*a^2 + a*b + b^2)

theorem independence_and_polynomial_value :
  ∃ a b : ℝ, is_independent_of_x a b ∧ a = -3 ∧ b = 1 ∧ polynomial_value a b = 8 := by
  sorry

end NUMINAMATH_CALUDE_independence_and_polynomial_value_l3650_365012


namespace NUMINAMATH_CALUDE_second_die_sides_l3650_365037

theorem second_die_sides (n : ℕ) (h : n > 0) :
  (1 / 2) * ((n - 1) / (2 * n)) = 21428571428571427 / 100000000000000000 →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_second_die_sides_l3650_365037


namespace NUMINAMATH_CALUDE_blood_flow_scientific_notation_l3650_365097

/-- The amount of blood flowing through the heart of a healthy adult per minute in mL -/
def blood_flow : ℝ := 4900

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem blood_flow_scientific_notation :
  to_scientific_notation blood_flow = ScientificNotation.mk 4.9 3 := by
  sorry

end NUMINAMATH_CALUDE_blood_flow_scientific_notation_l3650_365097


namespace NUMINAMATH_CALUDE_inverse_function_property_l3650_365047

theorem inverse_function_property (f : ℝ → ℝ) (hf : Function.Bijective f) 
  (h : ∀ x : ℝ, f (x + 1) + f (-x - 4) = 2) :
  ∀ x : ℝ, (Function.invFun f) (2011 - x) + (Function.invFun f) (x - 2009) = -3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_property_l3650_365047


namespace NUMINAMATH_CALUDE_bridge_length_proof_l3650_365017

theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  ∃ bridge_length : ℝ, bridge_length = 235 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l3650_365017


namespace NUMINAMATH_CALUDE_smallest_games_for_score_l3650_365031

theorem smallest_games_for_score (win_points loss_points final_score : ℤ)
  (win_points_pos : win_points > 0)
  (loss_points_pos : loss_points > 0)
  (final_score_pos : final_score > 0)
  (h : win_points = 25 ∧ loss_points = 13 ∧ final_score = 2007) :
  ∃ (wins losses : ℕ),
    wins * win_points - losses * loss_points = final_score ∧
    wins + losses = 87 ∧
    ∀ (w l : ℕ), w * win_points - l * loss_points = final_score →
      w + l ≥ 87 := by
sorry

end NUMINAMATH_CALUDE_smallest_games_for_score_l3650_365031


namespace NUMINAMATH_CALUDE_max_surface_area_l3650_365003

/-- A 3D structure made of unit cubes -/
structure CubeStructure where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculate the surface area of a CubeStructure -/
def surface_area (s : CubeStructure) : ℕ :=
  2 * (s.width * s.length + s.width * s.height + s.length * s.height)

/-- The specific cube structure from the problem -/
def problem_structure : CubeStructure :=
  { width := 2, length := 4, height := 2 }

theorem max_surface_area :
  surface_area problem_structure = 48 :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_l3650_365003


namespace NUMINAMATH_CALUDE_hexagon_area_equal_perimeter_l3650_365080

/-- The area of a regular hexagon with the same perimeter as a square of area 16 -/
theorem hexagon_area_equal_perimeter (square_area : ℝ) (square_side : ℝ) (hex_side : ℝ) :
  square_area = 16 →
  square_side^2 = square_area →
  4 * square_side = 6 * hex_side →
  (3 * hex_side^2 * Real.sqrt 3) / 2 = (32 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_area_equal_perimeter_l3650_365080


namespace NUMINAMATH_CALUDE_order_of_trig_functions_l3650_365033

theorem order_of_trig_functions : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_trig_functions_l3650_365033


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_eight_given_blue_three_or_six_l3650_365098

/-- Represents the possible outcomes of a die roll -/
inductive DieOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the outcome of rolling two dice -/
structure TwoDiceOutcome where
  red : DieOutcome
  blue : DieOutcome

/-- The sample space of all possible outcomes when rolling two dice -/
def sampleSpace : Set TwoDiceOutcome := sorry

/-- The event where the blue die shows either 3 or 6 -/
def blueThreeOrSix : Set TwoDiceOutcome := sorry

/-- The event where the sum of the numbers on both dice is greater than 8 -/
def sumGreaterThanEight : Set TwoDiceOutcome := sorry

/-- The probability of an event given a condition -/
def conditionalProbability (event condition : Set TwoDiceOutcome) : ℚ := sorry

theorem probability_sum_greater_than_eight_given_blue_three_or_six :
  conditionalProbability sumGreaterThanEight blueThreeOrSix = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_eight_given_blue_three_or_six_l3650_365098


namespace NUMINAMATH_CALUDE_weight_measurement_l3650_365027

def weights : List ℕ := [1, 3, 9, 27]

/-- The maximum weight that can be measured using the given weights -/
def max_weight : ℕ := 40

/-- The number of distinct weights that can be measured using the given weights -/
def distinct_weights : ℕ := 40

/-- Theorem stating the maximum weight and number of distinct weights that can be measured -/
theorem weight_measurement :
  (List.sum weights = max_weight) ∧
  (∀ w : ℕ, w ≤ max_weight → ∃ subset : List ℕ, subset.Sublist weights ∧ List.sum subset = w) ∧
  (distinct_weights = max_weight) := by
  sorry

end NUMINAMATH_CALUDE_weight_measurement_l3650_365027


namespace NUMINAMATH_CALUDE_max_dominoes_after_removal_l3650_365029

/-- Represents a chessboard with some squares removed -/
structure Chessboard :=
  (size : Nat)
  (removed : Nat)
  (removed_black : Nat)
  (removed_white : Nat)

/-- Calculates the maximum number of guaranteed dominoes -/
def max_guaranteed_dominoes (board : Chessboard) : Nat :=
  sorry

/-- Theorem stating the maximum number of guaranteed dominoes for the given problem -/
theorem max_dominoes_after_removal :
  ∀ (board : Chessboard),
    board.size = 8 ∧
    board.removed = 10 ∧
    board.removed_black > 0 ∧
    board.removed_white > 0 ∧
    board.removed_black + board.removed_white = board.removed →
    max_guaranteed_dominoes board = 23 :=
  sorry

end NUMINAMATH_CALUDE_max_dominoes_after_removal_l3650_365029


namespace NUMINAMATH_CALUDE_inequality_proof_l3650_365015

theorem inequality_proof (a b α β θ : ℝ) (ha : a > 0) (hb : b > 0) (hα : abs α > a) :
  (α * β - Real.sqrt (a^2 * β^2 + b^2 * α^2 - a^2 * b^2)) / (α^2 - a^2) ≤ 
  (β + b * Real.sin θ) / (α + a * Real.cos θ) ∧
  (β + b * Real.sin θ) / (α + a * Real.cos θ) ≤ 
  (α * β + Real.sqrt (a^2 * β^2 + b^2 * α^2 - a^2 * b^2)) / (α^2 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3650_365015


namespace NUMINAMATH_CALUDE_every_real_has_cube_root_real_number_line_bijection_correct_statements_l3650_365025

-- Statement 1: Every real number has a cube root
theorem every_real_has_cube_root : ∀ x : ℝ, ∃ y : ℝ, y^3 = x := by sorry

-- Statement 2: Bijection between real numbers and points on a number line
theorem real_number_line_bijection : ∃ f : ℝ → ℝ, Function.Bijective f := by sorry

-- Main theorem combining both statements
theorem correct_statements :
  (∀ x : ℝ, ∃ y : ℝ, y^3 = x) ∧ (∃ f : ℝ → ℝ, Function.Bijective f) := by sorry

end NUMINAMATH_CALUDE_every_real_has_cube_root_real_number_line_bijection_correct_statements_l3650_365025
