import Mathlib

namespace NUMINAMATH_CALUDE_max_fraction_sum_l800_80056

theorem max_fraction_sum (n : ℕ) (hn : n ≥ 2) :
  ∃ (a b c d : ℕ),
    a / b + c / d < 1 ∧
    a + c ≤ n ∧
    ∀ (a' b' c' d' : ℕ),
      a' / b' + c' / d' < 1 →
      a' + c' ≤ n →
      a' / b' + c' / d' ≤ a / (a + (a * c + 1)) + c / (c + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l800_80056


namespace NUMINAMATH_CALUDE_division_remainder_problem_l800_80048

theorem division_remainder_problem (a b : ℕ) (h1 : a - b = 1365) (h2 : a = 1634) 
  (h3 : ∃ (q : ℕ), q = 6 ∧ a = q * b + (a % b) ∧ a % b < b) : a % b = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l800_80048


namespace NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l800_80018

-- Define the conditions P and Q
def P (x : ℝ) : Prop := |x - 2| < 3
def Q (x : ℝ) : Prop := x^2 - 8*x + 15 < 0

-- Theorem statement
theorem P_necessary_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) := by
  sorry

end NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l800_80018


namespace NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l800_80042

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

-- Theorem statement
theorem linear_function_not_in_fourth_quadrant (k : ℝ) (h1 : k ≠ 2) :
  (∀ x > 0, f k x ≥ 0) → k > 2 := by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l800_80042


namespace NUMINAMATH_CALUDE_x_percentage_of_y_pay_l800_80011

/-- The percentage of Y's pay that X is paid, given the total pay and Y's pay -/
theorem x_percentage_of_y_pay (total_pay y_pay : ℝ) (h1 : total_pay = 700) (h2 : y_pay = 318.1818181818182) :
  (total_pay - y_pay) / y_pay * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_x_percentage_of_y_pay_l800_80011


namespace NUMINAMATH_CALUDE_weekend_ice_cream_total_l800_80091

/-- The total amount of ice cream consumed by 4 roommates over a weekend -/
def weekend_ice_cream_consumption (friday_total : ℝ) : ℝ :=
  let saturday_total := friday_total - (4 * 0.25)
  let sunday_total := 2 * saturday_total
  friday_total + saturday_total + sunday_total

/-- Theorem stating that the total ice cream consumption over the weekend is 10 pints -/
theorem weekend_ice_cream_total :
  weekend_ice_cream_consumption 3.25 = 10 := by
  sorry

#eval weekend_ice_cream_consumption 3.25

end NUMINAMATH_CALUDE_weekend_ice_cream_total_l800_80091


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l800_80088

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0,
    and |f(x)| ≥ 2 for all real x, prove that the coordinates of
    the focus of the parabolic curve are (0, 1/(4a) + 2). -/
theorem parabola_focus_coordinates
  (a b : ℝ) (ha : a ≠ 0)
  (hf : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 1 / (4 * a) + 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l800_80088


namespace NUMINAMATH_CALUDE_hexadecagon_triangles_l800_80030

/-- The number of vertices in a regular hexadecagon -/
def n : ℕ := 16

/-- Represents that no three vertices are collinear in a regular hexadecagon -/
axiom no_collinear_vertices : True

/-- The number of triangles formed by choosing 3 vertices from n vertices -/
def num_triangles : ℕ := Nat.choose n 3

theorem hexadecagon_triangles : num_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_hexadecagon_triangles_l800_80030


namespace NUMINAMATH_CALUDE_theater_occupancy_l800_80008

theorem theater_occupancy (total_chairs : ℕ) (total_people : ℕ) : 
  (3 * total_people = 5 * (4 * total_chairs / 5)) →  -- Three-fifths of people occupy four-fifths of chairs
  (total_chairs - (4 * total_chairs / 5) = 5) →      -- 5 chairs are empty
  (total_people = 33) :=                             -- Total people is 33
by
  sorry

#check theater_occupancy

end NUMINAMATH_CALUDE_theater_occupancy_l800_80008


namespace NUMINAMATH_CALUDE_photo_collection_inconsistency_l800_80013

/-- Represents the number of photos each person has --/
structure PhotoCollection where
  tom : ℕ
  tim : ℕ
  paul : ℕ
  jane : ℕ

/-- The problem statement --/
theorem photo_collection_inconsistency 
  (photos : PhotoCollection) 
  (total_photos : photos.tom + photos.tim + photos.paul + photos.jane = 200)
  (paul_more_than_tim : photos.paul = photos.tim + 10)
  (tim_less_than_total : photos.tim = 200 - 100) :
  False :=
by
  sorry


end NUMINAMATH_CALUDE_photo_collection_inconsistency_l800_80013


namespace NUMINAMATH_CALUDE_unique_pair_solution_l800_80040

theorem unique_pair_solution : 
  ∃! (p n : ℕ), 
    n > p ∧ 
    p.Prime ∧ 
    (∃ k : ℕ, k > 0 ∧ n^(n - p) = k^n) ∧ 
    p = 2 ∧ 
    n = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_pair_solution_l800_80040


namespace NUMINAMATH_CALUDE_partition_condition_l800_80007

theorem partition_condition (α β : ℕ+) : 
  (∃ (A B : Set ℕ+), 
    (A ∪ B = Set.univ) ∧ 
    (A ∩ B = ∅) ∧ 
    ({α * a | a ∈ A} = {β * b | b ∈ B})) ↔ 
  (α ∣ β ∧ α ≠ β) ∨ (β ∣ α ∧ α ≠ β) :=
sorry

end NUMINAMATH_CALUDE_partition_condition_l800_80007


namespace NUMINAMATH_CALUDE_pencil_count_l800_80015

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := 75

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := 112

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := 448

/-- The number of pencils Bob bought -/
def bob_pencils : ℕ := cindi_pencils + 20

/-- The total number of pencils bought by Donna, Marcia, and Bob -/
def total_pencils : ℕ := donna_pencils + marcia_pencils + bob_pencils

theorem pencil_count : total_pencils = 655 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l800_80015


namespace NUMINAMATH_CALUDE_probability_of_third_draw_l800_80003

/-- Represents the outcome of a single draw -/
inductive Ball : Type
| Hui : Ball
| Zhou : Ball
| Mei : Ball
| Li : Ball

/-- Represents the result of three draws -/
structure ThreeDraw :=
  (first : Ball)
  (second : Ball)
  (third : Ball)

/-- Checks if a ThreeDraw result meets the conditions -/
def isValidDraw (draw : ThreeDraw) : Prop :=
  ((draw.first = Ball.Hui ∨ draw.first = Ball.Zhou) ∧
   (draw.second ≠ Ball.Hui ∧ draw.second ≠ Ball.Zhou)) ∨
  ((draw.first ≠ Ball.Hui ∧ draw.first ≠ Ball.Zhou) ∧
   (draw.second = Ball.Hui ∨ draw.second = Ball.Zhou)) ∧
  (draw.third = Ball.Hui ∨ draw.third = Ball.Zhou)

/-- The total number of trials in the experiment -/
def totalTrials : Nat := 16

/-- The number of successful outcomes in the experiment -/
def successfulTrials : Nat := 2

/-- Theorem stating the probability of drawing both "惠" and "州" exactly on the third draw -/
theorem probability_of_third_draw :
  (successfulTrials : ℚ) / totalTrials = 1 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_of_third_draw_l800_80003


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l800_80028

theorem least_positive_integer_with_remainders : ∃ n : ℕ,
  n > 0 ∧
  n % 4 = 1 ∧
  n % 3 = 2 ∧
  n % 5 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 3 = 2 ∧ m % 5 = 3 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l800_80028


namespace NUMINAMATH_CALUDE_car_average_speed_l800_80025

/-- Given a car's speeds for two consecutive hours, calculate its average speed -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 90) (h2 : speed2 = 60) :
  (speed1 + speed2) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l800_80025


namespace NUMINAMATH_CALUDE_carly_butterfly_practice_l800_80075

/-- The number of hours Carly practices butterfly stroke per day -/
def butterfly_hours : ℝ := 3

/-- The number of days per week Carly practices butterfly stroke -/
def butterfly_days_per_week : ℕ := 4

/-- The number of hours Carly practices backstroke per day -/
def backstroke_hours : ℝ := 2

/-- The number of days per week Carly practices backstroke -/
def backstroke_days_per_week : ℕ := 6

/-- The total number of hours Carly practices swimming in a month -/
def total_hours_per_month : ℝ := 96

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

theorem carly_butterfly_practice :
  butterfly_hours * (butterfly_days_per_week * weeks_per_month) +
  backstroke_hours * (backstroke_days_per_week * weeks_per_month) =
  total_hours_per_month := by sorry

end NUMINAMATH_CALUDE_carly_butterfly_practice_l800_80075


namespace NUMINAMATH_CALUDE_max_value_part1_m_value_part2_l800_80059

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := -2 * x^2 + 4 * m * x - 1

-- Part 1
theorem max_value_part1 :
  ∀ θ : ℝ, 0 < θ ∧ θ < π/2 →
  (f 2 (Real.sin θ)) / (Real.sin θ) ≤ -2 * Real.sqrt 2 + 8 :=
sorry

-- Part 2
theorem m_value_part2 :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f m x ≤ 7) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f m x = 7) →
  m = -2.5 ∨ m = 2.5 :=
sorry

end NUMINAMATH_CALUDE_max_value_part1_m_value_part2_l800_80059


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l800_80099

theorem cubic_equation_solutions :
  let f (x : ℝ) := (10 * x - 1) ^ (1/3) + (20 * x + 1) ^ (1/3) - 3 * (5 * x) ^ (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 1/10 ∨ x = -45/973 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l800_80099


namespace NUMINAMATH_CALUDE_total_big_cats_l800_80004

def feline_sanctuary (lions tigers : ℕ) : ℕ :=
  let cougars := (lions + tigers) / 2
  lions + tigers + cougars

theorem total_big_cats :
  feline_sanctuary 12 14 = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_big_cats_l800_80004


namespace NUMINAMATH_CALUDE_mork_tax_rate_l800_80034

theorem mork_tax_rate (mork_income : ℝ) (mork_rate : ℝ) : 
  mork_rate > 0 →
  mork_income > 0 →
  (mork_rate / 100 * mork_income + 0.15 * (4 * mork_income)) / (5 * mork_income) = 0.21 →
  mork_rate = 45 := by
sorry

end NUMINAMATH_CALUDE_mork_tax_rate_l800_80034


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l800_80072

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.log x + 1)

def interval : Set ℝ := {x | 1 / Real.exp 2 ≤ x ∧ x ≤ 1}

theorem minimum_value_theorem (m : ℝ) (hm : ∀ x ∈ interval, f x ≥ m) :
  Real.log (abs m) = 1 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l800_80072


namespace NUMINAMATH_CALUDE_triangle_side_length_l800_80082

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for a valid triangle here
  True

-- Define the right angle at A
def RightAngleAtA (A B C : ℝ × ℝ) : Prop :=
  -- Add condition for right angle at A
  True

-- Define the length of a side
def Length (P Q : ℝ × ℝ) : ℝ :=
  -- Add definition for length between two points
  0

-- Define tangent of an angle
def Tan (A B C : ℝ × ℝ) : ℝ :=
  -- Add definition for tangent of angle C
  0

-- Define cosine of an angle
def Cos (A B C : ℝ × ℝ) : ℝ :=
  -- Add definition for cosine of angle B
  0

theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_right_angle : RightAngleAtA A B C)
  (h_BC_length : Length B C = 10)
  (h_tan_cos : Tan A B C = 3 * Cos A B C) :
  Length A B = 20 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l800_80082


namespace NUMINAMATH_CALUDE_rook_paths_on_chessboard_l800_80066

def rook_paths (n m k : ℕ) : ℕ :=
  if n + m ≠ k then 0
  else Nat.choose (n + m) n

theorem rook_paths_on_chessboard :
  (rook_paths 7 7 14 = 3432) ∧
  (rook_paths 7 7 12 = 57024) ∧
  (rook_paths 7 7 5 = 2000) := by
  sorry

end NUMINAMATH_CALUDE_rook_paths_on_chessboard_l800_80066


namespace NUMINAMATH_CALUDE_marcus_pebbles_l800_80053

theorem marcus_pebbles (initial_pebbles : ℕ) (current_pebbles : ℕ) 
  (h1 : initial_pebbles = 18)
  (h2 : current_pebbles = 39) :
  current_pebbles - (initial_pebbles - initial_pebbles / 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pebbles_l800_80053


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l800_80010

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_equals_open_interval : A ∩ B = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l800_80010


namespace NUMINAMATH_CALUDE_square_sum_equals_z_squared_l800_80041

theorem square_sum_equals_z_squared (x y z b a : ℝ) 
  (h1 : x * y + x^2 = b)
  (h2 : 1 / x^2 - 1 / y^2 = a)
  (h3 : z = x + y) :
  (x + y)^2 = z^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_z_squared_l800_80041


namespace NUMINAMATH_CALUDE_rectangular_hall_area_l800_80047

/-- Calculates the area of a rectangular hall given its length and breadth ratio. -/
def hall_area (length : ℝ) (breadth_ratio : ℝ) : ℝ :=
  length * (breadth_ratio * length)

/-- Theorem: The area of a rectangular hall with length 60 meters and breadth
    two-thirds of its length is 2400 square meters. -/
theorem rectangular_hall_area :
  hall_area 60 (2/3) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_area_l800_80047


namespace NUMINAMATH_CALUDE_sum_three_numbers_l800_80055

theorem sum_three_numbers (a b c N : ℝ) : 
  a + b + c = 72 →
  a - 7 = N →
  b + 7 = N →
  2 * c = N →
  N = 28.8 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l800_80055


namespace NUMINAMATH_CALUDE_inequality_proof_l800_80084

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  3 + a / b + b / c + c / a ≥ a + b + c + 1 / a + 1 / b + 1 / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l800_80084


namespace NUMINAMATH_CALUDE_equation_roots_l800_80009

theorem equation_roots (m n : ℝ) (hm : m ≠ 0) 
  (h : 2 * m * (-3)^2 - n * (-3) + 2 = 0) : 
  ∃ (x y : ℝ), 2 * m * x^2 + n * x + 2 = 0 ∧ 2 * m * y^2 + n * y + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l800_80009


namespace NUMINAMATH_CALUDE_muffin_count_arthur_muffins_l800_80002

/-- The total number of muffins Arthur wants to have -/
def total_muffins (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the total number of muffins is the sum of initial and additional muffins -/
theorem muffin_count (initial : ℕ) (additional : ℕ) :
  total_muffins initial additional = initial + additional :=
by sorry

/-- Theorem proving the specific case in the problem -/
theorem arthur_muffins :
  total_muffins 35 48 = 83 :=
by sorry

end NUMINAMATH_CALUDE_muffin_count_arthur_muffins_l800_80002


namespace NUMINAMATH_CALUDE_bathroom_tile_side_length_l800_80057

-- Define the dimensions of the bathroom
def bathroom_length : ℝ := 6
def bathroom_width : ℝ := 10

-- Define the number of tiles
def number_of_tiles : ℕ := 240

-- Define the side length of a tile
def tile_side_length : ℝ := 0.5

-- Theorem statement
theorem bathroom_tile_side_length :
  bathroom_length * bathroom_width = (number_of_tiles : ℝ) * tile_side_length^2 :=
by sorry

end NUMINAMATH_CALUDE_bathroom_tile_side_length_l800_80057


namespace NUMINAMATH_CALUDE_tangent_line_equation_l800_80092

noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.log x

theorem tangent_line_equation :
  let A : ℝ × ℝ := (1, f 1)
  let m : ℝ := deriv f 1
  (λ (x y : ℝ) => x + y - 2 = 0) = (λ (x y : ℝ) => y - A.2 = m * (x - A.1)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l800_80092


namespace NUMINAMATH_CALUDE_water_canteen_count_l800_80045

def water_problem (flow_rate : ℚ) (duration : ℚ) (additional_water : ℚ) (small_canteen_capacity : ℚ) : ℕ :=
  let total_water := flow_rate * duration + additional_water
  (total_water / small_canteen_capacity).ceil.toNat

theorem water_canteen_count :
  water_problem 9 8 7 6 = 14 := by sorry

end NUMINAMATH_CALUDE_water_canteen_count_l800_80045


namespace NUMINAMATH_CALUDE_mei_age_l800_80083

/-- Given the ages of Li, Zhang, Jung, and Mei, prove Mei's age is 13 --/
theorem mei_age (li_age zhang_age jung_age mei_age : ℕ) : 
  li_age = 12 →
  zhang_age = 2 * li_age →
  jung_age = zhang_age + 2 →
  mei_age = jung_age / 2 →
  mei_age = 13 := by
  sorry


end NUMINAMATH_CALUDE_mei_age_l800_80083


namespace NUMINAMATH_CALUDE_pencil_count_l800_80087

theorem pencil_count (pens pencils : ℕ) : 
  (5 * pencils = 6 * pens) → 
  (pencils = pens + 7) → 
  pencils = 42 := by sorry

end NUMINAMATH_CALUDE_pencil_count_l800_80087


namespace NUMINAMATH_CALUDE_problem_statement_l800_80026

/-- Given an invertible function g: ℝ → ℝ, prove that if g(0) = 3, g(-1) = 0, and g(3) = 6, then 0 - 3 = -3 -/
theorem problem_statement (g : ℝ → ℝ) (hg : Function.Bijective g) 
  (h1 : g 0 = 3) (h2 : g (-1) = 0) (h3 : g 3 = 6) : 0 - 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l800_80026


namespace NUMINAMATH_CALUDE_normal_vector_of_det_equation_l800_80076

/-- The determinant equation of a line -/
def det_equation (x y : ℝ) : Prop := x * 1 - y * 2 = 0

/-- Definition of a normal vector -/
def is_normal_vector (n : ℝ × ℝ) (line_eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), line_eq x y → n.1 * x + n.2 * y = 0

/-- Theorem: (1, -2) is a normal vector of the line represented by |x 2; y 1| = 0 -/
theorem normal_vector_of_det_equation :
  is_normal_vector (1, -2) det_equation :=
sorry

end NUMINAMATH_CALUDE_normal_vector_of_det_equation_l800_80076


namespace NUMINAMATH_CALUDE_counterexample_fifth_power_l800_80022

theorem counterexample_fifth_power : 144^5 + 121^5 + 95^5 + 30^5 = 159^5 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_fifth_power_l800_80022


namespace NUMINAMATH_CALUDE_string_average_length_l800_80064

/-- Given 6 strings where 2 strings have an average length of 70 cm
    and the other 4 strings have an average length of 85 cm,
    prove that the average length of all 6 strings is 80 cm. -/
theorem string_average_length :
  let total_strings : ℕ := 6
  let group1_strings : ℕ := 2
  let group2_strings : ℕ := 4
  let group1_avg_length : ℝ := 70
  let group2_avg_length : ℝ := 85
  (total_strings = group1_strings + group2_strings) →
  (group1_strings * group1_avg_length + group2_strings * group2_avg_length) / total_strings = 80 :=
by sorry

end NUMINAMATH_CALUDE_string_average_length_l800_80064


namespace NUMINAMATH_CALUDE_dog_cat_sum_l800_80005

/-- Represents a three-digit number composed of digits D, O, and G -/
def DOG (D O G : Nat) : Nat := 100 * D + 10 * O + G

/-- Represents a three-digit number composed of digits C, A, and T -/
def CAT (C A T : Nat) : Nat := 100 * C + 10 * A + T

/-- Theorem stating that if DOG + CAT = 1000 for different digits, then the sum of all digits is 28 -/
theorem dog_cat_sum (D O G C A T : Nat) 
  (h1 : D ≠ O ∧ D ≠ G ∧ D ≠ C ∧ D ≠ A ∧ D ≠ T ∧ 
        O ≠ G ∧ O ≠ C ∧ O ≠ A ∧ O ≠ T ∧ 
        G ≠ C ∧ G ≠ A ∧ G ≠ T ∧ 
        C ≠ A ∧ C ≠ T ∧ 
        A ≠ T)
  (h2 : D < 10 ∧ O < 10 ∧ G < 10 ∧ C < 10 ∧ A < 10 ∧ T < 10)
  (h3 : DOG D O G + CAT C A T = 1000) :
  D + O + G + C + A + T = 28 := by
  sorry

end NUMINAMATH_CALUDE_dog_cat_sum_l800_80005


namespace NUMINAMATH_CALUDE_final_price_approx_l800_80001

-- Define the initial cost price
def initial_cost : ℝ := 114.94

-- Define the profit percentages
def profit_A : ℝ := 0.35
def profit_B : ℝ := 0.45

-- Define the function to calculate selling price given cost price and profit percentage
def selling_price (cost : ℝ) (profit : ℝ) : ℝ := cost * (1 + profit)

-- Define the final selling price calculation
def final_price : ℝ := selling_price (selling_price initial_cost profit_A) profit_B

-- Theorem to prove
theorem final_price_approx :
  ∃ ε > 0, |final_price - 225| < ε :=
sorry

end NUMINAMATH_CALUDE_final_price_approx_l800_80001


namespace NUMINAMATH_CALUDE_closest_point_on_line_l800_80017

/-- The point on the line y = 2x - 1 that is closest to (3, 4) is (13/5, 21/5) -/
theorem closest_point_on_line (x y : ℝ) : 
  y = 2 * x - 1 → 
  (x - 3)^2 + (y - 4)^2 ≥ (13/5 - 3)^2 + (21/5 - 4)^2 :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l800_80017


namespace NUMINAMATH_CALUDE_distance_between_points_l800_80094

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 18)
  let p2 : ℝ × ℝ := (13, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 269 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l800_80094


namespace NUMINAMATH_CALUDE_unique_solution_l800_80062

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1) ∧
  2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1) ∧
  2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1)

/-- The theorem stating that (1, 1, 1) is the unique positive real solution -/
theorem unique_solution :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ system x y z ∧ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l800_80062


namespace NUMINAMATH_CALUDE_gross_profit_percentage_l800_80020

theorem gross_profit_percentage (sales_price gross_profit : ℝ) 
  (h1 : sales_price = 91)
  (h2 : gross_profit = 56) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 160 := by
sorry

end NUMINAMATH_CALUDE_gross_profit_percentage_l800_80020


namespace NUMINAMATH_CALUDE_abc_def_ratio_l800_80044

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 10) :
  a * b * c / (d * e * f) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l800_80044


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l800_80046

/-- The sum of the first n odd numbers -/
def triangular_sum (n : ℕ) : ℕ := n^2

/-- The sum of the digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_coin_sum :
  ∃ (n : ℕ), triangular_sum n = 3081 ∧ sum_of_digits n = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l800_80046


namespace NUMINAMATH_CALUDE_sin_cos_shift_l800_80054

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x - π / 3) + π / 6) := by sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l800_80054


namespace NUMINAMATH_CALUDE_rectangular_field_area_l800_80098

theorem rectangular_field_area (w : ℝ) (d : ℝ) (h1 : w = 15) (h2 : d = 17) :
  ∃ l : ℝ, w * l = 120 ∧ d^2 = w^2 + l^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l800_80098


namespace NUMINAMATH_CALUDE_remainder_sum_l800_80039

theorem remainder_sum (x y : ℤ) : 
  x % 80 = 75 → y % 120 = 115 → (x + y) % 40 = 30 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l800_80039


namespace NUMINAMATH_CALUDE_smallest_inverse_domain_l800_80061

def g (x : ℝ) : ℝ := -3 * (x - 1)^2 + 4

theorem smallest_inverse_domain (c : ℝ) : 
  (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_domain_l800_80061


namespace NUMINAMATH_CALUDE_ratio_nature_l800_80052

theorem ratio_nature (x : ℝ) (m n : ℝ) (hx : x > 0) (hmn : m * n ≠ 0) (hineq : m * x > n * x + n) :
  m / (m + n) = (x + 1) / (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ratio_nature_l800_80052


namespace NUMINAMATH_CALUDE_calculate_expression_l800_80065

theorem calculate_expression : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l800_80065


namespace NUMINAMATH_CALUDE_speed_ratio_is_seven_to_eight_l800_80069

-- Define the speeds of A and B
def v_A : ℝ := sorry
def v_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := 400

-- Define the time intervals
def time1 : ℝ := 3
def time2 : ℝ := 12

-- Theorem statement
theorem speed_ratio_is_seven_to_eight :
  -- Condition 1: After 3 minutes, A and B are equidistant from O
  (v_A * time1 = |initial_B_position - v_B * time1|) →
  -- Condition 2: After 12 minutes, A and B are again equidistant from O
  (v_A * time2 = |initial_B_position - v_B * time2|) →
  -- Conclusion: The ratio of A's speed to B's speed is 7:8
  (v_A / v_B = 7 / 8) := by
sorry

end NUMINAMATH_CALUDE_speed_ratio_is_seven_to_eight_l800_80069


namespace NUMINAMATH_CALUDE_given_equation_is_quadratic_l800_80093

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x

/-- Theorem: The given equation is a quadratic equation -/
theorem given_equation_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_given_equation_is_quadratic_l800_80093


namespace NUMINAMATH_CALUDE_square_of_1037_l800_80050

theorem square_of_1037 : (1037 : ℕ)^2 = 1074369 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_square_of_1037_l800_80050


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l800_80063

theorem largest_prime_factor_of_expression : 
  (Nat.factors (16^4 + 2 * 16^2 + 1 - 13^4)).maximum = some 71 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l800_80063


namespace NUMINAMATH_CALUDE_remaining_erasers_l800_80024

theorem remaining_erasers (total : ℕ) (yeonju_fraction : ℚ) (minji_fraction : ℚ)
  (h_total : total = 28)
  (h_yeonju : yeonju_fraction = 1 / 4)
  (h_minji : minji_fraction = 3 / 7) :
  total - (↑total * yeonju_fraction).floor - (↑total * minji_fraction).floor = 9 := by
  sorry

end NUMINAMATH_CALUDE_remaining_erasers_l800_80024


namespace NUMINAMATH_CALUDE_stating_triangle_division_theorem_l800_80079

/-- 
Represents the number of parts a triangle is divided into when each vertex
is connected to n points on the opposite side, assuming no three lines intersect
at the same point.
-/
def triangle_division (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- 
Theorem stating that when each vertex of a triangle is connected by straight lines
to n points on the opposite side, and no three lines intersect at the same point,
the triangle is divided into 3n^2 + 3n + 1 parts.
-/
theorem triangle_division_theorem (n : ℕ) :
  triangle_division n = 3 * n^2 + 3 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_stating_triangle_division_theorem_l800_80079


namespace NUMINAMATH_CALUDE_bryan_has_more_candies_l800_80023

-- Define the number of candies for Bryan and Ben
def bryan_skittles : ℕ := 50
def ben_mms : ℕ := 20

-- Theorem to prove Bryan has more candies and the difference is 30
theorem bryan_has_more_candies : 
  bryan_skittles > ben_mms ∧ bryan_skittles - ben_mms = 30 := by
  sorry

end NUMINAMATH_CALUDE_bryan_has_more_candies_l800_80023


namespace NUMINAMATH_CALUDE_largest_non_prime_sequence_l800_80095

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if a number is a two-digit positive integer -/
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

/-- The theorem stating the largest number in the sequence -/
theorem largest_non_prime_sequence :
  ∃ (n : ℕ), 
    (∀ k : ℕ, k ∈ Finset.range 7 → is_two_digit (n - k)) ∧ 
    (∀ k : ℕ, k ∈ Finset.range 7 → n - k < 50) ∧
    (∀ k : ℕ, k ∈ Finset.range 7 → ¬(is_prime (n - k))) ∧
    n = 30 := by
  sorry

end NUMINAMATH_CALUDE_largest_non_prime_sequence_l800_80095


namespace NUMINAMATH_CALUDE_gcd_cube_plus_three_cubed_l800_80077

theorem gcd_cube_plus_three_cubed (n : ℕ) (h : n > 3) :
  Nat.gcd (n^3 + 3^3) (n + 4) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_three_cubed_l800_80077


namespace NUMINAMATH_CALUDE_train_travel_time_l800_80049

/-- Given a train that travels 270 miles in 3 hours, prove that it takes 2 hours to travel an additional 180 miles at the same rate. -/
theorem train_travel_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ) :
  initial_distance = 270 →
  initial_time = 3 →
  additional_distance = 180 →
  (additional_distance / (initial_distance / initial_time)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l800_80049


namespace NUMINAMATH_CALUDE_square_side_length_l800_80038

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 64 → side * side = area → side = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l800_80038


namespace NUMINAMATH_CALUDE_inequality_solution_l800_80035

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 6) ≥ 1) ↔ (x ∈ Set.Icc (-4) (-2) ∪ Set.Icc 2 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l800_80035


namespace NUMINAMATH_CALUDE_jihye_wallet_money_l800_80012

/-- The total amount of money in Jihye's wallet -/
def total_money (note_value : ℕ) (note_count : ℕ) (coin_value : ℕ) : ℕ :=
  note_value * note_count + coin_value

/-- Theorem stating the total amount of money in Jihye's wallet -/
theorem jihye_wallet_money : total_money 1000 2 560 = 2560 := by
  sorry

end NUMINAMATH_CALUDE_jihye_wallet_money_l800_80012


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_six_l800_80000

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C,
    this theorem proves that under certain conditions, the perimeter is 6. -/
theorem triangle_perimeter_is_six 
  (a b c : ℝ) 
  (A B C : ℝ)
  (h1 : a * Real.cos C + Real.sqrt 3 * a * Real.sin C - b - c = 0)
  (h2 : a = 2)
  (h3 : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_six_l800_80000


namespace NUMINAMATH_CALUDE_data_transmission_time_l800_80031

/-- Proves that the time to send 80 blocks of 400 chunks each at 160 chunks per second is 3 minutes -/
theorem data_transmission_time :
  let num_blocks : ℕ := 80
  let chunks_per_block : ℕ := 400
  let transmission_rate : ℕ := 160
  let total_chunks : ℕ := num_blocks * chunks_per_block
  let transmission_time_seconds : ℕ := total_chunks / transmission_rate
  let transmission_time_minutes : ℚ := transmission_time_seconds / 60
  transmission_time_minutes = 3 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_l800_80031


namespace NUMINAMATH_CALUDE_florist_roses_l800_80033

/-- 
Given a florist who:
- Sells 15 roses
- Picks 21 more roses
- Ends up with 56 roses
Prove that she must have started with 50 roses
-/
theorem florist_roses (initial : ℕ) : 
  initial - 15 + 21 = 56 → initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l800_80033


namespace NUMINAMATH_CALUDE_complete_collection_size_jerry_collection_l800_80027

theorem complete_collection_size (initial_figures : ℕ) (figure_cost : ℕ) (additional_cost : ℕ) : ℕ :=
  let additional_figures := additional_cost / figure_cost
  initial_figures + additional_figures

theorem jerry_collection :
  complete_collection_size 7 8 72 = 16 := by
  sorry

end NUMINAMATH_CALUDE_complete_collection_size_jerry_collection_l800_80027


namespace NUMINAMATH_CALUDE_fraction_decimal_conversions_l800_80073

-- Define a function to round a rational number to n decimal places
def round_to_decimal_places (q : ℚ) (n : ℕ) : ℚ :=
  (↑(round (q * 10^n)) / 10^n)

theorem fraction_decimal_conversions :
  -- 1. 60/4 = 15 in both fraction and decimal form
  (60 : ℚ) / 4 = 15 ∧ 
  -- 2. 19/6 ≈ 3.167 when rounded to three decimal places
  round_to_decimal_places ((19 : ℚ) / 6) 3 = (3167 : ℚ) / 1000 ∧
  -- 3. 0.25 = 1/4
  (1 : ℚ) / 4 = (25 : ℚ) / 100 ∧
  -- 4. 0.08 = 2/25
  (2 : ℚ) / 25 = (8 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_decimal_conversions_l800_80073


namespace NUMINAMATH_CALUDE_lowest_number_of_students_twenty_four_divisible_lowest_number_is_twenty_four_l800_80081

theorem lowest_number_of_students (n : ℕ) : n > 0 ∧ 8 ∣ n ∧ 12 ∣ n → n ≥ 24 := by
  sorry

theorem twenty_four_divisible : 8 ∣ 24 ∧ 12 ∣ 24 := by
  sorry

theorem lowest_number_is_twenty_four : ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 12 ∣ n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_lowest_number_of_students_twenty_four_divisible_lowest_number_is_twenty_four_l800_80081


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_zero_conditions_l800_80090

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Part 1: Tangent line when a = 1
theorem tangent_line_at_zero (x : ℝ) :
  ∃ m b : ℝ, (deriv (f 1)) 0 = m ∧ f 1 0 = b ∧ m = 2 ∧ b = 0 := by sorry

-- Part 2: Range of a for exactly one zero in each interval
theorem zero_conditions (a : ℝ) :
  (∃! x : ℝ, x ∈ Set.Ioo (-1) 0 ∧ f a x = 0) ∧
  (∃! x : ℝ, x ∈ Set.Ioi 0 ∧ f a x = 0) ↔
  a ∈ Set.Iio (-1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_zero_conditions_l800_80090


namespace NUMINAMATH_CALUDE_min_value_of_product_l800_80037

/-- Given positive real numbers x₁, x₂, x₃, x₄ such that their sum is π,
    the product of (2sin²xᵢ + 1/sin²xᵢ) for i = 1 to 4 has a minimum value of 81. -/
theorem min_value_of_product (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0)
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) *
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) *
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) *
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_product_l800_80037


namespace NUMINAMATH_CALUDE_shortest_side_is_15_l800_80029

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of the first segment of the hypotenuse -/
  segment1 : ℝ
  /-- The length of the second segment of the hypotenuse -/
  segment2 : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- Assumption that segment1 is positive -/
  segment1_pos : segment1 > 0
  /-- Assumption that segment2 is positive -/
  segment2_pos : segment2 > 0
  /-- Assumption that radius is positive -/
  radius_pos : radius > 0

/-- The length of the shortest side in a right triangle with an inscribed circle -/
def shortest_side (t : RightTriangleWithInscribedCircle) : ℝ :=
  sorry

/-- Theorem stating that the shortest side is 15 units under given conditions -/
theorem shortest_side_is_15 (t : RightTriangleWithInscribedCircle) 
  (h1 : t.segment1 = 7) 
  (h2 : t.segment2 = 9) 
  (h3 : t.radius = 5) : 
  shortest_side t = 15 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_is_15_l800_80029


namespace NUMINAMATH_CALUDE_shaded_fraction_of_square_l800_80067

theorem shaded_fraction_of_square (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 4 →
  triangle_base = 3 →
  triangle_height = 2 →
  (square_side^2 - 2 * (triangle_base * triangle_height / 2)) / square_side^2 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_square_l800_80067


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l800_80014

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l800_80014


namespace NUMINAMATH_CALUDE_hay_delivery_ratio_l800_80074

theorem hay_delivery_ratio : 
  let initial_bales : ℕ := 10
  let initial_cost_per_bale : ℕ := 15
  let new_cost_per_bale : ℕ := 18
  let additional_cost : ℕ := 210
  let new_bales : ℕ := (initial_bales * initial_cost_per_bale + additional_cost) / new_cost_per_bale
  (new_bales : ℚ) / initial_bales = 2 := by
  sorry

end NUMINAMATH_CALUDE_hay_delivery_ratio_l800_80074


namespace NUMINAMATH_CALUDE_mike_ride_distance_l800_80060

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  base_fare : ℝ
  per_mile_rate : ℝ
  additional_fee : ℝ
  distance : ℝ

/-- Calculates the total fare for a taxi ride -/
def total_fare (ride : TaxiRide) : ℝ :=
  ride.base_fare + ride.per_mile_rate * ride.distance + ride.additional_fee

/-- Proves that Mike's ride was 42 miles long given the conditions -/
theorem mike_ride_distance (mike annie : TaxiRide) 
    (h1 : mike.base_fare = 2.5)
    (h2 : mike.per_mile_rate = 0.25)
    (h3 : mike.additional_fee = 0)
    (h4 : annie.base_fare = 2.5)
    (h5 : annie.per_mile_rate = 0.25)
    (h6 : annie.additional_fee = 5)
    (h7 : annie.distance = 22)
    (h8 : total_fare mike = total_fare annie) : mike.distance = 42 := by
  sorry

#check mike_ride_distance

end NUMINAMATH_CALUDE_mike_ride_distance_l800_80060


namespace NUMINAMATH_CALUDE_special_function_result_l800_80032

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

theorem special_function_result (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 ≠ 0) :
  (f 3 - f 4) / f 2 = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_special_function_result_l800_80032


namespace NUMINAMATH_CALUDE_system_solution_l800_80097

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ - y₁ = 2 ∧ x₁^2 - 2*x₁*y₁ - 3*y₁^2 = 0 ∧ x₁ = 3 ∧ y₁ = 1) ∧
    (x₂ - y₂ = 2 ∧ x₂^2 - 2*x₂*y₂ - 3*y₂^2 = 0 ∧ x₂ = 1 ∧ y₂ = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l800_80097


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l800_80078

theorem smallest_yellow_marbles (n : ℕ) (hn : n > 0) : ∃ m : ℕ,
  n / 2 + n / 3 + 12 + m = n ∧ 
  m = 0 ∧ 
  ∀ k : ℕ, k < m → ¬(∃ t : ℕ, t > 0 ∧ t / 2 + t / 3 + 12 + k = t) :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l800_80078


namespace NUMINAMATH_CALUDE_smallest_value_problem_l800_80085

theorem smallest_value_problem (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 → m = 77 →
  Nat.gcd m n = x + 7 →
  Nat.lcm m n = x * (x + 7) →
  ∃ (n_min : ℕ), n_min > 0 ∧ n_min ≤ n ∧ n_min = 22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_problem_l800_80085


namespace NUMINAMATH_CALUDE_a_divisibility_characterization_l800_80006

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ
  | 0 => 3
  | 1 => 9
  | (n + 2) => 4 * a (n + 1) - 3 * a n - 4 * (n + 2) + 2

/-- Predicate for n such that a_n is divisible by 9 -/
def is_divisible_by_9 (n : ℕ) : Prop :=
  n = 1 ∨ n % 9 = 7 ∨ n % 9 = 8

theorem a_divisibility_characterization :
  ∀ n : ℕ, 9 ∣ a n ↔ is_divisible_by_9 n :=
sorry

end NUMINAMATH_CALUDE_a_divisibility_characterization_l800_80006


namespace NUMINAMATH_CALUDE_slope_of_line_with_30_degree_inclination_l800_80058

theorem slope_of_line_with_30_degree_inclination :
  let angle_of_inclination : ℝ := 30 * π / 180
  let slope : ℝ := Real.tan angle_of_inclination
  slope = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_slope_of_line_with_30_degree_inclination_l800_80058


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l800_80096

theorem roots_of_quadratic (a b : ℝ) : 
  (a * b ≠ 0) →
  (a^2 + 2*b*a + a = 0) →
  (b^2 + 2*b*b + a = 0) →
  (a = -3 ∧ b = 1) := by
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l800_80096


namespace NUMINAMATH_CALUDE_bag_composition_for_expected_value_l800_80070

/-- Represents the contents of a bag of slips --/
structure BagOfSlips where
  threes : ℕ
  fives : ℕ
  eights : ℕ

/-- Calculates the expected value of a randomly drawn slip --/
def expectedValue (bag : BagOfSlips) : ℚ :=
  (3 * bag.threes + 5 * bag.fives + 8 * bag.eights) / 20

/-- Theorem statement --/
theorem bag_composition_for_expected_value :
  ∃ (bag : BagOfSlips),
    bag.threes + bag.fives + bag.eights = 20 ∧
    expectedValue bag = 57/10 ∧
    bag.threes = 4 ∧
    bag.fives = 10 ∧
    bag.eights = 6 := by
  sorry

end NUMINAMATH_CALUDE_bag_composition_for_expected_value_l800_80070


namespace NUMINAMATH_CALUDE_log_cos_acute_angle_l800_80019

theorem log_cos_acute_angle (A m n : ℝ) : 
  0 < A → A < π/2 →
  Real.log (1 + Real.sin A) = m →
  Real.log (1 / (1 - Real.sin A)) = n →
  Real.log (Real.cos A) = (1/2) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_log_cos_acute_angle_l800_80019


namespace NUMINAMATH_CALUDE_cubic_polynomial_theorem_l800_80036

-- Define the cubic polynomial whose roots are a, b, c
def cubic (x : ℝ) : ℝ := x^3 + 4*x^2 + 6*x + 9

-- Define the properties of P
def P_properties (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
  cubic a = 0 ∧ cubic b = 0 ∧ cubic c = 0 ∧
  P a = b + c ∧ P b = a + c ∧ P c = a + b ∧
  P (a + b + c) = -20

-- Theorem statement
theorem cubic_polynomial_theorem :
  ∀ (P : ℝ → ℝ) (a b c : ℝ),
  P_properties P a b c →
  (∀ x, P x = 16*x^3 + 64*x^2 + 90*x + 140) :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_theorem_l800_80036


namespace NUMINAMATH_CALUDE_john_ultramarathon_distance_l800_80080

/-- Calculates the total distance John can run after training -/
def johnRunningDistance (initialTime : ℝ) (timeIncrease : ℝ) (initialSpeed : ℝ) (speedIncrease : ℝ) : ℝ :=
  (initialTime * (1 + timeIncrease)) * (initialSpeed + speedIncrease)

theorem john_ultramarathon_distance :
  johnRunningDistance 8 0.75 8 4 = 168 := by
  sorry

end NUMINAMATH_CALUDE_john_ultramarathon_distance_l800_80080


namespace NUMINAMATH_CALUDE_village_population_l800_80021

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.15) = 3294 → P = 4080 := by
sorry

end NUMINAMATH_CALUDE_village_population_l800_80021


namespace NUMINAMATH_CALUDE_club_members_problem_l800_80089

theorem club_members_problem (current_members : ℕ) : 
  (2 * current_members + 5 = current_members + 15) → 
  current_members = 10 := by
sorry

end NUMINAMATH_CALUDE_club_members_problem_l800_80089


namespace NUMINAMATH_CALUDE_shifted_quadratic_function_l800_80043

/-- The original quadratic function -/
def original_function (x : ℝ) : ℝ := x^2

/-- The shifted function -/
def shifted_function (x : ℝ) : ℝ := (x - 3)^2 - 2

/-- Theorem stating that the shifted function is equivalent to shifting the original function -/
theorem shifted_quadratic_function (x : ℝ) : 
  shifted_function x = original_function (x - 3) - 2 := by sorry

end NUMINAMATH_CALUDE_shifted_quadratic_function_l800_80043


namespace NUMINAMATH_CALUDE_adam_strawberries_l800_80016

/-- The number of strawberries Adam had left -/
def strawberries_left : ℕ := 33

/-- The number of strawberries Adam ate -/
def strawberries_eaten : ℕ := 2

/-- The initial number of strawberries Adam picked -/
def initial_strawberries : ℕ := strawberries_left + strawberries_eaten

theorem adam_strawberries : initial_strawberries = 35 := by
  sorry

end NUMINAMATH_CALUDE_adam_strawberries_l800_80016


namespace NUMINAMATH_CALUDE_remaining_pages_l800_80071

-- Define the total number of pages
def total_pages : ℕ := 120

-- Define the percentage used for the science project
def science_project_percentage : ℚ := 25 / 100

-- Define the number of pages used for math homework
def math_homework_pages : ℕ := 10

-- Theorem statement
theorem remaining_pages :
  total_pages - (total_pages * science_project_percentage).floor - math_homework_pages = 80 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pages_l800_80071


namespace NUMINAMATH_CALUDE_y_value_l800_80068

theorem y_value : ∀ y : ℚ, (2 / 5 - 1 / 7 : ℚ) = 14 / y → y = 490 / 9 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l800_80068


namespace NUMINAMATH_CALUDE_job_completion_time_l800_80051

theorem job_completion_time (y : ℝ) : y > 0 → (
  (1 / (y + 4) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) ↔ y = 2
) := by sorry

end NUMINAMATH_CALUDE_job_completion_time_l800_80051


namespace NUMINAMATH_CALUDE_problem_solution_l800_80086

def A : Set ℝ := {x | x^2 + 5*x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*(m+1)*x + m^2 - 3 = 0}

theorem problem_solution :
  (A ∪ B 0 = {-6, 1, -3}) ∧
  (∀ m : ℝ, B m ⊆ A ↔ m ≤ -2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l800_80086
