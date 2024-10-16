import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_3_A_subset_B_iff_m_in_range_l3000_300033

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 18 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 8 ≤ x ∧ x ≤ m + 4}

-- Statement 1
theorem complement_A_intersect_B_when_m_3 : 
  (Set.univ \ A) ∩ B 3 = {x | -5 ≤ x ∧ x < -3 ∨ 6 < x ∧ x ≤ 7} := by sorry

-- Statement 2
theorem A_subset_B_iff_m_in_range : 
  ∀ m, A ∩ B m = A ↔ 2 ≤ m ∧ m ≤ 5 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_when_m_3_A_subset_B_iff_m_in_range_l3000_300033


namespace NUMINAMATH_CALUDE_parabola_minimum_value_l3000_300067

theorem parabola_minimum_value (m : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + 2*m*x + m + 2 ≥ -3) ∧ 
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + 2*m*x + m + 2 = -3) →
  m = 3 ∨ m = (1 - Real.sqrt 21) / 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_minimum_value_l3000_300067


namespace NUMINAMATH_CALUDE_smallest_cookie_count_l3000_300000

theorem smallest_cookie_count : ∃ (x : ℕ), x > 0 ∧
  x % 6 = 5 ∧ x % 8 = 7 ∧ x % 9 = 2 ∧
  ∀ (y : ℕ), y > 0 → y % 6 = 5 → y % 8 = 7 → y % 9 = 2 → x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_count_l3000_300000


namespace NUMINAMATH_CALUDE_train_length_calculation_l3000_300092

/-- Calculates the length of a train given its speed, the time it takes to pass a platform, and the length of the platform. -/
theorem train_length_calculation (train_speed : Real) (platform_pass_time : Real) (platform_length : Real) :
  train_speed = 60 →
  platform_pass_time = 23.998080153587715 →
  platform_length = 260 →
  let train_speed_mps := train_speed * 1000 / 3600
  let total_distance := train_speed_mps * platform_pass_time
  let train_length := total_distance - platform_length
  train_length = 139.968003071754 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3000_300092


namespace NUMINAMATH_CALUDE_min_product_of_tangents_l3000_300012

theorem min_product_of_tangents (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_tangents_l3000_300012


namespace NUMINAMATH_CALUDE_meaningful_square_root_l3000_300076

theorem meaningful_square_root (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2023) ↔ x ≥ 2023 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_square_root_l3000_300076


namespace NUMINAMATH_CALUDE_polyhedron_property_l3000_300001

/-- Represents a convex polyhedron with the given properties -/
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  t : ℕ  -- Number of triangular faces
  s : ℕ  -- Number of square faces
  euler_formula : V - E + F = 2
  face_count : F = 42
  face_types : F = t + s
  edge_relation : E = (3 * t + 4 * s) / 2
  vertex_degree : 13 * V = 2 * E

/-- The main theorem to be proved -/
theorem polyhedron_property (p : ConvexPolyhedron) : 100 * 3 + 10 * 2 + p.V = 337 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l3000_300001


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l3000_300002

theorem square_perimeter_sum (y : ℝ) (h1 : y^2 + (2*y)^2 = 145) (h2 : (2*y)^2 - y^2 = 105) :
  4*y + 8*y = 12 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l3000_300002


namespace NUMINAMATH_CALUDE_a_range_l3000_300056

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

def inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - a * x + 1 > 0

theorem a_range (a : ℝ) (h_a : a > 0) :
  (¬(is_monotonically_increasing (λ x => a^x)) ∨
   ¬(inequality_holds a)) ∧
  (is_monotonically_increasing (λ x => a^x) ∨
   inequality_holds a) →
  a ∈ Set.Ioc 0 1 ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_a_range_l3000_300056


namespace NUMINAMATH_CALUDE_students_taking_both_music_and_art_l3000_300052

theorem students_taking_both_music_and_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (neither : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 50) 
  (h3 : art = 20) 
  (h4 : neither = 440) : 
  total - neither - (music + art - (total - neither)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_both_music_and_art_l3000_300052


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3000_300082

theorem inequality_solution_set (x : ℝ) : (x - 1) / (x + 2) > 0 ↔ x < -2 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3000_300082


namespace NUMINAMATH_CALUDE_solution_to_system_l3000_300078

theorem solution_to_system (x y : ℝ) :
  x^5 + y^5 = 33 ∧ x + y = 3 →
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
sorry

end NUMINAMATH_CALUDE_solution_to_system_l3000_300078


namespace NUMINAMATH_CALUDE_q_div_p_equals_450_l3000_300087

def total_slips : ℕ := 50
def num_range : ℕ := 10
def slips_per_num : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := num_range / (total_slips.choose drawn_slips)
def q : ℚ := (num_range.choose 2) * (slips_per_num.choose 3) * (slips_per_num.choose 2) / (total_slips.choose drawn_slips)

theorem q_div_p_equals_450 : q / p = 450 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_450_l3000_300087


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3000_300058

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) - (a - b) / (a + b) = 0) : 
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3000_300058


namespace NUMINAMATH_CALUDE_pyramid_sculpture_surface_area_l3000_300051

/-- Represents a cube sculpture with three layers -/
structure CubeSculpture where
  top_layer : Nat
  middle_layer : Nat
  bottom_layer : Nat

/-- Calculates the painted surface area of a cube sculpture -/
def painted_surface_area (sculpture : CubeSculpture) : Nat :=
  sorry

/-- The specific sculpture described in the problem -/
def pyramid_sculpture : CubeSculpture :=
  { top_layer := 1
  , middle_layer := 5
  , bottom_layer := 13 }

theorem pyramid_sculpture_surface_area :
  painted_surface_area pyramid_sculpture = 31 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_sculpture_surface_area_l3000_300051


namespace NUMINAMATH_CALUDE_toy_cost_price_l3000_300091

theorem toy_cost_price (total_selling_price : ℕ) (num_toys_sold : ℕ) (num_toys_gain : ℕ) :
  total_selling_price = 18900 →
  num_toys_sold = 18 →
  num_toys_gain = 3 →
  ∃ (cost_price : ℕ),
    cost_price * num_toys_sold + cost_price * num_toys_gain = total_selling_price ∧
    cost_price = 900 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_price_l3000_300091


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l3000_300044

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 10*x - 6*y - 34) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l3000_300044


namespace NUMINAMATH_CALUDE_impossible_transformation_number_54_impossible_l3000_300031

/-- Represents the allowed operations on the number -/
inductive Operation
  | Multiply2
  | Multiply3
  | Divide2
  | Divide3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Multiply2 => n * 2
  | Operation.Multiply3 => n * 3
  | Operation.Divide2 => n / 2
  | Operation.Divide3 => n / 3

/-- Applies a sequence of operations to a number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- Returns the sum of exponents in the prime factorization of a number -/
def sumOfExponents (n : ℕ) : ℕ :=
  (Nat.factorization n).sum (fun _ e => e)

/-- Theorem stating that it's impossible to transform 12 into 54 with exactly 60 operations -/
theorem impossible_transformation :
  ∀ (ops : List Operation), ops.length = 60 → applyOperations 12 ops ≠ 54 := by
  sorry

/-- Corollary: The number 54 cannot appear on the screen after exactly one minute -/
theorem number_54_impossible : ∃ (ops : List Operation), ops.length = 60 ∧ applyOperations 12 ops = 54 → False := by
  sorry

end NUMINAMATH_CALUDE_impossible_transformation_number_54_impossible_l3000_300031


namespace NUMINAMATH_CALUDE_david_catches_cory_l3000_300059

/-- The length of the track in meters -/
def track_length : ℝ := 600

/-- Cory's initial lead in meters -/
def initial_lead : ℝ := 50

/-- David's speed relative to Cory's -/
def speed_ratio : ℝ := 1.5

/-- Number of laps David runs when he first catches up to Cory -/
def david_laps : ℝ := 2

theorem david_catches_cory :
  ∃ (cory_speed : ℝ), cory_speed > 0 →
  let david_speed := speed_ratio * cory_speed
  let catch_up_distance := david_laps * track_length
  catch_up_distance * (1 / david_speed - 1 / cory_speed) = initial_lead := by
  sorry

end NUMINAMATH_CALUDE_david_catches_cory_l3000_300059


namespace NUMINAMATH_CALUDE_sea_turtle_collection_age_difference_l3000_300048

/-- Converts a number from base 8 to base 10 -/
def octalToDecimal (n : ℕ) : ℕ :=
  (n % 10) + 8 * ((n / 10) % 10) + 64 * (n / 100)

theorem sea_turtle_collection_age_difference : 
  octalToDecimal 724 - octalToDecimal 560 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sea_turtle_collection_age_difference_l3000_300048


namespace NUMINAMATH_CALUDE_trefoils_per_case_l3000_300026

theorem trefoils_per_case (total_boxes : ℕ) (total_cases : ℕ) (boxes_per_case : ℕ) : 
  total_boxes = 54 → total_cases = 9 → boxes_per_case = total_boxes / total_cases → boxes_per_case = 6 := by
  sorry

end NUMINAMATH_CALUDE_trefoils_per_case_l3000_300026


namespace NUMINAMATH_CALUDE_negation_exists_product_zero_l3000_300039

open Real

theorem negation_exists_product_zero (f g : ℝ → ℝ) :
  (¬ ∃ x₀ : ℝ, f x₀ * g x₀ = 0) ↔ (∀ x : ℝ, f x ≠ 0 ∧ g x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_exists_product_zero_l3000_300039


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l3000_300079

/-- The area of a square with adjacent vertices at (1,3) and (4,6) is 18 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (4, 6)
  let distance_squared := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
  distance_squared = 18 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l3000_300079


namespace NUMINAMATH_CALUDE_line_canonical_form_l3000_300062

/-- Given two planes that intersect to form a line, prove that the line can be represented in canonical form. -/
theorem line_canonical_form (x y z : ℝ) : 
  (2*x - y + 3*z = 1) ∧ (5*x + 4*y - z = 7) →
  ∃ (t : ℝ), x = -11*t ∧ y = 17*t + 2 ∧ z = 13*t + 1 :=
by sorry

end NUMINAMATH_CALUDE_line_canonical_form_l3000_300062


namespace NUMINAMATH_CALUDE_henry_final_book_count_l3000_300069

def initial_books : ℕ := 99
def boxes_donated : ℕ := 3
def books_per_box : ℕ := 15
def room_books : ℕ := 21
def coffee_table_books : ℕ := 4
def kitchen_books : ℕ := 18
def free_books_taken : ℕ := 12

theorem henry_final_book_count :
  initial_books - 
  (boxes_donated * books_per_box + room_books + coffee_table_books + kitchen_books) + 
  free_books_taken = 23 := by
  sorry

end NUMINAMATH_CALUDE_henry_final_book_count_l3000_300069


namespace NUMINAMATH_CALUDE_fraction_equality_l3000_300007

/-- Given two amounts a and b, prove that the fraction of b that equals 2/3 of a is 2/3 -/
theorem fraction_equality (a b : ℚ) (h1 : a + b = 1210) (h2 : b = 484) : 
  ∃ x : ℚ, x * b = 2/3 * a ∧ x = 2/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3000_300007


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l3000_300068

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) ↔
  (x^2 / 25 + y^2 / 21 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l3000_300068


namespace NUMINAMATH_CALUDE_min_value_expression_l3000_300019

theorem min_value_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 ≥ 9 ∧
  ∃ x y z : ℝ, 2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 5 ∧
    (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3000_300019


namespace NUMINAMATH_CALUDE_smallest_number_l3000_300071

theorem smallest_number (a b c d : ℤ) (ha : a = -4) (hb : b = -3) (hc : c = 0) (hd : d = 1) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3000_300071


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_sqrt_27_times_sqrt_32_div_sqrt_6_l3000_300098

theorem sqrt_product_quotient :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
  Real.sqrt (a * b) / Real.sqrt c = Real.sqrt a * Real.sqrt b / Real.sqrt c :=
by sorry

theorem sqrt_27_times_sqrt_32_div_sqrt_6 :
  Real.sqrt 27 * Real.sqrt 32 / Real.sqrt 6 = 12 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_sqrt_27_times_sqrt_32_div_sqrt_6_l3000_300098


namespace NUMINAMATH_CALUDE_percentage_difference_in_gain_l3000_300020

def cost_price : ℝ := 400
def selling_price1 : ℝ := 360
def selling_price2 : ℝ := 340

def gain1 : ℝ := selling_price1 - cost_price
def gain2 : ℝ := selling_price2 - cost_price

def difference_in_gain : ℝ := gain1 - gain2

theorem percentage_difference_in_gain :
  (difference_in_gain / cost_price) * 100 = 5 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_in_gain_l3000_300020


namespace NUMINAMATH_CALUDE_remainder_six_divisor_count_l3000_300049

theorem remainder_six_divisor_count : 
  ∃! (n : ℕ), n > 6 ∧ 67 % n = 6 :=
sorry

end NUMINAMATH_CALUDE_remainder_six_divisor_count_l3000_300049


namespace NUMINAMATH_CALUDE_rachel_homework_l3000_300094

theorem rachel_homework (total_pages reading_pages biology_pages : ℕ) 
  (h1 : total_pages = 15)
  (h2 : reading_pages = 3)
  (h3 : biology_pages = 10) : 
  total_pages - reading_pages - biology_pages = 2 := by
sorry

end NUMINAMATH_CALUDE_rachel_homework_l3000_300094


namespace NUMINAMATH_CALUDE_matrix_cube_sum_l3000_300023

/-- Given a 3x3 complex matrix N of the form [d e f; e f d; f d e] where N^2 = I and def = -1,
    the possible values of d^3 + e^3 + f^3 are 2 and 4. -/
theorem matrix_cube_sum (d e f : ℂ) : 
  let N : Matrix (Fin 3) (Fin 3) ℂ := !![d, e, f; e, f, d; f, d, e]
  (N ^ 2 = 1 ∧ d * e * f = -1) →
  (d^3 + e^3 + f^3 = 2 ∨ d^3 + e^3 + f^3 = 4) :=
by sorry

end NUMINAMATH_CALUDE_matrix_cube_sum_l3000_300023


namespace NUMINAMATH_CALUDE_original_polygon_sides_l3000_300080

theorem original_polygon_sides (n : ℕ) : 
  (n + 1 - 2) * 180 = 1620 → n = 10 := by sorry

end NUMINAMATH_CALUDE_original_polygon_sides_l3000_300080


namespace NUMINAMATH_CALUDE_infinite_twin_pretty_numbers_l3000_300016

/-- A positive integer is a "pretty number" if each of its prime factors appears with an exponent of at least 2 in its prime factorization. -/
def is_pretty_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (∃ k : ℕ, k ≥ 2 ∧ p^k ∣ n)

/-- Two consecutive positive integers that are both "pretty numbers" are called "twin pretty numbers." -/
def is_twin_pretty_numbers (n : ℕ) : Prop :=
  is_pretty_number n ∧ is_pretty_number (n + 1)

/-- For any pair of twin pretty numbers, there exists a larger pair of twin pretty numbers. -/
theorem infinite_twin_pretty_numbers :
  ∀ n : ℕ, is_twin_pretty_numbers n →
    ∃ m : ℕ, m > n + 1 ∧ is_twin_pretty_numbers m :=
sorry

end NUMINAMATH_CALUDE_infinite_twin_pretty_numbers_l3000_300016


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3000_300097

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum_odd : a 1 + a 3 + a 5 = 105)
  (h_sum_even : a 2 + a 4 + a 6 = 99) :
  ∃ d : ℝ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3000_300097


namespace NUMINAMATH_CALUDE_percent_of_percent_l3000_300089

theorem percent_of_percent (y : ℝ) (h : y ≠ 0) :
  (0.6 * (0.3 * y)) / y * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l3000_300089


namespace NUMINAMATH_CALUDE_problem_solution_l3000_300065

theorem problem_solution (x y : ℚ) (hx : x = 5/7) (hy : y = 7/5) : 
  (1/3) * x^8 * y^9 + 1/7 = 64/105 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3000_300065


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3000_300041

theorem cubic_equation_solution (b : ℝ) : 
  let x := b
  let c := 0
  x^3 + c^2 = (b - x)^2 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3000_300041


namespace NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l3000_300050

theorem cauchy_schwarz_like_inequality (a b c d : ℝ) :
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_like_inequality_l3000_300050


namespace NUMINAMATH_CALUDE_endpoint_sum_thirteen_l3000_300061

/-- Given a line segment with one endpoint (6,1) and midpoint (3,7),
    the sum of the coordinates of the other endpoint is 13. -/
theorem endpoint_sum_thirteen (x y : ℝ) : 
  (6 + x) / 2 = 3 ∧ (1 + y) / 2 = 7 → x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_thirteen_l3000_300061


namespace NUMINAMATH_CALUDE_cranberry_calculation_l3000_300057

/-- The initial number of cranberries in the bog -/
def initial_cranberries : ℕ := 60000

/-- The fraction of cranberries harvested by humans -/
def human_harvest_fraction : ℚ := 2/5

/-- The number of cranberries eaten by elk -/
def elk_eaten : ℕ := 20000

/-- The number of cranberries left after harvesting and elk eating -/
def remaining_cranberries : ℕ := 16000

/-- Theorem stating that the initial number of cranberries is correct given the conditions -/
theorem cranberry_calculation :
  (1 - human_harvest_fraction) * initial_cranberries - elk_eaten = remaining_cranberries :=
by sorry

end NUMINAMATH_CALUDE_cranberry_calculation_l3000_300057


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l3000_300022

def probability_n_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1/2)^n

theorem fair_coin_probability_difference : 
  (probability_n_heads 4 3) - (probability_n_heads 4 4) = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l3000_300022


namespace NUMINAMATH_CALUDE_austin_started_with_80_l3000_300043

/-- The amount of money Austin started with, given the conditions of the problem. -/
def austin_starting_amount : ℚ :=
  let num_robots : ℕ := 7
  let robot_cost : ℚ := 875 / 100
  let total_tax : ℚ := 722 / 100
  let change : ℚ := 1153 / 100
  num_robots * robot_cost + total_tax + change

/-- Theorem stating that Austin started with $80. -/
theorem austin_started_with_80 : austin_starting_amount = 80 := by
  sorry

end NUMINAMATH_CALUDE_austin_started_with_80_l3000_300043


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l3000_300010

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side length opposite to angle A
  b : ℝ  -- side length opposite to angle B
  c : ℝ  -- side length opposite to angle C
  A : ℝ  -- angle A in radians
  B : ℝ  -- angle B in radians
  C : ℝ  -- angle C in radians

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 4/5

-- Theorem 1
theorem theorem_1 (t : Triangle) (h : triangle_conditions t) (h_A : t.A = Real.pi/6) :
  t.a = 5/3 := by sorry

-- Theorem 2
theorem theorem_2 (t : Triangle) (h : triangle_conditions t) 
  (h_area : (1/2) * t.a * t.c * Real.sin t.B = 3) :
  t.a = Real.sqrt 10 ∧ t.c = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l3000_300010


namespace NUMINAMATH_CALUDE_two_economic_reasons_exist_l3000_300054

/-- Represents a European country --/
structure EuropeanCountry where
  name : String

/-- Represents an economic reason for offering free or low-cost education to foreign citizens --/
structure EconomicReason where
  description : String

/-- Represents a government policy --/
structure GovernmentPolicy where
  description : String

/-- Predicate to check if a policy offers free or low-cost education to foreign citizens --/
def is_free_education_policy (policy : GovernmentPolicy) : Prop :=
  policy.description = "Offer free or low-cost education to foreign citizens"

/-- Predicate to check if a reason is valid for a given country and policy --/
def is_valid_reason (country : EuropeanCountry) (policy : GovernmentPolicy) (reason : EconomicReason) : Prop :=
  is_free_education_policy policy ∧ 
  (reason.description = "International Agreements" ∨ reason.description = "Addressing Demographic Changes")

/-- Theorem stating that there exist at least two distinct economic reasons for the policy --/
theorem two_economic_reasons_exist (country : EuropeanCountry) (policy : GovernmentPolicy) :
  is_free_education_policy policy →
  ∃ (reason1 reason2 : EconomicReason), 
    reason1 ≠ reason2 ∧ 
    is_valid_reason country policy reason1 ∧ 
    is_valid_reason country policy reason2 :=
sorry

end NUMINAMATH_CALUDE_two_economic_reasons_exist_l3000_300054


namespace NUMINAMATH_CALUDE_symmetric_series_sum_sqrt_l3000_300042

def symmetric_series (n : ℕ) : ℕ := 
  2 * (n * (n + 1) / 2) + (n + 1)

theorem symmetric_series_sum_sqrt (n : ℕ) : 
  Real.sqrt (symmetric_series n) = (n : ℝ) + 0.5 :=
sorry

end NUMINAMATH_CALUDE_symmetric_series_sum_sqrt_l3000_300042


namespace NUMINAMATH_CALUDE_glass_bottles_in_second_scenario_l3000_300009

/-- The weight of a glass bottle in grams -/
def glass_weight : ℕ := 200

/-- The weight of a plastic bottle in grams -/
def plastic_weight : ℕ := 50

/-- The number of glass bottles in the first scenario -/
def first_scenario_bottles : ℕ := 3

/-- The number of plastic bottles in the second scenario -/
def second_scenario_plastic : ℕ := 5

/-- The total weight in the first scenario in grams -/
def first_scenario_weight : ℕ := 600

/-- The total weight in the second scenario in grams -/
def second_scenario_weight : ℕ := 1050

/-- The weight difference between a glass and plastic bottle in grams -/
def weight_difference : ℕ := 150

theorem glass_bottles_in_second_scenario :
  ∃ x : ℕ, 
    first_scenario_bottles * glass_weight = first_scenario_weight ∧
    glass_weight = plastic_weight + weight_difference ∧
    x * glass_weight + second_scenario_plastic * plastic_weight = second_scenario_weight ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_glass_bottles_in_second_scenario_l3000_300009


namespace NUMINAMATH_CALUDE_no_integer_solution_3x2_plus_2_eq_y2_l3000_300034

theorem no_integer_solution_3x2_plus_2_eq_y2 :
  ∀ (x y : ℤ), 3 * x^2 + 2 ≠ y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_3x2_plus_2_eq_y2_l3000_300034


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l3000_300003

theorem stratified_sampling_medium_stores 
  (total_stores : ℕ) 
  (medium_stores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_stores = 300) 
  (h2 : medium_stores = 75) 
  (h3 : sample_size = 20) :
  ⌊(medium_stores : ℚ) / total_stores * sample_size⌋ = 5 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l3000_300003


namespace NUMINAMATH_CALUDE_decision_not_basic_l3000_300015

-- Define the type for flowchart structures
inductive FlowchartStructure
  | Sequence
  | Condition
  | Loop
  | Decision

-- Define the set of basic logic structures
def basic_logic_structures : Set FlowchartStructure :=
  {FlowchartStructure.Sequence, FlowchartStructure.Condition, FlowchartStructure.Loop}

-- Theorem: Decision structure is not in the set of basic logic structures
theorem decision_not_basic : FlowchartStructure.Decision ∉ basic_logic_structures := by
  sorry

end NUMINAMATH_CALUDE_decision_not_basic_l3000_300015


namespace NUMINAMATH_CALUDE_race_speed_ratio_l3000_300024

theorem race_speed_ratio (course_length : ℝ) (head_start : ℝ) 
  (h1 : course_length = 84)
  (h2 : head_start = 63)
  (h3 : course_length > head_start)
  (h4 : head_start > 0) :
  ∃ (speed_a speed_b : ℝ),
    speed_a > 0 ∧ speed_b > 0 ∧
    (course_length / speed_a = (course_length - head_start) / speed_b) ∧
    speed_a = 4 * speed_b :=
by sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l3000_300024


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3000_300096

/-- The distance between the foci of the hyperbola x^2 - 6x - 4y^2 - 8y = 27 is 4√10 -/
theorem hyperbola_foci_distance :
  ∃ (a b c : ℝ),
    (∀ x y : ℝ, x^2 - 6*x - 4*y^2 - 8*y = 27 ↔ (x - 3)^2 / a^2 - (y + 1)^2 / b^2 = 1) ∧
    c^2 = a^2 + b^2 ∧
    2*c = 4 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3000_300096


namespace NUMINAMATH_CALUDE_prob_head_fair_coin_l3000_300081

/-- A fair coin with two sides. -/
structure FairCoin where
  sides : Fin 2
  prob_head : ℝ
  prob_tail : ℝ
  sum_to_one : prob_head + prob_tail = 1
  equal_prob : prob_head = prob_tail

/-- The probability of getting a head in a fair coin toss is 1/2. -/
theorem prob_head_fair_coin (c : FairCoin) : c.prob_head = 1/2 := by
  sorry

#check prob_head_fair_coin

end NUMINAMATH_CALUDE_prob_head_fair_coin_l3000_300081


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_relation_l3000_300083

theorem circle_radius_from_area_circumference_relation : 
  ∀ r : ℝ, r > 0 → (3 * (2 * Real.pi * r) = Real.pi * r^2) → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_relation_l3000_300083


namespace NUMINAMATH_CALUDE_inequality_solutions_l3000_300064

theorem inequality_solutions :
  (∀ x : ℝ, 3 * x > 2 * (1 - x) ↔ x > 2/5) ∧
  (∀ x : ℝ, (3 * x - 7) / 2 ≤ x - 2 ∧ 4 * (x - 1) > 4 ↔ 2 < x ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l3000_300064


namespace NUMINAMATH_CALUDE_student_grade_problem_l3000_300086

theorem student_grade_problem (grade2 grade3 average : ℚ) 
  (h1 : grade2 = 80/100)
  (h2 : grade3 = 85/100)
  (h3 : average = 75/100)
  (h4 : (grade1 + grade2 + grade3) / 3 = average) :
  grade1 = 60/100 := by
  sorry

end NUMINAMATH_CALUDE_student_grade_problem_l3000_300086


namespace NUMINAMATH_CALUDE_base7_calculation_l3000_300029

/-- Represents a number in base 7 --/
def Base7 : Type := Nat

/-- Converts a base 7 number to its decimal representation --/
def toDecimal (n : Base7) : Nat := sorry

/-- Converts a decimal number to its base 7 representation --/
def toBase7 (n : Nat) : Base7 := sorry

/-- Adds two base 7 numbers --/
def addBase7 (a b : Base7) : Base7 := sorry

/-- Subtracts two base 7 numbers --/
def subBase7 (a b : Base7) : Base7 := sorry

theorem base7_calculation : 
  let a := toBase7 2000
  let b := toBase7 1256
  let c := toBase7 345
  let d := toBase7 1042
  subBase7 (addBase7 (subBase7 a b) c) d = toBase7 0 := by sorry

end NUMINAMATH_CALUDE_base7_calculation_l3000_300029


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3000_300070

/-- The speed of a boat in still water given downstream travel information -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 68)
  (h3 : downstream_time = 4) :
  downstream_distance / downstream_time - stream_speed = 13 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3000_300070


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3000_300085

/-- Given a parabola y = ax^2 where a < 0, its latus rectum has the equation y = -1/(4a) -/
theorem latus_rectum_of_parabola (a : ℝ) (h : a < 0) :
  let parabola := λ x : ℝ => a * x^2
  let latus_rectum := λ y : ℝ => y = -1 / (4 * a)
  ∀ x : ℝ, latus_rectum (parabola x) := by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3000_300085


namespace NUMINAMATH_CALUDE_optimal_chair_removal_l3000_300030

/-- Represents the number of chairs in a complete row -/
def chairs_per_row : ℕ := 11

/-- Represents the initial total number of chairs -/
def initial_chairs : ℕ := 110

/-- Represents the number of students attending the assembly -/
def students : ℕ := 70

/-- Represents the number of chairs to be removed -/
def chairs_to_remove : ℕ := 33

/-- Proves that removing 33 chairs results in the optimal arrangement -/
theorem optimal_chair_removal :
  let remaining_chairs := initial_chairs - chairs_to_remove
  (remaining_chairs % chairs_per_row = 0) ∧
  (remaining_chairs ≥ students) ∧
  (∀ n : ℕ, n < chairs_to_remove →
    ((initial_chairs - n) % chairs_per_row = 0) →
    (initial_chairs - n < students ∨ initial_chairs - n > remaining_chairs)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_chair_removal_l3000_300030


namespace NUMINAMATH_CALUDE_total_raisins_l3000_300093

theorem total_raisins (yellow_raisins black_raisins : ℝ) 
  (h1 : yellow_raisins = 0.3)
  (h2 : black_raisins = 0.4) :
  yellow_raisins + black_raisins = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_total_raisins_l3000_300093


namespace NUMINAMATH_CALUDE_log_sequence_a_is_geometric_l3000_300088

def sequence_a : ℕ → ℝ
  | 0 => 2
  | n + 1 => (sequence_a n) ^ 2

theorem log_sequence_a_is_geometric :
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → Real.log (sequence_a (n + 1)) = r * Real.log (sequence_a n) := by
  sorry

end NUMINAMATH_CALUDE_log_sequence_a_is_geometric_l3000_300088


namespace NUMINAMATH_CALUDE_c_months_equals_six_l3000_300055

def total_cost : ℚ := 435
def a_horses : ℕ := 12
def a_months : ℕ := 8
def b_horses : ℕ := 16
def b_months : ℕ := 9
def c_horses : ℕ := 18
def b_payment : ℚ := 180

theorem c_months_equals_six :
  ∃ (x : ℕ), 
    (b_payment / total_cost) * (a_horses * a_months + b_horses * b_months + c_horses * x) = 
    b_horses * b_months ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_c_months_equals_six_l3000_300055


namespace NUMINAMATH_CALUDE_angle_sum_tangent_l3000_300021

theorem angle_sum_tangent (a β : Real) (ha : 0 < a ∧ a < π/2) (hβ : 0 < β ∧ β < π/2)
  (tan_a : Real.tan a = 2) (tan_β : Real.tan β = 3) :
  a + β = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_tangent_l3000_300021


namespace NUMINAMATH_CALUDE_reading_reward_pie_chart_l3000_300027

theorem reading_reward_pie_chart (agree disagree neutral : ℕ) 
  (h_ratio : (agree : ℚ) / (disagree : ℚ) = 7 / 2 ∧ (agree : ℚ) / (neutral : ℚ) = 7 / 1) :
  (360 : ℚ) * (agree : ℚ) / ((agree : ℚ) + (disagree : ℚ) + (neutral : ℚ)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_reading_reward_pie_chart_l3000_300027


namespace NUMINAMATH_CALUDE_math_team_combinations_l3000_300004

theorem math_team_combinations (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 4) (h2 : boys = 6) : 
  (Nat.choose girls 3) * (Nat.choose boys 2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l3000_300004


namespace NUMINAMATH_CALUDE_xy_equals_four_l3000_300090

theorem xy_equals_four (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_x : x = w)
  (h_y : y = w)
  (h_w : w + w = w * w)
  (h_z : z = 3) : 
  x * y = 4 := by
sorry

end NUMINAMATH_CALUDE_xy_equals_four_l3000_300090


namespace NUMINAMATH_CALUDE_factor_polynomial_l3000_300040

theorem factor_polynomial (x : ℝ) : 45 * x^3 + 135 * x^2 = 45 * x^2 * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3000_300040


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainder_sum_l3000_300011

theorem unique_divisor_with_remainder_sum (a b c : ℕ) : ∃! n : ℕ,
  n > 3 ∧
  ∃ x y z r s t : ℕ,
    63 = n * x + r ∧
    91 = n * y + s ∧
    130 = n * z + t ∧
    r + s + t = 26 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainder_sum_l3000_300011


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3000_300063

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 13*x + 40 = 0 →
  3 + 4 + x > x ∧ 3 + x > 4 ∧ 4 + x > 3 →
  3 + 4 + x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3000_300063


namespace NUMINAMATH_CALUDE_triangle_circumcircle_point_length_l3000_300045

/-- Triangle PQR with sides PQ = 39, QR = 52, PR = 25 -/
structure Triangle :=
  (PQ QR PR : ℝ)
  (PQ_pos : PQ > 0)
  (QR_pos : QR > 0)
  (PR_pos : PR > 0)

/-- S is a point on the circumcircle of triangle PQR -/
structure CircumcirclePoint (t : Triangle) :=
  (S : ℝ × ℝ)

/-- S is on the perpendicular bisector of PR, not on the same side as Q -/
def onPerpendicularBisector (t : Triangle) (p : CircumcirclePoint t) : Prop := sorry

/-- The length of PS can be expressed as a√b where a and b are positive integers -/
def PSLength (t : Triangle) (p : CircumcirclePoint t) : ℕ × ℕ := sorry

/-- b is not divisible by the square of any prime -/
def notDivisibleBySquare : ℕ → Prop := sorry

theorem triangle_circumcircle_point_length 
  (t : Triangle) 
  (h1 : t.PQ = 39 ∧ t.QR = 52 ∧ t.PR = 25) 
  (p : CircumcirclePoint t) 
  (h2 : onPerpendicularBisector t p) 
  (h3 : let (a, b) := PSLength t p; notDivisibleBySquare b) : 
  let (a, b) := PSLength t p
  (a : ℕ) + Real.sqrt b = 54 := by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_point_length_l3000_300045


namespace NUMINAMATH_CALUDE_tetrahedron_acute_angle_vertex_l3000_300060

/-- A tetrahedron is represented by its four vertices in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- The plane angle at a vertex of a tetrahedron -/
def planeAngle (t : Tetrahedron) (v : Fin 4) (e1 e2 : Fin 4) : ℝ :=
  sorry

/-- Theorem: In any tetrahedron, there exists at least one vertex where all plane angles are acute -/
theorem tetrahedron_acute_angle_vertex (t : Tetrahedron) : 
  ∃ v : Fin 4, ∀ e1 e2 : Fin 4, e1 ≠ e2 → e1 ≠ v → e2 ≠ v → planeAngle t v e1 e2 < π / 2 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_acute_angle_vertex_l3000_300060


namespace NUMINAMATH_CALUDE_peach_crate_pigeonhole_l3000_300073

/-- The number of crates of peaches -/
def total_crates : ℕ := 154

/-- The minimum number of peaches in a crate -/
def min_peaches : ℕ := 130

/-- The maximum number of peaches in a crate -/
def max_peaches : ℕ := 160

/-- The number of possible peach counts per crate -/
def possible_counts : ℕ := max_peaches - min_peaches + 1

theorem peach_crate_pigeonhole :
  ∃ (n : ℕ), n = 4 ∧
  (∀ (m : ℕ), m > n →
    ∃ (distribution : Fin total_crates → ℕ),
      (∀ i, min_peaches ≤ distribution i ∧ distribution i ≤ max_peaches) ∧
      (∀ k, ¬(∃ (S : Finset (Fin total_crates)), S.card = m ∧ (∀ i ∈ S, distribution i = k)))) ∧
  (∃ (distribution : Fin total_crates → ℕ),
    (∀ i, min_peaches ≤ distribution i ∧ distribution i ≤ max_peaches) →
    ∃ (k : ℕ) (S : Finset (Fin total_crates)), S.card = n ∧ (∀ i ∈ S, distribution i = k)) := by
  sorry


end NUMINAMATH_CALUDE_peach_crate_pigeonhole_l3000_300073


namespace NUMINAMATH_CALUDE_share_distribution_theorem_l3000_300014

/-- Represents the share distribution problem among three children -/
def ShareDistribution (anusha_share babu_share esha_share k : ℚ) : Prop :=
  -- Total amount is 378
  anusha_share + babu_share + esha_share = 378 ∧
  -- Anusha's share is 84
  anusha_share = 84 ∧
  -- 12 times Anusha's share equals k times Babu's share
  12 * anusha_share = k * babu_share ∧
  -- k times Babu's share equals 6 times Esha's share
  k * babu_share = 6 * esha_share

/-- The main theorem stating that given the conditions, k equals 4 -/
theorem share_distribution_theorem :
  ∀ (anusha_share babu_share esha_share k : ℚ),
  ShareDistribution anusha_share babu_share esha_share k →
  k = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_share_distribution_theorem_l3000_300014


namespace NUMINAMATH_CALUDE_custom_mul_four_three_l3000_300077

/-- Custom multiplication operation -/
def customMul (a b : ℕ) : ℕ := a^2 + a * Nat.factorial b - b^2

/-- Theorem stating that 4 * 3 = 31 under the custom multiplication -/
theorem custom_mul_four_three : customMul 4 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_four_three_l3000_300077


namespace NUMINAMATH_CALUDE_evaluate_expression_l3000_300066

theorem evaluate_expression : (900^2 : ℝ) / (153^2 - 147^2) = 450 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3000_300066


namespace NUMINAMATH_CALUDE_no_triangle_with_given_conditions_l3000_300037

theorem no_triangle_with_given_conditions (a b c : ℕ+) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (gcd_one : Nat.gcd a.val (Nat.gcd b.val c.val) = 1)
  (div_a : a.val ∣ (b.val - c.val)^2)
  (div_b : b.val ∣ (c.val - a.val)^2)
  (div_c : c.val ∣ (a.val - b.val)^2) :
  ¬(a.val < b.val + c.val ∧ b.val < a.val + c.val ∧ c.val < a.val + b.val) :=
sorry

end NUMINAMATH_CALUDE_no_triangle_with_given_conditions_l3000_300037


namespace NUMINAMATH_CALUDE_product_equals_3408_l3000_300018

theorem product_equals_3408 : 213 * 16 = 3408 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_3408_l3000_300018


namespace NUMINAMATH_CALUDE_inequality_proof_l3000_300035

theorem inequality_proof (x : ℝ) (h : x ≥ 1) : x^5 - 1/x^4 ≥ 9*(x-1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3000_300035


namespace NUMINAMATH_CALUDE_max_value_implies_a_l3000_300084

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = -3 ∨ a = 3/8 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l3000_300084


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l3000_300053

/-- Given a triangle with sides a, b, and x, where a > b, 
    prove that the perimeter m satisfies 2a < m < 2(a+b) -/
theorem triangle_perimeter_range 
  (a b x : ℝ) 
  (h1 : a > b) 
  (h2 : a - b < x) 
  (h3 : x < a + b) : 
  2 * a < a + b + x ∧ a + b + x < 2 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l3000_300053


namespace NUMINAMATH_CALUDE_helen_raisin_cookies_l3000_300074

/-- Represents the number of cookies baked --/
structure CookieCount where
  yesterday_chocolate : ℕ
  yesterday_raisin : ℕ
  today_chocolate : ℕ
  today_raisin : ℕ
  total_chocolate : ℕ

/-- Helen's cookie baking scenario --/
def helen_cookies : CookieCount where
  yesterday_chocolate := 527
  yesterday_raisin := 527  -- This is what we want to prove
  today_chocolate := 554
  today_raisin := 554
  total_chocolate := 1081

/-- Theorem stating that Helen baked 527 raisin cookies yesterday --/
theorem helen_raisin_cookies : 
  helen_cookies.yesterday_raisin = 527 := by
  sorry

#check helen_raisin_cookies

end NUMINAMATH_CALUDE_helen_raisin_cookies_l3000_300074


namespace NUMINAMATH_CALUDE_largest_difference_theorem_l3000_300008

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_constraints (a b c d e f g h i : ℕ) : Prop :=
  a ∈ ({3, 5, 9} : Set ℕ) ∧
  b ∈ ({2, 3, 7} : Set ℕ) ∧
  c ∈ ({3, 4, 8, 9} : Set ℕ) ∧
  d ∈ ({2, 3, 7} : Set ℕ) ∧
  e ∈ ({3, 5, 9} : Set ℕ) ∧
  f ∈ ({1, 4, 7} : Set ℕ) ∧
  g ∈ ({4, 5, 9} : Set ℕ) ∧
  h = 2 ∧
  i ∈ ({4, 5, 9} : Set ℕ)

def number_from_digits (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem largest_difference_theorem (a b c d e f g h i : ℕ) :
  digit_constraints a b c d e f g h i →
  is_three_digit (number_from_digits a b c) →
  is_three_digit (number_from_digits d e f) →
  is_three_digit (number_from_digits g h i) →
  number_from_digits a b c - number_from_digits d e f = number_from_digits g h i →
  ∀ (x y z u v w : ℕ),
    digit_constraints x y z u v w g h i →
    is_three_digit (number_from_digits x y z) →
    is_three_digit (number_from_digits u v w) →
    number_from_digits x y z - number_from_digits u v w = number_from_digits g h i →
    number_from_digits g h i ≤ 529 →
  (a = 9 ∧ b = 2 ∧ c = 3 ∧ d = 3 ∧ e = 9 ∧ f = 4) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_theorem_l3000_300008


namespace NUMINAMATH_CALUDE_range_of_m_l3000_300006

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  monotonically_decreasing f →
  f (m + 1) < f (3 - 2 * m) →
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), m = -(Real.sin x)^2 - 2 * Real.sin x + 1) →
  m ∈ Set.Ioo (2/3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3000_300006


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3000_300005

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3000_300005


namespace NUMINAMATH_CALUDE_apples_in_box_l3000_300046

/-- The number of boxes containing apples -/
def num_boxes : ℕ := 5

/-- The number of apples removed from each box -/
def apples_removed : ℕ := 60

/-- The number of apples initially in each box -/
def apples_per_box : ℕ := 100

theorem apples_in_box : 
  (num_boxes * apples_per_box) - (num_boxes * apples_removed) = 2 * apples_per_box := by
  sorry

end NUMINAMATH_CALUDE_apples_in_box_l3000_300046


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l3000_300013

def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 60) : 
  min_additional_coins num_friends initial_coins = 60 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l3000_300013


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3000_300038

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def majorAxisLength (cylinderRadius : ℝ) (majorMinorRatio : ℝ) : ℝ :=
  2 * cylinderRadius * majorMinorRatio

/-- Theorem: The length of the major axis of the ellipse is 12 -/
theorem ellipse_major_axis_length :
  majorAxisLength 3 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3000_300038


namespace NUMINAMATH_CALUDE_largest_B_181_l3000_300095

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sequence B_k as defined in the problem -/
def B (k : ℕ) : ℝ := (binomial 2000 k : ℝ) * (0.1 ^ k)

/-- Theorem stating that B_181 is the largest among all B_k -/
theorem largest_B_181 : ∀ k : ℕ, k ≤ 2000 → B 181 ≥ B k := by sorry

end NUMINAMATH_CALUDE_largest_B_181_l3000_300095


namespace NUMINAMATH_CALUDE_handshake_count_l3000_300025

theorem handshake_count (n : ℕ) (total_handshakes : ℕ) : 
  n = 7 ∧ total_handshakes = n * (n - 1) / 2 → total_handshakes = 21 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l3000_300025


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3000_300075

theorem inequality_solution_set (x : ℝ) : 
  (1/2 - x) * (x - 1/3) > 0 ↔ 1/3 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3000_300075


namespace NUMINAMATH_CALUDE_fifth_group_size_l3000_300099

/-- Represents a choir split into groups -/
structure Choir :=
  (total_members : ℕ)
  (group1 : ℕ)
  (group2 : ℕ)
  (group3 : ℕ)
  (group4 : ℕ)
  (group5 : ℕ)

/-- The choir satisfies the given conditions -/
def choir_conditions (c : Choir) : Prop :=
  c.total_members = 150 ∧
  c.group1 = 18 ∧
  c.group2 = 29 ∧
  c.group3 = 34 ∧
  c.group4 = 23 ∧
  c.total_members = c.group1 + c.group2 + c.group3 + c.group4 + c.group5

/-- Theorem: The fifth group has 46 members -/
theorem fifth_group_size (c : Choir) (h : choir_conditions c) : c.group5 = 46 := by
  sorry

end NUMINAMATH_CALUDE_fifth_group_size_l3000_300099


namespace NUMINAMATH_CALUDE_arithmetic_sequence_l3000_300028

def a (n : ℕ) : ℤ := 3 * n + 1

theorem arithmetic_sequence :
  ∀ n : ℕ, a (n + 1) - a n = (3 : ℤ) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_l3000_300028


namespace NUMINAMATH_CALUDE_quarters_percentage_is_65_22_l3000_300032

/-- The number of dimes -/
def num_dimes : ℕ := 40

/-- The number of quarters -/
def num_quarters : ℕ := 30

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of all coins in cents -/
def total_value : ℕ := num_dimes * dime_value + num_quarters * quarter_value

/-- The value of all quarters in cents -/
def quarters_value : ℕ := num_quarters * quarter_value

/-- The percentage of the total value that is in quarters -/
def quarters_percentage : ℚ := (quarters_value : ℚ) / (total_value : ℚ) * 100

theorem quarters_percentage_is_65_22 : 
  ∀ ε > 0, |quarters_percentage - 65.22| < ε :=
sorry

end NUMINAMATH_CALUDE_quarters_percentage_is_65_22_l3000_300032


namespace NUMINAMATH_CALUDE_max_x_2009_l3000_300047

def sequence_property (x : ℕ → ℝ) :=
  ∀ n, x n - 2 * x (n + 1) + x (n + 2) ≤ 0

theorem max_x_2009 (x : ℕ → ℝ) 
  (h : sequence_property x)
  (h0 : x 0 = 1)
  (h20 : x 20 = 9)
  (h200 : x 200 = 6) :
  x 2009 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_x_2009_l3000_300047


namespace NUMINAMATH_CALUDE_order_of_abc_l3000_300072

theorem order_of_abc : ∀ (a b c : ℝ), 
  a = 2^(1/10) → 
  b = Real.log (1/2) → 
  c = (2/3)^Real.pi → 
  a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l3000_300072


namespace NUMINAMATH_CALUDE_cube_monotonically_increasing_l3000_300017

/-- A function f: ℝ → ℝ is monotonically increasing if for all x₁ < x₂, f(x₁) ≤ f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

/-- The cube function -/
def cube (x : ℝ) : ℝ := x^3

/-- The cube function is monotonically increasing on ℝ -/
theorem cube_monotonically_increasing : MonotonicallyIncreasing cube := by
  sorry


end NUMINAMATH_CALUDE_cube_monotonically_increasing_l3000_300017


namespace NUMINAMATH_CALUDE_refrigerator_price_correct_l3000_300036

/-- The purchase price of a refrigerator, given specific conditions on its sale and the sale of a mobile phone --/
def refrigerator_price : ℝ :=
  let mobile_price : ℝ := 8000
  let refrigerator_loss_percent : ℝ := 0.04
  let mobile_profit_percent : ℝ := 0.11
  let total_profit : ℝ := 280
  
  -- Define the equation to solve
  let equation (x : ℝ) : Prop :=
    (mobile_price * (1 + mobile_profit_percent) - mobile_price) - 
    (x - x * (1 - refrigerator_loss_percent)) = total_profit
  
  -- The solution (to be proved)
  15000

theorem refrigerator_price_correct : 
  refrigerator_price = 15000 := by sorry

end NUMINAMATH_CALUDE_refrigerator_price_correct_l3000_300036
