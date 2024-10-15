import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1054_105454

/-- Given a line L with equation 3x - 6y = 9 and a point P(-2, 3), 
    the line perpendicular to L passing through P has equation y = -2x - 1 -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L : Set (ℝ × ℝ) := {(x, y) | 3 * x - 6 * y = 9}
  let P : ℝ × ℝ := (-2, 3)
  let m : ℝ := 1/2  -- slope of the original line
  let m_perp : ℝ := -1/m  -- slope of the perpendicular line
  let perp_line : Set (ℝ × ℝ) := {(x, y) | y = m_perp * x + (P.2 - m_perp * P.1)}
  perp_line = {(x, y) | y = -2 * x - 1} := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l1054_105454


namespace NUMINAMATH_CALUDE_largest_when_third_digit_changed_l1054_105437

def original_number : ℚ := 0.08765

def change_third_digit : ℚ := 0.08865
def change_fourth_digit : ℚ := 0.08785
def change_fifth_digit : ℚ := 0.08768

theorem largest_when_third_digit_changed :
  change_third_digit > change_fourth_digit ∧
  change_third_digit > change_fifth_digit :=
by sorry

end NUMINAMATH_CALUDE_largest_when_third_digit_changed_l1054_105437


namespace NUMINAMATH_CALUDE_area_ratio_ACEG_to_hexadecagon_l1054_105448

/-- Regular hexadecagon with vertices ABCDEFGHIJKLMNOP -/
structure RegularHexadecagon where
  vertices : Fin 16 → ℝ × ℝ
  is_regular : sorry -- Additional properties to ensure it's a regular hexadecagon

/-- Area of a regular hexadecagon -/
def area_hexadecagon (h : RegularHexadecagon) : ℝ := sorry

/-- Quadrilateral ACEG formed by connecting every fourth vertex of the hexadecagon -/
def quadrilateral_ACEG (h : RegularHexadecagon) : Set (ℝ × ℝ) := sorry

/-- Area of quadrilateral ACEG -/
def area_ACEG (h : RegularHexadecagon) : ℝ := sorry

/-- The main theorem: The ratio of the area of ACEG to the area of the hexadecagon is √2/2 -/
theorem area_ratio_ACEG_to_hexadecagon (h : RegularHexadecagon) :
  (area_ACEG h) / (area_hexadecagon h) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_ACEG_to_hexadecagon_l1054_105448


namespace NUMINAMATH_CALUDE_actual_plot_area_l1054_105495

/-- Represents the scale of the map --/
def scale : ℝ := 3

/-- Represents the conversion factor from square miles to acres --/
def sq_mile_to_acre : ℝ := 640

/-- Represents the length of the rectangle on the map in cm --/
def map_length : ℝ := 20

/-- Represents the width of the rectangle on the map in cm --/
def map_width : ℝ := 12

/-- Theorem stating that the area of the actual plot is 1,382,400 acres --/
theorem actual_plot_area :
  (map_length * scale) * (map_width * scale) * sq_mile_to_acre = 1382400 := by
  sorry

end NUMINAMATH_CALUDE_actual_plot_area_l1054_105495


namespace NUMINAMATH_CALUDE_choir_arrangement_l1054_105434

theorem choir_arrangement (n : ℕ) : n ≥ 32400 ∧ 
  (∃ k : ℕ, n = k^2) ∧ 
  n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 →
  n = 32400 :=
sorry

end NUMINAMATH_CALUDE_choir_arrangement_l1054_105434


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l1054_105431

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation_in_one_variable (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 3 -/
def equation (x : ℝ) : ℝ := x^2 - 3

theorem equation_is_quadratic : is_quadratic_equation_in_one_variable equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l1054_105431


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l1054_105490

theorem probability_triangle_or_circle (total : ℕ) (triangles : ℕ) (circles : ℕ) 
  (h1 : total = 10) 
  (h2 : triangles = 4) 
  (h3 : circles = 4) : 
  (triangles + circles : ℚ) / total = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l1054_105490


namespace NUMINAMATH_CALUDE_teacher_weight_l1054_105422

theorem teacher_weight (num_students : ℕ) (avg_weight : ℝ) (weight_increase : ℝ) : 
  num_students = 24 →
  avg_weight = 35 →
  weight_increase = 0.4 →
  (num_students * avg_weight + (avg_weight + weight_increase) * (num_students + 1)) / (num_students + 1) - avg_weight = weight_increase →
  (num_students + 1) * (avg_weight + weight_increase) - num_students * avg_weight = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_teacher_weight_l1054_105422


namespace NUMINAMATH_CALUDE_line_through_two_points_l1054_105499

/-- 
Given a line with equation x = 8y + 5 that passes through points (m, n) and (m + 2, n + p),
prove that p = 1/4.
-/
theorem line_through_two_points (m n p : ℝ) : 
  (m = 8 * n + 5) ∧ (m + 2 = 8 * (n + p) + 5) → p = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l1054_105499


namespace NUMINAMATH_CALUDE_fib_2006_mod_10_l1054_105449

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_2006_mod_10 : fib 2006 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fib_2006_mod_10_l1054_105449


namespace NUMINAMATH_CALUDE_female_emu_ratio_is_half_l1054_105433

/-- Represents the emu farm setup and egg production --/
structure EmuFarm where
  num_pens : ℕ
  emus_per_pen : ℕ
  eggs_per_week : ℕ

/-- Calculates the ratio of female emus to total emus --/
def female_emu_ratio (farm : EmuFarm) : ℚ :=
  let total_emus := farm.num_pens * farm.emus_per_pen
  let eggs_per_day := farm.eggs_per_week / 7
  eggs_per_day / total_emus

/-- Theorem stating that the ratio of female emus to total emus is 1/2 --/
theorem female_emu_ratio_is_half (farm : EmuFarm) 
    (h1 : farm.num_pens = 4)
    (h2 : farm.emus_per_pen = 6)
    (h3 : farm.eggs_per_week = 84) : 
  female_emu_ratio farm = 1/2 := by
  sorry

#eval female_emu_ratio ⟨4, 6, 84⟩

end NUMINAMATH_CALUDE_female_emu_ratio_is_half_l1054_105433


namespace NUMINAMATH_CALUDE_sum_equals_3000_length_conversion_l1054_105463

-- Problem 1
theorem sum_equals_3000 : 1361 + 972 + 639 + 28 = 3000 := by sorry

-- Problem 2
theorem length_conversion :
  ∀ (meters decimeters centimeters : ℕ),
    meters * 10 + decimeters - (centimeters / 10) = 91 →
    9 * 10 + 9 - (80 / 10) = 91 := by sorry

end NUMINAMATH_CALUDE_sum_equals_3000_length_conversion_l1054_105463


namespace NUMINAMATH_CALUDE_at_least_one_less_than_one_l1054_105462

theorem at_least_one_less_than_one (a b c : ℝ) (ha : a < 3) (hb : b < 3) (hc : c < 3) :
  min a (min b c) < 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_one_l1054_105462


namespace NUMINAMATH_CALUDE_log_difference_equals_negative_nine_l1054_105484

theorem log_difference_equals_negative_nine :
  (Real.log 243 / Real.log 3) / (Real.log 27 / Real.log 3) -
  (Real.log 729 / Real.log 3) / (Real.log 81 / Real.log 3) = -9 := by
sorry

end NUMINAMATH_CALUDE_log_difference_equals_negative_nine_l1054_105484


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_perfect_square_digit_sums_l1054_105472

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Theorem: There exist two consecutive natural numbers greater than 1,000,000 
    whose sums of digits are perfect squares -/
theorem consecutive_numbers_with_perfect_square_digit_sums : 
  ∃ n : ℕ, n > 1000000 ∧ 
    is_perfect_square (sum_of_digits n) ∧ 
    is_perfect_square (sum_of_digits (n + 1)) := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_with_perfect_square_digit_sums_l1054_105472


namespace NUMINAMATH_CALUDE_negation_equivalence_l1054_105473

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x ≤ 1 ∨ x^2 > 4) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1054_105473


namespace NUMINAMATH_CALUDE_toms_tickets_toms_remaining_tickets_l1054_105405

/-- Tom's arcade tickets problem -/
theorem toms_tickets (whack_a_mole : ℕ) (skee_ball : ℕ) (spent : ℕ) : ℕ :=
  let total := whack_a_mole + skee_ball
  total - spent

/-- Proof of Tom's remaining tickets -/
theorem toms_remaining_tickets : toms_tickets 32 25 7 = 50 := by
  sorry

end NUMINAMATH_CALUDE_toms_tickets_toms_remaining_tickets_l1054_105405


namespace NUMINAMATH_CALUDE_difference_of_squares_255_745_l1054_105413

theorem difference_of_squares_255_745 : 255^2 - 745^2 = -490000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_255_745_l1054_105413


namespace NUMINAMATH_CALUDE_bookshelf_picking_l1054_105408

theorem bookshelf_picking (english_books math_books : ℕ) 
  (h1 : english_books = 6) 
  (h2 : math_books = 2) : 
  english_books + math_books = 8 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_picking_l1054_105408


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l1054_105488

/-- Two different lines in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem plane_perpendicularity (m n : Line3D) (α β : Plane3D) 
  (h1 : m ≠ n) (h2 : α ≠ β) (h3 : perpendicular m α) (h4 : parallel m β) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l1054_105488


namespace NUMINAMATH_CALUDE_not_on_line_l1054_105446

-- Define the quadratic function f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g(x)
def g (a b c x x_1 x_2 : ℝ) : ℝ := f a b c (x - x_1) + f a b c (x - x_2)

theorem not_on_line (a b c x_1 x_2 : ℝ) 
  (h1 : ∃ x_1 x_2, f a b c x_1 = 0 ∧ f a b c x_2 = 0) -- f has two zeros
  (h2 : f a b c 1 = 2 * a) -- f(1) = 2a
  (h3 : a > c) -- a > c
  (h4 : ∀ x ∈ Set.Icc 0 1, g a b c x x_1 x_2 ≤ 2 / a) -- max of g(x) in [0,1] is 2/a
  (h5 : ∃ x ∈ Set.Icc 0 1, g a b c x x_1 x_2 = 2 / a) -- max of g(x) in [0,1] is achieved
  : a + b ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_not_on_line_l1054_105446


namespace NUMINAMATH_CALUDE_improved_representation_of_100_l1054_105439

theorem improved_representation_of_100 :
  (222 / 2 : ℚ) - (22 / 2 : ℚ) = 100 := by sorry

end NUMINAMATH_CALUDE_improved_representation_of_100_l1054_105439


namespace NUMINAMATH_CALUDE_fraction_of_y_l1054_105467

theorem fraction_of_y (y : ℝ) (h : y > 0) : (9 * y / 20 + 3 * y / 10) / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_y_l1054_105467


namespace NUMINAMATH_CALUDE_constant_value_l1054_105481

/-- A function satisfying the given conditions -/
def f (c : ℝ) : ℝ → ℝ :=
  fun x ↦ sorry

/-- The theorem stating the problem conditions and conclusion -/
theorem constant_value (c : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x + 3 * f (c - x) = x) →
  f 2 = 2 →
  c = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l1054_105481


namespace NUMINAMATH_CALUDE_inequality_solution_l1054_105428

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1054_105428


namespace NUMINAMATH_CALUDE_fourth_sample_id_l1054_105402

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_ids : List ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- Checks if a given ID is part of the sample -/
def is_in_sample (s : SystematicSampling) (id : ℕ) : Prop :=
  ∃ k : ℕ, id = s.known_ids.head! + k * sampling_interval s

/-- The main theorem to prove -/
theorem fourth_sample_id (s : SystematicSampling)
  (h1 : s.total_students = 44)
  (h2 : s.sample_size = 4)
  (h3 : s.known_ids = [6, 28, 39]) :
  is_in_sample s 17 := by
  sorry

#check fourth_sample_id

end NUMINAMATH_CALUDE_fourth_sample_id_l1054_105402


namespace NUMINAMATH_CALUDE_division_problem_l1054_105400

theorem division_problem : (-1) / (-5) / (-1/5) = -1 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1054_105400


namespace NUMINAMATH_CALUDE_notebook_cost_example_l1054_105478

/-- The cost of notebooks given the number of notebooks, pages per notebook, and cost per page. -/
def notebook_cost (num_notebooks : ℕ) (pages_per_notebook : ℕ) (cost_per_page : ℚ) : ℚ :=
  (num_notebooks * pages_per_notebook : ℚ) * cost_per_page

/-- Theorem stating that the cost of 2 notebooks with 50 pages each, at 5 cents per page, is $5.00 -/
theorem notebook_cost_example : notebook_cost 2 50 (5 / 100) = 5 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_example_l1054_105478


namespace NUMINAMATH_CALUDE_keaton_ladder_climbs_l1054_105483

/-- Proves that Keaton climbed the ladder 20 times given the problem conditions -/
theorem keaton_ladder_climbs : 
  let keaton_ladder_height : ℕ := 30 * 12  -- 30 feet in inches
  let reece_ladder_height : ℕ := (30 - 4) * 12  -- 26 feet in inches
  let reece_climbs : ℕ := 15
  let total_length : ℕ := 11880  -- in inches
  ∃ (keaton_climbs : ℕ), 
    keaton_climbs * keaton_ladder_height + reece_climbs * reece_ladder_height = total_length ∧ 
    keaton_climbs = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_keaton_ladder_climbs_l1054_105483


namespace NUMINAMATH_CALUDE_fraction_product_equals_seven_fifty_fourths_l1054_105423

theorem fraction_product_equals_seven_fifty_fourths : 
  (7 : ℚ) / 4 * 8 / 12 * 14 / 6 * 18 / 30 * 16 / 24 * 35 / 49 * 27 / 54 * 40 / 20 = 7 / 54 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_seven_fifty_fourths_l1054_105423


namespace NUMINAMATH_CALUDE_parallelogram_network_l1054_105404

theorem parallelogram_network (first_set : ℕ) (total_parallelograms : ℕ) 
  (h1 : first_set = 8) 
  (h2 : total_parallelograms = 784) : 
  ∃ (second_set : ℕ), 
    second_set > 0 ∧ 
    (first_set - 1) * (second_set - 1) = total_parallelograms := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_network_l1054_105404


namespace NUMINAMATH_CALUDE_main_theorem_l1054_105406

/-- The logarithm function with base 2 -/
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

/-- The main function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log2 (x + a)

/-- The companion function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * log2 (4*x + a)

/-- The difference function F(x) -/
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

theorem main_theorem (a : ℝ) (h : a > 0) :
  (∀ x, f a x < -1 ↔ -a < x ∧ x < 1/2 - a) ∧
  (∀ x ∈ Set.Ioo 0 2, f a x < g a x ↔ 0 < a ∧ a ≤ 1) ∧
  (∃ M, M = 1 - (1/2) * log2 3 ∧ 
    ∀ x ∈ Set.Ioo 0 2, |F 1 x| ≤ M ∧
    ∃ x₀ ∈ Set.Ioo 0 2, |F 1 x₀| = M) :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l1054_105406


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1054_105444

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + 2*k*y + 4*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 2*z = 0 →
  x*z / (y^2) = 10 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1054_105444


namespace NUMINAMATH_CALUDE_choco_given_away_l1054_105426

/-- Represents the number of cookies in a dozen. -/
def dozen : ℕ := 12

/-- Represents the number of dozens of oatmeal raisin cookies baked. -/
def oatmeal_baked : ℚ := 3

/-- Represents the number of dozens of sugar cookies baked. -/
def sugar_baked : ℚ := 2

/-- Represents the number of dozens of chocolate chip cookies baked. -/
def choco_baked : ℚ := 4

/-- Represents the number of dozens of oatmeal raisin cookies given away. -/
def oatmeal_given : ℚ := 2

/-- Represents the number of dozens of sugar cookies given away. -/
def sugar_given : ℚ := 3/2

/-- Represents the total number of cookies Ann keeps. -/
def cookies_kept : ℕ := 36

/-- Theorem stating the number of dozens of chocolate chip cookies given away. -/
theorem choco_given_away : 
  (oatmeal_baked * dozen + sugar_baked * dozen + choco_baked * dozen - 
   oatmeal_given * dozen - sugar_given * dozen - cookies_kept) / dozen = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_choco_given_away_l1054_105426


namespace NUMINAMATH_CALUDE_original_square_side_length_l1054_105470

def is_valid_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ (n + k)^2 - n^2 = 47

theorem original_square_side_length :
  ∃! (n : ℕ), is_valid_square n ∧ n > 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_original_square_side_length_l1054_105470


namespace NUMINAMATH_CALUDE_sin_210_degrees_l1054_105491

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l1054_105491


namespace NUMINAMATH_CALUDE_log_inequality_l1054_105427

/-- Given a = log_3(2), b = log_2(3), and c = log_(1/2)(5), prove that c < a < b -/
theorem log_inequality (a b c : ℝ) 
  (ha : a = Real.log 2 / Real.log 3)
  (hb : b = Real.log 3 / Real.log 2)
  (hc : c = Real.log 5 / Real.log (1/2)) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1054_105427


namespace NUMINAMATH_CALUDE_product_sum_fractions_l1054_105496

theorem product_sum_fractions : (3 * 4 * 5 * 6) * (1/3 + 1/4 + 1/5 + 1/6) = 342 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l1054_105496


namespace NUMINAMATH_CALUDE_f_properties_l1054_105456

/-- The function f(x) defined as 2 / (2^x + 1) + m -/
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 / (2^x + 1) + m

/-- f is an odd function -/
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- f is decreasing on ℝ -/
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

theorem f_properties (m : ℝ) :
  (is_odd (f · m) → m = -1) ∧
  is_decreasing (f · m) ∧
  (∀ x ≤ 1, f x m ≥ f 1 m) ∧
  f (-1) m = 4/3 + m :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1054_105456


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1054_105445

theorem complex_number_modulus : 
  let z : ℂ := 2 / (1 + Complex.I) + (1 - Complex.I)^2
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1054_105445


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1054_105452

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1054_105452


namespace NUMINAMATH_CALUDE_xiao_ming_reading_progress_l1054_105450

/-- Calculates the starting page for the 6th day of reading -/
def starting_page_6th_day (total_pages book_pages_per_day days_read : ℕ) : ℕ :=
  book_pages_per_day * days_read + 1

/-- Proves that the starting page for the 6th day is 301 -/
theorem xiao_ming_reading_progress : starting_page_6th_day 500 60 5 = 301 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_reading_progress_l1054_105450


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l1054_105480

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of x^r in the expansion of (x + 1/(2x))^n -/
def coeff (n r : ℕ) : ℚ := (binomial n r) * (1 / 2 ^ r)

theorem binomial_expansion_properties (n : ℕ) (h : n ≥ 2) :
  (2 * coeff n 1 = coeff n 0 + coeff n 2 ↔ n = 8) ∧
  (n = 8 → coeff n 2 = 7) := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l1054_105480


namespace NUMINAMATH_CALUDE_equal_area_triangle_octagon_ratio_l1054_105420

/-- The ratio of side lengths of an equilateral triangle and a regular octagon with equal areas -/
theorem equal_area_triangle_octagon_ratio :
  ∀ (s_t s_o : ℝ),
  s_t > 0 → s_o > 0 →
  (s_t^2 * Real.sqrt 3) / 4 = 2 * s_o^2 * (1 + Real.sqrt 2) →
  s_t / s_o = Real.sqrt (8 * Real.sqrt 3 * (1 + Real.sqrt 2) / 3) :=
by sorry


end NUMINAMATH_CALUDE_equal_area_triangle_octagon_ratio_l1054_105420


namespace NUMINAMATH_CALUDE_rex_cards_left_is_150_l1054_105453

/-- The number of Pokemon cards collected by Nicole -/
def nicole_cards : ℕ := 400

/-- The number of Pokemon cards collected by Cindy -/
def cindy_cards : ℕ := 2 * nicole_cards

/-- The combined total of Nicole and Cindy's cards -/
def combined_total : ℕ := nicole_cards + cindy_cards

/-- The number of Pokemon cards collected by Rex -/
def rex_cards : ℕ := combined_total / 2

/-- The number of people Rex divides his cards among (including himself) -/
def number_of_people : ℕ := 4

/-- The number of cards Rex has left after dividing his cards equally -/
def rex_cards_left : ℕ := rex_cards / number_of_people

theorem rex_cards_left_is_150 : rex_cards_left = 150 := by sorry

end NUMINAMATH_CALUDE_rex_cards_left_is_150_l1054_105453


namespace NUMINAMATH_CALUDE_yellow_purple_difference_l1054_105464

/-- Represents the composition of candies in a box of rainbow nerds -/
structure RainbowNerdsBox where
  purple : ℕ
  yellow : ℕ
  green : ℕ
  total : ℕ
  green_yellow_relation : green = yellow - 2
  total_sum : total = purple + yellow + green

/-- Theorem stating the difference between yellow and purple candies -/
theorem yellow_purple_difference (box : RainbowNerdsBox) 
  (h_purple : box.purple = 10) 
  (h_total : box.total = 36) : 
  box.yellow - box.purple = 4 := by
  sorry


end NUMINAMATH_CALUDE_yellow_purple_difference_l1054_105464


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1054_105494

theorem rectangle_perimeter (width length : ℝ) (h1 : width = Real.sqrt 3) (h2 : length = Real.sqrt 6) :
  2 * (width + length) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1054_105494


namespace NUMINAMATH_CALUDE_marble_arrangement_count_l1054_105412

/-- The number of ways to arrange 7 red marbles and n blue marbles in a row,
    where n is the maximum number of blue marbles that can be arranged such that
    the number of adjacent same-color pairs equals the number of adjacent different-color pairs -/
def M : ℕ := sorry

/-- The maximum number of blue marbles that can be arranged with 7 red marbles
    such that the number of adjacent same-color pairs equals the number of adjacent different-color pairs -/
def n : ℕ := sorry

/-- The theorem stating that M modulo 1000 equals 716 -/
theorem marble_arrangement_count : M % 1000 = 716 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_count_l1054_105412


namespace NUMINAMATH_CALUDE_water_level_drop_l1054_105479

/-- The water level drop in a cylindrical container when two spheres are removed -/
theorem water_level_drop (container_radius : ℝ) (sphere_diameter : ℝ) : 
  container_radius = 5 →
  sphere_diameter = 5 →
  (π * container_radius^2 * (5/3)) = (2 * (4/3) * π * (sphere_diameter/2)^3) :=
by sorry

end NUMINAMATH_CALUDE_water_level_drop_l1054_105479


namespace NUMINAMATH_CALUDE_tank_filling_time_l1054_105466

def pipe1_rate : ℚ := 1 / 8
def pipe2_rate : ℚ := 1 / 12

def combined_rate : ℚ := pipe1_rate + pipe2_rate

theorem tank_filling_time : (1 : ℚ) / combined_rate = 24 / 5 := by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l1054_105466


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_equals_3_subset_of_complement_iff_m_in_range_l1054_105487

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < m + 2}
def B : Set ℝ := {x | -4 < x ∧ x < 4}

-- Theorem for part (I)
theorem intersection_and_union_when_m_equals_3 :
  (A 3 ∩ B = {x | 1 < x ∧ x < 4}) ∧
  (A 3 ∪ B = {x | -4 < x ∧ x < 5}) := by
  sorry

-- Theorem for part (II)
theorem subset_of_complement_iff_m_in_range :
  ∀ m : ℝ, A m ⊆ Bᶜ ↔ m ≤ -6 ∨ m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_equals_3_subset_of_complement_iff_m_in_range_l1054_105487


namespace NUMINAMATH_CALUDE_balls_in_boxes_count_l1054_105498

/-- The number of ways to place 4 different balls into 4 numbered boxes with exactly one empty box -/
def placeBallsInBoxes : ℕ :=
  -- We define this as a natural number, but don't provide the implementation
  sorry

/-- The theorem stating that there are 144 ways to place 4 different balls into 4 numbered boxes
    such that exactly one box is empty -/
theorem balls_in_boxes_count : placeBallsInBoxes = 144 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_count_l1054_105498


namespace NUMINAMATH_CALUDE_perpendicular_vectors_condition_l1054_105497

/-- Given two vectors in R², prove that if they satisfy certain conditions, then a specific component of one vector equals -1. -/
theorem perpendicular_vectors_condition (m : ℝ) : 
  let a : Fin 2 → ℝ := ![m, 3]
  let b : Fin 2 → ℝ := ![-2, 2]
  (∀ i, (a - b) i * b i = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_condition_l1054_105497


namespace NUMINAMATH_CALUDE_gena_hits_target_l1054_105476

/-- Calculates the number of hits given the total shots, initial shots, and additional shots per hit -/
def calculate_hits (total_shots initial_shots additional_shots_per_hit : ℕ) : ℕ :=
  (total_shots - initial_shots) / additional_shots_per_hit

/-- Theorem: Given the shooting range conditions, Gena hit the target 6 times -/
theorem gena_hits_target : 
  let initial_shots : ℕ := 5
  let additional_shots_per_hit : ℕ := 2
  let total_shots : ℕ := 17
  calculate_hits total_shots initial_shots additional_shots_per_hit = 6 := by
sorry

#eval calculate_hits 17 5 2

end NUMINAMATH_CALUDE_gena_hits_target_l1054_105476


namespace NUMINAMATH_CALUDE_min_squares_to_exceed_1000_l1054_105465

/-- The function that represents repeated squaring of a number -/
def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

/-- The theorem stating that 3 is the smallest number of squaring operations needed for 5 to exceed 1000 -/
theorem min_squares_to_exceed_1000 :
  (∀ k < 3, repeated_square 5 k ≤ 1000) ∧
  (repeated_square 5 3 > 1000) :=
sorry

end NUMINAMATH_CALUDE_min_squares_to_exceed_1000_l1054_105465


namespace NUMINAMATH_CALUDE_area_of_region_l1054_105460

/-- Rectangle with sides of length 2 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (is_2x2 : width = 2 ∧ height = 2)

/-- Equilateral triangle with side length 2 -/
structure EquilateralTriangle :=
  (side_length : ℝ)
  (is_side_2 : side_length = 2)

/-- Region R inside rectangle and outside triangle -/
structure Region (rect : Rectangle) (tri : EquilateralTriangle) :=
  (inside_rectangle : Prop)
  (outside_triangle : Prop)
  (distance_from_AD : ℝ → Prop)

/-- The theorem to be proved -/
theorem area_of_region 
  (rect : Rectangle) 
  (tri : EquilateralTriangle) 
  (R : Region rect tri) : 
  ∃ (area : ℝ), 
    area = (4 - Real.sqrt 3) / 6 ∧ 
    (∀ x, R.distance_from_AD x → 2/3 ≤ x ∧ x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_area_of_region_l1054_105460


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l1054_105421

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem sequence_is_increasing (a : ℕ → ℝ) 
  (h1 : a 1 < 0) 
  (h2 : ∀ n, a (n + 1) / a n = 1 / 3) : 
  is_increasing a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l1054_105421


namespace NUMINAMATH_CALUDE_min_socks_for_pair_l1054_105442

/-- Represents the color of a sock -/
inductive SockColor
| White
| Blue
| Grey

/-- Represents a sock with its color and whether it has a hole -/
structure Sock :=
  (color : SockColor)
  (hasHole : Bool)

/-- The contents of the sock box -/
def sockBox : List Sock := sorry

/-- The number of socks in the box -/
def totalSocks : Nat := sockBox.length

/-- The number of socks with holes -/
def socksWithHoles : Nat := 3

/-- The number of white socks -/
def whiteSocks : Nat := 2

/-- The number of blue socks -/
def blueSocks : Nat := 3

/-- The number of grey socks -/
def greySocks : Nat := 4

/-- Theorem stating that 7 is the minimum number of socks needed to guarantee a pair without holes -/
theorem min_socks_for_pair (draw : Nat → Sock) :
  ∃ (n : Nat), n ≤ 7 ∧
  ∃ (i j : Nat), i < j ∧ j < n ∧
  (draw i).color = (draw j).color ∧
  ¬(draw i).hasHole ∧ ¬(draw j).hasHole :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_pair_l1054_105442


namespace NUMINAMATH_CALUDE_a_plus_reward_is_ten_l1054_105455

/-- Represents the grading system and reward structure for Paul's courses. -/
structure GradingSystem where
  num_courses : ℕ
  reward_b_plus : ℚ
  reward_a : ℚ
  max_reward : ℚ

/-- Calculates the maximum reward Paul can receive given a grading system and A+ reward. -/
def max_reward (gs : GradingSystem) (reward_a_plus : ℚ) : ℚ :=
  let doubled_reward_b_plus := 2 * gs.reward_b_plus
  let doubled_reward_a := 2 * gs.reward_a
  max (gs.num_courses * doubled_reward_a)
      (((gs.num_courses - 1) * doubled_reward_a) + reward_a_plus)

/-- Theorem stating that the A+ reward must be $10 to achieve the maximum possible reward. -/
theorem a_plus_reward_is_ten (gs : GradingSystem) 
    (h_num_courses : gs.num_courses = 10)
    (h_reward_b_plus : gs.reward_b_plus = 5)
    (h_reward_a : gs.reward_a = 10)
    (h_max_reward : gs.max_reward = 190) :
    ∃ (reward_a_plus : ℚ), reward_a_plus = 10 ∧ max_reward gs reward_a_plus = gs.max_reward :=
  sorry


end NUMINAMATH_CALUDE_a_plus_reward_is_ten_l1054_105455


namespace NUMINAMATH_CALUDE_total_cars_count_l1054_105436

/-- The number of cars owned by Cathy, Lindsey, Carol, and Susan -/
def total_cars (cathy lindsey carol susan : ℕ) : ℕ :=
  cathy + lindsey + carol + susan

/-- Theorem stating the total number of cars owned by all four people -/
theorem total_cars_count :
  ∀ (cathy lindsey carol susan : ℕ),
    cathy = 5 →
    lindsey = cathy + 4 →
    carol = 2 * cathy →
    susan = carol - 2 →
    total_cars cathy lindsey carol susan = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_count_l1054_105436


namespace NUMINAMATH_CALUDE_range_of_m_for_p_range_of_m_for_p_and_q_l1054_105409

-- Define the equations for p and q
def p (x y m : ℝ) : Prop := x^2 / (m + 1) + y^2 / (4 - m) = 1
def q (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 2*m*y + 5 = 0

-- Define what it means for p to be an ellipse with foci on the x-axis
def p_is_ellipse (m : ℝ) : Prop := m + 1 > 0 ∧ 4 - m > 0 ∧ m + 1 ≠ 4 - m

-- Define what it means for q to be a circle
def q_is_circle (m : ℝ) : Prop := m^2 - 4 > 0

-- Theorem 1
theorem range_of_m_for_p (m : ℝ) :
  p_is_ellipse m → m > 3/2 ∧ m < 4 :=
sorry

-- Theorem 2
theorem range_of_m_for_p_and_q (m : ℝ) :
  p_is_ellipse m ∧ q_is_circle m → m > 2 ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_p_range_of_m_for_p_and_q_l1054_105409


namespace NUMINAMATH_CALUDE_fraction_simplification_l1054_105489

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  (a - b) / (2*a*b - b^2 - a^2) = 1 / (b - a) := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1054_105489


namespace NUMINAMATH_CALUDE_find_a_l1054_105468

def U : Set ℕ := {1, 3, 5, 7}

theorem find_a (M : Set ℕ) (a : ℕ) (h1 : M = {1, a}) 
  (h2 : (U \ M) = {5, 7}) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1054_105468


namespace NUMINAMATH_CALUDE_work_related_emails_l1054_105493

theorem work_related_emails (total : ℕ) (spam_percent : ℚ) (promo_percent : ℚ) (social_percent : ℚ)
  (h_total : total = 1200)
  (h_spam : spam_percent = 27 / 100)
  (h_promo : promo_percent = 18 / 100)
  (h_social : social_percent = 15 / 100) :
  (total : ℚ) * (1 - (spam_percent + promo_percent + social_percent)) = 480 := by
  sorry

end NUMINAMATH_CALUDE_work_related_emails_l1054_105493


namespace NUMINAMATH_CALUDE_english_sample_count_l1054_105414

/-- Represents the number of books for each subject -/
structure BookCount where
  chinese : ℕ
  math : ℕ
  english : ℕ

/-- Represents the ratio of books for each subject -/
structure BookRatio where
  chinese : ℕ
  math : ℕ
  english : ℕ

/-- Given a ratio of Chinese to English books and the number of Chinese books sampled,
    calculate the number of English books that should be sampled using stratified sampling. -/
def stratifiedSample (ratio : BookRatio) (chineseSampled : ℕ) : ℕ :=
  (ratio.english * chineseSampled) / ratio.chinese

/-- Theorem stating that given the specified ratio and number of Chinese books sampled,
    the number of English books to be sampled is 25. -/
theorem english_sample_count (ratio : BookRatio) (h1 : ratio.chinese = 2) (h2 : ratio.english = 5) :
  stratifiedSample ratio 10 = 25 := by
  sorry

#check english_sample_count

end NUMINAMATH_CALUDE_english_sample_count_l1054_105414


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l1054_105485

theorem no_real_solution_log_equation :
  ¬ ∃ (x : ℝ), (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 8*x + 15)) ∧
               (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 8*x + 15 > 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l1054_105485


namespace NUMINAMATH_CALUDE_room_width_calculation_l1054_105430

def room_length : ℝ := 25
def room_height : ℝ := 12
def door_length : ℝ := 6
def door_width : ℝ := 3
def window_length : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 8
def total_cost : ℝ := 7248

theorem room_width_calculation (x : ℝ) :
  (2 * (room_length * room_height + x * room_height) - 
   (door_length * door_width + ↑num_windows * window_length * window_width)) * 
   cost_per_sqft = total_cost →
  x = 15 := by sorry

end NUMINAMATH_CALUDE_room_width_calculation_l1054_105430


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l1054_105401

theorem ratio_sum_problem (a b c : ℝ) : 
  (a / b = 5 / 3) ∧ (c / b = 4 / 3) ∧ (b = 27) → a + b + c = 108 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l1054_105401


namespace NUMINAMATH_CALUDE_remainder_sum_squares_mod_11_l1054_105440

theorem remainder_sum_squares_mod_11 :
  (2 * (88134^2 + 88135^2 + 88136^2 + 88137^2 + 88138^2 + 88139^2)) % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_squares_mod_11_l1054_105440


namespace NUMINAMATH_CALUDE_lcm_of_24_and_16_l1054_105471

theorem lcm_of_24_and_16 :
  let n : ℕ := 24
  let m : ℕ := 16
  Nat.gcd n m = 8 →
  Nat.lcm n m = 48 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_24_and_16_l1054_105471


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1054_105443

theorem inequality_solution_set :
  let f : ℝ → ℝ := λ x ↦ 2 * x
  let integral_value : ℝ := ∫ x in (0:ℝ)..1, f x
  {x : ℝ | |x - 2| > integral_value} = Set.Ioi 3 ∪ Set.Iio 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1054_105443


namespace NUMINAMATH_CALUDE_domain_of_g_l1054_105424

-- Define the function f with domain [-2, 4]
def f : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 4 }

-- Define the function g as g(x) = f(x) + f(-x)
def g (x : ℝ) : Prop := x ∈ f ∧ (-x) ∈ f

-- Theorem stating that the domain of g is [-2, 2]
theorem domain_of_g : { x : ℝ | g x } = { x : ℝ | -2 ≤ x ∧ x ≤ 2 } := by
  sorry

end NUMINAMATH_CALUDE_domain_of_g_l1054_105424


namespace NUMINAMATH_CALUDE_imo_problem_6_l1054_105447

theorem imo_problem_6 (n : ℕ) (hn : n ≥ 2) :
  (∀ k : ℕ, k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n)) := by
  sorry

end NUMINAMATH_CALUDE_imo_problem_6_l1054_105447


namespace NUMINAMATH_CALUDE_range_of_a_l1054_105458

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 3)
  (h_f1 : f 1 > 1)
  (h_f2015 : f 2015 = (2 * a - 3) / (a + 1)) :
  -1 < a ∧ a < 2/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1054_105458


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_l1054_105477

theorem binomial_coefficient_identity (r m k : ℕ) (h1 : k ≤ m) (h2 : m ≤ r) :
  (Nat.choose r m) * (Nat.choose m k) = (Nat.choose r k) * (Nat.choose (r - k) (m - k)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_l1054_105477


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1054_105475

/-- Calculate the total cost for a group at a restaurant where adults pay and kids eat free -/
theorem restaurant_bill_calculation (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : 
  total_people = 12 →
  num_kids = 7 →
  adult_meal_cost = 3 →
  (total_people - num_kids) * adult_meal_cost = 15 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1054_105475


namespace NUMINAMATH_CALUDE_complex_equation_roots_l1054_105461

theorem complex_equation_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = 3 - I ∧ z₂ = -2 + I ∧ 
  z₁^2 - z₁ = 5 - 5*I ∧ 
  z₂^2 - z₂ = 5 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l1054_105461


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1054_105438

theorem circles_externally_tangent : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*y + 5 = 0}
  ∃ (p : ℝ × ℝ), p ∈ circle1 ∧ p ∈ circle2 ∧
  (∀ (q : ℝ × ℝ), q ≠ p → (q ∈ circle1 → q ∉ circle2) ∧ (q ∈ circle2 → q ∉ circle1)) :=
by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1054_105438


namespace NUMINAMATH_CALUDE_shells_calculation_l1054_105486

/-- Given an initial amount of shells and an additional amount added, 
    calculate the total amount of shells -/
def total_shells (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that with 5 pounds initial and 23 pounds added, 
    the total is 28 pounds -/
theorem shells_calculation :
  total_shells 5 23 = 28 := by
  sorry

end NUMINAMATH_CALUDE_shells_calculation_l1054_105486


namespace NUMINAMATH_CALUDE_max_value_of_f_l1054_105418

theorem max_value_of_f (x : ℝ) (h : 0 < x ∧ x < 2) : 
  ∃ (max_val : ℝ), max_val = 16/3 ∧ ∀ y ∈ Set.Ioo 0 2, x * (8 - 3 * x) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1054_105418


namespace NUMINAMATH_CALUDE_multiply_by_number_l1054_105474

theorem multiply_by_number (x : ℝ) (n : ℝ) : x = 5 → x * n = (16 - x) + 4 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_number_l1054_105474


namespace NUMINAMATH_CALUDE_polynomial_coefficient_values_l1054_105416

theorem polynomial_coefficient_values (a₅ a₄ a₃ a₂ a₁ a₀ : ℝ) :
  (∀ x : ℝ, x^5 = a₅*(2*x+1)^5 + a₄*(2*x+1)^4 + a₃*(2*x+1)^3 + a₂*(2*x+1)^2 + a₁*(2*x+1) + a₀) →
  a₅ = 1/32 ∧ a₄ = -5/32 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_values_l1054_105416


namespace NUMINAMATH_CALUDE_gcf_of_36_48_72_l1054_105435

theorem gcf_of_36_48_72 : Nat.gcd 36 (Nat.gcd 48 72) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_36_48_72_l1054_105435


namespace NUMINAMATH_CALUDE_third_ball_yarn_amount_l1054_105419

/-- The amount of yarn (in feet) used for each ball -/
structure YarnBalls where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Properties of the yarn balls based on the given conditions -/
def validYarnBalls (y : YarnBalls) : Prop :=
  y.first = y.second / 2 ∧ 
  y.third = 3 * y.first ∧ 
  y.second = 18

/-- Theorem stating that the third ball uses 27 feet of yarn -/
theorem third_ball_yarn_amount (y : YarnBalls) (h : validYarnBalls y) : 
  y.third = 27 := by
  sorry

end NUMINAMATH_CALUDE_third_ball_yarn_amount_l1054_105419


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1054_105410

theorem complex_expression_equality : 
  let z₁ : ℂ := (-2 * Real.sqrt 3 + I) / (1 + 2 * Real.sqrt 3 * I)
  let z₂ : ℂ := (Real.sqrt 2 / (1 - I)) ^ 2017
  z₁ + z₂ = Real.sqrt 2 / 2 + (Real.sqrt 2 / 2 + 1) * I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1054_105410


namespace NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l1054_105459

/-- A four-digit palindrome is a number between 1000 and 9999 of the form abba where a and b are digits and a ≠ 0 -/
def FourDigitPalindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem all_four_digit_palindromes_divisible_by_11 :
  ∀ n : ℕ, FourDigitPalindrome n → n % 11 = 0 := by
  sorry

#check all_four_digit_palindromes_divisible_by_11

end NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l1054_105459


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1054_105407

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The foci of a hyperbola -/
def foci (h : Hyperbola) : Point × Point := sorry

/-- Check if a point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop := sorry

/-- Check if three points are on the same circle -/
def on_same_circle (p1 p2 p3 : Point) : Prop := sorry

/-- Check if a circle is tangent to a line segment -/
def circle_tangent_to_segment (center radius : Point) (p1 p2 : Point) : Prop := sorry

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

theorem hyperbola_eccentricity (h : Hyperbola) (p : Point) :
  let (f1, f2) := foci h
  is_on_hyperbola h p ∧
  on_same_circle f1 f2 p ∧
  circle_tangent_to_segment origin f1 p f2 →
  eccentricity h = (3 + 6 * Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1054_105407


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1054_105417

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2 * x + 6| = 3 * x + 9) ↔ (x = -3) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1054_105417


namespace NUMINAMATH_CALUDE_nancy_bills_l1054_105425

/-- The number of 5-dollar bills Nancy has -/
def num_bills : ℕ := sorry

/-- The value of each bill in dollars -/
def bill_value : ℕ := 5

/-- The total amount of money Nancy has in dollars -/
def total_money : ℕ := 45

/-- Theorem stating that Nancy has 9 five-dollar bills -/
theorem nancy_bills : num_bills = 45 / 5 := by sorry

end NUMINAMATH_CALUDE_nancy_bills_l1054_105425


namespace NUMINAMATH_CALUDE_min_students_per_bench_l1054_105411

theorem min_students_per_bench (male_students : ℕ) (benches : ℕ) : 
  male_students = 29 →
  benches = 29 →
  let female_students := 4 * male_students
  let total_students := male_students + female_students
  (total_students + benches - 1) / benches = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_students_per_bench_l1054_105411


namespace NUMINAMATH_CALUDE_landscape_playground_ratio_l1054_105429

/-- Given a rectangular landscape with specific dimensions and a playground,
    prove the ratio of the playground's area to the total landscape area. -/
theorem landscape_playground_ratio :
  ∀ (length breadth playground_area : ℝ),
    breadth = 8 * length →
    breadth = 480 →
    playground_area = 3200 →
    playground_area / (length * breadth) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_landscape_playground_ratio_l1054_105429


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1054_105492

/-- The intersection point of two lines in 2D space -/
def intersection_point (a b c d e f : ℝ) : ℝ × ℝ := sorry

/-- Theorem: The point (-1, -2) is the unique intersection of the given lines -/
theorem intersection_of_lines :
  let line1 : ℝ → ℝ → Prop := λ x y => 2 * x + 3 * y + 8 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => x - y - 1 = 0
  let point := (-1, -2)
  (line1 point.1 point.2 ∧ line2 point.1 point.2) ∧
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = point) := by
sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1054_105492


namespace NUMINAMATH_CALUDE_binomial_8_2_l1054_105482

theorem binomial_8_2 : Nat.choose 8 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_2_l1054_105482


namespace NUMINAMATH_CALUDE_max_sum_of_digits_base8_less_than_1800_l1054_105432

/-- Represents the sum of digits in base 8 for a natural number -/
def sumOfDigitsBase8 (n : ℕ) : ℕ := sorry

/-- The greatest possible sum of digits in base 8 for numbers less than 1800 -/
def maxSumOfDigitsBase8LessThan1800 : ℕ := 23

/-- Theorem stating that the maximum sum of digits in base 8 for positive integers less than 1800 is 23 -/
theorem max_sum_of_digits_base8_less_than_1800 :
  ∀ n : ℕ, 0 < n → n < 1800 → sumOfDigitsBase8 n ≤ maxSumOfDigitsBase8LessThan1800 ∧
  ∃ m : ℕ, 0 < m ∧ m < 1800 ∧ sumOfDigitsBase8 m = maxSumOfDigitsBase8LessThan1800 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_base8_less_than_1800_l1054_105432


namespace NUMINAMATH_CALUDE_complex_real_implies_a_eq_neg_one_l1054_105451

theorem complex_real_implies_a_eq_neg_one (a : ℝ) :
  (Complex.I : ℂ) * (a + 1 : ℝ) = (0 : ℂ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_implies_a_eq_neg_one_l1054_105451


namespace NUMINAMATH_CALUDE_prob_different_colors_specific_l1054_105441

/-- The probability of drawing two chips of different colors with replacement -/
def prob_different_colors (blue red yellow : ℕ) : ℚ :=
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / total
  let prob_not_red := (blue + yellow) / total
  let prob_not_yellow := (blue + red) / total
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_specific :
  prob_different_colors 6 5 4 = 148 / 225 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_specific_l1054_105441


namespace NUMINAMATH_CALUDE_concert_attendance_l1054_105469

theorem concert_attendance (total_tickets : ℕ) 
  (before_start : ℚ) (after_first_song : ℚ) (during_middle : ℕ) 
  (h1 : total_tickets = 900)
  (h2 : before_start = 3/4)
  (h3 : after_first_song = 5/9)
  (h4 : during_middle = 80) : 
  total_tickets - (before_start * total_tickets + 
    after_first_song * (total_tickets - before_start * total_tickets) + 
    during_middle) = 20 := by
sorry

end NUMINAMATH_CALUDE_concert_attendance_l1054_105469


namespace NUMINAMATH_CALUDE_candidates_per_state_l1054_105415

theorem candidates_per_state (total_candidates : ℕ) : 
  (total_candidates : ℝ) * 0.06 + 80 = total_candidates * 0.07 → 
  total_candidates = 8000 := by
  sorry

end NUMINAMATH_CALUDE_candidates_per_state_l1054_105415


namespace NUMINAMATH_CALUDE_camp_cedar_boys_l1054_105403

theorem camp_cedar_boys (boys : ℕ) (girls : ℕ) (counselors : ℕ) : 
  girls = 3 * boys →
  counselors = 20 →
  boys + girls = 8 * counselors →
  boys = 40 := by
sorry

end NUMINAMATH_CALUDE_camp_cedar_boys_l1054_105403


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1054_105457

/-- The y-intercept of the line 3x - 5y = 7 is -7/5 -/
theorem y_intercept_of_line (x y : ℝ) :
  3 * x - 5 * y = 7 → x = 0 → y = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1054_105457
