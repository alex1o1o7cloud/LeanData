import Mathlib

namespace NUMINAMATH_CALUDE_roots_equation_value_l4121_412185

theorem roots_equation_value (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 4 = 0 → 
  x₂^2 - 3*x₂ - 4 = 0 → 
  x₁^2 - 4*x₁ - x₂ + 2*x₁*x₂ = -7 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_value_l4121_412185


namespace NUMINAMATH_CALUDE_function_properties_l4121_412134

/-- Given functions f and g on ℝ satisfying certain properties, 
    prove specific characteristics of their derivatives. -/
theorem function_properties
  (f g : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f (-x + 2))
  (h2 : ∀ x, g (-x + 1) - 2 = -(g (x + 1) - 2))
  (h3 : ∀ x, f (3 - x) + g (x - 1) = 2) :
  (deriv f 2022 = 0) ∧
  (∀ x, deriv g (-x) = -(deriv g x)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4121_412134


namespace NUMINAMATH_CALUDE_problem_solution_l4121_412189

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) (sum : x + y = 5) :
  x = (7 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4121_412189


namespace NUMINAMATH_CALUDE_compare_exponentials_l4121_412149

theorem compare_exponentials (a b c : ℝ) : 
  a = (0.4 : ℝ) ^ (0.3 : ℝ) → 
  b = (0.3 : ℝ) ^ (0.4 : ℝ) → 
  c = (0.3 : ℝ) ^ (-(0.2 : ℝ)) → 
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_compare_exponentials_l4121_412149


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l4121_412115

theorem logarithmic_equation_solution (x : ℝ) :
  (x > 0) →
  (5 * (Real.log x / Real.log (x / 9)) + 
   (Real.log (x^3) / Real.log (9 / x)) + 
   8 * (Real.log (x^2) / Real.log (9 * x^2)) = 2) ↔ 
  (x = 3 ∨ x = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l4121_412115


namespace NUMINAMATH_CALUDE_money_sharing_l4121_412161

theorem money_sharing (emma finn grace total : ℕ) : 
  emma = 45 →
  emma + finn + grace = total →
  3 * finn = 4 * emma →
  3 * grace = 5 * emma →
  total = 180 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l4121_412161


namespace NUMINAMATH_CALUDE_common_factor_of_polynomials_l4121_412119

theorem common_factor_of_polynomials (m : ℝ) : 
  ∃ (k₁ k₂ k₃ : ℝ → ℝ), 
    (m * (m - 3) + 2 * (3 - m) = (m - 2) * k₁ m) ∧
    (m^2 - 4*m + 4 = (m - 2) * k₂ m) ∧
    (m^4 - 16 = (m - 2) * k₃ m) := by
  sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomials_l4121_412119


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l4121_412120

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2*x) / Real.log (1/2)

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Iio (1/2)) := by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l4121_412120


namespace NUMINAMATH_CALUDE_remainder_theorem_l4121_412162

def polynomial (x : ℝ) : ℝ := 4*x^6 - x^5 - 8*x^4 + 3*x^2 + 5*x - 15

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (x - 3) * q x + 2079 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4121_412162


namespace NUMINAMATH_CALUDE_no_exact_change_for_57_can_make_change_for_15_l4121_412186

/-- Represents the available Tyro bill denominations -/
def tyro_bills : List ℕ := [35, 80]

/-- Checks if a given amount can be represented as a sum of available Tyro bills -/
def can_make_exact_change (amount : ℕ) : Prop :=
  ∃ (a b : ℕ), a * 35 + b * 80 = amount

/-- Checks if a given amount can be represented as a difference of sums of available Tyro bills -/
def can_make_change_with_subtraction (amount : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a * 35 + b * 80 - (c * 35 + d * 80) = amount

/-- Theorem stating that exact change cannot be made for 57 Tyros -/
theorem no_exact_change_for_57 : ¬ can_make_exact_change 57 := by sorry

/-- Theorem stating that change can be made for 15 Tyros using subtraction -/
theorem can_make_change_for_15 : can_make_change_with_subtraction 15 := by sorry

end NUMINAMATH_CALUDE_no_exact_change_for_57_can_make_change_for_15_l4121_412186


namespace NUMINAMATH_CALUDE_simplify_fraction_l4121_412170

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4121_412170


namespace NUMINAMATH_CALUDE_range_of_a_l4121_412140

-- Define the sets A and B
def A (a : ℝ) := {x : ℝ | a < x ∧ x < a + 1}
def B := {x : ℝ | 3 + 2*x - x^2 > 0}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, A a ∩ B = A a) ↔ (∀ a : ℝ, -1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4121_412140


namespace NUMINAMATH_CALUDE_existence_of_close_points_l4121_412103

theorem existence_of_close_points :
  ∃ (x y : ℝ), y = x^3 ∧ |y - (x^3 + |x| + 1)| ≤ 1/100 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_close_points_l4121_412103


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4121_412179

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 729 * x^3 + 8 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 78 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4121_412179


namespace NUMINAMATH_CALUDE_lauri_eating_days_l4121_412199

/-- The number of days Lauri ate apples -/
def lauriDays : ℕ := 15

/-- The fraction of an apple Simone ate per day -/
def simonePerDay : ℚ := 1/2

/-- The number of days Simone ate apples -/
def simoneDays : ℕ := 16

/-- The fraction of an apple Lauri ate per day -/
def lauriPerDay : ℚ := 1/3

/-- The total number of apples both girls ate -/
def totalApples : ℕ := 13

theorem lauri_eating_days : 
  simonePerDay * simoneDays + lauriPerDay * lauriDays = totalApples := by
  sorry

end NUMINAMATH_CALUDE_lauri_eating_days_l4121_412199


namespace NUMINAMATH_CALUDE_license_plate_increase_l4121_412157

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^3
  let new_plates := 26^4 * 10^2
  new_plates / old_plates = 26^2 / 10 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l4121_412157


namespace NUMINAMATH_CALUDE_exact_three_green_probability_l4121_412139

def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def num_trials : ℕ := 7
def num_green_selected : ℕ := 3

def probability_green : ℚ := green_marbles / total_marbles
def probability_purple : ℚ := purple_marbles / total_marbles

theorem exact_three_green_probability :
  (Nat.choose num_trials num_green_selected : ℚ) *
  (probability_green ^ num_green_selected) *
  (probability_purple ^ (num_trials - num_green_selected)) =
  860818 / 3421867 := by sorry

end NUMINAMATH_CALUDE_exact_three_green_probability_l4121_412139


namespace NUMINAMATH_CALUDE_solution_set_for_neg_eight_solution_range_for_a_l4121_412106

-- Define the inequality function
def inequality (x a : ℝ) : Prop :=
  |x - 3| + |x + 2| ≤ |a + 1|

-- Theorem 1: Solution set when a = -8
theorem solution_set_for_neg_eight :
  Set.Icc (-3 : ℝ) 4 = {x : ℝ | inequality x (-8)} :=
sorry

-- Theorem 2: Range of a for which the inequality has solutions
theorem solution_range_for_a :
  {a : ℝ | ∃ x, inequality x a} = Set.Iic (-6) ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_neg_eight_solution_range_for_a_l4121_412106


namespace NUMINAMATH_CALUDE_line_direction_vector_l4121_412196

/-- Prove that for a line passing through (1, -3) and (5, 3), 
    if its direction vector is of the form (3, c), then c = 9/2 -/
theorem line_direction_vector (c : ℚ) : 
  (∃ (t : ℚ), (1 + 3*t = 5) ∧ (-3 + c*t = 3)) → c = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l4121_412196


namespace NUMINAMATH_CALUDE_rectangle_original_length_l4121_412187

theorem rectangle_original_length :
  ∀ (original_length : ℝ),
    (original_length * 10 = 25 * 7.2) →
    original_length = 18 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_original_length_l4121_412187


namespace NUMINAMATH_CALUDE_increasing_f_implies_t_geq_5_l4121_412148

/-- A cubic function with a parameter t -/
def f (t : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + t*x + t

/-- The derivative of f with respect to x -/
def f' (t : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*x + t

theorem increasing_f_implies_t_geq_5 :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, Monotone (f t)) →
  t ≥ 5 := by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_t_geq_5_l4121_412148


namespace NUMINAMATH_CALUDE_common_remainder_proof_l4121_412137

theorem common_remainder_proof : 
  let n := 1398 - 7
  (n % 7 = 5) ∧ (n % 9 = 5) ∧ (n % 11 = 5) :=
by sorry

end NUMINAMATH_CALUDE_common_remainder_proof_l4121_412137


namespace NUMINAMATH_CALUDE_marble_selection_theorem_l4121_412108

def total_marbles : ℕ := 15
def red_marbles : ℕ := 1
def green_marbles : ℕ := 1
def blue_marbles : ℕ := 1
def yellow_marbles : ℕ := 1
def other_marbles : ℕ := 11
def marbles_to_choose : ℕ := 5

def choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem marble_selection_theorem :
  (choose_marbles 3 1 * choose_marbles 11 4) +
  (choose_marbles 3 2 * choose_marbles 11 3) +
  (choose_marbles 3 3 * choose_marbles 11 2) = 1540 := by
  sorry

#check marble_selection_theorem

end NUMINAMATH_CALUDE_marble_selection_theorem_l4121_412108


namespace NUMINAMATH_CALUDE_unique_function_property_l4121_412180

theorem unique_function_property (k : ℕ) (f : ℕ → ℕ) 
  (h1 : ∀ n, f n < f (n + 1)) 
  (h2 : ∀ n, f (f n) = n + 2 * k) : 
  ∀ n, f n = n + k := by
  sorry

end NUMINAMATH_CALUDE_unique_function_property_l4121_412180


namespace NUMINAMATH_CALUDE_confucius_travel_equation_l4121_412152

/-- Represents the scenario of Confucius and his students traveling to a school -/
def confucius_travel (x : ℝ) : Prop :=
  let student_speed := x
  let cart_speed := 1.5 * x
  let distance := 30
  let student_time := distance / student_speed
  let confucius_time := distance / cart_speed + 1
  student_time = confucius_time

/-- Theorem stating the equation that holds true for the travel scenario -/
theorem confucius_travel_equation (x : ℝ) (hx : x > 0) :
  confucius_travel x ↔ 30 / x = 30 / (1.5 * x) + 1 :=
sorry

end NUMINAMATH_CALUDE_confucius_travel_equation_l4121_412152


namespace NUMINAMATH_CALUDE_distance_is_27_l4121_412183

/-- The distance between two locations A and B, where two people walk towards each other, 
    meet, continue to their destinations, turn back, and meet again. -/
def distance_between_locations : ℝ :=
  let first_meeting_distance_from_A : ℝ := 10
  let second_meeting_distance_from_B : ℝ := 3
  first_meeting_distance_from_A + (2 * first_meeting_distance_from_A - second_meeting_distance_from_B)

/-- Theorem stating that the distance between locations A and B is 27 kilometers. -/
theorem distance_is_27 : distance_between_locations = 27 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_27_l4121_412183


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4121_412141

def M : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def N : Set ℝ := {x | -4 < x ∧ x ≤ 2}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4121_412141


namespace NUMINAMATH_CALUDE_archery_probabilities_l4121_412163

/-- Represents the probabilities of hitting different rings in archery --/
structure ArcheryProbabilities where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ
  ring7 : ℝ
  below7 : ℝ

/-- The given probabilities for archer Zhang Qiang --/
def zhangQiang : ArcheryProbabilities :=
  { ring10 := 0.24
  , ring9 := 0.28
  , ring8 := 0.19
  , ring7 := 0.16
  , below7 := 0.13 }

/-- The sum of all probabilities should be 1 --/
axiom probSum (p : ArcheryProbabilities) : p.ring10 + p.ring9 + p.ring8 + p.ring7 + p.below7 = 1

theorem archery_probabilities (p : ArcheryProbabilities) 
  (h : p = zhangQiang) : 
  (p.ring10 + p.ring9 = 0.52) ∧ 
  (p.ring10 + p.ring9 + p.ring8 + p.ring7 = 0.87) ∧ 
  (p.ring7 + p.below7 = 0.29) := by
  sorry


end NUMINAMATH_CALUDE_archery_probabilities_l4121_412163


namespace NUMINAMATH_CALUDE_can_empty_table_l4121_412195

/-- Represents a 2x2 table of natural numbers -/
def Table := Fin 2 → Fin 2 → ℕ

/-- Represents a move on the table -/
inductive Move
| RemoveRow (row : Fin 2) : Move
| DoubleColumn (col : Fin 2) : Move

/-- Applies a move to a table -/
def applyMove (t : Table) (m : Move) : Table :=
  match m with
  | Move.RemoveRow row => fun i j => if i = row ∧ t i j > 0 then t i j - 1 else t i j
  | Move.DoubleColumn col => fun i j => if j = col then 2 * t i j else t i j

/-- Checks if a table is empty (all cells are zero) -/
def isEmptyTable (t : Table) : Prop :=
  ∀ i j, t i j = 0

/-- The main theorem: any non-empty table can be emptied -/
theorem can_empty_table (t : Table) (h : ∀ i j, t i j > 0) :
  ∃ (moves : List Move), isEmptyTable (moves.foldl applyMove t) :=
sorry

end NUMINAMATH_CALUDE_can_empty_table_l4121_412195


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l4121_412122

theorem complex_sum_magnitude (a b c : ℂ) :
  Complex.abs a = 1 →
  Complex.abs b = 1 →
  Complex.abs c = 1 →
  a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3 →
  Complex.abs (a + b + c) = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l4121_412122


namespace NUMINAMATH_CALUDE_two_color_theorem_l4121_412100

/-- A type representing a region in the plane --/
def Region : Type := ℕ

/-- A type representing a color (either 0 or 1) --/
def Color : Type := Fin 2

/-- A function that determines if two regions are adjacent --/
def adjacent (n : ℕ) (r1 r2 : Region) : Prop := sorry

/-- A coloring of regions --/
def Coloring (n : ℕ) : Type := Region → Color

/-- A predicate that determines if a coloring is valid --/
def is_valid_coloring (n : ℕ) (c : Coloring n) : Prop :=
  ∀ r1 r2 : Region, adjacent n r1 r2 → c r1 ≠ c r2

/-- The main theorem: there exists a valid two-coloring for any number of circles --/
theorem two_color_theorem (n : ℕ) (h : n ≥ 1) :
  ∃ c : Coloring n, is_valid_coloring n c :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l4121_412100


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l4121_412172

theorem mean_of_three_numbers (n : ℤ) : 
  n = (17 + 23 + 2*n) / 3 → n = 40 := by
sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l4121_412172


namespace NUMINAMATH_CALUDE_replaced_student_weight_l4121_412118

theorem replaced_student_weight
  (n : ℕ)
  (initial_total_weight : ℝ)
  (new_student_weight : ℝ)
  (average_decrease : ℝ)
  (h1 : n = 10)
  (h2 : new_student_weight = 60)
  (h3 : average_decrease = 6)
  (h4 : initial_total_weight - (initial_total_weight / n) + (new_student_weight / n) = initial_total_weight - n * average_decrease) :
  initial_total_weight / n - new_student_weight + n * average_decrease = 120 := by
sorry

end NUMINAMATH_CALUDE_replaced_student_weight_l4121_412118


namespace NUMINAMATH_CALUDE_inequality_problem_l4121_412151

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * y > x * z) ∧
  (∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * (y - z) > 0) ∧
  (∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * z * (z - x) < 0) ∧
  ¬(∀ x y z, x < y ∧ y < z ∧ x * z < 0 → x * y^2 < z * y^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l4121_412151


namespace NUMINAMATH_CALUDE_nested_square_root_simplification_l4121_412104

theorem nested_square_root_simplification :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * (5 ^ (3/4)) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_simplification_l4121_412104


namespace NUMINAMATH_CALUDE_conic_section_foci_l4121_412145

-- Define the polar equation of the conic section
def polar_equation (ρ θ : ℝ) : Prop := ρ = 16 / (5 - 3 * Real.cos θ)

-- Define the focus coordinates
def focus1 : ℝ × ℝ := (0, 0)
def focus2 : ℝ × ℝ := (6, 0)

-- Theorem statement
theorem conic_section_foci (ρ θ : ℝ) :
  polar_equation ρ θ → (focus1 = (0, 0) ∧ focus2 = (6, 0)) :=
sorry

end NUMINAMATH_CALUDE_conic_section_foci_l4121_412145


namespace NUMINAMATH_CALUDE_decimal_division_l4121_412191

theorem decimal_division (x y : ℚ) (hx : x = 0.12) (hy : y = 0.04) :
  x / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l4121_412191


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l4121_412158

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) : 
  a^3 + 1/a^3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l4121_412158


namespace NUMINAMATH_CALUDE_school_population_l4121_412144

theorem school_population (total_sample : ℕ) (first_year_sample : ℕ) (third_year_sample : ℕ) (second_year_total : ℕ) :
  total_sample = 45 →
  first_year_sample = 20 →
  third_year_sample = 10 →
  second_year_total = 300 →
  ∃ (total_students : ℕ), total_students = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_school_population_l4121_412144


namespace NUMINAMATH_CALUDE_fraction_inequality_conditions_l4121_412178

theorem fraction_inequality_conditions (a b : ℝ) :
  (∀ x : ℝ, |((x^2 + a*x + b) / (x^2 + 2*x + 2))| < 1) ↔ (a = 2 ∧ 0 < b ∧ b < 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_conditions_l4121_412178


namespace NUMINAMATH_CALUDE_speed_difference_l4121_412155

/-- Two cars traveling in opposite directions -/
structure TwoCars where
  fast_speed : ℝ
  slow_speed : ℝ
  time : ℝ
  distance : ℝ

/-- The conditions of the problem -/
def problem_conditions (cars : TwoCars) : Prop :=
  cars.fast_speed = 55 ∧
  cars.time = 5 ∧
  cars.distance = 500 ∧
  cars.distance = (cars.fast_speed + cars.slow_speed) * cars.time

/-- The theorem to prove -/
theorem speed_difference (cars : TwoCars) 
  (h : problem_conditions cars) : 
  cars.fast_speed - cars.slow_speed = 10 := by
  sorry


end NUMINAMATH_CALUDE_speed_difference_l4121_412155


namespace NUMINAMATH_CALUDE_square_roots_of_four_l4121_412131

-- Define the square root property
def is_square_root (x y : ℝ) : Prop := y ^ 2 = x

-- Theorem statement
theorem square_roots_of_four :
  ∃ (a b : ℝ), a ≠ b ∧ is_square_root 4 a ∧ is_square_root 4 b ∧
  ∀ (c : ℝ), is_square_root 4 c → (c = a ∨ c = b) :=
sorry

end NUMINAMATH_CALUDE_square_roots_of_four_l4121_412131


namespace NUMINAMATH_CALUDE_books_taken_out_on_friday_l4121_412101

theorem books_taken_out_on_friday 
  (initial_books : ℕ) 
  (taken_out_tuesday : ℕ) 
  (brought_back_thursday : ℕ) 
  (final_books : ℕ) 
  (h1 : initial_books = 235)
  (h2 : taken_out_tuesday = 227)
  (h3 : brought_back_thursday = 56)
  (h4 : final_books = 29) :
  initial_books - taken_out_tuesday + brought_back_thursday - final_books = 35 :=
by sorry

end NUMINAMATH_CALUDE_books_taken_out_on_friday_l4121_412101


namespace NUMINAMATH_CALUDE_hole_depth_proof_l4121_412138

/-- The depth of the hole Mat is digging -/
def hole_depth : ℝ := 120

/-- Mat's height in cm -/
def mat_height : ℝ := 90

theorem hole_depth_proof :
  (mat_height = (3/4) * hole_depth) ∧
  (hole_depth - mat_height = mat_height - (1/2) * hole_depth) :=
by sorry

end NUMINAMATH_CALUDE_hole_depth_proof_l4121_412138


namespace NUMINAMATH_CALUDE_factors_of_72_l4121_412116

theorem factors_of_72 : Finset.card (Nat.divisors 72) = 12 := by sorry

end NUMINAMATH_CALUDE_factors_of_72_l4121_412116


namespace NUMINAMATH_CALUDE_function_properties_l4121_412193

open Real

theorem function_properties (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = Real.sin x * Real.cos x)
  (hg : ∀ x, g x = Real.sin x + Real.cos x) :
  (∀ x y, 0 < x ∧ x < y ∧ y < π/4 → f x < f y ∧ g x < g y) ∧
  (∃ x, f x + g x = 1/2 + Real.sqrt 2 ∧
    ∀ y, f y + g y ≤ 1/2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l4121_412193


namespace NUMINAMATH_CALUDE_sara_hotdog_cost_l4121_412136

/-- The cost of Sara's lunch items -/
structure LunchCost where
  total : ℝ
  salad : ℝ
  hotdog : ℝ

/-- Sara's lunch satisfies the given conditions -/
def sara_lunch : LunchCost where
  total := 10.46
  salad := 5.10
  hotdog := 10.46 - 5.10

/-- Theorem: Sara spent $5.36 on the hotdog -/
theorem sara_hotdog_cost : sara_lunch.hotdog = 5.36 := by
  sorry

end NUMINAMATH_CALUDE_sara_hotdog_cost_l4121_412136


namespace NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l4121_412169

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Point Q where extended sides BC and DE meet -/
def Q (octagon : RegularOctagon) : ℝ × ℝ := sorry

/-- Angle measure in degrees -/
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem regular_octagon_extended_sides_angle 
  (octagon : RegularOctagon) : 
  angle_measure (octagon.vertices 3) (Q octagon) (octagon.vertices 4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l4121_412169


namespace NUMINAMATH_CALUDE_pencils_lost_l4121_412107

theorem pencils_lost (initial_pencils : ℕ) (current_pencils : ℕ) (lost_pencils : ℕ) :
  initial_pencils = 30 →
  current_pencils = 16 →
  current_pencils = initial_pencils - lost_pencils - (initial_pencils - lost_pencils) / 3 →
  lost_pencils = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencils_lost_l4121_412107


namespace NUMINAMATH_CALUDE_students_not_enrolled_l4121_412150

theorem students_not_enrolled (total : ℕ) (football : ℕ) (swimming : ℕ) (both : ℕ) :
  total = 100 →
  football = 37 →
  swimming = 40 →
  both = 15 →
  total - (football + swimming - both) = 38 := by
sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l4121_412150


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_exact_l4121_412190

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 4/w = 1 → x + 2*y ≤ z + 2*w :=
by sorry

theorem min_value_exact (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + 2*y = 9 + 4*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_exact_l4121_412190


namespace NUMINAMATH_CALUDE_angle_A_is_pi_third_triangle_area_l4121_412166

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the vectors
def m (t : Triangle) : ℝ × ℝ := (t.a + t.b + t.c, 3 * t.c)
def n (t : Triangle) : ℝ × ℝ := (t.b, t.c + t.b - t.a)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem 1
theorem angle_A_is_pi_third (t : Triangle) 
  (h : parallel (m t) (n t)) : t.A = π / 3 := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle)
  (h1 : t.a = Real.sqrt 3)
  (h2 : t.b = 1)
  (h3 : t.A = π / 3) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_third_triangle_area_l4121_412166


namespace NUMINAMATH_CALUDE_square_value_l4121_412112

/-- Given that square times 3a equals -3a^2b, prove that square equals -ab -/
theorem square_value (a b : ℝ) (square : ℝ) (h : square * 3 * a = -3 * a^2 * b) :
  square = -a * b := by sorry

end NUMINAMATH_CALUDE_square_value_l4121_412112


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l4121_412160

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : Nat) (position : Nat) : Nat :=
  digit * (2 ^ position)

/-- Represents the binary number 110011 -/
def binaryNumber : List Nat := [1, 1, 0, 0, 1, 1]

/-- Converts a list of binary digits to its decimal representation -/
def listBinaryToDecimal (bits : List Nat) : Nat :=
  (List.zipWith binaryToDecimal bits (List.range bits.length)).sum

theorem binary_110011_equals_51 : listBinaryToDecimal binaryNumber = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l4121_412160


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l4121_412177

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 3) * (x + 1)

-- Theorem statement
theorem axis_of_symmetry :
  ∃ (a b c : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ 
  (a ≠ 0) ∧
  (- b / (2 * a) = -1) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l4121_412177


namespace NUMINAMATH_CALUDE_square_side_from_diagonal_difference_l4121_412110

/-- Given the difference between the diagonal and side of a square, 
    the side of the square can be uniquely determined. -/
theorem square_side_from_diagonal_difference (d_minus_a : ℝ) (d_minus_a_pos : 0 < d_minus_a) :
  ∃! a : ℝ, ∃ d : ℝ, 
    0 < a ∧ 
    d = Real.sqrt (2 * a ^ 2) ∧ 
    d - a = d_minus_a :=
by sorry

end NUMINAMATH_CALUDE_square_side_from_diagonal_difference_l4121_412110


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4121_412135

theorem sufficient_not_necessary (a b x : ℝ) :
  (∀ a b x : ℝ, x > a^2 + b^2 → x > 2*a*b) ∧
  (∃ a b x : ℝ, x > 2*a*b ∧ x ≤ a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4121_412135


namespace NUMINAMATH_CALUDE_three_heads_probability_l4121_412130

def prob_heads : ℚ := 1/2

theorem three_heads_probability :
  let prob_three_heads := prob_heads * prob_heads * prob_heads
  prob_three_heads = 1/8 := by sorry

end NUMINAMATH_CALUDE_three_heads_probability_l4121_412130


namespace NUMINAMATH_CALUDE_conference_theorem_l4121_412132

def conference_problem (total : ℕ) (coffee tea soda : ℕ) 
  (coffee_tea tea_soda coffee_soda : ℕ) (all_three : ℕ) : Prop :=
  let drank_at_least_one := coffee + tea + soda - coffee_tea - tea_soda - coffee_soda + all_three
  total - drank_at_least_one = 5

theorem conference_theorem : 
  conference_problem 30 15 13 9 7 4 3 2 := by sorry

end NUMINAMATH_CALUDE_conference_theorem_l4121_412132


namespace NUMINAMATH_CALUDE_live_streaming_fee_strategy2_revenue_total_profit_l4121_412184

-- Define the problem parameters
def total_items : ℕ := 600
def strategy1_items : ℕ := 200
def strategy2_items : ℕ := 400
def strategy2_phase1_items : ℕ := 100
def strategy2_phase2_items : ℕ := 300

-- Define the strategies
def strategy1_price (m : ℝ) : ℝ := 2 * m - 5
def strategy1_fee_rate : ℝ := 0.01
def strategy2_base_price (m : ℝ) : ℝ := 2.5 * m
def strategy2_discount1 : ℝ := 0.8
def strategy2_discount2 : ℝ := 0.8

-- Theorem statements
theorem live_streaming_fee (m : ℝ) :
  strategy1_items * strategy1_price m * strategy1_fee_rate = 4 * m - 10 := by sorry

theorem strategy2_revenue (m : ℝ) :
  strategy2_phase1_items * strategy2_base_price m * strategy2_discount1 +
  strategy2_phase2_items * strategy2_base_price m * strategy2_discount1 * strategy2_discount2 = 680 * m := by sorry

theorem total_profit (m : ℝ) :
  strategy1_items * strategy1_price m +
  (strategy2_phase1_items * strategy2_base_price m * strategy2_discount1 +
   strategy2_phase2_items * strategy2_base_price m * strategy2_discount1 * strategy2_discount2) -
  (strategy1_items * strategy1_price m * strategy1_fee_rate) -
  (total_items * m) = 476 * m - 990 := by sorry

end NUMINAMATH_CALUDE_live_streaming_fee_strategy2_revenue_total_profit_l4121_412184


namespace NUMINAMATH_CALUDE_smallest_even_abundant_l4121_412114

/-- A number is abundant if the sum of its proper divisors is greater than the number itself. -/
def is_abundant (n : ℕ) : Prop :=
  (Finset.filter (· < n) (Finset.range n)).sum (λ i => if n % i = 0 then i else 0) > n

/-- A number is even if it's divisible by 2. -/
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem smallest_even_abundant : ∀ n : ℕ, is_even n → is_abundant n → n ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_abundant_l4121_412114


namespace NUMINAMATH_CALUDE_increasing_power_function_l4121_412156

/-- A function f(x) = (m^2 - 2m - 2)x^(m^2 + m - 1) is increasing on (0, +∞) if and only if
    m^2 - 2m - 2 > 0 and m^2 + m - 1 > 0 -/
theorem increasing_power_function (m : ℝ) :
  let f := fun (x : ℝ) => (m^2 - 2*m - 2) * x^(m^2 + m - 1)
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ↔ 
  (m^2 - 2*m - 2 > 0 ∧ m^2 + m - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_increasing_power_function_l4121_412156


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l4121_412171

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(23 ∣ (1056 + y))) ∧ (23 ∣ (1056 + x)) → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l4121_412171


namespace NUMINAMATH_CALUDE_people_per_column_l4121_412194

theorem people_per_column (total_people : ℕ) (x : ℕ) : 
  total_people = 16 * x ∧ total_people = 12 * 40 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_l4121_412194


namespace NUMINAMATH_CALUDE_total_crayons_l4121_412182

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 9

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- Theorem stating that the total number of crayons after adding is 12 -/
theorem total_crayons : initial_crayons + added_crayons = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l4121_412182


namespace NUMINAMATH_CALUDE_weaving_problem_l4121_412175

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℕ  -- The sequence
  first_three_sum : a 1 + a 2 + a 3 = 9
  second_fourth_sixth_sum : a 2 + a 4 + a 6 = 15

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℕ :=
  (List.range n).map seq.a |>.sum

theorem weaving_problem (seq : ArithmeticSequence) : sum_n seq 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l4121_412175


namespace NUMINAMATH_CALUDE_least_value_quadratic_l4121_412109

theorem least_value_quadratic (x : ℝ) : 
  (∀ y : ℝ, 4 * y^2 + 8 * y + 3 = 1 → y ≥ -1) ∧ 
  (4 * (-1)^2 + 8 * (-1) + 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l4121_412109


namespace NUMINAMATH_CALUDE_axisymmetric_triangle_is_isosceles_l4121_412127

/-- A triangle is axisymmetric if it has an axis of symmetry. -/
def IsAxisymmetric (t : Triangle) : Prop := sorry

/-- A triangle is isosceles if it has at least two sides of equal length. -/
def IsIsosceles (t : Triangle) : Prop := sorry

/-- If a triangle is axisymmetric, then it is isosceles. -/
theorem axisymmetric_triangle_is_isosceles (t : Triangle) :
  IsAxisymmetric t → IsIsosceles t := by
  sorry

end NUMINAMATH_CALUDE_axisymmetric_triangle_is_isosceles_l4121_412127


namespace NUMINAMATH_CALUDE_deck_size_l4121_412142

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1 / 5 →
  (r : ℚ) / (r + b + 6) = 1 / 7 →
  r + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_deck_size_l4121_412142


namespace NUMINAMATH_CALUDE_sum_seven_times_difference_l4121_412113

theorem sum_seven_times_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x - y = 3 → x + y = 7 * (x - y) → x + y = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_times_difference_l4121_412113


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l4121_412192

theorem inequality_and_equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) ≥ Real.sqrt (a^2 + a*c + c^2)) ∧
  ((Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) = Real.sqrt (a^2 + a*c + c^2)) ↔ 
   (1/b = 1/a + 1/c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l4121_412192


namespace NUMINAMATH_CALUDE_max_value_of_g_l4121_412147

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3 ∧ ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → g y ≤ g x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l4121_412147


namespace NUMINAMATH_CALUDE_rationalize_denominator_l4121_412198

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℚ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧
    B = 7 ∧
    C = 9 ∧
    D = 13 ∧
    E = 5 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l4121_412198


namespace NUMINAMATH_CALUDE_range_of_m_and_n_l4121_412181

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + m > 0}
def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - n ≤ 0}

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- State the theorem
theorem range_of_m_and_n (m n : ℝ) : 
  P ∈ A m ∧ P ∉ B n → m > -1 ∧ n < 5 := by
  sorry


end NUMINAMATH_CALUDE_range_of_m_and_n_l4121_412181


namespace NUMINAMATH_CALUDE_cube_split_2017_l4121_412168

/-- The function that gives the first odd number in the split for m^3 -/
def first_split (m : ℕ) : ℕ := 2 * m * (m - 1) + 1

/-- The predicate that checks if a number is in the split for m^3 -/
def in_split (n m : ℕ) : Prop :=
  ∃ k, 0 < k ∧ k ≤ m^3 ∧ n = first_split m + 2 * (k - 1)

theorem cube_split_2017 :
  ∀ m : ℕ, m > 1 → (in_split 2017 m ↔ m = 47) :=
sorry

end NUMINAMATH_CALUDE_cube_split_2017_l4121_412168


namespace NUMINAMATH_CALUDE_largest_divisible_n_l4121_412121

theorem largest_divisible_n : ∀ n : ℕ+, n^3 + 144 ∣ n + 12 → n ≤ 780 :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l4121_412121


namespace NUMINAMATH_CALUDE_min_trips_for_28_containers_l4121_412154

/-- The minimum number of trips required to transport a given number of containers -/
def min_trips (total_containers : ℕ) (max_per_trip : ℕ) : ℕ :=
  (total_containers + max_per_trip - 1) / max_per_trip

theorem min_trips_for_28_containers :
  min_trips 28 5 = 6 := by
  sorry

#eval min_trips 28 5

end NUMINAMATH_CALUDE_min_trips_for_28_containers_l4121_412154


namespace NUMINAMATH_CALUDE_valid_numbers_count_l4121_412133

/-- A function that generates all valid five-digit numbers satisfying the conditions -/
def validNumbers : List Nat := sorry

/-- A predicate that checks if a number satisfies all conditions -/
def isValid (n : Nat) : Bool := sorry

/-- The main theorem stating that there are exactly 20 valid numbers -/
theorem valid_numbers_count : (validNumbers.filter isValid).length = 20 := by sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l4121_412133


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l4121_412176

-- Define a periodic function with period 2
def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f x

-- Define symmetry around x = 2
def symmetric_around_two (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) = f (2 - x)

-- Define decreasing on an interval
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Define acute angle
def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

-- Theorem statement
theorem sine_cosine_inequality
  (f : ℝ → ℝ)
  (h_periodic : periodic_function f)
  (h_symmetric : symmetric_around_two f)
  (h_decreasing : decreasing_on f (-3) (-2))
  (A B : ℝ)
  (h_acute_A : acute_angle A)
  (h_acute_B : acute_angle B)
  (h_triangle : A + B ≤ Real.pi / 2) :
  f (Real.sin A) > f (Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l4121_412176


namespace NUMINAMATH_CALUDE_y_not_less_than_four_by_at_least_one_l4121_412124

theorem y_not_less_than_four_by_at_least_one (y : ℝ) :
  (y ≥ 5) ↔ (y - 4 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_y_not_less_than_four_by_at_least_one_l4121_412124


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4121_412153

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4121_412153


namespace NUMINAMATH_CALUDE_absolute_value_equality_l4121_412146

theorem absolute_value_equality (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 0 ∧ c = 1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = -1)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l4121_412146


namespace NUMINAMATH_CALUDE_factorization_1_l4121_412174

theorem factorization_1 (m n : ℤ) : 3 * m * n - 6 * m^2 * n^2 = 3 * m * n * (1 - 2 * m * n) :=
by sorry

end NUMINAMATH_CALUDE_factorization_1_l4121_412174


namespace NUMINAMATH_CALUDE_polynomial_remainder_l4121_412102

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 3*x^2 + 5) % (x - 1) = 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4121_412102


namespace NUMINAMATH_CALUDE_exam_score_calculation_l4121_412123

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) :
  total_questions = 50 →
  correct_answers = 36 →
  marks_per_correct = 4 →
  marks_lost_per_wrong = 1 →
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong) = 130 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l4121_412123


namespace NUMINAMATH_CALUDE_assignment_ways_l4121_412126

def total_students : ℕ := 30
def selected_students : ℕ := 10
def group_size : ℕ := 5

def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem assignment_ways :
  (combination total_students selected_students * combination selected_students group_size) / 2 =
  (combination total_students selected_students * combination selected_students group_size) / 2 := by
  sorry

end NUMINAMATH_CALUDE_assignment_ways_l4121_412126


namespace NUMINAMATH_CALUDE_min_additional_games_correct_l4121_412143

/-- The minimum number of additional games needed for the Cheetahs to win at least 80% of all games -/
def min_additional_games : ℕ := 15

/-- The initial number of games played -/
def initial_games : ℕ := 5

/-- The initial number of games won by the Cheetahs -/
def initial_cheetah_wins : ℕ := 1

/-- Checks if the given number of additional games satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  (initial_cheetah_wins + n : ℚ) / (initial_games + n : ℚ) ≥ 4/5

theorem min_additional_games_correct :
  satisfies_condition min_additional_games ∧
  ∀ m : ℕ, m < min_additional_games → ¬satisfies_condition m :=
by sorry

end NUMINAMATH_CALUDE_min_additional_games_correct_l4121_412143


namespace NUMINAMATH_CALUDE_min_absolute_value_complex_l4121_412105

open Complex

theorem min_absolute_value_complex (z : ℂ) :
  (abs (z + I) + abs (z - 2 - I) = 2 * Real.sqrt 2) →
  (∃ (w : ℂ), abs w ≤ abs z ∧ abs (w + I) + abs (w - 2 - I) = 2 * Real.sqrt 2) →
  abs z ≥ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_absolute_value_complex_l4121_412105


namespace NUMINAMATH_CALUDE_bluegrass_percentage_in_x_l4121_412165

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def finalMixture (x y : SeedMixture) (xWeight : ℝ) : SeedMixture :=
  { ryegrass := xWeight * x.ryegrass + (1 - xWeight) * y.ryegrass,
    bluegrass := xWeight * x.bluegrass + (1 - xWeight) * y.bluegrass,
    fescue := xWeight * x.fescue + (1 - xWeight) * y.fescue }

theorem bluegrass_percentage_in_x 
  (x : SeedMixture)
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : y.ryegrass = 0.25)
  (h3 : y.fescue = 0.75)
  (h4 : (finalMixture x y 0.6667).ryegrass = 0.35)
  : x.bluegrass = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_bluegrass_percentage_in_x_l4121_412165


namespace NUMINAMATH_CALUDE_number_of_pens_purchased_l4121_412159

/-- Given the total cost of pens and pencils, the number of pencils, and the prices of pens and pencils,
    prove that the number of pens purchased is 30. -/
theorem number_of_pens_purchased 
  (total_cost : ℝ) 
  (num_pencils : ℕ) 
  (price_pencil : ℝ) 
  (price_pen : ℝ) 
  (h1 : total_cost = 510)
  (h2 : num_pencils = 75)
  (h3 : price_pencil = 2)
  (h4 : price_pen = 12) :
  (total_cost - num_pencils * price_pencil) / price_pen = 30 := by
  sorry


end NUMINAMATH_CALUDE_number_of_pens_purchased_l4121_412159


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l4121_412125

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (hr₁ : r₁ = 10) 
  (hr₂ : r₂ = 6) 
  (hd : contact_distance = 30) : 
  ∃ (center_distance : ℝ), center_distance = 2 * Real.sqrt 229 :=
by sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l4121_412125


namespace NUMINAMATH_CALUDE_product_of_six_consecutive_divisible_by_ten_l4121_412167

theorem product_of_six_consecutive_divisible_by_ten (n : ℕ+) :
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_six_consecutive_divisible_by_ten_l4121_412167


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l4121_412128

theorem arithmetic_sequence_length (a₁ aₙ d n : ℕ) : 
  a₁ = 6 → aₙ = 154 → d = 4 → aₙ = a₁ + (n - 1) * d → n = 38 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l4121_412128


namespace NUMINAMATH_CALUDE_prism_with_five_faces_has_nine_edges_l4121_412129

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  faces : ℕ

/-- The number of edges in a prism given its number of faces. -/
def num_edges (p : Prism) : ℕ :=
  if p.faces = 5 then 9 else 0  -- We only define it for the case of 5 faces

theorem prism_with_five_faces_has_nine_edges (p : Prism) (h : p.faces = 5) : 
  num_edges p = 9 := by
  sorry

#check prism_with_five_faces_has_nine_edges

end NUMINAMATH_CALUDE_prism_with_five_faces_has_nine_edges_l4121_412129


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l4121_412188

/-- Given that -1, a, b, c, -4 form a geometric sequence, prove that a * b * c = -8 -/
theorem geometric_sequence_product (a b c : ℝ) 
  (h : ∃ (r : ℝ), a = -1 * r ∧ b = a * r ∧ c = b * r ∧ -4 = c * r) : 
  a * b * c = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l4121_412188


namespace NUMINAMATH_CALUDE_students_not_enrolled_l4121_412164

theorem students_not_enrolled (total : ℕ) (math : ℕ) (chem : ℕ) (both : ℕ) :
  total = 60 ∧ math = 40 ∧ chem = 30 ∧ both = 25 →
  total - (math + chem - both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l4121_412164


namespace NUMINAMATH_CALUDE_girls_in_choir_l4121_412173

theorem girls_in_choir (orchestra_students band_students choir_students total_students boys_in_choir : ℕ)
  (h1 : orchestra_students = 20)
  (h2 : band_students = 2 * orchestra_students)
  (h3 : boys_in_choir = 12)
  (h4 : total_students = 88)
  (h5 : total_students = orchestra_students + band_students + choir_students) :
  choir_students - boys_in_choir = 16 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_choir_l4121_412173


namespace NUMINAMATH_CALUDE_nine_times_nines_digit_sum_l4121_412197

/-- Represents a number consisting of n nines -/
def nines (n : ℕ) : ℕ := (10^n - 1)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Theorem stating that the product of 9 and a number with 120 nines has a digit sum of 1080 -/
theorem nine_times_nines_digit_sum :
  sumOfDigits (9 * nines 120) = 1080 := by sorry

end NUMINAMATH_CALUDE_nine_times_nines_digit_sum_l4121_412197


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l4121_412117

theorem sunglasses_cap_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_cap_and_sunglasses : ℚ) :
  total_sunglasses = 50 →
  total_caps = 35 →
  prob_cap_and_sunglasses = 2/5 →
  (prob_cap_and_sunglasses * total_caps : ℚ) / total_sunglasses = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l4121_412117


namespace NUMINAMATH_CALUDE_sum_of_digits_power_6_13_l4121_412111

def power_6_13 : ℕ := 6^13

def ones_digit (n : ℕ) : ℕ := n % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem sum_of_digits_power_6_13 :
  ones_digit power_6_13 + tens_digit power_6_13 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_6_13_l4121_412111
