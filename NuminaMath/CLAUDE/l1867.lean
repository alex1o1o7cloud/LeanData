import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l1867_186786

theorem min_value_theorem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 1/a' + 1/b' + 1/c' ≥ 9/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1867_186786


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1867_186701

/-- Given that i is the imaginary unit, prove that (2*i)/(1+i) = 1+i -/
theorem complex_fraction_equality : (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1867_186701


namespace NUMINAMATH_CALUDE_four_letter_initials_count_l1867_186770

theorem four_letter_initials_count : 
  (Finset.range 10).card ^ 4 = 10000 := by sorry

end NUMINAMATH_CALUDE_four_letter_initials_count_l1867_186770


namespace NUMINAMATH_CALUDE_vlad_sister_height_difference_l1867_186700

/-- Converts feet and inches to total inches -/
def heightToInches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Represents the height difference between two people -/
def heightDifference (person1_feet : ℕ) (person1_inches : ℕ) (person2_feet : ℕ) (person2_inches : ℕ) : ℕ :=
  heightToInches person1_feet person1_inches - heightToInches person2_feet person2_inches

theorem vlad_sister_height_difference :
  heightDifference 6 3 2 10 = 41 := by sorry

end NUMINAMATH_CALUDE_vlad_sister_height_difference_l1867_186700


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l1867_186755

theorem two_digit_integer_problem (m n : ℕ) : 
  (10 ≤ m ∧ m < 100) →  -- m is a 2-digit positive integer
  (10 ≤ n ∧ n < 100) →  -- n is a 2-digit positive integer
  (n % 25 = 0) →        -- n is a multiple of 25
  ((m + n) / 2 : ℚ) = m + n / 100 →  -- average equals decimal representation
  max m n = 50 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l1867_186755


namespace NUMINAMATH_CALUDE_bakery_problem_l1867_186724

/-- The number of ways to distribute additional items into bins, given a minimum per bin -/
def distribute_items (total_items : ℕ) (num_bins : ℕ) (min_per_bin : ℕ) : ℕ :=
  Nat.choose (total_items - num_bins * min_per_bin + num_bins - 1) (num_bins - 1)

theorem bakery_problem :
  distribute_items 10 4 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bakery_problem_l1867_186724


namespace NUMINAMATH_CALUDE_sophia_reading_progress_l1867_186769

theorem sophia_reading_progress (total_pages : ℕ) (pages_read : ℕ) : 
  total_pages = 270 →
  pages_read = (total_pages - pages_read) + 90 →
  pages_read / total_pages = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_sophia_reading_progress_l1867_186769


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1867_186782

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ p q : ℝ, 16 - 4*x - x^2 = 0 ∧ x = p ∨ x = q) → 
  (∃ p q : ℝ, 16 - 4*p - p^2 = 0 ∧ 16 - 4*q - q^2 = 0 ∧ p + q = 4) :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1867_186782


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1867_186752

theorem max_value_trig_expression :
  let f : ℝ → ℝ := λ x => Real.sin (x + π/4) + Real.cos (x + π/4) - Real.tan (x + π/3)
  ∃ (max_val : ℝ), max_val = 1 - Real.sqrt 2 ∧
    ∀ x, -π/3 ≤ x ∧ x ≤ 0 → f x ≤ max_val :=
by
  sorry


end NUMINAMATH_CALUDE_max_value_trig_expression_l1867_186752


namespace NUMINAMATH_CALUDE_marks_fruit_consumption_l1867_186722

/-- Given that Mark had 10 pieces of fruit for the week, kept 2 for next week,
    and brought 3 to school on Friday, prove that he ate 5 pieces in the first four days. -/
theorem marks_fruit_consumption
  (total_fruit : ℕ)
  (kept_for_next_week : ℕ)
  (brought_to_school : ℕ)
  (h1 : total_fruit = 10)
  (h2 : kept_for_next_week = 2)
  (h3 : brought_to_school = 3) :
  total_fruit - kept_for_next_week - brought_to_school = 5 := by
  sorry

end NUMINAMATH_CALUDE_marks_fruit_consumption_l1867_186722


namespace NUMINAMATH_CALUDE_football_exercise_calories_l1867_186702

/-- Calculates the total calories burned during a stair-climbing exercise. -/
def total_calories_burned (round_trips : ℕ) (stairs_one_way : ℕ) (calories_per_stair : ℕ) : ℕ :=
  round_trips * (2 * stairs_one_way) * calories_per_stair

/-- Proves that given the specific conditions, the total calories burned is 16200. -/
theorem football_exercise_calories : 
  total_calories_burned 60 45 3 = 16200 := by
  sorry

end NUMINAMATH_CALUDE_football_exercise_calories_l1867_186702


namespace NUMINAMATH_CALUDE_unique_triple_l1867_186707

theorem unique_triple : 
  ∃! (a b c : ℕ), a ≥ b ∧ b ≥ c ∧ a^3 + 9*b^2 + 9*c + 7 = 1997 ∧ 
  a = 10 ∧ b = 10 ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l1867_186707


namespace NUMINAMATH_CALUDE_john_incentive_amount_l1867_186708

/-- Calculates the incentive amount given to an agent based on commission, advance fees, and amount paid. --/
def calculate_incentive (commission : ℕ) (advance_fees : ℕ) (amount_paid : ℕ) : Int :=
  (commission - advance_fees : Int) - amount_paid

/-- Proves that the incentive amount for John is -1780 Rs, indicating an excess payment. --/
theorem john_incentive_amount :
  let commission : ℕ := 25000
  let advance_fees : ℕ := 8280
  let amount_paid : ℕ := 18500
  calculate_incentive commission advance_fees amount_paid = -1780 := by
  sorry

end NUMINAMATH_CALUDE_john_incentive_amount_l1867_186708


namespace NUMINAMATH_CALUDE_percentage_of_muslim_boys_l1867_186753

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  hindu_percentage = 28 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 187 →
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_boys)) / total_boys = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_muslim_boys_l1867_186753


namespace NUMINAMATH_CALUDE_triangle_properties_l1867_186778

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0)
  (h2 : t.c = 2)
  (h3 : t.a + t.b = t.a * t.b) : 
  t.C = Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1867_186778


namespace NUMINAMATH_CALUDE_shirt_cost_l1867_186776

theorem shirt_cost (num_shirts : ℕ) (num_pants : ℕ) (pant_cost : ℕ) (total_cost : ℕ) : 
  num_shirts = 10 →
  num_pants = num_shirts / 2 →
  pant_cost = 8 →
  total_cost = 100 →
  num_shirts * (total_cost - num_pants * pant_cost) / num_shirts = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l1867_186776


namespace NUMINAMATH_CALUDE_faye_age_l1867_186742

/-- Represents the ages of the people in the problem -/
structure Ages where
  chad : ℕ
  diana : ℕ
  eduardo : ℕ
  faye : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  ages.diana + 5 = ages.eduardo ∧
  ages.eduardo = ages.chad + 6 ∧
  ages.faye = ages.chad + 4 ∧
  ages.diana = 17

/-- The theorem statement -/
theorem faye_age (ages : Ages) : age_conditions ages → ages.faye = 20 := by
  sorry

end NUMINAMATH_CALUDE_faye_age_l1867_186742


namespace NUMINAMATH_CALUDE_set_inequality_l1867_186759

def S : Set ℤ := {x | ∃ k : ℕ+, x = 2 * k + 1}
def A : Set ℤ := {x | ∃ k : ℕ+, x = 2 * k - 1}

theorem set_inequality : A ≠ S := by
  sorry

end NUMINAMATH_CALUDE_set_inequality_l1867_186759


namespace NUMINAMATH_CALUDE_problem_solution_l1867_186795

theorem problem_solution : 
  (0.064 ^ (-(1/3 : ℝ)) - (-(1/8 : ℝ))^0 + 16^(3/4 : ℝ) + 0.25^(1/2 : ℝ) = 10) ∧
  ((2 * Real.log 2 + Real.log 3) / (1 + (1/2 : ℝ) * Real.log 0.36 + (1/3 : ℝ) * Real.log 8) = 1) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1867_186795


namespace NUMINAMATH_CALUDE_subset_implies_membership_l1867_186788

theorem subset_implies_membership {α : Type*} (A B : Set α) (h : A ⊆ B) :
  ∀ x, x ∈ A → x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_membership_l1867_186788


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1867_186784

-- Define an arithmetic sequence with first term a₁ and common difference d
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_common_difference :
  ∃ d : ℝ, ∀ n : ℕ, arithmeticSequence 2 d n = 2 + (n - 1) * d ∧ arithmeticSequence 2 d 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1867_186784


namespace NUMINAMATH_CALUDE_even_perfect_square_factors_l1867_186715

/-- The number of even perfect square factors of 2^6 * 7^10 * 3^2 -/
theorem even_perfect_square_factors : 
  (Finset.filter (fun a => a % 2 = 0 ∧ 2 ≤ a) (Finset.range 7)).card *
  (Finset.filter (fun b => b % 2 = 0) (Finset.range 11)).card *
  (Finset.filter (fun c => c % 2 = 0) (Finset.range 3)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_even_perfect_square_factors_l1867_186715


namespace NUMINAMATH_CALUDE_video_dislikes_l1867_186765

theorem video_dislikes (likes : ℕ) (initial_dislikes : ℕ) (additional_dislikes : ℕ) : 
  likes = 3000 → 
  initial_dislikes = likes / 2 + 100 → 
  additional_dislikes = 1000 → 
  initial_dislikes + additional_dislikes = 2600 :=
by sorry

end NUMINAMATH_CALUDE_video_dislikes_l1867_186765


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_l1867_186799

theorem max_lateral_surface_area (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 10 → 2 * π * x * y ≤ 50 * π :=
by sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_l1867_186799


namespace NUMINAMATH_CALUDE_cow_count_l1867_186746

theorem cow_count (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 16) → cows = 8 := by
sorry

end NUMINAMATH_CALUDE_cow_count_l1867_186746


namespace NUMINAMATH_CALUDE_andrena_has_three_more_than_debelyn_l1867_186754

/-- Represents the number of dolls each person has -/
structure DollCounts where
  debelyn : ℕ
  christel : ℕ
  andrena : ℕ

/-- The initial state of doll ownership -/
def initial : DollCounts :=
  { debelyn := 20
  , christel := 24
  , andrena := 0 }

/-- The state after doll transfers -/
def final : DollCounts :=
  { debelyn := initial.debelyn - 2
  , christel := initial.christel - 5
  , andrena := initial.andrena + 2 + 5 }

theorem andrena_has_three_more_than_debelyn :
  final.andrena = final.christel + 2 →
  final.andrena - final.debelyn = 3 := by
  sorry

end NUMINAMATH_CALUDE_andrena_has_three_more_than_debelyn_l1867_186754


namespace NUMINAMATH_CALUDE_min_value_of_f_l1867_186725

def f (x : ℝ) (m : ℝ) := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 2) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m ∧ f x m = -6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1867_186725


namespace NUMINAMATH_CALUDE_marie_task_completion_time_l1867_186727

-- Define the start time of the first task
def start_time : Nat := 7 * 60  -- 7:00 AM in minutes since midnight

-- Define the end time of the second task
def end_second_task : Nat := 9 * 60 + 20  -- 9:20 AM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem statement
theorem marie_task_completion_time :
  let total_time_two_tasks := end_second_task - start_time
  let task_duration := total_time_two_tasks / 2
  let completion_time := end_second_task + 2 * task_duration
  completion_time = 11 * 60 + 40  -- 11:40 AM in minutes since midnight
:= by sorry

end NUMINAMATH_CALUDE_marie_task_completion_time_l1867_186727


namespace NUMINAMATH_CALUDE_student_count_l1867_186743

theorem student_count (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 16) 
  (h2 : rank_from_left = 6) : 
  rank_from_right + rank_from_left - 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1867_186743


namespace NUMINAMATH_CALUDE_sin_45_equals_sqrt2_div_2_l1867_186764

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def angle_45 (x y : ℝ) : Prop := x = y ∧ x > 0 ∧ y > 0

def right_isosceles_triangle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1 ∧ x = y ∧ x > 0 ∧ y > 0

theorem sin_45_equals_sqrt2_div_2 :
  ∀ x y : ℝ, unit_circle x y → angle_45 x y → right_isosceles_triangle x y →
  Real.sin (45 * π / 180) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_45_equals_sqrt2_div_2_l1867_186764


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1867_186756

theorem tan_alpha_plus_pi_fourth (α : Real) (h : Real.tan (α / 2) = 2) : 
  Real.tan (α + π / 4) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1867_186756


namespace NUMINAMATH_CALUDE_x_squared_plus_nine_y_squared_l1867_186731

theorem x_squared_plus_nine_y_squared (x y : ℝ) 
  (h1 : x + 3*y = 6) (h2 : x*y = -9) : x^2 + 9*y^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_nine_y_squared_l1867_186731


namespace NUMINAMATH_CALUDE_fiftiethTerm_l1867_186761

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
theorem fiftiethTerm : arithmeticSequenceTerm 2 5 50 = 247 := by
  sorry

end NUMINAMATH_CALUDE_fiftiethTerm_l1867_186761


namespace NUMINAMATH_CALUDE_min_abs_z_with_constraint_l1867_186798

theorem min_abs_z_with_constraint (z : ℂ) (h : Complex.abs (z - 2*Complex.I) + Complex.abs (z - 5) = 7) :
  Complex.abs z ≥ 20 * Real.sqrt 29 / 29 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_with_constraint_l1867_186798


namespace NUMINAMATH_CALUDE_smallest_y_absolute_equation_l1867_186737

theorem smallest_y_absolute_equation : 
  let y₁ := -46 / 5
  let y₂ := 64 / 5
  (∀ y : ℚ, |5 * y - 9| = 55 → y ≥ y₁) ∧ 
  |5 * y₁ - 9| = 55 ∧ 
  |5 * y₂ - 9| = 55 ∧
  y₁ < y₂ :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_absolute_equation_l1867_186737


namespace NUMINAMATH_CALUDE_profit_margin_increase_l1867_186797

theorem profit_margin_increase (initial_margin : ℝ) (final_margin : ℝ) : 
  initial_margin = 0.25 →
  final_margin = 0.40 →
  let initial_price := 1 + initial_margin
  let final_price := 1 + final_margin
  (final_price / initial_price - 1) * 100 = 12 := by
sorry

end NUMINAMATH_CALUDE_profit_margin_increase_l1867_186797


namespace NUMINAMATH_CALUDE_black_piece_position_l1867_186728

-- Define the structure of a piece
structure Piece :=
  (cubes : Fin 4 → Unit)
  (shape : String)

-- Define the rectangular prism
structure RectangularPrism :=
  (pieces : Fin 4 → Piece)
  (visible : Fin 4 → Bool)
  (bottom_layer : Fin 2 → Piece)

-- Define the positions
inductive Position
  | A | B | C | D

-- Define the properties of the black piece
def is_black_piece (p : Piece) : Prop :=
  p.shape = "T" ∧ 
  (∃ (i : Fin 4), i.val = 3 → p.cubes i = ())

-- Theorem statement
theorem black_piece_position (prism : RectangularPrism) 
  (h1 : ∃ (i : Fin 4), ¬prism.visible i)
  (h2 : ∃ (i : Fin 2), is_black_piece (prism.bottom_layer i))
  (h3 : ∃ (i : Fin 2), prism.bottom_layer i = prism.pieces 3) :
  ∃ (p : Piece), is_black_piece p ∧ p = prism.pieces 2 :=
sorry

end NUMINAMATH_CALUDE_black_piece_position_l1867_186728


namespace NUMINAMATH_CALUDE_replacement_concentration_theorem_l1867_186763

/-- Given an initial solution concentration, a replacement solution concentration,
    and the fraction of solution replaced, calculate the new concentration. -/
def new_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (fraction_replaced : ℝ) : ℝ :=
  (initial_conc * (1 - fraction_replaced) + replacement_conc * fraction_replaced)

/-- Theorem stating that replacing half of a 45% solution with a 25% solution
    results in a 35% solution. -/
theorem replacement_concentration_theorem :
  new_concentration 0.45 0.25 0.5 = 0.35 := by
  sorry

#eval new_concentration 0.45 0.25 0.5

end NUMINAMATH_CALUDE_replacement_concentration_theorem_l1867_186763


namespace NUMINAMATH_CALUDE_thread_length_calculation_l1867_186723

/-- The total length of thread required given an original length and an additional fraction -/
def total_length (original : ℝ) (additional_fraction : ℝ) : ℝ :=
  original + original * additional_fraction

/-- Theorem: Given a 12 cm thread and an additional three-quarters requirement, the total length is 21 cm -/
theorem thread_length_calculation : total_length 12 (3/4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_thread_length_calculation_l1867_186723


namespace NUMINAMATH_CALUDE_intersection_distance_is_sqrt_2_l1867_186780

-- Define the two equations
def equation1 (x y : ℝ) : Prop := x^2 + y = 12
def equation2 (x y : ℝ) : Prop := x + y = 12

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ equation1 x y ∧ equation2 x y}

-- State the theorem
theorem intersection_distance_is_sqrt_2 :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt 2 = Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_is_sqrt_2_l1867_186780


namespace NUMINAMATH_CALUDE_maple_leaf_high_basketball_score_l1867_186747

theorem maple_leaf_high_basketball_score :
  ∀ (x : ℚ) (y : ℕ),
    x > 0 →
    (1/3 : ℚ) * x + (3/8 : ℚ) * x + 18 + y = x →
    10 ≤ y →
    y ≤ 30 →
    y = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_maple_leaf_high_basketball_score_l1867_186747


namespace NUMINAMATH_CALUDE_pecan_pies_count_l1867_186739

def total_pies : ℕ := 13
def apple_pies : ℕ := 2
def pumpkin_pies : ℕ := 7

theorem pecan_pies_count : total_pies - apple_pies - pumpkin_pies = 4 := by
  sorry

end NUMINAMATH_CALUDE_pecan_pies_count_l1867_186739


namespace NUMINAMATH_CALUDE_earnings_difference_l1867_186796

/-- Given investment ratios, percentage return ratios, and total earnings,
    prove the difference between b's and a's earnings -/
theorem earnings_difference
  (inv_a inv_b inv_c : ℕ)
  (ret_a ret_b ret_c : ℕ)
  (total_earnings : ℕ)
  (h_inv_ratio : inv_a + inv_b + inv_c = 12)
  (h_inv_a : inv_a = 3)
  (h_inv_b : inv_b = 4)
  (h_inv_c : inv_c = 5)
  (h_ret_ratio : ret_a + ret_b + ret_c = 15)
  (h_ret_a : ret_a = 6)
  (h_ret_b : ret_b = 5)
  (h_ret_c : ret_c = 4)
  (h_total : total_earnings = 4350) :
  inv_b * ret_b - inv_a * ret_a = 150 := by
  sorry


end NUMINAMATH_CALUDE_earnings_difference_l1867_186796


namespace NUMINAMATH_CALUDE_transform_point_l1867_186779

def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem transform_point (p : ℝ × ℝ) :
  reflectX (rotate180 p) = (p.1, -p.2) := by sorry

end NUMINAMATH_CALUDE_transform_point_l1867_186779


namespace NUMINAMATH_CALUDE_smartphone_sample_correct_l1867_186706

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_item : ℕ

/-- Conditions for the smartphone sampling problem -/
def smartphone_sample : SystematicSample where
  population_size := 160
  sample_size := 20
  group_size := 8
  first_item := 2  -- This is what we want to prove

theorem smartphone_sample_correct :
  let s := smartphone_sample
  s.population_size = 160 ∧
  s.sample_size = 20 ∧
  s.group_size = 8 ∧
  (s.first_item + 8 * 8 + s.first_item + 9 * 8 = 140) →
  s.first_item = 2 := by sorry

end NUMINAMATH_CALUDE_smartphone_sample_correct_l1867_186706


namespace NUMINAMATH_CALUDE_number_between_5_and_9_greater_than_7_l1867_186732

theorem number_between_5_and_9_greater_than_7 : ∃! x : ℝ, 5 < x ∧ x < 9 ∧ 7 < x := by
  sorry

end NUMINAMATH_CALUDE_number_between_5_and_9_greater_than_7_l1867_186732


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l1867_186710

def i : ℂ := Complex.I

theorem complex_expression_evaluation : i * (1 - 2*i) = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l1867_186710


namespace NUMINAMATH_CALUDE_students_in_class_l1867_186751

theorem students_in_class (b : ℕ) : 
  100 < b ∧ b < 200 ∧ 
  b % 3 = 1 ∧ 
  b % 4 = 1 ∧ 
  b % 5 = 1 → 
  b = 101 ∨ b = 161 := by sorry

end NUMINAMATH_CALUDE_students_in_class_l1867_186751


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_10_l1867_186794

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_10 :
  unitsDigit (sumFactorials 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_10_l1867_186794


namespace NUMINAMATH_CALUDE_initial_concentration_is_40_percent_l1867_186775

-- Define the capacities and concentrations
def vessel1_capacity : ℝ := 2
def vessel2_capacity : ℝ := 6
def vessel2_concentration : ℝ := 0.6
def total_liquid : ℝ := 8
def new_vessel_capacity : ℝ := 10
def final_concentration : ℝ := 0.44

-- Define the unknown initial concentration of vessel 1
def vessel1_concentration : ℝ := sorry

-- Theorem statement
theorem initial_concentration_is_40_percent :
  vessel1_concentration * vessel1_capacity + 
  vessel2_concentration * vessel2_capacity = 
  final_concentration * new_vessel_capacity := by
  sorry

end NUMINAMATH_CALUDE_initial_concentration_is_40_percent_l1867_186775


namespace NUMINAMATH_CALUDE_range_of_k_for_inequality_l1867_186762

theorem range_of_k_for_inequality (k : ℝ) : 
  (∀ a b : ℝ, (a - b)^2 ≥ k * a * b) ↔ k ∈ Set.Icc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_for_inequality_l1867_186762


namespace NUMINAMATH_CALUDE_sum_first_and_ninth_term_l1867_186785

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sum_first_and_ninth_term : a 1 + a 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_and_ninth_term_l1867_186785


namespace NUMINAMATH_CALUDE_investment_rate_problem_l1867_186757

/-- Calculates simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_problem (principal : ℝ) (time : ℝ) (standardRate : ℝ) (additionalInterest : ℝ) :
  principal = 2500 →
  time = 2 →
  standardRate = 0.12 →
  additionalInterest = 300 →
  ∃ (rate : ℝ),
    simpleInterest principal rate time = simpleInterest principal standardRate time + additionalInterest ∧
    rate = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l1867_186757


namespace NUMINAMATH_CALUDE_geometric_sequence_theorem_l1867_186791

def geometric_sequence (a₁ : ℚ) (q : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n + 1 => a₁ * q^n

def sum_sequence (a : ℕ → ℚ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => sum_sequence a n + a (n + 1)

def b (S : ℕ → ℚ) (n : ℕ) : ℚ := S n + 1 / (S n)

def is_arithmetic_sequence (a b c : ℚ) : Prop := b - a = c - b

theorem geometric_sequence_theorem (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h1 : a 1 = 3/2)
  (h2 : ∀ n, S n = sum_sequence a n)
  (h3 : is_arithmetic_sequence (-2 * S 2) (S 3) (4 * S 4)) :
  (∃ q : ℚ, ∀ n, a n = geometric_sequence (3/2) q n) ∧
  (∃ l m : ℚ, ∀ n, l ≤ b S n ∧ b S n ≤ m ∧ m - l = 1/6) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_theorem_l1867_186791


namespace NUMINAMATH_CALUDE_intersection_line_ellipse_part1_intersection_line_ellipse_part2_l1867_186740

noncomputable section

-- Define the line and ellipse
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1
def ellipse (a : ℝ) (x y : ℝ) : Prop := 3 * x^2 + y^2 = a

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the area of a triangle given the coordinates of its vertices
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem intersection_line_ellipse_part1 (a : ℝ) (x1 y1 x2 y2 : ℝ) :
  line 1 x1 = y1 →
  line 1 x2 = y2 →
  ellipse a x1 y1 →
  ellipse a x2 y2 →
  distance x1 y1 x2 y2 = Real.sqrt 10 / 2 →
  a = 2 := by sorry

theorem intersection_line_ellipse_part2 (k a : ℝ) (x1 y1 x2 y2 : ℝ) :
  k ≠ 0 →
  line k x1 = y1 →
  line k x2 = y2 →
  ellipse a x1 y1 →
  ellipse a x2 y2 →
  x1 = -2 * x2 →
  ∃ (max_area : ℝ),
    (∀ (k' a' : ℝ) (x1' y1' x2' y2' : ℝ),
      k' ≠ 0 →
      line k' x1' = y1' →
      line k' x2' = y2' →
      ellipse a' x1' y1' →
      ellipse a' x2' y2' →
      x1' = -2 * x2' →
      triangle_area 0 0 x1' y1' x2' y2' ≤ max_area) ∧
    max_area = Real.sqrt 3 / 2 ∧
    a = 5 := by sorry

end NUMINAMATH_CALUDE_intersection_line_ellipse_part1_intersection_line_ellipse_part2_l1867_186740


namespace NUMINAMATH_CALUDE_intersection_point_on_line_and_x_axis_l1867_186734

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * y - 7 * x = 35

/-- The point of intersection -/
def intersection_point : ℝ × ℝ := (-5, 0)

/-- Theorem: The intersection point satisfies the line equation and lies on the x-axis -/
theorem intersection_point_on_line_and_x_axis :
  line_equation intersection_point.1 intersection_point.2 ∧ intersection_point.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_on_line_and_x_axis_l1867_186734


namespace NUMINAMATH_CALUDE_diagonal_length_l1867_186716

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = FG = 12
  (ex - fx)^2 + (ey - fy)^2 = 12^2 ∧
  (fx - gx)^2 + (fy - gy)^2 = 12^2 ∧
  -- GH = HE = 20
  (gx - hx)^2 + (gy - hy)^2 = 20^2 ∧
  (hx - ex)^2 + (hy - ey)^2 = 20^2 ∧
  -- Angle GHE = 90°
  (gx - hx) * (ex - hx) + (gy - hy) * (ey - hy) = 0

-- Theorem statement
theorem diagonal_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  let (ex, ey) := q.E
  let (gx, gy) := q.G
  (ex - gx)^2 + (ey - gy)^2 = 2 * 20^2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_l1867_186716


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l1867_186709

theorem square_of_negative_sqrt_two : (-Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l1867_186709


namespace NUMINAMATH_CALUDE_apple_pear_cost_l1867_186726

theorem apple_pear_cost (x y : ℝ) 
  (eq1 : x + 2*y = 194) 
  (eq2 : 2*x + 5*y = 458) : 
  x = 54 ∧ y = 70 := by
  sorry

end NUMINAMATH_CALUDE_apple_pear_cost_l1867_186726


namespace NUMINAMATH_CALUDE_simplified_expansion_terms_l1867_186789

/-- The number of terms in the simplified expansion of (x+y+z)^2008 + (x-y-z)^2008 -/
def num_terms : ℕ :=
  (Finset.range 1005).card + (Finset.range 1006).card

theorem simplified_expansion_terms :
  num_terms = 505815 :=
sorry

end NUMINAMATH_CALUDE_simplified_expansion_terms_l1867_186789


namespace NUMINAMATH_CALUDE_parabola_intersection_angle_l1867_186758

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line type -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Intersection point type -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem parabola_intersection_angle (C : Parabola) (F M : ℝ × ℝ) (l : Line) 
  (A B : IntersectionPoint) :
  C.equation = (fun x y => y^2 = 8*x) →
  F = (2, 0) →
  M = (-2, 2) →
  l.point = F →
  (C.equation A.x A.y ∧ C.equation B.x B.y) →
  (A.y - M.2) * (B.y - M.2) = -(A.x - M.1) * (B.x - M.1) →
  l.slope = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_angle_l1867_186758


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l1867_186705

theorem stratified_sampling_proportion (total_population : ℕ) (stratum_a : ℕ) (stratum_b : ℕ) (sample_size : ℕ) :
  total_population = stratum_a + stratum_b →
  total_population = 120 →
  stratum_a = 20 →
  stratum_b = 100 →
  sample_size = 12 →
  (sample_size * stratum_a) / total_population = 2 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l1867_186705


namespace NUMINAMATH_CALUDE_integer_solutions_of_polynomial_l1867_186774

theorem integer_solutions_of_polynomial (n : ℤ) : 
  n^5 - 2*n^4 - 7*n^2 - 7*n + 3 = 0 ↔ n = -1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_polynomial_l1867_186774


namespace NUMINAMATH_CALUDE_train_distance_problem_l1867_186738

/-- Proves that the distance between two stations is 450 km, given the conditions of the train problem. -/
theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : v2 > v1) (h4 : d > 0) :
  let t := d / v1
  let d1 := v1 * t
  let d2 := v2 * t
  d2 - d1 = 50 → d1 + d2 = 450 := by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l1867_186738


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1867_186730

theorem cost_price_calculation (C : ℝ) : 
  (0.9 * C = C - 0.1 * C) →
  (1.1 * C = C + 0.1 * C) →
  (1.1 * C - 0.9 * C = 50) →
  C = 250 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1867_186730


namespace NUMINAMATH_CALUDE_nelly_earnings_per_night_l1867_186717

/-- Calculates Nelly's earnings per night babysitting given the pizza party conditions -/
theorem nelly_earnings_per_night (total_people : ℕ) (pizza_cost : ℚ) (people_per_pizza : ℕ) (babysitting_nights : ℕ) : 
  total_people = 15 →
  pizza_cost = 12 →
  people_per_pizza = 3 →
  babysitting_nights = 15 →
  (total_people : ℚ) / (people_per_pizza : ℚ) * pizza_cost / (babysitting_nights : ℚ) = 4 := by
  sorry

#check nelly_earnings_per_night

end NUMINAMATH_CALUDE_nelly_earnings_per_night_l1867_186717


namespace NUMINAMATH_CALUDE_linear_function_through_points_l1867_186767

/-- Given a linear function y = ax + a where a is a constant, and the graph of this function 
    passes through the point (1,2), prove that the graph also passes through the point (-2,-1). -/
theorem linear_function_through_points (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f = λ x => a * x + a) → 
  (2 = a * 1 + a) → 
  (-1 = a * (-2) + a) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_through_points_l1867_186767


namespace NUMINAMATH_CALUDE_xOzSymmetry_of_A_l1867_186773

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Performs symmetry transformation with respect to the xOz plane -/
def xOzSymmetry (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- The original point A -/
def A : Point3D :=
  { x := 2, y := -3, z := 1 }

/-- The expected result after symmetry -/
def expectedResult : Point3D :=
  { x := 2, y := 3, z := 1 }

theorem xOzSymmetry_of_A : xOzSymmetry A = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_xOzSymmetry_of_A_l1867_186773


namespace NUMINAMATH_CALUDE_ana_bonita_age_difference_ana_bonita_age_difference_proof_l1867_186714

theorem ana_bonita_age_difference : ℕ → Prop := fun n =>
  ∀ (A B : ℕ),
    A = B + n →                    -- Ana is n years older than Bonita
    A - 1 = 3 * (B - 1) →          -- Last year Ana was 3 times as old as Bonita
    A = B * B →                    -- This year Ana's age is the square of Bonita's age
    n = 2                          -- The age difference is 2 years

-- The proof goes here
theorem ana_bonita_age_difference_proof : ana_bonita_age_difference 2 := by
  sorry

#check ana_bonita_age_difference_proof

end NUMINAMATH_CALUDE_ana_bonita_age_difference_ana_bonita_age_difference_proof_l1867_186714


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1867_186721

theorem floor_equation_solution (A B : ℝ) (hA : A ≥ 0) (hB : B ≥ 0) :
  (∀ x : ℝ, x > 1 → ⌊1 / (A * x + B / x)⌋ = 1 / (A * ⌊x⌋ + B / ⌊x⌋)) →
  A = 0 ∧ B = 1 := by
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1867_186721


namespace NUMINAMATH_CALUDE_tangent_circle_properties_l1867_186768

/-- A circle with center (1, 2) that is tangent to the x-axis -/
def TangentCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

/-- The center of the circle -/
def center : ℝ × ℝ := (1, 2)

/-- The radius of the circle -/
def radius : ℝ := 2

theorem tangent_circle_properties :
  (∀ p ∈ TangentCircle, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2) ∧
  (∃ p ∈ TangentCircle, p.2 = 0) ∧
  (∀ p ∈ TangentCircle, p.2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_properties_l1867_186768


namespace NUMINAMATH_CALUDE_simplify_expression_l1867_186777

theorem simplify_expression (m n : ℝ) : 
  m - (m^2 * n + 3 * m - 4 * n) + (2 * n * m^2 - 3 * n) = m^2 * n - 2 * m + n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1867_186777


namespace NUMINAMATH_CALUDE_base_n_multiple_of_five_l1867_186735

theorem base_n_multiple_of_five (n : ℕ) : 
  let count := Finset.filter (fun n => (2*n^5 + 3*n^4 + 5*n^3 + 2*n^2 + 3*n + 6) % 5 = 0) 
    (Finset.range 99 ∪ {100})
  (2 ≤ n) → (n ≤ 100) → Finset.card count = 40 := by
  sorry

end NUMINAMATH_CALUDE_base_n_multiple_of_five_l1867_186735


namespace NUMINAMATH_CALUDE_fraction_equality_l1867_186704

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (1 / a - 1 / b = 1 / 3) → (a * b / (a - b) = -3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1867_186704


namespace NUMINAMATH_CALUDE_min_queries_for_100_sets_l1867_186703

/-- Represents a query operation on two sets -/
inductive Query
  | intersect : ℕ → ℕ → Query
  | union : ℕ → ℕ → Query

/-- The result of a query operation -/
def QueryResult := Set ℕ

/-- A function that performs a query on two sets -/
def performQuery : Query → (ℕ → Set ℕ) → QueryResult := sorry

/-- A strategy is a sequence of queries -/
def Strategy := List Query

/-- Checks if a strategy determines all sets -/
def determinesAllSets (s : Strategy) (n : ℕ) : Prop := sorry

/-- The main theorem: 100 queries are necessary and sufficient -/
theorem min_queries_for_100_sets :
  (∃ (s : Strategy), s.length = 100 ∧ determinesAllSets s 100) ∧
  (∀ (s : Strategy), s.length < 100 → ¬determinesAllSets s 100) := by sorry

end NUMINAMATH_CALUDE_min_queries_for_100_sets_l1867_186703


namespace NUMINAMATH_CALUDE_computer_additions_l1867_186713

/-- Represents the number of additions a computer can perform per second. -/
def additions_per_second : ℕ := 15000

/-- Represents the duration in seconds for which we want to calculate the total additions. -/
def duration_in_seconds : ℕ := 2 * 3600 + 30 * 60

/-- Calculates the total number of additions performed by the computer. -/
def total_additions : ℕ := additions_per_second * duration_in_seconds

/-- Theorem stating that the computer performs 135,000,000 additions in the given time. -/
theorem computer_additions : total_additions = 135000000 := by
  sorry

end NUMINAMATH_CALUDE_computer_additions_l1867_186713


namespace NUMINAMATH_CALUDE_digital_earth_functions_l1867_186766

-- Define the concept of Digital Earth
structure DigitalEarth where
  integratesInfo : Bool
  displaysIn3D : Bool
  isDynamic : Bool
  providesExperimentalConditions : Bool

-- Define the correct description of Digital Earth functions
def correctDescription (de : DigitalEarth) : Prop :=
  de.integratesInfo ∧ de.displaysIn3D ∧ de.isDynamic ∧ de.providesExperimentalConditions

-- Theorem stating that the correct description accurately represents Digital Earth functions
theorem digital_earth_functions :
  ∀ (de : DigitalEarth), correctDescription de ↔ 
    (de.integratesInfo = true ∧ 
     de.displaysIn3D = true ∧ 
     de.isDynamic = true ∧ 
     de.providesExperimentalConditions = true) :=
by
  sorry

#check digital_earth_functions

end NUMINAMATH_CALUDE_digital_earth_functions_l1867_186766


namespace NUMINAMATH_CALUDE_positive_reals_inequality_l1867_186790

theorem positive_reals_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 + a * b) / (1 + a) + (1 + b * c) / (1 + b) + (1 + c * a) / (1 + c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_l1867_186790


namespace NUMINAMATH_CALUDE_apple_tripling_theorem_l1867_186720

theorem apple_tripling_theorem (a b c : ℕ) :
  (3 * a + b + c = (17/10) * (a + b + c)) →
  (a + 3 * b + c = (3/2) * (a + b + c)) →
  (a + b + 3 * c = (9/5) * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_apple_tripling_theorem_l1867_186720


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_eccentricity_l1867_186733

/-- Given a hyperbola and a circle that intersect to form a square, 
    prove that the eccentricity of the hyperbola is √(2 + √2) -/
theorem hyperbola_circle_intersection_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c = Real.sqrt (a^2 + b^2)) 
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → x^2 + y^2 = c^2 → x^2 = y^2) : 
  c / a = Real.sqrt (2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_eccentricity_l1867_186733


namespace NUMINAMATH_CALUDE_range_of_a_l1867_186718

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (f = λ x => x * |x^2 - a|) →
  (∃ x ∈ Set.Icc 1 2, f x < 2) →
  -1 < a ∧ a < 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1867_186718


namespace NUMINAMATH_CALUDE_book_reading_fraction_l1867_186787

theorem book_reading_fraction (total_pages : ℝ) (pages_read_more : ℝ) : 
  total_pages = 270.00000000000006 →
  pages_read_more = 90 →
  (total_pages / 2 + pages_read_more / 2) / total_pages = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_book_reading_fraction_l1867_186787


namespace NUMINAMATH_CALUDE_min_manhattan_distance_l1867_186781

-- Define the manhattan distance function
def manhattan_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

-- Define the line
def on_line (x y : ℝ) : Prop :=
  3 * x + 4 * y - 12 = 0

-- State the theorem
theorem min_manhattan_distance :
  ∃ (min_dist : ℝ),
    min_dist = (12 - Real.sqrt 34) / 4 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      on_ellipse x₁ y₁ → on_line x₂ y₂ →
      manhattan_distance x₁ y₁ x₂ y₂ ≥ min_dist :=
by
  sorry

end NUMINAMATH_CALUDE_min_manhattan_distance_l1867_186781


namespace NUMINAMATH_CALUDE_fraction_simplification_and_evaluation_l1867_186750

theorem fraction_simplification_and_evaluation (x : ℝ) (h : x ≠ 2) :
  (x^6 - 16*x^3 + 64) / (x^3 - 8) = x^3 - 8 ∧ 
  (6^6 - 16*6^3 + 64) / (6^3 - 8) = 208 :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_and_evaluation_l1867_186750


namespace NUMINAMATH_CALUDE_pizza_toppings_l1867_186760

theorem pizza_toppings (total_slices ham_slices pineapple_slices : ℕ) 
  (h_total : total_slices = 15)
  (h_ham : ham_slices = 9)
  (h_pineapple : pineapple_slices = 12)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ ham_slices ∨ slice ≤ pineapple_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = ham_slices + pineapple_slices - total_slices ∧
    both_toppings = 6 := by
  sorry


end NUMINAMATH_CALUDE_pizza_toppings_l1867_186760


namespace NUMINAMATH_CALUDE_unique_solution_exists_l1867_186771

/-- Represents a digit from 0 to 7 -/
def Digit := Fin 8

/-- Converts a three-digit number to its integer representation -/
def toInt (a b c : Digit) : Nat := a.val * 100 + b.val * 10 + c.val

/-- Converts a two-digit number to its integer representation -/
def toInt2 (d e : Digit) : Nat := d.val * 10 + e.val

theorem unique_solution_exists (a b c d e f g h : Digit) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
               f ≠ g ∧ f ≠ h ∧
               g ≠ h)
  (abc_eq : toInt a b c = 146)
  (equation : toInt a b c + toInt2 d e = toInt f g h) :
  toInt2 d e = 57 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l1867_186771


namespace NUMINAMATH_CALUDE_central_angle_regular_octagon_l1867_186741

/-- The central angle of a regular octagon is 45 degrees. -/
theorem central_angle_regular_octagon :
  let total_angle : ℝ := 360
  let num_sides : ℕ := 8
  let central_angle := total_angle / num_sides
  central_angle = 45 := by sorry

end NUMINAMATH_CALUDE_central_angle_regular_octagon_l1867_186741


namespace NUMINAMATH_CALUDE_percentage_problem_l1867_186729

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 60) : 0.8 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1867_186729


namespace NUMINAMATH_CALUDE_least_n_for_jumpy_l1867_186792

/-- A permutation of 2021 elements -/
def Permutation := Fin 2021 → Fin 2021

/-- A function that reorders up to 1232 elements in a permutation -/
def reorder_1232 (p : Permutation) : Permutation :=
  sorry

/-- A function that reorders up to n elements in a permutation -/
def reorder_n (n : ℕ) (p : Permutation) : Permutation :=
  sorry

/-- The identity permutation -/
def id_perm : Permutation :=
  sorry

theorem least_n_for_jumpy :
  ∀ n : ℕ,
    (∀ p : Permutation,
      ∃ q : Permutation,
        reorder_n n (reorder_1232 p) = id_perm) ↔
    n ≥ 1234 :=
  sorry

end NUMINAMATH_CALUDE_least_n_for_jumpy_l1867_186792


namespace NUMINAMATH_CALUDE_other_diagonal_length_l1867_186744

/-- A rhombus with known properties -/
structure Rhombus where
  /-- The length of one diagonal -/
  diagonal1 : ℝ
  /-- The area of one of the two equal triangles that make up the rhombus -/
  triangle_area : ℝ
  /-- Assumption that the diagonal1 is positive -/
  diagonal1_pos : 0 < diagonal1
  /-- Assumption that the triangle_area is positive -/
  triangle_area_pos : 0 < triangle_area

/-- The theorem stating the length of the other diagonal given specific conditions -/
theorem other_diagonal_length (r : Rhombus) (h1 : r.diagonal1 = 15) (h2 : r.triangle_area = 75) :
  ∃ diagonal2 : ℝ, diagonal2 = 20 ∧ r.diagonal1 * diagonal2 / 2 = 2 * r.triangle_area := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l1867_186744


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l1867_186783

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 250 * Real.pi) :
  A = Real.pi * r^2 → r = 5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l1867_186783


namespace NUMINAMATH_CALUDE_sequence_classification_l1867_186748

/-- Given a sequence {a_n} where the sum of the first n terms S_n = a^n - 2 (a is a constant, a ≠ 0),
    the sequence {a_n} forms either an arithmetic sequence or a geometric sequence from the second term onwards. -/
theorem sequence_classification (a : ℝ) (h_a : a ≠ 0) :
  let S : ℕ → ℝ := λ n => a ^ n - 2
  let a_seq : ℕ → ℝ := λ n => S n - S (n - 1)
  (∀ n : ℕ, n ≥ 2 → ∃ d : ℝ, a_seq (n + 1) - a_seq n = d) ∨
  (∀ n : ℕ, n ≥ 2 → ∃ r : ℝ, a_seq (n + 1) / a_seq n = r) :=
by sorry

end NUMINAMATH_CALUDE_sequence_classification_l1867_186748


namespace NUMINAMATH_CALUDE_koby_sparklers_count_l1867_186772

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of boxes Cherie has -/
def cherie_boxes : ℕ := 1

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

theorem koby_sparklers_count :
  koby_sparklers_per_box * koby_boxes +
  cherie_sparklers +
  koby_whistlers_per_box * koby_boxes +
  cherie_whistlers = total_fireworks :=
by sorry

end NUMINAMATH_CALUDE_koby_sparklers_count_l1867_186772


namespace NUMINAMATH_CALUDE_arithmetic_progression_possible_n_values_l1867_186712

theorem arithmetic_progression_possible_n_values : 
  ∃! (S : Finset ℕ), 
    S.Nonempty ∧ 
    (∀ n ∈ S, n > 1) ∧
    (S.card = 4) ∧
    (∀ n ∈ S, ∃ a : ℤ, 120 = n * (a + (3 * n / 2 : ℚ) - (3 / 2 : ℚ))) ∧
    (∀ n : ℕ, n > 1 → (∃ a : ℤ, 120 = n * (a + (3 * n / 2 : ℚ) - (3 / 2 : ℚ))) → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_possible_n_values_l1867_186712


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l1867_186745

/-- Represents a cone with given base radius and height -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere with given radius -/
structure Sphere where
  radius : ℝ

/-- Checks if a sphere is inscribed in a cone -/
def isInscribed (c : Cone) (s : Sphere) : Prop :=
  -- This is a placeholder for the actual geometric condition
  True

theorem inscribed_sphere_radius (c : Cone) (s : Sphere) 
  (h1 : c.baseRadius = 15)
  (h2 : c.height = 30)
  (h3 : isInscribed c s) :
  s.radius = 7.5 * Real.sqrt 5 - 7.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l1867_186745


namespace NUMINAMATH_CALUDE_draw_points_value_l1867_186793

/-- Represents the points system in a football competition --/
structure PointSystem where
  victory_points : ℕ
  draw_points : ℕ
  defeat_points : ℕ

/-- Represents the state of a team in the competition --/
structure TeamState where
  total_matches : ℕ
  matches_played : ℕ
  current_points : ℕ
  target_points : ℕ
  min_victories : ℕ

/-- The theorem to prove --/
theorem draw_points_value (ps : PointSystem) (ts : TeamState) : 
  ps.victory_points = 3 ∧ 
  ps.defeat_points = 0 ∧
  ts.total_matches = 20 ∧ 
  ts.matches_played = 5 ∧ 
  ts.current_points = 14 ∧ 
  ts.target_points = 40 ∧
  ts.min_victories = 6 →
  ps.draw_points = 2 := by
  sorry


end NUMINAMATH_CALUDE_draw_points_value_l1867_186793


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1867_186736

theorem infinite_series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series := fun n => 1 / ((2 * (n - 1) * a - (n - 2) * b) * (2 * n * a - (n - 1) * b))
  ∑' n, series n = 1 / ((2 * a - b) * 2 * b) :=
sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1867_186736


namespace NUMINAMATH_CALUDE_t_shirt_packages_l1867_186711

theorem t_shirt_packages (total_shirts : ℕ) (shirts_per_package : ℕ) (h1 : total_shirts = 51) (h2 : shirts_per_package = 3) :
  total_shirts / shirts_per_package = 17 :=
by sorry

end NUMINAMATH_CALUDE_t_shirt_packages_l1867_186711


namespace NUMINAMATH_CALUDE_cylinder_height_l1867_186749

/-- Proves that a cylinder with a circular base perimeter of 6 feet and a side surface
    formed by a rectangular plate with a diagonal of 10 feet has a height of 8 feet. -/
theorem cylinder_height (base_perimeter : ℝ) (diagonal : ℝ) (height : ℝ) : 
  base_perimeter = 6 → diagonal = 10 → height = 8 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l1867_186749


namespace NUMINAMATH_CALUDE_pure_imaginary_z_implies_a_plus_2i_modulus_l1867_186719

theorem pure_imaginary_z_implies_a_plus_2i_modulus (a : ℝ) : 
  let z : ℂ := (a + 3 * Complex.I) / (1 - 2 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (a + 2 * Complex.I) = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_z_implies_a_plus_2i_modulus_l1867_186719
