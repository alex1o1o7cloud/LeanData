import Mathlib

namespace NUMINAMATH_CALUDE_withdrawn_players_matches_l1195_119551

/-- Represents a table tennis tournament -/
structure TableTennisTournament where
  n : ℕ  -- Total number of players
  r : ℕ  -- Number of matches played among the 3 withdrawn players

/-- The number of matches played by remaining players -/
def remainingMatches (t : TableTennisTournament) : ℕ :=
  (t.n - 3) * (t.n - 4) / 2

/-- The total number of matches played in the tournament -/
def totalMatches (t : TableTennisTournament) : ℕ :=
  remainingMatches t + (3 * 2 - t.r)

/-- Theorem stating the number of matches played among withdrawn players -/
theorem withdrawn_players_matches (t : TableTennisTournament) : 
  t.n > 3 ∧ totalMatches t = 50 → t.r = 1 := by sorry

end NUMINAMATH_CALUDE_withdrawn_players_matches_l1195_119551


namespace NUMINAMATH_CALUDE_unique_solution_l1195_119565

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the main theorem
theorem unique_solution :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (3^x + x^4 = factorial y + 2019) ↔ (x = 6 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l1195_119565


namespace NUMINAMATH_CALUDE_erased_numbers_l1195_119592

def has_digit (n : ℕ) (d : ℕ) : Prop := ∃ k m : ℕ, n = k * 10 + d + m * 10

theorem erased_numbers (remaining_with_one : ℕ) (remaining_with_two : ℕ) (remaining_without_one_or_two : ℕ) :
  remaining_with_one = 20 →
  remaining_with_two = 19 →
  remaining_without_one_or_two = 30 →
  (∀ n : ℕ, n ≤ 100 → (has_digit n 1 ∨ has_digit n 2 ∨ (¬ has_digit n 1 ∧ ¬ has_digit n 2))) →
  100 - (remaining_with_one + remaining_with_two + remaining_without_one_or_two - 2) = 33 := by
  sorry

end NUMINAMATH_CALUDE_erased_numbers_l1195_119592


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l1195_119533

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l1195_119533


namespace NUMINAMATH_CALUDE_triangle_properties_l1195_119579

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 4)
def B : ℝ × ℝ := (-2, -1)
def C : ℝ × ℝ := (2, 3)

-- Define the height line from B to AC
def height_line (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 8

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, height_line x y ↔ (x + y - 3 = 0)) ∧
  triangle_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1195_119579


namespace NUMINAMATH_CALUDE_triangle_side_length_l1195_119596

open Real

-- Define the triangle
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  -- Given conditions
  c = 4 * sqrt 2 →
  B = π / 4 →  -- 45° in radians
  S = 2 →
  -- Area formula
  S = (1 / 2) * a * c * sin B →
  -- Law of Cosines
  b^2 = a^2 + c^2 - 2*a*c*(cos B) →
  -- Conclusion
  b = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1195_119596


namespace NUMINAMATH_CALUDE_red_car_speed_is_10_l1195_119545

/-- The speed of the black car in miles per hour -/
def black_car_speed : ℝ := 50

/-- The initial distance between the cars in miles -/
def initial_distance : ℝ := 20

/-- The time it takes for the black car to overtake the red car in hours -/
def overtake_time : ℝ := 0.5

/-- The speed of the red car in miles per hour -/
def red_car_speed : ℝ := 10

theorem red_car_speed_is_10 :
  red_car_speed = 10 :=
by sorry

end NUMINAMATH_CALUDE_red_car_speed_is_10_l1195_119545


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l1195_119582

theorem students_liking_both_desserts 
  (total_students : ℕ) 
  (like_apple_pie : ℕ) 
  (like_chocolate_cake : ℕ) 
  (like_neither : ℕ) 
  (h1 : total_students = 50)
  (h2 : like_apple_pie = 25)
  (h3 : like_chocolate_cake = 20)
  (h4 : like_neither = 15) :
  like_apple_pie + like_chocolate_cake - (total_students - like_neither) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l1195_119582


namespace NUMINAMATH_CALUDE_expand_expression_l1195_119552

theorem expand_expression (x y z : ℝ) : 
  (x + 10) * (3 * y + 5 * z + 15) = 3 * x * y + 5 * x * z + 15 * x + 30 * y + 50 * z + 150 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1195_119552


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1195_119543

theorem work_completion_theorem (original_days : ℕ) (reduced_days : ℕ) (additional_men : ℕ) : ∃ (original_men : ℕ), 
  original_days = 10 ∧ 
  reduced_days = 7 ∧ 
  additional_men = 10 ∧
  original_men * original_days = (original_men + additional_men) * reduced_days ∧
  original_men = 24 := by
sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l1195_119543


namespace NUMINAMATH_CALUDE_product_of_numbers_l1195_119512

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 30) (sum_cubes_eq : x^3 + y^3 = 9450) : x * y = -585 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1195_119512


namespace NUMINAMATH_CALUDE_no_valid_cd_l1195_119581

theorem no_valid_cd : ¬ ∃ (C D : ℕ+), 
  (Nat.lcm C D = 210) ∧ 
  (C : ℚ) / (D : ℚ) = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_no_valid_cd_l1195_119581


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1195_119525

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : GeometricSequence a) 
  (h_sum : a 4 + a 8 = -3) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1195_119525


namespace NUMINAMATH_CALUDE_positive_expression_l1195_119578

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  max ((a + b + c)^2 - 8*a*c) (max ((a + b + c)^2 - 8*b*c) ((a + b + c)^2 - 8*a*b)) > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l1195_119578


namespace NUMINAMATH_CALUDE_journey_distance_l1195_119544

theorem journey_distance (train_fraction bus_fraction : ℚ) (walk_distance : ℝ) 
  (h1 : train_fraction = 3/5)
  (h2 : bus_fraction = 7/20)
  (h3 : walk_distance = 6.5)
  (h4 : train_fraction + bus_fraction + (walk_distance / total_distance) = 1) :
  total_distance = 130 :=
by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l1195_119544


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l1195_119513

theorem constant_ratio_problem (x y : ℚ) (k : ℚ) : 
  (k = (5 * x - 3) / (y + 20)) → 
  (y = 2 ∧ x = 1 → k = 1/11) → 
  (y = 5 → x = 58/55) := by
  sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l1195_119513


namespace NUMINAMATH_CALUDE_one_ton_equals_2000_pounds_l1195_119536

/-- The weight of a blue whale's tongue in pounds -/
def tongue_weight_pounds : ℕ := 6000

/-- The weight of a blue whale's tongue in tons -/
def tongue_weight_tons : ℕ := 3

/-- The number of pounds in one ton -/
def pounds_per_ton : ℕ := tongue_weight_pounds / tongue_weight_tons

theorem one_ton_equals_2000_pounds : pounds_per_ton = 2000 := by sorry

end NUMINAMATH_CALUDE_one_ton_equals_2000_pounds_l1195_119536


namespace NUMINAMATH_CALUDE_division_sum_equals_111_l1195_119522

theorem division_sum_equals_111 : (111 / 3) + (222 / 6) + (333 / 9) = 111 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_equals_111_l1195_119522


namespace NUMINAMATH_CALUDE_twentieth_term_is_220_l1195_119548

def a (n : ℕ) : ℚ := (1/2) * n * (n + 2)

theorem twentieth_term_is_220 : a 20 = 220 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_220_l1195_119548


namespace NUMINAMATH_CALUDE_participation_difference_l1195_119508

def participants_2018 : ℕ := 150

def participants_2019 : ℕ := 2 * participants_2018 + 20

def participants_2020 : ℕ := participants_2019 / 2 - 40

def participants_2021 : ℕ := 30 + (participants_2018 - participants_2020)

theorem participation_difference : participants_2019 - participants_2020 = 200 := by
  sorry

end NUMINAMATH_CALUDE_participation_difference_l1195_119508


namespace NUMINAMATH_CALUDE_group_average_age_l1195_119519

theorem group_average_age 
  (num_women : ℕ) 
  (num_men : ℕ) 
  (avg_age_women : ℚ) 
  (avg_age_men : ℚ) 
  (h1 : num_women = 12) 
  (h2 : num_men = 18) 
  (h3 : avg_age_women = 28) 
  (h4 : avg_age_men = 40) : 
  (num_women * avg_age_women + num_men * avg_age_men) / (num_women + num_men : ℚ) = 352 / 10 := by
  sorry

end NUMINAMATH_CALUDE_group_average_age_l1195_119519


namespace NUMINAMATH_CALUDE_missing_number_is_1255_l1195_119560

def given_numbers : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]
def total_count : ℕ := 10
def average : ℕ := 750

theorem missing_number_is_1255 :
  let sum_given := given_numbers.sum
  let total_sum := total_count * average
  total_sum - sum_given = 1255 := by sorry

end NUMINAMATH_CALUDE_missing_number_is_1255_l1195_119560


namespace NUMINAMATH_CALUDE_find_x_l1195_119584

theorem find_x : ∃ x : ℝ, (5 * x) / (180 / 3) + 80 = 81 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1195_119584


namespace NUMINAMATH_CALUDE_divisibility_by_hundred_l1195_119555

theorem divisibility_by_hundred (n : ℕ) : 
  ∃ (k : ℕ), 100 ∣ (5^n + 12*n^2 + 12*n + 3) ↔ n = 5*k + 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_hundred_l1195_119555


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1195_119502

/-- The eight-digit number in the form 973m2158 -/
def eight_digit_number (m : ℕ) : ℕ := 973000000 + m * 10000 + 2158

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9 -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100000000) + ((n / 10000000) % 10) + ((n / 1000000) % 10) + 
  ((n / 100000) % 10) + ((n / 10000) % 10) + ((n / 1000) % 10) + 
  ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem divisible_by_nine (m : ℕ) : 
  (eight_digit_number m) % 9 = 0 ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1195_119502


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l1195_119535

theorem congruence_solutions_count : 
  ∃! (s : Finset ℤ), 
    (∀ x ∈ s, (x^3 + 3*x^2 + x + 3) % 25 = 0) ∧ 
    (∀ x, (x^3 + 3*x^2 + x + 3) % 25 = 0 → x % 25 ∈ s) ∧ 
    s.card = 6 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l1195_119535


namespace NUMINAMATH_CALUDE_watermelon_seeds_theorem_l1195_119521

/-- Calculates the total number of seeds in three watermelons -/
def total_seeds (slices1 slices2 slices3 seeds_per_slice1 seeds_per_slice2 seeds_per_slice3 : ℕ) : ℕ :=
  slices1 * seeds_per_slice1 + slices2 * seeds_per_slice2 + slices3 * seeds_per_slice3

/-- Proves that the total number of seeds in the given watermelons is 6800 -/
theorem watermelon_seeds_theorem :
  total_seeds 40 30 50 60 80 40 = 6800 := by
  sorry

#eval total_seeds 40 30 50 60 80 40

end NUMINAMATH_CALUDE_watermelon_seeds_theorem_l1195_119521


namespace NUMINAMATH_CALUDE_quadratic_root_difference_squares_l1195_119558

theorem quadratic_root_difference_squares (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₁^2 - x₂^2 = c^2 / a^2 → 
  b^4 - c^4 = 4 * a^3 * b * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_squares_l1195_119558


namespace NUMINAMATH_CALUDE_tan_double_angle_l1195_119575

theorem tan_double_angle (α β : Real) 
  (h1 : Real.tan (α + β) = 7)
  (h2 : Real.tan (α - β) = 1) :
  Real.tan (2 * α) = -4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1195_119575


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l1195_119585

theorem sum_of_consecutive_integers (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l1195_119585


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1195_119588

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x ≥ |x + 1| + 1} = {x : ℝ | x > 0.5} := by sorry

-- Part II
theorem range_of_a_part_ii :
  {a : ℝ | {x : ℝ | x ≤ -1} ⊆ {x : ℝ | f a x + 3*x ≤ 0}} = {a : ℝ | -4 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1195_119588


namespace NUMINAMATH_CALUDE_positive_root_equation_l1195_119567

theorem positive_root_equation : ∃ x : ℝ, x > 0 ∧ x^3 - 3*x^2 - x - Real.sqrt 2 = 0 :=
by
  use 2 + Real.sqrt 2
  sorry

end NUMINAMATH_CALUDE_positive_root_equation_l1195_119567


namespace NUMINAMATH_CALUDE_min_cars_in_group_l1195_119524

/-- Represents the properties of a group of cars -/
structure CarGroup where
  total : ℕ
  withAC : ℕ
  withStripes : ℕ
  withACNoStripes : ℕ

/-- The conditions of the car group problem -/
def validCarGroup (g : CarGroup) : Prop :=
  g.total - g.withAC = 47 ∧
  g.withStripes ≥ 55 ∧
  g.withACNoStripes ≤ 45

/-- The theorem stating the minimum number of cars in the group -/
theorem min_cars_in_group (g : CarGroup) (h : validCarGroup g) : g.total ≥ 102 := by
  sorry

#check min_cars_in_group

end NUMINAMATH_CALUDE_min_cars_in_group_l1195_119524


namespace NUMINAMATH_CALUDE_first_player_can_force_odd_result_l1195_119500

/-- A game where two players insert operations between numbers 1 to 100 --/
def NumberGame : Type := List (Fin 100 → ℕ) → Prop

/-- The set of possible operations in the game --/
inductive Operation
| Add
| Subtract
| Multiply

/-- A strategy for a player in the game --/
def Strategy : Type := List Operation → Operation

/-- The result of applying operations to a list of numbers --/
def applyOperations (nums : List ℕ) (ops : List Operation) : ℕ := sorry

/-- A winning strategy ensures an odd result --/
def winningStrategy (s : Strategy) : Prop :=
  ∀ (opponent : Strategy), 
    ∃ (finalOps : List Operation), 
      Odd (applyOperations (List.range 100) finalOps)

/-- Theorem: There exists a winning strategy for the first player --/
theorem first_player_can_force_odd_result :
  ∃ (s : Strategy), winningStrategy s :=
sorry

end NUMINAMATH_CALUDE_first_player_can_force_odd_result_l1195_119500


namespace NUMINAMATH_CALUDE_function_extrema_l1195_119553

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 4*x + 4

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4

-- Theorem statement
theorem function_extrema (a : ℝ) : 
  (f' a 1 = -3) → 
  (a = 1/3) ∧ 
  (∀ x, f (1/3) x ≤ 28/3) ∧ 
  (∀ x, f (1/3) x ≥ -4/3) ∧
  (∃ x, f (1/3) x = 28/3) ∧ 
  (∃ x, f (1/3) x = -4/3) :=
sorry

end NUMINAMATH_CALUDE_function_extrema_l1195_119553


namespace NUMINAMATH_CALUDE_scaled_job_workforce_l1195_119557

/-- Calculates the number of men needed for a scaled job given the original workforce and timelines. -/
def men_needed_for_scaled_job (original_men : ℕ) (original_days : ℕ) (scale_factor : ℕ) (new_days : ℕ) : ℕ :=
  (original_men * original_days * scale_factor) / new_days

/-- Proves that 600 men are needed for a job 3 times the original size, given the original conditions. -/
theorem scaled_job_workforce :
  men_needed_for_scaled_job 250 16 3 20 = 600 := by
  sorry

#eval men_needed_for_scaled_job 250 16 3 20

end NUMINAMATH_CALUDE_scaled_job_workforce_l1195_119557


namespace NUMINAMATH_CALUDE_first_character_lines_l1195_119542

/-- Represents the number of lines for each character in Jerry's skit script. -/
structure ScriptLines where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Defines the conditions for Jerry's skit script. -/
def valid_script (s : ScriptLines) : Prop :=
  s.first = s.second + 8 ∧
  s.third = 2 ∧
  s.second = 3 * s.third + 6

/-- Theorem stating that the first character has 20 lines in a valid script. -/
theorem first_character_lines (s : ScriptLines) (h : valid_script s) : s.first = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_character_lines_l1195_119542


namespace NUMINAMATH_CALUDE_henri_reads_1800_words_l1195_119566

/-- Calculates the number of words read given total free time, movie durations, and reading rate. -/
def words_read (total_time : ℝ) (movie1_duration : ℝ) (movie2_duration : ℝ) (reading_rate : ℝ) : ℝ :=
  (total_time - movie1_duration - movie2_duration) * reading_rate * 60

/-- Proves that Henri reads 1800 words given the specified conditions. -/
theorem henri_reads_1800_words :
  words_read 8 3.5 1.5 10 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_henri_reads_1800_words_l1195_119566


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1195_119564

/-- Proves that cos(70°)sin(80°) + cos(20°)sin(10°) = 1/2 -/
theorem trig_identity_proof : 
  Real.cos (70 * π / 180) * Real.sin (80 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1195_119564


namespace NUMINAMATH_CALUDE_school_election_l1195_119540

theorem school_election (total_students : ℕ) : total_students = 2000 :=
  let voter_percentage : ℚ := 25 / 100
  let winner_vote_percentage : ℚ := 55 / 100
  let loser_vote_percentage : ℚ := 1 - winner_vote_percentage
  let vote_difference : ℕ := 50
  have h1 : (winner_vote_percentage * voter_percentage * total_students : ℚ) = 
            (loser_vote_percentage * voter_percentage * total_students + vote_difference : ℚ) := by sorry
  sorry

end NUMINAMATH_CALUDE_school_election_l1195_119540


namespace NUMINAMATH_CALUDE_january_has_greatest_difference_l1195_119599

-- Define the sales data for each month
def january_sales : (Nat × Nat) := (5, 2)
def february_sales : (Nat × Nat) := (6, 4)
def march_sales : (Nat × Nat) := (5, 5)
def april_sales : (Nat × Nat) := (4, 6)
def may_sales : (Nat × Nat) := (3, 5)

-- Define the percentage difference function
def percentage_difference (sales : Nat × Nat) : ℚ :=
  let (drummers, buglers) := sales
  (↑(max drummers buglers - min drummers buglers) / ↑(min drummers buglers)) * 100

-- Theorem statement
theorem january_has_greatest_difference :
  percentage_difference january_sales >
  max (percentage_difference february_sales)
    (max (percentage_difference march_sales)
      (max (percentage_difference april_sales)
        (percentage_difference may_sales))) :=
by sorry

end NUMINAMATH_CALUDE_january_has_greatest_difference_l1195_119599


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1195_119509

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1195_119509


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1195_119547

/-- Given that a and b are inversely proportional and a = 3b when a + b = 60,
    prove that b = -67.5 when a = -10 -/
theorem inverse_proportion_problem (a b : ℝ) (k : ℝ) : 
  (∀ x y, x * y = k → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) →  -- inverse proportion
  (∃ a' b', a' + b' = 60 ∧ a' = 3 * b') →                   -- condition when sum is 60
  (a = -10) →                                               -- given a value
  (b = -67.5) :=                                            -- to prove
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1195_119547


namespace NUMINAMATH_CALUDE_rope_cutting_l1195_119580

theorem rope_cutting (l : ℚ) : 
  l > 0 ∧ (1 / l).isInt ∧ (2 / l).isInt → (3 / l) ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_l1195_119580


namespace NUMINAMATH_CALUDE_certain_number_proof_l1195_119593

theorem certain_number_proof : ∃ x : ℚ, x - (390 / 5) = (4 - (210 / 7)) + 114 ∧ x = 166 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1195_119593


namespace NUMINAMATH_CALUDE_candy_bar_profit_l1195_119570

/-- Calculates the profit from selling candy bars given the following conditions:
  * 1500 candy bars are bought
  * Buy price is 4 bars for $3
  * 1200 bars are sold at 3 bars for $2
  * 300 bars are sold at 10 bars for $8
-/
theorem candy_bar_profit : 
  let total_bars : ℕ := 1500
  let buy_price : ℚ := 3 / 4
  let sell_price_1 : ℚ := 2 / 3
  let sell_price_2 : ℚ := 8 / 10
  let sold_bars_1 : ℕ := 1200
  let sold_bars_2 : ℕ := 300
  let cost : ℚ := total_bars * buy_price
  let revenue : ℚ := sold_bars_1 * sell_price_1 + sold_bars_2 * sell_price_2
  let profit : ℚ := revenue - cost
  profit = -85 := by sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l1195_119570


namespace NUMINAMATH_CALUDE_min_pieces_correct_l1195_119528

/-- Represents a chessboard of size n x n -/
structure Chessboard (n : ℕ) where
  size : ℕ
  size_pos : size > 0
  size_eq : size = n

/-- A piece on the chessboard -/
structure Piece (n : ℕ) where
  x : Fin n
  y : Fin n

/-- A configuration of pieces on the chessboard -/
def Configuration (n : ℕ) := List (Piece n)

/-- Checks if a configuration satisfies the line coverage property -/
def satisfiesLineCoverage (n : ℕ) (config : Configuration n) : Prop := sorry

/-- The minimum number of pieces required for a valid configuration -/
def minPieces (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 * n else 2 * n + 1

/-- Theorem stating the minimum number of pieces required for a valid configuration -/
theorem min_pieces_correct (n : ℕ) (h : n > 0) :
  ∀ (config : Configuration n),
    satisfiesLineCoverage n config →
    config.length ≥ minPieces n :=
  sorry

end NUMINAMATH_CALUDE_min_pieces_correct_l1195_119528


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l1195_119534

def red_lamps : ℕ := 4
def blue_lamps : ℕ := 3
def green_lamps : ℕ := 3
def total_lamps : ℕ := red_lamps + blue_lamps + green_lamps
def lamps_turned_on : ℕ := 5

def probability_leftmost_green_off_second_right_blue_on : ℚ := 63 / 100

theorem lava_lamp_probability :
  let total_arrangements := Nat.choose total_lamps red_lamps * Nat.choose (total_lamps - red_lamps) blue_lamps
  let leftmost_green_arrangements := Nat.choose (total_lamps - 1) (green_lamps - 1) * Nat.choose (total_lamps - green_lamps) red_lamps * Nat.choose (total_lamps - green_lamps - red_lamps) (blue_lamps - 1)
  let second_right_blue_on_arrangements := Nat.choose (total_lamps - 2) (blue_lamps - 1)
  let remaining_on_lamps := Nat.choose (total_lamps - 2) (lamps_turned_on - 1)
  (leftmost_green_arrangements * second_right_blue_on_arrangements * remaining_on_lamps : ℚ) / (total_arrangements * Nat.choose total_lamps lamps_turned_on) = probability_leftmost_green_off_second_right_blue_on :=
by sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l1195_119534


namespace NUMINAMATH_CALUDE_charity_event_equation_l1195_119506

theorem charity_event_equation :
  ∀ x : ℕ,
  (x + (12 - x) = 12) →  -- Total number of banknotes is 12
  (x ≤ 12) →             -- Ensure x doesn't exceed total banknotes
  (x + 5 * (12 - x) = 48) -- The equation correctly represents the problem
  :=
by
  sorry

end NUMINAMATH_CALUDE_charity_event_equation_l1195_119506


namespace NUMINAMATH_CALUDE_coupon_a_best_at_220_l1195_119539

def coupon_a_discount (price : ℝ) : ℝ := 0.12 * price

def coupon_b_discount (price : ℝ) : ℝ := 25

def coupon_c_discount (price : ℝ) : ℝ := 0.2 * (price - 120)

theorem coupon_a_best_at_220 :
  let price := 220
  coupon_a_discount price > coupon_b_discount price ∧
  coupon_a_discount price > coupon_c_discount price :=
by sorry

end NUMINAMATH_CALUDE_coupon_a_best_at_220_l1195_119539


namespace NUMINAMATH_CALUDE_thousand_ring_date_l1195_119518

/-- Represents a time with hour and minute components -/
structure Time where
  hour : Nat
  minute : Nat

/-- Represents a date with year, month, and day components -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Counts the number of bell rings from a given start time and date until the nth ring -/
def countBellRings (startTime : Time) (startDate : Date) (n : Nat) : Date :=
  sorry

/-- The bell ringing pattern: once at 45 minutes past each hour and according to the hour every hour -/
axiom bell_pattern : ∀ (t : Time), (t.minute = 45 ∧ t.hour ≠ 0) ∨ (t.minute = 0 ∧ t.hour ≠ 0)

/-- The starting time is 10:30 AM on January 1, 2021 -/
def startTime : Time := { hour := 10, minute := 30 }
def startDate : Date := { year := 2021, month := 1, day := 1 }

/-- The theorem to prove -/
theorem thousand_ring_date : 
  countBellRings startTime startDate 1000 = { year := 2021, month := 1, day := 11 } :=
sorry

end NUMINAMATH_CALUDE_thousand_ring_date_l1195_119518


namespace NUMINAMATH_CALUDE_jug_problem_l1195_119549

theorem jug_problem (Cx Cy : ℝ) (h1 : Cx > 0) (h2 : Cy > 0) : 
  (1/6 : ℝ) * Cx = (2/3 : ℝ) * Cy → 
  (1/9 : ℝ) * Cx = (1/3 : ℝ) * Cy - (1/3 : ℝ) * Cy := by
sorry

end NUMINAMATH_CALUDE_jug_problem_l1195_119549


namespace NUMINAMATH_CALUDE_parabola_directrix_l1195_119537

/-- The directrix of a parabola x^2 + 12y = 0 -/
theorem parabola_directrix (x y : ℝ) : 
  (∀ x y, x^2 + 12*y = 0) → (∃ k : ℝ, k = 3 ∧ y = k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1195_119537


namespace NUMINAMATH_CALUDE_sin_period_omega_l1195_119598

/-- 
Given a function y = sin(ωx - π/3) with ω > 0 and a minimum positive period of π,
prove that ω = 2.
-/
theorem sin_period_omega (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x, ∃ y, y = Real.sin (ω * x - π / 3)) 
  (h3 : ∀ T > 0, (∀ x, Real.sin (ω * (x + T) - π / 3) = Real.sin (ω * x - π / 3)) → T ≥ π) 
  (h4 : ∀ x, Real.sin (ω * (x + π) - π / 3) = Real.sin (ω * x - π / 3)) : ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_period_omega_l1195_119598


namespace NUMINAMATH_CALUDE_power_mod_37_l1195_119516

theorem power_mod_37 (n : ℕ) (h1 : n < 37) (h2 : (6 * n) % 37 = 1) :
  (Nat.pow (Nat.pow 2 n) 4 - 3) % 37 = 35 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_37_l1195_119516


namespace NUMINAMATH_CALUDE_trigonometric_sum_equality_l1195_119591

theorem trigonometric_sum_equality : 
  Real.cos (π / 3) + Real.sin (π / 3) - Real.sqrt (3 / 4) + (Real.tan (π / 4))⁻¹ = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equality_l1195_119591


namespace NUMINAMATH_CALUDE_median_of_special_list_l1195_119571

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of elements in the list -/
def total_elements : ℕ := sum_of_first_n 200

/-- The position of the median elements -/
def median_position : ℕ × ℕ := (total_elements / 2, total_elements / 2 + 1)

/-- The value that appears at the median positions -/
def median_value : ℕ := 141

/-- The median of the list -/
def list_median : ℚ := (median_value : ℚ)

theorem median_of_special_list : list_median = 141 := by sorry

end NUMINAMATH_CALUDE_median_of_special_list_l1195_119571


namespace NUMINAMATH_CALUDE_symmetric_point_of_M_l1195_119583

/-- The symmetric point of (x, y) with respect to the x-axis is (x, -y) -/
def symmetricPointXAxis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Given M(-2, -3), its symmetric point with respect to the x-axis is (-2, 3) -/
theorem symmetric_point_of_M : 
  let M : ℝ × ℝ := (-2, -3)
  symmetricPointXAxis M = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_of_M_l1195_119583


namespace NUMINAMATH_CALUDE_expression_evaluation_l1195_119517

theorem expression_evaluation (a b : ℚ) (h1 : a = -3) (h2 : b = 1/3) :
  (a - 3*b) * (a + 3*b) + (a - 3*b)^2 = 24 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1195_119517


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1195_119511

/-- The set A -/
def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}

/-- The set B -/
def B : Set (ℝ × ℝ) := {p | (p.1 - 5) / (p.1 - 4) = 1}

/-- The symmetric difference of two sets -/
def symmetricDifference (X Y : Set α) : Set α :=
  (X \ Y) ∪ (Y \ X)

/-- Theorem: The symmetric difference of A and B -/
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {p : ℝ × ℝ | p.2 = p.1 + 1 ∧ p.1 ≠ 4} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1195_119511


namespace NUMINAMATH_CALUDE_total_fuel_consumption_l1195_119561

/-- Fuel consumption for city driving in liters per km -/
def city_fuel_rate : ℝ := 6

/-- Fuel consumption for highway driving in liters per km -/
def highway_fuel_rate : ℝ := 4

/-- Distance of the city-only trip in km -/
def city_trip : ℝ := 50

/-- Distance of the highway-only trip in km -/
def highway_trip : ℝ := 35

/-- Distance of the mixed trip's city portion in km -/
def mixed_trip_city : ℝ := 15

/-- Distance of the mixed trip's highway portion in km -/
def mixed_trip_highway : ℝ := 10

/-- Theorem stating that the total fuel consumption for all trips is 570 liters -/
theorem total_fuel_consumption :
  city_fuel_rate * city_trip +
  highway_fuel_rate * highway_trip +
  city_fuel_rate * mixed_trip_city +
  highway_fuel_rate * mixed_trip_highway = 570 := by
  sorry


end NUMINAMATH_CALUDE_total_fuel_consumption_l1195_119561


namespace NUMINAMATH_CALUDE_max_k_value_l1195_119523

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 6 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 3/2 ∧ ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ 
    6 = (3/2)^2 * (x'^2 / y'^2 + y'^2 / x'^2) + (3/2) * (x' / y' + y' / x') :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1195_119523


namespace NUMINAMATH_CALUDE_calculator_change_l1195_119538

/-- Calculates the change received after buying three types of calculators. -/
theorem calculator_change (total_money : ℕ) (basic_cost : ℕ) : 
  total_money = 100 →
  basic_cost = 8 →
  total_money - (basic_cost + 2 * basic_cost + 3 * (2 * basic_cost)) = 28 := by
  sorry

#check calculator_change

end NUMINAMATH_CALUDE_calculator_change_l1195_119538


namespace NUMINAMATH_CALUDE_line_arrangements_l1195_119573

theorem line_arrangements (n : ℕ) (h : n = 6) :
  (n - 1) * Nat.factorial (n - 1) = 600 :=
by sorry

end NUMINAMATH_CALUDE_line_arrangements_l1195_119573


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1195_119572

theorem complex_equation_solution (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1195_119572


namespace NUMINAMATH_CALUDE_largest_integer_with_four_digit_square_l1195_119520

theorem largest_integer_with_four_digit_square : ∃ N : ℕ, 
  (∀ n : ℕ, n^2 ≥ 10000 → N ≤ n) ∧ 
  (N^2 < 10000) ∧
  (N^2 ≥ 1000) ∧
  N = 99 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_with_four_digit_square_l1195_119520


namespace NUMINAMATH_CALUDE_intersection_distance_implies_a_value_l1195_119531

-- Define the curve C
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve_C a x y ∧ line_l x y}

-- Theorem statement
theorem intersection_distance_implies_a_value (a : ℝ) :
  (∃ (A B : ℝ × ℝ), A ∈ intersection_points a ∧ B ∈ intersection_points a ∧ 
    A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 10) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_implies_a_value_l1195_119531


namespace NUMINAMATH_CALUDE_amaya_total_marks_l1195_119501

/-- Represents the marks scored in different subjects -/
structure Marks where
  music : ℕ
  socialStudies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total marks across all subjects -/
def totalMarks (m : Marks) : ℕ := m.music + m.socialStudies + m.arts + m.maths

/-- Theorem stating the total marks Amaya scored given the conditions -/
theorem amaya_total_marks :
  ∀ (m : Marks),
    m.music = 70 →
    m.socialStudies = m.music + 10 →
    m.maths = m.arts - 20 →
    m.maths = (9 : ℕ) * m.arts / 10 →
    totalMarks m = 530 := by
  sorry

#check amaya_total_marks

end NUMINAMATH_CALUDE_amaya_total_marks_l1195_119501


namespace NUMINAMATH_CALUDE_fraction_problem_l1195_119532

theorem fraction_problem (f : ℚ) : f * 12 + 5 = 11 → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1195_119532


namespace NUMINAMATH_CALUDE_black_fraction_after_changes_l1195_119576

/-- Represents the fraction of the triangle that remains black after each change. -/
def black_fraction_after_change : ℚ := 8/9

/-- Represents the fraction of the triangle that is always black (the central triangle). -/
def always_black_fraction : ℚ := 1/9

/-- Represents the number of changes applied to the triangle. -/
def num_changes : ℕ := 4

/-- Theorem stating the fractional part of the original area that remains black after the changes. -/
theorem black_fraction_after_changes :
  (black_fraction_after_change ^ num_changes) * (1 - always_black_fraction) + always_black_fraction = 39329/59049 := by
  sorry

end NUMINAMATH_CALUDE_black_fraction_after_changes_l1195_119576


namespace NUMINAMATH_CALUDE_students_on_right_side_l1195_119556

theorem students_on_right_side (total : ℕ) (left : ℕ) (right : ℕ) : 
  total = 63 → left = 36 → right = total - left → right = 27 := by
  sorry

end NUMINAMATH_CALUDE_students_on_right_side_l1195_119556


namespace NUMINAMATH_CALUDE_lcm_of_16_and_24_l1195_119510

theorem lcm_of_16_and_24 :
  let n : ℕ := 16
  let m : ℕ := 24
  let g : ℕ := 8
  Nat.gcd n m = g →
  Nat.lcm n m = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_16_and_24_l1195_119510


namespace NUMINAMATH_CALUDE_total_squares_6x6_grid_l1195_119562

/-- The number of squares of a given size in a grid --/
def count_squares (grid_size : ℕ) (square_size : ℕ) : ℕ :=
  (grid_size - square_size + 1) ^ 2

/-- The total number of squares in a 6x6 grid --/
theorem total_squares_6x6_grid :
  let grid_size := 6
  let square_sizes := [1, 2, 3, 4]
  (square_sizes.map (count_squares grid_size)).sum = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_6x6_grid_l1195_119562


namespace NUMINAMATH_CALUDE_set_union_problem_l1195_119563

theorem set_union_problem (a b : ℝ) : 
  let A : Set ℝ := {3, 2^a}
  let B : Set ℝ := {a, b}
  (A ∩ B = {2}) → (A ∪ B = {1, 2, 3}) := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1195_119563


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_negation_greater_than_neg_200_l1195_119505

theorem largest_multiple_of_8_negation_greater_than_neg_200 :
  ∀ n : ℤ, (n % 8 = 0 ∧ -n > -200) → n ≤ 192 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_negation_greater_than_neg_200_l1195_119505


namespace NUMINAMATH_CALUDE_sum_xyz_equals_zero_l1195_119587

theorem sum_xyz_equals_zero 
  (x y z a b c : ℝ) 
  (eq1 : x + y - z = a - b)
  (eq2 : x - y + z = b - c)
  (eq3 : -x + y + z = c - a) : 
  x + y + z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_zero_l1195_119587


namespace NUMINAMATH_CALUDE_f_properties_l1195_119515

-- Define the function f(x) = x|x - 2|
def f (x : ℝ) : ℝ := x * abs (x - 2)

-- Theorem for the monotonicity intervals and inequality solution
theorem f_properties :
  (∀ x y, x ≤ y ∧ y ≤ 1 → f x ≤ f y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f x ≤ f y) ∧
  (∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f y ≤ f x) ∧
  (∀ x, f x < 3 ↔ x < 3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1195_119515


namespace NUMINAMATH_CALUDE_sector_central_angle_l1195_119590

/-- Given a sector with radius 10 and area 50π/3, its central angle is π/3. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (h1 : r = 10) (h2 : S = 50 * Real.pi / 3) :
  S = 1/2 * r^2 * (Real.pi/3) := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1195_119590


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1195_119527

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  isArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1195_119527


namespace NUMINAMATH_CALUDE_units_digit_problem_l1195_119507

theorem units_digit_problem : (25^3 + 17^3) * 12^2 % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1195_119507


namespace NUMINAMATH_CALUDE_min_digit_sum_of_sum_l1195_119597

/-- Two-digit number type -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n ≤ 99 }

/-- Function to get the digits of a natural number -/
def digits (n : ℕ) : List ℕ := sorry

/-- Function to sum the digits of a natural number -/
def digitSum (n : ℕ) : ℕ := (digits n).sum

/-- Predicate to check if two two-digit numbers have exactly one common digit -/
def hasOneCommonDigit (a b : TwoDigitNumber) : Prop := sorry

/-- Theorem: The smallest possible digit sum of S, where S is the sum of two two-digit numbers
    with exactly one common digit, and S is a three-digit number, is 2. -/
theorem min_digit_sum_of_sum (a b : TwoDigitNumber) 
  (h1 : hasOneCommonDigit a b) 
  (h2 : a.val + b.val ≥ 100 ∧ a.val + b.val ≤ 999) : 
  ∃ (S : ℕ), S = a.val + b.val ∧ digitSum S = 2 ∧ 
  ∀ (T : ℕ), T = a.val + b.val → digitSum T ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_digit_sum_of_sum_l1195_119597


namespace NUMINAMATH_CALUDE_wife_account_percentage_l1195_119503

def total_income : ℝ := 200000
def children_count : ℕ := 3
def children_percentage : ℝ := 0.15
def orphan_house_percentage : ℝ := 0.05
def final_amount : ℝ := 40000

theorem wife_account_percentage :
  let children_total := children_count * children_percentage * total_income
  let remaining_after_children := total_income - children_total
  let orphan_house_amount := orphan_house_percentage * remaining_after_children
  let remaining_after_orphan := remaining_after_children - orphan_house_amount
  let wife_amount := remaining_after_orphan - final_amount
  (wife_amount / total_income) * 100 = 32.25 := by sorry

end NUMINAMATH_CALUDE_wife_account_percentage_l1195_119503


namespace NUMINAMATH_CALUDE_pole_intersection_height_l1195_119559

/-- Given two poles with heights 30 and 60 units, placed 50 units apart,
    the height of the intersection of the lines joining the top of each pole
    to the foot of the opposite pole is 20 units. -/
theorem pole_intersection_height :
  let h₁ : ℝ := 30  -- Height of the first pole
  let h₂ : ℝ := 60  -- Height of the second pole
  let d : ℝ := 50   -- Distance between the poles
  let m₁ : ℝ := (0 - h₁) / d  -- Slope of the first line
  let m₂ : ℝ := (0 - h₂) / (-d)  -- Slope of the second line
  let x : ℝ := (h₁ - 0) / (m₂ - m₁)  -- x-coordinate of intersection
  let y : ℝ := m₁ * x + h₁  -- y-coordinate of intersection
  y = 20 := by sorry

end NUMINAMATH_CALUDE_pole_intersection_height_l1195_119559


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l1195_119594

theorem same_terminal_side_angle :
  ∃ α : ℝ, 0 ≤ α ∧ α < 360 ∧ ∃ k : ℤ, α = k * 360 - 30 ∧ α = 330 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l1195_119594


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1195_119589

theorem max_y_coordinate_sin_3theta (θ : Real) :
  let r := λ θ : Real => Real.sin (3 * θ)
  let y := λ θ : Real => r θ * Real.sin θ
  ∃ (max_y : Real), max_y = 9/64 ∧ ∀ θ', y θ' ≤ max_y := by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l1195_119589


namespace NUMINAMATH_CALUDE_money_distribution_l1195_119574

theorem money_distribution (a b c : ℤ) 
  (total : a + b + c = 500)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 310) :
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1195_119574


namespace NUMINAMATH_CALUDE_N_mod_52_l1195_119550

/-- The number formed by concatenating integers from 1 to 51 -/
def N : ℕ := sorry

/-- The remainder when N is divided by 52 -/
def remainder : ℕ := N % 52

theorem N_mod_52 : remainder = 13 := by sorry

end NUMINAMATH_CALUDE_N_mod_52_l1195_119550


namespace NUMINAMATH_CALUDE_smallest_product_is_zero_l1195_119568

def S : Set Int := {-10, -6, 0, 2, 5}

theorem smallest_product_is_zero :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y = 0 ∧ 
  ∀ (a b : Int), a ∈ S → b ∈ S → x * y ≤ a * b :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_is_zero_l1195_119568


namespace NUMINAMATH_CALUDE_two_parts_divisibility_l1195_119514

theorem two_parts_divisibility (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ 13 * x + 17 * y = 283 → 
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a + b = 283 ∧ 13 ∣ a ∧ 17 ∣ b :=
by sorry

end NUMINAMATH_CALUDE_two_parts_divisibility_l1195_119514


namespace NUMINAMATH_CALUDE_nails_for_smaller_planks_eq_eight_l1195_119577

/-- The number of large planks used for the walls -/
def large_planks : ℕ := 13

/-- The number of nails needed for each large plank -/
def nails_per_plank : ℕ := 17

/-- The total number of nails needed for the house wall -/
def total_nails : ℕ := 229

/-- The number of nails needed for smaller planks -/
def nails_for_smaller_planks : ℕ := total_nails - (large_planks * nails_per_plank)

theorem nails_for_smaller_planks_eq_eight :
  nails_for_smaller_planks = 8 := by
  sorry

end NUMINAMATH_CALUDE_nails_for_smaller_planks_eq_eight_l1195_119577


namespace NUMINAMATH_CALUDE_lambda_5_lower_bound_l1195_119530

/-- The ratio of the longest distance to the shortest distance for n points in a plane -/
def lambda (n : ℕ) : ℝ := sorry

/-- Theorem: For 5 points in a plane, the ratio of the longest distance to the shortest distance
    is greater than or equal to 2 sin 54° -/
theorem lambda_5_lower_bound : lambda 5 ≥ 2 * Real.sin (54 * π / 180) := by sorry

end NUMINAMATH_CALUDE_lambda_5_lower_bound_l1195_119530


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l1195_119554

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l1195_119554


namespace NUMINAMATH_CALUDE_radio_survey_males_not_listening_l1195_119586

theorem radio_survey_males_not_listening (males_listening : ℕ) 
  (females_not_listening : ℕ) (total_listening : ℕ) (total_not_listening : ℕ) 
  (h1 : males_listening = 70)
  (h2 : females_not_listening = 110)
  (h3 : total_listening = 145)
  (h4 : total_not_listening = 160) :
  total_not_listening - females_not_listening = 50 :=
by
  sorry

#check radio_survey_males_not_listening

end NUMINAMATH_CALUDE_radio_survey_males_not_listening_l1195_119586


namespace NUMINAMATH_CALUDE_sum_of_factorials_last_two_digits_l1195_119541

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def isExcluded (n : ℕ) : Bool := n % 3 == 0 && n % 5 == 0

def sumOfFactorials : ℕ := 
  (List.range 100).foldl (λ acc n => 
    if !isExcluded (n + 1) then 
      (acc + lastTwoDigits (factorial (n + 1))) % 100 
    else 
      acc
  ) 0

theorem sum_of_factorials_last_two_digits : 
  sumOfFactorials = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_factorials_last_two_digits_l1195_119541


namespace NUMINAMATH_CALUDE_seed_packet_combinations_l1195_119546

/-- Represents the cost of a sunflower seed packet -/
def sunflower_cost : ℕ := 4

/-- Represents the cost of a lavender seed packet -/
def lavender_cost : ℕ := 1

/-- Represents the cost of a marigold seed packet -/
def marigold_cost : ℕ := 3

/-- Represents the total budget -/
def total_budget : ℕ := 60

/-- Counts the number of non-negative integer solutions to the equation -/
def count_solutions : ℕ := sorry

/-- Theorem stating that there are exactly 72 different combinations of seed packets -/
theorem seed_packet_combinations : count_solutions = 72 := by sorry

end NUMINAMATH_CALUDE_seed_packet_combinations_l1195_119546


namespace NUMINAMATH_CALUDE_divisibility_of_2_pow_62_plus_1_l1195_119504

theorem divisibility_of_2_pow_62_plus_1 :
  ∃ k : ℕ, 2^62 + 1 = k * (2^31 + 2^16 + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_2_pow_62_plus_1_l1195_119504


namespace NUMINAMATH_CALUDE_equation_solutions_l1195_119529

theorem equation_solutions : ∃! (s : Set ℝ), 
  (∀ x ∈ s, (x - 4)^4 + (x - 6)^4 = 16) ∧
  (s = {4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1195_119529


namespace NUMINAMATH_CALUDE_average_study_time_difference_l1195_119569

def daily_differences : List ℤ := [15, -5, 25, 35, -15, 10, 20]

def days_in_week : ℕ := 7

theorem average_study_time_difference : 
  (daily_differences.sum : ℚ) / days_in_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l1195_119569


namespace NUMINAMATH_CALUDE_min_abs_sum_squared_matrix_l1195_119595

-- Define the matrix type
def Matrix2x2 (α : Type) := Fin 2 → Fin 2 → α

-- Define the matrix multiplication
def matMul (A B : Matrix2x2 ℤ) : Matrix2x2 ℤ :=
  λ i j => (Finset.univ.sum λ k => A i k * B k j)

-- Define the identity matrix
def identityMatrix : Matrix2x2 ℤ :=
  λ i j => if i = j then 9 else 0

-- Define the absolute value sum
def absSum (a b c d : ℤ) : ℤ :=
  |a| + |b| + |c| + |d|

theorem min_abs_sum_squared_matrix :
  ∃ (a b c d : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (matMul (λ i j => match i, j with
      | 0, 0 => a
      | 0, 1 => b
      | 1, 0 => c
      | 1, 1 => d) (λ i j => match i, j with
      | 0, 0 => a
      | 0, 1 => b
      | 1, 0 => c
      | 1, 1 => d)) = identityMatrix ∧
    (∀ (a' b' c' d' : ℤ),
      a' ≠ 0 → b' ≠ 0 → c' ≠ 0 → d' ≠ 0 →
      (matMul (λ i j => match i, j with
        | 0, 0 => a'
        | 0, 1 => b'
        | 1, 0 => c'
        | 1, 1 => d') (λ i j => match i, j with
        | 0, 0 => a'
        | 0, 1 => b'
        | 1, 0 => c'
        | 1, 1 => d')) = identityMatrix →
      absSum a b c d ≤ absSum a' b' c' d') ∧
    absSum a b c d = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_min_abs_sum_squared_matrix_l1195_119595


namespace NUMINAMATH_CALUDE_organization_size_l1195_119526

/-- Represents a committee in the organization -/
def Committee := Fin 6

/-- Represents a member of the organization -/
structure Member where
  committees : Finset Committee
  member_in_three : committees.card = 3

/-- The organization with its members -/
structure Organization where
  members : Finset Member
  all_triples_covered : ∀ (c1 c2 c3 : Committee), c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 →
    ∃! m : Member, m ∈ members ∧ c1 ∈ m.committees ∧ c2 ∈ m.committees ∧ c3 ∈ m.committees

theorem organization_size (org : Organization) : org.members.card = 20 := by
  sorry

end NUMINAMATH_CALUDE_organization_size_l1195_119526
