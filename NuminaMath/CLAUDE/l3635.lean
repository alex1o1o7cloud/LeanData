import Mathlib

namespace batsman_average_after_25th_innings_l3635_363597

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: A batsman's average after 25 innings -/
theorem batsman_average_after_25th_innings 
  (stats : BatsmanStats)
  (h1 : stats.innings = 24)
  (h2 : newAverage stats 80 = stats.average + 3)
  : newAverage stats 80 = 8 := by
  sorry


end batsman_average_after_25th_innings_l3635_363597


namespace banana_arrangements_eq_60_l3635_363559

def banana_arrangements : ℕ :=
  let total_letters : ℕ := 6
  let b_count : ℕ := 1
  let n_count : ℕ := 2
  let a_count : ℕ := 3
  Nat.factorial total_letters / (Nat.factorial b_count * Nat.factorial n_count * Nat.factorial a_count)

theorem banana_arrangements_eq_60 : banana_arrangements = 60 := by
  sorry

end banana_arrangements_eq_60_l3635_363559


namespace coefficient_a3b3_is_1400_l3635_363515

/-- The coefficient of a^3b^3 in (a+b)^6(c+1/c)^8 -/
def coefficient_a3b3 : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 8 4)

/-- Theorem: The coefficient of a^3b^3 in (a+b)^6(c+1/c)^8 is 1400 -/
theorem coefficient_a3b3_is_1400 : coefficient_a3b3 = 1400 := by
  sorry

end coefficient_a3b3_is_1400_l3635_363515


namespace f_2012_equals_2_l3635_363545

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2012_equals_2 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 3) = -f x)
  (h_f_1 : f 1 = 2) : 
  f 2012 = 2 := by
  sorry

end f_2012_equals_2_l3635_363545


namespace complement_intersection_theorem_l3635_363521

def I : Set ℕ := Set.univ
def A : Set ℕ := {1,2,3,4,5,6}
def B : Set ℕ := {2,3,5}

theorem complement_intersection_theorem :
  (I \ B) ∩ A = {1,4,6} := by sorry

end complement_intersection_theorem_l3635_363521


namespace valid_distributions_count_l3635_363580

/-- Represents a triangular array of 8 rows -/
def TriangularArray := Fin 8 → Fin 8 → ℕ

/-- The bottom row of the triangular array -/
def BottomRow := Fin 8 → Fin 2

/-- Checks if a number is a multiple of 5 -/
def IsMultipleOf5 (n : ℕ) : Prop := ∃ k, n = 5 * k

/-- Calculates the value of a square based on the two squares below it -/
def CalculateSquareValue (arr : TriangularArray) (row : Fin 8) (col : Fin 8) : ℕ :=
  if row = 0 then arr 0 col
  else arr (row - 1) col + arr (row - 1) (col + 1)

/-- Builds the triangular array from the bottom row -/
def BuildArray (bottom : BottomRow) : TriangularArray :=
  sorry

/-- Counts the number of valid bottom row distributions -/
def CountValidDistributions : ℕ :=
  sorry

/-- The main theorem stating that the count of valid distributions is 32 -/
theorem valid_distributions_count :
  CountValidDistributions = 32 :=
sorry

end valid_distributions_count_l3635_363580


namespace little_john_height_l3635_363572

/-- Conversion factor from centimeters to meters -/
def cm_to_m : ℝ := 0.01

/-- Conversion factor from millimeters to meters -/
def mm_to_m : ℝ := 0.001

/-- Little John's height in meters, centimeters, and millimeters -/
def height_m : ℝ := 2
def height_cm : ℝ := 8
def height_mm : ℝ := 3

/-- Theorem stating that Little John's height in meters is 2.083 -/
theorem little_john_height : 
  height_m + height_cm * cm_to_m + height_mm * mm_to_m = 2.083 := by
  sorry

end little_john_height_l3635_363572


namespace largest_common_term_l3635_363555

theorem largest_common_term (n m : ℕ) : 
  163 = 3 + 8 * n ∧ 
  163 = 5 + 9 * m ∧ 
  163 ≤ 200 ∧ 
  ∀ k, k > 163 → k ≤ 200 → (k - 3) % 8 ≠ 0 ∨ (k - 5) % 9 ≠ 0 :=
by sorry

end largest_common_term_l3635_363555


namespace divisibility_relation_l3635_363567

theorem divisibility_relation :
  (∀ n : ℤ, n % 6 = 0 → n % 2 = 0) ∧
  (∃ n : ℤ, n % 2 = 0 ∧ n % 6 ≠ 0) := by
  sorry

end divisibility_relation_l3635_363567


namespace fourth_person_win_prob_is_one_thirtieth_l3635_363518

/-- Represents the probability of winning for the fourth person in a
    coin-flipping game with four players where the first to get heads wins. -/
def fourth_person_win_probability : ℚ := 1 / 30

/-- The probability of getting tails on a fair coin flip. -/
def prob_tails : ℚ := 1 / 2

/-- The number of players in the game. -/
def num_players : ℕ := 4

/-- Theorem stating that the probability of the fourth person winning
    in a coin-flipping game with four players is 1/30. -/
theorem fourth_person_win_prob_is_one_thirtieth :
  fourth_person_win_probability = 
    (prob_tails ^ num_players) / (1 - prob_tails ^ num_players) :=
sorry

end fourth_person_win_prob_is_one_thirtieth_l3635_363518


namespace line_through_M_and_origin_parallel_line_perpendicular_line_main_theorem_l3635_363537

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3*x + 4*y + 5 = 0
def l₂ (x y : ℝ) : Prop := 2*x - 3*y - 8 = 0
def l₃ (x y : ℝ) : Prop := 2*x + y + 5 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, -2)

-- Theorem for the line passing through M and the origin
theorem line_through_M_and_origin :
  ∃ (k : ℝ), ∀ (x y : ℝ), l₁ x y ∧ l₂ x y → (y = k * x) ∧ k = -2 :=
sorry

-- Theorem for the parallel line
theorem parallel_line :
  ∃ (t : ℝ), ∀ (x y : ℝ), l₁ (M.1) (M.2) ∧ l₂ (M.1) (M.2) →
    (2*x + y + t = 0) ∧ t = 0 :=
sorry

-- Theorem for the perpendicular line
theorem perpendicular_line :
  ∃ (s : ℝ), ∀ (x y : ℝ), l₁ (M.1) (M.2) ∧ l₂ (M.1) (M.2) →
    (x - 2*y + s = 0) ∧ s = -5 :=
sorry

-- Main theorem combining all conditions
theorem main_theorem :
  (∀ (x y : ℝ), l₁ x y ∧ l₂ x y → 2*x + y = 0) ∧
  (∀ (x y : ℝ), l₁ x y ∧ l₂ x y → x - 2*y - 5 = 0) :=
sorry

end line_through_M_and_origin_parallel_line_perpendicular_line_main_theorem_l3635_363537


namespace coefficient_x_cubed_in_expansion_l3635_363588

theorem coefficient_x_cubed_in_expansion : 
  let expansion := (fun x => (2 * x + 1) * (x - 1)^5)
  ∃ a b c d e f, ∀ x, 
    expansion x = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f ∧ 
    c = -10 :=
by sorry

end coefficient_x_cubed_in_expansion_l3635_363588


namespace inverse_mod_78_l3635_363532

theorem inverse_mod_78 (h : (7⁻¹ : ZMod 78) = 55) : (49⁻¹ : ZMod 78) = 61 := by
  sorry

end inverse_mod_78_l3635_363532


namespace darius_drove_679_miles_l3635_363507

/-- The number of miles Julia drove -/
def julia_miles : ℕ := 998

/-- The total number of miles Darius and Julia drove -/
def total_miles : ℕ := 1677

/-- The number of miles Darius drove -/
def darius_miles : ℕ := total_miles - julia_miles

theorem darius_drove_679_miles : darius_miles = 679 := by sorry

end darius_drove_679_miles_l3635_363507


namespace walking_speed_calculation_l3635_363540

/-- Given a person who jogs and walks, this theorem proves their walking speed. -/
theorem walking_speed_calculation 
  (jog_speed : ℝ) 
  (jog_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : jog_speed = 2) 
  (h2 : jog_distance = 3) 
  (h3 : total_time = 3) : 
  jog_distance / (total_time - jog_distance / jog_speed) = 2 := by
  sorry

end walking_speed_calculation_l3635_363540


namespace alpha_equals_five_l3635_363533

-- Define the grid as a 3x3 matrix of natural numbers
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define a predicate to check if a number is a non-zero digit
def IsNonZeroDigit (n : Nat) : Prop := 0 < n ∧ n ≤ 9

-- Define a predicate to check if all elements in the grid are distinct
def AllDistinct (g : Grid) : Prop :=
  ∀ i j k l, (i, j) ≠ (k, l) → g i j ≠ g k l

-- Define a predicate to check if all elements in the grid are non-zero digits
def AllNonZeroDigits (g : Grid) : Prop :=
  ∀ i j, IsNonZeroDigit (g i j)

-- Define a predicate to check if all horizontal expressions are correct
def HorizontalExpressionsCorrect (g : Grid) : Prop :=
  (g 0 0 + g 0 1 = g 0 2) ∧
  (g 1 0 - g 1 1 = g 1 2) ∧
  (g 2 0 * g 2 1 = g 2 2)

-- Define a predicate to check if all vertical expressions are correct
def VerticalExpressionsCorrect (g : Grid) : Prop :=
  (g 0 0 + g 1 0 = g 2 0) ∧
  (g 0 1 - g 1 1 = g 2 1) ∧
  (g 0 2 * g 1 2 = g 2 2)

-- Main theorem
theorem alpha_equals_five (g : Grid) (α : Nat)
  (h1 : AllDistinct g)
  (h2 : AllNonZeroDigits g)
  (h3 : HorizontalExpressionsCorrect g)
  (h4 : VerticalExpressionsCorrect g)
  (h5 : ∃ i j, g i j = α) :
  α = 5 := by
  sorry

end alpha_equals_five_l3635_363533


namespace cubic_polynomial_theorem_l3635_363504

def is_monic_cubic (q : ℝ → ℂ) : Prop :=
  ∃ a b c : ℝ, ∀ x, q x = x^3 + a*x^2 + b*x + c

theorem cubic_polynomial_theorem (q : ℝ → ℂ) 
  (h_monic : is_monic_cubic q)
  (h_root : q (2 - 3*I) = 0)
  (h_const : q 0 = -72) :
  ∀ x, q x = x^3 - (100/13)*x^2 + (236/13)*x - 936/13 :=
by sorry

end cubic_polynomial_theorem_l3635_363504


namespace family_heights_l3635_363520

/-- Represents the heights of family members and proves statements about their relationships -/
theorem family_heights (binbin_height mother_height : Real) 
  (father_taller_by : Real) (h1 : binbin_height = 1.46) 
  (h2 : father_taller_by = 0.32) (h3 : mother_height = 1.5) : 
  (binbin_height + father_taller_by = 1.78) ∧ 
  ((binbin_height + father_taller_by) - mother_height = 0.28) := by
  sorry

#check family_heights

end family_heights_l3635_363520


namespace star_operations_l3635_363516

/-- The ☆ operation for rational numbers -/
def star (a b : ℚ) : ℚ := a * b - a + b

/-- Theorem stating the results of the given operations -/
theorem star_operations :
  (star 2 (-3) = -11) ∧ (star (-2) (star 1 3) = -3) := by
  sorry

end star_operations_l3635_363516


namespace ellipse_properties_l3635_363550

/-- An ellipse with focal length 2 passing through the point (3/2, √6) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  focal_length : a^2 - b^2 = 1
  passes_through : (3/2)^2 / a^2 + 6 / b^2 = 1

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2 / 9 + y^2 / 8 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The trajectory of point E -/
def trajectory_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x ≠ 3 ∧ x ≠ -3 → (x^2 / 9 - y^2 / 8 = 1 ↔
    ∃ x₁ y₁ : ℝ, x₁^2 / e.a^2 + y₁^2 / e.b^2 = 1 ∧ x₁ ≠ 0 ∧ |x₁| < e.a ∧
      y / y₁ = (x + e.a) / (x₁ + e.a) ∧
      y / (-y₁) = (x - e.a) / (x₁ - e.a))

/-- The main theorem to be proved -/
theorem ellipse_properties (e : Ellipse) :
  standard_equation e ∧ trajectory_equation e :=
sorry

end ellipse_properties_l3635_363550


namespace min_tiles_for_region_l3635_363571

/-- The number of tiles needed to cover a rectangular region -/
def tiles_needed (tile_length : ℕ) (tile_width : ℕ) (region_length : ℕ) (region_width : ℕ) : ℕ :=
  let region_area := region_length * region_width
  let tile_area := tile_length * tile_width
  (region_area + tile_area - 1) / tile_area

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

/-- Theorem stating the minimum number of tiles needed to cover the given region -/
theorem min_tiles_for_region : 
  tiles_needed 5 6 (3 * feet_to_inches) (4 * feet_to_inches) = 58 := by
  sorry

#eval tiles_needed 5 6 (3 * feet_to_inches) (4 * feet_to_inches)

end min_tiles_for_region_l3635_363571


namespace largest_value_l3635_363564

def expr_a : ℕ := 3 + 1 + 2 + 8
def expr_b : ℕ := 3 * 1 + 2 + 8
def expr_c : ℕ := 3 + 1 * 2 + 8
def expr_d : ℕ := 3 + 1 + 2 * 8
def expr_e : ℕ := 3 * 1 * 2 * 8

theorem largest_value :
  expr_e ≥ expr_a ∧ 
  expr_e ≥ expr_b ∧ 
  expr_e ≥ expr_c ∧ 
  expr_e ≥ expr_d :=
by sorry

end largest_value_l3635_363564


namespace friends_for_games_only_l3635_363575

-- Define the variables
def movie : ℕ := 10
def picnic : ℕ := 20
def movie_and_picnic : ℕ := 4
def movie_and_games : ℕ := 2
def picnic_and_games : ℕ := 0
def all_three : ℕ := 2
def total_students : ℕ := 31

-- Theorem to prove
theorem friends_for_games_only : 
  ∃ (movie_only picnic_only games_only : ℕ),
    movie_only + picnic_only + games_only + movie_and_picnic + movie_and_games + picnic_and_games + all_three = total_students ∧
    movie_only + movie_and_picnic + movie_and_games + all_three = movie ∧
    picnic_only + movie_and_picnic + picnic_and_games + all_three = picnic ∧
    games_only = 1 := by
  sorry

end friends_for_games_only_l3635_363575


namespace oranges_thrown_away_l3635_363526

theorem oranges_thrown_away (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 40 → new = 21 → final = 36 → initial - (initial - final + new) = 25 := by
  sorry

end oranges_thrown_away_l3635_363526


namespace geometric_sum_first_8_terms_l3635_363544

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/2
  let n : ℕ := 8
  geometric_sum a r n = 85/128 := by
sorry

end geometric_sum_first_8_terms_l3635_363544


namespace mode_not_necessarily_same_l3635_363586

-- Define the number of shots
def num_shots : ℕ := 10

-- Define the average score for both persons
def average_score : ℝ := 8

-- Define the variances for person A and B
def variance_A : ℝ := 1.2
def variance_B : ℝ := 1.6

-- Define a type for a person's shooting results
structure ShootingResult where
  scores : List ℝ
  average : ℝ
  variance : ℝ

-- Define person A and B's shooting results
def person_A : ShootingResult := {
  scores := [], -- We don't know the actual scores
  average := average_score,
  variance := variance_A
}

def person_B : ShootingResult := {
  scores := [], -- We don't know the actual scores
  average := average_score,
  variance := variance_B
}

-- Theorem: It cannot be concluded that the mode of person A and B's scores must be the same
theorem mode_not_necessarily_same (A B : ShootingResult) 
  (h1 : A.scores.length = num_shots) 
  (h2 : B.scores.length = num_shots)
  (h3 : A.average = average_score) 
  (h4 : B.average = average_score)
  (h5 : A.variance = variance_A) 
  (h6 : B.variance = variance_B) : 
  ¬ (∀ (mode_A mode_B : ℝ), 
    (mode_A ∈ A.scores ∧ (∀ x ∈ A.scores, (A.scores.count mode_A) ≥ (A.scores.count x))) →
    (mode_B ∈ B.scores ∧ (∀ y ∈ B.scores, (B.scores.count mode_B) ≥ (B.scores.count y))) →
    mode_A = mode_B) :=
sorry

end mode_not_necessarily_same_l3635_363586


namespace inequality_proof_l3635_363578

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^6 - a^2 + 4) * (b^6 - b^2 + 4) * (c^6 - c^2 + 4) * (d^6 - d^2 + 4) ≥ (a + b + c + d)^4 := by
  sorry

end inequality_proof_l3635_363578


namespace shaded_area_in_square_with_triangles_l3635_363539

/-- The area of the shaded region in a square with two right-angle triangles -/
theorem shaded_area_in_square_with_triangles (square_side : ℝ) (triangle_leg : ℝ)
  (h_square : square_side = 40)
  (h_triangle : triangle_leg = 25) :
  square_side ^ 2 - 2 * (triangle_leg ^ 2 / 2) = 975 :=
by sorry

end shaded_area_in_square_with_triangles_l3635_363539


namespace sixth_number_in_sequence_l3635_363589

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n

theorem sixth_number_in_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_sum : a 2 + a 3 = 24) :
  a 6 = 128 := by
sorry

end sixth_number_in_sequence_l3635_363589


namespace derivative_of_sine_at_pi_sixth_l3635_363563

/-- Given f(x) = sin(2x + π/6), prove that f'(π/6) = 0 -/
theorem derivative_of_sine_at_pi_sixth (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (2 * x + π / 6)) :
  deriv f (π / 6) = 0 := by
  sorry

end derivative_of_sine_at_pi_sixth_l3635_363563


namespace complex_equation_solution_l3635_363500

theorem complex_equation_solution (x y : ℝ) : 
  (2*x - 1 : ℂ) + (y + 1 : ℂ) * I = (x - y : ℂ) - (x + y : ℂ) * I → x = 3 ∧ y = -2 :=
by sorry

end complex_equation_solution_l3635_363500


namespace polynomial_sum_simplification_l3635_363585

theorem polynomial_sum_simplification :
  ∀ x : ℝ, (2 * x^3 - 3 * x^2 + 5 * x - 6) + (5 * x^4 - 2 * x^3 - 4 * x^2 - x + 8) =
            5 * x^4 - 7 * x^2 + 4 * x + 2 := by
  sorry

end polynomial_sum_simplification_l3635_363585


namespace business_value_proof_l3635_363599

theorem business_value_proof (total_shares : ℚ) (man_shares : ℚ) (sold_fraction : ℚ) (sale_price : ℚ) :
  total_shares = 1 →
  man_shares = 1 / 3 →
  sold_fraction = 3 / 5 →
  sale_price = 2000 →
  (man_shares * sold_fraction * total_shares⁻¹) * (total_shares / (man_shares * sold_fraction)) * sale_price = 10000 :=
by sorry

end business_value_proof_l3635_363599


namespace misread_number_correction_l3635_363594

theorem misread_number_correction (n : ℕ) (incorrect_avg correct_avg misread_value : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 18)
  (h3 : correct_avg = 22)
  (h4 : misread_value = 26) :
  ∃ (actual_value : ℚ), 
    n * correct_avg = n * incorrect_avg - misread_value + actual_value ∧ 
    actual_value = 66 := by
  sorry

end misread_number_correction_l3635_363594


namespace leona_earnings_l3635_363510

/-- Given an hourly rate calculated from earning $24.75 for 3 hours,
    prove that the earnings for 5 hours at the same rate will be $41.25. -/
theorem leona_earnings (hourly_rate : ℝ) (h1 : hourly_rate * 3 = 24.75) :
  hourly_rate * 5 = 41.25 := by
  sorry

end leona_earnings_l3635_363510


namespace sum_and_count_theorem_l3635_363514

def sum_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_in_range (a b : ℕ) : ℕ := ((b - a) / 2) + 1

theorem sum_and_count_theorem :
  sum_range 60 80 + count_even_in_range 60 80 = 1481 := by
  sorry

end sum_and_count_theorem_l3635_363514


namespace sqrt_one_minus_sqrt_two_squared_l3635_363527

theorem sqrt_one_minus_sqrt_two_squared (h : 1 < Real.sqrt 2) :
  Real.sqrt ((1 - Real.sqrt 2) ^ 2) = Real.sqrt 2 - 1 := by
  sorry

end sqrt_one_minus_sqrt_two_squared_l3635_363527


namespace intersection_of_A_and_B_l3635_363543

def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

def B : Set ℝ := {x | |x - 1| > 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | (-1 < x ∧ x < 0) ∨ (2 < x ∧ x < 5)} :=
by sorry

end intersection_of_A_and_B_l3635_363543


namespace square_side_lengths_l3635_363591

theorem square_side_lengths (a b : ℕ) : 
  a > b → a ^ 2 - b ^ 2 = 2001 → a ∈ ({1001, 335, 55, 49} : Set ℕ) :=
by sorry

end square_side_lengths_l3635_363591


namespace fruit_eating_arrangements_l3635_363554

theorem fruit_eating_arrangements : 
  let total_fruits : ℕ := 4 + 2 + 1
  let apples : ℕ := 4
  let oranges : ℕ := 2
  let bananas : ℕ := 1
  (Nat.factorial total_fruits) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) = 105 := by
  sorry

end fruit_eating_arrangements_l3635_363554


namespace fraction_simplification_l3635_363577

theorem fraction_simplification :
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by
  sorry

end fraction_simplification_l3635_363577


namespace one_element_set_l3635_363548

def A (k : ℝ) : Set ℝ := {x | k * x^2 - 4 * x + 2 = 0}

theorem one_element_set (k : ℝ) :
  (∃! x, x ∈ A k) → (k = 0 ∧ A k = {1/2}) ∨ (k = 2 ∧ A k = {1}) := by
  sorry

end one_element_set_l3635_363548


namespace factor_t_squared_minus_64_l3635_363536

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_t_squared_minus_64_l3635_363536


namespace stratified_sampling_result_l3635_363592

/-- Calculates the number of students selected from a class in stratified sampling -/
def stratified_sample (class_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (class_size * sample_size) / total_size

/-- Represents the stratified sampling scenario -/
structure StratifiedSampling where
  class1_size : ℕ
  class2_size : ℕ
  total_sample_size : ℕ

/-- Theorem stating the result of the stratified sampling problem -/
theorem stratified_sampling_result (s : StratifiedSampling) 
  (h1 : s.class1_size = 36)
  (h2 : s.class2_size = 42)
  (h3 : s.total_sample_size = 13) :
  stratified_sample s.class2_size (s.class1_size + s.class2_size) s.total_sample_size = 7 := by
  sorry

#eval stratified_sample 42 (36 + 42) 13

end stratified_sampling_result_l3635_363592


namespace bike_rental_fixed_fee_bike_rental_fixed_fee_proof_l3635_363570

/-- The fixed fee for renting a bike, given the total cost formula and a specific rental case. -/
theorem bike_rental_fixed_fee : ℝ → Prop :=
  fun fixed_fee =>
    let total_cost := fun (hours : ℝ) => fixed_fee + 7 * hours
    total_cost 9 = 80 → fixed_fee = 17

/-- Proof of the bike rental fixed fee theorem -/
theorem bike_rental_fixed_fee_proof : bike_rental_fixed_fee 17 := by
  sorry

end bike_rental_fixed_fee_bike_rental_fixed_fee_proof_l3635_363570


namespace raft_travel_time_l3635_363556

/-- The number of days it takes for a ship to travel from Chongqing to Shanghai -/
def ship_cq_to_sh : ℝ := 5

/-- The number of days it takes for a ship to travel from Shanghai to Chongqing -/
def ship_sh_to_cq : ℝ := 7

/-- The number of days it takes for a raft to drift from Chongqing to Shanghai -/
def raft_cq_to_sh : ℝ := 35

/-- Theorem stating that the raft travel time satisfies the given conditions -/
theorem raft_travel_time :
  1 / ship_cq_to_sh - 1 / raft_cq_to_sh = 1 / ship_sh_to_cq + 1 / raft_cq_to_sh :=
by sorry

end raft_travel_time_l3635_363556


namespace expression_evaluation_l3635_363562

theorem expression_evaluation : (3 / 2) * 12 - 3 = 15 := by
  sorry

end expression_evaluation_l3635_363562


namespace permutation_identities_l3635_363525

def A (n m : ℕ) : ℕ := (n :: List.range m).prod

theorem permutation_identities :
  (∀ n m : ℕ, A (n + 1) (m + 1) - A n m = n^2 * A (n - 1) (m - 1)) ∧
  (∀ n m : ℕ, A n m = n * A (n - 1) (m - 1)) := by
  sorry

end permutation_identities_l3635_363525


namespace conference_left_handed_fraction_l3635_363595

/-- Represents the fraction of left-handed participants in a conference -/
def left_handed_fraction (total : ℕ) (red : ℕ) (blue : ℕ) (red_left : ℚ) (blue_left : ℚ) : ℚ :=
  (red_left * red + blue_left * blue) / total

/-- Theorem stating the fraction of left-handed participants in the conference -/
theorem conference_left_handed_fraction :
  ∀ (total : ℕ) (red : ℕ) (blue : ℕ),
  total > 0 →
  red + blue = total →
  red = blue →
  left_handed_fraction total red blue (1/3) (2/3) = 1/2 := by
sorry

end conference_left_handed_fraction_l3635_363595


namespace max_min_sum_l3635_363584

noncomputable def y (x : ℝ) : ℝ :=
  (2 * Real.sin x ^ 2 + Real.sin (3 * x / 2) - 4) / (Real.sin x ^ 2 + 2 * Real.cos x ^ 2)

theorem max_min_sum (M m : ℝ) 
  (hM : ∀ x, y x ≤ M) 
  (hm : ∀ x, m ≤ y x) 
  (hM_exists : ∃ x, y x = M) 
  (hm_exists : ∃ x, y x = m) : 
  M + m = -4 := by
  sorry

end max_min_sum_l3635_363584


namespace cube_volume_ratio_l3635_363542

theorem cube_volume_ratio : 
  let cube1_edge_length : ℚ := 10  -- in inches
  let cube2_edge_length : ℚ := 5 * 12  -- 5 feet converted to inches
  let volume_ratio := (cube1_edge_length / cube2_edge_length) ^ 3
  volume_ratio = 1 / 216 := by
sorry

end cube_volume_ratio_l3635_363542


namespace square_root_equation_l3635_363519

theorem square_root_equation (a : ℝ) : Real.sqrt (a^2) = 3 → a = 3 ∨ a = -3 := by
  sorry

end square_root_equation_l3635_363519


namespace iphone_price_decrease_l3635_363561

def initial_price : ℝ := 1000
def first_month_decrease : ℝ := 0.1
def final_price : ℝ := 720

theorem iphone_price_decrease : 
  let price_after_first_month := initial_price * (1 - first_month_decrease)
  let second_month_decrease := (price_after_first_month - final_price) / price_after_first_month
  second_month_decrease = 0.2 := by
  sorry

end iphone_price_decrease_l3635_363561


namespace farmer_land_calculation_l3635_363509

/-- Proves that if 90% of a farmer's land is cleared, and 20% of the cleared land
    is planted with tomatoes covering 360 acres, then the total land owned by the
    farmer is 2000 acres. -/
theorem farmer_land_calculation (total_land : ℝ) (cleared_land : ℝ) (tomato_land : ℝ) :
  cleared_land = 0.9 * total_land →
  tomato_land = 0.2 * cleared_land →
  tomato_land = 360 →
  total_land = 2000 := by
  sorry

end farmer_land_calculation_l3635_363509


namespace base_10_to_base_12_250_l3635_363529

def base_12_digit (n : ℕ) : Char :=
  if n < 10 then Char.ofNat (n + 48)
  else if n = 10 then 'A'
  else 'B'

def to_base_12 (n : ℕ) : List Char :=
  if n < 12 then [base_12_digit n]
  else (to_base_12 (n / 12)) ++ [base_12_digit (n % 12)]

theorem base_10_to_base_12_250 :
  to_base_12 250 = ['1', 'A'] :=
sorry

end base_10_to_base_12_250_l3635_363529


namespace polynomial_expansion_l3635_363535

theorem polynomial_expansion (t : ℚ) : 
  (3*t^3 + 2*t^2 - 4*t + 3) * (-4*t^3 + 3*t - 5) = 
  -12*t^6 - 8*t^5 + 25*t^4 - 21*t^3 - 22*t^2 + 29*t - 15 := by
  sorry

end polynomial_expansion_l3635_363535


namespace sequence_roots_theorem_l3635_363547

theorem sequence_roots_theorem (b c : ℕ → ℝ) : 
  (∀ n : ℕ, n ≥ 1 → b n ≤ c n) → 
  (∀ n : ℕ, n ≥ 1 → (b (n + 1))^2 + (b n) * (b (n + 1)) + (c n) = 0 ∧ 
                     (c (n + 1))^2 + (b n) * (c (n + 1)) + (c n) = 0) →
  (∀ n : ℕ, n ≥ 1 → b n = 0 ∧ c n = 0) :=
by sorry

end sequence_roots_theorem_l3635_363547


namespace parallel_perpendicular_transitivity_l3635_363534

-- Define the space
variable (S : Type*) [MetricSpace S]

-- Define lines and planes
variable (Line Plane : Type*)

-- Define the lines m and n, and the plane α
variable (m n : Line) (α : Plane)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the distinct relation for lines
variable (distinct : Line → Line → Prop)

-- Theorem statement
theorem parallel_perpendicular_transitivity 
  (h_distinct : distinct m n)
  (h_parallel : parallel m n)
  (h_perpendicular : perpendicular n α) :
  perpendicular m α :=
sorry

end parallel_perpendicular_transitivity_l3635_363534


namespace banana_permutations_l3635_363538

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2

theorem banana_permutations :
  (word_length.factorial) / (a_count.factorial * n_count.factorial) = 60 := by
  sorry

end banana_permutations_l3635_363538


namespace sum_of_other_digits_l3635_363553

def is_form_76h4 (n : ℕ) : Prop :=
  ∃ h : ℕ, n = 7000 + 600 + 10 * h + 4

theorem sum_of_other_digits (n : ℕ) (h : ℕ) :
  is_form_76h4 n →
  h = 1 →
  n % 9 = 0 →
  (7 + 6 + 4 : ℕ) = 12 :=
by
  sorry

end sum_of_other_digits_l3635_363553


namespace complex_number_quadrant_l3635_363541

theorem complex_number_quadrant : 
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_quadrant_l3635_363541


namespace count_nondegenerate_triangles_l3635_363552

/-- A point in the integer grid -/
structure GridPoint where
  s : Nat
  t : Nat
  s_bound : s ≤ 4
  t_bound : t ≤ 4

/-- A triangle represented by three grid points -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.s - p1.s) * (p3.t - p1.t) = (p3.s - p1.s) * (p2.t - p1.t)

/-- Predicate to check if a triangle is nondegenerate -/
def nondegenerate (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all valid grid points -/
def gridPoints : Finset GridPoint :=
  sorry

/-- The set of all possible triangles formed by grid points -/
def allTriangles : Finset GridTriangle :=
  sorry

/-- The set of all nondegenerate triangles -/
def nondegenerateTriangles : Finset GridTriangle :=
  sorry

theorem count_nondegenerate_triangles :
  Finset.card nondegenerateTriangles = 2170 :=
sorry

end count_nondegenerate_triangles_l3635_363552


namespace sum_and_product_zero_l3635_363568

theorem sum_and_product_zero (a b : ℝ) 
  (h1 : 2*a + 2*b + a*b = 1) 
  (h2 : a + b + 3*a*b = -2) : 
  a + b + a*b = 0 := by
sorry

end sum_and_product_zero_l3635_363568


namespace cookie_price_l3635_363565

/-- Proves that the price of each cookie is $0.50 given the conditions of the basketball team's sales and purchases. -/
theorem cookie_price (cupcake_count : ℕ) (cupcake_price : ℚ) (cookie_count : ℕ) 
  (basketball_count : ℕ) (basketball_price : ℚ) (drink_count : ℕ) (drink_price : ℚ) :
  cupcake_count = 50 →
  cupcake_price = 2 →
  cookie_count = 40 →
  basketball_count = 2 →
  basketball_price = 40 →
  drink_count = 20 →
  drink_price = 2 →
  ∃ (cookie_price : ℚ),
    cupcake_count * cupcake_price + cookie_count * cookie_price = 
    basketball_count * basketball_price + drink_count * drink_price ∧
    cookie_price = 1/2 := by
  sorry

#check cookie_price

end cookie_price_l3635_363565


namespace max_overtakes_relay_race_l3635_363546

/-- Represents a relay race between two teams -/
structure RelayRace where
  num_runners : ℕ
  num_segments : ℕ
  runners_per_team : ℕ

/-- Represents the maximum number of overtakes in a relay race -/
def max_overtakes (race : RelayRace) : ℕ :=
  2 * (race.num_runners - 1)

/-- Theorem stating the maximum number of overtakes in the specific relay race scenario -/
theorem max_overtakes_relay_race :
  ∀ (race : RelayRace),
    race.num_runners = 20 →
    race.num_segments = 20 →
    race.runners_per_team = 20 →
    max_overtakes race = 38 := by
  sorry

end max_overtakes_relay_race_l3635_363546


namespace extended_quadrilateral_area_l3635_363579

/-- Represents a quadrilateral with extended sides --/
structure ExtendedQuadrilateral where
  -- Original quadrilateral sides
  WZ : ℝ
  WX : ℝ
  XY : ℝ
  YZ : ℝ
  -- Extended sides
  ZW' : ℝ
  XX' : ℝ
  YY' : ℝ
  Z'W : ℝ
  -- Area of original quadrilateral
  area : ℝ

/-- Theorem stating the area of the extended quadrilateral --/
theorem extended_quadrilateral_area 
  (q : ExtendedQuadrilateral) 
  (h1 : q.WZ = 10 ∧ q.ZW' = 10)
  (h2 : q.WX = 6 ∧ q.XX' = 6)
  (h3 : q.XY = 7 ∧ q.YY' = 7)
  (h4 : q.YZ = 12 ∧ q.Z'W = 12)
  (h5 : q.area = 15) :
  ∃ (area_extended : ℝ), area_extended = 45 := by
  sorry

end extended_quadrilateral_area_l3635_363579


namespace point_in_fourth_quadrant_l3635_363566

/-- The point P(1+m^2, -1) lies in the fourth quadrant for any real number m. -/
theorem point_in_fourth_quadrant (m : ℝ) : 
  let x : ℝ := 1 + m^2
  let y : ℝ := -1
  x > 0 ∧ y < 0 := by
sorry


end point_in_fourth_quadrant_l3635_363566


namespace trig_product_value_l3635_363576

theorem trig_product_value : 
  Real.sin (4/3 * Real.pi) * Real.cos (5/6 * Real.pi) * Real.tan (-4/3 * Real.pi) = -3 * Real.sqrt 3 / 4 := by
  sorry

end trig_product_value_l3635_363576


namespace four_line_angles_l3635_363513

/-- Given four lines on a plane with angles α, β, and γ between some of them,
    prove that the angles between the remaining pairs of lines are as stated. -/
theorem four_line_angles (α β γ : ℝ) 
  (h_α : α = 110)
  (h_β : β = 60)
  (h_γ : γ = 80) :
  ∃ x y z : ℝ, 
    x = α - γ ∧ 
    z = β - x ∧
    y = α - β ∧
    x = 30 ∧ 
    y = 50 ∧ 
    z = 30 :=
by sorry

end four_line_angles_l3635_363513


namespace percentage_female_on_duty_l3635_363505

def total_on_duty : ℕ := 200
def female_ratio_on_duty : ℚ := 1/2
def total_female_officers : ℕ := 1000

theorem percentage_female_on_duty :
  (female_ratio_on_duty * total_on_duty) / total_female_officers * 100 = 10 := by
  sorry

end percentage_female_on_duty_l3635_363505


namespace acid_dilution_l3635_363581

/-- Given m ounces of an m% acid solution, when x ounces of water are added,
    a new solution of (m-20)% concentration is formed. Assuming m > 25,
    prove that x = 20m / (m-20). -/
theorem acid_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m * m / 100 = (m - 20) / 100 * (m + x)) → x = 20 * m / (m - 20) := by
  sorry

end acid_dilution_l3635_363581


namespace triomino_corner_reachability_l3635_363551

/-- Represents an L-triomino on a board -/
structure Triomino where
  center : Nat × Nat
  leg1 : Nat × Nat
  leg2 : Nat × Nat

/-- Represents a board of size m × n -/
structure Board (m n : Nat) where
  triomino : Triomino

/-- Defines a valid initial position of the triomino -/
def initial_position (m n : Nat) : Board m n :=
  { triomino := { center := (0, 0), leg1 := (0, 1), leg2 := (1, 0) } }

/-- Defines a valid rotation of the triomino -/
def can_rotate (b : Board m n) : Prop :=
  ∃ new_position : Triomino, true  -- We assume any rotation is possible

/-- Defines if a triomino can reach the bottom right corner -/
def can_reach_corner (m n : Nat) : Prop :=
  ∃ final_position : Board m n, 
    final_position.triomino.center = (m - 1, n - 1)

/-- The main theorem to be proved -/
theorem triomino_corner_reachability (m n : Nat) :
  can_reach_corner m n ↔ m % 2 = 1 ∧ n % 2 = 1 := by sorry

end triomino_corner_reachability_l3635_363551


namespace emily_game_lives_l3635_363524

def game_lives (initial : ℕ) (lost : ℕ) (final : ℕ) : ℕ :=
  final - (initial - lost)

theorem emily_game_lives :
  game_lives 42 25 41 = 24 := by
  sorry

end emily_game_lives_l3635_363524


namespace ellipse_area_irrational_l3635_363596

/-- The area of an ellipse with rational semi-major and semi-minor axes is irrational -/
theorem ellipse_area_irrational (a b : ℚ) (h_a : a > 0) (h_b : b > 0) : 
  Irrational (Real.pi * (a * b)) := by
  sorry

end ellipse_area_irrational_l3635_363596


namespace quadratic_no_real_roots_l3635_363501

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 - x + 2 ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l3635_363501


namespace divide_fractions_and_mixed_number_l3635_363517

theorem divide_fractions_and_mixed_number :
  (5 : ℚ) / 6 / (1 + 3 / 9) = 5 / 8 := by
  sorry

end divide_fractions_and_mixed_number_l3635_363517


namespace prob_four_correct_zero_l3635_363557

/-- Represents the number of people and letters -/
def n : ℕ := 5

/-- The probability of exactly (n-1) people receiving their correct letter
    in a random distribution of n letters to n people -/
def prob_n_minus_one_correct (n : ℕ) : ℝ := 
  if n ≥ 2 then 0 else 1

/-- Theorem stating that the probability of exactly 4 out of 5 people
    receiving their correct letter is 0 -/
theorem prob_four_correct_zero : 
  prob_n_minus_one_correct n = 0 := by sorry

end prob_four_correct_zero_l3635_363557


namespace count_divisible_integers_l3635_363522

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 3)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (1764 : ℤ) ∣ (m^2 - 3) → m ∈ S) ∧
    S.card = 5 := by
  sorry

end count_divisible_integers_l3635_363522


namespace range_of_power_function_l3635_363590

/-- The range of f(x) = x^k + c on [0, ∞) is [c, ∞) when k > 0 -/
theorem range_of_power_function (k : ℝ) (c : ℝ) (h : k > 0) :
  Set.range (fun x : ℝ => x^k + c) = Set.Ici c :=
by sorry


end range_of_power_function_l3635_363590


namespace village_population_l3635_363560

theorem village_population (population : ℝ) : 
  (0.9 * population = 45000) → population = 50000 := by
  sorry

end village_population_l3635_363560


namespace zero_in_M_l3635_363593

def M : Set ℝ := {x | x^2 - 3 ≤ 0}

theorem zero_in_M : (0 : ℝ) ∈ M := by
  sorry

end zero_in_M_l3635_363593


namespace volume_ratio_is_twenty_l3635_363582

-- Define the dimensions of the shapes
def cube_edge : ℝ := 1  -- 1 meter
def cuboid_width : ℝ := 0.5  -- 50 cm in meters
def cuboid_length : ℝ := 0.5  -- 50 cm in meters
def cuboid_height : ℝ := 0.2  -- 20 cm in meters

-- Define the volume functions
def cube_volume (edge : ℝ) : ℝ := edge ^ 3
def cuboid_volume (width length height : ℝ) : ℝ := width * length * height

-- Theorem statement
theorem volume_ratio_is_twenty :
  (cube_volume cube_edge) / (cuboid_volume cuboid_width cuboid_length cuboid_height) = 20 := by
  sorry

end volume_ratio_is_twenty_l3635_363582


namespace wario_expected_wide_right_misses_l3635_363502

/-- Represents a football kicker's field goal statistics -/
structure KickerStats where
  totalAttempts : ℕ
  missRate : ℚ
  missTypes : Fin 4 → ℚ
  underFortyYardsSuccessRate : ℚ

/-- Represents the conditions for a specific game -/
structure GameConditions where
  attempts : ℕ
  attemptsUnderForty : ℕ
  windSpeed : ℚ

/-- Calculates the expected number of wide right misses for a kicker in a game -/
def expectedWideRightMisses (stats : KickerStats) (game : GameConditions) : ℚ :=
  (game.attempts : ℚ) * stats.missRate * (stats.missTypes 3)

/-- Theorem stating that Wario's expected wide right misses in the next game is 1 -/
theorem wario_expected_wide_right_misses :
  let warioStats : KickerStats := {
    totalAttempts := 80,
    missRate := 1/3,
    missTypes := λ _ => 1/4,
    underFortyYardsSuccessRate := 7/10
  }
  let gameConditions : GameConditions := {
    attempts := 12,
    attemptsUnderForty := 9,
    windSpeed := 18
  }
  expectedWideRightMisses warioStats gameConditions = 1 := by sorry

end wario_expected_wide_right_misses_l3635_363502


namespace min_value_expression_l3635_363574

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + z^2 = 8) : 
  (x + y) / z + (y + z) / x^2 + (z + x) / y^2 ≥ 4 := by
  sorry

end min_value_expression_l3635_363574


namespace cube_sum_and_reciprocal_l3635_363587

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^3 + 1/x^3 = 332 := by
  sorry

end cube_sum_and_reciprocal_l3635_363587


namespace distribute_five_balls_three_boxes_l3635_363573

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Stirling number of the second kind: number of ways to partition n objects into k non-empty subsets -/
def stirling2 (n k : ℕ) : ℕ := sorry

theorem distribute_five_balls_three_boxes : 
  distribute_balls 5 3 = 41 := by sorry

end distribute_five_balls_three_boxes_l3635_363573


namespace polynomial_evaluation_l3635_363569

theorem polynomial_evaluation : 
  let x : ℤ := -2
  2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5 = 5 := by sorry

end polynomial_evaluation_l3635_363569


namespace cube_surface_area_from_prism_l3635_363528

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_from_prism (l w h : ℝ) (h1 : l = 8) (h2 : w = 2) (h3 : h = 32) :
  let prism_volume := l * w * h
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 384 := by sorry

end cube_surface_area_from_prism_l3635_363528


namespace inequality_solution_condition_l3635_363523

theorem inequality_solution_condition (a : ℝ) : 
  (∃! x y : ℤ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
    (∀ z : ℤ, z < 0 → ((z + a) / 2 ≥ 1) ↔ (z = x ∨ z = y))) 
  → 4 ≤ a ∧ a < 5 :=
by sorry

end inequality_solution_condition_l3635_363523


namespace bus_speed_problem_l3635_363558

/-- Given a bus that stops for 15 minutes per hour and has an average speed of 45 km/hr
    including stoppages, its average speed excluding stoppages is 60 km/hr. -/
theorem bus_speed_problem (stop_time : ℝ) (avg_speed_with_stops : ℝ) :
  stop_time = 15 →
  avg_speed_with_stops = 45 →
  ∃ (avg_speed_without_stops : ℝ),
    avg_speed_without_stops = 60 ∧
    avg_speed_with_stops * 1 = avg_speed_without_stops * ((60 - stop_time) / 60) := by
  sorry

end bus_speed_problem_l3635_363558


namespace total_cost_is_correct_l3635_363598

-- Define the types of cows
inductive CowType
| Holstein
| Jersey

-- Define the cost of each cow type
def cow_cost : CowType → Nat
| CowType.Holstein => 260
| CowType.Jersey => 170

-- Define the number of hearts in a standard deck
def hearts_in_deck : Nat := 52

-- Define the number of cows
def total_cows : Nat := 2 * hearts_in_deck

-- Define the ratio of Holstein to Jersey cows
def holstein_ratio : Nat := 3
def jersey_ratio : Nat := 2

-- Define the sales tax rate
def sales_tax_rate : Rat := 5 / 100

-- Define the transportation cost per cow
def transport_cost_per_cow : Nat := 20

-- Define the function to calculate the total cost
def total_cost : ℚ :=
  let holstein_count := (holstein_ratio * total_cows) / (holstein_ratio + jersey_ratio)
  let jersey_count := (jersey_ratio * total_cows) / (holstein_ratio + jersey_ratio)
  let base_cost := holstein_count * cow_cost CowType.Holstein + jersey_count * cow_cost CowType.Jersey
  let sales_tax := base_cost * sales_tax_rate
  let transport_cost := total_cows * transport_cost_per_cow
  (base_cost + sales_tax + transport_cost : ℚ)

-- Theorem statement
theorem total_cost_is_correct : total_cost = 26324.50 := by
  sorry

end total_cost_is_correct_l3635_363598


namespace pauls_crayons_l3635_363511

theorem pauls_crayons (erasers_birthday : ℕ) (crayons_left : ℕ) (eraser_crayon_diff : ℕ) 
  (h1 : erasers_birthday = 406)
  (h2 : crayons_left = 336)
  (h3 : eraser_crayon_diff = 70)
  (h4 : erasers_birthday = crayons_left + eraser_crayon_diff) :
  crayons_left + eraser_crayon_diff = 406 := by
  sorry

end pauls_crayons_l3635_363511


namespace parabola_triangle_area_l3635_363508

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- The focus of the parabola x^2 = 4y -/
def focus : Point := ⟨0, 1⟩

/-- The directrix of the parabola x^2 = 4y -/
def directrix : ℝ := -1

theorem parabola_triangle_area 
  (P : Point) 
  (h_P : P ∈ Parabola) 
  (M : Point) 
  (h_M : M.y = directrix) 
  (h_perp : (P.x - M.x) * (P.y - M.y) + (P.y - M.y) * (M.y - directrix) = 0) 
  (h_dist : (P.x - M.x)^2 + (P.y - M.y)^2 = 25) : 
  (1/2) * |P.x - M.x| * |P.y - focus.y| = 10 :=
sorry

end parabola_triangle_area_l3635_363508


namespace arithmetic_sequence_sum_l3635_363583

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 1 →
  a 1 + a 3 + a 5 = 21 →
  a 2 + a 4 + a 6 = 42 := by
  sorry

end arithmetic_sequence_sum_l3635_363583


namespace sum_of_numbers_l3635_363530

/-- Represents the numbers of various individuals in a mathematical problem. -/
structure Numbers where
  k : ℕ
  joyce : ℕ
  xavier : ℕ
  coraline : ℕ
  jayden : ℕ
  mickey : ℕ
  yvonne : ℕ
  natalie : ℕ

/-- The conditions of the problem are satisfied. -/
def satisfies_conditions (n : Numbers) : Prop :=
  n.k > 1 ∧
  n.joyce = 5 * n.k ∧
  n.xavier = 4 * n.joyce ∧
  n.coraline = n.xavier + 50 ∧
  n.jayden = n.coraline - 40 ∧
  n.mickey = n.jayden + 20 ∧
  n.yvonne = (n.xavier + n.joyce) * n.k ∧
  n.natalie = (n.yvonne - n.coraline) / 2

/-- The theorem to be proved. -/
theorem sum_of_numbers (n : Numbers) 
  (h : satisfies_conditions n) : 
  n.joyce + n.xavier + n.coraline + n.jayden + n.mickey + n.yvonne + n.natalie = 365 := by
  sorry

end sum_of_numbers_l3635_363530


namespace f_sum_negative_l3635_363531

/-- The function f satisfying the given conditions -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

/-- The theorem statement -/
theorem f_sum_negative (m : ℝ) (a b : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → (f m x₁ - f m x₂) / (x₁ - x₂) < 0) →
  a < 0 →
  0 < b →
  abs a < abs b →
  f m a + f m b < 0 :=
by sorry

end f_sum_negative_l3635_363531


namespace tournament_three_cycle_l3635_363549

/-- Represents a tournament with n contestants. -/
structure Tournament (n : ℕ) where
  -- n ≥ 3
  contestants_count : n ≥ 3
  -- Represents the result of matches between contestants
  defeats : Fin n → Fin n → Prop
  -- Each pair of contestants plays exactly one match
  one_match (i j : Fin n) : i ≠ j → (defeats i j ∨ defeats j i) ∧ ¬(defeats i j ∧ defeats j i)
  -- No contestant wins all their matches
  no_perfect_winner (i : Fin n) : ∃ j : Fin n, j ≠ i ∧ defeats j i

/-- 
There exist three contestants A, B, and C such that A defeats B, B defeats C, and C defeats A.
-/
theorem tournament_three_cycle {n : ℕ} (t : Tournament n) :
  ∃ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    t.defeats a b ∧ t.defeats b c ∧ t.defeats c a :=
sorry

end tournament_three_cycle_l3635_363549


namespace minimum_value_theorem_l3635_363503

theorem minimum_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 - b + 4 ≤ 0) :
  ∃ (min : ℝ), min = 14/5 ∧ ∀ x, x = (2*a + 3*b)/(a + b) → x ≥ min :=
sorry

end minimum_value_theorem_l3635_363503


namespace simplify_expression_l3635_363506

theorem simplify_expression : 
  (512 : ℝ)^(1/4) * (343 : ℝ)^(1/2) = 28 * (14 : ℝ)^(1/4) :=
by
  sorry

end simplify_expression_l3635_363506


namespace smallest_digit_change_l3635_363512

def original_sum : ℕ := 2457
def correct_sum : ℕ := 2547
def discrepancy : ℕ := correct_sum - original_sum

def num1 : ℕ := 731
def num2 : ℕ := 964
def num3 : ℕ := 852

def is_smallest_change (d : ℕ) : Prop :=
  d ≤ 9 ∧ 
  (num1 - d * 100 + num2 + num3 = correct_sum) ∧
  ∀ (d' : ℕ), d' < d → (num1 - d' * 100 + num2 + num3 ≠ correct_sum ∧
                        num1 + num2 - d' * 100 + num3 ≠ correct_sum ∧
                        num1 + num2 + num3 - d' * 100 ≠ correct_sum)

theorem smallest_digit_change :
  is_smallest_change 1 :=
sorry

end smallest_digit_change_l3635_363512
