import Mathlib

namespace water_needed_for_lemonade_l3891_389152

-- Define the ratio of water to lemon juice
def water_ratio : ℚ := 4
def lemon_juice_ratio : ℚ := 1

-- Define the total volume in gallons
def total_volume : ℚ := 3

-- Define the conversion factor from gallons to quarts
def quarts_per_gallon : ℚ := 4

-- Theorem statement
theorem water_needed_for_lemonade :
  let total_ratio : ℚ := water_ratio + lemon_juice_ratio
  let total_quarts : ℚ := total_volume * quarts_per_gallon
  let quarts_per_part : ℚ := total_quarts / total_ratio
  let water_quarts : ℚ := water_ratio * quarts_per_part
  water_quarts = 9.6 := by
  sorry

end water_needed_for_lemonade_l3891_389152


namespace integral_f_equals_pi_over_2_plus_4_over_3_l3891_389116

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 1 then Real.sqrt (1 - x^2)
  else if 1 ≤ x ∧ x ≤ 2 then x^2 - 1
  else 0

theorem integral_f_equals_pi_over_2_plus_4_over_3 :
  ∫ x in (-1)..(2), f x = π / 2 + 4 / 3 := by
  sorry

end integral_f_equals_pi_over_2_plus_4_over_3_l3891_389116


namespace arithmetic_progression_five_digit_term_l3891_389157

theorem arithmetic_progression_five_digit_term (n : ℕ) (k : ℕ) : 
  let a : ℕ → ℤ := λ i => -1 + (i - 1) * 19
  let is_all_fives : ℤ → Prop := λ x => ∃ m : ℕ, x = 5 * ((10^m - 1) / 9)
  (∃ n, is_all_fives (a n)) ↔ k = 3 ∧ 19 * n - 20 = 5 * ((10^k - 1) / 9) :=
by sorry

end arithmetic_progression_five_digit_term_l3891_389157


namespace large_rectangle_perimeter_large_rectangle_perimeter_proof_l3891_389191

/-- The perimeter of a rectangle composed of three squares with perimeter 24 each
    and three rectangles with perimeter 16 each is 52. -/
theorem large_rectangle_perimeter : ℝ → Prop :=
  fun (perimeter : ℝ) =>
    let square_perimeter := 24
    let small_rectangle_perimeter := 16
    let square_side := square_perimeter / 4
    let small_rectangle_width := (small_rectangle_perimeter / 2) - square_side
    let large_rectangle_height := square_side + small_rectangle_width
    let large_rectangle_width := 3 * square_side
    perimeter = 2 * (large_rectangle_height + large_rectangle_width) ∧
    perimeter = 52

/-- Proof of the theorem -/
theorem large_rectangle_perimeter_proof : large_rectangle_perimeter 52 := by
  sorry

#check large_rectangle_perimeter_proof

end large_rectangle_perimeter_large_rectangle_perimeter_proof_l3891_389191


namespace sqrt_sum_equals_sum_of_sqrts_l3891_389184

theorem sqrt_sum_equals_sum_of_sqrts : 
  Real.sqrt (36 + 14 * Real.sqrt 6 + 14 * Real.sqrt 5 + 6 * Real.sqrt 30) = 
  Real.sqrt 15 + Real.sqrt 10 + Real.sqrt 8 + Real.sqrt 3 := by
  sorry

end sqrt_sum_equals_sum_of_sqrts_l3891_389184


namespace kaylin_age_is_33_l3891_389163

def freyja_age : ℕ := 10
def eli_age : ℕ := freyja_age + 9
def sarah_age : ℕ := 2 * eli_age
def kaylin_age : ℕ := sarah_age - 5

theorem kaylin_age_is_33 : kaylin_age = 33 := by
  sorry

end kaylin_age_is_33_l3891_389163


namespace probability_odd_sum_rows_l3891_389166

/-- Represents a 4x3 grid filled with numbers 1 to 12 --/
def Grid := Fin 4 → Fin 3 → Fin 12

/-- Checks if a list of numbers has an odd sum --/
def has_odd_sum (row : List (Fin 12)) : Prop :=
  (row.map (fun n => n.val + 1)).sum % 2 = 1

/-- Represents a valid grid configuration --/
def valid_grid (g : Grid) : Prop :=
  ∀ i : Fin 4, has_odd_sum [g i 0, g i 1, g i 2]

/-- The total number of ways to arrange 12 numbers in a 4x3 grid --/
def total_arrangements : ℕ := 479001600

/-- The number of valid arrangements (where each row has an odd sum) --/
def valid_arrangements : ℕ := 21600

theorem probability_odd_sum_rows :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 22176 :=
sorry

end probability_odd_sum_rows_l3891_389166


namespace complete_square_quadratic_l3891_389179

/-- Given a quadratic equation 4x^2 - 8x - 320 = 0, prove that when transformed
    into the form (x+p)^2 = q by completing the square, the value of q is 81. -/
theorem complete_square_quadratic :
  ∃ (p : ℝ), ∀ (x : ℝ),
    (4 * x^2 - 8 * x - 320 = 0) ↔ ((x + p)^2 = 81) :=
by sorry

end complete_square_quadratic_l3891_389179


namespace locus_is_ellipse_l3891_389111

/-- The locus of points P(x,y) satisfying the given conditions forms an ellipse -/
theorem locus_is_ellipse (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 0)
  let directrix : ℝ → Prop := λ t => t = 9
  let dist_ratio : ℝ := 1/3
  let dist_to_A : ℝ := Real.sqrt ((x - A.1)^2 + (y - A.2)^2)
  let dist_to_directrix : ℝ := |x - 9|
  dist_to_A / dist_to_directrix = dist_ratio →
  x^2/9 + y^2/8 = 1 :=
by sorry

end locus_is_ellipse_l3891_389111


namespace marble_probability_l3891_389158

theorem marble_probability (total_marbles : ℕ) 
  (prob_both_black : ℚ) (box1 box2 : ℕ) :
  total_marbles = 30 →
  box1 + box2 = total_marbles →
  prob_both_black = 3/5 →
  box1 > 0 ∧ box2 > 0 →
  ∃ (black1 black2 : ℕ),
    black1 ≤ box1 ∧ black2 ≤ box2 ∧
    (black1 : ℚ) / box1 * (black2 : ℚ) / box2 = prob_both_black →
    ((box1 - black1 : ℚ) / box1 * (box2 - black2 : ℚ) / box2 = 4/25) :=
by sorry

#check marble_probability

end marble_probability_l3891_389158


namespace tournament_matches_divisible_by_two_l3891_389180

/-- Represents a single elimination tennis tournament -/
structure TennisTournament where
  total_players : ℕ
  bye_players : ℕ
  first_round_players : ℕ

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : TennisTournament) : ℕ :=
  t.total_players - 1

/-- Theorem: The total number of matches in the specified tournament is divisible by 2 -/
theorem tournament_matches_divisible_by_two :
  ∃ (t : TennisTournament), 
    t.total_players = 128 ∧ 
    t.bye_players = 32 ∧ 
    t.first_round_players = 96 ∧ 
    ∃ (k : ℕ), total_matches t = 2 * k := by
  sorry

end tournament_matches_divisible_by_two_l3891_389180


namespace sequence_product_l3891_389130

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) / b n = b (n + 2) / b (n + 1)

/-- The main theorem -/
theorem sequence_product (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 11 = 8 →
  geometric_sequence b →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
sorry

end sequence_product_l3891_389130


namespace watch_cost_is_20_l3891_389173

-- Define the given conditions
def evans_initial_money : ℕ := 1
def money_given_to_evan : ℕ := 12
def additional_money_needed : ℕ := 7

-- Define the cost of the watch
def watch_cost : ℕ := evans_initial_money + money_given_to_evan + additional_money_needed

-- Theorem to prove
theorem watch_cost_is_20 : watch_cost = 20 := by
  sorry

end watch_cost_is_20_l3891_389173


namespace work_completion_men_count_l3891_389148

/-- Proves that the number of men in the second group is 15, given the conditions of the problem -/
theorem work_completion_men_count : 
  ∀ (work : ℕ) (men1 men2 days1 days2 : ℕ),
    men1 = 18 →
    days1 = 20 →
    days2 = 24 →
    work = men1 * days1 →
    work = men2 * days2 →
    men2 = 15 :=
by
  sorry

end work_completion_men_count_l3891_389148


namespace equation_proof_l3891_389143

theorem equation_proof : 361 + 2 * 19 * 6 + 36 = 625 := by
  sorry

end equation_proof_l3891_389143


namespace unique_group_size_l3891_389133

theorem unique_group_size (n : ℕ) (k : ℕ) : 
  (∀ (i j : Fin n), i ≠ j → ∃! (call : Bool), call) →
  (∀ (subset : Finset (Fin n)), subset.card = n - 2 → 
    (subset.sum (λ i => (subset.filter (λ j => j ≠ i)).card) / 2) = 3^k) →
  n = 5 :=
sorry

end unique_group_size_l3891_389133


namespace cars_without_ac_l3891_389142

theorem cars_without_ac (total : ℕ) (min_racing : ℕ) (max_ac_no_racing : ℕ)
  (h_total : total = 100)
  (h_min_racing : min_racing = 51)
  (h_max_ac_no_racing : max_ac_no_racing = 49) :
  total - (max_ac_no_racing + (min_racing - max_ac_no_racing)) = 49 := by
  sorry

end cars_without_ac_l3891_389142


namespace difference_of_squares_l3891_389199

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l3891_389199


namespace contrapositive_odd_sum_l3891_389155

theorem contrapositive_odd_sum (x y : ℤ) :
  (¬(Odd (x + y)) → ¬(Odd x ∧ Odd y)) ↔
  (∀ x y : ℤ, (Odd x ∧ Odd y) → Odd (x + y)) :=
by sorry

end contrapositive_odd_sum_l3891_389155


namespace star_to_maltese_cross_l3891_389168

/-- Represents a four-pointed star -/
structure FourPointedStar :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Represents a frame -/
structure Frame :=
  (corners : Fin 4 → ℝ × ℝ)

/-- Represents a part of the cut star -/
structure StarPart :=
  (vertices : Fin 3 → ℝ × ℝ)

/-- Represents a Maltese cross -/
structure MalteseCross :=
  (vertices : Fin 8 → ℝ × ℝ)

/-- Function to cut a FourPointedStar into 4 StarParts -/
def cutStar (star : FourPointedStar) : Fin 4 → StarPart :=
  sorry

/-- Function to arrange StarParts in a Frame -/
def arrangeParts (parts : Fin 4 → StarPart) (frame : Frame) : MalteseCross :=
  sorry

/-- Theorem stating that a FourPointedStar can be cut and arranged to form a MalteseCross -/
theorem star_to_maltese_cross (star : FourPointedStar) (frame : Frame) :
  ∃ (arrangement : MalteseCross), arrangement = arrangeParts (cutStar star) frame :=
sorry

end star_to_maltese_cross_l3891_389168


namespace pizza_topping_combinations_l3891_389176

theorem pizza_topping_combinations (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) + (n.choose 3) = 92 := by sorry

end pizza_topping_combinations_l3891_389176


namespace f_equals_g_l3891_389117

-- Define the functions f and g
def f (x : ℝ) : ℝ := x - 1
def g (t : ℝ) : ℝ := t - 1

-- Theorem stating that f and g represent the same function
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end f_equals_g_l3891_389117


namespace complex_number_proof_l3891_389160

def i : ℂ := Complex.I

def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_number_proof (z : ℂ) 
  (h1 : is_real (z + 2*i)) 
  (h2 : is_real (z / (2 - i))) : 
  z = 4 - 2*i ∧ Complex.abs (z / (1 + i)) = Real.sqrt 10 := by
  sorry

end complex_number_proof_l3891_389160


namespace blue_faces_cube_l3891_389159

theorem blue_faces_cube (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 :=
by
  sorry

end blue_faces_cube_l3891_389159


namespace bridesmaid_dresses_completion_time_l3891_389188

def dress_hours : List Nat := [15, 18, 20, 22, 24, 26, 28]
def weekly_pattern : List Nat := [5, 3, 6, 4]
def finalization_hours : Nat := 10

def total_sewing_hours : Nat := dress_hours.sum
def total_hours : Nat := total_sewing_hours + finalization_hours
def cycle_hours : Nat := weekly_pattern.sum

def weeks_to_complete : Nat :=
  let full_cycles := (total_hours + cycle_hours - 1) / cycle_hours
  full_cycles * 4 - 3

theorem bridesmaid_dresses_completion_time :
  weeks_to_complete = 37 := by sorry

end bridesmaid_dresses_completion_time_l3891_389188


namespace trigonometric_equation_solution_l3891_389154

theorem trigonometric_equation_solution (x : ℝ) :
  (2 * Real.sin x ^ 3 + 2 * Real.sin x ^ 2 * Real.cos x - Real.sin x * Real.cos x ^ 2 - Real.cos x ^ 3 = 0) ↔
  (∃ n : ℤ, x = -π / 4 + n * π) ∨
  (∃ k : ℤ, x = Real.arctan (Real.sqrt 2 / 2) + k * π) ∨
  (∃ k : ℤ, x = -Real.arctan (Real.sqrt 2 / 2) + k * π) := by
  sorry

end trigonometric_equation_solution_l3891_389154


namespace root_difference_squared_l3891_389194

theorem root_difference_squared (p q : ℚ) : 
  (6 * p^2 - 7 * p - 20 = 0) → 
  (6 * q^2 - 7 * q - 20 = 0) → 
  (p - q)^2 = 529 / 36 := by
sorry

end root_difference_squared_l3891_389194


namespace angle_measure_in_special_triangle_l3891_389114

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b² = a² + ac + c², then the measure of angle B is 120°. -/
theorem angle_measure_in_special_triangle (a b c : ℝ) (h : b^2 = a^2 + a*c + c^2) :
  let angle_B := Real.arccos (-1/2)
  angle_B = 2 * π / 3 := by
  sorry

end angle_measure_in_special_triangle_l3891_389114


namespace t_upper_bound_F_positive_l3891_389129

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := t / x - f x
noncomputable def F (x : ℝ) : ℝ := f x - 1 / Real.exp x + 2 / (Real.exp 1 * x)

-- Theorem 1
theorem t_upper_bound (t : ℝ) :
  (∀ x > 0, g t x ≤ f x) → t ≤ -2 / Real.exp 1 :=
sorry

-- Theorem 2
theorem F_positive (x : ℝ) :
  x > 0 → F x > 0 :=
sorry

end t_upper_bound_F_positive_l3891_389129


namespace transaction_fraction_proof_l3891_389125

theorem transaction_fraction_proof (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ) :
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + mabel_transactions / 10 →
  jade_transactions = 83 →
  jade_transactions = cal_transactions + 17 →
  3 * cal_transactions = 2 * anthony_transactions :=
by
  sorry

#check transaction_fraction_proof

end transaction_fraction_proof_l3891_389125


namespace jacket_markup_percentage_l3891_389174

/-- Proves that the markup percentage is 40% given the conditions of the jacket sale problem -/
theorem jacket_markup_percentage (purchase_price : ℝ) (selling_price : ℝ) (markup_percentage : ℝ) 
  (sale_discount : ℝ) (gross_profit : ℝ) :
  purchase_price = 48 →
  selling_price = purchase_price + markup_percentage * selling_price →
  sale_discount = 0.2 →
  gross_profit = 16 →
  (1 - sale_discount) * selling_price - purchase_price = gross_profit →
  markup_percentage = 0.4 := by
sorry

end jacket_markup_percentage_l3891_389174


namespace simplify_trig_expression_l3891_389182

theorem simplify_trig_expression :
  (Real.sin (30 * π / 180) + Real.sin (50 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (50 * π / 180)) =
  Real.tan (40 * π / 180) := by sorry

end simplify_trig_expression_l3891_389182


namespace activities_equally_popular_l3891_389134

def dodgeball : Rat := 10 / 25
def artWorkshop : Rat := 12 / 30
def movieScreening : Rat := 18 / 45
def quizBowl : Rat := 16 / 40

theorem activities_equally_popular :
  dodgeball = artWorkshop ∧
  artWorkshop = movieScreening ∧
  movieScreening = quizBowl := by
  sorry

end activities_equally_popular_l3891_389134


namespace triangle_trip_distance_l3891_389162

/-- Given a right-angled triangle DEF with F as the right angle, 
    where DF = 2000 and DE = 4500, prove that DE + EF + DF = 10531 -/
theorem triangle_trip_distance (DE DF EF : ℝ) : 
  DE = 4500 → 
  DF = 2000 → 
  EF ^ 2 = DE ^ 2 - DF ^ 2 → 
  DE + EF + DF = 10531 := by
sorry

end triangle_trip_distance_l3891_389162


namespace total_molecular_weight_l3891_389136

-- Define atomic weights
def carbon_weight : ℝ := 12.01
def hydrogen_weight : ℝ := 1.008
def oxygen_weight : ℝ := 16.00

-- Define molecular formulas
def ascorbic_acid_carbon : ℕ := 6
def ascorbic_acid_hydrogen : ℕ := 8
def ascorbic_acid_oxygen : ℕ := 6

def citric_acid_carbon : ℕ := 6
def citric_acid_hydrogen : ℕ := 8
def citric_acid_oxygen : ℕ := 7

-- Define number of moles
def ascorbic_acid_moles : ℕ := 7
def citric_acid_moles : ℕ := 5

-- Calculate molecular weights
def ascorbic_acid_weight : ℝ :=
  (ascorbic_acid_carbon * carbon_weight) +
  (ascorbic_acid_hydrogen * hydrogen_weight) +
  (ascorbic_acid_oxygen * oxygen_weight)

def citric_acid_weight : ℝ :=
  (citric_acid_carbon * carbon_weight) +
  (citric_acid_hydrogen * hydrogen_weight) +
  (citric_acid_oxygen * oxygen_weight)

-- Theorem statement
theorem total_molecular_weight :
  (ascorbic_acid_moles * ascorbic_acid_weight) +
  (citric_acid_moles * citric_acid_weight) = 2193.488 :=
by sorry

end total_molecular_weight_l3891_389136


namespace intersection_of_A_and_B_l3891_389105

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l3891_389105


namespace functional_equation_solution_l3891_389119

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end functional_equation_solution_l3891_389119


namespace equal_side_length_l3891_389137

/-- An isosceles right-angled triangle with side lengths a, a, and c, where the sum of squares of sides is 725 --/
structure IsoscelesRightTriangle where
  a : ℝ
  c : ℝ
  isosceles : c^2 = 2 * a^2
  sum_of_squares : a^2 + a^2 + c^2 = 725

/-- The length of each equal side in the isosceles right-angled triangle is 13.5 --/
theorem equal_side_length (t : IsoscelesRightTriangle) : t.a = 13.5 := by
  sorry

end equal_side_length_l3891_389137


namespace boat_downstream_time_l3891_389140

def boat_problem (boat_speed : ℝ) (stream_speed : ℝ) (upstream_time : ℝ) : Prop :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance := upstream_speed * upstream_time
  let downstream_time := distance / downstream_speed
  downstream_time = 1

theorem boat_downstream_time :
  boat_problem 15 3 1.5 := by sorry

end boat_downstream_time_l3891_389140


namespace prob_first_qualified_on_third_test_l3891_389161

/-- The probability of obtaining the first qualified product on the third test. -/
def P_epsilon_3 (pass_rate : ℝ) (fail_rate : ℝ) : ℝ :=
  fail_rate^2 * pass_rate

/-- The theorem stating that P(ε = 3) is equal to (1/4)² × (3/4) given the specified pass and fail rates. -/
theorem prob_first_qualified_on_third_test :
  let pass_rate : ℝ := 3/4
  let fail_rate : ℝ := 1/4
  P_epsilon_3 pass_rate fail_rate = (1/4)^2 * (3/4) :=
by sorry

end prob_first_qualified_on_third_test_l3891_389161


namespace quadratic_complex_root_l3891_389153

/-- Given a quadratic equation x^2 + px + q = 0 with real coefficients,
    if 1 + i is a root, then q = 2. -/
theorem quadratic_complex_root (p q : ℝ) : 
  (∀ x : ℂ, x^2 + p * x + q = 0 ↔ x = (1 + I) ∨ x = (1 - I)) → q = 2 := by
  sorry

end quadratic_complex_root_l3891_389153


namespace smallest_multiple_of_5_and_24_l3891_389121

theorem smallest_multiple_of_5_and_24 : ∃ n : ℕ, n > 0 ∧ n % 5 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 5 = 0 → m % 24 = 0 → n ≤ m := by
  sorry

end smallest_multiple_of_5_and_24_l3891_389121


namespace smallest_cube_ending_580_l3891_389146

theorem smallest_cube_ending_580 : 
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 580 ∧ ∀ m : ℕ, m > 0 ∧ m^3 % 1000 = 580 → m ≥ n :=
by
  -- The proof goes here
  sorry

end smallest_cube_ending_580_l3891_389146


namespace greatest_XPM_l3891_389185

/-- A function that checks if a number is a two-digit number with equal digits -/
def is_two_digit_equal (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

/-- A function that checks if a number is a one-digit prime -/
def is_one_digit_prime (n : ℕ) : Prop :=
  n < 10 ∧ Nat.Prime n

/-- The main theorem -/
theorem greatest_XPM :
  ∀ M N XPM : ℕ,
  is_two_digit_equal M →
  is_one_digit_prime N →
  N ≠ M / 11 →
  M * N = XPM →
  100 ≤ XPM ∧ XPM ≤ 999 →
  XPM ≤ 462 :=
sorry

end greatest_XPM_l3891_389185


namespace stratified_sampling_most_appropriate_l3891_389110

-- Define the type for sampling methods
inductive SamplingMethod
| Lottery
| RandomNumber
| Stratified
| Systematic

-- Define the company's production
structure Company where
  sedanModels : Nat
  significantDifferences : Bool

-- Define the appropriateness of a sampling method
def isAppropriate (method : SamplingMethod) (company : Company) : Prop :=
  method = SamplingMethod.Stratified ∧ 
  company.sedanModels > 1 ∧ 
  company.significantDifferences

-- Theorem statement
theorem stratified_sampling_most_appropriate (company : Company) 
  (h1 : company.sedanModels = 3) 
  (h2 : company.significantDifferences = true) :
  isAppropriate SamplingMethod.Stratified company := by
  sorry

end stratified_sampling_most_appropriate_l3891_389110


namespace liam_juice_consumption_l3891_389167

/-- The number of glasses of juice Liam drinks in a given time period -/
def glasses_of_juice (time_minutes : ℕ) : ℕ :=
  time_minutes / 20

theorem liam_juice_consumption : glasses_of_juice 340 = 17 := by
  sorry

end liam_juice_consumption_l3891_389167


namespace quadratic_root_bound_l3891_389193

theorem quadratic_root_bound (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_real_roots : ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ |x| ≤ 2 * |c / b| := by
  sorry

end quadratic_root_bound_l3891_389193


namespace intersection_chord_length_l3891_389124

noncomputable def line_l (x y : ℝ) : Prop := x + 2*y = 0

noncomputable def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 2

theorem intersection_chord_length :
  ∀ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    A ≠ B →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 1435 / 35 := by
  sorry

end intersection_chord_length_l3891_389124


namespace star_ratio_equals_two_thirds_l3891_389109

-- Define the ⋆ operation
def star (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem star_ratio_equals_two_thirds :
  (star 3 2 : ℚ) / (star 2 3 : ℚ) = 2/3 := by sorry

end star_ratio_equals_two_thirds_l3891_389109


namespace perpendicular_vectors_l3891_389106

/-- Given two vectors a and b in ℝ², and a real number k, 
    we define vector c as a sum of a and k * b. 
    If b is perpendicular to c, then k equals -3. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) :
  a = (10, 20) →
  b = (5, 5) →
  let c := (a.1 + k * b.1, a.2 + k * b.2)
  (b.1 * c.1 + b.2 * c.2 = 0) →
  k = -3 := by
  sorry

end perpendicular_vectors_l3891_389106


namespace number_puzzle_l3891_389150

theorem number_puzzle (N : ℚ) : (5/4) * N = (4/5) * N + 45 → N = 100 := by
  sorry

end number_puzzle_l3891_389150


namespace mountain_speed_decrease_l3891_389190

/-- The problem of finding the percentage decrease in vehicle speed when ascending a mountain. -/
theorem mountain_speed_decrease (initial_speed : ℝ) (ascend_distance descend_distance : ℝ) 
  (total_time : ℝ) (descend_increase : ℝ) :
  initial_speed = 30 →
  ascend_distance = 60 →
  descend_distance = 72 →
  total_time = 6 →
  descend_increase = 0.2 →
  ∃ (x : ℝ),
    x = 0.5 ∧
    (ascend_distance / (initial_speed * (1 - x))) + 
    (descend_distance / (initial_speed * (1 + descend_increase))) = total_time :=
by sorry

end mountain_speed_decrease_l3891_389190


namespace ashutosh_completion_time_l3891_389149

/-- The time it takes Suresh to complete the job alone -/
def suresh_time : ℝ := 15

/-- The time Suresh works on the job -/
def suresh_work_time : ℝ := 9

/-- The time it takes Ashutosh to complete the remaining job -/
def ashutosh_remaining_time : ℝ := 14

/-- The time it takes Ashutosh to complete the job alone -/
def ashutosh_time : ℝ := 35

theorem ashutosh_completion_time :
  (suresh_work_time / suresh_time) + 
  ((1 - suresh_work_time / suresh_time) / ashutosh_time) = 
  (1 / ashutosh_remaining_time) := by
  sorry

#check ashutosh_completion_time

end ashutosh_completion_time_l3891_389149


namespace direct_proportion_m_value_l3891_389113

/-- A linear function y = mx + b is a direct proportion if and only if b = 0 -/
def is_direct_proportion (m b : ℝ) : Prop := b = 0

/-- Given that y = mx + (m - 2) is a direct proportion function, prove that m = 2 -/
theorem direct_proportion_m_value (m : ℝ) 
  (h : is_direct_proportion m (m - 2)) : m = 2 := by
  sorry

end direct_proportion_m_value_l3891_389113


namespace multiple_with_binary_digits_l3891_389135

theorem multiple_with_binary_digits (n : ℕ) : ∃ m : ℕ, 
  (m % n = 0) ∧ 
  (Nat.digits 2 m).length = n ∧ 
  (∀ d ∈ Nat.digits 2 m, d = 0 ∨ d = 1) :=
sorry

end multiple_with_binary_digits_l3891_389135


namespace lucas_150_mod_5_l3891_389147

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

/-- The 150th term of the Lucas sequence modulo 5 is equal to 3 -/
theorem lucas_150_mod_5 : lucas 149 % 5 = 3 := by
  sorry

end lucas_150_mod_5_l3891_389147


namespace exists_monomial_with_conditions_l3891_389122

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers. -/
structure Monomial (α : Type*) [Semiring α] where
  coeff : α
  powers : List (Nat × Nat)

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def Monomial.degree {α : Type*} [Semiring α] (m : Monomial α) : Nat :=
  m.powers.foldl (fun acc (_, pow) => acc + pow) 0

/-- A monomial contains specific variables if they appear in its power list. -/
def Monomial.containsVariables {α : Type*} [Semiring α] (m : Monomial α) (vars : List Nat) : Prop :=
  ∀ v ∈ vars, ∃ (pow : Nat), (v, pow) ∈ m.powers

/-- There exists a monomial with coefficient 3, containing variables x and y, and having a total degree of 3. -/
theorem exists_monomial_with_conditions :
  ∃ (m : Monomial ℕ),
    m.coeff = 3 ∧
    m.containsVariables [1, 2] ∧  -- Let 1 represent x and 2 represent y
    m.degree = 3 := by
  sorry

end exists_monomial_with_conditions_l3891_389122


namespace evaluate_expression_l3891_389181

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 := by
  sorry

end evaluate_expression_l3891_389181


namespace sqrt_x_minus_one_real_l3891_389127

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_real_l3891_389127


namespace geometric_sequence_alternating_l3891_389107

def is_alternating_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) < 0

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_alternating
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_sum1 : a 1 + a 2 = -3/2)
  (h_sum2 : a 4 + a 5 = 12) :
  is_alternating_sequence a :=
sorry

end geometric_sequence_alternating_l3891_389107


namespace square_root_sum_difference_l3891_389197

theorem square_root_sum_difference (x y : ℝ) : 
  x = Real.sqrt 7 + Real.sqrt 3 →
  y = Real.sqrt 7 - Real.sqrt 3 →
  x * y = 4 ∧ x^2 + y^2 = 20 := by
  sorry

end square_root_sum_difference_l3891_389197


namespace min_unique_score_above_90_l3891_389100

/-- Represents the scoring system for the modified AHSME exam -/
def score (c w : ℕ) : ℕ := 35 + 5 * c - 2 * w

/-- Represents the total number of questions in the exam -/
def total_questions : ℕ := 35

/-- Theorem stating that 91 is the minimum score above 90 with a unique solution -/
theorem min_unique_score_above_90 :
  ∀ s : ℕ, s > 90 →
  (∃! (c w : ℕ), c + w ≤ total_questions ∧ score c w = s) →
  s ≥ 91 :=
sorry

end min_unique_score_above_90_l3891_389100


namespace tip_percentage_calculation_l3891_389198

theorem tip_percentage_calculation (total_bill : ℝ) (food_price : ℝ) (tax_rate : ℝ) : 
  total_bill = 198 →
  food_price = 150 →
  tax_rate = 0.1 →
  (total_bill - food_price * (1 + tax_rate)) / (food_price * (1 + tax_rate)) = 0.2 :=
by sorry

end tip_percentage_calculation_l3891_389198


namespace complex_number_problem_l3891_389169

theorem complex_number_problem (α β : ℂ) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ α + β = x ∧ Complex.I * (α - 3 * β) = y) →
  β = 4 + Complex.I →
  α = 12 - Complex.I := by
sorry

end complex_number_problem_l3891_389169


namespace solve_equation_l3891_389138

theorem solve_equation : ∃ x : ℝ, (2 * x + 7) / 5 = 17 ∧ x = 39 := by
  sorry

end solve_equation_l3891_389138


namespace second_derivative_f_l3891_389151

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x + x * Real.sin x

theorem second_derivative_f (x : ℝ) (hx : x ≠ 0) :
  (deriv^[2] f) x = (2 * x * Real.exp x * (1 - x)) / x^4 + Real.cos x - x * Real.sin x :=
sorry

end second_derivative_f_l3891_389151


namespace cone_lateral_surface_l3891_389164

theorem cone_lateral_surface (l r : ℝ) (h : l > 0) (k : r > 0) : 
  (2 * π * r) / l = 4 * π / 3 → r / l = 2 / 3 := by
sorry

end cone_lateral_surface_l3891_389164


namespace square_perimeter_from_diagonal_l3891_389172

theorem square_perimeter_from_diagonal (d : ℝ) (h : d = 12) :
  let side := d / Real.sqrt 2
  4 * side = 24 * Real.sqrt 2 := by sorry

end square_perimeter_from_diagonal_l3891_389172


namespace preservation_time_at_33_l3891_389102

/-- The preservation time function -/
noncomputable def preservation_time (k b x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the preservation time at 33°C given conditions -/
theorem preservation_time_at_33 (k b : ℝ) :
  preservation_time k b 0 = 192 →
  preservation_time k b 22 = 48 →
  preservation_time k b 33 = 24 := by
  sorry

end preservation_time_at_33_l3891_389102


namespace davids_english_marks_l3891_389156

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  english : ℕ
  average : ℚ

/-- Theorem stating that given David's marks in other subjects and his average,
    his English marks must be 90 -/
theorem davids_english_marks (david : StudentMarks) 
  (math_marks : david.mathematics = 92)
  (physics_marks : david.physics = 85)
  (chemistry_marks : david.chemistry = 87)
  (biology_marks : david.biology = 85)
  (avg_marks : david.average = 87.8)
  : david.english = 90 := by
  sorry

#check davids_english_marks

end davids_english_marks_l3891_389156


namespace fencing_cost_rectangle_l3891_389132

/-- Proves that for a rectangular field with sides in the ratio 3:4 and an area of 8748 sq. m, 
    the cost of fencing at 25 paise per metre is 94.5 rupees. -/
theorem fencing_cost_rectangle (length width : ℝ) (area perimeter cost_per_meter total_cost : ℝ) : 
  length / width = 4 / 3 →
  area = 8748 →
  area = length * width →
  perimeter = 2 * (length + width) →
  cost_per_meter = 0.25 →
  total_cost = perimeter * cost_per_meter →
  total_cost = 94.5 := by
sorry

end fencing_cost_rectangle_l3891_389132


namespace hyperbola_asymptote_slope_l3891_389120

/-- Given a hyperbola with equation mx^2 + y^2 = 1, if one of its asymptotes has a slope of 2, then m = -4 -/
theorem hyperbola_asymptote_slope (m : ℝ) : 
  (∃ (x y : ℝ), m * x^2 + y^2 = 1) →  -- Hyperbola equation exists
  (∃ (k : ℝ), k = 2 ∧ k^2 = -m) →    -- One asymptote has slope 2
  m = -4 :=
by sorry

end hyperbola_asymptote_slope_l3891_389120


namespace digit_addition_puzzle_l3891_389192

/-- Represents a four-digit number ABCD --/
def FourDigitNumber (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem digit_addition_puzzle :
  ∃ (possible_d : Finset ℕ),
    (∀ a b c d : ℕ,
      a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧  -- Digits are less than 10
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- Digits are distinct
      FourDigitNumber a a b c + FourDigitNumber b c a d = FourDigitNumber d b c d →  -- AABC + BCAD = DBCD
      d ∈ possible_d) ∧
    possible_d.card = 9  -- There are 9 possible values for D
  := by sorry

end digit_addition_puzzle_l3891_389192


namespace baker_extra_donuts_l3891_389139

theorem baker_extra_donuts (total_donuts : ℕ) (num_boxes : ℕ) 
  (h1 : total_donuts = 48) 
  (h2 : num_boxes = 7) : 
  total_donuts % num_boxes = 6 := by
  sorry

end baker_extra_donuts_l3891_389139


namespace circle_tripled_radius_l3891_389170

theorem circle_tripled_radius (r : ℝ) (h : r > 0) :
  let new_r := 3 * r
  let original_area := π * r^2
  let new_area := π * new_r^2
  let original_circumference := 2 * π * r
  let new_circumference := 2 * π * new_r
  (new_area = 9 * original_area) ∧ (new_circumference = 3 * original_circumference) := by
  sorry

end circle_tripled_radius_l3891_389170


namespace quadratic_equation_solution_l3891_389175

theorem quadratic_equation_solution (t s : ℝ) : t = 8 * s^2 + 2 * s → t = 5 →
  s = (-1 + Real.sqrt 41) / 8 ∨ s = (-1 - Real.sqrt 41) / 8 := by
  sorry

end quadratic_equation_solution_l3891_389175


namespace divisors_of_cube_l3891_389103

theorem divisors_of_cube (n : ℕ) : 
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n} ∧ d.card = 5) →
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n^3} ∧ (d.card = 13 ∨ d.card = 16)) :=
by sorry

end divisors_of_cube_l3891_389103


namespace wages_calculation_l3891_389145

/-- The wages calculation problem -/
theorem wages_calculation 
  (initial_workers : ℕ) 
  (initial_days : ℕ) 
  (initial_wages : ℚ) 
  (new_workers : ℕ) 
  (new_days : ℕ) 
  (h1 : initial_workers = 15) 
  (h2 : initial_days = 6) 
  (h3 : initial_wages = 9450) 
  (h4 : new_workers = 19) 
  (h5 : new_days = 5) : 
  (initial_wages / (initial_workers * initial_days : ℚ)) * (new_workers * new_days) = 9975 :=
by sorry

end wages_calculation_l3891_389145


namespace classroom_capacity_l3891_389189

theorem classroom_capacity (total_students : ℕ) (num_classrooms : ℕ) 
  (h1 : total_students = 390) (h2 : num_classrooms = 13) :
  total_students / num_classrooms = 30 := by
  sorry

end classroom_capacity_l3891_389189


namespace product_of_divisors_equals_3_30_5_40_l3891_389108

/-- The product of divisors function -/
def productOfDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the product of all divisors of N equals 3^30 * 5^40, then N = 3^3 * 5^4 -/
theorem product_of_divisors_equals_3_30_5_40 (N : ℕ) :
  productOfDivisors N = 3^30 * 5^40 → N = 3^3 * 5^4 := by sorry

end product_of_divisors_equals_3_30_5_40_l3891_389108


namespace divisible_by_1998_digit_sum_l3891_389112

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, if n is divisible by 1998, 
    then the sum of its digits is greater than or equal to 27 -/
theorem divisible_by_1998_digit_sum (n : ℕ) : 
  n % 1998 = 0 → sum_of_digits n ≥ 27 := by sorry

end divisible_by_1998_digit_sum_l3891_389112


namespace exercise_book_count_l3891_389183

/-- Given a ratio of pencils to pens to exercise books and the number of pencils,
    calculate the number of exercise books. -/
theorem exercise_book_count (pencil_ratio : ℕ) (pen_ratio : ℕ) (book_ratio : ℕ) 
    (pencil_count : ℕ) (h1 : pencil_ratio = 14) (h2 : pen_ratio = 4) (h3 : book_ratio = 3) 
    (h4 : pencil_count = 140) : 
    (pencil_count * book_ratio) / pencil_ratio = 30 := by
  sorry

end exercise_book_count_l3891_389183


namespace impossibility_of_all_prime_combinations_l3891_389186

/-- A digit is a natural number from 0 to 9 -/
def Digit := {n : ℕ // n < 10}

/-- A two-digit number formed from two digits -/
def TwoDigitNumber (d1 d2 : Digit) : ℕ := d1.val * 10 + d2.val

/-- Predicate to check if a natural number is prime -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem impossibility_of_all_prime_combinations :
  ∀ (d1 d2 d3 d4 : Digit),
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 →
    ∃ (i j : Fin 4),
      i ≠ j ∧
      ¬(IsPrime (TwoDigitNumber (([d1, d2, d3, d4].get i) : Digit) (([d1, d2, d3, d4].get j) : Digit))) :=
by sorry

end impossibility_of_all_prime_combinations_l3891_389186


namespace unique_number_theorem_l3891_389178

def A₁ (n : ℕ) : Prop := n < 12
def A₂ (n : ℕ) : Prop := ¬(7 ∣ n)
def A₃ (n : ℕ) : Prop := 5 * n < 70

def B₁ (n : ℕ) : Prop := 12 * n > 1000
def B₂ (n : ℕ) : Prop := 10 ∣ n
def B₃ (n : ℕ) : Prop := n > 100

def C₁ (n : ℕ) : Prop := 4 ∣ n
def C₂ (n : ℕ) : Prop := 11 * n < 1000
def C₃ (n : ℕ) : Prop := 9 ∣ n

def D₁ (n : ℕ) : Prop := n < 20
def D₂ (n : ℕ) : Prop := Nat.Prime n
def D₃ (n : ℕ) : Prop := 7 ∣ n

def at_least_one_true (p q r : Prop) : Prop := p ∨ q ∨ r
def at_least_one_false (p q r : Prop) : Prop := ¬p ∨ ¬q ∨ ¬r

theorem unique_number_theorem (n : ℕ) : 
  (at_least_one_true (A₁ n) (A₂ n) (A₃ n)) ∧ 
  (at_least_one_false (A₁ n) (A₂ n) (A₃ n)) ∧
  (at_least_one_true (B₁ n) (B₂ n) (B₃ n)) ∧ 
  (at_least_one_false (B₁ n) (B₂ n) (B₃ n)) ∧
  (at_least_one_true (C₁ n) (C₂ n) (C₃ n)) ∧ 
  (at_least_one_false (C₁ n) (C₂ n) (C₃ n)) ∧
  (at_least_one_true (D₁ n) (D₂ n) (D₃ n)) ∧ 
  (at_least_one_false (D₁ n) (D₂ n) (D₃ n)) →
  n = 89 := by
  sorry

end unique_number_theorem_l3891_389178


namespace woods_width_l3891_389195

theorem woods_width (area : ℝ) (length : ℝ) (width : ℝ) 
  (h1 : area = 24) 
  (h2 : length = 3) 
  (h3 : area = length * width) : width = 8 := by
sorry

end woods_width_l3891_389195


namespace new_train_distance_calculation_l3891_389115

/-- The distance traveled by the new train given the distance traveled by the old train and the percentage increase -/
def new_train_distance (old_distance : ℝ) (percent_increase : ℝ) : ℝ :=
  old_distance * (1 + percent_increase)

/-- Theorem: Given that a new train travels 30% farther than an old train in the same time,
    and the old train travels 300 miles, the new train travels 390 miles. -/
theorem new_train_distance_calculation :
  new_train_distance 300 0.3 = 390 := by
  sorry

#eval new_train_distance 300 0.3

end new_train_distance_calculation_l3891_389115


namespace vlads_pen_price_ratio_l3891_389196

/-- The ratio of gel pen price to ballpoint pen price given the conditions in Vlad's pen purchase problem -/
theorem vlads_pen_price_ratio :
  ∀ (x y : ℕ) (b g : ℝ),
  x > 0 → y > 0 → b > 0 → g > 0 →
  (x + y) * g = 4 * (x * b + y * g) →
  (x + y) * b = (1 / 2) * (x * b + y * g) →
  g = 8 * b := by
sorry

end vlads_pen_price_ratio_l3891_389196


namespace oil_leak_calculation_l3891_389144

theorem oil_leak_calculation (total_leak : ℕ) (leak_during_fix : ℕ) 
  (h1 : total_leak = 6206)
  (h2 : leak_during_fix = 3731) :
  total_leak - leak_during_fix = 2475 := by
  sorry

end oil_leak_calculation_l3891_389144


namespace library_book_count_l3891_389128

def library_books (initial_books : ℕ) (books_bought_two_years_ago : ℕ) (additional_books_last_year : ℕ) (books_donated : ℕ) : ℕ :=
  initial_books + books_bought_two_years_ago + (books_bought_two_years_ago + additional_books_last_year) - books_donated

theorem library_book_count : 
  library_books 500 300 100 200 = 1000 := by
  sorry

end library_book_count_l3891_389128


namespace manufacturing_employee_percentage_l3891_389165

theorem manufacturing_employee_percentage 
  (total_degrees : ℝ) 
  (manufacturing_degrees : ℝ) 
  (h1 : total_degrees = 360) 
  (h2 : manufacturing_degrees = 72) : 
  (manufacturing_degrees / total_degrees) * 100 = 20 := by
sorry

end manufacturing_employee_percentage_l3891_389165


namespace alice_wins_coin_game_l3891_389126

def coin_game (initial_coins : ℕ) : Prop :=
  ∃ (k : ℕ),
    k^2 ≤ initial_coins ∧
    initial_coins < k * (k + 1) - 1

theorem alice_wins_coin_game :
  coin_game 1331 :=
sorry

end alice_wins_coin_game_l3891_389126


namespace complex_multiplication_l3891_389118

def i : ℂ := Complex.I

theorem complex_multiplication :
  (1 + i) * (3 - i) = 4 + 2*i := by sorry

end complex_multiplication_l3891_389118


namespace distribute_seven_balls_three_boxes_l3891_389187

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes :
  distribute_balls 7 3 = 36 := by
  sorry


end distribute_seven_balls_three_boxes_l3891_389187


namespace slices_per_pizza_l3891_389141

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) 
  (h1 : total_pizzas = 17) 
  (h2 : total_slices = 68) : 
  total_slices / total_pizzas = 4 := by
  sorry

end slices_per_pizza_l3891_389141


namespace container_capacity_problem_l3891_389123

/-- Represents a rectangular container with dimensions and capacity -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- The problem statement -/
theorem container_capacity_problem (c1 c2 : Container) :
  c1.height = 4 →
  c1.width = 2 →
  c1.length = 8 →
  c1.capacity = 64 →
  c2.height = 3 * c1.height →
  c2.width = 2 * c1.width →
  c2.length = c1.length →
  c2.capacity = 384 := by
  sorry

end container_capacity_problem_l3891_389123


namespace soccer_team_starters_l3891_389101

theorem soccer_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quadruplets_in_lineup : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  quadruplets_in_lineup = 3 →
  (Nat.choose quadruplets quadruplets_in_lineup) * (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 880 :=
by sorry

end soccer_team_starters_l3891_389101


namespace avery_donation_ratio_l3891_389104

/-- Proves that the ratio of pants to shirts is 2:1 given the conditions of Avery's donation --/
theorem avery_donation_ratio :
  ∀ (pants : ℕ) (shorts : ℕ),
  let shirts := 4
  shorts = pants / 2 →
  shirts + pants + shorts = 16 →
  pants / shirts = 2 := by
sorry

end avery_donation_ratio_l3891_389104


namespace range_of_a_l3891_389171

/-- A function that is decreasing on R and defined piecewise --/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x) else -x^2 - 2*x + 1

/-- f is decreasing on R --/
axiom f_decreasing : ∀ x y : ℝ, x < y → f y < f x

/-- The theorem to prove --/
theorem range_of_a (a : ℝ) : f (a - 1) ≥ f (-a^2 + 1) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l3891_389171


namespace triangle_side_length_l3891_389177

namespace TriangleProof

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  AC : ℝ
  BC : ℝ
  cos_A : ℝ
  cos_B : ℝ
  h_cos_A : cos_A = 3/5
  h_cos_B : cos_B = 5/13
  h_AC : AC = 3

/-- The main theorem to prove -/
theorem triangle_side_length (t : Triangle) : t.AB = 14/5 := by
  sorry

end TriangleProof

end triangle_side_length_l3891_389177


namespace quadratic_roots_cube_l3891_389131

theorem quadratic_roots_cube (A B C : ℝ) (r s : ℝ) (h1 : A ≠ 0) :
  A * r^2 + B * r + C = 0 →
  A * s^2 + B * s + C = 0 →
  r ≠ s →
  ∃ q, (r^3)^2 + ((B^3 - 3*A*B*C) / A^3) * r^3 + q = 0 ∧
       (s^3)^2 + ((B^3 - 3*A*B*C) / A^3) * s^3 + q = 0 :=
by sorry

end quadratic_roots_cube_l3891_389131
