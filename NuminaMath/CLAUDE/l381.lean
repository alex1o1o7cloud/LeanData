import Mathlib

namespace fraction_problem_l381_38101

theorem fraction_problem (x : ℚ) : 
  x / (4 * x - 4) = 3 / 7 → x = 12 / 5 := by
  sorry

end fraction_problem_l381_38101


namespace mean_equality_implies_values_l381_38140

theorem mean_equality_implies_values (x y : ℝ) : 
  (2 + 11 + 6 + x) / 4 = (14 + 9 + y) / 3 → x = -35 ∧ y = -35 := by
  sorry

end mean_equality_implies_values_l381_38140


namespace lipschitz_periodic_bound_l381_38112

/-- A function f is k-Lipschitz if |f(x) - f(y)| ≤ k|x - y| for all x, y in the domain -/
def is_k_lipschitz (f : ℝ → ℝ) (k : ℝ) :=
  ∀ x y, |f x - f y| ≤ k * |x - y|

/-- A function f is periodic with period T if f(x + T) = f(x) for all x -/
def is_periodic (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x, f (x + T) = f x

theorem lipschitz_periodic_bound
  (f : ℝ → ℝ)
  (h_lipschitz : is_k_lipschitz f 1)
  (h_periodic : is_periodic f 2) :
  ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| ≤ 1 :=
sorry

end lipschitz_periodic_bound_l381_38112


namespace triangle_area_proof_l381_38183

theorem triangle_area_proof (z₁ z₂ : ℂ) (h1 : Complex.abs z₂ = 4) 
  (h2 : 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0) : 
  let O : ℂ := 0
  let P : ℂ := z₁
  let Q : ℂ := z₂
  Real.sqrt 3 * (Complex.abs (z₁ - O) * Complex.abs (z₂ - O) * Real.sin (Real.pi / 3)) = 4 * Real.sqrt 3 :=
by sorry

end triangle_area_proof_l381_38183


namespace central_cell_value_l381_38176

/-- Represents a 3x3 grid with numbers from 0 to 8 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two cells are adjacent -/
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Checks if the grid satisfies the consecutive number condition -/
def consecutive_condition (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, adjacent (i, j) (k, l) → (g i j).val + 1 = (g k l).val ∨ (g k l).val + 1 = (g i j).val

/-- Returns the sum of corner cell values in the grid -/
def corner_sum (g : Grid) : ℕ :=
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val

/-- The main theorem to be proved -/
theorem central_cell_value (g : Grid) 
  (h_consec : consecutive_condition g) 
  (h_corner_sum : corner_sum g = 18) :
  (g 1 1).val = 2 :=
sorry

end central_cell_value_l381_38176


namespace negative_square_times_cube_l381_38190

theorem negative_square_times_cube (x : ℝ) : (-x)^2 * x^3 = x^5 := by
  sorry

end negative_square_times_cube_l381_38190


namespace cookie_distribution_l381_38127

/-- The number of cookies Uncle Jude gave to Tim -/
def cookies_to_tim : ℕ := 15

/-- The total number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 256

/-- The number of cookies Uncle Jude gave to Mike -/
def cookies_to_mike : ℕ := 23

/-- The number of cookies Uncle Jude kept in the fridge -/
def cookies_in_fridge : ℕ := 188

theorem cookie_distribution :
  cookies_to_tim + cookies_to_mike + cookies_in_fridge + 2 * cookies_to_tim = total_cookies :=
by sorry

end cookie_distribution_l381_38127


namespace reflection_of_circle_center_l381_38107

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center : ℝ × ℝ := reflect_about_y_eq_neg_x original_center
  reflected_center = (3, -8) := by sorry

end reflection_of_circle_center_l381_38107


namespace prob_A_and_B_l381_38195

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.55

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.60

/-- The theorem stating that the probability of both A and B occurring simultaneously
    is equal to the product of their individual probabilities -/
theorem prob_A_and_B : prob_A * prob_B = 0.33 := by
  sorry

end prob_A_and_B_l381_38195


namespace fractional_exponent_simplification_l381_38116

theorem fractional_exponent_simplification (a : ℝ) (ha : a > 0) :
  a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by
  sorry

end fractional_exponent_simplification_l381_38116


namespace time_addition_and_digit_sum_l381_38145

/-- Represents time in a 12-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  isPM : Bool

/-- Represents a duration of time -/
structure Duration where
  hours : Nat
  minutes : Nat
  seconds : Nat

def addTime (t : Time) (d : Duration) : Time :=
  sorry

def sumDigits (t : Time) : Nat :=
  sorry

theorem time_addition_and_digit_sum :
  let initialTime : Time := ⟨3, 25, 15, true⟩
  let duration : Duration := ⟨137, 59, 59⟩
  let newTime := addTime initialTime duration
  newTime = ⟨9, 25, 14, true⟩ ∧ sumDigits newTime = 21 := by
  sorry

end time_addition_and_digit_sum_l381_38145


namespace no_solution_l381_38120

theorem no_solution : ¬∃ (k j x : ℝ), 
  (64 / k = 8) ∧ 
  (k * j = 128) ∧ 
  (j - x = k) ∧ 
  (x^2 + j = 3 * k) := by
  sorry

end no_solution_l381_38120


namespace inequality_theorem_l381_38150

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_prod : a * b * c = 1) 
  (h_ineq : a^2011 + b^2011 + c^2011 < 1/a^2011 + 1/b^2011 + 1/c^2011) : 
  a + b + c < 1/a + 1/b + 1/c := by
sorry

end inequality_theorem_l381_38150


namespace inequality_proof_l381_38102

theorem inequality_proof (a b c : ℝ) (h1 : c > 0) (h2 : a ≠ b) 
  (h3 : a^4 - 2019*a = c) (h4 : b^4 - 2019*b = c) : 
  -Real.sqrt c < a * b ∧ a * b < 0 := by
  sorry

end inequality_proof_l381_38102


namespace bills_remaining_money_bills_remaining_money_proof_l381_38134

/-- Calculates the amount of money Bill is left with after selling fool's gold and paying a fine -/
theorem bills_remaining_money (ounces_sold : ℕ) (price_per_ounce : ℕ) (fine : ℕ) : ℕ :=
  let total_earned := ounces_sold * price_per_ounce
  total_earned - fine

/-- Proves that Bill is left with $22 given the specific conditions -/
theorem bills_remaining_money_proof :
  bills_remaining_money 8 9 50 = 22 := by
  sorry

end bills_remaining_money_bills_remaining_money_proof_l381_38134


namespace snack_eaters_left_eq_30_l381_38158

/-- Represents the number of snack eaters who left after the second group of outsiders joined -/
def snack_eaters_left (initial_people : ℕ) (initial_snackers : ℕ) (first_outsiders : ℕ) (second_outsiders : ℕ) (final_snackers : ℕ) : ℕ :=
  let total_after_first := initial_snackers + first_outsiders
  let remaining_after_half_left := total_after_first / 2
  let total_after_second := remaining_after_half_left + second_outsiders
  let before_final_half_left := final_snackers * 2
  total_after_second - before_final_half_left

theorem snack_eaters_left_eq_30 :
  snack_eaters_left 200 100 20 10 20 = 30 := by
  sorry

end snack_eaters_left_eq_30_l381_38158


namespace birthday_money_allocation_l381_38180

theorem birthday_money_allocation (total : ℚ) (books snacks apps games : ℚ) : 
  total = 50 ∧ 
  books = (1 : ℚ) / 4 * total ∧
  snacks = (3 : ℚ) / 10 * total ∧
  apps = (7 : ℚ) / 20 * total ∧
  games = total - (books + snacks + apps) →
  games = 5 := by sorry

end birthday_money_allocation_l381_38180


namespace karen_cookie_distribution_l381_38103

/-- Calculates the number of cookies each person in Karen's class receives -/
def cookies_per_person (total_cookies : ℕ) (kept_cookies : ℕ) (grandparents_cookies : ℕ) (class_size : ℕ) : ℕ :=
  (total_cookies - kept_cookies - grandparents_cookies) / class_size

/-- Theorem stating that each person in Karen's class receives 2 cookies -/
theorem karen_cookie_distribution :
  cookies_per_person 50 10 8 16 = 2 := by
  sorry

#eval cookies_per_person 50 10 8 16

end karen_cookie_distribution_l381_38103


namespace evaluate_expression_l381_38136

theorem evaluate_expression : 8^6 * 27^6 * 8^27 * 27^8 = 2^99 * 3^42 := by
  sorry

end evaluate_expression_l381_38136


namespace triangle_area_l381_38163

theorem triangle_area (P Q R : ℝ) (r R : ℝ) (h1 : r = 3) (h2 : R = 15) 
  (h3 : 2 * Real.cos Q = Real.cos P + Real.cos R) : 
  ∃ (area : ℝ), area = 27 * Real.sqrt 21 := by
  sorry

end triangle_area_l381_38163


namespace equation_solution_l381_38157

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 4.5 ∧ x₂ = -3) ∧ 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -3 → (18 / (x^2 - 9) - 3 / (x - 3) = 2 ↔ (x = x₁ ∨ x = x₂))) := by
  sorry

end equation_solution_l381_38157


namespace power_product_equals_ten_thousand_l381_38100

theorem power_product_equals_ten_thousand : (2 ^ 4) * (5 ^ 4) = 10000 := by
  sorry

end power_product_equals_ten_thousand_l381_38100


namespace john_reps_per_set_l381_38167

/-- Given the weight per rep, number of sets, and total weight moved,
    calculate the number of reps per set. -/
def reps_per_set (weight_per_rep : ℕ) (num_sets : ℕ) (total_weight : ℕ) : ℕ :=
  (total_weight / weight_per_rep) / num_sets

/-- Prove that under the given conditions, John does 10 reps per set. -/
theorem john_reps_per_set :
  let weight_per_rep : ℕ := 15
  let num_sets : ℕ := 3
  let total_weight : ℕ := 450
  reps_per_set weight_per_rep num_sets total_weight = 10 := by
sorry

end john_reps_per_set_l381_38167


namespace percentage_of_students_passed_l381_38198

/-- Given an examination where 700 students appeared and 455 failed,
    prove that 35% of students passed the examination. -/
theorem percentage_of_students_passed (total : ℕ) (failed : ℕ) (h1 : total = 700) (h2 : failed = 455) :
  (total - failed : ℚ) / total * 100 = 35 := by
  sorry

end percentage_of_students_passed_l381_38198


namespace rainy_days_calculation_l381_38125

/-- Calculates the number of rainy days in a week given cycling conditions --/
def rainy_days (rain_speed : ℕ) (snow_speed : ℕ) (snow_days : ℕ) (total_distance : ℕ) : ℕ :=
  let snow_distance := snow_speed * snow_days
  (total_distance - snow_distance) / rain_speed

theorem rainy_days_calculation :
  rainy_days 90 30 4 390 = 3 := by
  sorry

end rainy_days_calculation_l381_38125


namespace total_score_is_248_l381_38154

/-- Calculates the total score across 4 subjects given 3 scores and the 4th as their average -/
def totalScoreAcross4Subjects (geography math english : ℕ) : ℕ :=
  let history := (geography + math + english) / 3
  geography + math + english + history

/-- Proves that given the specific scores, the total across 4 subjects is 248 -/
theorem total_score_is_248 :
  totalScoreAcross4Subjects 50 70 66 = 248 := by
  sorry

#eval totalScoreAcross4Subjects 50 70 66

end total_score_is_248_l381_38154


namespace perpendicular_tangents_ratio_l381_38124

/-- Given a line ax - by - 2 = 0 and a curve y = x³ with perpendicular tangents at point P(1,1),
    the value of b/a is -3. -/
theorem perpendicular_tangents_ratio (a b : ℝ) : 
  (∃ (x y : ℝ), a * x - b * y - 2 = 0 ∧ y = x^3) →  -- Line and curve equations
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ 
    (∀ (t : ℝ), a * t - b * (t^3) - 2 = 0 ↔ a * (x - t) + b * (y - t^3) = 0)) →  -- Perpendicular tangents at P(1,1)
  b / a = -3 := by
sorry

end perpendicular_tangents_ratio_l381_38124


namespace M₁_on_curve_M₂_not_on_curve_M₃_a_value_l381_38192

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (3 * t, 2 * t^2 + 1)

-- Define the points
def M₁ : ℝ × ℝ := (0, 1)
def M₂ : ℝ × ℝ := (5, 4)
def M₃ (a : ℝ) : ℝ × ℝ := (6, a)

-- Theorem statements
theorem M₁_on_curve : ∃ t : ℝ, curve_C t = M₁ := by sorry

theorem M₂_not_on_curve : ¬ ∃ t : ℝ, curve_C t = M₂ := by sorry

theorem M₃_a_value : ∃ a : ℝ, (∃ t : ℝ, curve_C t = M₃ a) → a = 9 := by sorry

end M₁_on_curve_M₂_not_on_curve_M₃_a_value_l381_38192


namespace yellow_ball_estimate_l381_38139

/-- Represents the contents of a bag with red and yellow balls -/
structure BagContents where
  red_balls : ℕ
  yellow_balls : ℕ

/-- Represents the result of multiple trials of drawing balls -/
structure TrialResults where
  num_trials : ℕ
  avg_red_ratio : ℝ

/-- Estimates the number of yellow balls in the bag based on trial results -/
def estimate_yellow_balls (bag : BagContents) (trials : TrialResults) : ℕ :=
  sorry

theorem yellow_ball_estimate (bag : BagContents) (trials : TrialResults) :
  bag.red_balls = 10 ∧ 
  trials.num_trials = 20 ∧ 
  trials.avg_red_ratio = 0.4 →
  estimate_yellow_balls bag trials = 15 :=
sorry

end yellow_ball_estimate_l381_38139


namespace inequality_property_l381_38184

theorem inequality_property (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a * c < b * c := by
  sorry

end inequality_property_l381_38184


namespace hyperbola_asymptotes_l381_38159

/-- Given a hyperbola with equation x²/4 - y²/9 = 1, its asymptotes are y = ±(3/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 4 - y^2 / 9 = 1 →
  ∃ (k : ℝ), k = 3/2 ∧ (y = k*x ∨ y = -k*x) :=
by sorry

end hyperbola_asymptotes_l381_38159


namespace indeterminate_roots_of_related_quadratic_l381_38128

/-- Given positive numbers a, b, c, and a quadratic equation with two equal real roots,
    the nature of the roots of a related quadratic equation cannot be determined. -/
theorem indeterminate_roots_of_related_quadratic
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_equal_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (∀ y : ℝ, a * y^2 + b * y + c = 0 → y = x)) :
  ∃ (r₁ r₂ : ℝ), (a + 1) * r₁^2 + (b + 2) * r₁ + (c + 1) = 0 ∧
                 (a + 1) * r₂^2 + (b + 2) * r₂ + (c + 1) = 0 ∧
                 (r₁ = r₂ ∨ r₁ ≠ r₂) :=
sorry

end indeterminate_roots_of_related_quadratic_l381_38128


namespace good_pair_exists_l381_38151

theorem good_pair_exists (m : ℕ) : ∃ n : ℕ, n > m ∧ 
  ∃ a b : ℕ, m * n = a^2 ∧ (m + 1) * (n + 1) = b^2 := by
  sorry

end good_pair_exists_l381_38151


namespace pure_imaginary_complex_product_l381_38196

theorem pure_imaginary_complex_product (a : ℝ) : 
  (Complex.im ((1 + a * Complex.I) * (3 - Complex.I)) ≠ 0 ∧ 
   Complex.re ((1 + a * Complex.I) * (3 - Complex.I)) = 0) → 
  a = -3 := by
  sorry

end pure_imaginary_complex_product_l381_38196


namespace sports_club_overlap_l381_38156

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) : 
  total = 150 → badminton = 75 → tennis = 85 → neither = 15 → 
  badminton + tennis - (total - neither) = 25 := by
sorry

end sports_club_overlap_l381_38156


namespace complement_A_intersect_B_l381_38142

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {-2, -1, 0}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end complement_A_intersect_B_l381_38142


namespace concert_drive_l381_38126

/-- Given a total distance and a distance already driven, calculate the remaining distance. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem: Given a total distance of 78 miles and having driven 32 miles, 
    the remaining distance to drive is 46 miles. -/
theorem concert_drive : remaining_distance 78 32 = 46 := by
  sorry

end concert_drive_l381_38126


namespace roots_sum_product_l381_38185

theorem roots_sum_product (a b : ℝ) : 
  (a^4 - 6*a - 1 = 0) →
  (b^4 - 6*b - 1 = 0) →
  (a ≠ b) →
  (ab + 2*a + 2*b = 1.5 + Real.sqrt 3) :=
by sorry

end roots_sum_product_l381_38185


namespace trigonometric_shift_l381_38147

/-- Proves that √3 * sin(2x) - cos(2x) is equivalent to 2 * sin(2(x + π/12)) --/
theorem trigonometric_shift (x : ℝ) : 
  Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x) = 2 * Real.sin (2 * (x + Real.pi / 12)) :=
by sorry

end trigonometric_shift_l381_38147


namespace q_value_l381_38194

theorem q_value (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 12) : q = 6 + 2 * Real.sqrt 6 := by
  sorry

end q_value_l381_38194


namespace max_pairs_remaining_l381_38173

/-- Represents the total number of shoe pairs -/
def total_pairs : ℕ := 27

/-- Represents the number of individual shoes lost -/
def shoes_lost : ℕ := 9

/-- Theorem stating the maximum number of complete pairs remaining after losing shoes -/
theorem max_pairs_remaining (total : ℕ) (lost : ℕ) : 
  total = total_pairs → lost = shoes_lost → total - lost ≤ 18 := by
  sorry

#check max_pairs_remaining

end max_pairs_remaining_l381_38173


namespace roots_of_equation_l381_38168

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => x * (x - 3)^2 * (5 + x)
  {x : ℝ | f x = 0} = {0, 3, -5} := by
sorry

end roots_of_equation_l381_38168


namespace triangle_area_from_altitudes_l381_38174

/-- A triangle with given altitudes has a specific area -/
theorem triangle_area_from_altitudes (h₁ h₂ h₃ : ℝ) (h_pos₁ : h₁ > 0) (h_pos₂ : h₂ > 0) (h_pos₃ : h₃ > 0) :
  h₁ = 12 → h₂ = 15 → h₃ = 20 → ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * h₁ = 2 * 150) ∧ (b * h₂ = 2 * 150) ∧ (c * h₃ = 2 * 150) :=
by sorry

#check triangle_area_from_altitudes

end triangle_area_from_altitudes_l381_38174


namespace rebecca_eggs_l381_38135

/-- The number of marbles Rebecca has -/
def marbles : ℕ := 6

/-- The difference between the number of eggs and marbles -/
def egg_marble_difference : ℕ := 14

/-- The number of eggs Rebecca has -/
def eggs : ℕ := marbles + egg_marble_difference

theorem rebecca_eggs : eggs = 20 := by
  sorry

end rebecca_eggs_l381_38135


namespace final_score_calculation_l381_38132

theorem final_score_calculation (innovation_score comprehensive_score language_score : ℝ)
  (innovation_weight comprehensive_weight language_weight : ℝ) :
  innovation_score = 88 →
  comprehensive_score = 80 →
  language_score = 75 →
  innovation_weight = 5 →
  comprehensive_weight = 3 →
  language_weight = 2 →
  (innovation_score * innovation_weight + comprehensive_score * comprehensive_weight + language_score * language_weight) /
    (innovation_weight + comprehensive_weight + language_weight) = 83 := by
  sorry

end final_score_calculation_l381_38132


namespace vector_properties_l381_38155

open Real

/-- Given vectors satisfying certain conditions, prove parallelism and angle between vectors -/
theorem vector_properties (a b c : ℝ × ℝ) : 
  (3 • a - 2 • b = (2, 6)) → 
  (a + 2 • b = (6, 2)) → 
  (c = (1, 1)) → 
  (∃ (k : ℝ), a = k • c) ∧ 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.sqrt 2 / 2 := by
  sorry

end vector_properties_l381_38155


namespace sum_of_squares_in_ratio_l381_38160

theorem sum_of_squares_in_ratio (a b c : ℚ) : 
  (a : ℚ) + b + c = 15 →
  b = 2 * a →
  c = 4 * a →
  a^2 + b^2 + c^2 = 4725 / 49 := by
sorry

end sum_of_squares_in_ratio_l381_38160


namespace frog_hop_probability_l381_38171

/-- Represents a position on the 4x4 grid -/
inductive Position
| Inner : Fin 2 → Fin 2 → Position
| Edge : Fin 4 → Fin 4 → Position

/-- Represents a possible hop direction -/
inductive Direction
| Up | Down | Left | Right

/-- The grid size -/
def gridSize : Nat := 4

/-- The maximum number of hops -/
def maxHops : Nat := 5

/-- Function to determine if a position is on the edge -/
def isEdge (p : Position) : Bool :=
  match p with
  | Position.Edge _ _ => true
  | _ => false

/-- Function to perform a single hop -/
def hop (p : Position) (d : Direction) : Position :=
  sorry

/-- Function to calculate the probability of reaching an edge within n hops -/
def probReachEdge (start : Position) (n : Nat) : Rat :=
  sorry

/-- The starting position (second square in the second row) -/
def startPosition : Position := Position.Inner 1 1

/-- The main theorem to prove -/
theorem frog_hop_probability :
  probReachEdge startPosition maxHops = 94 / 256 := by
  sorry

end frog_hop_probability_l381_38171


namespace junior_score_l381_38144

theorem junior_score (total_students : ℕ) (junior_percent senior_percent : ℚ) 
  (overall_avg senior_avg junior_score : ℚ) : 
  junior_percent = 1/5 →
  senior_percent = 4/5 →
  junior_percent + senior_percent = 1 →
  overall_avg = 85 →
  senior_avg = 83 →
  (junior_percent * junior_score + senior_percent * senior_avg = overall_avg) →
  junior_score = 93 := by
sorry

end junior_score_l381_38144


namespace base7_arithmetic_l381_38152

/-- Represents a number in base 7 --/
structure Base7 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 7

/-- Addition operation for Base7 numbers --/
def add_base7 (a b : Base7) : Base7 := sorry

/-- Subtraction operation for Base7 numbers --/
def sub_base7 (a b : Base7) : Base7 := sorry

/-- Conversion from a natural number to Base7 --/
def nat_to_base7 (n : Nat) : Base7 := sorry

theorem base7_arithmetic :
  let a := nat_to_base7 24
  let b := nat_to_base7 356
  let c := nat_to_base7 105
  let d := nat_to_base7 265
  sub_base7 (add_base7 a b) c = d := by sorry

end base7_arithmetic_l381_38152


namespace room_painting_problem_l381_38165

/-- The total area of a room painted by two painters working together --/
def room_area (painter1_rate : ℝ) (painter2_rate : ℝ) (slowdown : ℝ) (time : ℝ) : ℝ :=
  time * (painter1_rate + painter2_rate - slowdown)

theorem room_painting_problem :
  let painter1_rate := 1 / 6
  let painter2_rate := 1 / 8
  let slowdown := 5
  let time := 4
  room_area painter1_rate painter2_rate slowdown time = 120 := by
sorry

end room_painting_problem_l381_38165


namespace recurring_decimal_sum_l381_38186

/-- Represents a recurring decimal with a single digit repeating -/
def RecurringDecimal (d : ℕ) : ℚ :=
  d / 9

theorem recurring_decimal_sum :
  let a := RecurringDecimal 5
  let b := RecurringDecimal 1
  let c := RecurringDecimal 3
  let d := RecurringDecimal 6
  a + b - c + d = 1 := by sorry

end recurring_decimal_sum_l381_38186


namespace sum_x₁_x₂_equals_three_l381_38191

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  p₂ : ℝ
  h_prob_sum : p₁ + p₂ = 1
  h_prob_pos : 0 < p₁ ∧ 0 < p₂

/-- The expected value of a discrete random variable -/
def expected_value (X : DiscreteRV) : ℝ := X.x₁ * X.p₁ + X.x₂ * X.p₂

/-- The variance of a discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  X.p₁ * (X.x₁ - expected_value X)^2 + X.p₂ * (X.x₂ - expected_value X)^2

/-- Theorem stating the sum of x₁ and x₂ for the given conditions -/
theorem sum_x₁_x₂_equals_three (X : DiscreteRV)
  (h_p₁ : X.p₁ = 2/3)
  (h_p₂ : X.p₂ = 1/3)
  (h_order : X.x₁ < X.x₂)
  (h_exp : expected_value X = 4/3)
  (h_var : variance X = 2/9) :
  X.x₁ + X.x₂ = 3 := by
  sorry

end sum_x₁_x₂_equals_three_l381_38191


namespace gym_membership_duration_is_three_years_l381_38172

/-- Calculates the duration of a gym membership in years given the monthly cost,
    down payment, and total cost. -/
def gym_membership_duration (monthly_cost : ℚ) (down_payment : ℚ) (total_cost : ℚ) : ℚ :=
  ((total_cost - down_payment) / monthly_cost) / 12

/-- Proves that given the specific costs, the gym membership duration is 3 years. -/
theorem gym_membership_duration_is_three_years :
  gym_membership_duration 12 50 482 = 3 := by
  sorry

#eval gym_membership_duration 12 50 482

end gym_membership_duration_is_three_years_l381_38172


namespace median_length_l381_38137

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10
  right_angle : a^2 + b^2 = c^2

/-- The length of the median to the longest side of the triangle -/
def median_to_longest_side (t : RightTriangle) : ℝ := 5

/-- Theorem: The length of the median to the longest side is 5 -/
theorem median_length (t : RightTriangle) : median_to_longest_side t = 5 := by
  sorry

end median_length_l381_38137


namespace mean_equality_implies_z_l381_38130

theorem mean_equality_implies_z (z : ℚ) : 
  (7 + 10 + 23) / 3 = (18 + z) / 2 → z = 26 / 3 := by
sorry

end mean_equality_implies_z_l381_38130


namespace max_value_of_expression_l381_38175

theorem max_value_of_expression (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) :
  ∃ (max_val : ℝ), max_val = 15 ∧ ∀ (x' y' : ℝ), 2 * x'^2 - 6 * x' + y'^2 = 0 →
    x'^2 + y'^2 + 2 * x' ≤ max_val :=
by sorry

end max_value_of_expression_l381_38175


namespace evaluate_expression_l381_38162

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 0) : z * (z - 4 * x) = 0 := by
  sorry

end evaluate_expression_l381_38162


namespace min_value_of_product_l381_38138

theorem min_value_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 47 := by
  sorry

end min_value_of_product_l381_38138


namespace complex_number_in_fourth_quadrant_l381_38141

/-- The complex number -2i+1 corresponds to a point in the fourth quadrant of the complex plane -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := -2 * Complex.I + 1
  (z.re > 0) ∧ (z.im < 0) := by sorry

end complex_number_in_fourth_quadrant_l381_38141


namespace total_visitors_is_440_l381_38109

/-- Represents the survey results of visitors to a Picasso painting exhibition -/
structure SurveyResults where
  totalVisitors : ℕ
  didNotEnjoyOrUnderstand : ℕ
  enjoyedAndUnderstood : ℕ

/-- The conditions of the survey results -/
def surveyConditions (results : SurveyResults) : Prop :=
  results.didNotEnjoyOrUnderstand = 110 ∧
  results.enjoyedAndUnderstood = 3 * results.totalVisitors / 4 ∧
  results.totalVisitors = results.enjoyedAndUnderstood + results.didNotEnjoyOrUnderstand

/-- The theorem stating that given the survey conditions, the total number of visitors is 440 -/
theorem total_visitors_is_440 (results : SurveyResults) :
  surveyConditions results → results.totalVisitors = 440 := by
  sorry

#check total_visitors_is_440

end total_visitors_is_440_l381_38109


namespace percentage_not_receiving_muffin_l381_38166

theorem percentage_not_receiving_muffin (total_percentage : ℝ) (muffin_percentage : ℝ) 
  (h1 : total_percentage = 100) 
  (h2 : muffin_percentage = 38) : 
  total_percentage - muffin_percentage = 62 := by
  sorry

end percentage_not_receiving_muffin_l381_38166


namespace exists_valid_sequence_l381_38117

/-- A sequence of natural numbers satisfying the given conditions -/
def ValidSequence (s : List Nat) : Prop :=
  s.length > 10 ∧
  s.sum = 20 ∧
  3 ∉ s ∧
  ∀ i j, i ≤ j → j < s.length → (s.take (j + 1)).drop i ≠ [3]

/-- Theorem stating the existence of a valid sequence -/
theorem exists_valid_sequence : ∃ s : List Nat, ValidSequence s := by
  sorry

end exists_valid_sequence_l381_38117


namespace coffee_stock_problem_l381_38123

/-- Proves that given the conditions of the coffee stock problem, 
    the percentage of the initial stock that was decaffeinated is 20%. -/
theorem coffee_stock_problem (initial_stock : ℝ) (additional_purchase : ℝ) 
  (decaf_percent_new : ℝ) (total_decaf_percent : ℝ) :
  initial_stock = 400 →
  additional_purchase = 100 →
  decaf_percent_new = 60 →
  total_decaf_percent = 28.000000000000004 →
  (initial_stock * (20 / 100) + additional_purchase * (decaf_percent_new / 100)) / 
  (initial_stock + additional_purchase) * 100 = total_decaf_percent :=
by
  sorry

end coffee_stock_problem_l381_38123


namespace older_ate_twelve_l381_38113

/-- Represents the pancake eating scenario -/
structure PancakeScenario where
  initial_pancakes : ℕ
  final_pancakes : ℕ
  younger_eats : ℕ
  older_eats : ℕ
  grandma_bakes : ℕ

/-- Calculates the number of pancakes eaten by the older grandchild -/
def older_grandchild_pancakes (scenario : PancakeScenario) : ℕ :=
  let net_reduction := scenario.younger_eats + scenario.older_eats - scenario.grandma_bakes
  let cycles := (scenario.initial_pancakes - scenario.final_pancakes) / net_reduction
  scenario.older_eats * cycles

/-- Theorem stating that the older grandchild ate 12 pancakes in the given scenario -/
theorem older_ate_twelve (scenario : PancakeScenario) 
  (h1 : scenario.initial_pancakes = 19)
  (h2 : scenario.final_pancakes = 11)
  (h3 : scenario.younger_eats = 1)
  (h4 : scenario.older_eats = 3)
  (h5 : scenario.grandma_bakes = 2) :
  older_grandchild_pancakes scenario = 12 := by
  sorry

#eval older_grandchild_pancakes { 
  initial_pancakes := 19, 
  final_pancakes := 11, 
  younger_eats := 1, 
  older_eats := 3, 
  grandma_bakes := 2 
}

end older_ate_twelve_l381_38113


namespace parallel_vectors_sum_magnitude_l381_38133

/-- Given two parallel vectors p and q, prove that their sum has a magnitude of √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) :
  p = (2, -3) →
  q.1 = x ∧ q.2 = 6 →
  (∃ (k : ℝ), q = k • p) →
  ‖p + q‖ = Real.sqrt 13 := by
  sorry

end parallel_vectors_sum_magnitude_l381_38133


namespace pumpkin_pie_degrees_l381_38148

/-- Represents the preference distribution of pies in a class --/
structure PiePreference where
  total : ℕ
  peach : ℕ
  apple : ℕ
  blueberry : ℕ
  pumpkin : ℕ
  banana : ℕ

/-- Calculates the degrees for a given pie in a pie chart --/
def degreesForPie (pref : PiePreference) (pieCount : ℕ) : ℚ :=
  (pieCount : ℚ) / (pref.total : ℚ) * 360

/-- Theorem stating the degrees for pumpkin pie in Jeremy's class --/
theorem pumpkin_pie_degrees (pref : PiePreference) 
  (h1 : pref.total = 40)
  (h2 : pref.peach = 14)
  (h3 : pref.apple = 9)
  (h4 : pref.blueberry = 7)
  (h5 : pref.pumpkin = pref.banana)
  (h6 : pref.pumpkin + pref.banana = pref.total - (pref.peach + pref.apple + pref.blueberry)) :
  degreesForPie pref pref.pumpkin = 45 := by
  sorry


end pumpkin_pie_degrees_l381_38148


namespace ones_digit_of_largest_power_of_two_dividing_32_factorial_l381_38188

def largest_power_of_two_dividing_factorial (n : ℕ) : ℕ :=
  (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (2^(largest_power_of_two_dividing_factorial 32)) = 4 := by
  sorry

end ones_digit_of_largest_power_of_two_dividing_32_factorial_l381_38188


namespace divisibility_property_l381_38189

theorem divisibility_property (p : ℕ) (hp : p > 3) (hodd : Odd p) :
  ∃ k : ℤ, (p - 3) ^ ((p - 1) / 2) - 1 = k * (p - 4) := by
  sorry

end divisibility_property_l381_38189


namespace angle_U_is_90_degrees_l381_38199

-- Define the hexagon FIGURE
structure Hexagon where
  F : ℝ
  I : ℝ
  U : ℝ
  G : ℝ
  R : ℝ
  E : ℝ

-- Define the conditions
def hexagon_conditions (h : Hexagon) : Prop :=
  h.F = h.I ∧ h.I = h.U ∧ 
  h.G + h.E = 180 ∧ 
  h.R + h.U = 180 ∧
  h.F + h.I + h.U + h.G + h.R + h.E = 720

-- Theorem statement
theorem angle_U_is_90_degrees (h : Hexagon) 
  (hc : hexagon_conditions h) : h.U = 90 := by sorry

end angle_U_is_90_degrees_l381_38199


namespace distance_thirty_students_l381_38122

/-- The distance between the first and last student in a line of students -/
def distance_between_ends (num_students : ℕ) (gap_distance : ℝ) : ℝ :=
  (num_students - 1 : ℝ) * gap_distance

/-- Theorem: For 30 students standing in a line with 3 meters between adjacent students,
    the distance between the first and last student is 87 meters. -/
theorem distance_thirty_students :
  distance_between_ends 30 3 = 87 := by
  sorry

end distance_thirty_students_l381_38122


namespace right_triangle_expansion_l381_38146

theorem right_triangle_expansion : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  a^2 + b^2 = c^2 ∧
  (a + 100)^2 + (b + 100)^2 = (c + 140)^2 := by
  sorry

end right_triangle_expansion_l381_38146


namespace molecular_weight_N2O5_l381_38105

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (n_count : ℕ) (o_count : ℕ) : ℝ :=
  n_count * atomic_weight_N + o_count * atomic_weight_O

/-- Theorem stating that the molecular weight of N2O5 is 108.02 g/mol -/
theorem molecular_weight_N2O5 : 
  molecular_weight 2 5 = 108.02 := by sorry

end molecular_weight_N2O5_l381_38105


namespace min_workers_for_profit_l381_38169

/-- Represents the problem of determining the minimum number of workers needed for profit --/
theorem min_workers_for_profit 
  (maintenance_fee : ℝ) 
  (hourly_wage : ℝ) 
  (hourly_production : ℝ) 
  (widget_price : ℝ) 
  (work_hours : ℝ)
  (h1 : maintenance_fee = 600)
  (h2 : hourly_wage = 20)
  (h3 : hourly_production = 3)
  (h4 : widget_price = 2.80)
  (h5 : work_hours = 8) :
  (∃ n : ℕ, n * hourly_production * work_hours * widget_price > 
             maintenance_fee + n * hourly_wage * work_hours) ∧
  (∀ m : ℕ, m * hourly_production * work_hours * widget_price > 
             maintenance_fee + m * hourly_wage * work_hours → m ≥ 7) :=
by sorry

end min_workers_for_profit_l381_38169


namespace roses_in_garden_l381_38149

theorem roses_in_garden (total_pink : ℕ) (roses_per_row : ℕ) 
  (h1 : roses_per_row = 20)
  (h2 : total_pink = 40) : 
  (total_pink / (roses_per_row * (1 - 1/2) * (1 - 3/5))) = 10 := by
  sorry

end roses_in_garden_l381_38149


namespace interior_edge_sum_is_twenty_l381_38129

/-- A rectangular picture frame with specific properties -/
structure PictureFrame where
  outer_length : ℝ
  outer_width : ℝ
  frame_width : ℝ
  frame_area : ℝ
  outer_length_given : outer_length = 7
  frame_width_given : frame_width = 1
  frame_area_given : frame_area = 24
  positive_dimensions : outer_length > 0 ∧ outer_width > 0

/-- The sum of the interior edge lengths of the picture frame -/
def interior_edge_sum (frame : PictureFrame) : ℝ :=
  2 * (frame.outer_length - 2 * frame.frame_width) + 2 * (frame.outer_width - 2 * frame.frame_width)

/-- Theorem stating that the sum of interior edge lengths is 20 -/
theorem interior_edge_sum_is_twenty (frame : PictureFrame) :
  interior_edge_sum frame = 20 := by
  sorry


end interior_edge_sum_is_twenty_l381_38129


namespace evaluate_expression_l381_38178

theorem evaluate_expression : (-2 : ℤ) ^ (3^2) + 2 ^ (3^2) = 0 := by
  sorry

end evaluate_expression_l381_38178


namespace adam_final_score_l381_38170

def trivia_game (first_half_correct : ℕ) (second_half_correct : ℕ) 
                (first_half_points : ℕ) (second_half_points : ℕ) 
                (bonus_points : ℕ) (penalty : ℕ) (total_questions : ℕ) : ℕ :=
  let correct_points := first_half_correct * first_half_points + second_half_correct * second_half_points
  let total_correct := first_half_correct + second_half_correct
  let bonus := (total_correct / 3) * bonus_points
  let incorrect := total_questions - total_correct
  let penalty_points := incorrect * penalty
  correct_points + bonus - penalty_points

theorem adam_final_score : 
  trivia_game 15 12 3 5 2 1 35 = 115 := by sorry

end adam_final_score_l381_38170


namespace nested_radical_value_l381_38164

theorem nested_radical_value : 
  ∃ x : ℝ, x = Real.sqrt (20 + x) ∧ x > 0 → x = 5 := by
  sorry

end nested_radical_value_l381_38164


namespace painting_time_relation_l381_38131

/-- Time taken by Taylor to paint the room alone -/
def taylor_time : ℝ := 12

/-- Time taken by Taylor and Jennifer together to paint the room -/
def combined_time : ℝ := 5.45454545455

/-- Time taken by Jennifer to paint the room alone -/
def jennifer_time : ℝ := 10.1538461538

/-- Theorem stating the relationship between individual and combined painting times -/
theorem painting_time_relation :
  1 / taylor_time + 1 / jennifer_time = 1 / combined_time :=
sorry

end painting_time_relation_l381_38131


namespace second_row_starts_with_531_l381_38115

-- Define the grid type
def Grid := Fin 3 → Fin 3 → Nat

-- Define the valid range of numbers
def ValidNumber (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 5

-- No repetition in rows
def NoRowRepetition (grid : Grid) : Prop :=
  ∀ i j k, j ≠ k → grid i j ≠ grid i k

-- No repetition in columns
def NoColumnRepetition (grid : Grid) : Prop :=
  ∀ i j k, i ≠ k → grid i j ≠ grid k j

-- Divisibility condition
def DivisibilityCondition (grid : Grid) : Prop :=
  ∀ i j, i > 0 → grid i j % grid (i-1) j = 0 ∧
  ∀ i j, j > 0 → grid i j % grid i (j-1) = 0

-- All numbers are valid
def AllValidNumbers (grid : Grid) : Prop :=
  ∀ i j, ValidNumber (grid i j)

-- Main theorem
theorem second_row_starts_with_531 (grid : Grid) 
  (h1 : NoRowRepetition grid)
  (h2 : NoColumnRepetition grid)
  (h3 : DivisibilityCondition grid)
  (h4 : AllValidNumbers grid) :
  grid 1 0 = 5 ∧ grid 1 1 = 1 ∧ grid 1 2 = 3 := by
  sorry

end second_row_starts_with_531_l381_38115


namespace rectangle_dimensions_l381_38111

theorem rectangle_dimensions (length width : ℝ) : 
  length > 0 → width > 0 → 
  length * width = 120 → 
  2 * (length + width) = 46 → 
  min length width = 8 :=
by
  sorry

end rectangle_dimensions_l381_38111


namespace percentage_calculation_l381_38108

theorem percentage_calculation (x : ℝ) (h : 0.255 * x = 153) : 0.678 * x = 406.8 := by
  sorry

end percentage_calculation_l381_38108


namespace second_group_size_l381_38187

/-- The number of persons in the first group -/
def first_group : ℕ := 78

/-- The number of days the first group works -/
def first_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_hours : ℕ := 5

/-- The number of days the second group works -/
def second_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_hours : ℕ := 6

/-- The total man-hours required to complete the job -/
def total_man_hours : ℕ := first_group * first_days * first_hours

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (second_days * second_hours)

theorem second_group_size :
  second_group = 130 := by sorry

end second_group_size_l381_38187


namespace bowler_previous_wickets_l381_38197

/-- Bowling average calculation -/
def bowling_average (runs : ℚ) (wickets : ℚ) : ℚ := runs / wickets

theorem bowler_previous_wickets 
  (initial_average : ℚ)
  (last_match_wickets : ℚ)
  (last_match_runs : ℚ)
  (average_decrease : ℚ)
  (h1 : initial_average = 12.4)
  (h2 : last_match_wickets = 7)
  (h3 : last_match_runs = 26)
  (h4 : average_decrease = 0.4) :
  ∃ (previous_wickets : ℚ),
    previous_wickets = 145 ∧
    bowling_average (initial_average * previous_wickets + last_match_runs) (previous_wickets + last_match_wickets) = initial_average - average_decrease :=
sorry

end bowler_previous_wickets_l381_38197


namespace expression_value_l381_38110

theorem expression_value (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : x = 4) :
  3 * x^2 + 12 * x * y + y^2 = 1 := by
  sorry

end expression_value_l381_38110


namespace no_integer_solution_l381_38179

theorem no_integer_solution : ¬∃ (a b : ℤ), a^2 + b^2 = 10^100 + 3 := by sorry

end no_integer_solution_l381_38179


namespace proportion_equality_l381_38177

/-- Given a proportion x : 6 :: 2 : 0.19999999999999998, prove that x = 60 -/
theorem proportion_equality : 
  ∀ x : ℝ, (x / 6 = 2 / 0.19999999999999998) → x = 60 := by
  sorry

end proportion_equality_l381_38177


namespace quadratic_real_roots_l381_38119

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k + Complex.I)*x - 2 - k*Complex.I = 0) ↔ (k = 1 ∨ k = -1) := by
  sorry

end quadratic_real_roots_l381_38119


namespace circles_symmetry_implies_sin_cos_theta_l381_38161

/-- Circle C₁ -/
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + a*x = 0

/-- Circle C₂ -/
def C₂ (a θ : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*a*x + y*Real.tan θ = 0

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

/-- Main theorem -/
theorem circles_symmetry_implies_sin_cos_theta (a θ : ℝ) :
  (∀ x y, C₁ a x y ↔ C₁ a ((2*x-1)/2) (2*x-1-y)) →
  (∀ x y, C₂ a θ x y ↔ C₂ a θ ((2*x-1)/2) (2*x-1-y)) →
  Real.sin θ * Real.cos θ = -2/5 := by sorry

end circles_symmetry_implies_sin_cos_theta_l381_38161


namespace hexagon_interior_angles_sum_l381_38104

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end hexagon_interior_angles_sum_l381_38104


namespace expression_simplification_l381_38143

theorem expression_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (2 * x) / (x + 1) - (2 * x + 4) / (x^2 - 1) / ((x + 2) / (x^2 - 2 * x + 1)) = 2 / (x + 1) :=
by sorry

end expression_simplification_l381_38143


namespace arithmetic_sequence_sum_l381_38121

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- Given an arithmetic sequence a where a₁ + a₃ + a₅ = 3, prove a₂ + a₄ = 2 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ArithmeticSequence a) 
    (h2 : a 1 + a 3 + a 5 = 3) : a 2 + a 4 = 2 := by
  sorry

end arithmetic_sequence_sum_l381_38121


namespace congruence_solution_l381_38106

theorem congruence_solution (x : ℤ) :
  x ≡ 6 [ZMOD 17] → 15 * x + 2 ≡ 7 [ZMOD 17] := by
  sorry

end congruence_solution_l381_38106


namespace next_five_even_sum_l381_38182

/-- Given a sum 'a' of 5 consecutive even positive integers, 
    the sum of the next 5 consecutive even integers is a + 50 -/
theorem next_five_even_sum (a : ℕ) (x : ℕ) 
  (h1 : x > 0) 
  (h2 : a = x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) : 
  (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18) = a + 50 := by
  sorry

end next_five_even_sum_l381_38182


namespace mary_money_left_l381_38181

/-- The amount of money Mary has left after her purchases -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 2 * drink_cost + medium_pizza_cost + 2 * large_pizza_cost
  50 - total_cost

/-- Theorem stating that the amount of money Mary has left is 50 - 10p -/
theorem mary_money_left (p : ℝ) : money_left p = 50 - 10 * p := by
  sorry

end mary_money_left_l381_38181


namespace floor_of_5_7_l381_38118

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by sorry

end floor_of_5_7_l381_38118


namespace complex_multiplication_l381_38114

theorem complex_multiplication : (1 - 2*Complex.I) * (3 + 4*Complex.I) * (-1 + Complex.I) = -9 + 13*Complex.I := by
  sorry

end complex_multiplication_l381_38114


namespace divisor_problem_l381_38153

theorem divisor_problem (n : ℤ) (d : ℤ) : 
  (∃ k : ℤ, n = 18 * k + 10) → 
  (∃ q : ℤ, 2 * n = d * q + 2) → 
  d = 18 := by
sorry

end divisor_problem_l381_38153


namespace quadratic_roots_from_intersections_l381_38193

/-- Given a quadratic function f(x) = ax² + bx + c, if its graph intersects
    the x-axis at (1,0) and (4,0), then the solutions to ax² + bx + c = 0
    are x₁ = 1 and x₂ = 4. -/
theorem quadratic_roots_from_intersections
  (a b c : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a * x^2 + b * x + c) :
  f 1 = 0 → f 4 = 0 →
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 4 ∧ ∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂ :=
sorry

end quadratic_roots_from_intersections_l381_38193
