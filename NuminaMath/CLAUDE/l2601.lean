import Mathlib

namespace NUMINAMATH_CALUDE_parallel_planes_sum_l2601_260132

/-- Given two planes α and β with normal vectors (x, 1, -2) and (-1, y, 1/2) respectively,
    if α is parallel to β, then x + y = 15/4 -/
theorem parallel_planes_sum (x y : ℝ) : 
  let n₁ : Fin 3 → ℝ := ![x, 1, -2]
  let n₂ : Fin 3 → ℝ := ![-1, y, 1/2]
  (∃ (k : ℝ), ∀ i, n₁ i = k * n₂ i) →
  x + y = 15/4 := by
sorry

end NUMINAMATH_CALUDE_parallel_planes_sum_l2601_260132


namespace NUMINAMATH_CALUDE_solution_pairs_l2601_260155

/-- The type of pairs of positive integers satisfying the divisibility condition -/
def SolutionPairs : Type := 
  {p : Nat × Nat // p.1 > 0 ∧ p.2 > 0 ∧ (2^(2^p.1) + 1) * (2^(2^p.2) + 1) % (p.1 * p.2) = 0}

/-- The theorem stating the solution pairs -/
theorem solution_pairs : 
  {p : SolutionPairs | p.val = (1, 1) ∨ p.val = (1, 5) ∨ p.val = (5, 1)} = 
  {p : SolutionPairs | true} := by sorry

end NUMINAMATH_CALUDE_solution_pairs_l2601_260155


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l2601_260152

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 3 * x - 9/4 = 0) ↔ (k > -1 ∨ k < -1) ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l2601_260152


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2601_260158

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 15 players where each player plays every other player once, 
    the total number of games played is 105. -/
theorem chess_tournament_games : num_games 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2601_260158


namespace NUMINAMATH_CALUDE_area_of_triangle_AEC_l2601_260141

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the properties of the rectangle and point E
def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry

def on_segment (E C D : ℝ × ℝ) : Prop := sorry

def segment_ratio (D E C : ℝ × ℝ) (r : ℚ) : Prop := sorry

def triangle_area (A D E : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_AEC 
  (h_rectangle : is_rectangle A B C D)
  (h_on_segment : on_segment E C D)
  (h_ratio : segment_ratio D E C (3/2))
  (h_area_ADE : triangle_area A D E = 27) :
  triangle_area A E C = 18 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AEC_l2601_260141


namespace NUMINAMATH_CALUDE_evaluate_expression_l2601_260166

theorem evaluate_expression : (528 : ℤ) * 528 - (527 * 529) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2601_260166


namespace NUMINAMATH_CALUDE_house_number_theorem_l2601_260153

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ := (n / 100) + ((n / 10) % 10) + (n % 10)

def all_digits_same (n : ℕ) : Prop :=
  (n / 100) = ((n / 10) % 10) ∧ ((n / 10) % 10) = (n % 10)

def two_digits_same (n : ℕ) : Prop :=
  (n / 100 = (n / 10) % 10) ∨ ((n / 10) % 10 = n % 10) ∨ (n / 100 = n % 10)

def all_digits_different (n : ℕ) : Prop :=
  (n / 100) ≠ ((n / 10) % 10) ∧ ((n / 10) % 10) ≠ (n % 10) ∧ (n / 100) ≠ (n % 10)

theorem house_number_theorem :
  (∃! n : ℕ, is_three_digit n ∧ digit_sum n = 24 ∧ all_digits_same n) ∧
  (∃ l : List ℕ, l.length = 3 ∧ ∀ n ∈ l, is_three_digit n ∧ digit_sum n = 24 ∧ two_digits_same n) ∧
  (∃ l : List ℕ, l.length = 6 ∧ ∀ n ∈ l, is_three_digit n ∧ digit_sum n = 24 ∧ all_digits_different n) :=
sorry

end NUMINAMATH_CALUDE_house_number_theorem_l2601_260153


namespace NUMINAMATH_CALUDE_linear_function_properties_l2601_260143

/-- Linear function definition -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (2*m + 1)*x + m - 2

theorem linear_function_properties :
  ∀ m : ℝ,
  (∀ x, linear_function m x = 0 → x = 0) → m = 2 ∧
  (linear_function m 0 = -3) → m = -1 ∧
  (∀ x, ∃ k, linear_function m x = x + k) → m = 0 ∧
  (∀ x, x < 0 → linear_function m x > 0) → -1/2 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2601_260143


namespace NUMINAMATH_CALUDE_fixed_monthly_charge_l2601_260111

-- Define the fixed monthly charge for internet service
def F : ℝ := sorry

-- Define the charge for calls in January
def C : ℝ := sorry

-- Define the total bill for January
def january_bill : ℝ := 50

-- Define the total bill for February
def february_bill : ℝ := 76

-- Theorem to prove the fixed monthly charge for internet service
theorem fixed_monthly_charge :
  (F + C = january_bill) →
  (F + 2 * C = february_bill) →
  F = 24 := by sorry

end NUMINAMATH_CALUDE_fixed_monthly_charge_l2601_260111


namespace NUMINAMATH_CALUDE_equation_solution_l2601_260148

theorem equation_solution : 
  ∃ x : ℝ, (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/3)) ∧ 
  (x = ((((1 + Real.sqrt 17) / 2) ^ 3 - 2) ^ 2)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2601_260148


namespace NUMINAMATH_CALUDE_picture_area_l2601_260136

theorem picture_area (x y : ℕ) (h1 : x > 1) (h2 : y > 1) (h3 : (3 * x + 3) * (y + 2) = 110) : x * y = 28 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l2601_260136


namespace NUMINAMATH_CALUDE_females_in_coach_class_l2601_260165

theorem females_in_coach_class 
  (total_passengers : ℕ) 
  (female_percentage : ℚ) 
  (first_class_percentage : ℚ) 
  (male_first_class_fraction : ℚ) 
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 30 / 100)
  (h3 : first_class_percentage = 10 / 100)
  (h4 : male_first_class_fraction = 1 / 3) :
  ↑((total_passengers : ℚ) * female_percentage - 
    (total_passengers : ℚ) * first_class_percentage * (1 - male_first_class_fraction)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_females_in_coach_class_l2601_260165


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l2601_260164

/-- The number of ways to arrange two types of indistinguishable objects in a row -/
def arrange_books (n m : ℕ) : ℕ := Nat.choose (n + m) n

/-- Theorem: Arranging 5 and 6 indistinguishable objects in 11 positions yields 462 ways -/
theorem book_arrangement_theorem :
  arrange_books 5 6 = 462 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l2601_260164


namespace NUMINAMATH_CALUDE_book_area_l2601_260101

/-- The area of a rectangular book with length 5 inches and width 10 inches is 50 square inches. -/
theorem book_area : 
  let length : ℝ := 5
  let width : ℝ := 10
  let area := length * width
  area = 50 := by sorry

end NUMINAMATH_CALUDE_book_area_l2601_260101


namespace NUMINAMATH_CALUDE_continuity_definition_relation_l2601_260178

-- Define a real-valued function
variable (f : ℝ → ℝ)
-- Define a point x₀
variable (x₀ : ℝ)

-- Define what it means for f to be defined at x₀
def is_defined_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ y : ℝ, f x₀ = y

-- State the theorem
theorem continuity_definition_relation :
  (ContinuousAt f x₀ → is_defined_at f x₀) ∧
  ¬(is_defined_at f x₀ → ContinuousAt f x₀) :=
sorry

end NUMINAMATH_CALUDE_continuity_definition_relation_l2601_260178


namespace NUMINAMATH_CALUDE_steven_jill_peach_difference_l2601_260170

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 19

/-- The number of peaches Jill has -/
def jill_peaches : ℕ := 6

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - 18

/-- Theorem: Steven has 13 more peaches than Jill -/
theorem steven_jill_peach_difference : steven_peaches - jill_peaches = 13 := by
  sorry

end NUMINAMATH_CALUDE_steven_jill_peach_difference_l2601_260170


namespace NUMINAMATH_CALUDE_system_solution_l2601_260127

-- Define the system of equations
def equation1 (x y : ℚ) : Prop := 2 * x - 3 * y = 5
def equation2 (x y : ℚ) : Prop := 4 * x + y = 9

-- Define the solution
def solution : ℚ × ℚ := (16/7, -1/7)

-- Theorem statement
theorem system_solution :
  let (x, y) := solution
  equation1 x y ∧ equation2 x y := by sorry

end NUMINAMATH_CALUDE_system_solution_l2601_260127


namespace NUMINAMATH_CALUDE_mod_fifteen_equivalence_l2601_260154

theorem mod_fifteen_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 7615 [ZMOD 15] ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_mod_fifteen_equivalence_l2601_260154


namespace NUMINAMATH_CALUDE_harvest_duration_proof_l2601_260180

/-- Calculates the number of weeks the harvest lasted. -/
def harvest_duration (weekly_earnings : ℕ) (total_earnings : ℕ) : ℕ :=
  total_earnings / weekly_earnings

/-- Proves that the harvest lasted 89 weeks given the conditions. -/
theorem harvest_duration_proof (weekly_earnings total_earnings : ℕ) 
  (h1 : weekly_earnings = 2)
  (h2 : total_earnings = 178) :
  harvest_duration weekly_earnings total_earnings = 89 := by
  sorry

end NUMINAMATH_CALUDE_harvest_duration_proof_l2601_260180


namespace NUMINAMATH_CALUDE_g_tan_squared_l2601_260112

open Real

noncomputable def g (x : ℝ) : ℝ := 1 / ((x - 1) / x)

theorem g_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/2) :
  g (tan t ^ 2) = tan t ^ 2 - tan t ^ 4 :=
by sorry

end NUMINAMATH_CALUDE_g_tan_squared_l2601_260112


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l2601_260191

theorem largest_multiple_of_9_under_100 : ∃ n : ℕ, n * 9 = 99 ∧ 
  99 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ 99 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l2601_260191


namespace NUMINAMATH_CALUDE_age_difference_l2601_260150

theorem age_difference (patrick michael monica : ℕ) : 
  patrick * 5 = michael * 3 →
  michael * 5 = monica * 3 →
  patrick + michael + monica = 147 →
  monica - patrick = 48 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l2601_260150


namespace NUMINAMATH_CALUDE_max_students_is_25_l2601_260161

/-- Represents the field trip problem with given conditions --/
structure FieldTrip where
  bus_rental : ℕ
  bus_capacity : ℕ
  admission_cost : ℕ
  total_budget : ℕ

/-- Calculates the maximum number of students that can go on the field trip --/
def max_students (trip : FieldTrip) : ℕ :=
  min
    ((trip.total_budget - trip.bus_rental) / trip.admission_cost)
    trip.bus_capacity

/-- Theorem stating that the maximum number of students for the given conditions is 25 --/
theorem max_students_is_25 :
  let trip : FieldTrip := {
    bus_rental := 100,
    bus_capacity := 25,
    admission_cost := 10,
    total_budget := 350
  }
  max_students trip = 25 := by
  sorry


end NUMINAMATH_CALUDE_max_students_is_25_l2601_260161


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2601_260110

/-- Proves that mixing 175 gallons of 15% alcohol solution with 75 gallons of 35% alcohol solution 
    results in 250 gallons of 21% alcohol solution. -/
theorem alcohol_mixture_proof :
  let solution_1_volume : ℝ := 175
  let solution_1_concentration : ℝ := 0.15
  let solution_2_volume : ℝ := 75
  let solution_2_concentration : ℝ := 0.35
  let total_volume : ℝ := 250
  let final_concentration : ℝ := 0.21
  (solution_1_volume + solution_2_volume = total_volume) ∧
  (solution_1_volume * solution_1_concentration + solution_2_volume * solution_2_concentration = 
   total_volume * final_concentration) :=
by sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2601_260110


namespace NUMINAMATH_CALUDE_smallest_positive_root_of_f_l2601_260182

open Real

theorem smallest_positive_root_of_f (f : ℝ → ℝ) :
  (∀ x, f x = sin x + 2 * cos x + 3 * tan x) →
  (∃ x ∈ Set.Ioo 3 4, f x = 0) ∧
  (∀ x ∈ Set.Ioo 0 3, f x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_root_of_f_l2601_260182


namespace NUMINAMATH_CALUDE_fraction_division_eval_l2601_260104

theorem fraction_division_eval : (7 / 3) / (8 / 15) = 35 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_eval_l2601_260104


namespace NUMINAMATH_CALUDE_calvins_weight_after_training_l2601_260117

/-- Calculates the final weight after a period of constant weight loss -/
def final_weight (initial_weight : ℕ) (weight_loss_per_month : ℕ) (months : ℕ) : ℕ :=
  initial_weight - weight_loss_per_month * months

/-- Theorem stating that Calvin's weight after one year of training is 154 pounds -/
theorem calvins_weight_after_training :
  final_weight 250 8 12 = 154 := by
  sorry

end NUMINAMATH_CALUDE_calvins_weight_after_training_l2601_260117


namespace NUMINAMATH_CALUDE_local_minimum_condition_l2601_260185

/-- The function f(x) = x^3 + (x-a)^2 has a local minimum at x = 2 if and only if a = 8 -/
theorem local_minimum_condition (a : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), 
    x^3 + (x - a)^2 ≥ 2^3 + (2 - a)^2) ↔ a = 8 := by
  sorry


end NUMINAMATH_CALUDE_local_minimum_condition_l2601_260185


namespace NUMINAMATH_CALUDE_rice_price_decrease_l2601_260173

/-- Calculates the percentage decrease in price given the original and new quantities that can be purchased with the same amount of money. -/
def price_decrease_percentage (original_quantity : ℕ) (new_quantity : ℕ) : ℚ :=
  (1 - original_quantity / new_quantity) * 100

/-- Theorem stating that if 20 kg of rice can now buy 25 kg after a price decrease, the percentage decrease is 20%. -/
theorem rice_price_decrease : price_decrease_percentage 20 25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rice_price_decrease_l2601_260173


namespace NUMINAMATH_CALUDE_complement_of_P_l2601_260130

-- Define the universal set R as the set of real numbers
def R : Set ℝ := Set.univ

-- Define set P
def P : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_of_P : 
  Set.compl P = {x : ℝ | x < 1} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_P_l2601_260130


namespace NUMINAMATH_CALUDE_twenty_three_in_base_two_l2601_260168

theorem twenty_three_in_base_two : 
  (23 : ℕ) = 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 := by
  sorry

end NUMINAMATH_CALUDE_twenty_three_in_base_two_l2601_260168


namespace NUMINAMATH_CALUDE_prob_red_ball_one_third_l2601_260126

/-- A bag containing red and yellow balls -/
structure Bag where
  red_balls : ℕ
  yellow_balls : ℕ

/-- The probability of drawing a red ball from the bag -/
def prob_red_ball (bag : Bag) : ℚ :=
  bag.red_balls / (bag.red_balls + bag.yellow_balls)

/-- The theorem stating the probability of drawing a red ball -/
theorem prob_red_ball_one_third (bag : Bag) 
  (h1 : bag.red_balls = 1) 
  (h2 : bag.yellow_balls = 2) : 
  prob_red_ball bag = 1/3 := by
  sorry

#check prob_red_ball_one_third

end NUMINAMATH_CALUDE_prob_red_ball_one_third_l2601_260126


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2601_260102

def M : Set ℕ := {0, 1}

def N : Set ℕ := {y | ∃ x ∈ M, y = x^2 + 1}

theorem intersection_of_M_and_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2601_260102


namespace NUMINAMATH_CALUDE_percentage_defective_meters_l2601_260194

def total_meters : ℕ := 4000
def rejected_meters : ℕ := 2

theorem percentage_defective_meters :
  (rejected_meters : ℝ) / total_meters * 100 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_percentage_defective_meters_l2601_260194


namespace NUMINAMATH_CALUDE_ratio_of_system_l2601_260156

theorem ratio_of_system (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_system_l2601_260156


namespace NUMINAMATH_CALUDE_circumradius_eq_one_l2601_260187

/-- Three unit circles passing through a common point -/
structure ThreeIntersectingCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  center3 : ℝ × ℝ
  commonPoint : ℝ × ℝ
  radius : ℝ
  radius_eq_one : radius = 1
  passes_through_common : 
    dist center1 commonPoint = radius ∧
    dist center2 commonPoint = radius ∧
    dist center3 commonPoint = radius

/-- The three intersection points forming triangle ABC -/
def intersectionPoints (c : ThreeIntersectingCircles) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- The circumcenter of triangle ABC -/
def circumcenter (points : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

/-- The circumradius of triangle ABC -/
def circumradius (c : ThreeIntersectingCircles) : ℝ :=
  let points := intersectionPoints c
  dist (circumcenter points) points.1

/-- Theorem: The circumradius of triangle ABC is equal to 1 -/
theorem circumradius_eq_one (c : ThreeIntersectingCircles) :
  circumradius c = 1 :=
sorry

end NUMINAMATH_CALUDE_circumradius_eq_one_l2601_260187


namespace NUMINAMATH_CALUDE_badge_exchange_l2601_260134

theorem badge_exchange (x : ℕ) : 
  -- Vasya initially had 5 more badges than Tolya
  let vasya_initial := x + 5
  -- Vasya exchanged 24% of his badges for 20% of Tolya's badges
  let vasya_final := vasya_initial - (24 * vasya_initial) / 100 + (20 * x) / 100
  let tolya_final := x - (20 * x) / 100 + (24 * vasya_initial) / 100
  -- After the exchange, Vasya had one badge less than Tolya
  vasya_final + 1 = tolya_final →
  -- Prove that Tolya initially had 45 badges and Vasya initially had 50 badges
  x = 45 ∧ vasya_initial = 50 := by
sorry

end NUMINAMATH_CALUDE_badge_exchange_l2601_260134


namespace NUMINAMATH_CALUDE_linear_function_passes_through_point_l2601_260114

/-- A linear function f(x) = kx + k - 1 passes through the point (-1, -1) for any real k. -/
theorem linear_function_passes_through_point
  (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ k * x + k - 1
  f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_point_l2601_260114


namespace NUMINAMATH_CALUDE_average_fish_is_75_l2601_260183

/-- The number of fish in Boast Pool -/
def boast_pool : ℕ := 75

/-- The number of fish in Onum Lake -/
def onum_lake : ℕ := boast_pool + 25

/-- The number of fish in Riddle Pond -/
def riddle_pond : ℕ := onum_lake / 2

/-- The total number of fish in all three bodies of water -/
def total_fish : ℕ := boast_pool + onum_lake + riddle_pond

/-- The number of bodies of water -/
def num_bodies : ℕ := 3

/-- Theorem stating that the average number of fish in all three bodies of water is 75 -/
theorem average_fish_is_75 : total_fish / num_bodies = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_fish_is_75_l2601_260183


namespace NUMINAMATH_CALUDE_inequality_proof_l2601_260139

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) :
  (1 / (2 - a)) + (1 / (2 - b)) + (1 / (2 - c)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2601_260139


namespace NUMINAMATH_CALUDE_john_yearly_music_cost_l2601_260142

/-- Calculates the yearly cost of music for John given his buying habits --/
theorem john_yearly_music_cost
  (hours_per_month : ℕ)
  (song_length_minutes : ℕ)
  (song_cost_cents : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : song_length_minutes = 3)
  (h3 : song_cost_cents = 50) :
  (hours_per_month * 60 / song_length_minutes) * song_cost_cents * 12 = 240000 :=
by sorry

end NUMINAMATH_CALUDE_john_yearly_music_cost_l2601_260142


namespace NUMINAMATH_CALUDE_circle_condition_l2601_260135

-- Define the equation
def circle_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0

-- Theorem statement
theorem circle_condition (a : ℝ) :
  (∃ h k r, ∀ x y, circle_equation a x y ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l2601_260135


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2601_260188

/-- An equilateral triangle with a point inside --/
structure EquilateralTriangleWithPoint where
  /-- The side length of the equilateral triangle --/
  side_length : ℝ
  /-- The perpendicular distance from the point to the first side --/
  dist1 : ℝ
  /-- The perpendicular distance from the point to the second side --/
  dist2 : ℝ
  /-- The perpendicular distance from the point to the third side --/
  dist3 : ℝ
  /-- The side length is positive --/
  side_positive : side_length > 0
  /-- All distances are positive --/
  dist_positive : dist1 > 0 ∧ dist2 > 0 ∧ dist3 > 0

/-- The theorem stating the relationship between the side length and the perpendicular distances --/
theorem equilateral_triangle_side_length 
  (t : EquilateralTriangleWithPoint) 
  (h1 : t.dist1 = 2) 
  (h2 : t.dist2 = 3) 
  (h3 : t.dist3 = 4) : 
  t.side_length = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2601_260188


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l2601_260133

def B : Set ℕ := {n | ∃ x : ℕ, x > 0 ∧ n = (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l2601_260133


namespace NUMINAMATH_CALUDE_charity_fundraising_l2601_260125

theorem charity_fundraising (people : ℕ) (total_amount : ℕ) (amount_per_person : ℕ) :
  people = 8 →
  total_amount = 3000 →
  amount_per_person * people = total_amount →
  amount_per_person = 375 := by
  sorry

end NUMINAMATH_CALUDE_charity_fundraising_l2601_260125


namespace NUMINAMATH_CALUDE_adults_average_age_l2601_260179

def robotics_camp_problem (total_members : ℕ) (overall_average_age : ℝ)
  (num_girls num_boys num_adults : ℕ) (girls_average_age boys_average_age : ℝ) : Prop :=
  total_members = 50 ∧
  overall_average_age = 20 ∧
  num_girls = 25 ∧
  num_boys = 18 ∧
  num_adults = 7 ∧
  girls_average_age = 18 ∧
  boys_average_age = 19 ∧
  (total_members : ℝ) * overall_average_age =
    (num_girls : ℝ) * girls_average_age +
    (num_boys : ℝ) * boys_average_age +
    (num_adults : ℝ) * ((1000 - 450 - 342) / 7)

theorem adults_average_age
  (total_members : ℕ) (overall_average_age : ℝ)
  (num_girls num_boys num_adults : ℕ) (girls_average_age boys_average_age : ℝ)
  (h : robotics_camp_problem total_members overall_average_age
    num_girls num_boys num_adults girls_average_age boys_average_age) :
  (1000 - 450 - 342) / 7 = (total_members * overall_average_age -
    num_girls * girls_average_age - num_boys * boys_average_age) / num_adults :=
by sorry

end NUMINAMATH_CALUDE_adults_average_age_l2601_260179


namespace NUMINAMATH_CALUDE_loan_duration_l2601_260115

/-- Proves that the first part of a loan was lent for 8 years given specific conditions -/
theorem loan_duration (total sum : ℕ) (second_part : ℕ) (first_rate second_rate : ℚ) (second_duration : ℕ) : 
  total = 2730 →
  second_part = 1680 →
  first_rate = 3 / 100 →
  second_rate = 5 / 100 →
  second_duration = 3 →
  ∃ (first_duration : ℕ), 
    (total - second_part) * first_rate * first_duration = second_part * second_rate * second_duration ∧
    first_duration = 8 :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_l2601_260115


namespace NUMINAMATH_CALUDE_square_difference_153_147_l2601_260159

theorem square_difference_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_153_147_l2601_260159


namespace NUMINAMATH_CALUDE_finite_prime_triples_l2601_260147

theorem finite_prime_triples (k : ℕ) :
  Set.Finite {triple : ℕ × ℕ × ℕ | 
    let (p, q, r) := triple
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (q * r - k) % p = 0 ∧
    (p * r - k) % q = 0 ∧
    (p * q - k) % r = 0} :=
by sorry

end NUMINAMATH_CALUDE_finite_prime_triples_l2601_260147


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l2601_260184

-- Define the binary number as a list of bits (0 or 1)
def binary_number : List Nat := [1, 1, 0, 0, 1, 1]

-- Define the function to convert binary to decimal
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Theorem to prove
theorem binary_110011_equals_51 :
  binary_to_decimal binary_number = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l2601_260184


namespace NUMINAMATH_CALUDE_loan_duration_l2601_260116

/-- Proves that the first part of a loan is lent for 8 years given specific conditions -/
theorem loan_duration (total_sum interest_rate1 interest_rate2 duration2 : ℚ) 
  (second_part : ℚ) : 
  total_sum = 2743 →
  second_part = 1688 →
  interest_rate1 = 3/100 →
  interest_rate2 = 5/100 →
  duration2 = 3 →
  let first_part := total_sum - second_part
  let duration1 := (second_part * interest_rate2 * duration2) / (first_part * interest_rate1)
  duration1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_loan_duration_l2601_260116


namespace NUMINAMATH_CALUDE_f_value_at_11pi_over_6_l2601_260138

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, (is_periodic f q ∧ q > 0) → p ≤ q

theorem f_value_at_11pi_over_6 (f : ℝ → ℝ) :
  is_odd f →
  smallest_positive_period f π →
  (∀ x ∈ Set.Ioo 0 (π/2), f x = 2 * Real.sin x) →
  f (11*π/6) = -1 := by sorry

end NUMINAMATH_CALUDE_f_value_at_11pi_over_6_l2601_260138


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2601_260163

/-- Represents the total number of students in the school -/
def total_students : ℕ := 600 + 500 + 400

/-- Represents the number of students in the first grade -/
def first_grade_students : ℕ := 600

/-- Represents the number of first-grade students in the sample -/
def first_grade_sample : ℕ := 30

/-- Theorem stating that the total sample size is 75 given the conditions -/
theorem stratified_sample_size :
  ∃ (n : ℕ),
    n * first_grade_students = total_students * first_grade_sample ∧
    n = 75 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2601_260163


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_intersecting_circles_l2601_260124

/-- Given two circles with radii R and r that touch a common line and intersect each other,
    the radius ρ of the circumcircle of the triangle formed by their two points of tangency
    and one point of intersection is equal to √(R * r). -/
theorem circumcircle_radius_of_intersecting_circles (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  ∃ (ρ : ℝ), ρ > 0 ∧ ρ * ρ = R * r := by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_intersecting_circles_l2601_260124


namespace NUMINAMATH_CALUDE_no_integer_solution_l2601_260193

theorem no_integer_solution : ¬ ∃ (x : ℤ), (x + 12 > 15) ∧ (-3*x > -9) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2601_260193


namespace NUMINAMATH_CALUDE_zero_in_P_and_two_not_in_P_l2601_260149

-- Define the set P
def P : Set Int := sorry

-- Define the properties of P
axiom P_contains_positive : ∃ x : Int, x > 0 ∧ x ∈ P
axiom P_contains_negative : ∃ x : Int, x < 0 ∧ x ∈ P
axiom P_contains_odd : ∃ x : Int, x % 2 ≠ 0 ∧ x ∈ P
axiom P_contains_even : ∃ x : Int, x % 2 = 0 ∧ x ∈ P
axiom P_not_contains_neg_one : -1 ∉ P
axiom P_closed_under_addition : ∀ x y : Int, x ∈ P → y ∈ P → (x + y) ∈ P

-- Theorem to prove
theorem zero_in_P_and_two_not_in_P : 0 ∈ P ∧ 2 ∉ P := by
  sorry

end NUMINAMATH_CALUDE_zero_in_P_and_two_not_in_P_l2601_260149


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2601_260195

theorem quadratic_equal_roots (a b c : ℝ) 
  (h1 : b ≠ c) 
  (h2 : ∃ x : ℝ, (b - c) * x^2 + (a - b) * x + (c - a) = 0 ∧ 
       ((b - c) * (2 * x) + (a - b) = 0)) : 
  c = (a + b) / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2601_260195


namespace NUMINAMATH_CALUDE_prob_two_dice_shows_two_l2601_260107

def num_sides : ℕ := 8

def prob_at_least_one_two (n : ℕ) : ℚ :=
  1 - ((n - 1) / n)^2

theorem prob_two_dice_shows_two :
  prob_at_least_one_two num_sides = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_dice_shows_two_l2601_260107


namespace NUMINAMATH_CALUDE_sum_base3_equals_10200_l2601_260105

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun digit acc => acc * 3 + digit) 0

/-- Represents a number in base 3 -/
structure Base3 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 3

/-- Addition of Base3 numbers -/
def addBase3 (a b : Base3) : Base3 :=
  sorry

theorem sum_base3_equals_10200 :
  let a := Base3.mk [1] (by simp)
  let b := Base3.mk [2, 1] (by simp)
  let c := Base3.mk [2, 1, 2] (by simp)
  let d := Base3.mk [1, 2, 1, 2] (by simp)
  let result := Base3.mk [0, 0, 2, 0, 1] (by simp)
  addBase3 (addBase3 (addBase3 a b) c) d = result :=
sorry

end NUMINAMATH_CALUDE_sum_base3_equals_10200_l2601_260105


namespace NUMINAMATH_CALUDE_principal_amount_l2601_260189

-- Define the interest rate and time
def r : ℝ := 0.07
def t : ℝ := 2

-- Define the difference between C.I. and S.I.
def difference : ℝ := 49

-- State the theorem
theorem principal_amount (P : ℝ) :
  P * ((1 + r)^t - 1 - t * r) = difference → P = 10000 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l2601_260189


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2601_260108

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 4}
def N : Finset ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2601_260108


namespace NUMINAMATH_CALUDE_polynomial_factor_l2601_260176

/-- The polynomial with parameters a and b -/
def P (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + 48 * x^2 - 24 * x + 4

/-- The factor of the polynomial -/
def F (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 1

/-- Theorem stating that the polynomial P has the factor F when a = -16 and b = -36 -/
theorem polynomial_factor (x : ℝ) : ∃ (Q : ℝ → ℝ), P (-16) (-36) x = F x * Q x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l2601_260176


namespace NUMINAMATH_CALUDE_remaining_balance_calculation_l2601_260167

def initial_balance : ℝ := 50
def coffee_expense : ℝ := 10
def tumbler_expense : ℝ := 30

theorem remaining_balance_calculation :
  initial_balance - (coffee_expense + tumbler_expense) = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_calculation_l2601_260167


namespace NUMINAMATH_CALUDE_special_triangle_perimeter_l2601_260174

/-- A triangle with sides that are consecutive natural numbers and largest angle twice the smallest -/
structure SpecialTriangle where
  n : ℕ
  side1 : ℕ := n - 1
  side2 : ℕ := n
  side3 : ℕ := n + 1
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angle_sum : angleA + angleB + angleC = π
  angle_relation : angleC = 2 * angleA
  law_of_sines : (n - 1) / Real.sin angleA = n / Real.sin angleB
  law_of_cosines : (n - 1)^2 = (n + 1)^2 + n^2 - 2 * (n + 1) * n * Real.cos angleC

/-- The perimeter of the special triangle is 15 -/
theorem special_triangle_perimeter (t : SpecialTriangle) : t.side1 + t.side2 + t.side3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_perimeter_l2601_260174


namespace NUMINAMATH_CALUDE_ninth_row_sum_l2601_260103

/-- Yang Hui's Triangle (Pascal's Triangle) -/
def yangHuiTriangle (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- Sum of elements in a row of Yang Hui's Triangle -/
def rowSum (n : ℕ) : ℕ :=
  (List.range (n + 1)).map (yangHuiTriangle n) |>.sum

/-- Theorem: The sum of all numbers in the 9th row of Yang Hui's Triangle is 2^8 -/
theorem ninth_row_sum : rowSum 8 = 2^8 := by
  sorry

end NUMINAMATH_CALUDE_ninth_row_sum_l2601_260103


namespace NUMINAMATH_CALUDE_time_left_before_movie_l2601_260100

def movie_time_minutes : ℕ := 2 * 60

def homework_time : ℕ := 30

def room_cleaning_time : ℕ := homework_time / 2

def dog_walking_time : ℕ := homework_time + 5

def trash_taking_time : ℕ := homework_time / 6

def total_chore_time : ℕ := homework_time + room_cleaning_time + dog_walking_time + trash_taking_time

theorem time_left_before_movie : movie_time_minutes - total_chore_time = 35 := by
  sorry

end NUMINAMATH_CALUDE_time_left_before_movie_l2601_260100


namespace NUMINAMATH_CALUDE_find_unknown_numbers_l2601_260137

/-- Given four real numbers A, B, C, and D satisfying certain conditions,
    prove that they have specific values. -/
theorem find_unknown_numbers (A B C D : ℝ) 
    (h1 : 0.05 * A = 0.20 * 650 + 0.10 * B)
    (h2 : A + B = 4000)
    (h3 : C = 2 * B)
    (h4 : A + B + C = 0.40 * D) :
    A = 3533.3333333333335 ∧ 
    B = 466.6666666666667 ∧ 
    C = 933.3333333333334 ∧ 
    D = 12333.333333333334 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_numbers_l2601_260137


namespace NUMINAMATH_CALUDE_value_of_x_l2601_260109

theorem value_of_x (x y z d e f : ℝ) 
  (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)
  (h1 : x * y / (x + 2 * y) = d)
  (h2 : x * z / (2 * x + z) = e)
  (h3 : y * z / (y + 2 * z) = f) :
  x = 3 * d * e * f / (d * e - 2 * d * f + e * f) := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2601_260109


namespace NUMINAMATH_CALUDE_number_equation_l2601_260198

theorem number_equation : ∃ n : ℝ, (n - 5) * 4 = n * 2 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2601_260198


namespace NUMINAMATH_CALUDE_train_speed_l2601_260106

/-- Calculates the speed of a train passing through a tunnel -/
theorem train_speed (train_length : ℝ) (tunnel_length : ℝ) (time_minutes : ℝ) :
  train_length = 1 →
  tunnel_length = 70 →
  time_minutes = 6 →
  (train_length + tunnel_length) / (time_minutes / 60) = 710 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2601_260106


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l2601_260175

/-- The number of perfect square factors of 345600 -/
def perfectSquareFactors : ℕ := 16

/-- The prime factorization of 345600 -/
def n : ℕ := 2^6 * 3^3 * 5^2

/-- A function that counts the number of perfect square factors of n -/
def countPerfectSquareFactors (n : ℕ) : ℕ := sorry

theorem count_perfect_square_factors :
  countPerfectSquareFactors n = perfectSquareFactors := by sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l2601_260175


namespace NUMINAMATH_CALUDE_y_intercept_range_l2601_260123

-- Define the points A and B
def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (2, 3)

-- Define the line l: x + y - c = 0
def line_l (c : ℝ) (x y : ℝ) : Prop := x + y - c = 0

-- Define what it means for a point to be on the line
def point_on_line (p : ℝ × ℝ) (c : ℝ) : Prop :=
  line_l c p.1 p.2

-- Define what it means for a line to intersect a segment
def intersects_segment (c : ℝ) : Prop :=
  ∃ t : ℝ, t ∈ (Set.Icc 0 1) ∧
    point_on_line ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2) c

-- State the theorem
theorem y_intercept_range :
  ∀ c : ℝ, intersects_segment c → c ∈ Set.Icc (-3) 5 :=
sorry

end NUMINAMATH_CALUDE_y_intercept_range_l2601_260123


namespace NUMINAMATH_CALUDE_noemi_initial_money_l2601_260169

def roulette_loss : Int := 600
def blackjack_win : Int := 400
def poker_loss : Int := 400
def baccarat_win : Int := 500
def meal_cost : Int := 200
def final_amount : Int := 1800

theorem noemi_initial_money :
  ∃ (initial_money : Int),
    initial_money = 
      roulette_loss + blackjack_win + poker_loss + baccarat_win + meal_cost + final_amount :=
by
  sorry

end NUMINAMATH_CALUDE_noemi_initial_money_l2601_260169


namespace NUMINAMATH_CALUDE_beth_friends_count_l2601_260121

theorem beth_friends_count (initial_packs : ℝ) (additional_packs : ℝ) (final_packs : ℝ) :
  initial_packs = 4 →
  additional_packs = 6 →
  final_packs = 6.4 →
  ∃ (num_friends : ℝ),
    num_friends > 0 ∧
    final_packs = additional_packs + initial_packs / num_friends ∧
    num_friends = 10 := by
  sorry

end NUMINAMATH_CALUDE_beth_friends_count_l2601_260121


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_for_1000_l2601_260122

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k ≤ d - 1 ∧ (n - k) % d = 0 ∧
  ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem least_subtraction_for_1000 :
  ∃ (k : Nat), k = 398 ∧ 
  (427398 - k) % 1000 = 0 ∧
  ∀ (m : Nat), m < k → (427398 - m) % 1000 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_for_1000_l2601_260122


namespace NUMINAMATH_CALUDE_algebraic_identity_l2601_260146

theorem algebraic_identity (a b : ℝ) : a * b - 2 * (a * b) = -(a * b) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l2601_260146


namespace NUMINAMATH_CALUDE_minimum_value_and_max_when_half_l2601_260181

noncomputable def f (a x : ℝ) : ℝ := 1 - 2*a - 2*a*Real.cos x - 2*(Real.sin x)^2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a ≤ 2 then -a^2/2 - 2*a - 1
  else 1 - 4*a

theorem minimum_value_and_max_when_half (a : ℝ) :
  (∀ x, f a x ≥ g a) ∧
  (g a = 1/2 → a = -1 ∧ ∃ x, f (-1) x = 5 ∧ ∀ y, f (-1) y ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_minimum_value_and_max_when_half_l2601_260181


namespace NUMINAMATH_CALUDE_tangent_line_point_on_circle_l2601_260145

/-- Given a circle C defined by x^2 + y^2 = 1 and a line L defined by ax + by = 1 
    that is tangent to C, prove that the point (a, b) lies on C. -/
theorem tangent_line_point_on_circle (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (a*x + b*y = 1 → x^2 + y^2 > 1 ∨ a*x + b*y > 1)) → 
  a^2 + b^2 = 1 := by
  sorry

#check tangent_line_point_on_circle

end NUMINAMATH_CALUDE_tangent_line_point_on_circle_l2601_260145


namespace NUMINAMATH_CALUDE_distance_between_intersections_l2601_260186

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := ∃ θ : ℝ, x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sin θ

-- Define the ray
def ray (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∧ x ≥ 0

-- Define the intersection points
def intersectionC₁ (x y : ℝ) : Prop := C₁ x y ∧ ray x y
def intersectionC₂ (x y : ℝ) : Prop := C₂ x y ∧ ray x y

-- Theorem statement
theorem distance_between_intersections :
  ∃ (A B : ℝ × ℝ),
    intersectionC₁ A.1 A.2 ∧
    intersectionC₂ B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 - 2 * Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_CALUDE_distance_between_intersections_l2601_260186


namespace NUMINAMATH_CALUDE_a_monotonically_decreasing_iff_t_lt_3_l2601_260162

/-- The sequence a_n defined as -n^2 + tn for positive integers n and constant t -/
def a (n : ℕ+) (t : ℝ) : ℝ := -n.val^2 + t * n.val

/-- A sequence is monotonically decreasing if each term is less than the previous term -/
def monotonically_decreasing (s : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, s (n + 1) < s n

/-- The main theorem: the sequence a_n is monotonically decreasing iff t < 3 -/
theorem a_monotonically_decreasing_iff_t_lt_3 (t : ℝ) :
  monotonically_decreasing (a · t) ↔ t < 3 := by
  sorry

end NUMINAMATH_CALUDE_a_monotonically_decreasing_iff_t_lt_3_l2601_260162


namespace NUMINAMATH_CALUDE_smallest_student_count_l2601_260177

/-- Represents the number of students in each grade --/
structure StudentCounts where
  grade9 : ℕ
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Checks if the given student counts satisfy the required ratios --/
def satisfiesRatios (counts : StudentCounts) : Prop :=
  3 * counts.grade10 = 2 * counts.grade12 ∧
  7 * counts.grade11 = 4 * counts.grade12 ∧
  5 * counts.grade9 = 3 * counts.grade12

/-- Calculates the total number of students --/
def totalStudents (counts : StudentCounts) : ℕ :=
  counts.grade9 + counts.grade10 + counts.grade11 + counts.grade12

/-- Theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (counts : StudentCounts),
    satisfiesRatios counts ∧
    totalStudents counts = 298 ∧
    (∀ (other : StudentCounts),
      satisfiesRatios other → totalStudents other ≥ 298) :=
  sorry

end NUMINAMATH_CALUDE_smallest_student_count_l2601_260177


namespace NUMINAMATH_CALUDE_solve_parking_problem_l2601_260151

def parking_problem (initial_balance : ℝ) (first_ticket_cost : ℝ) (num_full_cost_tickets : ℕ) (third_ticket_fraction : ℝ) (roommate_share : ℝ) : Prop :=
  let total_cost := first_ticket_cost * num_full_cost_tickets + first_ticket_cost * third_ticket_fraction
  let james_share := total_cost * (1 - roommate_share)
  initial_balance - james_share = 325

theorem solve_parking_problem :
  parking_problem 500 150 2 (1/3) (1/2) :=
by
  sorry

#check solve_parking_problem

end NUMINAMATH_CALUDE_solve_parking_problem_l2601_260151


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2601_260192

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (2 * x + 9) = 11 → x = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2601_260192


namespace NUMINAMATH_CALUDE_solve_equation_l2601_260160

theorem solve_equation (x : ℝ) : 2 - 2 / (1 - x) = 2 / (1 - x) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2601_260160


namespace NUMINAMATH_CALUDE_largest_sample_number_l2601_260140

/-- Systematic sampling from a set of numbered items -/
def systematic_sample (total : ℕ) (first : ℕ) (second : ℕ) : ℕ := 
  let interval := second - first
  let sample_size := total / interval
  first + interval * (sample_size - 1)

/-- The largest number in a systematic sample from 500 items -/
theorem largest_sample_number : 
  systematic_sample 500 7 32 = 482 := by
  sorry

end NUMINAMATH_CALUDE_largest_sample_number_l2601_260140


namespace NUMINAMATH_CALUDE_tan_alpha_results_l2601_260197

theorem tan_alpha_results (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  (Real.sin (2*α) / (Real.sin α^2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_results_l2601_260197


namespace NUMINAMATH_CALUDE_cathys_wallet_theorem_l2601_260113

/-- Calculates the remaining money in Cathy's wallet after receiving money from parents, buying a book, and saving some money. -/
def cathys_remaining_money (initial_amount dad_contribution book_cost savings_rate : ℚ) : ℚ :=
  let mom_contribution := 2 * dad_contribution
  let total_received := initial_amount + dad_contribution + mom_contribution
  let after_book_purchase := total_received - book_cost
  let savings_amount := savings_rate * after_book_purchase
  after_book_purchase - savings_amount

/-- Theorem stating that Cathy's remaining money is $57.60 given the initial conditions. -/
theorem cathys_wallet_theorem :
  cathys_remaining_money 12 25 15 (1/5) = 288/5 := by sorry

end NUMINAMATH_CALUDE_cathys_wallet_theorem_l2601_260113


namespace NUMINAMATH_CALUDE_triangle_problem_l2601_260157

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →
  a + c = 6 →
  (1/2) * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 →
  B = π / 3 ∧ b = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2601_260157


namespace NUMINAMATH_CALUDE_box_dimensions_l2601_260190

def is_valid_box (a b : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ b ∧ a^2 * b = a^2 + 4*a*b

theorem box_dimensions :
  ∀ a b : ℕ, is_valid_box a b ↔ (a = 8 ∧ b = 2) ∨ (a = 5 ∧ b = 5) :=
sorry

end NUMINAMATH_CALUDE_box_dimensions_l2601_260190


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2601_260144

def type_a_cost : ℚ := 9
def type_a_quantity : ℕ := 4
def type_b_extra_cost : ℚ := 5
def type_b_quantity : ℕ := 2
def clay_pot_extra_cost : ℚ := 20
def soil_cost_reduction : ℚ := 2
def fertilizer_percentage : ℚ := 1.5
def gardening_tools_percentage : ℚ := 0.75

def total_cost : ℚ :=
  type_a_cost * type_a_quantity +
  (type_a_cost + type_b_extra_cost) * type_b_quantity +
  (type_a_cost + clay_pot_extra_cost) +
  (type_a_cost - soil_cost_reduction) +
  (type_a_cost * fertilizer_percentage) +
  ((type_a_cost + clay_pot_extra_cost) * gardening_tools_percentage)

theorem total_cost_is_correct : total_cost = 135.25 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2601_260144


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2601_260128

open Set

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2601_260128


namespace NUMINAMATH_CALUDE_two_colors_sufficient_l2601_260129

/-- Represents a key on the ring -/
structure Key where
  position : Fin 8
  color : Bool

/-- Represents the ring of keys -/
def KeyRing : Type := Fin 8 → Key

/-- A coloring scheme is valid if it allows each key to be uniquely identified -/
def is_valid_coloring (ring : KeyRing) : Prop :=
  ∀ (i j : Fin 8), i ≠ j → 
    ∃ (k : ℕ), (ring ((i + k) % 8)).color ≠ (ring ((j + k) % 8)).color

/-- There exists a valid coloring scheme using only two colors -/
theorem two_colors_sufficient : 
  ∃ (ring : KeyRing), (∀ k, (ring k).color = true ∨ (ring k).color = false) ∧ is_valid_coloring ring := by
  sorry


end NUMINAMATH_CALUDE_two_colors_sufficient_l2601_260129


namespace NUMINAMATH_CALUDE_tax_discount_commute_mathville_problem_l2601_260120

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_price : 0 < price) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) := by
  sorry

/-- Calculates Bob's method: tax first, then discount --/
def bob_method (price tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate)

/-- Calculates Alice's method: discount first, then tax --/
def alice_method (price tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

theorem mathville_problem (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h_tax : tax_rate = 0.08) (h_discount : discount_rate = 0.25) (h_price : price = 120) :
  bob_method price tax_rate discount_rate - alice_method price tax_rate discount_rate = 0 := by
  sorry

end NUMINAMATH_CALUDE_tax_discount_commute_mathville_problem_l2601_260120


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l2601_260131

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 0 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l2601_260131


namespace NUMINAMATH_CALUDE_joanne_earnings_l2601_260118

/-- Joanne's work schedule and earnings calculation -/
theorem joanne_earnings :
  let main_job_hours : ℕ := 8
  let main_job_rate : ℚ := 16
  let part_time_hours : ℕ := 2
  let part_time_rate : ℚ := 27/2  -- $13.50 represented as a fraction
  let days_worked : ℕ := 5
  
  let main_job_daily := main_job_hours * main_job_rate
  let part_time_daily := part_time_hours * part_time_rate
  let total_daily := main_job_daily + part_time_daily
  let total_weekly := total_daily * days_worked
  
  total_weekly = 775
:= by sorry


end NUMINAMATH_CALUDE_joanne_earnings_l2601_260118


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2601_260119

theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℝ),
    (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
    a 1 = 4 →                            -- First term
    a 10 = 93312 →                       -- Last term
    a 7 = 186624 :=                      -- Seventh term
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l2601_260119


namespace NUMINAMATH_CALUDE_digit_repetition_property_l2601_260196

def repeat_digit (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^n - 1) / 9

theorem digit_repetition_property (n : ℕ) (h : n > 0) :
  (repeat_digit 6 n)^2 + repeat_digit 8 n = repeat_digit 4 (2*n) :=
sorry

end NUMINAMATH_CALUDE_digit_repetition_property_l2601_260196


namespace NUMINAMATH_CALUDE_tan_theta_equation_l2601_260199

open Real

theorem tan_theta_equation (θ : ℝ) (h1 : π/4 < θ ∧ θ < π/2) 
  (h2 : tan θ + tan (3*θ) + tan (5*θ) = 0) : tan θ = sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_equation_l2601_260199


namespace NUMINAMATH_CALUDE_mileage_difference_l2601_260171

/-- The difference between advertised and actual mileage -/
theorem mileage_difference (advertised_mpg : ℝ) (tank_capacity : ℝ) (miles_driven : ℝ) :
  advertised_mpg = 35 →
  tank_capacity = 12 →
  miles_driven = 372 →
  advertised_mpg - (miles_driven / tank_capacity) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mileage_difference_l2601_260171


namespace NUMINAMATH_CALUDE_delta_value_l2601_260172

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ - 1 → Δ = -11 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2601_260172
