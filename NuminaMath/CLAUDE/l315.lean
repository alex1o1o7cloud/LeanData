import Mathlib

namespace factorization_of_4x_cubed_minus_x_l315_31527

theorem factorization_of_4x_cubed_minus_x (x : ℝ) : 4 * x^3 - x = x * (2*x + 1) * (2*x - 1) := by
  sorry

end factorization_of_4x_cubed_minus_x_l315_31527


namespace cubic_point_tangent_l315_31515

theorem cubic_point_tangent (a : ℝ) (h : a^3 = 27) : 
  Real.tan (π / a) = Real.sqrt 3 := by
  sorry

end cubic_point_tangent_l315_31515


namespace right_triangle_hypotenuse_and_perimeter_l315_31552

theorem right_triangle_hypotenuse_and_perimeter :
  let a : ℝ := 8.5
  let b : ℝ := 15
  let h : ℝ := Real.sqrt (a^2 + b^2)
  let perimeter : ℝ := a + b + h
  h = 17.25 ∧ perimeter = 40.75 := by
  sorry

end right_triangle_hypotenuse_and_perimeter_l315_31552


namespace bowling_tournament_sequences_l315_31571

/-- Represents a tournament with a fixed number of players and rounds. -/
structure Tournament :=
  (num_players : ℕ)
  (num_rounds : ℕ)
  (outcomes_per_match : ℕ)

/-- Calculates the number of possible award distribution sequences for a given tournament. -/
def award_sequences (t : Tournament) : ℕ :=
  t.outcomes_per_match ^ t.num_rounds

/-- Theorem stating that a tournament with 5 players, 4 rounds, and 2 outcomes per match has 16 possible award sequences. -/
theorem bowling_tournament_sequences :
  ∃ t : Tournament, t.num_players = 5 ∧ t.num_rounds = 4 ∧ t.outcomes_per_match = 2 ∧ award_sequences t = 16 :=
sorry

end bowling_tournament_sequences_l315_31571


namespace F_neg_one_eq_zero_l315_31502

noncomputable def F (x : ℝ) : ℝ := Real.sqrt (abs (x + 1)) + (9 / Real.pi) * Real.arctan (Real.sqrt (abs (x + 1)))

theorem F_neg_one_eq_zero : F (-1) = 0 := by
  sorry

end F_neg_one_eq_zero_l315_31502


namespace new_tv_cost_l315_31525

/-- The cost of a new TV given the dimensions and price of an old TV, and the price difference per square inch. -/
theorem new_tv_cost (old_width old_height old_cost new_width new_height price_diff : ℝ) :
  old_width = 24 →
  old_height = 16 →
  old_cost = 672 →
  new_width = 48 →
  new_height = 32 →
  price_diff = 1 →
  (old_cost / (old_width * old_height) - price_diff) * (new_width * new_height) = 1152 := by
  sorry

end new_tv_cost_l315_31525


namespace bakery_outdoor_tables_l315_31568

/-- Given a bakery setup with indoor and outdoor tables, prove the number of outdoor tables. -/
theorem bakery_outdoor_tables
  (indoor_tables : ℕ)
  (indoor_chairs_per_table : ℕ)
  (outdoor_chairs_per_table : ℕ)
  (total_chairs : ℕ)
  (h1 : indoor_tables = 8)
  (h2 : indoor_chairs_per_table = 3)
  (h3 : outdoor_chairs_per_table = 3)
  (h4 : total_chairs = 60) :
  (total_chairs - indoor_tables * indoor_chairs_per_table) / outdoor_chairs_per_table = 12 := by
  sorry

end bakery_outdoor_tables_l315_31568


namespace factorization_equality_l315_31578

theorem factorization_equality (x y : ℝ) : x^3*y - x*y = x*y*(x - 1)*(x + 1) := by
  sorry

end factorization_equality_l315_31578


namespace rectangle_length_from_square_l315_31546

theorem rectangle_length_from_square (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 20 →
  rect_width = 14 →
  4 * square_side = 2 * (rect_width + rect_length) →
  rect_length = 26 := by
sorry

end rectangle_length_from_square_l315_31546


namespace equation_solution_l315_31577

theorem equation_solution :
  ∀ x : ℚ, (1 / 4 : ℚ) + 7 / x = 13 / x + (1 / 9 : ℚ) → x = 216 / 5 := by
  sorry

end equation_solution_l315_31577


namespace same_terminal_side_as_405_degrees_l315_31524

theorem same_terminal_side_as_405_degrees : ∀ (k : ℤ),
  ∃ (n : ℤ), 405 = n * 360 + 45 ∧ (k * 360 + 45) % 360 = 45 := by
  sorry

end same_terminal_side_as_405_degrees_l315_31524


namespace min_probability_cards_unique_min_probability_cards_l315_31581

/-- Represents the probability of a card being red-side up after flips -/
def probability_red_up (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2) / 676

/-- The statement to prove -/
theorem min_probability_cards :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 →
    (probability_red_up 13 ≤ probability_red_up k ∧
     probability_red_up 38 ≤ probability_red_up k) :=
by sorry

/-- Uniqueness of the minimum probability cards -/
theorem unique_min_probability_cards :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 →
    (k ≠ 13 ∧ k ≠ 38 →
      probability_red_up 13 < probability_red_up k ∧
      probability_red_up 38 < probability_red_up k) :=
by sorry

end min_probability_cards_unique_min_probability_cards_l315_31581


namespace complex_square_equation_l315_31507

theorem complex_square_equation : 
  ∀ z : ℂ, z^2 = -57 - 48*I ↔ z = 3 - 8*I ∨ z = -3 + 8*I :=
by sorry

end complex_square_equation_l315_31507


namespace largest_unreachable_score_l315_31539

/-- 
Given that:
1. Easy questions earn 3 points.
2. Harder questions earn 7 points.
3. Scores are achieved by combinations of these point values.

Prove that 11 is the largest integer that cannot be expressed as a linear combination of 3 and 7 
with non-negative integer coefficients.
-/
theorem largest_unreachable_score : 
  ∀ n : ℕ, n > 11 → ∃ x y : ℕ, n = 3 * x + 7 * y :=
by sorry

end largest_unreachable_score_l315_31539


namespace triangle_area_from_square_sides_l315_31562

theorem triangle_area_from_square_sides (a b c : Real) 
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) : 
  (1/2) * a * b = 24 := by
sorry

end triangle_area_from_square_sides_l315_31562


namespace solution_set_l315_31523

theorem solution_set (x : ℝ) : 
  x > 4 → x^3 - 8*x^2 + 16*x > 64 ∧ x^2 - 4*x + 5 > 0 := by
  sorry

end solution_set_l315_31523


namespace r_earnings_l315_31522

/-- Represents the daily earnings of individuals p, q, and r -/
structure Earnings where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The conditions given in the problem -/
def problem_conditions (e : Earnings) : Prop :=
  e.p + e.q + e.r = 1980 / 9 ∧
  e.p + e.r = 600 / 5 ∧
  e.q + e.r = 910 / 7

/-- The theorem stating that under the given conditions, r earns 30 rs per day -/
theorem r_earnings (e : Earnings) : problem_conditions e → e.r = 30 := by
  sorry

end r_earnings_l315_31522


namespace auspicious_count_l315_31594

/-- Returns true if n is an auspicious number (multiple of 6 with digit sum 6) -/
def isAuspicious (n : Nat) : Bool :=
  n % 6 = 0 && (n / 100 + (n / 10) % 10 + n % 10 = 6)

/-- Count of auspicious numbers between 100 and 999 -/
def countAuspicious : Nat :=
  (List.range 900).map (· + 100)
    |>.filter isAuspicious
    |>.length

theorem auspicious_count : countAuspicious = 12 := by
  sorry

end auspicious_count_l315_31594


namespace fireflies_win_by_five_l315_31520

/-- Represents a basketball team's score -/
structure TeamScore where
  initial : ℕ
  final_baskets : ℕ
  basket_value : ℕ

/-- Calculates the final score of a team -/
def final_score (team : TeamScore) : ℕ :=
  team.initial + team.final_baskets * team.basket_value

/-- Represents the scores of both teams in the basketball game -/
structure GameScore where
  hornets : TeamScore
  fireflies : TeamScore

/-- The theorem stating the final score difference between Fireflies and Hornets -/
theorem fireflies_win_by_five (game : GameScore)
  (h1 : game.hornets = ⟨86, 2, 2⟩)
  (h2 : game.fireflies = ⟨74, 7, 3⟩) :
  final_score game.fireflies - final_score game.hornets = 5 := by
  sorry

#check fireflies_win_by_five

end fireflies_win_by_five_l315_31520


namespace indeterminate_neutral_eight_year_boys_l315_31538

structure Classroom where
  total_children : Nat
  happy_children : Nat
  sad_children : Nat
  neutral_children : Nat
  total_boys : Nat
  total_girls : Nat
  happy_boys : Nat
  happy_girls : Nat
  sad_boys : Nat
  sad_girls : Nat
  age_seven_total : Nat
  age_seven_boys : Nat
  age_seven_girls : Nat
  age_eight_total : Nat
  age_eight_boys : Nat
  age_eight_girls : Nat
  age_nine_total : Nat
  age_nine_boys : Nat
  age_nine_girls : Nat

def classroom : Classroom := {
  total_children := 60,
  happy_children := 30,
  sad_children := 10,
  neutral_children := 20,
  total_boys := 16,
  total_girls := 44,
  happy_boys := 6,
  happy_girls := 12,
  sad_boys := 6,
  sad_girls := 4,
  age_seven_total := 20,
  age_seven_boys := 8,
  age_seven_girls := 12,
  age_eight_total := 25,
  age_eight_boys := 5,
  age_eight_girls := 20,
  age_nine_total := 15,
  age_nine_boys := 3,
  age_nine_girls := 12
}

theorem indeterminate_neutral_eight_year_boys (c : Classroom) : 
  c = classroom → 
  ¬∃ (n : Nat), n = c.age_eight_boys - (number_of_happy_eight_year_boys + number_of_sad_eight_year_boys) :=
by sorry

end indeterminate_neutral_eight_year_boys_l315_31538


namespace weeks_in_month_is_four_l315_31531

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := sorry

/-- The standard work hours per week -/
def standard_hours_per_week : ℕ := 20

/-- The number of months worked -/
def months_worked : ℕ := 2

/-- The additional hours worked due to covering a shift -/
def additional_hours : ℕ := 20

/-- The total hours worked over the period -/
def total_hours_worked : ℕ := 180

theorem weeks_in_month_is_four :
  weeks_in_month = 4 :=
by sorry

end weeks_in_month_is_four_l315_31531


namespace max_value_theorem_l315_31575

theorem max_value_theorem (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 2*x + y + z = 4) : 
  x^2 + x*(y + z) + y*z ≤ 4 :=
by sorry

end max_value_theorem_l315_31575


namespace percy_swimming_hours_l315_31555

/-- Percy's daily swimming hours on weekdays -/
def weekday_hours : ℕ := 2

/-- Number of weekdays Percy swims per week -/
def weekdays_per_week : ℕ := 5

/-- Percy's weekend swimming hours -/
def weekend_hours : ℕ := 3

/-- Number of weeks -/
def num_weeks : ℕ := 4

/-- Total swimming hours over the given number of weeks -/
def total_swimming_hours : ℕ := 
  num_weeks * (weekday_hours * weekdays_per_week + weekend_hours)

theorem percy_swimming_hours : total_swimming_hours = 52 := by
  sorry

end percy_swimming_hours_l315_31555


namespace area_ratio_of_squares_l315_31559

/-- Given three square regions A, B, and C with perimeters 16, 20, and 40 units respectively,
    the ratio of the area of region B to the area of region C is 1/4 -/
theorem area_ratio_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (pa : 4 * a = 16) (pb : 4 * b = 20) (pc : 4 * c = 40) :
  (b ^ 2) / (c ^ 2) = 1 / 4 := by
  sorry

end area_ratio_of_squares_l315_31559


namespace jason_gave_nine_cards_l315_31543

/-- The number of Pokemon cards Jason started with -/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason has left -/
def remaining_cards : ℕ := 4

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem jason_gave_nine_cards : cards_given = 9 := by sorry

end jason_gave_nine_cards_l315_31543


namespace exists_phi_and_x0_for_sin_product_equals_one_l315_31529

theorem exists_phi_and_x0_for_sin_product_equals_one : 
  ∃ (φ : ℝ) (x₀ : ℝ), Real.sin x₀ * Real.sin (x₀ + φ) = 1 := by
  sorry

end exists_phi_and_x0_for_sin_product_equals_one_l315_31529


namespace target_hit_probability_l315_31592

theorem target_hit_probability (p_a p_b : ℝ) (h_a : p_a = 0.9) (h_b : p_b = 0.8) 
  (h_independent : True) : 1 - (1 - p_a) * (1 - p_b) = 0.98 := by
  sorry

end target_hit_probability_l315_31592


namespace star_1993_1935_l315_31598

-- Define the operation *
def star (x y : ℤ) : ℤ := x - y

-- State the theorem
theorem star_1993_1935 : star 1993 1935 = 58 := by
  -- Assumptions
  have h1 : ∀ x : ℤ, star x x = 0 := by sorry
  have h2 : ∀ x y z : ℤ, star x (star y z) = star (star x y) z := by sorry
  
  -- Proof
  sorry

end star_1993_1935_l315_31598


namespace probability_theorem_l315_31574

def is_valid_pair (b c : Int) : Prop :=
  (b.natAbs ≤ 6) ∧ (c.natAbs ≤ 6)

def has_non_real_or_non_positive_roots (b c : Int) : Prop :=
  (b^2 < 4*c) ∨ (b ≥ 0) ∨ (b^2 ≤ 4*c)

def total_pairs : Nat := 13 * 13

def valid_pairs : Nat := 150

theorem probability_theorem :
  (Nat.cast valid_pairs / Nat.cast total_pairs : ℚ) = 150 / 169 := by
  sorry

end probability_theorem_l315_31574


namespace problem_solution_l315_31540

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (a x : ℝ) : ℝ := 1/2 * x^2 - 4*a*x + a * log x + a + 1/2

/-- The function g(x) as defined in the problem -/
noncomputable def g (a x : ℝ) : ℝ := f a x + 2*a

/-- The derivative of g(x) -/
noncomputable def g' (a x : ℝ) : ℝ := x - 4*a + a/x

theorem problem_solution (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    g' a x₁ = 0 ∧ g' a x₂ = 0 ∧
    g a x₁ + g a x₂ ≥ g' a (x₁ * x₂)) →
  1/4 < a ∧ a ≤ 1 :=
by sorry

end problem_solution_l315_31540


namespace difference_max_min_both_l315_31533

def total_students : ℕ := 1500

def spanish_min : ℕ := 1050
def spanish_max : ℕ := 1125

def french_min : ℕ := 525
def french_max : ℕ := 675

def min_both : ℕ := spanish_min + french_min - total_students
def max_both : ℕ := spanish_max + french_max - total_students

theorem difference_max_min_both : max_both - min_both = 225 := by
  sorry

end difference_max_min_both_l315_31533


namespace quadratic_root_is_one_l315_31556

/-- 
Given a quadratic function f(x) = x^2 + ax + b, where:
- The graph of f intersects the y-axis at (0, b)
- The graph of f intersects the x-axis at (b, 0)
- b ≠ 0

Prove that the other root of f(x) = 0 is equal to 1.
-/
theorem quadratic_root_is_one (a b : ℝ) (hb : b ≠ 0) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  (f 0 = b) → (f b = 0) → ∃ c, c ≠ b ∧ f c = 0 ∧ c = 1 := by
  sorry


end quadratic_root_is_one_l315_31556


namespace smallest_k_value_l315_31509

theorem smallest_k_value (a b c d x y z t : ℝ) :
  ∃ k : ℝ, k = 1 ∧ 
  (∀ k' : ℝ, (a * d - b * c + y * z - x * t + (a + c) * (y + t) - (b + d) * (x + z) ≤ 
    k' * (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (x^2 + y^2) + Real.sqrt (z^2 + t^2))^2) → 
  k ≤ k') ∧
  (a * d - b * c + y * z - x * t + (a + c) * (y + t) - (b + d) * (x + z) ≤ 
    k * (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (x^2 + y^2) + Real.sqrt (z^2 + t^2))^2) := by
  sorry

end smallest_k_value_l315_31509


namespace min_value_sum_ratios_l315_31535

theorem min_value_sum_ratios (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / a + (a + b + c) / b + (a + b + c) / c ≥ 9 ∧
  ((a + b + c) / a + (a + b + c) / b + (a + b + c) / c = 9 ↔ a = b ∧ b = c) :=
by sorry

end min_value_sum_ratios_l315_31535


namespace problem_1_problem_2_l315_31518

-- Define the ★ operation
def star (a b : ℤ) : ℤ := a * b - 1

-- Theorem 1
theorem problem_1 : star (-1) 3 = -4 := by sorry

-- Theorem 2
theorem problem_2 : star (-2) (star (-3) (-4)) = -21 := by sorry

end problem_1_problem_2_l315_31518


namespace condition_relationship_l315_31573

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a + b = 1 → 4 * a * b ≤ 1) ∧
  (∃ a b, 4 * a * b ≤ 1 ∧ a + b ≠ 1) :=
by
  sorry

end condition_relationship_l315_31573


namespace power_sum_equals_123_l315_31560

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^10 + b^10 = 123 -/
theorem power_sum_equals_123 (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end power_sum_equals_123_l315_31560


namespace right_triangle_side_lengths_l315_31589

/-- A right-angled triangle with given incircle and circumcircle radii -/
structure RightTriangle where
  -- The radius of the incircle
  inradius : ℝ
  -- The radius of the circumcircle
  circumradius : ℝ
  -- The lengths of the three sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- Conditions
  inradius_positive : 0 < inradius
  circumradius_positive : 0 < circumradius
  right_angle : a ^ 2 + b ^ 2 = c ^ 2
  incircle_condition : a + b - c = 2 * inradius
  circumcircle_condition : c = 2 * circumradius

/-- The main theorem stating the side lengths of the triangle -/
theorem right_triangle_side_lengths (t : RightTriangle) 
    (h1 : t.inradius = 8)
    (h2 : t.circumradius = 41) :
    (t.a = 18 ∧ t.b = 80 ∧ t.c = 82) ∨ (t.a = 80 ∧ t.b = 18 ∧ t.c = 82) := by
  sorry

end right_triangle_side_lengths_l315_31589


namespace rectangle_area_l315_31564

theorem rectangle_area (x y : ℕ) : 
  1 ≤ x ∧ x < 10 ∧ 1 ≤ y ∧ y < 10 →
  ∃ n : ℕ, (1100 * x + 11 * y) = n^2 →
  x * y = 28 :=
by sorry

end rectangle_area_l315_31564


namespace positive_c_geq_one_l315_31579

theorem positive_c_geq_one (a b : ℕ+) (c : ℝ) 
  (h_c_pos : c > 0) 
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : 
  c ≥ 1 := by
sorry

end positive_c_geq_one_l315_31579


namespace prob_four_ones_twelve_dice_l315_31587

def n : ℕ := 12  -- total number of dice
def k : ℕ := 4   -- number of dice showing 1
def s : ℕ := 6   -- number of sides on each die

-- Probability of rolling exactly k ones out of n dice
def prob_exactly_k_ones : ℚ :=
  (Nat.choose n k : ℚ) * (1 / s) ^ k * ((s - 1) / s) ^ (n - k)

theorem prob_four_ones_twelve_dice :
  prob_exactly_k_ones = 495 * 390625 / 2176782336 := by
  sorry

end prob_four_ones_twelve_dice_l315_31587


namespace lois_final_book_count_l315_31563

def calculate_final_books (initial_books : ℕ) : ℕ :=
  let books_after_giving := initial_books - (initial_books / 4)
  let nonfiction_books := (books_after_giving * 60) / 100
  let kept_nonfiction := nonfiction_books / 2
  let fiction_books := books_after_giving - nonfiction_books
  let kept_fiction := fiction_books - (fiction_books / 3)
  let new_books := 12
  kept_nonfiction + kept_fiction + new_books

theorem lois_final_book_count :
  calculate_final_books 150 = 76 := by
  sorry

end lois_final_book_count_l315_31563


namespace redistribution_amount_l315_31528

def earnings : List ℕ := [18, 22, 26, 32, 47]

theorem redistribution_amount (earnings : List ℕ) (h1 : earnings = [18, 22, 26, 32, 47]) :
  let total := earnings.sum
  let equalShare := total / earnings.length
  let maxEarning := earnings.maximum?
  maxEarning.map (λ max => max - equalShare) = some 18 := by
  sorry

end redistribution_amount_l315_31528


namespace no_real_roots_l315_31532

theorem no_real_roots : ∀ x : ℝ, x^2 + |x| + 1 ≠ 0 := by
  sorry

end no_real_roots_l315_31532


namespace remainder_sum_powers_mod_five_l315_31599

theorem remainder_sum_powers_mod_five :
  (9^7 + 8^8 + 7^9) % 5 = 2 := by
  sorry

end remainder_sum_powers_mod_five_l315_31599


namespace num_assignment_plans_l315_31591

/-- The number of male doctors -/
def num_male_doctors : ℕ := 6

/-- The number of female doctors -/
def num_female_doctors : ℕ := 4

/-- The number of male doctors to be selected -/
def selected_male_doctors : ℕ := 3

/-- The number of female doctors to be selected -/
def selected_female_doctors : ℕ := 2

/-- The number of regions -/
def num_regions : ℕ := 5

/-- Function to calculate the number of assignment plans -/
def calculate_assignment_plans : ℕ := sorry

/-- Theorem stating the number of different assignment plans -/
theorem num_assignment_plans : 
  calculate_assignment_plans = 12960 := by sorry

end num_assignment_plans_l315_31591


namespace integer_solution_correct_rational_solution_correct_l315_31554

-- Define the equation
def equation (x y : ℚ) : Prop := 2 * x^3 + x * y - 7 = 0

-- Define the set of integer solutions
def integer_solutions : Set (ℤ × ℤ) :=
  {(1, 5), (-1, -9), (7, -97), (-7, -99)}

-- Define the rational solution function
def rational_solution (x : ℚ) : ℚ := 7 / x - 2 * x^2

-- Theorem for integer solutions
theorem integer_solution_correct :
  ∀ (x y : ℤ), (x, y) ∈ integer_solutions → equation (x : ℚ) (y : ℚ) :=
sorry

-- Theorem for rational solutions
theorem rational_solution_correct :
  ∀ (x : ℚ), x ≠ 0 → equation x (rational_solution x) :=
sorry

end integer_solution_correct_rational_solution_correct_l315_31554


namespace cartesian_to_polar_conversion_l315_31567

theorem cartesian_to_polar_conversion (x y ρ θ : Real) :
  x = -1 ∧ y = Real.sqrt 3 →
  ρ = 2 ∧ θ = 2 * Real.pi / 3 →
  ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y ∧ ρ^2 = x^2 + y^2 :=
by sorry

end cartesian_to_polar_conversion_l315_31567


namespace power_four_times_base_equals_power_five_l315_31596

theorem power_four_times_base_equals_power_five (a : ℝ) : a^4 * a = a^5 := by
  sorry

end power_four_times_base_equals_power_five_l315_31596


namespace pages_used_l315_31526

def cards_per_page : ℕ := 3
def new_cards : ℕ := 3
def old_cards : ℕ := 9

theorem pages_used :
  (new_cards + old_cards) / cards_per_page = 4 :=
by sorry

end pages_used_l315_31526


namespace unique_number_property_l315_31530

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end unique_number_property_l315_31530


namespace interest_difference_theorem_l315_31558

theorem interest_difference_theorem (P : ℝ) : 
  let r : ℝ := 5 / 100  -- 5% interest rate
  let t : ℝ := 2        -- 2 years
  let simple_interest := P * r * t
  let compound_interest := P * ((1 + r) ^ t - 1)
  compound_interest - simple_interest = 20 → P = 8000 := by
sorry

end interest_difference_theorem_l315_31558


namespace container_count_l315_31557

theorem container_count (x y : ℕ) : 
  27 * x = 65 * y + 34 → 
  y ≤ 44 → 
  x + y = 66 :=
by sorry

end container_count_l315_31557


namespace dave_money_l315_31586

theorem dave_money (dave_amount : ℝ) : 
  (2 / 3 * (3 * dave_amount - 12) = 84) → dave_amount = 46 := by
  sorry

end dave_money_l315_31586


namespace sqrt_inequality_l315_31545

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (3 - x) - Real.sqrt (x + 1) > (1 : ℝ) / 2 ↔ 
  -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8 := by
sorry

end sqrt_inequality_l315_31545


namespace actual_tissue_diameter_l315_31561

/-- Given a circular piece of tissue magnified by an electron microscope, 
    this theorem proves that the actual diameter of the tissue is 0.001 centimeters. -/
theorem actual_tissue_diameter 
  (magnification : ℝ) 
  (magnified_diameter : ℝ) 
  (h1 : magnification = 1000)
  (h2 : magnified_diameter = 1) : 
  magnified_diameter / magnification = 0.001 := by
  sorry

end actual_tissue_diameter_l315_31561


namespace remaining_cube_volume_l315_31570

/-- The volume of a cube with edge length a -/
def cube_volume (a : ℝ) : ℝ := a ^ 3

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The volume of the remaining part after cutting small cubes from vertices -/
def remaining_volume (edge_length : ℝ) (small_cube_volume : ℝ) : ℝ :=
  cube_volume edge_length - (cube_vertices : ℝ) * small_cube_volume

/-- Theorem: The volume of a cube with edge length 3 cm, after removing 
    small cubes of volume 1 cm³ from each of its vertices, is 19 cm³ -/
theorem remaining_cube_volume : 
  remaining_volume 3 1 = 19 := by
  sorry

end remaining_cube_volume_l315_31570


namespace equal_strawberry_division_l315_31500

def strawberry_division (brother_baskets : ℕ) (strawberries_per_basket : ℕ) : ℕ :=
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let kimberly_strawberries := 8 * brother_strawberries
  let parents_strawberries := kimberly_strawberries - 93
  let total_strawberries := kimberly_strawberries + brother_strawberries + parents_strawberries
  total_strawberries / 4

theorem equal_strawberry_division :
  strawberry_division 3 15 = 168 := by
  sorry

end equal_strawberry_division_l315_31500


namespace correct_proposition_l315_31588

-- Define proposition P
def P : Prop := ∀ x : ℝ, x^2 ≥ 0

-- Define proposition Q
def Q : Prop := ∃ x : ℚ, x^2 ≠ 3

-- Theorem to prove
theorem correct_proposition : P ∨ (¬Q) := by
  sorry

end correct_proposition_l315_31588


namespace right_triangle_area_l315_31593

/-- The area of a right triangle with given side lengths -/
theorem right_triangle_area 
  (X Y Z : ℝ × ℝ) -- Points in 2D plane
  (h_right : (X.1 - Y.1) * (X.1 - Z.1) + (X.2 - Y.2) * (X.2 - Z.2) = 0) -- Right angle at X
  (h_xy : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 15) -- XY = 15
  (h_xz : Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 10) -- XZ = 10
  (h_median : ∃ M : ℝ × ℝ, M.1 = (Y.1 + Z.1) / 2 ∧ M.2 = (Y.2 + Z.2) / 2 ∧ 
    (X.1 - M.1) * (Y.1 - Z.1) + (X.2 - M.2) * (Y.2 - Z.2) = 0) -- Median bisects angle X
  : (1 / 2 : ℝ) * 15 * 10 = 75 := by
  sorry

end right_triangle_area_l315_31593


namespace total_cost_of_items_l315_31553

/-- The total cost of items given their price relationships -/
theorem total_cost_of_items (chair_price : ℝ) : 
  chair_price > 0 →
  let table_price := 3 * chair_price
  let couch_price := 5 * table_price
  couch_price = 300 →
  chair_price + table_price + couch_price = 380 := by
  sorry

end total_cost_of_items_l315_31553


namespace second_hole_depth_calculation_l315_31512

/-- Calculates the depth of a second hole given the conditions of two digging projects -/
def second_hole_depth (workers1 hours1 depth1 workers2 hours2 : ℕ) : ℚ :=
  let man_hours1 := workers1 * hours1
  let man_hours2 := workers2 * hours2
  (man_hours2 * depth1 : ℚ) / man_hours1

theorem second_hole_depth_calculation (workers1 hours1 depth1 extra_workers hours2 : ℕ) :
  second_hole_depth workers1 hours1 depth1 (workers1 + extra_workers) hours2 = 40 :=
by
  -- The proof goes here
  sorry

#eval second_hole_depth 45 8 30 80 6

end second_hole_depth_calculation_l315_31512


namespace multiply_divide_equality_l315_31521

theorem multiply_divide_equality : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end multiply_divide_equality_l315_31521


namespace percentage_relation_l315_31566

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.08 * x) (h2 : b = 0.16 * x) :
  a = 0.5 * b := by sorry

end percentage_relation_l315_31566


namespace usual_work_week_l315_31505

/-- Proves that given the conditions, the employee's usual work week is 40 hours -/
theorem usual_work_week (hourly_rate : ℝ) (weekly_salary : ℝ) (worked_fraction : ℝ) :
  hourly_rate = 15 →
  weekly_salary = 480 →
  worked_fraction = 4 / 5 →
  worked_fraction * (weekly_salary / hourly_rate) = 40 := by
sorry

end usual_work_week_l315_31505


namespace triangle_properties_l315_31549

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (t : Triangle) 
  (h1 : 3 * Real.cos t.A * Real.cos t.C * (Real.tan t.A * Real.tan t.C - 1) = 1)
  (h2 : t.a + t.c = 3 * Real.sqrt 3 / 2)
  (h3 : t.b = Real.sqrt 3) : 
  Real.sin (2 * t.B - 5 * Real.pi / 6) = (7 - 4 * Real.sqrt 6) / 18 ∧ 
  t.a * t.c * Real.sin t.B / 2 = 15 * Real.sqrt 2 / 32 := by
  sorry

end triangle_properties_l315_31549


namespace max_k_value_l315_31595

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 4 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 4 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 = 4^2 * (x^2 / y^2 + y^2 / x^2) + 4 * (x / y + y / x) :=
sorry

end max_k_value_l315_31595


namespace chocolate_division_l315_31572

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) :
  total_chocolate = 60 / 7 →
  num_piles = 5 →
  piles_given = 2 →
  piles_given * (total_chocolate / num_piles) = 24 / 7 := by
  sorry

end chocolate_division_l315_31572


namespace abc_sum_product_l315_31580

theorem abc_sum_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^4 + b^4 + c^4 = 128) :
  a*b + b*c + c*a = -8 := by
sorry

end abc_sum_product_l315_31580


namespace sequence_terms_coprime_l315_31550

def sequence_a : ℕ → ℕ
  | 0 => 2
  | n + 1 => (sequence_a n)^2 - sequence_a n + 1

theorem sequence_terms_coprime (m n : ℕ) (h : m ≠ n) : 
  Nat.gcd (sequence_a m) (sequence_a n) = 1 := by
  sorry

end sequence_terms_coprime_l315_31550


namespace complex_fraction_simplification_l315_31551

theorem complex_fraction_simplification :
  (3 + Complex.I) / (1 - Complex.I) = 1 + 2 * Complex.I := by
  sorry

end complex_fraction_simplification_l315_31551


namespace irrational_x_with_rational_expressions_l315_31503

theorem irrational_x_with_rational_expressions (x : ℝ) :
  Irrational x →
  ∃ q₁ q₂ : ℚ, (x^3 - 6*x : ℝ) = (q₁ : ℝ) ∧ (x^4 - 8*x^2 : ℝ) = (q₂ : ℝ) →
  x = Real.sqrt 6 ∨ x = -Real.sqrt 6 ∨
  x = 1 + Real.sqrt 3 ∨ x = -(1 + Real.sqrt 3) ∨
  x = 1 - Real.sqrt 3 ∨ x = -(1 - Real.sqrt 3) :=
by sorry

end irrational_x_with_rational_expressions_l315_31503


namespace largest_expression_l315_31534

theorem largest_expression (P Q : ℝ) (h1 : P = 1000) (h2 : Q = 0.01) :
  (P / Q > P + Q) ∧ (P / Q > P * Q) ∧ (P / Q > Q / P) ∧ (P / Q > P - Q) :=
by sorry

end largest_expression_l315_31534


namespace girls_combined_score_is_87_l315_31513

-- Define the schools
structure School where
  boys_score : ℝ
  girls_score : ℝ
  combined_score : ℝ

-- Define the problem parameters
def cedar : School := { boys_score := 68, girls_score := 80, combined_score := 73 }
def drake : School := { boys_score := 75, girls_score := 88, combined_score := 83 }
def combined_boys_score : ℝ := 74

-- Theorem statement
theorem girls_combined_score_is_87 :
  ∃ (cedar_boys cedar_girls drake_boys drake_girls : ℕ),
    (cedar_boys : ℝ) * cedar.boys_score + (cedar_girls : ℝ) * cedar.girls_score = 
      (cedar_boys + cedar_girls : ℝ) * cedar.combined_score ∧
    (drake_boys : ℝ) * drake.boys_score + (drake_girls : ℝ) * drake.girls_score = 
      (drake_boys + drake_girls : ℝ) * drake.combined_score ∧
    ((cedar_boys : ℝ) * cedar.boys_score + (drake_boys : ℝ) * drake.boys_score) / 
      (cedar_boys + drake_boys : ℝ) = combined_boys_score ∧
    ((cedar_girls : ℝ) * cedar.girls_score + (drake_girls : ℝ) * drake.girls_score) / 
      (cedar_girls + drake_girls : ℝ) = 87 := by
  sorry

end girls_combined_score_is_87_l315_31513


namespace Q_characterization_l315_31519

def Ω : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 2008}

def superior (p q : ℝ × ℝ) : Prop := p.1 ≤ q.1 ∧ p.2 ≥ q.2

def Q : Set (ℝ × ℝ) := {q ∈ Ω | ∀ p ∈ Ω, superior p q → p = q}

theorem Q_characterization : Q = {p ∈ Ω | p.1^2 + p.2^2 = 2008 ∧ p.1 ≤ 0 ∧ p.2 ≥ 0} := by sorry

end Q_characterization_l315_31519


namespace original_fraction_proof_l315_31506

theorem original_fraction_proof (x y : ℚ) : 
  (1.15 * x) / (0.92 * y) = 15 / 16 → x / y = 3 / 4 := by
  sorry

end original_fraction_proof_l315_31506


namespace equation_solution_l315_31511

theorem equation_solution :
  let f (x : ℝ) := 4 * (3 * x)^2 + (3 * x) + 5 - (3 * (9 * x^2 + 3 * x + 3))
  ∀ x : ℝ, f x = 0 ↔ x = (1 + Real.sqrt 5) / 3 ∨ x = (1 - Real.sqrt 5) / 3 := by
  sorry

end equation_solution_l315_31511


namespace simplify_expression_l315_31514

theorem simplify_expression (s : ℝ) : 120 * s - 32 * s = 88 * s := by
  sorry

end simplify_expression_l315_31514


namespace calculation_proof_l315_31582

theorem calculation_proof :
  (6.42 - 2.8 + 3.58 = 7.2) ∧ (0.36 / (0.4 * (6.1 - 4.6)) = 0.6) := by
  sorry

end calculation_proof_l315_31582


namespace golden_ratio_greater_than_half_l315_31536

theorem golden_ratio_greater_than_half : (Real.sqrt 5 - 1) / 2 > 1 / 2 := by
  sorry

end golden_ratio_greater_than_half_l315_31536


namespace percentage_increase_in_students_l315_31541

theorem percentage_increase_in_students (students_this_year students_last_year : ℕ) 
  (h1 : students_this_year = 960)
  (h2 : students_last_year = 800) :
  (students_this_year - students_last_year) / students_last_year * 100 = 20 := by
  sorry

end percentage_increase_in_students_l315_31541


namespace summer_reading_goal_l315_31583

/-- The number of books Carlos read in June -/
def june_books : ℕ := 42

/-- The number of books Carlos read in July -/
def july_books : ℕ := 28

/-- The number of books Carlos read in August -/
def august_books : ℕ := 30

/-- Carlos' goal for the number of books to read during summer vacation -/
def summer_goal : ℕ := june_books + july_books + august_books

theorem summer_reading_goal : summer_goal = 100 := by
  sorry

end summer_reading_goal_l315_31583


namespace walking_speeds_l315_31585

/-- The speeds of two people walking on a highway -/
theorem walking_speeds (x y : ℝ) : 
  (30 * x - 30 * y = 300) →  -- If both walk eastward for 30 minutes, A catches up with B
  (2 * x + 2 * y = 300) →    -- If they walk towards each other, they meet after 2 minutes
  (x = 80 ∧ y = 70) :=        -- Then A's speed is 80 m/min and B's speed is 70 m/min
by
  sorry

end walking_speeds_l315_31585


namespace reciprocal_of_two_thirds_l315_31548

def reciprocal (a b : ℚ) : ℚ := b / a

theorem reciprocal_of_two_thirds :
  reciprocal (2 : ℚ) 3 = (3 : ℚ) / 2 := by
  sorry

end reciprocal_of_two_thirds_l315_31548


namespace no_solution_exists_l315_31537

theorem no_solution_exists : ¬∃ (a b : ℕ+), 
  a * b + 52 = 20 * Nat.lcm a b + 15 * Nat.gcd a b := by
  sorry

end no_solution_exists_l315_31537


namespace value_of_d_l315_31576

theorem value_of_d (r s t u d : ℕ+) 
  (h1 : r^5 = s^4)
  (h2 : t^3 = u^2)
  (h3 : t - r = 19)
  (h4 : d = u - s) :
  d = 757 := by
  sorry

end value_of_d_l315_31576


namespace sum_34_27_base5_l315_31504

def base10_to_base5 (n : ℕ) : List ℕ :=
  sorry

theorem sum_34_27_base5 :
  base10_to_base5 (34 + 27) = [2, 2, 1] :=
sorry

end sum_34_27_base5_l315_31504


namespace jason_pokemon_cards_l315_31542

theorem jason_pokemon_cards (initial_cards new_cards : ℕ) 
  (h1 : initial_cards = 676)
  (h2 : new_cards = 224) :
  initial_cards + new_cards = 900 := by
  sorry

end jason_pokemon_cards_l315_31542


namespace min_max_bound_l315_31501

theorem min_max_bound (x₁ x₂ x₃ : ℝ) (h_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  1 ≤ (x₁ + 3*x₂ + 5*x₃)*(x₁ + x₂/3 + x₃/5) ∧ 
  (x₁ + 3*x₂ + 5*x₃)*(x₁ + x₂/3 + x₃/5) ≤ 9/5 := by
  sorry

#check min_max_bound

end min_max_bound_l315_31501


namespace characterize_satisfying_functions_l315_31547

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x * (f (x + 1) - f x) = f x) ∧
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|)

/-- The theorem stating the form of functions satisfying the conditions -/
theorem characterize_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfyingFunction f →
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ |k| ≤ 1 :=
by sorry

end characterize_satisfying_functions_l315_31547


namespace grandparents_uncle_difference_l315_31517

/-- Represents the money Gwen received from each family member -/
structure MoneyReceived where
  dad : ℕ
  mom : ℕ
  uncle : ℕ
  aunt : ℕ
  cousin : ℕ
  grandparents : ℕ

/-- The amount of money Gwen received for her birthday -/
def gwens_birthday_money : MoneyReceived :=
  { dad := 5
  , mom := 10
  , uncle := 8
  , aunt := 3
  , cousin := 6
  , grandparents := 15
  }

/-- Theorem stating the difference between money received from grandparents and uncle -/
theorem grandparents_uncle_difference :
  gwens_birthday_money.grandparents - gwens_birthday_money.uncle = 7 := by
  sorry

end grandparents_uncle_difference_l315_31517


namespace hall_length_l315_31565

/-- Hall represents a rectangular hall with specific properties -/
structure Hall where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- Properties of the hall -/
def hall_properties (h : Hall) : Prop :=
  h.width = 15 ∧
  h.volume = 1687.5 ∧
  2 * (h.length * h.width) = 2 * (h.length * h.height) + 2 * (h.width * h.height)

/-- Theorem stating that a hall with the given properties has a length of 15 meters -/
theorem hall_length (h : Hall) (hp : hall_properties h) : h.length = 15 := by
  sorry

end hall_length_l315_31565


namespace marbles_remainder_l315_31510

theorem marbles_remainder (a b c : ℤ) 
  (ha : a % 8 = 5)
  (hb : b % 8 = 7)
  (hc : c % 8 = 2) : 
  (a + b + c) % 8 = 6 := by
sorry

end marbles_remainder_l315_31510


namespace remainder_3_1000_mod_7_l315_31516

theorem remainder_3_1000_mod_7 : 3^1000 % 7 = 4 := by
  sorry

end remainder_3_1000_mod_7_l315_31516


namespace total_waiting_time_l315_31584

def days_first_appointment : ℕ := 4
def days_second_appointment : ℕ := 20
def weeks_for_effectiveness : ℕ := 2
def days_per_week : ℕ := 7

theorem total_waiting_time :
  days_first_appointment + days_second_appointment + (weeks_for_effectiveness * days_per_week) = 38 := by
  sorry

end total_waiting_time_l315_31584


namespace smallest_valid_seating_l315_31569

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  totalChairs : ℕ
  seatedPeople : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone. -/
def validSeating (table : CircularTable) : Prop :=
  table.seatedPeople > 0 ∧ 
  table.totalChairs / table.seatedPeople ≤ 4

/-- The theorem stating the smallest number of people that can be seated while satisfying the condition. -/
theorem smallest_valid_seating (table : CircularTable) : 
  table.totalChairs = 72 → 
  (∀ n : ℕ, n < table.seatedPeople → ¬validSeating ⟨table.totalChairs, n⟩) →
  validSeating table →
  table.seatedPeople = 18 :=
sorry

end smallest_valid_seating_l315_31569


namespace polygon_interior_angles_sum_l315_31597

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- The original number of sides of the polygon -/
def original_sides : ℕ := 7

/-- The number of sides after doubling -/
def doubled_sides : ℕ := 2 * original_sides

theorem polygon_interior_angles_sum : 
  sum_interior_angles doubled_sides = 2160 := by
  sorry

end polygon_interior_angles_sum_l315_31597


namespace solution_set_of_inequality_l315_31544

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x^2 - x - 6 > 0) ↔ (x < -3/2 ∨ x > 2) :=
by sorry

end solution_set_of_inequality_l315_31544


namespace legitimate_paths_count_l315_31590

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Defines the grid dimensions -/
def gridWidth : Nat := 12
def gridHeight : Nat := 4

/-- Defines the start and end points -/
def pointA : GridPoint := ⟨0, 0⟩
def pointB : GridPoint := ⟨gridWidth - 1, gridHeight - 1⟩

/-- Checks if a path is legitimate based on the column restrictions -/
def isLegitimate (path : List GridPoint) : Bool :=
  path.all fun p =>
    (p.x ≠ 2 || p.y = 0 || p.y = 1 || p.y = gridHeight - 1) &&
    (p.x ≠ 4 || p.y = 0 || p.y = gridHeight - 2 || p.y = gridHeight - 1)

/-- Counts the number of legitimate paths from A to B -/
def countLegitimatePaths : Nat :=
  sorry -- The actual implementation would go here

/-- The main theorem to prove -/
theorem legitimate_paths_count :
  countLegitimatePaths = 1289 := by
  sorry

end legitimate_paths_count_l315_31590


namespace triangle_angle_calculation_l315_31508

theorem triangle_angle_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC
  -- b = √2
  b = Real.sqrt 2 →
  -- c = 1
  c = 1 →
  -- B = 45°
  B = 45 * π / 180 →
  -- Then C = 30°
  C = 30 * π / 180 := by
sorry

end triangle_angle_calculation_l315_31508
