import Mathlib

namespace NUMINAMATH_CALUDE_trail_mix_almonds_l2769_276976

theorem trail_mix_almonds (walnuts : ℝ) (total_nuts : ℝ) (almonds : ℝ) : 
  walnuts = 0.25 → total_nuts = 0.5 → almonds = total_nuts - walnuts → almonds = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_almonds_l2769_276976


namespace NUMINAMATH_CALUDE_least_congruent_number_proof_l2769_276914

/-- The least five-digit positive integer congruent to 7 (mod 18) and 4 (mod 9) -/
def least_congruent_number : ℕ := 10012

theorem least_congruent_number_proof :
  (least_congruent_number ≥ 10000) ∧
  (least_congruent_number < 100000) ∧
  (least_congruent_number % 18 = 7) ∧
  (least_congruent_number % 9 = 4) ∧
  (∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 18 = 7 ∧ n % 9 = 4 → n ≥ least_congruent_number) :=
by sorry

end NUMINAMATH_CALUDE_least_congruent_number_proof_l2769_276914


namespace NUMINAMATH_CALUDE_unitedNations75thAnniversary_l2769_276901

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (advanceDay d n)

-- Define the founding day of the United Nations
def unitedNationsFoundingDay : DayOfWeek := DayOfWeek.Wednesday

-- Define the number of days to advance for the 75th anniversary
def daysToAdvance : Nat := 93

-- Theorem statement
theorem unitedNations75thAnniversary :
  advanceDay unitedNationsFoundingDay daysToAdvance = DayOfWeek.Friday :=
sorry

end NUMINAMATH_CALUDE_unitedNations75thAnniversary_l2769_276901


namespace NUMINAMATH_CALUDE_basketball_score_difference_l2769_276989

theorem basketball_score_difference 
  (tim joe ken : ℕ) 
  (h1 : tim > joe)
  (h2 : tim = ken / 2)
  (h3 : tim + joe + ken = 100)
  (h4 : tim = 30) :
  tim - joe = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_difference_l2769_276989


namespace NUMINAMATH_CALUDE_original_workers_count_l2769_276968

/-- Represents the number of days required to complete the work. -/
def original_days : ℕ := 12

/-- Represents the number of days saved after additional workers joined. -/
def days_saved : ℕ := 4

/-- Represents the number of additional workers who joined. -/
def additional_workers : ℕ := 5

/-- Represents the number of additional workers working at twice the original rate. -/
def double_rate_workers : ℕ := 3

/-- Represents the number of additional workers working at the original rate. -/
def normal_rate_workers : ℕ := 2

/-- Theorem stating that the original number of workers is 16. -/
theorem original_workers_count : ℕ := by
  sorry

#check original_workers_count

end NUMINAMATH_CALUDE_original_workers_count_l2769_276968


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2769_276945

theorem matrix_equation_solution : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0.5, 1]
  M^4 - 3 • M^3 + 2 • M^2 = !![6, 12; 3, 6] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2769_276945


namespace NUMINAMATH_CALUDE_initial_men_correct_l2769_276902

/-- Represents the initial number of men employed by Nhai -/
def initial_men : ℕ := 100

/-- Represents the total length of the highway in kilometers -/
def highway_length : ℕ := 2

/-- Represents the initial number of days allocated for the project -/
def total_days : ℕ := 50

/-- Represents the initial number of work hours per day -/
def initial_hours_per_day : ℕ := 8

/-- Represents the number of days after which 1/3 of the work is completed -/
def days_for_one_third : ℕ := 25

/-- Represents the fraction of work completed after 25 days -/
def work_completed_fraction : ℚ := 1/3

/-- Represents the number of additional men hired -/
def additional_men : ℕ := 60

/-- Represents the new number of work hours per day after hiring additional men -/
def new_hours_per_day : ℕ := 10

theorem initial_men_correct :
  initial_men * total_days * initial_hours_per_day =
  (initial_men + additional_men) * (total_days - days_for_one_third) * new_hours_per_day :=
by sorry

end NUMINAMATH_CALUDE_initial_men_correct_l2769_276902


namespace NUMINAMATH_CALUDE_unit_circle_y_coordinate_l2769_276981

theorem unit_circle_y_coordinate 
  (α : Real) 
  (h1 : -3*π/2 < α ∧ α < 0) 
  (h2 : Real.cos (α - π/3) = -Real.sqrt 3 / 3) : 
  ∃ (x₀ y₀ : Real), 
    x₀^2 + y₀^2 = 1 ∧ 
    y₀ = Real.sin α ∧
    y₀ = (-Real.sqrt 6 - 3) / 6 :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_y_coordinate_l2769_276981


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2769_276923

/-- A rhombus with diagonals in ratio 1:2 and shorter diagonal 4 cm has side length 2√5 cm -/
theorem rhombus_side_length (d1 d2 side : ℝ) : 
  d1 > 0 → -- shorter diagonal is positive
  d2 = 2 * d1 → -- ratio of diagonals is 1:2
  d1 = 4 → -- shorter diagonal is 4 cm
  side^2 = (d1/2)^2 + (d2/2)^2 → -- Pythagorean theorem for half-diagonals
  side = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2769_276923


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2769_276960

theorem quadratic_inequality_solution_set (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x - b - 3/4 > 0) ↔ -3 < b ∧ b < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2769_276960


namespace NUMINAMATH_CALUDE_scooter_price_l2769_276962

-- Define the upfront payment and the percentage paid
def upfront_payment : ℝ := 240
def percentage_paid : ℝ := 20

-- State the theorem
theorem scooter_price : 
  (upfront_payment / (percentage_paid / 100)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_scooter_price_l2769_276962


namespace NUMINAMATH_CALUDE_sum_of_squares_with_inequality_l2769_276993

theorem sum_of_squares_with_inequality (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ (k : ℕ), n = 5 * k) 
  (h3 : ∃ (a b : ℤ), n = a^2 + b^2) : 
  ∃ (x y : ℤ), n = x^2 + y^2 ∧ x^2 ≥ 4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_inequality_l2769_276993


namespace NUMINAMATH_CALUDE_prob_three_out_of_five_l2769_276957

def prob_single_win : ℚ := 2/3

theorem prob_three_out_of_five :
  let n : ℕ := 5
  let k : ℕ := 3
  let p : ℚ := prob_single_win
  (n.choose k) * p^k * (1-p)^(n-k) = 80/243 := by
sorry

end NUMINAMATH_CALUDE_prob_three_out_of_five_l2769_276957


namespace NUMINAMATH_CALUDE_product_abcd_l2769_276953

theorem product_abcd (a b c d : ℚ) 
  (eq1 : 4*a - 2*b + 3*c + 5*d = 22)
  (eq2 : 2*(d+c) = b - 2)
  (eq3 : 4*b - c = a + 1)
  (eq4 : c + 1 = 2*d) :
  a * b * c * d = -30751860 / 11338912 := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l2769_276953


namespace NUMINAMATH_CALUDE_simplify_expression_l2769_276980

theorem simplify_expression (x : ℝ) : 2 * x^8 / x^4 = 2 * x^4 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2769_276980


namespace NUMINAMATH_CALUDE_prop_p_or_q_false_iff_a_in_range_l2769_276942

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, a^2 * x^2 + a * x - 2 = 0 ∧ -1 ≤ x ∧ x ≤ 1

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- Theorem statement
theorem prop_p_or_q_false_iff_a_in_range (a : ℝ) :
  (¬(p a ∨ q a)) ↔ ((-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)) :=
sorry

end NUMINAMATH_CALUDE_prop_p_or_q_false_iff_a_in_range_l2769_276942


namespace NUMINAMATH_CALUDE_game_probabilities_l2769_276906

/-- Represents the outcome of a single trial -/
inductive Outcome
  | win
  | loss

/-- Represents the result of 4 trials -/
def GameResult := List Outcome

/-- Counts the number of wins in a game result -/
def countWins : GameResult → Nat
  | [] => 0
  | (Outcome.win :: rest) => 1 + countWins rest
  | (Outcome.loss :: rest) => countWins rest

/-- The sample space of all possible game results -/
def sampleSpace : List GameResult := sorry

/-- The probability of an event occurring -/
def probability (event : GameResult → Bool) : Rat :=
  (sampleSpace.filter event).length / sampleSpace.length

/-- Winning at least once -/
def winAtLeastOnce (result : GameResult) : Bool :=
  countWins result ≥ 1

/-- Winning at most twice -/
def winAtMostTwice (result : GameResult) : Bool :=
  countWins result ≤ 2

theorem game_probabilities :
  probability winAtLeastOnce = 5/16 ∧
  probability winAtMostTwice = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_game_probabilities_l2769_276906


namespace NUMINAMATH_CALUDE_clockHandsOpposite_eq_48_l2769_276911

/-- The number of times clock hands are in a straight line but opposite in direction in a day -/
def clockHandsOpposite : ℕ :=
  let hoursOnClockFace : ℕ := 12
  let hoursInDay : ℕ := 24
  let occurrencesPerHour : ℕ := 2
  hoursInDay * occurrencesPerHour

/-- Theorem stating that clock hands are in a straight line but opposite in direction 48 times a day -/
theorem clockHandsOpposite_eq_48 : clockHandsOpposite = 48 := by
  sorry

end NUMINAMATH_CALUDE_clockHandsOpposite_eq_48_l2769_276911


namespace NUMINAMATH_CALUDE_power_of_special_sum_l2769_276944

theorem power_of_special_sum (a b : ℝ) 
  (h : b = Real.sqrt (1 - 2*a) + Real.sqrt (2*a - 1) + 3) : 
  a ^ b = (1/8 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_power_of_special_sum_l2769_276944


namespace NUMINAMATH_CALUDE_divide_by_repeating_decimal_l2769_276961

theorem divide_by_repeating_decimal :
  ∃ (x : ℚ), (∀ (n : ℕ), x = (3 * 10^n - 3) / (9 * 10^n)) ∧ (8 / x = 24) := by
  sorry

end NUMINAMATH_CALUDE_divide_by_repeating_decimal_l2769_276961


namespace NUMINAMATH_CALUDE_product_five_reciprocal_squares_sum_l2769_276959

theorem product_five_reciprocal_squares_sum (a b : ℕ) (h : a * b = 5) :
  (1 : ℝ) / (a^2 : ℝ) + (1 : ℝ) / (b^2 : ℝ) = 1.04 := by
  sorry

end NUMINAMATH_CALUDE_product_five_reciprocal_squares_sum_l2769_276959


namespace NUMINAMATH_CALUDE_books_ratio_l2769_276979

/-- Given the number of books for Loris, Lamont, and Darryl, 
    prove that the ratio of Lamont's books to Darryl's books is 2:1 -/
theorem books_ratio (Loris Lamont Darryl : ℕ) : 
  Loris + 3 = Lamont →  -- Loris needs three more books to have the same as Lamont
  Darryl = 20 →  -- Darryl has 20 books
  Loris + Lamont + Darryl = 97 →  -- Total number of books is 97
  Lamont / Darryl = 2 := by
sorry

end NUMINAMATH_CALUDE_books_ratio_l2769_276979


namespace NUMINAMATH_CALUDE_inequality_range_l2769_276955

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, |x + a| - |x + 1| < 2 * a) ↔ a > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2769_276955


namespace NUMINAMATH_CALUDE_no_valid_n_for_ap_l2769_276999

theorem no_valid_n_for_ap : ¬∃ (n : ℕ), n > 1 ∧ 
  180 % n = 0 ∧ 
  ∃ (k : ℕ), k^2 = (180 / n : ℚ) - (3/2 : ℚ) * n + (3/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_for_ap_l2769_276999


namespace NUMINAMATH_CALUDE_rachel_rona_age_ratio_l2769_276939

/-- Given the ages of Rachel, Rona, and Collete, prove that the ratio of Rachel's age to Rona's age is 2:1 -/
theorem rachel_rona_age_ratio (rachel_age rona_age collete_age : ℕ) : 
  rachel_age > rona_age →
  collete_age = rona_age / 2 →
  rona_age = 8 →
  rachel_age - collete_age = 12 →
  rachel_age / rona_age = 2 := by
  sorry


end NUMINAMATH_CALUDE_rachel_rona_age_ratio_l2769_276939


namespace NUMINAMATH_CALUDE_diagonals_intersect_l2769_276978

-- Define a regular 30-sided polygon
def RegularPolygon30 : Type := Unit

-- Define the sine function (simplified for this context)
noncomputable def sin (angle : ℝ) : ℝ := sorry

-- Define the cosine function (simplified for this context)
noncomputable def cos (angle : ℝ) : ℝ := sorry

-- Theorem statement
theorem diagonals_intersect (polygon : RegularPolygon30) : 
  (sin (6 * π / 180) * sin (18 * π / 180) * sin (84 * π / 180) = 
   sin (12 * π / 180) * sin (12 * π / 180) * sin (48 * π / 180)) ∧
  (sin (6 * π / 180) * sin (36 * π / 180) * sin (54 * π / 180) = 
   sin (30 * π / 180) * sin (12 * π / 180) * sin (12 * π / 180)) ∧
  (sin (36 * π / 180) * sin (18 * π / 180) * sin (6 * π / 180) = 
   cos (36 * π / 180) * cos (36 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_diagonals_intersect_l2769_276978


namespace NUMINAMATH_CALUDE_f_of_zero_l2769_276991

/-- Given functions g and f, prove that f(0) = 2 - 2∛2 -/
theorem f_of_zero (g : ℝ → ℝ) (f : ℝ → ℝ) 
  (hg : ∀ x, g x = 2 - x^3)
  (hf : ∀ x, f (g x) = x^3 - 2*x) :
  f 0 = 2 - 2 * Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_f_of_zero_l2769_276991


namespace NUMINAMATH_CALUDE_circle_tangent_lines_l2769_276938

/-- Given a circle with equation (x-1)^2 + (y+3)^2 = 4 and a point (-1, -1),
    the tangent lines from this point to the circle have equations x = -1 or y = -1 -/
theorem circle_tangent_lines (x y : ℝ) :
  let circle := (x - 1)^2 + (y + 3)^2 = 4
  let point := ((-1 : ℝ), (-1 : ℝ))
  let tangent1 := x = -1
  let tangent2 := y = -1
  (∃ (t : ℝ), circle ∧ (tangent1 ∨ tangent2) ∧
    (point.1 = t ∧ point.2 = -1) ∨ (point.1 = -1 ∧ point.2 = t)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_lines_l2769_276938


namespace NUMINAMATH_CALUDE_prob_less_than_130_l2769_276994

-- Define the normal distribution
def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

-- Define the cumulative distribution function (CDF) for the normal distribution
def normal_cdf (μ σ : ℝ) : ℝ → ℝ := sorry

-- Define the probability of a score being within μ ± kσ
def prob_within_k_sigma (k : ℝ) : ℝ := sorry

-- Theorem to prove
theorem prob_less_than_130 :
  let μ : ℝ := 110
  let σ : ℝ := 20
  ∃ ε > 0, |normal_cdf μ σ 130 - 0.97725| < ε :=
sorry

end NUMINAMATH_CALUDE_prob_less_than_130_l2769_276994


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2769_276937

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 11 ∧ x₂ = 2 - Real.sqrt 11 ∧
    x₁^2 - 4*x₁ - 7 = 0 ∧ x₂^2 - 4*x₂ - 7 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 1/3 ∧ y₂ = -2 ∧
    3*y₁^2 + 5*y₁ - 2 = 0 ∧ 3*y₂^2 + 5*y₂ - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2769_276937


namespace NUMINAMATH_CALUDE_quadratic_at_most_one_solution_l2769_276920

theorem quadratic_at_most_one_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 3 * x + 1 = 0) ∨ (∀ x y : ℝ, a * x^2 + 3 * x + 1 = 0 → a * y^2 + 3 * y + 1 = 0 → x = y) ↔
  a = 0 ∨ a ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_at_most_one_solution_l2769_276920


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l2769_276988

theorem unique_triplet_solution :
  ∀ x y z : ℕ,
    (1 + x / (y + z : ℚ))^2 + (1 + y / (z + x : ℚ))^2 + (1 + z / (x + y : ℚ))^2 = 27/4
    ↔ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l2769_276988


namespace NUMINAMATH_CALUDE_master_wang_parts_per_day_l2769_276921

/-- The number of parts Master Wang processed per day -/
def parts_per_day (a : ℕ) : ℚ :=
  (a + 3 : ℚ) / 8

/-- Theorem stating that the number of parts processed per day is (a + 3) / 8 -/
theorem master_wang_parts_per_day (a : ℕ) :
  parts_per_day a = (a + 3 : ℚ) / 8 := by
  sorry

#check master_wang_parts_per_day

end NUMINAMATH_CALUDE_master_wang_parts_per_day_l2769_276921


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l2769_276943

/-- Represents the number of ways to arrange frogs with given constraints -/
def frog_arrangements (total : ℕ) (green : ℕ) (red : ℕ) (blue : ℕ) : ℕ :=
  2 * (Nat.factorial blue) * (Nat.factorial red) * (Nat.factorial green)

/-- Theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 8 2 3 3 = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l2769_276943


namespace NUMINAMATH_CALUDE_car_to_stream_distance_l2769_276969

/-- The distance from the car to the stream in miles -/
def distance_car_to_stream : ℝ := 0.2

/-- The total distance hiked in miles -/
def total_distance : ℝ := 0.7

/-- The distance from the stream to the meadow in miles -/
def distance_stream_to_meadow : ℝ := 0.4

/-- The distance from the meadow to the campsite in miles -/
def distance_meadow_to_campsite : ℝ := 0.1

theorem car_to_stream_distance :
  distance_car_to_stream = total_distance - distance_stream_to_meadow - distance_meadow_to_campsite :=
by sorry

end NUMINAMATH_CALUDE_car_to_stream_distance_l2769_276969


namespace NUMINAMATH_CALUDE_three_pieces_per_box_l2769_276929

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the tape needed for a single box -/
def tapeForBox (box : BoxDimensions) : ℕ :=
  box.length + 2 * box.width

/-- The number of 15x30 boxes -/
def numSmallBoxes : ℕ := 5

/-- The number of 40x40 boxes -/
def numLargeBoxes : ℕ := 2

/-- The dimensions of the small boxes -/
def smallBox : BoxDimensions :=
  { length := 30, width := 15 }

/-- The dimensions of the large boxes -/
def largeBox : BoxDimensions :=
  { length := 40, width := 40 }

/-- The total amount of tape needed -/
def totalTape : ℕ := 540

/-- Theorem: Each box needs 3 pieces of tape -/
theorem three_pieces_per_box :
  (∃ (n : ℕ), n > 0 ∧
    n * (numSmallBoxes * tapeForBox smallBox + numLargeBoxes * tapeForBox largeBox) = totalTape * n ∧
    n * (numSmallBoxes + numLargeBoxes) = 3 * n * (numSmallBoxes + numLargeBoxes)) := by
  sorry


end NUMINAMATH_CALUDE_three_pieces_per_box_l2769_276929


namespace NUMINAMATH_CALUDE_potato_bag_weight_l2769_276922

def bag_weight : ℝ → Prop := λ w => w = 36 / (w / 2)

theorem potato_bag_weight : ∃ w : ℝ, bag_weight w ∧ w = 36 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l2769_276922


namespace NUMINAMATH_CALUDE_orange_seller_gain_percentage_l2769_276946

theorem orange_seller_gain_percentage 
  (loss_rate : ℝ) 
  (initial_sale_quantity : ℝ) 
  (new_sale_quantity : ℝ) 
  (loss_percentage : ℝ) : 
  loss_rate = 0.1 → 
  initial_sale_quantity = 10 → 
  new_sale_quantity = 6 → 
  loss_percentage = 10 → 
  ∃ (G : ℝ), G = 50 ∧ 
    (1 + G / 100) * (1 - loss_rate) * initial_sale_quantity / new_sale_quantity = 1 := by
  sorry

end NUMINAMATH_CALUDE_orange_seller_gain_percentage_l2769_276946


namespace NUMINAMATH_CALUDE_expression_simplification_l2769_276917

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 - 2) :
  (m^2 - 4*m + 4) / (m - 1) / ((3 / (m - 1)) - m - 1) = (-3 + 4 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2769_276917


namespace NUMINAMATH_CALUDE_heptagon_exterior_angle_sum_l2769_276933

/-- The exterior angle sum of a heptagon is 360 degrees. -/
theorem heptagon_exterior_angle_sum : ℝ :=
  360

#check heptagon_exterior_angle_sum

end NUMINAMATH_CALUDE_heptagon_exterior_angle_sum_l2769_276933


namespace NUMINAMATH_CALUDE_cookout_ratio_l2769_276974

def cookout_2004 : ℕ := 60
def cookout_2005 : ℕ := cookout_2004 / 2
def cookout_2006 : ℕ := 20

theorem cookout_ratio : 
  (cookout_2006 : ℚ) / cookout_2005 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cookout_ratio_l2769_276974


namespace NUMINAMATH_CALUDE_function_properties_l2769_276930

noncomputable def f (x φ : ℝ) : ℝ := Real.sin x * Real.cos φ + Real.cos x * Real.sin φ

theorem function_properties (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  -- The smallest positive period of f is 2π
  (∃ (T : ℝ), T > 0 ∧ T = 2 * π ∧ ∀ (x : ℝ), f x φ = f (x + T) φ) ∧
  -- If the graph of y = f(2x + π/4) is symmetric about x = π/6, then φ = 11π/12
  (∀ (x : ℝ), f (2 * (π/6 - x) + π/4) φ = f (2 * (π/6 + x) + π/4) φ → φ = 11 * π / 12) ∧
  -- If f(α - 2π/3) = √2/4, then sin 2α = -3/4
  (∀ (α : ℝ), f (α - 2 * π / 3) φ = Real.sqrt 2 / 4 → Real.sin (2 * α) = -3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2769_276930


namespace NUMINAMATH_CALUDE_hanks_total_donation_l2769_276990

/-- Proves that Hank's total donation is $200 given his earnings and donation percentages --/
theorem hanks_total_donation :
  let carwash_earnings : ℚ := 100
  let carwash_donation_percent : ℚ := 90 / 100
  let bake_sale_earnings : ℚ := 80
  let bake_sale_donation_percent : ℚ := 75 / 100
  let lawn_mowing_earnings : ℚ := 50
  let lawn_mowing_donation_percent : ℚ := 100 / 100
  
  carwash_earnings * carwash_donation_percent +
  bake_sale_earnings * bake_sale_donation_percent +
  lawn_mowing_earnings * lawn_mowing_donation_percent = 200 := by
  sorry


end NUMINAMATH_CALUDE_hanks_total_donation_l2769_276990


namespace NUMINAMATH_CALUDE_shirt_tie_combinations_l2769_276964

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of ties available. -/
def num_ties : ℕ := 7

/-- The number of specific shirt-tie pairs that cannot be worn together. -/
def num_restricted_pairs : ℕ := 3

/-- The total number of possible shirt-tie combinations. -/
def total_combinations : ℕ := num_shirts * num_ties

/-- The number of allowable shirt-tie combinations. -/
def allowable_combinations : ℕ := total_combinations - num_restricted_pairs

theorem shirt_tie_combinations : allowable_combinations = 53 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_combinations_l2769_276964


namespace NUMINAMATH_CALUDE_grid_midpoint_theorem_l2769_276913

theorem grid_midpoint_theorem (points : Finset (ℤ × ℤ)) :
  points.card = 5 →
  ∃ p q : ℤ × ℤ, p ∈ points ∧ q ∈ points ∧ p ≠ q ∧
    Even (p.1 + q.1) ∧ Even (p.2 + q.2) :=
by sorry

end NUMINAMATH_CALUDE_grid_midpoint_theorem_l2769_276913


namespace NUMINAMATH_CALUDE_constant_b_value_l2769_276995

theorem constant_b_value (a b c : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5/2) * (a * x^2 + b * x + c) = 
    6 * x^4 - 17 * x^3 + 11 * x^2 - 7/2 * x + 5/3) →
  b = -3 := by
sorry

end NUMINAMATH_CALUDE_constant_b_value_l2769_276995


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2769_276963

theorem smallest_three_digit_multiple_of_17 :
  (∀ n : ℕ, n ≥ 100 ∧ n < 102 → ¬(17 ∣ n)) ∧ (17 ∣ 102) := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2769_276963


namespace NUMINAMATH_CALUDE_polygon_with_special_angle_property_l2769_276965

theorem polygon_with_special_angle_property (n : ℕ) 
  (h : (n - 2) * 180 = 2 * 360) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_special_angle_property_l2769_276965


namespace NUMINAMATH_CALUDE_complement_M_in_U_l2769_276918

def U : Set ℕ := {x | x < 5 ∧ x > 0}
def M : Set ℕ := {x | x ^ 2 - 5 * x + 6 = 0}

theorem complement_M_in_U : U \ M = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l2769_276918


namespace NUMINAMATH_CALUDE_system_of_equations_solution_transformed_system_solution_l2769_276940

theorem system_of_equations_solution (x y : ℝ) :
  x + 2*y = 9 ∧ 2*x + y = 6 → x - y = -3 ∧ x + y = 5 := by sorry

theorem transformed_system_solution (m n x y : ℝ) :
  (m = 5 ∧ n = 4 ∧ 2*m - 3*n = -2 ∧ 3*m + 5*n = 35) →
  (2*(x+2) - 3*(y-1) = -2 ∧ 3*(x+2) + 5*(y-1) = 35 → x = 3 ∧ y = 5) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_transformed_system_solution_l2769_276940


namespace NUMINAMATH_CALUDE_count_multiples_eq_31_l2769_276947

/-- Count of positive integers less than 151 that are multiples of either 6 or 8, but not both -/
def count_multiples : ℕ := 
  let multiples_of_6 := Finset.filter (fun n => n % 6 = 0) (Finset.range 151)
  let multiples_of_8 := Finset.filter (fun n => n % 8 = 0) (Finset.range 151)
  (multiples_of_6 ∪ multiples_of_8).card - (multiples_of_6 ∩ multiples_of_8).card

theorem count_multiples_eq_31 : count_multiples = 31 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_eq_31_l2769_276947


namespace NUMINAMATH_CALUDE_f_2023_equals_2_l2769_276983

theorem f_2023_equals_2 (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2)) :
  f 2023 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_equals_2_l2769_276983


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l2769_276950

theorem girls_to_boys_ratio (total_students : ℕ) (girls_present : ℕ) (boys_absent : ℕ)
  (h1 : total_students = 250)
  (h2 : girls_present = 140)
  (h3 : boys_absent = 40) :
  (girls_present : ℚ) / (total_students - girls_present - boys_absent) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l2769_276950


namespace NUMINAMATH_CALUDE_sum_of_factors_36_l2769_276996

theorem sum_of_factors_36 : (List.sum (List.filter (λ x => 36 % x = 0) (List.range 37))) = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_36_l2769_276996


namespace NUMINAMATH_CALUDE_closest_point_l2769_276900

def v (t : ℝ) : ℝ × ℝ × ℝ := (3 - 4*t, -1 + 3*t, 2 + 5*t)

def a : ℝ × ℝ × ℝ := (-2, 7, 0)

def direction : ℝ × ℝ × ℝ := (-4, 3, 5)

theorem closest_point (t : ℝ) : 
  (v t - a) • direction = 0 ↔ t = 17/25 := by sorry

end NUMINAMATH_CALUDE_closest_point_l2769_276900


namespace NUMINAMATH_CALUDE_min_additional_weeks_to_win_l2769_276919

def prize_per_week : ℕ := 100
def weeks_already_won : ℕ := 2
def puppy_cost : ℕ := 1000

theorem min_additional_weeks_to_win (prize_per_week weeks_already_won puppy_cost : ℕ) :
  let amount_saved := prize_per_week * weeks_already_won
  let additional_amount_needed := puppy_cost - amount_saved
  (additional_amount_needed + prize_per_week - 1) / prize_per_week = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_additional_weeks_to_win_l2769_276919


namespace NUMINAMATH_CALUDE_magic_card_profit_l2769_276936

/-- Calculates the profit from selling a Magic card that triples in value -/
theorem magic_card_profit (initial_cost : ℝ) : 
  initial_cost > 0 → 2 * initial_cost = (3 * initial_cost) - initial_cost := by
  sorry

#check magic_card_profit

end NUMINAMATH_CALUDE_magic_card_profit_l2769_276936


namespace NUMINAMATH_CALUDE_grouping_schemes_l2769_276925

theorem grouping_schemes (drivers : Finset α) (ticket_sellers : Finset β) :
  (drivers.card = 4) → (ticket_sellers.card = 4) →
  (Finset.product drivers ticket_sellers).card = 24 := by
sorry

end NUMINAMATH_CALUDE_grouping_schemes_l2769_276925


namespace NUMINAMATH_CALUDE_price_change_theorem_l2769_276924

theorem price_change_theorem (p : ℝ) : 
  (1 + p / 100) * (1 - p / 200) = 1 + p / 300 → p = 100 / 3 :=
by sorry

end NUMINAMATH_CALUDE_price_change_theorem_l2769_276924


namespace NUMINAMATH_CALUDE_triangle_area_is_50_l2769_276910

/-- A square with side length 10 and lower left vertex at (0, 10) -/
structure Square where
  side : ℝ
  lower_left : ℝ × ℝ
  h_side : side = 10
  h_lower_left : lower_left = (0, 10)

/-- An isosceles triangle with base 10 on y-axis and lower right vertex at (0, 10) -/
structure IsoscelesTriangle where
  base : ℝ
  lower_right : ℝ × ℝ
  h_base : base = 10
  h_lower_right : lower_right = (0, 10)

/-- The area of the triangle formed by connecting the top vertex of the isosceles triangle
    to the top left vertex of the square -/
def triangle_area (s : Square) (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of the formed triangle is 50 square units -/
theorem triangle_area_is_50 (s : Square) (t : IsoscelesTriangle) :
  triangle_area s t = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_50_l2769_276910


namespace NUMINAMATH_CALUDE_cube_root_of_x_plus_y_l2769_276928

theorem cube_root_of_x_plus_y (x y : ℝ) : 
  (∃ (z : ℝ), z^2 = x + 2 ∧ (z = 3 ∨ z = -3)) → 
  (-y = -1) → 
  ∃ (w : ℝ), w^3 = x + y ∧ w = 2 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_x_plus_y_l2769_276928


namespace NUMINAMATH_CALUDE_bagel_shop_benches_l2769_276975

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- The problem statement -/
theorem bagel_shop_benches :
  let seating_capacity_base7 : ℕ := 321
  let people_per_bench : ℕ := 3
  let num_benches : ℕ := (base7ToBase10 seating_capacity_base7) / people_per_bench
  num_benches = 54 := by sorry

end NUMINAMATH_CALUDE_bagel_shop_benches_l2769_276975


namespace NUMINAMATH_CALUDE_f_13_equals_223_l2769_276987

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_13_equals_223 : f 13 = 223 := by
  sorry

end NUMINAMATH_CALUDE_f_13_equals_223_l2769_276987


namespace NUMINAMATH_CALUDE_school_play_scenes_l2769_276909

theorem school_play_scenes (Tom Ben Sam Nick Chris : ℕ) : 
  Tom = 8 ∧ Chris = 5 ∧ 
  Ben > Chris ∧ Ben < Tom ∧
  Sam > Chris ∧ Sam < Tom ∧
  Nick > Chris ∧ Nick < Tom ∧
  (∀ scene : ℕ, scene ≤ (Tom + Ben + Sam + Nick + Chris) / 2) ∧
  (∀ pair : ℕ × ℕ, pair.1 ≠ pair.2 → pair.1 ≤ 5 ∧ pair.2 ≤ 5 → 
    ∃ scene : ℕ, scene ≤ (Tom + Ben + Sam + Nick + Chris) / 2) →
  (Tom + Ben + Sam + Nick + Chris) / 2 = 16 := by
sorry

end NUMINAMATH_CALUDE_school_play_scenes_l2769_276909


namespace NUMINAMATH_CALUDE_sales_and_profit_l2769_276986

theorem sales_and_profit (x : ℤ) (y : ℝ) : 
  (8 ≤ x ∧ x ≤ 15) →
  (y = -5 * (x : ℝ) + 150) →
  (y = 105 ↔ x = 9) →
  (y = 95 ↔ x = 11) →
  (y = 85 ↔ x = 13) →
  (∃ (x : ℤ), 8 ≤ x ∧ x ≤ 15 ∧ (x - 8) * (-5 * x + 150) = 425 ↔ x = 13) :=
by sorry

end NUMINAMATH_CALUDE_sales_and_profit_l2769_276986


namespace NUMINAMATH_CALUDE_susan_strawberry_eating_l2769_276941

theorem susan_strawberry_eating (basket_capacity : ℕ) (total_picked : ℕ) (handful_size : ℕ) :
  basket_capacity = 60 →
  total_picked = 75 →
  handful_size = 5 →
  (total_picked - basket_capacity) / (total_picked / handful_size) = 1 := by
  sorry

end NUMINAMATH_CALUDE_susan_strawberry_eating_l2769_276941


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_to_2003_l2769_276927

theorem sum_of_last_two_digits_of_8_to_2003 :
  ∃ (n : ℕ), 8^2003 ≡ n [ZMOD 100] ∧ (n / 10 % 10 + n % 10 = 5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_to_2003_l2769_276927


namespace NUMINAMATH_CALUDE_B_power_15_minus_4_power_14_l2769_276956

def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 5; 0, 2]

theorem B_power_15_minus_4_power_14 :
  B^15 - 4 • B^14 = !![0, 5; 0, -2] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_4_power_14_l2769_276956


namespace NUMINAMATH_CALUDE_total_bottles_proof_l2769_276998

/-- Represents the total number of bottles -/
def total_bottles : ℕ := 180

/-- Represents the number of bottles containing only cider -/
def cider_bottles : ℕ := 40

/-- Represents the number of bottles containing only beer -/
def beer_bottles : ℕ := 80

/-- Represents the number of bottles given to the first house -/
def first_house_bottles : ℕ := 90

/-- Proves that the total number of bottles is 180 given the problem conditions -/
theorem total_bottles_proof :
  total_bottles = cider_bottles + beer_bottles + (2 * first_house_bottles - cider_bottles - beer_bottles) :=
by sorry

end NUMINAMATH_CALUDE_total_bottles_proof_l2769_276998


namespace NUMINAMATH_CALUDE_compound_interest_principal_l2769_276912

def simple_interest_rate : ℝ := 0.14
def compound_interest_rate : ℝ := 0.07
def simple_interest_time : ℝ := 6
def compound_interest_time : ℝ := 2
def simple_interest_amount : ℝ := 603.75

theorem compound_interest_principal (P_SI : ℝ) (P_CI : ℝ) 
  (h1 : P_SI * simple_interest_rate * simple_interest_time = simple_interest_amount)
  (h2 : P_SI * simple_interest_rate * simple_interest_time = 
        1/2 * (P_CI * ((1 + compound_interest_rate) ^ compound_interest_time - 1)))
  (h3 : P_SI = 603.75 / (simple_interest_rate * simple_interest_time)) :
  P_CI = 8333.33 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l2769_276912


namespace NUMINAMATH_CALUDE_probability_factor_90_less_than_8_l2769_276954

def positive_factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => x > 0 ∧ n % x = 0) (Finset.range (n + 1))

def factors_less_than (n k : ℕ) : Finset ℕ :=
  Finset.filter (λ x => x < k) (positive_factors n)

theorem probability_factor_90_less_than_8 :
  (Finset.card (factors_less_than 90 8) : ℚ) / (Finset.card (positive_factors 90) : ℚ) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_90_less_than_8_l2769_276954


namespace NUMINAMATH_CALUDE_pentagon_sum_edges_vertices_l2769_276952

/-- A pentagon is a polygon with 5 sides -/
structure Pentagon where
  edges : ℕ
  vertices : ℕ
  is_pentagon : edges = 5 ∧ vertices = 5

/-- The sum of edges and vertices in a pentagon is 10 -/
theorem pentagon_sum_edges_vertices (p : Pentagon) : p.edges + p.vertices = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_edges_vertices_l2769_276952


namespace NUMINAMATH_CALUDE_hyperbola_focus_product_l2769_276903

-- Define the hyperbola C
def hyperbola (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 9 - y^2 / m = 1

-- Define the foci F₁ and F₂
def is_focus (F : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola x y m ∧ 
  ((F.1 - x)^2 + (F.2 - y)^2 = 16 ∨ (F.1 - x)^2 + (F.2 - y)^2 = 16)

-- Define point P on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) (m : ℝ) : Prop :=
  hyperbola P.1 P.2 m

-- Define the dot product condition
def perpendicular_vectors (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Define the directrix condition
def directrix_through_focus (F : ℝ × ℝ) : Prop :=
  F.1 = -4

-- Main theorem
theorem hyperbola_focus_product (m : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  is_focus F₁ m →
  is_focus F₂ m →
  point_on_hyperbola P m →
  perpendicular_vectors P F₁ F₂ →
  (directrix_through_focus F₁ ∨ directrix_through_focus F₂) →
  ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 14^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_product_l2769_276903


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_k_is_1_A_intersect_B_nonempty_iff_k_geq_neg_1_l2769_276916

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def B (k : ℝ) : Set ℝ := {x : ℝ | x - k ≤ 0}

-- Define the complement of B in the universal set U (which is ℝ in this case)
def C_U_B (k : ℝ) : Set ℝ := {x : ℝ | x - k > 0}

theorem intersection_A_complement_B_when_k_is_1 :
  A ∩ C_U_B 1 = {x : ℝ | 1 < x ∧ x < 3} := by sorry

theorem A_intersect_B_nonempty_iff_k_geq_neg_1 :
  ∀ k : ℝ, (A ∩ B k).Nonempty ↔ k ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_k_is_1_A_intersect_B_nonempty_iff_k_geq_neg_1_l2769_276916


namespace NUMINAMATH_CALUDE_carpet_breadth_l2769_276932

/-- The breadth of the first carpet in meters -/
def b : ℝ := 6

/-- The length of the first carpet in meters -/
def l : ℝ := 1.44 * b

/-- The length of the second carpet in meters -/
def l2 : ℝ := 1.4 * l

/-- The breadth of the second carpet in meters -/
def b2 : ℝ := 1.25 * b

/-- The cost of the second carpet in rupees -/
def cost : ℝ := 4082.4

/-- The rate of the carpet in rupees per square meter -/
def rate : ℝ := 45

theorem carpet_breadth :
  b = 6 ∧
  l = 1.44 * b ∧
  l2 = 1.4 * l ∧
  b2 = 1.25 * b ∧
  cost = rate * l2 * b2 :=
by sorry

end NUMINAMATH_CALUDE_carpet_breadth_l2769_276932


namespace NUMINAMATH_CALUDE_two_digit_integers_theorem_l2769_276905

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def share_digit (a b : ℕ) : Prop :=
  (a / 10 = b / 10) ∨ (a / 10 = b % 10) ∨ (a % 10 = b / 10) ∨ (a % 10 = b % 10)

theorem two_digit_integers_theorem (a b : ℕ) :
  is_two_digit a ∧ is_two_digit b ∧
  ((a = b + 12) ∨ (b = a + 12)) ∧
  share_digit a b ∧
  ((digit_sum a = digit_sum b + 3) ∨ (digit_sum b = digit_sum a + 3)) →
  (∃ t : ℕ, 2 ≤ t ∧ t ≤ 8 ∧ a = 11 * t + 10 ∧ b = 11 * t - 2) ∨
  (∃ s : ℕ, 1 ≤ s ∧ s ≤ 6 ∧ a = 11 * s + 1 ∧ b = 11 * s + 13) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_integers_theorem_l2769_276905


namespace NUMINAMATH_CALUDE_lock_cost_l2769_276926

def total_cost : ℝ := 360

theorem lock_cost (helmet : ℝ) (bicycle : ℝ) (lock : ℝ) : 
  bicycle = 5 * helmet → 
  lock = helmet / 2 → 
  bicycle + helmet + lock = total_cost → 
  lock = 27.72 := by sorry

end NUMINAMATH_CALUDE_lock_cost_l2769_276926


namespace NUMINAMATH_CALUDE_imoProblem1995_l2769_276997

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points -/
def triangleArea (p q r : Point) : ℝ := sorry

/-- Checks if three points are collinear -/
def areCollinear (p q r : Point) : Prop := sorry

theorem imoProblem1995 (n : ℕ) (h_n : n > 3) :
  (∃ (A : Fin n → Point) (r : Fin n → ℝ),
    (∀ (i j k : Fin n), i < j → j < k → ¬areCollinear (A i) (A j) (A k)) ∧
    (∀ (i j k : Fin n), i < j → j < k → 
      triangleArea (A i) (A j) (A k) = r i + r j + r k)) ↔ 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_imoProblem1995_l2769_276997


namespace NUMINAMATH_CALUDE_largest_base5_three_digit_to_base10_l2769_276985

-- Define a function to convert a base-5 number to base 10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define the largest three-digit number in base 5
def largestBase5ThreeDigit : List Nat := [4, 4, 4]

-- Theorem statement
theorem largest_base5_three_digit_to_base10 :
  base5ToBase10 largestBase5ThreeDigit = 124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_three_digit_to_base10_l2769_276985


namespace NUMINAMATH_CALUDE_sum_of_squares_l2769_276904

theorem sum_of_squares (x y : ℝ) : 
  x * y = 10 → x^2 * y + x * y^2 + x + y = 80 → x^2 + y^2 = 3980 / 121 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2769_276904


namespace NUMINAMATH_CALUDE_radio_range_is_125_l2769_276984

/-- The range of radios for two teams traveling in opposite directions --/
def radio_range (speed1 speed2 time : ℝ) : ℝ :=
  speed1 * time + speed2 * time

/-- Theorem: The radio range for the given scenario is 125 miles --/
theorem radio_range_is_125 :
  radio_range 20 30 2.5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_radio_range_is_125_l2769_276984


namespace NUMINAMATH_CALUDE_product_equals_one_l2769_276970

/-- A geometric sequence with a specific property -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  root_property : a 3 * a 15 = 1 ∧ a 3 + a 15 = 6

/-- The product of five consecutive terms equals 1 -/
theorem product_equals_one (seq : GeometricSequence) :
  seq.a 7 * seq.a 8 * seq.a 9 * seq.a 10 * seq.a 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l2769_276970


namespace NUMINAMATH_CALUDE_shelter_dogs_l2769_276951

theorem shelter_dogs (dogs cats : ℕ) : 
  (dogs : ℚ) / cats = 15 / 7 →
  dogs / (cats + 16) = 15 / 11 →
  dogs = 60 :=
by sorry

end NUMINAMATH_CALUDE_shelter_dogs_l2769_276951


namespace NUMINAMATH_CALUDE_least_common_period_is_24_l2769_276973

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) + f (x - 4) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  IsPeriod f p ∧ ∀ q, IsPeriod f q → p ≤ q

theorem least_common_period_is_24 :
  ∃ p : ℝ, p = 24 ∧
    (∀ f : ℝ → ℝ, FunctionalEquation f → IsLeastPeriod f p) ∧
    (∀ q : ℝ, (∀ f : ℝ → ℝ, FunctionalEquation f → IsLeastPeriod f q) → p ≤ q) :=
sorry

end NUMINAMATH_CALUDE_least_common_period_is_24_l2769_276973


namespace NUMINAMATH_CALUDE_max_sum_with_lcm_constraint_max_sum_with_lcm_constraint_achievable_l2769_276907

theorem max_sum_with_lcm_constraint (m n : ℕ) : 
  m > 0 → n > 0 → m < 500 → n < 500 → Nat.lcm m n = (m - n)^2 → m + n ≤ 840 := by
  sorry

theorem max_sum_with_lcm_constraint_achievable : 
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m < 500 ∧ n < 500 ∧ Nat.lcm m n = (m - n)^2 ∧ m + n = 840 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_lcm_constraint_max_sum_with_lcm_constraint_achievable_l2769_276907


namespace NUMINAMATH_CALUDE_remainder_problem_l2769_276934

theorem remainder_problem (f y z : ℤ) 
  (hf : f % 5 = 3)
  (hy : y % 5 = 4)
  (hz : z % 7 = 6)
  (hsum : (f + y) % 15 = 7) :
  (f + y + z) % 35 = 3 ∧ (f + y + z) % 105 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2769_276934


namespace NUMINAMATH_CALUDE_course_selection_count_l2769_276949

/-- The number of courses available -/
def total_courses : ℕ := 7

/-- The number of courses each student must choose -/
def courses_to_choose : ℕ := 4

/-- The number of special courses (A and B) that cannot be chosen together -/
def special_courses : ℕ := 2

/-- The number of different course selection schemes -/
def selection_schemes : ℕ := Nat.choose total_courses courses_to_choose - 
  (Nat.choose special_courses special_courses * Nat.choose (total_courses - special_courses) (courses_to_choose - special_courses))

theorem course_selection_count : selection_schemes = 25 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_count_l2769_276949


namespace NUMINAMATH_CALUDE_workshop_transfer_l2769_276935

theorem workshop_transfer (w : ℕ) (n : ℕ) (x : ℕ) : 
  w ≥ 63 →
  w ≤ 64 →
  31 * w + n * (n + 1) / 2 = 1994 →
  (n = 4 ∧ x = 4) ∨ (n = 2 ∧ x = 21) :=
sorry

end NUMINAMATH_CALUDE_workshop_transfer_l2769_276935


namespace NUMINAMATH_CALUDE_equation_solutions_l2769_276971

def equation (x y n : ℤ) : Prop :=
  x^3 - 3*x*y^2 + y^3 = n

theorem equation_solutions (n : ℤ) (hn : n > 0) :
  (∃ (x y : ℤ), equation x y n → 
    equation (y - x) (-x) n ∧ equation (-y) (x - y) n) ∧
  (n = 2891 → ¬∃ (x y : ℤ), equation x y n) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2769_276971


namespace NUMINAMATH_CALUDE_extension_point_coordinates_l2769_276967

/-- Given two points P₁ and P₂ in ℝ², and a point P on the extension line of P₁P₂ 
    such that |⃗P₁P| = 2|⃗PP₂|, prove that P has coordinates (-2, 11). -/
theorem extension_point_coordinates (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) → 
  P₂ = (0, 5) → 
  (∃ t : ℝ, P = P₁ + t • (P₂ - P₁)) → 
  ‖P - P₁‖ = 2 * ‖P₂ - P‖ → 
  P = (-2, 11) := by
  sorry

end NUMINAMATH_CALUDE_extension_point_coordinates_l2769_276967


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l2769_276948

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * x (n + 1) - x n

theorem no_perfect_squares_in_sequence : ∀ n : ℕ, ¬∃ k : ℤ, x n = k * k := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l2769_276948


namespace NUMINAMATH_CALUDE_circle_locus_line_theorem_l2769_276977

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the locus of M
def locus_M (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - Real.sqrt 3)

-- Define the length product condition
def length_product_condition (k : ℝ) : Prop :=
  let d := |Real.sqrt 3 * k| / Real.sqrt (1 + k^2)
  let AB := 2 * Real.sqrt (4 - d^2)
  let CD := 4 * (1 + k^2) / (1 + 4 * k^2)
  AB * CD = 8 * Real.sqrt 10 / 5

-- Main theorem
theorem circle_locus_line_theorem :
  ∀ (k : ℝ),
  (∀ (x y : ℝ), circle_O x y → locus_M (x/2) (y/2)) ∧
  (length_product_condition k ↔ (k = 1 ∨ k = -1)) :=
sorry

end NUMINAMATH_CALUDE_circle_locus_line_theorem_l2769_276977


namespace NUMINAMATH_CALUDE_george_first_half_correct_l2769_276966

def trivia_game (first_half_correct : ℕ) (second_half_correct : ℕ) (points_per_question : ℕ) (total_score : ℕ) : Prop :=
  first_half_correct * points_per_question + second_half_correct * points_per_question = total_score

theorem george_first_half_correct :
  ∃ (x : ℕ), trivia_game x 4 3 30 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_george_first_half_correct_l2769_276966


namespace NUMINAMATH_CALUDE_equation_solutions_l2769_276958

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1) ∧
  (∀ x : ℝ, 2*x^3 = -16 ↔ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2769_276958


namespace NUMINAMATH_CALUDE_sams_initial_points_l2769_276915

theorem sams_initial_points :
  ∀ initial_points : ℕ,
  initial_points + 3 = 95 →
  initial_points = 92 :=
by
  sorry

end NUMINAMATH_CALUDE_sams_initial_points_l2769_276915


namespace NUMINAMATH_CALUDE_weight_placement_theorem_l2769_276982

def factorial_double (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => (2 * k + 1) * factorial_double k

def weight_placement_ways (n : ℕ) : ℕ :=
  factorial_double n

theorem weight_placement_theorem (n : ℕ) (h : n > 0) :
  weight_placement_ways n = factorial_double n :=
by
  sorry

end NUMINAMATH_CALUDE_weight_placement_theorem_l2769_276982


namespace NUMINAMATH_CALUDE_set_operations_l2769_276972

def U : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
def A : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem set_operations :
  (A ∩ B = {x | 0 < x ∧ x ≤ 1}) ∧
  (B ∪ (U \ A) = {x | (-5 ≤ x ∧ x ≤ 1) ∨ (3 < x ∧ x ≤ 5)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2769_276972


namespace NUMINAMATH_CALUDE_segment_HY_length_is_15_sqrt3_div_2_l2769_276992

/-- Regular hexagon with side length 3 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Point Y on the extension of side CD such that CY = 4CD -/
def extend_side (h : RegularHexagon) (C D Y : ℝ × ℝ) : Prop :=
  dist C Y = 4 * dist C D ∧ dist C D = h.side_length

/-- The length of segment HY in a regular hexagon with extended side -/
def segment_HY_length (h : RegularHexagon) (H Y : ℝ × ℝ) : ℝ :=
  dist H Y

/-- Theorem: The length of segment HY is 15√3/2 -/
theorem segment_HY_length_is_15_sqrt3_div_2 (h : RegularHexagon) 
  (C D E F G H Y : ℝ × ℝ) 
  (hex : RegularHexagon) 
  (ext : extend_side hex C D Y) :
  segment_HY_length hex H Y = 15 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_segment_HY_length_is_15_sqrt3_div_2_l2769_276992


namespace NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l2769_276931

theorem opposite_signs_abs_sum_less_abs_diff (x y : ℝ) (h : x * y < 0) :
  |x + y| < |x - y| := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_abs_sum_less_abs_diff_l2769_276931


namespace NUMINAMATH_CALUDE_banana_permutations_count_l2769_276908

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_count :
  banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_count_l2769_276908
