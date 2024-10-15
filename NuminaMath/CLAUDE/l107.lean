import Mathlib

namespace NUMINAMATH_CALUDE_parabola_transformation_l107_10778

/-- A parabola is defined by its coefficient and horizontal shift -/
structure Parabola where
  a : ℝ
  h : ℝ

/-- The equation of a parabola y = a(x-h)^2 -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2

/-- The transformation that shifts a parabola horizontally -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, h := p.h + shift }

theorem parabola_transformation (p1 p2 : Parabola) :
  p1.a = 2 ∧ p1.h = 0 ∧ p2.a = 2 ∧ p2.h = 3 →
  ∃ (shift : ℝ), shift = 3 ∧ horizontal_shift p1 shift = p2 :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l107_10778


namespace NUMINAMATH_CALUDE_pass_probability_theorem_l107_10776

/-- A highway with n intersecting roads. -/
structure Highway where
  n : ℕ
  k : ℕ
  h1 : 0 < n
  h2 : k ≤ n

/-- The probability of a car passing through the k-th intersection on a highway. -/
def pass_probability (h : Highway) : ℚ :=
  (2 * h.k * h.n - 2 * h.k^2 + 2 * h.k - 1) / (h.n^2 : ℚ)

/-- Theorem stating the probability of a car passing through the k-th intersection. -/
theorem pass_probability_theorem (h : Highway) :
  pass_probability h = (2 * h.k * h.n - 2 * h.k^2 + 2 * h.k - 1) / (h.n^2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_pass_probability_theorem_l107_10776


namespace NUMINAMATH_CALUDE_subset_condition_l107_10783

def P : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def S (a : ℝ) : Set ℝ := {x | a*x + 2 = 0}

theorem subset_condition (a : ℝ) : S a ⊆ P ↔ a = 0 ∨ a = 2 ∨ a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l107_10783


namespace NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l107_10733

/-- An arithmetic progression where the sum of the first 15 terms is three times the sum of the first 8 terms -/
def ArithmeticProgression (a d : ℝ) : Prop :=
  let S : ℕ → ℝ := λ n => n / 2 * (2 * a + (n - 1) * d)
  S 15 = 3 * S 8

theorem ratio_first_term_to_common_difference 
  {a d : ℝ} (h : ArithmeticProgression a d) : 
  a / d = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l107_10733


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l107_10739

-- Define the property of lines being skew
def are_skew_lines (a b : Line3D) : Prop := sorry

-- Define the property of lines having no common points
def have_no_common_points (a b : Line3D) : Prop := sorry

-- Theorem stating that "are_skew_lines" is a sufficient but not necessary condition for "have_no_common_points"
theorem skew_lines_sufficient_not_necessary (a b : Line3D) :
  (are_skew_lines a b → have_no_common_points a b) ∧
  ¬(have_no_common_points a b → are_skew_lines a b) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l107_10739


namespace NUMINAMATH_CALUDE_jackies_break_duration_l107_10749

/-- Represents Jackie's push-up performance --/
structure PushupPerformance where
  pushups_per_10sec : ℕ
  pushups_per_minute_with_breaks : ℕ
  num_breaks : ℕ

/-- Calculates the duration of each break in seconds --/
def break_duration (perf : PushupPerformance) : ℕ :=
  let pushups_per_minute := perf.pushups_per_10sec * 6
  let total_break_time := (pushups_per_minute - perf.pushups_per_minute_with_breaks) * (10 / perf.pushups_per_10sec)
  total_break_time / perf.num_breaks

/-- Theorem: Jackie's break duration is 8 seconds --/
theorem jackies_break_duration :
  let jackie : PushupPerformance := ⟨5, 22, 2⟩
  break_duration jackie = 8 := by
  sorry

end NUMINAMATH_CALUDE_jackies_break_duration_l107_10749


namespace NUMINAMATH_CALUDE_folded_carbon_copies_l107_10703

/-- Represents the number of carbon copies produced given the initial number of sheets,
    carbon papers, and whether the setup is folded or not -/
def carbonCopies (sheets : ℕ) (carbons : ℕ) (folded : Bool) : ℕ :=
  if folded then
    2 * (sheets - 1)
  else
    carbons

/-- Theorem stating that with 3 sheets, 2 carbons, and folded setup, 4 carbon copies are produced -/
theorem folded_carbon_copies :
  carbonCopies 3 2 true = 4 := by sorry

end NUMINAMATH_CALUDE_folded_carbon_copies_l107_10703


namespace NUMINAMATH_CALUDE_square_perimeter_when_area_equals_side_l107_10737

theorem square_perimeter_when_area_equals_side : ∀ s : ℝ,
  s > 0 → s^2 = s → 4 * s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_when_area_equals_side_l107_10737


namespace NUMINAMATH_CALUDE_function_bound_l107_10753

open Real

theorem function_bound (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (π/4), f x = sin (2*x) - Real.sqrt 3 * cos (2*x)) →
  (∀ x ∈ Set.Ioo 0 (π/4), |f x| < m) →
  m ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_function_bound_l107_10753


namespace NUMINAMATH_CALUDE_pastry_solution_l107_10709

/-- Represents the number of pastries each person has -/
structure Pastries where
  calvin : ℕ
  phoebe : ℕ
  frank : ℕ
  grace : ℕ

/-- The conditions of the pastry problem -/
def pastry_problem (p : Pastries) : Prop :=
  p.grace = 30 ∧
  p.calvin > p.frank ∧
  p.phoebe > p.frank ∧
  p.calvin = p.grace - 5 ∧
  p.phoebe = p.grace - 5 ∧
  p.calvin + p.phoebe + p.frank + p.grace = 97

/-- The theorem stating the solution to the pastry problem -/
theorem pastry_solution (p : Pastries) (h : pastry_problem p) :
  p.calvin - p.frank = 8 ∧ p.phoebe - p.frank = 8 := by
  sorry

end NUMINAMATH_CALUDE_pastry_solution_l107_10709


namespace NUMINAMATH_CALUDE_summer_camp_probability_l107_10774

/-- Given a summer camp with 30 kids total, 22 in coding, and 19 in robotics,
    the probability of selecting two kids from different workshops is 32/39. -/
theorem summer_camp_probability (total : ℕ) (coding : ℕ) (robotics : ℕ) 
  (h_total : total = 30)
  (h_coding : coding = 22)
  (h_robotics : robotics = 19) :
  (total.choose 2 - (coding - (coding + robotics - total)).choose 2 - (robotics - (coding + robotics - total)).choose 2) / total.choose 2 = 32 / 39 :=
by sorry

end NUMINAMATH_CALUDE_summer_camp_probability_l107_10774


namespace NUMINAMATH_CALUDE_average_string_length_l107_10736

theorem average_string_length :
  let string1 : ℚ := 2
  let string2 : ℚ := 5
  let string3 : ℚ := 3
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_string_length_l107_10736


namespace NUMINAMATH_CALUDE_cos_180_degrees_l107_10716

theorem cos_180_degrees : Real.cos (π) = -1 := by sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l107_10716


namespace NUMINAMATH_CALUDE_triangle_inequality_l107_10761

theorem triangle_inequality (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l107_10761


namespace NUMINAMATH_CALUDE_cheese_cookie_price_l107_10721

/-- Proves that the price of a pack of cheese cookies is $1 -/
theorem cheese_cookie_price (boxes_per_carton : ℕ) (packs_per_box : ℕ) (cost_dozen_cartons : ℕ) 
  (h1 : boxes_per_carton = 12)
  (h2 : packs_per_box = 10)
  (h3 : cost_dozen_cartons = 1440) :
  (cost_dozen_cartons : ℚ) / ((12 * boxes_per_carton * packs_per_box) : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cheese_cookie_price_l107_10721


namespace NUMINAMATH_CALUDE_range_of_a_l107_10711

-- Define p and q as predicates on real numbers
def p (a : ℝ) (x : ℝ) : Prop := x ≥ a
def q (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, p a x → q x) ∧ (∃ x, q x ∧ ¬p a x) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l107_10711


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l107_10790

theorem consecutive_squares_sum (n : ℕ) (h : 2 * n + 1 = 144169^2) :
  ∃ (a : ℕ), a^2 + (a + 1)^2 = n + 1 :=
sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l107_10790


namespace NUMINAMATH_CALUDE_trio_songs_count_l107_10724

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  hanna : Nat
  mary : Nat
  alina : Nat
  tina : Nat

/-- Calculates the total number of songs sung by the trios -/
def totalSongs (counts : SongCounts) : Nat :=
  (counts.hanna + counts.mary + counts.alina + counts.tina) / 3

/-- Theorem stating the conditions and the result to be proved -/
theorem trio_songs_count (counts : SongCounts) 
  (hanna_most : counts.hanna = 7 ∧ counts.hanna > counts.alina ∧ counts.hanna > counts.tina)
  (mary_least : counts.mary = 4 ∧ counts.mary < counts.alina ∧ counts.mary < counts.tina)
  (alina_tina_between : counts.alina > 4 ∧ counts.alina < 7 ∧ counts.tina > 4 ∧ counts.tina < 7)
  : totalSongs counts = 7 := by
  sorry

end NUMINAMATH_CALUDE_trio_songs_count_l107_10724


namespace NUMINAMATH_CALUDE_unanswered_questions_l107_10735

/-- Represents the scoring system and results for a math contest. -/
structure ContestScoring where
  total_questions : ℕ
  new_correct_points : ℕ
  new_unanswered_points : ℕ
  old_start_points : ℕ
  old_correct_points : ℕ
  old_wrong_points : ℕ
  new_score : ℕ
  old_score : ℕ

/-- Theorem stating that given the contest scoring system and Alice's scores,
    the number of unanswered questions is 8. -/
theorem unanswered_questions (cs : ContestScoring)
  (h1 : cs.total_questions = 30)
  (h2 : cs.new_correct_points = 6)
  (h3 : cs.new_unanswered_points = 3)
  (h4 : cs.old_start_points = 40)
  (h5 : cs.old_correct_points = 5)
  (h6 : cs.old_wrong_points = 2)
  (h7 : cs.new_score = 108)
  (h8 : cs.old_score = 94) :
  ∃ (c w u : ℕ), c + w + u = cs.total_questions ∧
                 cs.new_correct_points * c + cs.new_unanswered_points * u = cs.new_score ∧
                 cs.old_start_points + cs.old_correct_points * c - cs.old_wrong_points * w = cs.old_score ∧
                 u = 8 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_questions_l107_10735


namespace NUMINAMATH_CALUDE_x_value_l107_10788

theorem x_value (x : ℚ) 
  (eq1 : 9 * x^2 + 8 * x - 1 = 0) 
  (eq2 : 27 * x^2 + 65 * x - 8 = 0) : 
  x = 1/9 := by
sorry

end NUMINAMATH_CALUDE_x_value_l107_10788


namespace NUMINAMATH_CALUDE_franklin_valentines_l107_10777

/-- The number of Valentines Mrs. Franklin initially had -/
def initial_valentines : ℕ := 58

/-- The number of Valentines Mrs. Franklin gave to her students -/
def given_valentines : ℕ := 42

/-- The number of Valentines Mrs. Franklin has now -/
def remaining_valentines : ℕ := initial_valentines - given_valentines

/-- Theorem stating that Mrs. Franklin now has 16 Valentines -/
theorem franklin_valentines : remaining_valentines = 16 := by
  sorry

end NUMINAMATH_CALUDE_franklin_valentines_l107_10777


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l107_10755

def C : Set Nat := {37, 39, 42, 43, 47}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ (∀ (m : Nat), m ∈ C → Nat.minFac n ≤ Nat.minFac m) ∧ n = 42 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l107_10755


namespace NUMINAMATH_CALUDE_max_band_members_l107_10718

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (f : BandFormation) (totalMembers : ℕ) : Prop :=
  totalMembers < 100 ∧
  totalMembers = f.rows * f.membersPerRow + 3 ∧
  totalMembers = (f.rows - 3) * (f.membersPerRow + 2)

/-- Theorem stating the maximum number of band members --/
theorem max_band_members :
  ∃ (m : ℕ) (f : BandFormation),
    isValidFormation f m ∧
    ∀ (n : ℕ) (g : BandFormation), isValidFormation g n → n ≤ m :=
  by sorry

end NUMINAMATH_CALUDE_max_band_members_l107_10718


namespace NUMINAMATH_CALUDE_phil_initial_money_l107_10772

/-- The amount of money Phil started with, given his purchases and remaining quarters. -/
theorem phil_initial_money (pizza_cost soda_cost jeans_cost : ℚ)
  (quarters_left : ℕ) (quarter_value : ℚ) :
  pizza_cost = 2.75 →
  soda_cost = 1.50 →
  jeans_cost = 11.50 →
  quarters_left = 97 →
  quarter_value = 0.25 →
  pizza_cost + soda_cost + jeans_cost + (quarters_left : ℚ) * quarter_value = 40 :=
by sorry

end NUMINAMATH_CALUDE_phil_initial_money_l107_10772


namespace NUMINAMATH_CALUDE_tower_heights_count_l107_10742

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of different tower heights achievable -/
def calculateTowerHeights (numBricks : ℕ) (dimensions : BrickDimensions) : ℕ :=
  let minHeight := numBricks * dimensions.length
  let maxAdditionalHeight := numBricks * (dimensions.height - dimensions.length)
  (maxAdditionalHeight / 5 + 1 : ℕ)

/-- Theorem stating the number of different tower heights achievable -/
theorem tower_heights_count (numBricks : ℕ) (dimensions : BrickDimensions) :
  numBricks = 78 →
  dimensions = { length := 3, width := 8, height := 20 } →
  calculateTowerHeights numBricks dimensions = 266 := by
  sorry

end NUMINAMATH_CALUDE_tower_heights_count_l107_10742


namespace NUMINAMATH_CALUDE_log_sum_equals_five_l107_10775

-- Define the logarithm functions
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10
noncomputable def log_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log_sum_equals_five : lg 25 + log_3 27 + lg 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_five_l107_10775


namespace NUMINAMATH_CALUDE_cow_count_is_18_l107_10779

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs for a given AnimalCount -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads for a given AnimalCount -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that the number of cows is 18 given the problem conditions -/
theorem cow_count_is_18 (count : AnimalCount) :
  totalLegs count = 2 * totalHeads count + 36 →
  count.cows = 18 := by
  sorry

#check cow_count_is_18

end NUMINAMATH_CALUDE_cow_count_is_18_l107_10779


namespace NUMINAMATH_CALUDE_min_value_of_f_l107_10763

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f :
  ∃ (min : ℝ), min = -1 / Real.exp 1 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l107_10763


namespace NUMINAMATH_CALUDE_lizard_eye_difference_l107_10797

theorem lizard_eye_difference : ∀ (E W S : ℕ),
  E = 3 →
  W = 3 * E →
  S = 7 * W →
  S + W - E = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_lizard_eye_difference_l107_10797


namespace NUMINAMATH_CALUDE_three_pumps_fill_time_l107_10773

-- Define the pumps and tank
variable (T : ℝ) -- Volume of the tank
variable (X Y Z : ℝ) -- Rates at which pumps X, Y, and Z fill the tank

-- Define the conditions
axiom cond1 : T = 3 * (X + Y)
axiom cond2 : T = 6 * (X + Z)
axiom cond3 : T = 4.5 * (Y + Z)

-- Define the theorem
theorem three_pumps_fill_time : 
  T / (X + Y + Z) = 36 / 13 := by sorry

end NUMINAMATH_CALUDE_three_pumps_fill_time_l107_10773


namespace NUMINAMATH_CALUDE_equilateral_triangle_paths_l107_10730

/-- Represents the number of paths in an equilateral triangle of side length n --/
def f (n : ℕ) : ℕ := n.factorial

/-- 
Theorem: The number of paths from the top triangle to the middle triangle 
in the bottom row of an equilateral triangle with side length n, 
where paths can only move downward and never revisit a triangle, is equal to n!.
-/
theorem equilateral_triangle_paths (n : ℕ) : f n = n.factorial := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_paths_l107_10730


namespace NUMINAMATH_CALUDE_coat_price_l107_10741

theorem coat_price (W : ℝ) (h1 : 2*W - 1.9*W = 4) : 1.9*W = 76 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_l107_10741


namespace NUMINAMATH_CALUDE_grid_toothpicks_l107_10746

/-- Calculates the total number of toothpicks in a grid with internal lines -/
def total_toothpicks (length width spacing : ℕ) : ℕ :=
  let vertical_lines := length / spacing + 1 + length % spacing
  let horizontal_lines := width / spacing + 1 + width % spacing
  vertical_lines * width + horizontal_lines * length

/-- Proves that a grid of 50x40 toothpicks with internal lines every 10 toothpicks uses 4490 toothpicks -/
theorem grid_toothpicks : total_toothpicks 50 40 10 = 4490 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_l107_10746


namespace NUMINAMATH_CALUDE_special_number_in_list_l107_10743

theorem special_number_in_list (l : List ℝ) (n : ℝ) (h1 : l.length = 21) 
  (h2 : n ∈ l) (h3 : n = 4 * ((l.sum - n) / 20)) : 
  n = (1 / 6 : ℝ) * l.sum :=
by
  sorry

end NUMINAMATH_CALUDE_special_number_in_list_l107_10743


namespace NUMINAMATH_CALUDE_log_equation_implies_sum_l107_10752

theorem log_equation_implies_sum (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^4 + (Real.log y / Real.log 3)^4 + 8 = 
       8 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 2 + y^Real.sqrt 2 = 13 := by
sorry

end NUMINAMATH_CALUDE_log_equation_implies_sum_l107_10752


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l107_10758

theorem polynomial_divisibility (a b c d : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d = 5 * k) →
  (∃ ka kb kc kd : ℤ, a = 5 * ka ∧ b = 5 * kb ∧ c = 5 * kc ∧ d = 5 * kd) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l107_10758


namespace NUMINAMATH_CALUDE_origin_and_point_opposite_sides_l107_10702

/-- Determines if two points are on opposite sides of a line -/
def areOnOppositeSides (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (a * x1 + b * y1 + c) * (a * x2 + b * y2 + c) < 0

theorem origin_and_point_opposite_sides :
  areOnOppositeSides 0 0 2 1 (-6) 2 1 := by
  sorry

end NUMINAMATH_CALUDE_origin_and_point_opposite_sides_l107_10702


namespace NUMINAMATH_CALUDE_problem_statement_l107_10789

theorem problem_statement : ∃ x : ℝ, x * (1/2)^2 = 2^3 ∧ x = 32 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l107_10789


namespace NUMINAMATH_CALUDE_sqrt_problem_l107_10715

theorem sqrt_problem (x : ℝ) (h : (Real.sqrt x - 8) / 13 = 6) :
  ⌊(x^2 - 45) / 23⌋ = 2380011 := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_l107_10715


namespace NUMINAMATH_CALUDE_herd_size_l107_10738

/-- Given a herd of cows divided among four sons, prove that the total number of cows is 224 --/
theorem herd_size (herd : ℕ) : herd = 224 :=
  by
  have h1 : (3 : ℚ) / 7 + 1 / 3 + 1 / 6 + (herd - 16 : ℚ) / herd = 1 := by sorry
  have h2 : (herd - 16 : ℚ) / herd = 1 - (3 / 7 + 1 / 3 + 1 / 6) := by sorry
  have h3 : (herd - 16 : ℚ) / herd = 1 / 14 := by sorry
  have h4 : (16 : ℚ) / herd = 1 / 14 := by sorry
  sorry

end NUMINAMATH_CALUDE_herd_size_l107_10738


namespace NUMINAMATH_CALUDE_bedroom_doors_count_l107_10705

theorem bedroom_doors_count : 
  ∀ (outside_doors bedroom_doors : ℕ) 
    (outside_door_cost bedroom_door_cost total_cost : ℚ),
  outside_doors = 2 →
  outside_door_cost = 20 →
  bedroom_door_cost = outside_door_cost / 2 →
  total_cost = 70 →
  outside_doors * outside_door_cost + bedroom_doors * bedroom_door_cost = total_cost →
  bedroom_doors = 3 := by
sorry

end NUMINAMATH_CALUDE_bedroom_doors_count_l107_10705


namespace NUMINAMATH_CALUDE_solve_equation_l107_10793

theorem solve_equation (x : ℝ) (h : Real.sqrt (3 / x + 1) = 5 / 3) : x = 27 / 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l107_10793


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l107_10792

theorem algebraic_expression_value (a b : ℝ) 
  (h1 : a * b = 2) 
  (h2 : a - b = 3) : 
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l107_10792


namespace NUMINAMATH_CALUDE_sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_proof_l107_10708

/-- The sum of the exterior angles of a pentagon is 360 degrees. -/
theorem sum_exterior_angles_pentagon : ℝ :=
  360

/-- A pentagon has 5 sides. -/
def pentagon_sides : ℕ := 5

/-- The sum of the exterior angles of any polygon with n sides. -/
def sum_exterior_angles (n : ℕ) : ℝ := 360

theorem sum_exterior_angles_pentagon_proof :
  sum_exterior_angles pentagon_sides = sum_exterior_angles_pentagon :=
by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_proof_l107_10708


namespace NUMINAMATH_CALUDE_ab_value_l107_10760

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l107_10760


namespace NUMINAMATH_CALUDE_factors_of_60_l107_10717

/-- The number of positive factors of 60 -/
def num_factors_60 : ℕ := sorry

/-- Theorem stating that the number of positive factors of 60 is 12 -/
theorem factors_of_60 : num_factors_60 = 12 := by sorry

end NUMINAMATH_CALUDE_factors_of_60_l107_10717


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l107_10720

def M : Set ℝ := {x | x^2 - x - 2 = 0}
def N : Set ℝ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l107_10720


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l107_10729

theorem square_sum_lower_bound (x y θ : ℝ) 
  (h : (x * Real.cos θ + y * Real.sin θ)^2 + x * Real.sin θ - y * Real.cos θ = 1) : 
  x^2 + y^2 ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l107_10729


namespace NUMINAMATH_CALUDE_original_room_population_l107_10700

theorem original_room_population (initial_population : ℕ) : 
  (initial_population / 3 : ℚ) = 18 → initial_population = 54 :=
by
  intro h
  sorry

#check original_room_population

end NUMINAMATH_CALUDE_original_room_population_l107_10700


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l107_10751

theorem arithmetic_sequence_length :
  ∀ (a₁ aₙ d n : ℤ),
    a₁ = -3 →
    aₙ = 45 →
    d = 4 →
    aₙ = a₁ + (n - 1) * d →
    n = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l107_10751


namespace NUMINAMATH_CALUDE_simplify_fraction_l107_10796

theorem simplify_fraction : 
  (5^1004)^2 - (5^1002)^2 / (5^1003)^2 - (5^1001)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l107_10796


namespace NUMINAMATH_CALUDE_ellipse_k_range_l107_10750

/-- The range of k for an ellipse with equation x²/(3-k) + y²/(5+k) = 1 and foci on the y-axis -/
theorem ellipse_k_range :
  ∀ k : ℝ,
  (∀ x y : ℝ, x^2 / (3 - k) + y^2 / (5 + k) = 1) →
  (5 + k > 3 - k) →
  (3 - k > 0) →
  (5 + k > 0) →
  -1 < k ∧ k < 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l107_10750


namespace NUMINAMATH_CALUDE_colors_in_box_is_seven_l107_10728

/-- The number of colors in each color box, given the total number of pencils and people who bought a color box. -/
def colors_per_box (total_pencils : ℕ) (total_people : ℕ) : ℕ :=
  total_pencils / total_people

/-- Theorem stating that the number of colors in each color box is 7, given the problem conditions. -/
theorem colors_in_box_is_seven : 
  let total_people : ℕ := 6  -- Chloe and 5 friends
  let total_pencils : ℕ := 42
  colors_per_box total_pencils total_people = 7 := by
  sorry

#eval colors_per_box 42 6  -- This should output 7

end NUMINAMATH_CALUDE_colors_in_box_is_seven_l107_10728


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l107_10769

/-- A function satisfying the given functional equation for all real numbers -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + f (f y)) = 2 * x + f (f y) - f (f x)

/-- The theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l107_10769


namespace NUMINAMATH_CALUDE_value_of_expression_l107_10732

theorem value_of_expression (x : ℝ) (h : x = 3) : 5 - 2 * x^2 = -13 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l107_10732


namespace NUMINAMATH_CALUDE_sequence_sum_inequality_l107_10707

theorem sequence_sum_inequality (a : ℕ → ℝ) (S : ℕ → ℝ) (x : ℝ) :
  a 1 = 1 →
  (∀ n, 2 * a (n + 1) = a n) →
  (∀ n : ℕ, ∀ t ∈ Set.Icc (-1 : ℝ) 1, x^2 + t*x + 1 > S n) →
  x ∈ Set.Iic (((-1:ℝ) - Real.sqrt 5) / 2) ∪ Set.Ici ((1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_inequality_l107_10707


namespace NUMINAMATH_CALUDE_drawer_pull_cost_l107_10722

/-- Given the conditions of Amanda's kitchen upgrade, prove the cost of each drawer pull. -/
theorem drawer_pull_cost (num_knobs : ℕ) (cost_per_knob : ℚ) (num_pulls : ℕ) (total_cost : ℚ) :
  num_knobs = 18 →
  cost_per_knob = 5/2 →
  num_pulls = 8 →
  total_cost = 77 →
  (total_cost - num_knobs * cost_per_knob) / num_pulls = 4 := by
  sorry

end NUMINAMATH_CALUDE_drawer_pull_cost_l107_10722


namespace NUMINAMATH_CALUDE_product_of_fractions_l107_10799

theorem product_of_fractions : 
  (8 / 4) * (14 / 7) * (20 / 10) * (25 / 50) * (9 / 18) * (12 / 6) * (21 / 42) * (16 / 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l107_10799


namespace NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l107_10726

/-- Given 100 pounds of cucumbers initially composed of 99% water by weight,
    when the water composition changes to 98% by weight due to evaporation,
    the new total weight of the cucumbers is 50 pounds. -/
theorem cucumber_weight_after_evaporation
  (initial_weight : ℝ)
  (initial_water_percentage : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_weight = 100)
  (h2 : initial_water_percentage = 0.99)
  (h3 : final_water_percentage = 0.98)
  : ∃ (final_weight : ℝ), final_weight = 50 ∧
    (1 - initial_water_percentage) * initial_weight =
    (1 - final_water_percentage) * final_weight :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l107_10726


namespace NUMINAMATH_CALUDE_travel_time_is_50_minutes_l107_10782

/-- Represents a tram system with stations A and B -/
structure TramSystem where
  departure_interval : ℕ  -- Interval between tram departures from A in minutes
  journey_time : ℕ        -- Time for a tram to travel from A to B in minutes

/-- Represents a person cycling from B to A -/
structure Cyclist where
  trams_encountered : ℕ   -- Number of trams encountered during the journey

/-- Calculates the time taken for the cyclist to travel from B to A -/
def travel_time (system : TramSystem) (cyclist : Cyclist) : ℕ :=
  cyclist.trams_encountered * system.departure_interval

/-- Theorem stating the travel time for the given scenario -/
theorem travel_time_is_50_minutes 
  (system : TramSystem) 
  (cyclist : Cyclist) 
  (h1 : system.departure_interval = 5)
  (h2 : system.journey_time = 15)
  (h3 : cyclist.trams_encountered = 10) :
  travel_time system cyclist = 50 := by
  sorry

#eval travel_time ⟨5, 15⟩ ⟨10⟩

end NUMINAMATH_CALUDE_travel_time_is_50_minutes_l107_10782


namespace NUMINAMATH_CALUDE_club_members_count_l107_10727

theorem club_members_count (n : ℕ) (h : n > 2) :
  (2 : ℚ) / ((n : ℚ) - 1) = (1 : ℚ) / 5 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l107_10727


namespace NUMINAMATH_CALUDE_marks_remaining_money_l107_10759

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial_amount : ℕ) (num_items : ℕ) (item_cost : ℕ) : ℕ :=
  initial_amount - num_items * item_cost

/-- Proves that Mark has $35 left after buying books -/
theorem marks_remaining_money :
  remaining_money 85 10 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_marks_remaining_money_l107_10759


namespace NUMINAMATH_CALUDE_problem_statement_l107_10768

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : Real.exp a * (1 - Real.log b) = 1) :
  (1 < b ∧ b < Real.exp 1) ∧
  (b - a > 1) ∧
  (a > Real.log b) ∧
  (Real.exp a - Real.log b > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l107_10768


namespace NUMINAMATH_CALUDE_total_vessels_l107_10719

theorem total_vessels (x y z w : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hxy : x < y) (hyz : y < z) (hzw : z < w) :
  let cruise_ships := x
  let cargo_ships := y * x
  let sailboats := y * x + z
  let fishing_boats := (y * x + z) / w
  cruise_ships + cargo_ships + sailboats + fishing_boats = x * (2 * y + 1) + z * (1 + 1 / w) :=
by sorry

end NUMINAMATH_CALUDE_total_vessels_l107_10719


namespace NUMINAMATH_CALUDE_cars_lifted_is_six_l107_10794

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := 2 * people_per_car

/-- The number of cars being lifted -/
def cars_lifted : ℕ := 6

/-- The number of trucks being lifted -/
def trucks_lifted : ℕ := 3

/-- Theorem stating that the number of cars being lifted is 6 -/
theorem cars_lifted_is_six : cars_lifted = 6 := by sorry

end NUMINAMATH_CALUDE_cars_lifted_is_six_l107_10794


namespace NUMINAMATH_CALUDE_negation_existential_statement_l107_10786

theorem negation_existential_statement :
  ¬(∃ (x : ℝ), x^2 - x + 2 > 0) ≠ (∀ (x : ℝ), x^2 - x + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_existential_statement_l107_10786


namespace NUMINAMATH_CALUDE_reseating_women_problem_l107_10712

/-- Represents the number of ways n women can be reseated under the given conditions --/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 3 => T (n + 2) + T (n + 1) + if n = 0 then 1 else T n

/-- The problem statement --/
theorem reseating_women_problem :
  T 15 = 987 := by
  sorry

end NUMINAMATH_CALUDE_reseating_women_problem_l107_10712


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_measure_l107_10731

/-- In a right triangle where the ratio of the measures of the acute angles is 7:2,
    the measure of the smaller angle is 20°. -/
theorem right_triangle_acute_angle_measure (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧  -- Angles are positive
  α + β + 90 = 180 ∧  -- Sum of angles in a triangle is 180°
  α / β = 7 / 2 ∧  -- Ratio of acute angles
  α > β  -- α is the larger acute angle
  → β = 20 := by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_measure_l107_10731


namespace NUMINAMATH_CALUDE_hcf_of_two_numbers_l107_10714

theorem hcf_of_two_numbers (a b : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (a ≤ b) → 
  (b = 1071) → 
  (∃ k : ℕ, Nat.lcm a b = k * 11 * 17) → 
  Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_two_numbers_l107_10714


namespace NUMINAMATH_CALUDE_mikes_investment_interest_l107_10791

/-- Calculates the total interest earned from a two-part investment --/
def total_interest (total_investment : ℚ) (amount_at_lower_rate : ℚ) (lower_rate : ℚ) (higher_rate : ℚ) : ℚ :=
  let amount_at_higher_rate := total_investment - amount_at_lower_rate
  let interest_lower := amount_at_lower_rate * lower_rate
  let interest_higher := amount_at_higher_rate * higher_rate
  interest_lower + interest_higher

/-- Theorem stating that Mike's investment yields $624 in interest --/
theorem mikes_investment_interest :
  total_interest 6000 1800 (9/100) (11/100) = 624 := by
  sorry

end NUMINAMATH_CALUDE_mikes_investment_interest_l107_10791


namespace NUMINAMATH_CALUDE_max_cables_cut_specific_case_l107_10748

/-- Represents a computer network -/
structure ComputerNetwork where
  total_computers : ℕ
  initial_cables : ℕ
  initial_clusters : ℕ
  final_clusters : ℕ

/-- Calculates the maximum number of cables that can be cut in a computer network -/
def max_cables_cut (network : ComputerNetwork) : ℕ :=
  network.initial_cables - (network.total_computers - network.final_clusters)

/-- Theorem stating the maximum number of cables that can be cut in the given scenario -/
theorem max_cables_cut_specific_case :
  let network := ComputerNetwork.mk 200 345 1 8
  max_cables_cut network = 153 := by
  sorry

#eval max_cables_cut (ComputerNetwork.mk 200 345 1 8)

end NUMINAMATH_CALUDE_max_cables_cut_specific_case_l107_10748


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l107_10706

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of enchanted crystals available. -/
def num_crystals : ℕ := 6

/-- The number of herbs incompatible with one specific crystal. -/
def incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
by sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l107_10706


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l107_10740

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (marks_per_correct : ℕ) 
  (marks_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 50)
  (h3 : marks_per_correct = 3)
  (h4 : marks_per_incorrect = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums ≤ total_sums ∧
    (marks_per_correct : ℤ) * correct_sums - 
    (marks_per_incorrect : ℤ) * (total_sums - correct_sums) = total_marks ∧
    correct_sums = 22 := by
  sorry

#check sandy_correct_sums

end NUMINAMATH_CALUDE_sandy_correct_sums_l107_10740


namespace NUMINAMATH_CALUDE_conic_equation_not_parabola_l107_10762

/-- Represents a conic section equation of the form mx² + ny² = 1 -/
structure ConicEquation where
  m : ℝ
  n : ℝ

/-- Defines the possible types of conic sections -/
inductive ConicType
  | Circle
  | Ellipse
  | Hyperbola
  | Parabola

/-- States that a conic equation cannot represent a parabola -/
theorem conic_equation_not_parabola (eq : ConicEquation) : 
  ∃ (t : ConicType), t ≠ ConicType.Parabola ∧ 
  (∀ (x y : ℝ), eq.m * x^2 + eq.n * y^2 = 1 → 
    ∃ (a b c d e f : ℝ), a * x^2 + b * y^2 + c * x * y + d * x + e * y + f = 0) :=
sorry

end NUMINAMATH_CALUDE_conic_equation_not_parabola_l107_10762


namespace NUMINAMATH_CALUDE_circle_number_placement_l107_10723

theorem circle_number_placement :
  ∃ (a₁ b₁ c₁ d₁ e₁ a₂ b₂ c₂ d₂ e₂ : ℕ),
    (1 ≤ a₁ ∧ a₁ ≤ 9) ∧ (1 ≤ b₁ ∧ b₁ ≤ 9) ∧ (1 ≤ c₁ ∧ c₁ ≤ 9) ∧ (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ e₁ ∧ e₁ ≤ 9) ∧
    (1 ≤ a₂ ∧ a₂ ≤ 9) ∧ (1 ≤ b₂ ∧ b₂ ≤ 9) ∧ (1 ≤ c₂ ∧ c₂ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (1 ≤ e₂ ∧ e₂ ≤ 9) ∧
    b₁ - d₁ = 2 ∧ d₁ - a₁ = 3 ∧ a₁ - c₁ = 1 ∧
    b₂ - d₂ = 2 ∧ d₂ - a₂ = 3 ∧ a₂ - c₂ = 1 ∧
    a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ a₁ ≠ e₁ ∧
    b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ b₁ ≠ e₁ ∧
    c₁ ≠ d₁ ∧ c₁ ≠ e₁ ∧
    d₁ ≠ e₁ ∧
    a₂ ≠ b₂ ∧ a₂ ≠ c₂ ∧ a₂ ≠ d₂ ∧ a₂ ≠ e₂ ∧
    b₂ ≠ c₂ ∧ b₂ ≠ d₂ ∧ b₂ ≠ e₂ ∧
    c₂ ≠ d₂ ∧ c₂ ≠ e₂ ∧
    d₂ ≠ e₂ ∧
    (a₁ ≠ a₂ ∨ b₁ ≠ b₂ ∨ c₁ ≠ c₂ ∨ d₁ ≠ d₂ ∨ e₁ ≠ e₂) :=
by sorry

end NUMINAMATH_CALUDE_circle_number_placement_l107_10723


namespace NUMINAMATH_CALUDE_shoe_probability_l107_10757

theorem shoe_probability (total_pairs : ℕ) (black_pairs brown_pairs gray_pairs : ℕ)
  (h1 : total_pairs = black_pairs + brown_pairs + gray_pairs)
  (h2 : total_pairs = 15)
  (h3 : black_pairs = 8)
  (h4 : brown_pairs = 4)
  (h5 : gray_pairs = 3) :
  let total_shoes := 2 * total_pairs
  let prob_black := (2 * black_pairs / total_shoes) * (black_pairs / (total_shoes - 1))
  let prob_brown := (2 * brown_pairs / total_shoes) * (brown_pairs / (total_shoes - 1))
  let prob_gray := (2 * gray_pairs / total_shoes) * (gray_pairs / (total_shoes - 1))
  prob_black + prob_brown + prob_gray = 89 / 435 :=
by sorry

end NUMINAMATH_CALUDE_shoe_probability_l107_10757


namespace NUMINAMATH_CALUDE_japanese_students_fraction_l107_10784

theorem japanese_students_fraction (J : ℚ) (h1 : J > 0) : 
  let S := 3 * J
  let seniors_studying := (1/3) * S
  let juniors_studying := (3/4) * J
  let total_students := S + J
  (seniors_studying + juniors_studying) / total_students = 7/16 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_fraction_l107_10784


namespace NUMINAMATH_CALUDE_triangle_area_is_one_l107_10770

-- Define the complex number z on the unit circle
def z : ℂ :=
  sorry

-- Define the condition |z| = 1
axiom z_on_unit_circle : Complex.abs z = 1

-- Define the vertices of the triangle
def vertex1 : ℂ := z
def vertex2 : ℂ := z^2
def vertex3 : ℂ := z + z^2

-- Define the function to calculate the area of a triangle given three complex points
def triangle_area (a b c : ℂ) : ℝ :=
  sorry

-- State the theorem
theorem triangle_area_is_one :
  triangle_area vertex1 vertex2 vertex3 = 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_is_one_l107_10770


namespace NUMINAMATH_CALUDE_solve_equation_l107_10725

theorem solve_equation (y : ℝ) : (45 / 75 = Real.sqrt (3 * y / 75)) → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l107_10725


namespace NUMINAMATH_CALUDE_harmonic_mean_of_2_3_6_l107_10764

theorem harmonic_mean_of_2_3_6 : 
  3 = 3 / (1 / 2 + 1 / 3 + 1 / 6) := by sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_2_3_6_l107_10764


namespace NUMINAMATH_CALUDE_staircase_arrangement_7_steps_l107_10704

/-- The number of ways 3 people can stand on a staircase with n steps,
    where each step can accommodate at most 2 people and
    the positions of people on the same step are not distinguished. -/
def staircase_arrangements (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The number of ways 3 people can stand on a 7-step staircase is 336,
    given that each step can accommodate at most 2 people and
    the positions of people on the same step are not distinguished. -/
theorem staircase_arrangement_7_steps :
  staircase_arrangements 7 = 336 := by
  sorry

end NUMINAMATH_CALUDE_staircase_arrangement_7_steps_l107_10704


namespace NUMINAMATH_CALUDE_exists_circle_with_n_grid_points_l107_10780

/-- A grid point is a point with integer coordinates -/
def GridPoint : Type := ℤ × ℤ

/-- A circle is defined by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Count the number of grid points within a circle -/
def countGridPointsInCircle (c : Circle) : ℕ :=
  sorry

/-- Main theorem: For any positive integer n, there exists a circle with exactly n grid points -/
theorem exists_circle_with_n_grid_points (n : ℕ) (hn : n > 0) :
  ∃ (c : Circle), countGridPointsInCircle c = n :=
sorry

end NUMINAMATH_CALUDE_exists_circle_with_n_grid_points_l107_10780


namespace NUMINAMATH_CALUDE_stating_count_quadrilaterals_correct_l107_10713

/-- 
For a convex n-gon, count_quadrilaterals n returns the number of ways to choose 
four vertices that form a quadrilateral with sides that are diagonals of the n-gon.
-/
def count_quadrilaterals (n : ℕ) : ℕ := 
  n / 4 * Nat.choose (n - 5) 3

/-- 
Theorem stating that count_quadrilaterals correctly counts the number of ways 
to choose four vertices forming a quadrilateral with diagonal sides in an n-gon.
-/
theorem count_quadrilaterals_correct (n : ℕ) : 
  count_quadrilaterals n = n / 4 * Nat.choose (n - 5) 3 := by
  sorry

#eval count_quadrilaterals 10  -- Example evaluation

end NUMINAMATH_CALUDE_stating_count_quadrilaterals_correct_l107_10713


namespace NUMINAMATH_CALUDE_ladder_slide_approx_l107_10754

noncomputable def ladder_slide (ladder_length : Real) (initial_distance : Real) (slip_distance : Real) : Real :=
  let initial_height := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height := initial_height - slip_distance
  let new_distance := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance

theorem ladder_slide_approx :
  ∃ (ε : Real), ε > 0 ∧ ε < 0.1 ∧ 
  |ladder_slide 30 11 5 - 3.7| < ε :=
sorry

end NUMINAMATH_CALUDE_ladder_slide_approx_l107_10754


namespace NUMINAMATH_CALUDE_morning_ribbons_l107_10734

theorem morning_ribbons (initial : ℕ) (afternoon : ℕ) (remaining : ℕ) : 
  initial = 38 → afternoon = 16 → remaining = 8 → initial - afternoon - remaining = 14 := by
  sorry

end NUMINAMATH_CALUDE_morning_ribbons_l107_10734


namespace NUMINAMATH_CALUDE_wrong_quotient_problem_l107_10766

theorem wrong_quotient_problem (dividend : ℕ) (correct_divisor wrong_divisor correct_quotient : ℕ) 
  (h1 : dividend % correct_divisor = 0)
  (h2 : correct_divisor = 21)
  (h3 : wrong_divisor = 12)
  (h4 : correct_quotient = 24)
  (h5 : dividend = correct_divisor * correct_quotient) :
  dividend / wrong_divisor = 42 := by
  sorry

end NUMINAMATH_CALUDE_wrong_quotient_problem_l107_10766


namespace NUMINAMATH_CALUDE_second_child_birth_year_l107_10787

/-- 
Given a couple married in 1980 with two children, one born in 1982 and the other
in an unknown year, if their combined ages equal the years of marriage in 1986,
then the second child was born in 1992.
-/
theorem second_child_birth_year 
  (marriage_year : Nat) 
  (first_child_birth_year : Nat) 
  (second_child_birth_year : Nat) 
  (h1 : marriage_year = 1980)
  (h2 : first_child_birth_year = 1982)
  (h3 : (1986 - first_child_birth_year) + (1986 - second_child_birth_year) + (1986 - marriage_year) = 1986) :
  second_child_birth_year = 1992 := by
sorry

end NUMINAMATH_CALUDE_second_child_birth_year_l107_10787


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l107_10795

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  C = π / 3 →
  b = 8 →
  (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3 →
  -- Definitions
  a > 0 →
  b > 0 →
  c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Theorem statements
  c = 7 ∧ Real.cos (B - C) = 13/14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l107_10795


namespace NUMINAMATH_CALUDE_hundredth_bracket_numbers_l107_10710

def bracket_sequence (n : ℕ) : ℕ := 
  if n % 4 = 1 then 1
  else if n % 4 = 2 then 2
  else if n % 4 = 3 then 3
  else 1

def first_number_in_group (group : ℕ) : ℕ := 2 * group - 1

theorem hundredth_bracket_numbers :
  let group := (100 - 1) / 3 + 1
  let first_num := first_number_in_group group - 2
  bracket_sequence 100 = 2 ∧ first_num = 65 ∧ first_num + 2 = 67 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_bracket_numbers_l107_10710


namespace NUMINAMATH_CALUDE_square_field_side_length_l107_10756

theorem square_field_side_length (area : ℝ) (side : ℝ) : 
  area = 400 → side ^ 2 = area → side = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_field_side_length_l107_10756


namespace NUMINAMATH_CALUDE_matrix_cube_sum_l107_10745

/-- Definition of the matrix N -/
def N (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, c, b],
    ![c, b, a],
    ![b, a, c]]

/-- The theorem statement -/
theorem matrix_cube_sum (a b c : ℂ) :
  (N a b c)^2 = 1 → a * b * c = 1 → a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_cube_sum_l107_10745


namespace NUMINAMATH_CALUDE_airline_services_overlap_l107_10767

theorem airline_services_overlap (wireless_percent : Real) (snacks_percent : Real) 
  (wireless_percent_hyp : wireless_percent = 35) 
  (snacks_percent_hyp : snacks_percent = 70) :
  (max_overlap : Real) → max_overlap ≤ 35 ∧ 
  ∃ (overlap : Real), overlap ≤ max_overlap ∧ 
                      overlap ≤ wireless_percent ∧ 
                      overlap ≤ snacks_percent :=
by sorry

end NUMINAMATH_CALUDE_airline_services_overlap_l107_10767


namespace NUMINAMATH_CALUDE_minimum_orchestra_size_l107_10781

theorem minimum_orchestra_size : ∃ n : ℕ, n > 0 ∧ 
  n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧
  ∀ m : ℕ, m > 0 → m % 9 = 0 → m % 10 = 0 → m % 11 = 0 → m ≥ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_minimum_orchestra_size_l107_10781


namespace NUMINAMATH_CALUDE_inversion_preserves_angle_l107_10701

-- Define a type for geometric objects (circles or lines)
inductive GeometricObject
  | Circle : ℝ → ℝ → ℝ → GeometricObject  -- center_x, center_y, radius
  | Line : ℝ → ℝ → ℝ → GeometricObject    -- a, b, c for ax + by + c = 0

-- Define the inversion transformation
def inversion (center : ℝ × ℝ) (k : ℝ) (obj : GeometricObject) : GeometricObject :=
  sorry

-- Define the angle between two geometric objects
def angle_between (obj1 obj2 : GeometricObject) : ℝ :=
  sorry

-- State the theorem
theorem inversion_preserves_angle (center : ℝ × ℝ) (k : ℝ) (obj1 obj2 : GeometricObject) :
  angle_between obj1 obj2 = angle_between (inversion center k obj1) (inversion center k obj2) :=
  sorry

end NUMINAMATH_CALUDE_inversion_preserves_angle_l107_10701


namespace NUMINAMATH_CALUDE_multiplication_error_correction_l107_10747

theorem multiplication_error_correction (N : ℝ) (x : ℝ) : 
  (((N * x - N / 5) / (N * x)) * 100 = 93.33333333333333) → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_error_correction_l107_10747


namespace NUMINAMATH_CALUDE_election_votes_proof_l107_10744

theorem election_votes_proof (total_votes : ℕ) (second_candidate_votes : ℕ) : 
  -- Given conditions
  total_votes = 27500 ∧ 
  (20000 : ℚ) / total_votes = 8011 / 11000 ∧
  total_votes = 2500 + second_candidate_votes + 20000 →
  -- Conclusion
  second_candidate_votes = 5000 := by
sorry


end NUMINAMATH_CALUDE_election_votes_proof_l107_10744


namespace NUMINAMATH_CALUDE_car_washing_time_l107_10771

theorem car_washing_time (x : ℝ) : 
  x > 0 → 
  x + (1/4) * x = 100 → 
  x = 80 :=
by sorry

end NUMINAMATH_CALUDE_car_washing_time_l107_10771


namespace NUMINAMATH_CALUDE_binomial_square_special_case_l107_10785

theorem binomial_square_special_case (a b : ℝ) : (2*a - 3*b)^2 = 4*a^2 - 12*a*b + 9*b^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_special_case_l107_10785


namespace NUMINAMATH_CALUDE_mildred_spending_l107_10798

def total_given : ℕ := 100
def amount_left : ℕ := 40
def candice_spent : ℕ := 35

theorem mildred_spending :
  total_given - amount_left - candice_spent = 25 :=
by sorry

end NUMINAMATH_CALUDE_mildred_spending_l107_10798


namespace NUMINAMATH_CALUDE_union_equals_reals_implies_a_is_negative_one_l107_10765

-- Define the sets S and P
def S : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 2}
def P (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem union_equals_reals_implies_a_is_negative_one (a : ℝ) :
  S ∪ P a = Set.univ → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_implies_a_is_negative_one_l107_10765
