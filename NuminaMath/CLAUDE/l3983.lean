import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_l3983_398397

theorem smallest_number (A B C : ℚ) (hA : A = 1/2) (hB : B = 9/10) (hC : C = 2/5) :
  C ≤ A ∧ C ≤ B := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3983_398397


namespace NUMINAMATH_CALUDE_min_wins_to_advance_exactly_ten_wins_l3983_398309

def football_advancement (total_matches win_matches loss_matches : ℕ) : Prop :=
  let draw_matches := total_matches - win_matches - loss_matches
  3 * win_matches + draw_matches ≥ 33

theorem min_wins_to_advance :
  ∀ win_matches : ℕ,
    football_advancement 15 win_matches 2 →
    win_matches ≥ 10 :=
by
  sorry

theorem exactly_ten_wins :
  football_advancement 15 10 2 ∧
  ∀ win_matches : ℕ, win_matches < 10 → ¬(football_advancement 15 win_matches 2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_wins_to_advance_exactly_ten_wins_l3983_398309


namespace NUMINAMATH_CALUDE_linda_original_correct_l3983_398343

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := 20

/-- The amount of money Lucy would give to Linda -/
def transfer_amount : ℕ := 5

/-- Linda's original amount of money -/
def linda_original : ℕ := 10

/-- Theorem stating that Linda's original amount is correct -/
theorem linda_original_correct : 
  lucy_original - transfer_amount = linda_original + transfer_amount := by
  sorry

end NUMINAMATH_CALUDE_linda_original_correct_l3983_398343


namespace NUMINAMATH_CALUDE_max_product_xy_l3983_398383

theorem max_product_xy (x y : ℝ) :
  (Real.sqrt (x + y - 1) + x^4 + y^4 - 1/8 ≤ 0) →
  (x * y ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_max_product_xy_l3983_398383


namespace NUMINAMATH_CALUDE_joel_stuffed_animals_l3983_398356

/-- The number of stuffed animals Joel collected -/
def stuffed_animals : ℕ := 18

/-- The number of action figures Joel collected -/
def action_figures : ℕ := 42

/-- The number of board games Joel collected -/
def board_games : ℕ := 2

/-- The number of puzzles Joel collected -/
def puzzles : ℕ := 13

/-- The total number of toys Joel donated -/
def total_toys : ℕ := 108

/-- The number of toys that were Joel's own -/
def joels_toys : ℕ := 22

/-- The number of toys Joel's sister gave him -/
def sisters_toys : ℕ := (joels_toys / 2)

theorem joel_stuffed_animals :
  stuffed_animals + action_figures + board_games + puzzles + sisters_toys + joels_toys = total_toys :=
by sorry

end NUMINAMATH_CALUDE_joel_stuffed_animals_l3983_398356


namespace NUMINAMATH_CALUDE_brick_width_is_10cm_l3983_398357

/-- Proves that the width of a brick is 10 cm given the specified conditions -/
theorem brick_width_is_10cm 
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_length : ℝ) (brick_height : ℝ)
  (num_bricks : ℕ)
  (h_wall_length : wall_length = 29)
  (h_wall_width : wall_width = 2)
  (h_wall_height : wall_height = 0.75)
  (h_brick_length : brick_length = 20)
  (h_brick_height : brick_height = 7.5)
  (h_num_bricks : num_bricks = 29000)
  : ∃ (brick_width : ℝ), 
    wall_length * wall_width * wall_height * 1000000 = 
    num_bricks * brick_length * brick_width * brick_height ∧ 
    brick_width = 10 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_is_10cm_l3983_398357


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3983_398375

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricPoint (p : Point2D) : Point2D :=
  ⟨-p.x, -p.y⟩

theorem symmetric_point_coordinates :
  let p : Point2D := ⟨1, -2⟩
  let q : Point2D := symmetricPoint p
  q.x = -1 ∧ q.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3983_398375


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3983_398331

theorem arccos_one_over_sqrt_two (π : Real) :
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3983_398331


namespace NUMINAMATH_CALUDE_interior_angle_sum_increases_interior_angle_sum_formula_l3983_398315

/-- The sum of interior angles of a polygon with k sides -/
def interior_angle_sum (k : ℕ) : ℝ := (k - 2) * 180

/-- Theorem: The sum of interior angles increases as the number of sides increases -/
theorem interior_angle_sum_increases (k : ℕ) (h : k ≥ 3) :
  interior_angle_sum k < interior_angle_sum (k + 1) := by
  sorry

/-- Theorem: The sum of interior angles of a k-sided polygon is (k-2) * 180° -/
theorem interior_angle_sum_formula (k : ℕ) (h : k ≥ 3) :
  interior_angle_sum k = (k - 2) * 180 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_increases_interior_angle_sum_formula_l3983_398315


namespace NUMINAMATH_CALUDE_divisibility_theorem_l3983_398348

theorem divisibility_theorem (a b n : ℕ) (h : a^n ∣ b) : a^(n+1) ∣ ((a+1)^b - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l3983_398348


namespace NUMINAMATH_CALUDE_bobs_weekly_profit_l3983_398385

/-- Calculates the weekly profit for Bob's muffin business -/
theorem bobs_weekly_profit (muffins_per_day : ℕ) (buy_price : ℚ) (sell_price : ℚ) (days_per_week : ℕ) :
  muffins_per_day = 12 →
  buy_price = 3/4 →
  sell_price = 3/2 →
  days_per_week = 7 →
  (sell_price - buy_price) * muffins_per_day * days_per_week = 63 := by
sorry

#eval (3/2 : ℚ) - (3/4 : ℚ)
#eval ((3/2 : ℚ) - (3/4 : ℚ)) * 12
#eval (((3/2 : ℚ) - (3/4 : ℚ)) * 12) * 7

end NUMINAMATH_CALUDE_bobs_weekly_profit_l3983_398385


namespace NUMINAMATH_CALUDE_log_inequality_l3983_398319

theorem log_inequality (h1 : 4^5 < 7^4) (h2 : 11^4 < 7^5) : 
  Real.log 11 / Real.log 7 < Real.log 243 / Real.log 81 ∧ 
  Real.log 243 / Real.log 81 < Real.log 7 / Real.log 4 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3983_398319


namespace NUMINAMATH_CALUDE_tan_non_intersection_l3983_398388

theorem tan_non_intersection :
  ∀ y : ℝ, ∃ k : ℤ, (2 * (π/8) + π/4) = k * π + π/2 :=
by sorry

end NUMINAMATH_CALUDE_tan_non_intersection_l3983_398388


namespace NUMINAMATH_CALUDE_a_plus_b_value_l3983_398360

-- Define the functions f and h
def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x - 6

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (∀ x, h (f a b x) = 4 * x + 3) → a + b = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l3983_398360


namespace NUMINAMATH_CALUDE_smallest_digit_change_correct_change_l3983_398314

def original_sum : ℕ := 738 + 625 + 841
def incorrect_sum : ℕ := 2104
def correct_sum : ℕ := 2204

def change_digit (n : ℕ) (place : ℕ) (new_digit : ℕ) : ℕ :=
  n - (n / 10^place % 10) * 10^place + new_digit * 10^place

theorem smallest_digit_change :
  ∀ (d : ℕ),
    d < 6 →
    ¬∃ (n : ℕ) (place : ℕ),
      (n = 738 ∨ n = 625 ∨ n = 841) ∧
      change_digit n place d + 
        (if n = 738 then 625 + 841
         else if n = 625 then 738 + 841
         else 738 + 625) = correct_sum :=
by sorry

theorem correct_change :
  change_digit 625 2 5 + 738 + 841 = correct_sum :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_change_correct_change_l3983_398314


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3983_398320

theorem sin_300_degrees : Real.sin (300 * Real.pi / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3983_398320


namespace NUMINAMATH_CALUDE_bus_problem_solution_l3983_398330

def bus_problem (initial : ℕ) 
  (stop1_off stop1_on : ℕ) 
  (stop2_off stop2_on : ℕ) 
  (stop3_off stop3_on : ℕ) : ℕ :=
  initial - stop1_off + stop1_on - stop2_off + stop2_on - stop3_off + stop3_on

theorem bus_problem_solution : 
  bus_problem 10 3 2 1 4 2 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_solution_l3983_398330


namespace NUMINAMATH_CALUDE_jaylen_kristin_bell_pepper_ratio_l3983_398364

/-- Prove that the ratio of Jaylen's bell peppers to Kristin's bell peppers is 2:1 -/
theorem jaylen_kristin_bell_pepper_ratio :
  let jaylen_carrots : ℕ := 5
  let jaylen_cucumbers : ℕ := 2
  let kristin_bell_peppers : ℕ := 2
  let kristin_green_beans : ℕ := 20
  let jaylen_green_beans : ℕ := kristin_green_beans / 2 - 3
  let jaylen_total_vegetables : ℕ := 18
  let jaylen_bell_peppers : ℕ := jaylen_total_vegetables - (jaylen_carrots + jaylen_cucumbers + jaylen_green_beans)
  
  (jaylen_bell_peppers : ℚ) / kristin_bell_peppers = 2 := by
  sorry


end NUMINAMATH_CALUDE_jaylen_kristin_bell_pepper_ratio_l3983_398364


namespace NUMINAMATH_CALUDE_cuboid_edge_lengths_l3983_398396

theorem cuboid_edge_lengths :
  ∀ a b c : ℕ,
  (a * b * c + a * b + b * c + c * a + a + b + c = 2000) →
  ({a, b, c} : Finset ℕ) = {28, 22, 2} := by
sorry

end NUMINAMATH_CALUDE_cuboid_edge_lengths_l3983_398396


namespace NUMINAMATH_CALUDE_sequence_sign_change_l3983_398378

theorem sequence_sign_change (a₀ c : ℝ) (h₁ : a₀ > 0) (h₂ : c > 0) : 
  ∃ (a : ℕ → ℝ), a 0 = a₀ ∧ 
  (∀ n, a (n + 1) = (a n + c) / (1 - a n * c)) ∧
  (∀ n, n < 1990 → a n > 0) ∧
  a 1990 < 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_sign_change_l3983_398378


namespace NUMINAMATH_CALUDE_james_training_sessions_l3983_398365

/-- James' training schedule -/
structure TrainingSchedule where
  hoursPerSession : ℕ
  daysOffPerWeek : ℕ
  totalHoursPerYear : ℕ

/-- Calculate the number of training sessions per day -/
def sessionsPerDay (schedule : TrainingSchedule) : ℚ :=
  let daysPerWeek : ℕ := 7
  let weeksPerYear : ℕ := 52
  let trainingDaysPerYear : ℕ := (daysPerWeek - schedule.daysOffPerWeek) * weeksPerYear
  let hoursPerDay : ℚ := schedule.totalHoursPerYear / trainingDaysPerYear
  hoursPerDay / schedule.hoursPerSession

/-- Theorem: James trains 2 times per day -/
theorem james_training_sessions (james : TrainingSchedule) 
  (h1 : james.hoursPerSession = 4)
  (h2 : james.daysOffPerWeek = 2)
  (h3 : james.totalHoursPerYear = 2080) : 
  sessionsPerDay james = 2 := by
  sorry


end NUMINAMATH_CALUDE_james_training_sessions_l3983_398365


namespace NUMINAMATH_CALUDE_acute_triangle_angles_l3983_398381

-- Define an acute triangle
def is_acute_triangle (a b c : ℝ) : Prop :=
  0 < a ∧ a < 90 ∧
  0 < b ∧ b < 90 ∧
  0 < c ∧ c < 90 ∧
  a + b + c = 180

-- Theorem statement
theorem acute_triangle_angles (a b c : ℝ) :
  is_acute_triangle a b c →
  ∃ (x y z : ℝ), is_acute_triangle x y z ∧ x > 45 ∧ y > 45 ∧ z > 45 :=
by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_angles_l3983_398381


namespace NUMINAMATH_CALUDE_rightmost_two_digits_l3983_398334

theorem rightmost_two_digits : ∃ n : ℕ, (4^127 + 5^129 + 7^131) % 100 = 52 + 100 * n := by
  sorry

end NUMINAMATH_CALUDE_rightmost_two_digits_l3983_398334


namespace NUMINAMATH_CALUDE_probability_is_one_over_930_l3983_398387

/-- Represents a sequence of 40 distinct real numbers -/
def Sequence := { s : Fin 40 → ℝ // Function.Injective s }

/-- The operation that compares and potentially swaps adjacent elements -/
def operation (s : Sequence) : Sequence := sorry

/-- The probability that the 20th element moves to the 30th position after one operation -/
def probability_20_to_30 (s : Sequence) : ℚ := sorry

/-- Theorem stating that the probability is 1/930 -/
theorem probability_is_one_over_930 (s : Sequence) : 
  probability_20_to_30 s = 1 / 930 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_over_930_l3983_398387


namespace NUMINAMATH_CALUDE_seokjin_paper_count_l3983_398312

theorem seokjin_paper_count (jimin_count : ℕ) (difference : ℕ) 
  (h1 : jimin_count = 41)
  (h2 : difference = 1)
  (h3 : jimin_count = seokjin_count + difference) :
  seokjin_count = 40 := by
  sorry

end NUMINAMATH_CALUDE_seokjin_paper_count_l3983_398312


namespace NUMINAMATH_CALUDE_problem_solution_l3983_398377

-- Define the solution set for x(x-2) < 0
def solution_set := {x : ℝ | x * (x - 2) < 0}

-- Define the proposed incorrect solution set
def incorrect_set := {x : ℝ | x < 0 ∨ x > 2}

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Theorem statement
theorem problem_solution :
  (solution_set ≠ incorrect_set) ∧
  (∀ t : Triangle, t.A > t.B ↔ Real.sin t.A > Real.sin t.B) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3983_398377


namespace NUMINAMATH_CALUDE_job_completion_time_l3983_398328

/-- Represents the time to complete a job given initial and final workforce conditions -/
def total_completion_time (n k t : ℕ) : ℝ :=
  t + 4 * (n + k)

/-- Theorem stating the total time to complete the job -/
theorem job_completion_time (n k t : ℕ) :
  (3 / 4 : ℝ) / t = n / total_completion_time n k t ∧
  (1 / 4 : ℝ) / (total_completion_time n k t - t) = (n + k) / (total_completion_time n k t - t) →
  total_completion_time n k t = t + 4 * (n + k) := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l3983_398328


namespace NUMINAMATH_CALUDE_no_integer_solution_x2_plus_y2_eq_3z2_l3983_398359

theorem no_integer_solution_x2_plus_y2_eq_3z2 :
  ∀ (x y z : ℤ), x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_x2_plus_y2_eq_3z2_l3983_398359


namespace NUMINAMATH_CALUDE_more_even_products_l3983_398373

def S : Finset Nat := {1, 2, 3, 4, 5}

def pairs : Finset (Nat × Nat) :=
  S.product S |>.filter (λ (a, b) => a ≤ b)

def products : Finset Nat :=
  pairs.image (λ (a, b) => a * b)

def evenProducts : Finset Nat :=
  products.filter (λ x => x % 2 = 0)

def oddProducts : Finset Nat :=
  products.filter (λ x => x % 2 ≠ 0)

theorem more_even_products :
  Finset.card evenProducts > Finset.card oddProducts :=
by sorry

end NUMINAMATH_CALUDE_more_even_products_l3983_398373


namespace NUMINAMATH_CALUDE_ice_cream_price_l3983_398380

theorem ice_cream_price (game_cost : ℚ) (num_ice_creams : ℕ) (h1 : game_cost = 60) (h2 : num_ice_creams = 24) :
  game_cost / num_ice_creams = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_price_l3983_398380


namespace NUMINAMATH_CALUDE_toy_sales_earnings_difference_l3983_398394

theorem toy_sales_earnings_difference :
  let bert_initial_price : ℝ := 18
  let bert_initial_quantity : ℕ := 10
  let bert_discount_percentage : ℝ := 0.15
  let bert_discounted_quantity : ℕ := 3

  let tory_initial_price : ℝ := 20
  let tory_initial_quantity : ℕ := 15
  let tory_discount_percentage : ℝ := 0.10
  let tory_discounted_quantity : ℕ := 7

  let tax_rate : ℝ := 0.05

  let bert_earnings : ℝ := 
    (bert_initial_price * bert_initial_quantity - 
     bert_discount_percentage * bert_initial_price * bert_discounted_quantity) * 
    (1 + tax_rate)

  let tory_earnings : ℝ := 
    (tory_initial_price * tory_initial_quantity - 
     tory_discount_percentage * tory_initial_price * tory_discounted_quantity) * 
    (1 + tax_rate)

  tory_earnings - bert_earnings = 119.805 :=
by sorry

end NUMINAMATH_CALUDE_toy_sales_earnings_difference_l3983_398394


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3983_398310

/-- Given a quadratic function f(x) = ax^2 - 2x + c where the solution set of f(x) > 0 is {x | x ≠ 1/a},
    this theorem states that the minimum value of f(2) is 0 and when f(2) is minimum,
    the maximum value of m that satisfies f(x) + 4 ≥ m(x-2) for all x > 2 is 2√2. -/
theorem quadratic_function_properties (a c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 - 2*x + c)
  (h2 : ∀ x, f x > 0 ↔ x ≠ 1/a) :
  (∃ (f_min : ℝ → ℝ), (∀ x, f_min x = (1/2) * x^2 - 2*x + 2) ∧ 
   f_min 2 = 0 ∧ 
   (∀ m : ℝ, (∀ x > 2, f_min x + 4 ≥ m * (x - 2)) ↔ m ≤ 2 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3983_398310


namespace NUMINAMATH_CALUDE_zoe_has_16_crayons_l3983_398305

/-- The number of crayons in each student's box -/
structure CrayonBoxes where
  karen : ℕ
  beatrice : ℕ
  gilbert : ℕ
  judah : ℕ
  xavier : ℕ
  yasmine : ℕ
  zoe : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (boxes : CrayonBoxes) : Prop :=
  boxes.karen = 2 * boxes.beatrice ∧
  boxes.beatrice = 2 * boxes.gilbert ∧
  boxes.gilbert = 4 * boxes.judah ∧
  2 * boxes.gilbert = boxes.xavier ∧
  boxes.xavier = boxes.yasmine + 16 ∧
  boxes.yasmine = 3 * boxes.zoe ∧
  boxes.karen = 128

/-- The theorem to be proved -/
theorem zoe_has_16_crayons (boxes : CrayonBoxes) 
  (h : satisfies_conditions boxes) : boxes.zoe = 16 := by
  sorry

end NUMINAMATH_CALUDE_zoe_has_16_crayons_l3983_398305


namespace NUMINAMATH_CALUDE_work_completion_time_l3983_398351

/-- Proves that if A is thrice as fast as B and together they can do a work in 15 days, 
    then A alone can do the work in 20 days. -/
theorem work_completion_time 
  (a b : ℝ) -- Work rates of A and B
  (h1 : a = 3 * b) -- A is thrice as fast as B
  (h2 : (a + b) * 15 = 1) -- Together, A and B can do the work in 15 days
  : a * 20 = 1 := by -- A alone can do the work in 20 days
sorry


end NUMINAMATH_CALUDE_work_completion_time_l3983_398351


namespace NUMINAMATH_CALUDE_root_conditions_imply_sum_l3983_398353

/-- Given two polynomial equations with specific root conditions, prove that 100p + q = 502 -/
theorem root_conditions_imply_sum (p q : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ 
    (∀ z : ℝ, (z + p) * (z + q) * (z + 5) / (z + 2)^2 = 0 ↔ (z = x ∨ z = y)) ∧
    (z = -2 → (z + p) * (z + q) * (z + 5) ≠ 0)) →
  (∃ (u v : ℝ), u ≠ v ∧ 
    (∀ w : ℝ, (w + 2*p) * (w + 2) * (w + 3) / ((w + q) * (w + 5)) = 0 ↔ (w = u ∨ w = v)) ∧
    ((w = -q ∨ w = -5) → (w + 2*p) * (w + 2) * (w + 3) ≠ 0)) →
  100 * p + q = 502 := by
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_sum_l3983_398353


namespace NUMINAMATH_CALUDE_no_solution_lcm_gcd_equation_l3983_398386

theorem no_solution_lcm_gcd_equation : ¬ ∃ (n : ℕ+), Nat.lcm n 120 = Nat.gcd n 120 + 300 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_lcm_gcd_equation_l3983_398386


namespace NUMINAMATH_CALUDE_year_2049_is_jisi_l3983_398369

/-- Represents the Heavenly Stems -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Heavenly Stems and Earthly Branches system -/
structure StemBranchYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

def next_stem (s : HeavenlyStem) : HeavenlyStem := sorry
def next_branch (b : EarthlyBranch) : EarthlyBranch := sorry

def advance_year (y : StemBranchYear) (n : ℕ) : StemBranchYear := sorry

theorem year_2049_is_jisi (year_2017 : StemBranchYear) 
  (h2017 : year_2017 = ⟨HeavenlyStem.Ding, EarthlyBranch.You⟩) :
  advance_year year_2017 32 = ⟨HeavenlyStem.Ji, EarthlyBranch.Si⟩ := by
  sorry

end NUMINAMATH_CALUDE_year_2049_is_jisi_l3983_398369


namespace NUMINAMATH_CALUDE_existence_of_constant_g_l3983_398341

-- Define the necessary types and functions
def Graph : Type := sorry
def circumference (G : Graph) : ℕ := sorry
def chromaticNumber (G : Graph) : ℕ := sorry
def containsSubgraph (G H : Graph) : Prop := sorry
def TK (r : ℕ) : Graph := sorry

-- The main theorem
theorem existence_of_constant_g : 
  ∃ g : ℕ, ∀ (G : Graph) (r : ℕ), 
    circumference G ≥ g → chromaticNumber G ≥ r → containsSubgraph G (TK r) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_constant_g_l3983_398341


namespace NUMINAMATH_CALUDE_travel_statements_correct_l3983_398300

/-- Represents a traveler (cyclist or motorcyclist) --/
structure Traveler where
  startTime : ℝ
  arrivalTime : ℝ
  distanceTraveled : ℝ → ℝ
  speed : ℝ → ℝ

/-- The travel scenario between two towns --/
structure TravelScenario where
  cyclist : Traveler
  motorcyclist : Traveler
  totalDistance : ℝ

/-- Properties of the travel scenario --/
def TravelScenario.properties (scenario : TravelScenario) : Prop :=
  -- The total distance is 80km
  scenario.totalDistance = 80 ∧
  -- The cyclist starts 3 hours before the motorcyclist
  scenario.cyclist.startTime + 3 = scenario.motorcyclist.startTime ∧
  -- The cyclist arrives 1 hour before the motorcyclist
  scenario.cyclist.arrivalTime + 1 = scenario.motorcyclist.arrivalTime ∧
  -- The cyclist's speed pattern (acceleration then constant)
  (∃ t₀ : ℝ, ∀ t, t ≥ scenario.cyclist.startTime → 
    (t ≤ t₀ → scenario.cyclist.speed t < scenario.cyclist.speed (t + 1)) ∧
    (t > t₀ → scenario.cyclist.speed t = scenario.cyclist.speed t₀)) ∧
  -- The motorcyclist's constant speed
  (∀ t₁ t₂, scenario.motorcyclist.speed t₁ = scenario.motorcyclist.speed t₂) ∧
  -- The catch-up time
  (∃ t : ℝ, t = scenario.motorcyclist.startTime + 1.5 ∧
    scenario.cyclist.distanceTraveled t = scenario.motorcyclist.distanceTraveled t)

/-- The main theorem stating the correctness of all statements --/
theorem travel_statements_correct (scenario : TravelScenario) 
  (h : scenario.properties) : 
  -- Statement 1: Timing difference
  (scenario.cyclist.startTime + 3 = scenario.motorcyclist.startTime ∧
   scenario.cyclist.arrivalTime + 1 = scenario.motorcyclist.arrivalTime) ∧
  -- Statement 2: Speed patterns
  (∃ t₀ : ℝ, ∀ t, t ≥ scenario.cyclist.startTime → 
    (t ≤ t₀ → scenario.cyclist.speed t < scenario.cyclist.speed (t + 1)) ∧
    (t > t₀ → scenario.cyclist.speed t = scenario.cyclist.speed t₀)) ∧
  (∀ t₁ t₂, scenario.motorcyclist.speed t₁ = scenario.motorcyclist.speed t₂) ∧
  -- Statement 3: Catch-up time
  (∃ t : ℝ, t = scenario.motorcyclist.startTime + 1.5 ∧
    scenario.cyclist.distanceTraveled t = scenario.motorcyclist.distanceTraveled t) :=
by sorry

end NUMINAMATH_CALUDE_travel_statements_correct_l3983_398300


namespace NUMINAMATH_CALUDE_four_digit_perfect_squares_l3983_398303

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

theorem four_digit_perfect_squares :
  (∀ n : ℕ, is_four_digit n ∧ all_even_digits n ∧ ∃ k, n = k^2 ↔ 
    n = 4624 ∨ n = 6084 ∨ n = 6400 ∨ n = 8464) ∧
  (¬ ∃ n : ℕ, is_four_digit n ∧ all_odd_digits n ∧ ∃ k, n = k^2) :=
sorry

end NUMINAMATH_CALUDE_four_digit_perfect_squares_l3983_398303


namespace NUMINAMATH_CALUDE_shopping_trip_expenses_l3983_398390

theorem shopping_trip_expenses (T : ℝ) (h_positive : T > 0) : 
  let clothing_percent : ℝ := 0.50
  let other_percent : ℝ := 0.30
  let clothing_tax : ℝ := 0.05
  let other_tax : ℝ := 0.10
  let total_tax_percent : ℝ := 0.055
  let food_percent : ℝ := 1 - clothing_percent - other_percent

  clothing_tax * clothing_percent * T + other_tax * other_percent * T = total_tax_percent * T →
  food_percent = 0.20 := by
sorry

end NUMINAMATH_CALUDE_shopping_trip_expenses_l3983_398390


namespace NUMINAMATH_CALUDE_pages_left_to_read_l3983_398368

theorem pages_left_to_read (total_pages read_pages : ℕ) 
  (h1 : total_pages = 563)
  (h2 : read_pages = 147) :
  total_pages - read_pages = 416 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l3983_398368


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l3983_398371

theorem max_value_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 5 + 6 * y * z * Real.sqrt 3 + 9 * z * x ≤ Real.sqrt 5 + 3 * Real.sqrt 3 + 9/2 :=
by sorry

theorem max_value_achievable : 
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^2 + y^2 + z^2 = 1 ∧
  3 * x * y * Real.sqrt 5 + 6 * y * z * Real.sqrt 3 + 9 * z * x = Real.sqrt 5 + 3 * Real.sqrt 3 + 9/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l3983_398371


namespace NUMINAMATH_CALUDE_even_function_properties_l3983_398347

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem even_function_properties (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_decreasing : is_decreasing_on f (-5) (-2))
  (h_max : ∀ x, -5 ≤ x ∧ x ≤ -2 → f x ≤ 7) :
  is_increasing_on f 2 5 ∧ ∀ x, 2 ≤ x ∧ x ≤ 5 → f x ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_even_function_properties_l3983_398347


namespace NUMINAMATH_CALUDE_circle_ratio_after_radius_increase_l3983_398333

theorem circle_ratio_after_radius_increase (r : ℝ) : 
  let new_radius : ℝ := r + 2
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_radius_increase_l3983_398333


namespace NUMINAMATH_CALUDE_logarithm_inequality_l3983_398362

theorem logarithm_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  Real.log (Real.sqrt (a * b)) = (Real.log a + Real.log b) / 2 ∧ 
  Real.log (Real.sqrt (a * b)) < Real.log ((a + b) / 2) ∧
  Real.log ((a + b) / 2) < Real.log ((a^2 + b^2) / 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l3983_398362


namespace NUMINAMATH_CALUDE_average_weight_abc_l3983_398398

/-- Given the weights of three individuals a, b, and c, prove that their average weight is 45 kg -/
theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 40 →   -- average weight of a and b is 40 kg
  (b + c) / 2 = 47 →   -- average weight of b and c is 47 kg
  b = 39 →             -- weight of b is 39 kg
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l3983_398398


namespace NUMINAMATH_CALUDE_strawberry_weight_sum_l3983_398324

/-- The weight of Marco's strawberries in pounds -/
def marco_strawberries : ℕ := 3

/-- The weight of Marco's dad's strawberries in pounds -/
def dad_strawberries : ℕ := 17

/-- The total weight of Marco's and his dad's strawberries -/
def total_strawberries : ℕ := marco_strawberries + dad_strawberries

theorem strawberry_weight_sum :
  total_strawberries = 20 := by sorry

end NUMINAMATH_CALUDE_strawberry_weight_sum_l3983_398324


namespace NUMINAMATH_CALUDE_angle_z_is_100_l3983_398302

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.X > 0 ∧ t.Y > 0 ∧ t.Z > 0 ∧ t.X + t.Y + t.Z = 180

-- Define the given conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.X + t.Y = 80 ∧ t.X = 2 * t.Y

-- Theorem statement
theorem angle_z_is_100 (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_conditions t) : 
  t.Z = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_z_is_100_l3983_398302


namespace NUMINAMATH_CALUDE_total_turnips_proof_l3983_398311

/-- The number of turnips grown by Sally -/
def sally_turnips : ℕ := 113

/-- The number of turnips grown by Mary -/
def mary_turnips : ℕ := 129

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := sally_turnips + mary_turnips

theorem total_turnips_proof : total_turnips = 242 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_proof_l3983_398311


namespace NUMINAMATH_CALUDE_square_difference_l3983_398382

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3983_398382


namespace NUMINAMATH_CALUDE_maria_assembly_time_l3983_398393

/-- Represents the time taken to assemble furniture items -/
structure AssemblyTime where
  chairs : Nat
  tables : Nat
  bookshelf : Nat
  tv_stand : Nat

/-- Calculates the total assembly time for all furniture items -/
def total_assembly_time (time : AssemblyTime) (num_chairs num_tables : Nat) : Nat :=
  num_chairs * time.chairs + num_tables * time.tables + time.bookshelf + time.tv_stand

/-- Theorem: The total assembly time for Maria's furniture is 100 minutes -/
theorem maria_assembly_time :
  let time : AssemblyTime := { chairs := 8, tables := 12, bookshelf := 25, tv_stand := 35 }
  total_assembly_time time 2 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_maria_assembly_time_l3983_398393


namespace NUMINAMATH_CALUDE_center_C_range_l3983_398325

-- Define the points and line
def A : ℝ × ℝ := (0, 3)
def l (x : ℝ) : ℝ := 2 * x - 4

-- Define circle C
def C (a : ℝ) : ℝ × ℝ := (a, l a)
def radius_C : ℝ := 1

-- Define moving point M
def M : ℝ × ℝ → Prop := λ (x, y) => (x^2 + (y - 3)^2) = 4 * (x^2 + y^2)

-- Define the intersection condition
def intersects (C : ℝ × ℝ) (M : ℝ × ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), M (x, y) ∧ (x - C.1)^2 + (y - C.2)^2 = radius_C^2

-- Theorem statement
theorem center_C_range (a : ℝ) :
  (C a).2 = l (C a).1 →  -- Center of C lies on line l
  intersects (C a) M →   -- M intersects with C
  0 ≤ a ∧ a ≤ 12/5 :=
by sorry

end NUMINAMATH_CALUDE_center_C_range_l3983_398325


namespace NUMINAMATH_CALUDE_christophers_gabrielas_age_ratio_l3983_398332

/-- Proves that given Christopher is 2 times as old as Gabriela and Christopher is 24 years old, 
    the ratio of Christopher's age to Gabriela's age nine years ago is 5:1. -/
theorem christophers_gabrielas_age_ratio : 
  ∀ (christopher_age gabriela_age : ℕ),
    christopher_age = 2 * gabriela_age →
    christopher_age = 24 →
    (christopher_age - 9) / (gabriela_age - 9) = 5 := by
  sorry

end NUMINAMATH_CALUDE_christophers_gabrielas_age_ratio_l3983_398332


namespace NUMINAMATH_CALUDE_ratio_approximation_l3983_398355

/-- The set of numbers from 1 to 10^13 in powers of 10 -/
def powerSet : Set ℕ := {n | ∃ k : ℕ, k ≤ 13 ∧ n = 10^k}

/-- The largest element in the set -/
def largestElement : ℕ := 10^13

/-- The sum of all elements in the set except the largest -/
def sumOfOthers : ℕ := (largestElement - 1) / 9

/-- The ratio of the largest element to the sum of others -/
def ratio : ℚ := largestElement / sumOfOthers

theorem ratio_approximation : ∃ ε > 0, abs (ratio - 9) < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_approximation_l3983_398355


namespace NUMINAMATH_CALUDE_existence_of_rationals_l3983_398363

theorem existence_of_rationals (f : ℚ → ℤ) : ∃ a b : ℚ, (f a + f b) / 2 ≤ f ((a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_rationals_l3983_398363


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3983_398384

-- Problem 1
theorem problem_1 (x y : ℝ) : (-4 * x * y^3) * (-2 * x)^2 = -16 * x^3 * y^3 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (3*x - 2) * (2*x - 3) - (x - 1) * (6*x + 5) = -12*x + 11 := by sorry

-- Problem 3
theorem problem_3 : (3 * (10^2)) * (5 * (10^5)) = (1.5 : ℝ) * (10^8) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3983_398384


namespace NUMINAMATH_CALUDE_birth_ticket_cost_l3983_398391

/-- The cost of a ticket to Mars at a given time -/
def ticket_cost (years_since_birth : ℕ) : ℚ := sorry

/-- The cost is halved every 10 years -/
axiom cost_halves (y : ℕ) : ticket_cost (y + 10) = ticket_cost y / 2

/-- When Matty is 30, a ticket costs $125,000 -/
axiom cost_at_30 : ticket_cost 30 = 125000

/-- The cost of a ticket to Mars when Matty was born was $1,000,000 -/
theorem birth_ticket_cost : ticket_cost 0 = 1000000 := by sorry

end NUMINAMATH_CALUDE_birth_ticket_cost_l3983_398391


namespace NUMINAMATH_CALUDE_total_revenue_is_4586_80_l3983_398379

structure PhoneModel where
  name : String
  initialInventory : ℕ
  price : ℚ
  discountRate : ℚ
  taxRate : ℚ
  damaged : ℕ
  finalInventory : ℕ

def calculateRevenue (model : PhoneModel) : ℚ :=
  let discountedPrice := model.price * (1 - model.discountRate)
  let priceAfterTax := discountedPrice * (1 + model.taxRate)
  let soldUnits := model.initialInventory - model.finalInventory - model.damaged
  soldUnits * priceAfterTax

def totalRevenue (models : List PhoneModel) : ℚ :=
  models.map calculateRevenue |>.sum

def phoneModels : List PhoneModel := [
  { name := "Samsung Galaxy S20", initialInventory := 14, price := 800, discountRate := 0.1, taxRate := 0.12, damaged := 2, finalInventory := 10 },
  { name := "iPhone 12", initialInventory := 8, price := 1000, discountRate := 0.15, taxRate := 0.1, damaged := 1, finalInventory := 5 },
  { name := "Google Pixel 5", initialInventory := 7, price := 700, discountRate := 0.05, taxRate := 0.08, damaged := 0, finalInventory := 8 },
  { name := "OnePlus 8T", initialInventory := 6, price := 600, discountRate := 0.2, taxRate := 0.15, damaged := 1, finalInventory := 3 }
]

theorem total_revenue_is_4586_80 :
  totalRevenue phoneModels = 4586.8 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_4586_80_l3983_398379


namespace NUMINAMATH_CALUDE_touch_point_theorem_l3983_398389

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  -- The length of the hypotenuse
  hypotenuse : ℝ
  -- The radius of the inscribed circle
  radius : ℝ
  -- Assumption that the hypotenuse is positive
  hypotenuse_pos : hypotenuse > 0
  -- Assumption that the radius is positive
  radius_pos : radius > 0

/-- The length from one vertex to where the circle touches the hypotenuse -/
def touchPoint (t : RightTriangleWithInscribedCircle) : Set ℝ :=
  {x : ℝ | x = t.hypotenuse / 2 - t.radius ∨ x = t.hypotenuse / 2 + t.radius}

theorem touch_point_theorem (t : RightTriangleWithInscribedCircle) 
    (h1 : t.hypotenuse = 10) (h2 : t.radius = 2) : 
    touchPoint t = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_touch_point_theorem_l3983_398389


namespace NUMINAMATH_CALUDE_a_neg_one_necessary_not_sufficient_l3983_398367

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + (a + 2) * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 2 = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ (x y : ℝ), l₁ a x y ↔ l₂ a x y

-- State the theorem
theorem a_neg_one_necessary_not_sufficient :
  (∀ a : ℝ, parallel a → a = -1) ∧ 
  ¬(∀ a : ℝ, a = -1 → parallel a) :=
sorry

end NUMINAMATH_CALUDE_a_neg_one_necessary_not_sufficient_l3983_398367


namespace NUMINAMATH_CALUDE_cheese_calories_per_serving_l3983_398316

/-- Represents the number of calories in a serving of cheese -/
def calories_per_serving (total_servings : ℕ) (eaten_servings : ℕ) (remaining_calories : ℕ) : ℕ :=
  remaining_calories / (total_servings - eaten_servings)

/-- Theorem stating that the number of calories in a serving of cheese is 110 -/
theorem cheese_calories_per_serving :
  calories_per_serving 16 5 1210 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cheese_calories_per_serving_l3983_398316


namespace NUMINAMATH_CALUDE_x_range_when_p_and_not_q_x_in_range_implies_p_and_not_q_l3983_398317

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x : ℝ) : Prop := 1/(3 - x) > 1

-- Define the set representing the range of x
def range_x : Set ℝ := {x | x < -3 ∨ (1 < x ∧ x ≤ 2) ∨ x ≥ 3}

-- Theorem statement
theorem x_range_when_p_and_not_q (x : ℝ) :
  p x ∧ ¬(q x) → x ∈ range_x :=
by
  sorry

-- Theorem for the converse (to show equivalence)
theorem x_in_range_implies_p_and_not_q (x : ℝ) :
  x ∈ range_x → p x ∧ ¬(q x) :=
by
  sorry

end NUMINAMATH_CALUDE_x_range_when_p_and_not_q_x_in_range_implies_p_and_not_q_l3983_398317


namespace NUMINAMATH_CALUDE_cars_without_features_l3983_398308

theorem cars_without_features (total : ℕ) (air_bag : ℕ) (power_windows : ℕ) (both : ℕ)
  (h1 : total = 65)
  (h2 : air_bag = 45)
  (h3 : power_windows = 30)
  (h4 : both = 12) :
  total - (air_bag + power_windows - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cars_without_features_l3983_398308


namespace NUMINAMATH_CALUDE_barium_chloride_molecular_weight_l3983_398374

/-- The molecular weight of one mole of Barium chloride, given the molecular weight of 4 moles. -/
theorem barium_chloride_molecular_weight :
  let moles : ℝ := 4
  let total_weight : ℝ := 828
  let one_mole_weight : ℝ := total_weight / moles
  one_mole_weight = 207 := by sorry

end NUMINAMATH_CALUDE_barium_chloride_molecular_weight_l3983_398374


namespace NUMINAMATH_CALUDE_max_plates_buyable_l3983_398399

/-- The cost of a pan -/
def pan_cost : ℕ := 3

/-- The cost of a pot -/
def pot_cost : ℕ := 5

/-- The cost of a plate -/
def plate_cost : ℕ := 10

/-- The total budget -/
def total_budget : ℕ := 100

/-- The minimum number of each item to buy -/
def min_items : ℕ := 2

/-- A function to calculate the total cost of the purchase -/
def total_cost (pans pots plates : ℕ) : ℕ :=
  pan_cost * pans + pot_cost * pots + plate_cost * plates

/-- The main theorem stating the maximum number of plates that can be bought -/
theorem max_plates_buyable :
  ∃ (pans pots plates : ℕ),
    pans ≥ min_items ∧
    pots ≥ min_items ∧
    plates ≥ min_items ∧
    total_cost pans pots plates = total_budget ∧
    plates = 8 ∧
    ∀ (p : ℕ), p > plates →
      ∀ (x y : ℕ), x ≥ min_items → y ≥ min_items →
        total_cost x y p ≠ total_budget :=
by sorry

end NUMINAMATH_CALUDE_max_plates_buyable_l3983_398399


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3983_398301

def A : Set ℝ := {x | x > -2}
def B : Set ℝ := {x | 1 - x > 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3983_398301


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l3983_398339

-- Define the polar equation of the circle
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Define the Cartesian equation of the circle
def cartesian_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Theorem statement
theorem circle_radius_is_one :
  ∀ ρ θ x y : ℝ,
  polar_equation ρ θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  cartesian_equation x y →
  1 = (x^2 + y^2).sqrt :=
by sorry


end NUMINAMATH_CALUDE_circle_radius_is_one_l3983_398339


namespace NUMINAMATH_CALUDE_unique_divisible_by_18_l3983_398321

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem unique_divisible_by_18 :
  ∀ n : ℕ, n < 10 →
    (is_divisible_by (7120 + n) 18 ↔ n = 8) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_18_l3983_398321


namespace NUMINAMATH_CALUDE_table_tennis_team_members_l3983_398358

theorem table_tennis_team_members : ∃ (x : ℕ), x > 0 ∧ x ≤ 33 ∧ 
  (∃ (s r : ℕ), s + r = x ∧ 4 * s + 3 * r + 2 * x = 33) :=
by
  sorry

end NUMINAMATH_CALUDE_table_tennis_team_members_l3983_398358


namespace NUMINAMATH_CALUDE_gasoline_distribution_impossible_l3983_398326

theorem gasoline_distribution_impossible : 
  ¬ ∃ (A B C : ℝ), 
    A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧
    A + B + C = 50 ∧ 
    A = B + 10 ∧ 
    C + 26 = B :=
by sorry

end NUMINAMATH_CALUDE_gasoline_distribution_impossible_l3983_398326


namespace NUMINAMATH_CALUDE_divisibility_by_six_l3983_398392

theorem divisibility_by_six (a x : ℤ) : 
  (∃ k : ℤ, a * (x^3 + a^2 * x^2 + a^2 - 1) = 6 * k) ↔ 
  (∃ t : ℤ, x = 3 * t ∨ x = 3 * t - a^2) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l3983_398392


namespace NUMINAMATH_CALUDE_mike_percentage_l3983_398350

def phone_cost : ℝ := 1300
def additional_needed : ℝ := 780

theorem mike_percentage : 
  (phone_cost - additional_needed) / phone_cost * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_mike_percentage_l3983_398350


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3983_398336

def set_A : Set ℝ := {x | (x - 1) / (x + 3) < 0}
def set_B : Set ℝ := {x | abs x < 2}

theorem intersection_of_A_and_B : 
  set_A ∩ set_B = {x | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3983_398336


namespace NUMINAMATH_CALUDE_b_alone_time_l3983_398304

/-- The time it takes for A and B together to complete the task -/
def time_AB : ℝ := 3

/-- The time it takes for B and C together to complete the task -/
def time_BC : ℝ := 6

/-- The time it takes for A and C together to complete the task -/
def time_AC : ℝ := 4.5

/-- The rate at which A completes the task -/
def rate_A : ℝ := sorry

/-- The rate at which B completes the task -/
def rate_B : ℝ := sorry

/-- The rate at which C completes the task -/
def rate_C : ℝ := sorry

theorem b_alone_time (h1 : rate_A + rate_B = 1 / time_AB)
                     (h2 : rate_B + rate_C = 1 / time_BC)
                     (h3 : rate_A + rate_C = 1 / time_AC) :
  1 / rate_B = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_b_alone_time_l3983_398304


namespace NUMINAMATH_CALUDE_volume_sphere_minus_cylinder_l3983_398346

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem volume_sphere_minus_cylinder (R : ℝ) (r : ℝ) (h : ℝ) :
  R = 7 →
  r = 4 →
  h = 2 * Real.sqrt 33 →
  (4 / 3 * π * R^3 - π * r^2 * h) = ((1372 / 3 : ℝ) - 32 * Real.sqrt 33) * π :=
by sorry

end NUMINAMATH_CALUDE_volume_sphere_minus_cylinder_l3983_398346


namespace NUMINAMATH_CALUDE_triangle_sides_with_inscribed_rhombus_l3983_398340

/-- A right triangle with a 60° angle and an inscribed rhombus -/
structure TriangleWithRhombus where
  /-- Side length of the inscribed rhombus -/
  rhombus_side : ℝ
  /-- The rhombus shares the 60° angle with the triangle -/
  shares_angle : Bool
  /-- All vertices of the rhombus lie on the sides of the triangle -/
  vertices_on_sides : Bool

/-- Theorem about the sides of the triangle given the inscribed rhombus -/
theorem triangle_sides_with_inscribed_rhombus 
  (t : TriangleWithRhombus) 
  (h1 : t.rhombus_side = 6) 
  (h2 : t.shares_angle) 
  (h3 : t.vertices_on_sides) : 
  ∃ (a b c : ℝ), a = 9 ∧ b = 9 * Real.sqrt 3 ∧ c = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_with_inscribed_rhombus_l3983_398340


namespace NUMINAMATH_CALUDE_science_club_membership_l3983_398322

theorem science_club_membership (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_science_club_membership_l3983_398322


namespace NUMINAMATH_CALUDE_all_X_composite_except_101_l3983_398337

def X (n : ℕ) : ℕ := 
  (10^(2*n + 1) - 1) / 9

theorem all_X_composite_except_101 (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ X n = a * b :=
sorry

end NUMINAMATH_CALUDE_all_X_composite_except_101_l3983_398337


namespace NUMINAMATH_CALUDE_football_team_probability_l3983_398354

/-- Given a group of 10 people with 2 from football teams and 8 from basketball teams,
    proves the probability that both randomly selected people are from football teams,
    given that one is from a football team, is 1/9. -/
theorem football_team_probability :
  let total_people : ℕ := 10
  let football_people : ℕ := 2
  let basketball_people : ℕ := 8
  let total_selections : ℕ := 9  -- Total ways to select given one is from football
  let both_football : ℕ := 1     -- Ways to select both from football given one is from football
  football_people + basketball_people = total_people →
  (both_football : ℚ) / total_selections = 1 / 9 :=
by sorry

end NUMINAMATH_CALUDE_football_team_probability_l3983_398354


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3983_398344

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (h_radius : r = 5) :
  2 * π * r^2 + 2 * π * r * h = 170 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3983_398344


namespace NUMINAMATH_CALUDE_parallelogram_not_symmetrical_l3983_398345

-- Define a type for shapes
inductive Shape
  | Circle
  | Rectangle
  | IsoscelesTrapezoid
  | Parallelogram

-- Define a property for symmetry
def is_symmetrical (s : Shape) : Prop :=
  match s with
  | Shape.Circle => True
  | Shape.Rectangle => True
  | Shape.IsoscelesTrapezoid => True
  | Shape.Parallelogram => False

-- Theorem statement
theorem parallelogram_not_symmetrical :
  ∃ (s : Shape), ¬(is_symmetrical s) ∧ s = Shape.Parallelogram :=
sorry

end NUMINAMATH_CALUDE_parallelogram_not_symmetrical_l3983_398345


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3983_398306

theorem quadratic_inequality_condition (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3983_398306


namespace NUMINAMATH_CALUDE_circle_circumference_approximation_l3983_398323

/-- The circumference of a circle with radius 0.4997465213085514 meters is approximately 3.140093 meters. -/
theorem circle_circumference_approximation :
  let r : ℝ := 0.4997465213085514
  let π : ℝ := Real.pi
  let C : ℝ := 2 * π * r
  ∃ ε > 0, |C - 3.140093| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_approximation_l3983_398323


namespace NUMINAMATH_CALUDE_C1_intersects_C2_l3983_398395

-- Define the line C1
def C1 (x : ℝ) : ℝ := 2 * x - 3

-- Define the circle C2
def C2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 25

-- Theorem stating that C1 and C2 intersect
theorem C1_intersects_C2 : ∃ (x y : ℝ), y = C1 x ∧ C2 x y := by
  sorry

end NUMINAMATH_CALUDE_C1_intersects_C2_l3983_398395


namespace NUMINAMATH_CALUDE_carla_tile_counting_l3983_398376

theorem carla_tile_counting (tiles : ℕ) (books : ℕ) (book_counts : ℕ) (total_counts : ℕ)
  (h1 : tiles = 38)
  (h2 : books = 75)
  (h3 : book_counts = 3)
  (h4 : total_counts = 301)
  : ∃ (tile_counts : ℕ), tile_counts * tiles + book_counts * books = total_counts ∧ tile_counts = 2 := by
  sorry

end NUMINAMATH_CALUDE_carla_tile_counting_l3983_398376


namespace NUMINAMATH_CALUDE_gold_quarter_value_ratio_l3983_398370

theorem gold_quarter_value_ratio : 
  let melted_value_per_ounce : ℚ := 100
  let quarter_weight : ℚ := 1 / 5
  let spent_value : ℚ := 1 / 4
  (melted_value_per_ounce * quarter_weight) / spent_value = 80 := by
  sorry

end NUMINAMATH_CALUDE_gold_quarter_value_ratio_l3983_398370


namespace NUMINAMATH_CALUDE_binomial_20_10_l3983_398372

theorem binomial_20_10 (h1 : Nat.choose 18 9 = 48620) 
                       (h2 : Nat.choose 18 10 = 43758) 
                       (h3 : Nat.choose 18 11 = 24310) : 
  Nat.choose 20 10 = 97240 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l3983_398372


namespace NUMINAMATH_CALUDE_group_arrangements_eq_40_l3983_398329

/-- The number of ways to divide 2 teachers and 6 students into two groups,
    each consisting of 1 teacher and 3 students. -/
def group_arrangements : ℕ :=
  (Nat.choose 2 1) * (Nat.choose 6 3)

/-- Theorem stating that the number of group arrangements is 40. -/
theorem group_arrangements_eq_40 : group_arrangements = 40 := by
  sorry

end NUMINAMATH_CALUDE_group_arrangements_eq_40_l3983_398329


namespace NUMINAMATH_CALUDE_max_price_of_roses_and_peonies_l3983_398335

-- Define the price of a rose and a peony
variable (R P : ℝ)

-- Define the conditions
def condition1 : Prop := 4 * R + 5 * P ≥ 27
def condition2 : Prop := 6 * R + 3 * P ≤ 27

-- Define the objective function
def objective : ℝ := 3 * R + 4 * P

-- Theorem statement
theorem max_price_of_roses_and_peonies :
  condition1 R P → condition2 R P → ∃ (max : ℝ), max = 36 ∧ objective R P ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_price_of_roses_and_peonies_l3983_398335


namespace NUMINAMATH_CALUDE_sum_sequence_37th_term_l3983_398342

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_sequence_37th_term
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : arithmetic_sequence b)
  (ha1 : a 1 = 25)
  (hb1 : b 1 = 75)
  (hab2 : a 2 + b 2 = 100) :
  a 37 + b 37 = 100 := by
sorry

end NUMINAMATH_CALUDE_sum_sequence_37th_term_l3983_398342


namespace NUMINAMATH_CALUDE_equal_division_of_cards_l3983_398361

theorem equal_division_of_cards (total_cards : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) : 
  total_cards = 455 → num_friends = 5 → cards_per_friend = total_cards / num_friends → cards_per_friend = 91 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_cards_l3983_398361


namespace NUMINAMATH_CALUDE_inequality_solution_l3983_398307

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / (x * (x + 2)) < 1 / 4) ↔ (x < -1 ∨ x > 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3983_398307


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3983_398349

/-- A line with equal intercepts on both axes passing through (2, -3) -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (2, -3) -/
  point_condition : -3 = m * 2 + b
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b = -b / m

/-- The equation of a line with equal intercepts passing through (2, -3) is either x + y + 1 = 0 or 3x + 2y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = -1 ∧ l.b = 1) ∨ (l.m = -3/2 ∧ l.b = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3983_398349


namespace NUMINAMATH_CALUDE_employee_payment_percentage_l3983_398338

theorem employee_payment_percentage (total_payment y_payment x_payment : ℝ) :
  total_payment = 770 →
  y_payment = 350 →
  x_payment + y_payment = total_payment →
  x_payment / y_payment = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_percentage_l3983_398338


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3983_398352

-- Define the quadratic function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f (-12) (-2) x > 0} = {x : ℝ | -1/2 < x ∧ x < 1/3} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a (-1) x ≥ 0) ↔ a ≥ 1/8 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3983_398352


namespace NUMINAMATH_CALUDE_remaining_seeds_l3983_398313

def initial_seeds : ℝ := 8.75
def sowed_seeds : ℝ := 2.75

theorem remaining_seeds :
  initial_seeds - sowed_seeds = 6 := by sorry

end NUMINAMATH_CALUDE_remaining_seeds_l3983_398313


namespace NUMINAMATH_CALUDE_weight_of_BaCl2_l3983_398318

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The number of moles of BaCl2 -/
def moles_BaCl2 : ℝ := 8

/-- The molecular weight of BaCl2 in g/mol -/
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

/-- The total weight of BaCl2 in grams -/
def total_weight_BaCl2 : ℝ := molecular_weight_BaCl2 * moles_BaCl2

theorem weight_of_BaCl2 : total_weight_BaCl2 = 1665.84 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_BaCl2_l3983_398318


namespace NUMINAMATH_CALUDE_pizza_slice_ratio_l3983_398327

theorem pizza_slice_ratio : 
  ∀ (total_slices lunch_slices : ℕ),
    total_slices = 12 →
    lunch_slices ≤ total_slices →
    (total_slices - lunch_slices) / 3 + 4 = total_slices - lunch_slices →
    lunch_slices = total_slices / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_ratio_l3983_398327


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3983_398366

theorem solution_set_inequality (x : ℝ) : 
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3983_398366
